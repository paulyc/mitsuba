/* This file is based on work conducted at Weta Digital */

#include <mitsuba/layer/layer.h>
#include <mitsuba/layer/microfacet.h>
#include <mitsuba/layer/hg.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/brent.h>
#include <mitsuba/render/scene.h>
#include <Eigen/SparseLU>
#include <boost/bind.hpp>

#if defined(__OSX__)
	#include <dispatch/dispatch.h>
	#define MTS_GCD 1
#endif

#if defined(MTS_HAS_FFTW)
	#include <fftw3.h>
#endif

#define DROP_THRESHOLD 1e-9

MTS_NAMESPACE_BEGIN

Layer::Layer(const DVector &nodes, const DVector &weights, size_t nFourierOrders)
  : m_tau(0.0f), m_nodes(nodes), m_weights(weights) {
	size_t n = (size_t) nodes.size();
	m_modes.resize(nFourierOrders);
	for (size_t i=0; i<nFourierOrders; ++i)
		m_modes[i] = LayerMode(n);
    m_albedo = 0;
}

void Layer::add(const Layer &layer1, const Layer &layer2, Layer &output, bool homogeneous) {
	SAssert(output.getNodeCount() == layer1.getNodeCount()
		 && output.getNodeCount() == layer2.getNodeCount()
	     && output.getFourierOrders() == layer1.getFourierOrders()
		 && output.getFourierOrders() == layer2.getFourierOrders());

	int n = output.getNodeCount() / 2;
	SMatrix I(n, n);
	I.setIdentity();

	/* Special case: it is possible to save quite a bit of computation when we
	   know that both layers are homogeneous and of the same type */
	if (homogeneous) {
		#if defined(MTS_GCD)
			dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
			dispatch_apply(output.getFourierOrders(), queue, ^(size_t i) {
		#elif defined(MTS_OPENMP)
			#pragma omp parallel for schedule(dynamic)
			for (int i=0; i<output.getFourierOrders(); ++i) {
		#else
			for (int i=0; i<output.getFourierOrders(); ++i) {
		#endif
		{
			const LayerMode &l1 = layer1[i], &l2 = layer2[i];
			LayerMode &lo = output[i];
			SMatrix Rb, Ttb;

			if (l1.isDiagonal() && l2.isDiagonal()) {
				Rb = SMatrix(n, n); // zero
				Ttb = l2.transmissionTopBottom() * l1.transmissionTopBottom();
			} else {
				/* Gain for downward radiation */
				Eigen::SparseLU<SMatrix, Eigen::AMDOrdering<int> > G_tb;
				G_tb.compute(I - l1.reflectionBottom() * l2.reflectionTop());

				/* Transmission at the bottom due to illumination at the top */
				SMatrix result = G_tb.solve(l1.transmissionTopBottom());
				Ttb = l2.transmissionTopBottom() * result;

				/* Reflection at the bottom */
				SMatrix temp = l1.reflectionBottom() * l2.transmissionBottomTop();
				result = G_tb.solve(temp);
				Rb = l2.reflectionBottom() + l2.transmissionTopBottom() * result;
			}

			#if defined(DROP_THRESHOLD)
				Ttb.prune((Float) 1, (Float) DROP_THRESHOLD);
				Rb.prune((Float) 1, (Float) DROP_THRESHOLD);
			#endif

			lo.transmissionTopBottom() = Ttb;
			lo.transmissionBottomTop() = Ttb;
			lo.reflectionTop() = Rb;
			lo.reflectionBottom() = Rb;
		}
		#if defined(MTS_GCD)
			} );
		#else
			}
		#endif
	} else {
		#if defined(MTS_GCD)
			dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
			dispatch_apply(output.getFourierOrders(), queue, ^(size_t i) {
		#elif defined(MTS_OPENMP)
			#pragma omp parallel for schedule(dynamic)
			for (int i=0; i<output.getFourierOrders(); ++i) {
		#else
			for (int i=0; i<output.getFourierOrders(); ++i) {
		#endif
		{
			const LayerMode &l1 = layer1[i], &l2 = layer2[i];
			LayerMode &lo = output[i];

			/* Gain for downward radiation */
			Eigen::SparseLU<SMatrix, Eigen::AMDOrdering<int> > G_tb;
			G_tb.compute(I - l1.reflectionBottom() * l2.reflectionTop());

			/* Gain for upward radiation */
			Eigen::SparseLU<SMatrix, Eigen::AMDOrdering<int> > G_bt;
			G_bt.compute(I - l2.reflectionTop() * l1.reflectionBottom());

			/* Transmission at the bottom due to illumination at the top */
			SMatrix result = G_tb.solve(l1.transmissionTopBottom());
			SMatrix Ttb = l2.transmissionTopBottom() * result;

			/* Reflection at the bottom */
			SMatrix temp = l1.reflectionBottom() * l2.transmissionBottomTop();
			result = G_tb.solve(temp);
			SMatrix Rb = l2.reflectionBottom() + l2.transmissionTopBottom() * result;

			/* Transmission at the top due to illumination at the bottom */
			result = G_bt.solve(l2.transmissionBottomTop());
			SMatrix Tbt = l1.transmissionBottomTop() * result;

			/* Reflection at the top */
			temp = l2.reflectionTop() * l1.transmissionTopBottom();
			result = G_bt.solve(temp);
			SMatrix Rt = l1.reflectionTop() + l1.transmissionBottomTop() * result;

			#if defined(DROP_THRESHOLD)
				Ttb.prune((Float) 1, (Float) DROP_THRESHOLD);
				Tbt.prune((Float) 1, (Float) DROP_THRESHOLD);
				Rb.prune((Float) 1, (Float) DROP_THRESHOLD);
				Rt.prune((Float) 1, (Float) DROP_THRESHOLD);
			#endif

			lo.transmissionTopBottom() = Ttb;
			lo.transmissionBottomTop() = Tbt;
			lo.reflectionTop() = Rt;
			lo.reflectionBottom() = Rb;
		}
		#if defined(MTS_GCD)
			} );
		#else
			}
		#endif
	}

	output.m_tau = layer1.m_tau + layer2.m_tau;
}

void Layer::expand(Float desiredSize, bool homogeneous) {
	if (desiredSize == m_tau)
		return;
	SAssertEx(desiredSize > m_tau, "expand(): Attempted to reduce the size of a layer!");

	Layer temp(*this);

	uint64_t count = (uint64_t) (desiredSize / m_tau) - 1;

	SLog(EDebug, "Expanding layer from tau=%e -> %e (i.e. %e times its size)",
			m_tau, desiredSize, (Float) count);

	ref<Timer> timer = new Timer();
	int doublings = 0;

	do {
		if (count & 1)
			temp.addToTop(*this, homogeneous);

		count /= 2;

		if (count == 0)
			break;

		addToTop(*this, homogeneous);
		doublings++;
	} while (true);

	SLog(EDebug, "Done after %i doublings, took %i ms", doublings, timer->getMilliseconds());

	*this = temp;
}

void Layer::setHenyeyGreenstein(Float albedo, Float g) {
	int n = getNodeCount();
	m_albedo = albedo;
	#if defined(MTS_GCD)
		__block ref<Mutex> mutex = new Mutex();
		__block std::vector<Quartet> quartets;
	#else
		ref<Mutex> mutex = new Mutex();
		std::vector<Quartet> quartets;
	#endif

	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(n, queue, ^(size_t i) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (int i=0; i<n; ++i) {
	#else
		for (int i=0; i<n; ++i) {
	#endif

	{
		/* Parallel loop over 'i' */

		Float mu_i = m_nodes[i];
		std::vector<Float> result;
		std::vector<Quartet> quartetLocal;
		quartetLocal.reserve(getFourierOrders() * n);

		for (int o=0; o<=i; ++o) {
			Float mu_o = m_nodes[o];
			hgFourierSeries(mu_o, mu_i, g, getFourierOrders(), ERROR_GOAL, result);

			for (int l=0; l<std::min(getFourierOrders(), (int) result.size()); ++l) {
				Float value = result[l] * albedo * M_PI * (l == 0 ? 2 : 1);
				quartetLocal.push_back(Quartet(l, o, i, value));
				if (i != o)
					quartetLocal.push_back(Quartet(l, i, o, value));
			}
		}
		mutex->lock();
		quartets.insert(quartets.end(), quartetLocal.begin(), quartetLocal.end());
		mutex->unlock();
	}

	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif

	initializeFromQuartets(quartets);
	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].scaleColumns(m_weights);
}

void Layer::setVMF(Float albedo, Float kappa) {
	int n = getNodeCount();
	m_albedo = albedo;
	#if defined(MTS_GCD)
		__block ref<Mutex> mutex = new Mutex();
		__block std::vector<Quartet> quartets;
	#else
		ref<Mutex> mutex = new Mutex();
		std::vector<Quartet> quartets;
	#endif
	Float scale;
	if (kappa == 0)
		scale = albedo / (4*M_PI);
	else
		scale = albedo * kappa / (4*M_PI * std::sinh(kappa));

	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(n, queue, ^(size_t i) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (int i=0; i<n; ++i) {
	#else
		for (int i=0; i<n; ++i) {
	#endif

	{
		/* Parallel loop over 'i' */

		Float mu_i = m_nodes[i];
		std::vector<Float> result;
		std::vector<Quartet> quartetLocal;
		quartetLocal.reserve(getFourierOrders() * n);

		for (int o=0; o<=i; ++o) {
			Float mu_o = m_nodes[o];
			if (kappa == 0) {
				Float value = scale * M_PI * 2;
				quartetLocal.push_back(Quartet(0, o, i, value));
				if (i != o)
					quartetLocal.push_back(Quartet(0, i, o, value));
				continue;
			}

			Float A = kappa * mu_i * mu_o;
			Float B = kappa * std::sqrt((1-mu_i*mu_i)*(1-mu_o*mu_o));

			expcosFourierSeries(A, B, ERROR_GOAL, result);

			for (int l=0; l<std::min(getFourierOrders(), (int) result.size()); ++l) {
				Float value = result[l] * scale * M_PI * (l == 0 ? 2 : 1);

				quartetLocal.push_back(Quartet(l, o, i, value));
				if (i != o)
					quartetLocal.push_back(Quartet(l, i, o, value));
			}
		}
		mutex->lock();
		quartets.insert(quartets.end(), quartetLocal.begin(), quartetLocal.end());
		mutex->unlock();
	}

	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif
	initializeFromQuartets(quartets);

	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].scaleColumns(m_weights);
}


void Layer::setMicrofacet(Float eta, Float k, Float alpha, bool conserveEnergy, int fourierOrders) {
	int n = getNodeCount(), h = n/2;

	fourierOrders = std::max(fourierOrders, getFourierOrders());

	#if defined(MTS_GCD)
		__block ref<Mutex> mutex = new Mutex();
		__block std::vector<Quartet> quartets;
	#else
		ref<Mutex> mutex = new Mutex();
		std::vector<Quartet> quartets;
	#endif

	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(n, queue, ^(size_t i) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (int i=0; i<n; ++i) {
	#else
		for (int i=0; i<n; ++i) {
	#endif
	{
		/* Parallel loop over 'i' */
		Float mu_i = m_nodes[i];
		std::vector<Float> result;
		std::vector<Quartet> quartetLocal;
		quartetLocal.reserve(getFourierOrders() * n);

		for (int o=0; o<n; ++o) {
			Float mu_o = m_nodes[o];
			/* Sign flip due to different convention (depth values increase opposite to the normal direction) */
			microfacetFourierSeries(-mu_o, -mu_i, eta, k, alpha, ERROR_GOAL, fourierOrders, result);

			for (int l=0; l<std::min(getFourierOrders(), (int) result.size()); ++l)
				quartetLocal.push_back(Quartet(l, o, i, result[l]));
		}

		mutex->lock();
		quartets.insert(quartets.end(), quartetLocal.begin(), quartetLocal.end());
		mutex->unlock();
	}

	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif

	initializeFromQuartets(quartets);

	/* Add a pseudo-diffuse term to capture lost energy */
	if (conserveEnergy && k == 0) {
		/* Case 1: Dielectrics */
		DVector W = m_weights.tail(h).cwiseProduct(m_nodes.tail(h)) * 2 * M_PI;
		LayerMode &l = m_modes[0];

		DVector Mb  = (W.asDiagonal() * DMatrix(l.reflectionBottom())).colwise().sum();
		DVector Mt  = (W.asDiagonal() * DMatrix(l.reflectionTop())).colwise().sum();
		DVector Mtb = (W.asDiagonal() * DMatrix(l.transmissionTopBottom())).colwise().sum();
		DVector Mbt = (W.asDiagonal() * DMatrix(l.transmissionBottomTop())).colwise().sum();

		/* Determine how much energy we'd like to put into the transmission component
		   (proportional to the current reflection/reflaction split) */
		DVector Atb = (DVector::Ones(h) - Mt - Mtb).cwiseProduct(Mtb.cwiseQuotient(Mt + Mtb));
		DVector Abt = (DVector::Ones(h) - Mb - Mbt).cwiseProduct(Mbt.cwiseQuotient(Mb + Mbt));
		Atb = Atb.cwiseMax(DVector::Zero(h));
		Abt = Abt.cwiseMax(DVector::Zero(h));

		/* Create a correction matrix which contains as much of the desired
		   energy as possible, while maintaining symmetry and energy conservation */
		DMatrix Ctb = Abt*Atb.transpose() / std::max(W.dot(Abt), W.dot(Atb) / (eta*eta));
		DMatrix Cbt = Ctb.transpose() / (eta*eta);

		sparsify(DMatrix(l.transmissionTopBottom()) + Ctb, l.transmissionTopBottom());
		sparsify(DMatrix(l.transmissionBottomTop()) + Cbt, l.transmissionBottomTop());

		/* Update missing energy terms */
		Mtb = (W.asDiagonal() * DMatrix(l.transmissionTopBottom())).colwise().sum();
		Mbt = (W.asDiagonal() * DMatrix(l.transmissionBottomTop())).colwise().sum();

		/* Put the rest of the missing energy into the reflection component */
		DVector At = DVector::Ones(h) - Mt - Mtb;
		DVector Ab = DVector::Ones(h) - Mb - Mbt;
		At = At.cwiseMax(DVector::Zero(h));
		Ab = Ab.cwiseMax(DVector::Zero(h));

		sparsify(DMatrix(l.reflectionTop())    + At*At.transpose() / W.dot(At), l.reflectionTop());
		sparsify(DMatrix(l.reflectionBottom()) + Ab*Ab.transpose() / W.dot(Ab), l.reflectionBottom());
	} else if (conserveEnergy && k != 0) {
		/* Case 2: Conductors */
		DVector W = m_weights.tail(h).cwiseProduct(m_nodes.tail(h)) * 2 * M_PI;

		/* Compute a reference matrix for a material *without* Fresnel effects */
		#if defined(MTS_GCD)
		__block DMatrix refMatrix(n, n);
		#else
		DMatrix refMatrix(n, n);
		#endif

		#if defined(MTS_GCD)
			dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
			dispatch_apply(n, queue, ^(size_t i) {
		#elif defined(MTS_OPENMP)
			#pragma omp parallel for schedule(dynamic)
			for (int i=0; i<n; ++i) {
		#else
			for (int i=0; i<n; ++i) {
		#endif

		{
			/* Parallel loop over 'i' */
			Float mu_i = m_nodes[i];
			std::vector<Float> result;
			for (int o=0; o<n; ++o) {
				Float mu_o = m_nodes[o];
				microfacetFourierSeries(-mu_o, -mu_i, 0.0f, 1.0f, alpha, ERROR_GOAL, fourierOrders, result);
				refMatrix(o, i) = result.size() > 0 ? result[0] : 0.0f;
			}
		}

		#if defined(MTS_GCD)
			} );
		#else
			}
		#endif
		DMatrix reflectionTopRef = DMatrix(refMatrix).block(0, h, h, h); /// XXX only compute needed block..

		DVector Mt = DVector::Ones(h) - (W.asDiagonal() * reflectionTopRef).colwise().sum().transpose();
		Mt = Mt.cwiseMax(DVector::Zero(h));

		Float F = fresnelDiffuseConductor(eta, k);
		Float E = 1 - W.dot(Mt) * INV_PI;

		Float factor = F*E / (1-F*(1-E));

		DMatrix C = Mt * Mt.transpose() * (factor/ W.dot(Mt));

		sparsify(DMatrix(m_modes[0].reflectionTop()) + C, m_modes[0].reflectionTop());
		sparsify(DMatrix(m_modes[0].reflectionBottom()) + C, m_modes[0].reflectionBottom());
	}

	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].scaleColumns(m_weights.cwiseProduct(m_nodes.cwiseAbs()) * M_PI * (l == 0 ? 2 : 1));
}

void Layer::setMatusik(const fs::path &path, int ch, int fourierOrders) {
#if !defined(MTS_HAS_FFTW)
	SLog(EError, "setMatusik(): You need to recompile Mitsuba with support for FFTW!");
#else
	Color3 scale(1.0/1500.0, 1.15/1500.0, 1.66/1500.0);

	#if defined(MTS_GCD)
		__block ref<Mutex> mutex = new Mutex();
		__block std::vector<Quartet> quartets;
	#else
		ref<Mutex> mutex = new Mutex();
		std::vector<Quartet> quartets;
	#endif

	ref<FileStream> fs = new FileStream(path, FileStream::EReadOnly);
	fs->setByteOrder(Stream::ELittleEndian);

	int res_theta_h = fs->readInt();
	int res_theta_d = fs->readInt();
	int res_phi_d = fs->readInt();

	ref<Timer> timer = new Timer();
	//SLog(EInfo, "Loading Matusik-style BRDF data file \"%s\" (%ix%ix%i)",
	//	path.filename().string().c_str(), res_theta_h, res_theta_d, res_phi_d);

	int fileSize = 3 * res_theta_h * res_theta_d * res_phi_d;

	double *storage = new double[fileSize];
	fs->readDoubleArray(storage, fileSize);

	fftw_plan_with_nthreads(1);

	fourierOrders = std::max(fourierOrders, getFourierOrders());
	int fftSize = fourierOrders * 4;
	fftw_plan plan = fftw_plan_r2r_1d(fftSize, NULL, NULL, FFTW_REDFT00, FFTW_ESTIMATE);

	int n = getNodeCount(), nEntries = n*n;
	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(nEntries, queue, ^(size_t entry) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (int entry=0; entry<nEntries; ++entry) {
	#else
		for (int entry=0; entry<nEntries; ++entry) {
	#endif
	{
		int i = entry / n, o = entry % n;

		Float cosThetaI = m_nodes[i],
		      sinThetaI = std::sqrt(1-cosThetaI*cosThetaI),
		      cosThetaO = m_nodes[o],
			  sinThetaO = std::sqrt(1-cosThetaO*cosThetaO);

		Vector wi(sinThetaI, 0, cosThetaI);

		if (cosThetaI * cosThetaO > 0 || cosThetaI < 0) {
			#if defined(MTS_GCD)
				return;
			#else
				continue;
			#endif
		}

		double *data = (double *) fftw_malloc(fftSize * sizeof(double));

		for (int j=0; j<fftSize; ++j) {
			/* Parallel loop over 'i' */
			Float phi_d = M_PI * j/(Float) (fftSize-1),
				  cosPhi = std::cos(phi_d),
				  sinPhi = std::sin(phi_d);
			Vector wo(-sinThetaO*cosPhi, -sinThetaO*sinPhi, -cosThetaO);

			Vector half = normalize(wi + wo);

			Float theta_half = std::acos(half.z);
			Float phi_half = std::atan2(half.y, half.x);

			Vector diff =
				(Transform::rotate(Vector(0, 1, 0), -radToDeg(theta_half)) *
				 Transform::rotate(Vector(0, 0, 1), -radToDeg(phi_half)))(wi);

			int theta_half_idx = std::min(std::max(0, (int) std::sqrt(
				((theta_half / (M_PI/2.0))*res_theta_h) * res_theta_h)), res_theta_h-1);

			Float theta_diff = std::acos(diff.z);
			Float phi_diff = std::atan2(diff.y, diff.x);

			if (phi_diff < 0)
				phi_diff += M_PI;

			int phi_diff_idx = std::min(std::max(0, int(phi_diff / M_PI * res_phi_d)), res_phi_d - 1);

			int theta_diff_idx = std::min(std::max(0, int(theta_diff / (M_PI * 0.5) * res_theta_d)),
				res_theta_d - 1);

			int ind = phi_diff_idx +
				theta_diff_idx * res_phi_d +
				theta_half_idx * res_phi_d * res_theta_d;

			data[j] = storage[ind+ch*res_theta_h*res_theta_d*res_phi_d] * scale[ch] * 2; /// XXX too dark?
		}
		double *spectrum = (double *) fftw_malloc(fftSize * sizeof(double));
		fftw_execute_r2r(plan, data, spectrum);

		for (int j=0; j<fftSize; ++j)
			spectrum[j] /= (double) (fftSize-1);
		spectrum[0] /= 2;
		spectrum[fftSize-1] /= 2;

		double ref = std::abs(spectrum[0]);
		size_t sparseSize = 0;
		double partialSum = 0;
		if (ref != 0) {
			sparseSize = (size_t) getFourierOrders();
			for (size_t j=(size_t) getFourierOrders()-1; j>=1; --j) {
				double value = (float) spectrum[j];
				partialSum += std::abs(value);
				if (partialSum <= ref * ERROR_GOAL)
					sparseSize = j;
			}
		}

		mutex->lock();
		for (size_t l=0; l<sparseSize; ++l)
			quartets.push_back(Quartet(l, o, i, spectrum[l]));
		mutex->unlock();
		fftw_free(spectrum);
		fftw_free(data);
	}
	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif

	fftw_destroy_plan(plan);
	delete[] storage;
	SLog(EInfo, "Done (took %i ms)", timer->getMilliseconds());

	initializeFromQuartets(quartets);

	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].scaleColumns(m_weights.cwiseProduct(m_nodes.cwiseAbs()) * M_PI * (l == 0 ? 2 : 1));
#endif
}

void Layer::scale(Float scale) {
    m_albedo *= scale;
	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].scale(scale);
}

void Layer::addScaled(const Layer &layer, Float scale) {
    SAssert(layer.getFourierOrders() == getFourierOrders() &&
           layer.getNodeCount() == getNodeCount());
    m_albedo += scale * layer.m_albedo;
	for (int l=0; l<getFourierOrders(); ++l)
		m_modes[l].addScaled(layer.m_modes[l], scale);
}

void Layer::setDiffuse(Float albedo) {
	int n = getNodeCount(), h = n/2;

	for (int i=0; i<n; ++i) {
		for (int o=0; o<n; ++o) {
			if ((i < h && o >= h) || (o < h && i >= h))
				m_modes[0].insert(o, i, albedo * INV_PI);
		}
	}

	m_modes[0].makeCompressed();
	m_modes[0].scaleColumns(m_weights.cwiseProduct(m_nodes.cwiseAbs()) * M_PI * 2);
}


void Layer::solveDiscreteOrdinates(Float tau) {
	SLog(EError, "TODO: port over to sparse matrix rewrite");
#if 0
	int n = getNodeCount(), h = n/2;

	Eigen::EigenSolver<DMatrix> eigenSolver(n);
	DMatrix V(n, n), A(n, n), B(n, n);
	DVector lambda(n);

	#if defined(MTS_OPENMP)
		#pragma omp parallel for firstprivate(eigenSolver, V, A, B, lambda) schedule(dynamic)
	#endif
	for (int i=0; i<getFourierOrders(); ++i) {
		LayerMode &mode = m_modes[i];

		// Solve a non-symmetric eigenvalue problem for the constant coefficients of a system of 1st-order ODEs
		eigenSolver.compute(m_nodes.cwiseInverse().asDiagonal() * (mode.matrix() - DMatrix::Identity(n, n)));

		if (eigenSolver.eigenvalues().imag().array().abs().maxCoeff() > 1e-5f)
			SLog(EError, "Discrete ordinates solver: transport matrix has complex eigenvalues!");

		lambda = (eigenSolver.eigenvalues().real() * tau).array().exp().matrix();
		V = eigenSolver.eigenvectors().real();

		/* Boundary conditions: specified upwards radiation at the bottom, downwards radiation at the top */
		A << V.block(0, 0, h, n) * lambda.asDiagonal(), V.block(h, 0, h, n);

		/* Desired output: upwards radiation at the top, downwards radiation at the bottom */
		B << V.block(0, 0, h, n), V.block(h, 0, h, n) * lambda.asDiagonal();

		mode.matrix() = B * A.inverse(); // (we unfortunately do need to compute the full inverse matrix..)
	}

	m_tau = tau;
#endif
}

void Layer::setAbsorbing(Float tau) {
	int n = getNodeCount() / 2;

	SMatrix I(n, n);
	I.setIdentity();

	for (int i=0; i<getFourierOrders(); ++i) {
		LayerMode &mode = m_modes[i];
		SMatrix Ttb = I * std::exp(-tau);

		mode.reflectionTop().setZero();
		mode.reflectionBottom().setZero();
		mode.transmissionTopBottom() = Ttb;
		mode.transmissionBottomTop() = Ttb;
	}
}

void Layer::solveAddingDoubling(Float tau) {
	/* Heuristic for choosing the initial width of a layer based on
	   "Discrete Space Theory of Radiative Transfer" by Grant and Hunt
	   Proc. R. Soc. London 1969 */
	m_tau = std::min(m_nodes.cwiseAbs().minCoeff() / (1-m_albedo/2),
			(Float) std::pow(2.0, -15.0));

	int doublings = (int) std::ceil(std::log(tau / m_tau) / std::log(2.0));
	m_tau = tau * std::pow((Float) 2.0f, (Float) -doublings);

	int n = getNodeCount() / 2;

	SMatrix I(n, n);
	I.setIdentity();

	SMatrix rowScale;
	sparseDiag(m_nodes.tail(n).cwiseInverse() * m_tau, rowScale);

	for (int i=0; i<getFourierOrders(); ++i) {
		LayerMode &mode = m_modes[i];
		SMatrix Rt = rowScale * mode.reflectionTop();
		SMatrix Ttb = I + rowScale * (mode.transmissionTopBottom() - I);

		mode.reflectionTop() = Rt;
		mode.reflectionBottom() = Rt;
		mode.transmissionTopBottom() = Ttb;
		mode.transmissionBottomTop() = Ttb;
	}

	for (int i=0; i<doublings; ++i)
		addToTop(*this, true);
}

void Layer::solve(Float tau) {
	solveAddingDoubling(tau);
}

void Layer::clearBottomAndTransmission() {
	for (int l=0; l<getFourierOrders(); ++l) {
		LayerMode &mode = m_modes[l];
		mode.transmissionBottomTop().setZero();
		mode.transmissionTopBottom().setZero();
		mode.reflectionBottom().setZero();
	}
}

void Layer::initializeFromQuartets(std::vector<Quartet> &quartets) {
	std::vector< std::vector<Triplet> >
		tripletsTbt(getFourierOrders()),
		tripletsTtb(getFourierOrders()),
		tripletsRb(getFourierOrders()),
		tripletsRt(getFourierOrders());

	size_t approxSize = quartets.size() / (4*getFourierOrders());

	ref<Timer> timer = new Timer();
	for (int i=0; i<getFourierOrders(); ++i) {
		tripletsTbt[i].reserve(approxSize);
		tripletsTtb[i].reserve(approxSize);
		tripletsRb[i].reserve(approxSize);
		tripletsRt[i].reserve(approxSize);
	}

	int n = getNodeCount()/2;
	for (size_t i=0; i<quartets.size(); ++i) {
		const Quartet &quartet = quartets[i];

		if (quartet.o < n && quartet.i < n)
			tripletsTbt[quartet.l].push_back(Triplet(quartet.o, quartet.i, quartet.value));
		else if (quartet.o >= n && quartet.i >= n)
			tripletsTtb[quartet.l].push_back(Triplet(quartet.o-n, quartet.i-n, quartet.value));
		else if (quartet.o <n && quartet.i >= n)
			tripletsRt[quartet.l].push_back(Triplet(quartet.o, quartet.i-n, quartet.value));
		else if (quartet.o >=n && quartet.i < n)
			tripletsRb[quartet.l].push_back(Triplet(quartet.o-n, quartet.i, quartet.value));
		else
			SLog(EError, "Internal error!");
	}

	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(getFourierOrders(), queue, ^(size_t l) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (int l=0; l<getFourierOrders(); ++l) {
	#else
		for (int l=0; l<getFourierOrders(); ++l) {
	#endif
	{
		m_modes[l].reflectionTop()        .setFromTriplets(tripletsRt [l].begin(), tripletsRt [l].end());
		m_modes[l].reflectionBottom()     .setFromTriplets(tripletsRb [l].begin(), tripletsRb [l].end());
		m_modes[l].transmissionTopBottom().setFromTriplets(tripletsTtb[l].begin(), tripletsTtb[l].end());
		m_modes[l].transmissionBottomTop().setFromTriplets(tripletsTbt[l].begin(), tripletsTbt[l].end());
	}

	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif
}

int initializeQuadrature(int n, DVector &nodes, DVector &weights) {
	if (n % 2 != 0) /* Need an even number of nodes */
		++n;

	int belowCritical = 0;

	nodes.resize(n);
	weights.resize(n);

	gaussLobatto(n, nodes.data(), weights.data());

	/* Rearrange the nodes so that they are nicer to deal with in adding doubling */
	weights.head(n/2).reverseInPlace();
	nodes.head(n/2).reverseInPlace();

	return belowCritical;
}

static Float vmfG(Float kappa, Float g = 0) {
	if (kappa == 0)
		return -g;
	Float coth = kappa > 6 ? 1 : ((std::exp(2*kappa)+1)/(std::exp(2*kappa)-1));
	return coth-1/kappa-g;
}

Float vmfKappa(Float g) {
	if (g == 0)
		return 0;
	else if (g < 0)
		SLog(EError, "Error: vMF distribution cannot be created for g<0.");

	BrentSolver brentSolver(100, 1e-6f);
	BrentSolver::Result result = brentSolver.solve(
		boost::bind(&vmfG, _1, g), 0, 1000);
	SAssert(result.success);
	return result.x;
}

std::pair<int, int> parameterHeuristicMicrofacet(Float alpha, Float eta, Float k) {
    alpha = std::min(alpha, (Float) 1);
    if (eta < 1)
        eta = 1/eta;
    if (std::abs(eta-1.5046f) < 1e-5f)
        eta = 1.5f;

    static const Float c[][9] = {
        /* IOR    A_n      B_n     C_n       D_n      A_m      B_m      C_m      D_m                                 */
        {  0.0, 35.275,  14.136,  29.287,  1.8765,   39.814,  88.992, -98.998,  39.261  },  /* Generic conductor     */
        {  1.1, 256.47, -73.180,  99.807,  37.383,  110.782,  57.576,  94.725,  14.001  },  /* Dielectric, eta = 1.1 */
        {  1.3, 100.264, 28.187,  64.425,  14.850,   45.809,  17.785, -7.8543,  12.892  },  /* Dielectric, eta = 1.3 */
        {  1.5, 74.176,  27.470,  42.454,  9.6437,   31.700,  44.896, -45.016,  19.643  },  /* Dielectric, eta = 1.5 */
        {  1.7, 80.098,  17.016,  50.656,  7.2798,   46.549,  58.592, -73.585,  25.473  },  /* Dielectric, eta = 1.7 */
    };

    int i0 = 0, i1 = 0;

    if (k == 0) { /* Dielectric case */
        for (int i=1; i<4; ++i) {
            if (eta >= c[i][0] && eta <= c[i+1][0]) {
                if (eta == c[i][0]) {
                    i1 = i0 = i;
                } else if (eta == c[i+1][0]) {
                    i0 = i1 = i+1;
                } else {
                    i0 = i; i1 = i+1;
                }
            }
        }

        if (!i0)
            throw std::runtime_error("Index of refraction is out of bounds (must be between 1.1 and 1.7)!");
    }

    Float n0 = std::max(c[i0][1] + c[i0][2]*std::pow(std::log(alpha), (Float) 4)*alpha, c[i0][3]+c[i0][4]*std::pow(alpha, (Float) -1.2f));
    Float n1 = std::max(c[i1][1] + c[i1][2]*std::pow(std::log(alpha), (Float) 4)*alpha, c[i1][3]+c[i1][4]*std::pow(alpha, (Float) -1.2f));
    Float m0 = std::max(c[i0][5] + c[i0][6]*std::pow(std::log(alpha), (Float) 4)*alpha, c[i0][7]+c[i0][8]*std::pow(alpha, (Float) -1.2f));
    Float m1 = std::max(c[i1][5] + c[i1][6]*std::pow(std::log(alpha), (Float) 4)*alpha, c[i1][7]+c[i1][8]*std::pow(alpha, (Float) -1.2f));

    return std::make_pair((int) std::ceil(std::max(n0, n1)), (int) std::ceil(std::max(m0, m1)));
}

std::pair<int, int> parameterHeuristicHG(Float g) {
    g = std::abs(g);
    Float m = 5.4f/(1.0f - g) - 1.3f;
    Float n = 8.6f/(1.0f - g) - 0.2f;
    return std::make_pair((int) std::ceil(n), (int) std::ceil(m));
}

std::pair<int, int> parameterHeuristicVMF(Float kappa) {
    return parameterHeuristicHG(vmfG(std::abs(kappa)));
}

MTS_NAMESPACE_END
