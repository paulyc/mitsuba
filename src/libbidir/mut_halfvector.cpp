/*
	This file is part of Mitsuba, a physically based rendering system.

	Copyright (c) 2007-2014 by Wenzel Jakob and others.

	Mitsuba is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License Version 3
	as published by the Free Software Foundation.

	Mitsuba is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/bidir/mut_halfvector.h>
#include <mitsuba/bidir/manifold.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/vmf.h>

// #define USE_EIGEN_TEST 1
#if USE_EIGEN_TEST==1
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#include <Eigen/LU>
#include <Eigen/Geometry>
#endif

using std::min;
using std::max;
using std::abs;
using std::sqrt;

#define MTS_REL_POS_EPS     ShadowEpsilon

MTS_NAMESPACE_BEGIN

static StatsCounter statsAccepted("Halfvector perturbation",
		"Acceptance rate", EPercentage);
static StatsCounter statsGenerated("Halfvector perturbation",
		"Successful generation rate", EPercentage);
static StatsCounter statsAvgBreakup("Halfvector perturbation",
									"Average breakup point", EAverage);
static StatsCounter statsNonReversible("Halfvector perturbation",
		"Non-reversible walks", EPercentage);
static StatsCounter statsFailLensSubpath("Halfvector perturbation",
		"Lens perturbation failed", EPercentage);
static StatsCounter statsConnectionFailed("Halfvector perturbation",
		"Last connection failed", EPercentage);

HalfvectorPerturbation::HalfvectorPerturbation(const Scene *scene, Sampler *sampler,
		  MemoryPool &pool, const Float probFactor) : 
		MutatorBase(scene, sampler, pool), m_probFactor(probFactor) {
	m_manifold = new PathManifold(scene, 6);
}

HalfvectorPerturbation::~HalfvectorPerturbation() {
}

Mutator::EMutationType HalfvectorPerturbation::getType() const {
	return EHalfvectorPerturbation;
}

Float HalfvectorPerturbation::suitability(const Path &path) const {
	return 1.0f;
}

#if USE_EIGEN_TEST==1
namespace
{
void filloutConstraintMatrix(PathManifold &manifold, Eigen::MatrixXd& A)
{
  const int numConstraints = (int)manifold.size() - 2;
  BDAssert(A.rows() == 2 * numConstraints && A.cols() == 2 * numConstraints);
  A.setZero();

  for (int j = 0, i = 0; j < numConstraints; ++j) {
    if (j - 1 >= 0) {
      A(2 * i, 2 * (j - 1)) = manifold.vertex(j + 1).a(0, 0);
      A(2 * i, 2 * (j - 1) + 1) = manifold.vertex(j + 1).a(0, 1);
      A(2 * i + 1, 2 * (j - 1)) = manifold.vertex(j + 1).a(1, 0);
      A(2 * i + 1, 2 * (j - 1) + 1) = manifold.vertex(j + 1).a(1, 1);
    }
    A(2 * i, 2 * j) = manifold.vertex(j + 1).b(0, 0);
    A(2 * i, 2 * j + 1) = manifold.vertex(j + 1).b(0, 1);
    A(2 * i + 1, 2 * j) = manifold.vertex(j + 1).b(1, 0);
    A(2 * i + 1, 2 * j + 1) = manifold.vertex(j + 1).b(1, 1);

    if (j + 1 < numConstraints) {
      A(2 * i, 2 * (j + 1)) = manifold.vertex(j + 1).c(0, 0);
      A(2 * i, 2 * (j + 1) + 1) = manifold.vertex(j + 1).c(0, 1);
      A(2 * i + 1, 2 * (j + 1)) = manifold.vertex(j + 1).c(1, 0);
      A(2 * i + 1, 2 * (j + 1) + 1) = manifold.vertex(j + 1).c(1, 1);
    }
    ++i;
  }
}
}
#endif

static inline Float computeStepsize(const Float roughness) {
    return /*std::sqrt*/(roughness * (Float)(0.797884560802865356)); /* sqrt(2/M_PI) ~= 0.7978.. */
}

static inline void cacheRoughness(const ref<PathManifold>& m, HalfvectorPerturbation::CachedTransitionPdf& cachedPdf) {
    const int numConstraints = (int)m->size() - 2;
    cachedPdf.vtxRoughtness.resize(numConstraints);
    cachedPdf.totalRoughness = 0;
    for(int j = 0;j<numConstraints;++j)
    {
        const Float roughness = m->vertex(j+1).roughness;
        cachedPdf.vtxRoughtness[j] = roughness;
        cachedPdf.totalRoughness += roughness;
    }
}

static inline Float nonspecularProb(Float alpha, const Vector& h, const Point2& img_sample, Vector2 dh_du, Vector2 dh_dv, const Vector2& meanImageJump) {
	if (alpha == std::numeric_limits<Float>::infinity() || h.z == 0.f)
		return 1.f;
	else if(alpha == 0.f)
		return 0.f;
	
	/* Compute dh as a maximum possible step */
	const Float dh = dot(dh_du, dh_dv) >= 0.f ? (dh_du+dh_dv).length() : (dh_du-dh_dv).length();
	
	/* Distance to the current position in plane-plane half vector space */
	const Float curH = hypotf(h.x, h.y) / abs(h.z);
	
	/* The ratio of two 2D Gaussian functions, written as a difference of the exponents */
	const Float dLenSq = ((curH+dh)*(curH+dh) - curH*curH);
	
	/* Compute the actual acceptance probability ratio of Gaussians as an estimate of change in the measurement contribution */
	Float a = math::fastexp(-dLenSq/(alpha*alpha));
	BDAssert(a >= 0.f && a <= 1.f);
	
	/* Estimate image-space feature size*/
	Float featureArea = 0.f;
	{
		/* Undo the scaling of derivatives */
		dh_du *= meanImageJump.x; dh_dv *= meanImageJump.y;
		
		/* Invert from hv space to image space */
		const Matrix2x2 dh_duv(dh_du[0], dh_dv[0],
							   dh_du[1], dh_dv[1]);
		Matrix2x2 duv_dh;
		if(!dh_duv.invert(duv_dh))
			return 1.f;
		
		/* Gaussian's singular values to clamp against the image plane */
		const Vector2 du_dh(duv_dh(0,0), duv_dh(0,1)),
					  dv_dh(duv_dh(1,0), duv_dh(1,1));
		
		/* Image-space extents of an ellipse inscribed into Gaussian */
		// TODO: what Gaussian percentile for ellipse?
		const Point2 du0 = img_sample - du_dh, du1 = img_sample + du_dh,
					 dv0 = img_sample - dv_dh, dv1 = img_sample + dv_dh;
		
		// TODO: clip by normalized image plane
		
		/* Ellipse axes and area */
		const Float el_a = 0.5f * (du1 - du0).length(),
					el_b = 0.5f * (dv1 - dv0).length();
		// TODO: apply aspect ratio compression
		featureArea = M_PI * el_a * el_b;
		//BDAssert(featureArea >= 0.f && featureArea <= 1.f);
	}
	
	/* Enforce more breakups for smaller (in image space) features */
	const Float prob_jumpoff = 1.f - math::clamp(featureArea, 0.f, 1.f);
	a = max(prob_jumpoff, a);
	
	return a;
}

bool HalfvectorPerturbation::computeBreakupProbabilities(const Path& path, vector_rd& rayDiffs) const {
	const int k = path.length();
	const int numConstraints = k - 3;
	m_breakupPmf.clear();

    const bool fullHSLT = true;//source.length() == numConstraints + 3;

	/* Compute all derivatives along the path */
	m_manifold->init(path, 1, k);
	if(!m_manifold->computeDerivatives()) {
		Log(EWarn, "Found a non-manifold path: %s", path.toString().c_str());
		return false;
	}

	/* Support only perspective camera for now */
	const Sensor* sensor = m_scene->getSensor();
	BDAssert(!sensor->needsApertureSample());
	BDAssert(!sensor->needsTimeSample());

	/* Take last vertex and its image plane position */
	const PathVertex* vs = path.vertex(k-1);

	/* Compute ray differential from the first vertex */
	sensor->sampleRayDifferential(m_rayDifferential, vs->getSamplePosition(), Point2(0.5f, 0.5f), Float(0.5f));
  
	/* Scale differential according to 2% change on the image plane */
	static const Float imgStepSize = 0.02f;  // 2% pixels to jump on average
	const Vector2 imageRes((Float)sensor->getFilm()->getCropSize().x, (Float)sensor->getFilm()->getCropSize().y);
	const Vector2 meanImageJump = imageRes * imgStepSize;
	m_rayDifferential.scaleDifferentialUV(meanImageJump);

	/* Compute dx(k-2) - differential offset at the next vertex after camera */
	const PathVertex* vt = path.vertex(k-2);
	Intersection its = vt->getIntersection();

	/* Compute differentials of v2 in dpdu/dpdv space of the vertex */
	its.computePartials(m_rayDifferential);
	Vector2 dv2_du(its.dudx, its.dvdx);
	Vector2 dv2_dv(its.dudy, its.dvdy);

	rayDiffs.resize(numConstraints);
	// based on LU decomposition:
	const int n = numConstraints+1; // last vertex

	Matrix2x2* Li = (Matrix2x2*)alloca((n+1)*sizeof(Matrix2x2));
	Matrix2x2* A = (Matrix2x2*)alloca((n+1)*sizeof(Matrix2x2));
	Matrix2x2* H = (Matrix2x2*)alloca((n+1)*sizeof(Matrix2x2));

	// first pass: compute and store Li and A
	if(!m_manifold->vertex(n-1).b.invert2x2(Li[n-1]))
		return false;
	for(int k=n-2;k>0;k--) {
		A[k] = m_manifold->vertex(k).c * Li[k+1];
		// L = b - au
		const Matrix2x2 t = m_manifold->vertex(k).b - A[k]*m_manifold->vertex(k+1).a;
		if(!(t.invert2x2(Li[k])))
			return false;
	}

#if USE_EIGEN_TEST == 1
	Eigen::MatrixXd M(2 * numConstraints, 2 * numConstraints);
	filloutConstraintMatrix(*m_manifold, M);

	// Check if the matrix is invertible
	const double det = M.determinant();
	if (std::abs(det) < 1e-50)
		return false;

	// Compute the inverse 
	const Eigen::MatrixXd Mi = M.inverse();
#endif
	
	/* Construct breakup pdf */
	m_breakupPmf.reserve(numConstraints+1);
	Float breakup_sum = 0.f;

	/* Transform offset at v2 for a step in pixel i and pixel j coordinates
	   to half vector at position c */
	for(int hi=1;hi<1+numConstraints;hi++) {

#if USE_EIGEN_TEST == 1
		// Take block from the last row of Mi (corresponding to the first surface vertex from camera)
		const Eigen::Matrix2d Bi = Mi.block(2 * (numConstraints-1), 2 * (hi-1), 2, 2);
		const Eigen::Matrix2d reference_x_to_h = Bi.inverse();
#endif

		// now finding the block x1 -> h[hi].
		// we pretend we have some input matrices h[.] = 0 except h[hi] = I.
		memset(H, 0, sizeof(Matrix2x2)*(n+1));
		H[hi].setIdentity();

		for(int k=hi-1;k>0;k--)
			H[k] = -A[k] * H[k+1];

		Matrix2x2 x, x_to_h;
		const int fixed_vertex = n-1; // the one just after the camera
		x = Li[1] * H[1]; // x[1]
		for(int k=2;k<=fixed_vertex;k++) // would be needed if tracing from the eye (or put the rest upside down)
			x = Li[k] * (H[k] - m_manifold->vertex(k).a * x); // x[k] = (.. x[k-1])

		// now x is the matrix that maps h[hi] -> x[fixed_vertex]. we need to invert that.
		// note that if(numConstraints == 1): x_to_h = m_manifold->vertex(1).b;
		if(!x.invert2x2(x_to_h))
			return false;

		// now we're not interested in the matrix x_to_h, but in a rotation matrix R
		// plus two eigenvalues which together span the ellipse that defines the ray differential
		// step in the space of this half vector (which is already plane/plane space,
		// i.e. the Beckmann space of slopes).

		// these are the projected half vector steps when changing pixel u/v,
		// i.e. these are first-order differentials:
		const Vector2 hu = x_to_h * dv2_du;
		const Vector2 hv = x_to_h * dv2_dv;
		
#if USE_EIGEN_TEST == 1
		BDAssert(fabs(x_to_h(0,0) / reference_x_to_h(0,0) - 1.0) < 5e-1f);
		BDAssert(fabs(x_to_h(0,1) / reference_x_to_h(0,1) - 1.0) < 5e-1f);
		BDAssert(fabs(x_to_h(1,0) / reference_x_to_h(1,0) - 1.0) < 5e-1f);
		BDAssert(fabs(x_to_h(1,1) / reference_x_to_h(1,1) - 1.0) < 5e-1f);
#endif

		// NaN check with fallback to iso rd!
		BDAssert(hu[0] == hu[0]);
		BDAssert(hu[1] == hu[1]);
		BDAssert(hv[0] == hv[0]);
		BDAssert(hv[1] == hv[1]);

		// store that as R0's column vectors, so R0 will transform
		// one-pixel deltas in screen space to half vectors in ray diff rotation in tangent space
		Matrix2x2 R0(hu[0], hv[0],
					 hu[1], hv[1]);

		// now R0 is the matrix that transforms single-pixel steps in screen space
		// to p/p half vector offsets at vertex v[c+1]. the transformed basis vectors (column vectors
		// of R0) are non-orthogonal and not normalised.
		//
		// for the anisotropy, we'd like to know the main axes of the ellipse (one std dev from
		// the center), because we can adjust this later on according to the bsdf without
		// accidentally rotating the main axes again.
		//
		// perform an incomplete 2x2 svd on R0:
		// find U such that U * R0 * R0’ * U’=diag
		// Su = R0*R0'
		Matrix2x2 Rp, Su;
		R0.transpose(Rp);
		Su = R0 * Rp;

		// this works, but involves cumbersome trig:
		const Float phi = - 0.5f * std::atan2(Su(0,1) + Su(1,0), Su(0,0) - Su(1,1));
		const Float sin_phi = std::sin(phi), cos_phi = std::cos(phi);
		BDAssert(phi == phi);
		BDAssert(sin_phi == sin_phi);
		BDAssert(cos_phi == cos_phi);

		// write back new rotation matrix (= U')
		rayDiffs[hi-1].R = Matrix2x2(
			cos_phi,  sin_phi,
			-sin_phi, cos_phi);

		// find the singular values from U
		const Float Su_sum = Su(0,0) + Su(1,1);
		const Float Su_dif = math::safe_sqrt((Su(0,0)-Su(1,1))*(Su(0,0)-Su(1,1)) + 4.0f*Su(0,1)*Su(1,0));
		// now the two axes will have anisotropic step lengths:
		rayDiffs[hi-1].v = Vector2(
			sqrt(max(Float(1e-25), (Su_sum + Su_dif)*Float(0.5))),
			sqrt(max(Float(1e-25), (Su_sum - Su_dif)*Float(0.5))));
		
		/* Compute the break-up probabilities */
		const Vector2 dh_du = rayDiffs[hi-1].R * Vector2(rayDiffs[hi-1].v.x, 0.f),
				      dh_dv = rayDiffs[hi-1].R * Vector2(0.f, rayDiffs[hi-1].v.y);
		// used to decide which vertex to break up the path into lens subchain (a,b) and
		// half vector perturbation sub chain (b,c):
		const Point2 smpPos(vs->getSamplePosition().x / imageRes.x, vs->getSamplePosition().y / imageRes.y);
		const Float p = nonspecularProb(m_manifold->vertex(hi).roughness, m_manifold->vertex(hi).m, smpPos, dh_du, dh_dv, meanImageJump);
		m_breakupPmf.append(p);
		breakup_sum += p;

        ///* Do not use ray differentials in case of partial HSLT */
        //if(!fullHSLT) {
        //    rayDiffs[hi-1].v = Vector2(1e30f, 1e30f);
        //    rayDiffs[hi-1].R.setIdentity();
        //}
	}
	
	/* Add the expected HSLT probability: sample a vertex, then p_hslt = 1 - p_vtx */
	Float hslt_p = 0.f;
	if(breakup_sum <= RCPOVERFLOW)
		hslt_p = 1.f;
	else {
		for(size_t i=0;i<numConstraints;++i) {
			const Float p = m_breakupPmf[i];
			hslt_p += max(0.f, 1.f - p) * (p / breakup_sum);
		}
		//BDAssert(hslt_p >= 0.f && hslt_p <= 1.f);
	}
	m_breakupPmf.append(hslt_p*(breakup_sum+hslt_p));
	m_breakupPmf.normalize();
	
	return true;
}

bool HalfvectorPerturbation::sampleSubpath(const Path& path, int& a, int& b, int& c) {
	const int k = path.length();
	const int numVerts = k+1;
	
	/* Compute ray differentials in half-vector space */
	if(!computeBreakupProbabilities(path, m_fwPdf.halfvectorDifferentials))
		return false;
	BDAssert(m_breakupPmf.getSum() > 0.f);
	
	// TODO: lens perturbation from a to b, h-v mutation of a subpath from b to c
	a = numVerts-2;
	c = 1;
	
	b = 2 + (int)m_breakupPmf.sample(m_sampler->next1D());
	
	//if(b < a-1 && breakupPmf[a-1-2]*breakupPmf.getSum() < 0.1f)
	//	b = a;
	
	//b = max(c+1, a-3);
	//b = a;
	
	BDAssert(path.vertex(b)->isConnectable());
	
	if(!path.vertex(b)->isConnectable())
		return false;
	
	return true;
}

bool HalfvectorPerturbation::perturbEndpoints(Path& path) {
	/* Perturb the vertex on the light if needed */
	PathVertex *vertex = path.vertex(0);
	PathVertex *succ = path.vertex(1);
	if(vertex->isConnectable() && succ->isConnectable()) {
		const Float pdf = vertex->pdf[EImportance] * m_probFactor * m_probFactor;
		const Float stddev = 1.0f / sqrt(2*M_PI * pdf);
		if(!succ->perturbPosition(m_scene, m_sampler, stddev))
			return false;

		vertex->update(m_scene, NULL, succ, EImportance, (EMeasure)succ->measure);
	}

	return true;
}

bool HalfvectorPerturbation::perturbLensSubpath(const Path& source, Path& proposal, const int a, const int b) {
	/* Resample the end point at the light */
	if(!perturbEndpoints(proposal))
		return false;

	if(a == b)
		return true;

	const int k = source.length();
	const int step = -1;
	const ETransportMode mode = (step == 1) ? EImportance : ERadiance;
	
	if (a != 0 && a != k) {
		/* Sample the first vertex */
		const PathVertex
		*pred_old     = source.vertex(a-step),
		*vertex_old   = source.vertex(a),
		*succ_old     = source.vertex(a+step);
		const PathEdge
		*succEdge_old = source.edge(mode == EImportance ? a : a-1);
		PathVertex
		*pred         = proposal.vertex(a-step),
		*vertex       = proposal.vertex(a),
		*succ         = proposal.vertex(a+step);
		PathEdge
		*predEdge     = proposal.edge(mode == EImportance ? a-step : a-1-step),
		*succEdge     = proposal.edge(mode == EImportance ? a : a-1);
		
		Float prob_old = std::max(INV_FOURPI, vertex_old->evalPdf(m_scene, pred_old, succ_old, mode, ESolidAngle));
		
		VonMisesFisherDistr vMF(VonMisesFisherDistr::forPeakValue(prob_old * m_probFactor * m_probFactor));
		
		Vector sampled = vMF.sample(m_sampler->next2D());
		Vector wo_old = normalize(succ_old->getPosition()
								  - vertex_old->getPosition());
		Vector wo_new = Frame(wo_old).toWorld(sampled);
		
		if (!vertex->perturbDirection(m_scene, pred, predEdge, succEdge, succ, wo_new,
									  succEdge_old->length, succ_old->getType(), mode)) {
			return false;
		}
	} else {
		const PathVertex *vertex_old = source.vertex(a);
		const PathVertex *succ_old = source.vertex(a+step);
		if (!succ_old->isConnectable())
			return false;
		
		PathVertex *vertex = proposal.vertex(a);
		PathVertex *succ = proposal.vertex(a+step);
		const PathEdge *succEdge_old = source.edge(mode == EImportance ? a : a-1);
		PathEdge *succEdge = proposal.edge(mode == EImportance ? a : a-1);
		
		*succ = *succ_old;
		*succEdge = *succEdge_old;
		
		Float pdf = vertex_old->pdf[mode] * m_probFactor * m_probFactor;
		Float stddev = 1.0f / std::sqrt(2*M_PI * pdf);
		if (!succ->perturbPosition(m_scene, m_sampler, stddev))
			return false;
		
		vertex->update(m_scene, NULL, succ, mode, (EMeasure) succ_old->measure);
	}
	
	/* Generate subsequent vertices between a .. b deterministically */
	if(!perturbSubpath(source, proposal, a, b, 0))
		return false;
	
	return true;
}

bool HalfvectorPerturbation::perturbHalfvectors(vector_hv& hs, const Path& source, const int b, const int c) {
	const int numConstraints = (int)hs.size();
	BDAssert(m_manifold->size()-2 == (uint64_t)numConstraints);
	
	/* Scale dependent on the distribution of roughness along the path */
	const Float totalRoughness = m_fwPdf.totalRoughness;
	
	/* Perturb half-vector constraints */
	for(int j=0;j<numConstraints;++j) {
		const PathManifold::SimpleVertex& vm = m_manifold->vertex(j+1);
		
		/* Reset to 0 to avoid accumulated drift from previous mutations */
		if(vm.degenerate) {
			hs[j] = Vector2(0, 0);
			continue;
		}
		
#if 1
		const Float stepsize = computeStepsize(vm.roughness);
		const Float rdWeight = vm.roughness / totalRoughness;
		const Float stdv0 = min(m_fwPdf.halfvectorDifferentials[j].v.x * rdWeight, stepsize);
		const Float stdv1 = min(m_fwPdf.halfvectorDifferentials[j].v.y * rdWeight, stepsize);
		Point2 g = warp::squareToStdNormal(m_sampler->next2D());
		Vector2 d(g.x * stdv0, g.y * stdv1);
		hs[j] += m_fwPdf.halfvectorDifferentials[j].R * d;
#else
		/* DEBUG: Constant jump sizes */
		const Float r2 = 0.005f;
		const Point2 offset = warp::squareToUniformDisk(m_sampler->next2D()) * r2;
		
		/* Generate an offset for half vector constraint */
		//const Float r = r2 * std::pow(r2/r1, -m_sampler->next1D());
		//const Float phi = m_sampler->next1D() * 2 * M_PI;
		//const Vector2 offset(r*std::cos(phi), r*std::sin(phi));
		hs[j] += Vector2(offset);
#endif
	}
	
	return true;
}

bool HalfvectorPerturbation::sampleMutation(
		Path &source, Path &proposal, MutationRecord &muRec, const MutationRecord& sourceMuRec) {
	const int k = source.length();

	statsAccepted.incrementBase();
	statsGenerated.incrementBase();
	
	// TODO: Remove. remember pointer to current so we know which array is for which path in Q():
	m_current = &source;
	
	PathVertex *v_p, *vs, *vt, *v_n;
	PathEdge *e_p, *e, *e_n;
	bool vs_reg, vt_reg;
	
	/* Sample desired subpath mutation */
	int a, b, c;
	statsFailLensSubpath.incrementBase();
	if(!sampleSubpath(source, a, b, c)) {
		++statsFailLensSubpath;
		return false;
	}
	muRec = MutationRecord(EHalfvectorPerturbation, 0, k, k, Spectrum(1.f));
	muRec.extra[0] = a; muRec.extra[1] = b; muRec.extra[2] = c;
	
	/* Cache source (forward) breakup probability */
	m_fwPdf.breakupPdf = 1.f;
	if(m_breakupPmf.getSum() != 0.f)
		m_fwPdf.breakupPdf = m_breakupPmf[b-2];
	
	// TODO: trim and init b, c only
	m_manifold->init(source, c, b);

	/* Cache source (forward) values */
	cacheRoughness(m_manifold, m_fwPdf);
	m_manifold->computeDerivatives();
	m_manifold->computeTransferMatrices();
	m_fwPdf.transferMx = m_manifold->fullG(source);

	/* Remember source half vectors and convert them into plane-plane domain */
	const int numConstraints = b-c-1;
	m_h_orig.resize(numConstraints);
	m_h_perturbed.resize(numConstraints);
	for(int j=0;j<numConstraints;++j) {
		const Vector m = m_manifold->vertex(j+1).m;
		m_h_orig[j] = Vector2(m.x, m.y)/m.z;
		m_h_perturbed[j] = m_h_orig[j];
	}

	/* Perturb all half-vectors */
	if(!perturbHalfvectors(m_h_perturbed, source, b, c))
		return false;

	/* Allocate memory for the proposed path */
	proposal.clear();
	source.clone(proposal, m_pool);

	/* Perform a lens perturbation for a selected subpath from a to b */
	if(!perturbLensSubpath(source, proposal, a, b)) {
		++statsFailLensSubpath;
		goto fail;
	}
	
	statsAvgBreakup.incrementBase();
	statsAvgBreakup += b;
	
	/* Set the perturbed end point */
	// TODO: only the 'b' point
	m_manifold->initEndpoints(proposal.vertex(c), proposal.vertex(c+1), proposal.vertex(b-1), proposal.vertex(b));
	if(!m_manifold->computeDerivatives()) {
		Log(EWarn, "Found a non-manifold path: %s", source.toString().c_str());
		goto fail;
	}
	
	/* Try to find the object-space path corresponding to the perturbed configuration of half-vectors */
	if(!m_manifold->find(m_h_perturbed))
		goto fail;

	/* Update the proposed path */
	if(!m_manifold->update(proposal, c, b))
		goto fail;

	/* Connect the last vertex */
	statsConnectionFailed.incrementBase();
	v_p = proposal.vertexOrNull(b-2);
	vs = proposal.vertex(b-1);
	vt = proposal.vertex(b);
	v_n = proposal.vertexOrNull(b+1);
	e_p = proposal.edgeOrNull(b-2);
	e = proposal.edge(b-1);
	e_n = proposal.edgeOrNull(b);
	vs_reg = source.vertex(b-1)->isConnectable();
	vt_reg = source.vertex(b)->isConnectable();
	if(!PathVertex::connect(m_scene,
		v_p, e_p,
		vs, e, vt,
		e_n, v_n,
		vs_reg ? EArea : EDiscrete, vt_reg ? EArea : EDiscrete)) {
		++statsConnectionFailed;
		goto fail;
	}

	/* Cache proposal (backwards) values */
	cacheRoughness(m_manifold, m_bwPdf);
	m_manifold->computeTransferMatrices();
	m_bwPdf.transferMx = m_manifold->fullG(proposal);

	/* Reconstruct original end points and update the derivatives */
	m_manifold->initEndpoints(source.vertex(c), source.vertex(c+1), source.vertex(b-1), source.vertex(b));
	// TODO: Optimize: recompute only a and c matrix derivatives at the post-first and pre-last vertices
	if(!m_manifold->computeDerivatives()) {
		Log(EWarn, "Found a non-manifold reversible path: %s", source.toString().c_str());
		goto fail;
	}

	/* Check reversibility */
	statsNonReversible.incrementBase();
	if(!m_manifold->find(m_h_orig)) {
		++statsNonReversible;
		goto fail;
	}

	/* Check if we landed into the same submanifold */
	for(int j = 1;j<=numConstraints;++j) {
		const Point p0 = source.vertex(c+j)->getPosition();
		const Point p1 = m_manifold->vertex(j).p;
		const Float relerr = (p0 - p1).lengthSquared() / max(p0.x*p0.x, max(p0.y*p0.y, p0.z*p0.z));
		if(relerr > MTS_REL_POS_EPS*MTS_REL_POS_EPS) {
			++statsNonReversible;
			goto fail;
		}
	}

	/* Always update pixel position BEFORE recomputing ray differentials */
	proposal.vertex(k-1)->updateSamplePosition(proposal.vertex(k-2));
	BDAssert(source.matchesConfiguration(proposal));

    /* Cache proposal (backwards) breakups and ray differentials */
    computeBreakupProbabilities(proposal, m_bwPdf.halfvectorDifferentials);
    m_bwPdf.breakupPdf = 1.f;
    if(m_breakupPmf.getSum() != 0.f)
        m_bwPdf.breakupPdf = m_breakupPmf[b-2];

	++statsGenerated;
	return true;
fail:
	proposal.release(m_pool);
	return false;
}

Float HalfvectorPerturbation::Q(const Path &source, const Path &proposal,
		const MutationRecord &muRec) const {
	const int k = proposal.length();
	Spectrum weight = Spectrum(1.0); // Always 1, since we regenerate there is no fixed subpath
	const int a = muRec.extra[0], b = muRec.extra[1], c = muRec.extra[2];
	const int numConstraints = b-c-1;
	
    const CachedTransitionPdf& cachedPdf = (&source == m_current) ? m_bwPdf : m_fwPdf;

	/* Account for breakup probability */
	weight *= cachedPdf.breakupPdf;

	/* Light endpoint perturbation probability */
	if(source.vertex(0)->isConnectable() && source.vertex(1)->isConnectable()) {
		Frame frame(source.vertex(1)->getGeometricNormal());
		Float stddev = 1.0f / sqrt(2*M_PI * source.vertex(0)->pdf[EImportance] * m_probFactor * m_probFactor);
		Float pdf = source.vertex(1)->perturbPositionPdf(proposal.vertex(1), stddev);
		if(pdf <= RCPOVERFLOW)
			goto q_failed;
		weight /= pdf;
	}
	
	/* Lens mutation probability */
	if(a != b)
	{
		const int step = -1;
		const ETransportMode mode = (step == 1) ? EImportance : ERadiance;
		if (a != 0 && a != k) {
			/* Compute the density of the first vertex */
			const PathVertex
				*pred         = source.vertex(a-step),
				*vertex       = source.vertex(a),
				*succ_old     = source.vertex(a+step),
				*succ_new     = proposal.vertex(a+step);
			
			/* Compute outgoing density wrt. proj. SA measure */
			Float prob_old = std::max(INV_FOURPI, vertex->evalPdf(m_scene.get(), pred, succ_old, mode, ESolidAngle));
			Vector wo_old = normalize(succ_old->getPosition() - vertex->getPosition());
			Vector wo_new = normalize(succ_new->getPosition() - vertex->getPosition());
			Float dp = dot(wo_old, wo_new);
			VonMisesFisherDistr vMF(VonMisesFisherDistr::forPeakValue(prob_old * m_probFactor * m_probFactor));
			Float prob = vMF.eval(dp);
			if (vertex->isOnSurface())
				prob /= absDot(wo_new, vertex->getShadingNormal());

			if (prob <= RCPOVERFLOW)
				goto q_failed;
			weight /= prob;
			
			/* Catch very low probabilities which round to +inf in the above division operation */
			if (!std::isfinite(weight.average()))
				goto q_failed;
		} else {
			Frame frame(source.vertex(a+step)->getGeometricNormal());
			Float stddev = 1.0f / std::sqrt(2*M_PI * source.vertex(a)->pdf[mode] * m_probFactor * m_probFactor);
			Float pdf = source.vertex(a+step)->perturbPositionPdf(proposal.vertex(a+step), stddev);
			if (pdf <= RCPOVERFLOW)
				goto q_failed;
			weight /= pdf;
		}
	}

	/* Compute the simplified measurement contribution function in half-vector space */
	{
		const Float totalRoughness = cachedPdf.totalRoughness;
		
		for(int i=c;i<=a;++i) {
			const PathVertex* vp = proposal.vertex(i-1);
			const PathVertex* v = proposal.vertex(i);
			const PathVertex* vn = proposal.vertex(i+1);
			const PathEdge& edgep = *proposal.edge(i-1);
			const PathEdge& edge = *proposal.edge(i);

			const ETransportMode mode = (i < b) ? EImportance : ERadiance;
			
			/* BSDF at the vertex without geometric term */
			if(mode == EImportance)
				weight *= edge.evalCached(v, vn, PathEdge::EValueImp | PathEdge::ETransmittance);
			else
				weight *= edgep.evalCached(vp, v, PathEdge::EValueRad | ( i != b ? PathEdge::ETransmittance : 0));
			
			/* Skip fixed end points of the path */
			if(i == c || i == b || i == a)
				continue;
			
			/* For specular paths all the terms degenerate to trivial identity */
			if(!v->isConnectable())
				continue;
			
			/* Apply domain transformations (half vector -> projected outgoing direction) */
			{
				BDAssert(v->isSurfaceInteraction());
				Vector wo = edge.d;
				Vector wi = -edgep.d;
				if(mode == ERadiance)
					std::swap(wi, wo);
				const Vector n = v->getShadingNormal();
				const Vector gn = v->getGeometricNormal();
				Float eta = v->getIntersection().getBSDF()->getEta();
				if(dot(wi, gn) < 0)
					eta = 1 / eta;
				
				/* Compute the half vector */
				Vector h = wi + wo * eta;
				const Float hl = h.length();
				BDAssert(hl != 0.f);
				h /= hl;
				
				/* Transform from projected half vector constraint (do/dh Jacobian)
					Note that dot(h, wi) + eta * dot(h, wo) == hl */
				Float do_dh = hl*hl / (eta*eta*dot(wo, h));
				
				/* Cosines caused by projected solid angle for wo and 'solid angle' -> 'parallel-plane' for h */
				const Float doth = dot(n, h);
				do_dh *= dot(n, wo) * doth * doth * doth;
				
				weight *= abs(do_dh);
			}
			
			/* Now account for perturbation probabilities */
			/* Skip fixed end points of the path */
			if(i > c && i < b)
			{
				const int j = i-c-1; // constraint index

				const Float roughness = cachedPdf.vtxRoughtness[j];
				const Float stepsize = computeStepsize(roughness);
				// shorthands to half vecs:
				Vector2 h, th;
				if (&source == m_current) {
					// would need to divide out forward transition probability current -> tentative T(c->t).
					// ray diff is around proposal, so we can only evaluate T(t->c) here, which we need to multiply.
					h  = m_h_orig[j];
					th = m_h_perturbed[j];
				} else {
                    h = m_h_perturbed[j];
                    th = m_h_orig[j];
				}
				
				// eval Gaussian:
				const Vector2 dh = th - h;
				Matrix2x2 Rinv;
				cachedPdf.halfvectorDifferentials[j].R.transpose(Rinv); // rotation matrix. also no Jacobian due to this.
				const Float rdWeight = roughness / totalRoughness;
				const Vector2 d = Rinv * dh; // distance aligned with main axes of ray differential ellipse in half vector plane/plane space
				const Float stdv0 = min(cachedPdf.halfvectorDifferentials[j].v.x * rdWeight, stepsize);
				const Float stdv1 = min(cachedPdf.halfvectorDifferentials[j].v.y * rdWeight, stepsize);
				const Float gauss = 1.0f/(2.0f*M_PI * stdv0*stdv1) * math::fastexp(-0.5f * (d.x*d.x/(stdv0*stdv0) + d.y*d.y/(stdv1*stdv1)));
				if(gauss < RCPOVERFLOW)
					goto q_failed;
				weight *= gauss;
			}
			BDAssert(weight == weight);
		}

		/* Transform the probability from half-vector space to object area measure for segment b-c */
		weight *= cachedPdf.transferMx;
	}

	Float lumWeight = weight.getLuminance();
	BDAssert(lumWeight >= 0 && (lumWeight == lumWeight));
	if (lumWeight <= RCPOVERFLOW)
		goto q_failed;

	lumWeight = 1.f / lumWeight;
	goto q_succeeded;
q_failed:
	lumWeight = 0.f;
q_succeeded:
	return lumWeight;
}

void HalfvectorPerturbation::accept(const MutationRecord &muRec) {
	++statsAccepted;
}

MTS_IMPLEMENT_CLASS(HalfvectorPerturbation, false, Mutator)

MTS_NAMESPACE_END
