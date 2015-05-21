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

#include <mitsuba/bidir/manifold.h>
#include <mitsuba/bidir/path.h>
#include <mitsuba/core/statistics.h>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_DEBUG
#include <Eigen/LU>
#include <Eigen/Geometry>

#define MTS_MANIFOLD_EPSILON        Epsilon
#define MTS_MANIFOLD_HV_EPSILON     Epsilon

MTS_NAMESPACE_BEGIN

/* Some statistics counters */
static StatsCounter statsStepFailed(
		"Specular manifold", "Retries (step failed)", EPercentage);
static StatsCounter statsStepTooFar(
		"Specular manifold", "Retries (step increased distance)", EPercentage);
static StatsCounter statsStepSuccess(
		"Specular manifold", "Successful steps", EPercentage);
static StatsCounter statsAvgIterations(
		"Specular manifold", "Iterations: Avg. per walk", EAverage);
static StatsCounter statsAvgIterationsSuccess(
		"Specular manifold", "Iterations: Avg. per successful walk", EAverage);
static StatsCounter statsMaxIterationsSuccess(
		"Specular manifold", "Iterations: Max per successful walk", EMaximumValue);
static StatsCounter statsAvgManifoldSize(
		"Specular manifold", "Avg. manifold size", EAverage);
static StatsCounter statsSuccessfulWalks(
		"Specular manifold", "Successful walks", EPercentage);
static StatsCounter statsMediumSuccess(
		"Specular manifold", "Successful walks w/ media", EPercentage);
static StatsCounter statsNonManifold(
		"Specular manifold", "Non-manifold", EPercentage);
static StatsCounter statsUpdateFailed(
		"Specular manifold", "Update failed", EPercentage);
static StatsCounter statsMaxManifold(
		"Specular manifold", "Max. manifold size", EMaximumValue);
static StatsCounter statsFullMxInversion(
		"Specular manifold", "Full matrix inversion", EPercentage);

PathManifold::PathManifold(const Scene *scene, int maxIterations)
  : m_scene(scene), m_det(1.f) {
	m_maxIterations = maxIterations > 0 ? maxIterations :
		MTS_MANIFOLD_MAX_ITERATIONS;
}

bool PathManifold::init(const Path &path, int start, int end) {
	int step = start < end ? 1 : -1;
	if (path.vertex(start)->isSupernode())
		start += step;
	if (path.vertex(end)->isSupernode())
		end -= step;

	const PathVertex
		*vs = path.vertex(start),
		*ve = path.vertex(end);

	m_time = vs->getTime();
	m_vertices.resize(std::abs(end-start)+1);

	initEndpoints(vs, path.vertex(start+step), path.vertex(end-step), ve);

	for (int i=start + step, iv=1; i != end; i += step, ++iv) {
		const PathVertex
			*pred = path.vertex(i-step),
			*vertex = path.vertex(i),
			*succ = path.vertex(i+step);
		SimpleVertex& v = m_vertices[iv];

		if (vertex->isSurfaceInteraction()) {
			const Intersection &its = vertex->getIntersection();
			const BSDF *bsdf = its.getBSDF();

			v.p = its.p;
			v.dpdu = its.dpdu;
			v.dpdv = its.dpdv;
			v.object = bsdf;
			v.degenerate = !vertex->isConnectable();
			bsdf->getFrame(its, v.shFrame);
			bsdf->getFrameDerivative(its, v.shFrameDu, v.shFrameDv);

			v.roughness = getEffectiveRoughness(vertex);

			Vector wPred = pred->getPosition() - v.p;
			Vector wSucc = succ->getPosition() - v.p;

			if (dot(v.shFrame.n, wPred) * dot(v.shFrame.n, wSucc) < 0) {
				v.type = ERefraction;
				v.eta = bsdf->getEta();
			} else {
				v.type = EReflection;
				v.eta = 1.0f;
			}
		} else if (vertex->isMediumInteraction()) {
			const MediumSamplingRecord &mRec = vertex->getMediumSamplingRecord();

			Vector wi = pred->getPosition() - mRec.p;
			Float invLength = 1.0f / wi.length();
			wi *= invLength;

			/* For medium, take a mean cosine of the phase function as a mean "roughness" */
			v.roughness = 1 - std::abs(mRec.getPhaseFunction()->getMeanCosine());

			v.p = mRec.p;
			v.shFrame.n = Normal(wi);

			coordinateSystem(wi, v.shFrame.s, v.shFrame.t);

			v.dpdu = v.shFrame.s;
			v.dpdv = v.shFrame.t;
			v.shFrameDu.n = v.shFrame.s * invLength;
			v.shFrameDv.n = v.shFrame.t * invLength;

			v.object = mRec.getPhaseFunction();
			v.eta = 1.0f;
			v.degenerate = false;
			v.type = EMedium;
		} else {
			Log(EError, "Unknown vertex type! : %s", vertex->toString().c_str());
		}
	}

	#if MTS_MANIFOLD_DEBUG == 1
		cout << "==========================================" << endl;
		cout << "Initialized path manifold: " << toString() << endl;
	#endif

	return true;
}

void PathManifold::initEndpoints(const PathVertex* vs, const PathVertex* vpred_s, const PathVertex* vpred_e, const PathVertex* ve) {
	/* Create the initial vertex that is pinned in position by default */
	{
		SimpleVertex& v = m_vertices[0];
		v.type = EPinnedPosition;
		v.p = vs->getPosition();
		v.degenerate = false;

		/* When the endpoint is on an orthographic camera or directional light
		   source, switch to a directionally pinned vertex instead */
		if (vs->getType() & (PathVertex::ESensorSample | PathVertex::EEmitterSample)) {
			const PositionSamplingRecord &pRec = vs->getPositionSamplingRecord();
			uint32_t type = static_cast<const AbstractEmitter *>(pRec.object)->getType();
			if (type & AbstractEmitter::EDeltaDirection)
				v.type = EPinnedDirection;

			v.shFrame = Frame(!pRec.n.isZero() ?
					pRec.n : Normal(normalize(vpred_s->getPosition() - vs->getPosition())));
		} else {
			BDAssert(vs->isSurfaceInteraction() || vs->isMediumInteraction());
			if (!vs->isMediumInteraction())
				vs->getIntersection().getBSDF()->getFrame(vs->getIntersection(), v.shFrame);
			else
				v.shFrame = Frame(normalize(vpred_s->getPosition() - vs->getPosition()));
		}
		v.dpdu = v.shFrame.s;
		v.dpdv = v.shFrame.t;
	}

	/* Last vertex */
	{
		SimpleVertex& v = m_vertices.back();
		v.type = EMovable;
		v.p = ve->getPosition();
		v.degenerate = false;

		if (ve->getType() & (PathVertex::ESensorSample | PathVertex::EEmitterSample)) {
			const PositionSamplingRecord &pRec = ve->getPositionSamplingRecord();
			uint32_t type = static_cast<const AbstractEmitter *>(pRec.object)->getType();
			if (type & AbstractEmitter::EDeltaDirection)
				v.type = EPinnedDirection;
			v.shFrame = Frame(!pRec.n.isZero() ?
					pRec.n : Normal(normalize(vpred_e->getPosition() - ve->getPosition())));
		} else {
			BDAssert(ve->isSurfaceInteraction() || ve->isMediumInteraction());
			if (!ve->isMediumInteraction())
				ve->getIntersection().getBSDF()->getFrame(ve->getIntersection(), v.shFrame);
			else
				v.shFrame = Frame(normalize(vpred_e->getPosition() - ve->getPosition()));
		}
		v.dpdu = v.shFrame.s;
		v.dpdv = v.shFrame.t;
	}
}

bool PathManifold::computeDerivatives() {
	const int n = static_cast<int>(m_vertices.size()) - 1;

	m_vertices[0].Tp.setZero();
	m_vertices[n].Tp.setIdentity();

	if (n == 1) /* Nothing to do */
		return true;

	/* Matrix assembly stage */
	for (int i=0; i<n; ++i) {
		SimpleVertex *v = &m_vertices[i];

		Vector wo = v[1].p - v[0].p;
		Float ilo = wo.length();

		if (ilo == 0)
			return false;
		ilo = 1/ilo; wo *= ilo;

		if (v[0].type == EPinnedPosition) {
			v[0].a.setZero();
			v[0].b.setIdentity();
			v[0].c.setZero();
			continue;
		} else if (v[0].type == EPinnedDirection) {
			Vector dC_dnext_u = (v[1].dpdu - wo * dot(wo, v[1].dpdu)) * ilo;
			Vector dC_dnext_v = (v[1].dpdv - wo * dot(wo, v[1].dpdv)) * ilo;
			Vector dC_dcur_u = (wo * dot(wo, v[0].dpdu) - v[0].dpdu) * ilo;
			Vector dC_dcur_v = (wo * dot(wo, v[0].dpdv) - v[0].dpdv) * ilo;

			v[0].a.setZero();
			v[0].b = Matrix2x2(
				Vector2(dot(dC_dcur_u, v[0].dpdu), dot(dC_dcur_u, v[0].dpdv)),
				Vector2(dot(dC_dcur_v, v[0].dpdu), dot(dC_dcur_v, v[0].dpdv))
			);
			v[0].c = Matrix2x2(
				Vector2(dot(dC_dnext_u, v[0].dpdu), dot(dC_dnext_u, v[0].dpdv)),
				Vector2(dot(dC_dnext_v, v[0].dpdu), dot(dC_dnext_v, v[0].dpdv))
			);
			continue;
		}

		Vector wi = v[-1].p - v[0].p;
		Float ili = wi.length();

		if (ili == 0)
			return false;

		ili = 1/ili; wi *= ili;

		if (v[0].type == EReflection || v[0].type == ERefraction) {
			Float eta = v[0].eta;
			if (dot(wi, v[0].shFrame.n) < 0)
				eta = 1 / eta;

			/* Compute the half vector and a few useful projections */
			Vector H = wi + eta * wo;
			const bool indexMatched = v[0].type == ERefraction && eta == 1.0f;
			if (!indexMatched) {
				/* Generally compute derivatives with respect to the normalized
				   half-vector. When given an index-matched refraction event,
				   don't perform this normalization, since the desired vertex
				   configuration is actually where H = 0.
				   Otherwise compute derivatives for unnormalized half vector.
				   For normalized one they are infinite in this case. */
				Float ilh = dot(v[0].shFrame.n, H);
				if (ilh == 0) {
					Log(EWarn, "On-surface vertex has wrong configuration:\n%s\n", v[0].toString().c_str());
					return false;	/* Report as a non-manifold in this case. Usually wrong path. */
				}
				ilh = 1 / ilh;
				H *= ilh;
				ilo *= eta * ilh; ili *= ilh;
			}

			/* Local shading tangent frame */
			const Normal n = v[0].shFrame.n;
			const Vector s = v[0].shFrame.s;
			const Vector t = v[0].shFrame.t;

			/* Derivatives of C with respect to x_{i-1} */
			Vector dH_du = (v[-1].dpdu - wi * dot(wi, v[-1].dpdu)) * ili,
			       dH_dv = (v[-1].dpdv - wi * dot(wi, v[-1].dpdv)) * ili;

			if (!indexMatched) {
				dH_du -= H * dot(dH_du, n);
				dH_dv -= H * dot(dH_dv, n);
			}

			v[0].a = Matrix2x2(
				dot(dH_du, s), dot(dH_dv, s),
				dot(dH_du, t), dot(dH_dv, t));

			/* Derivatives of C with respect to x_i */
			dH_du = -v[0].dpdu * (ili + ilo) + wi * (dot(wi, v[0].dpdu) * ili)
			                                 + wo * (dot(wo, v[0].dpdu) * ilo);
			dH_dv = -v[0].dpdv * (ili + ilo) + wi * (dot(wi, v[0].dpdv) * ili)
			                                 + wo * (dot(wo, v[0].dpdv) * ilo);

			if (!indexMatched) {
				dH_du -= H * (dot(H, v[0].shFrameDu.n) + dot(dH_du, n));
				dH_dv -= H * (dot(H, v[0].shFrameDv.n) + dot(dH_dv, n));
			}

			/* (h * T)' = h' * T + h * T' */
			v[0].b = Matrix2x2(
				dot(dH_du, s) + dot(H, v[0].shFrameDu.s),   // ds/du
				dot(dH_dv, s) + dot(H, v[0].shFrameDv.s),   // ds/dv
				dot(dH_du, t) + dot(H, v[0].shFrameDu.t),   // dt/du
				dot(dH_dv, t) + dot(H, v[0].shFrameDv.t));  // dt/dv

			/* Derivatives of C with respect to x_{i+1} */
			dH_du = (v[1].dpdu - wo * dot(wo, v[1].dpdu)) * ilo;
			dH_dv = (v[1].dpdv - wo * dot(wo, v[1].dpdv)) * ilo;

			if (!indexMatched) {
				dH_du -= H * dot(dH_du, n);
				dH_dv -= H * dot(dH_dv, n);
			}

			v[0].c = Matrix2x2(
				dot(dH_du, s), dot(dH_dv, s),
				dot(dH_du, t), dot(dH_dv, t));

			/* Store the microfacet normal wrt. the local (orthonormal) shading frame */
			Vector m = v[0].shFrame.toLocal(indexMatched ? H : normalize(H));
			v[0].m = m * math::signum(Frame::cosTheta(m));
		} else if (v[0].type == EMedium) {
			Vector dwi_dpred_u = (v[-1].dpdu - wi * dot(wi, v[-1].dpdu)) * ili;
			Vector dwi_dpred_v = (v[-1].dpdv - wi * dot(wi, v[-1].dpdv)) * ili;
			Vector dwi_dcur_u  = (-v[0].dpdu + wi * dot(wi, v[ 0].dpdu)) * ili;
			Vector dwi_dcur_v  = (-v[0].dpdv + wi * dot(wi, v[ 0].dpdv)) * ili;

			Vector t, dt_dpred_u, dt_dpred_v, dt_dcur_u, dt_dcur_v;

			/* Compute the local frame and derivatives thereof */
			if (std::abs(wi.x) > std::abs(wi.y)) {
				Float tl = 1.0f / std::sqrt(wi.x * wi.x + wi.z * wi.z);
				t = Vector(wi.z * tl, 0.0f, -wi.x * tl);

				dt_dpred_u = Vector(dwi_dpred_u.z*tl, 0.0f, -dwi_dpred_u.x*tl);
				dt_dpred_v = Vector(dwi_dpred_v.z*tl, 0.0f, -dwi_dpred_v.x*tl);
				dt_dcur_u  = Vector(dwi_dcur_u.z*tl,  0.0f, -dwi_dcur_u.x*tl);
				dt_dcur_v  = Vector(dwi_dcur_v.z*tl,  0.0f, -dwi_dcur_v.x*tl);
			} else {
				Float tl = 1.0f / std::sqrt(wi.y * wi.y + wi.z * wi.z);
				t = Vector(0.0f, wi.z * tl, -wi.y * tl);

				dt_dpred_u = Vector(0.0f, dwi_dpred_u.z*tl, -dwi_dpred_u.y*tl);
				dt_dpred_v = Vector(0.0f, dwi_dpred_v.z*tl, -dwi_dpred_v.y*tl);
				dt_dcur_u  = Vector(0.0f, dwi_dcur_u.z*tl,  -dwi_dcur_u.y*tl);
				dt_dcur_v  = Vector(0.0f, dwi_dcur_v.z*tl,  -dwi_dcur_v.y*tl);
			}

			dt_dpred_u -= t * dot(t, dt_dpred_u);
			dt_dpred_v -= t * dot(t, dt_dpred_v);
			dt_dcur_u  -= t * dot(t, dt_dcur_u);
			dt_dcur_v  -= t * dot(t, dt_dcur_v);

			Vector s = cross(t, wi);
			Vector ds_dpred_u = cross(dt_dpred_u, wi) + cross(t, dwi_dpred_u);
			Vector ds_dpred_v = cross(dt_dpred_v, wi) + cross(t, dwi_dpred_v);
			Vector ds_dcur_u  = cross(dt_dcur_u, wi)  + cross(t, dwi_dcur_u);
			Vector ds_dcur_v  = cross(dt_dcur_v, wi)  + cross(t, dwi_dcur_v);

			/* Some tangential projections */
			Vector2
				t_cur_dpdu (dot(v[ 0].dpdu, s), dot(v[ 0].dpdu, t)),
				t_cur_dpdv (dot(v[ 0].dpdv, s), dot(v[ 0].dpdv, t)),
				t_next_dpdu(dot(v[ 1].dpdu, s), dot(v[ 1].dpdu, t)),
				t_next_dpdv(dot(v[ 1].dpdv, s), dot(v[ 1].dpdv, t)),
				t_wo = Vector2(dot(wo, s), dot(wo, t));

			v[0].a = Matrix2x2(
				Vector2(dot(ds_dpred_u, wo), dot(dt_dpred_u, wo)),
				Vector2(dot(ds_dpred_v, wo), dot(dt_dpred_v, wo))
			);

			v[0].b = Matrix2x2(
				(t_wo * dot(wo, v[0].dpdu) - t_cur_dpdu) * ilo +
				Vector2(dot(ds_dcur_u, wo), dot(dt_dcur_u, wo)),
				(t_wo * dot(wo, v[0].dpdv) - t_cur_dpdv) * ilo +
				Vector2(dot(ds_dcur_v, wo), dot(dt_dcur_v, wo)));

			v[0].c = Matrix2x2(
				(t_next_dpdu - t_wo * dot(wo, v[1].dpdu)) * ilo,
				(t_next_dpdv - t_wo * dot(wo, v[1].dpdv)) * ilo);

			v[0].m = v[0].shFrame.toLocal(wo);
		} else {
			Log(EError, "Unknown vertex type!");
		}
	}

	return true;
}

bool PathManifold::computeTransferMatrices() {
	const int n = static_cast<int>(m_vertices.size() - 1);
	m_det = 1.f;
	if (n < 2)
		return true;
	
	/* Find the tangent space with respect to translation of the last
	   vertex. For this, we must solve a tridiagonal system. The following is
	   simplified version of the block tridiagonal LU factorization algorithm
	   for this specific problem */
	Float det;
	Matrix2x2 Li;
	if (!m_vertices[0].b.invert2x2(Li, det))
		return false;
	m_det *= det;

	Matrix2x2* u = (Matrix2x2*)alloca(sizeof(Matrix2x2)*(n-1));
	for (int i=0; i < n - 1; ++i) {
		u[i] = Li * m_vertices[i].c;
		const Matrix2x2 t = m_vertices[i+1].b - m_vertices[i+1].a * u[i];
		if (!t.invert2x2(Li, det))
			return false;
		m_det *= det;
	}

	m_vertices[n-1].Tp = -Li * m_vertices[n-1].c;

	for (int i=n-2; i>=0; --i)
		m_vertices[i].Tp = -u[i] * m_vertices[i+1].Tp;
	return true;
}

bool PathManifold::project(const Vector &d) {
	const SimpleVertex &last = m_vertices[m_vertices.size()-1];
	const Float du = dot(d, last.dpdu), dv = dot(d, last.dpdv);

	Ray ray(Point(0.0f), Vector(1.0f), 0); // make gcc happy
	Intersection its;

	m_proposal.clear();
	for (size_t i=0; i<m_vertices.size(); ++i) {
		m_proposal.push_back(m_vertices[i]);
		SimpleVertex &vertex = m_proposal[i];

		if (i == 0) {
			const Point p0 = m_vertices[0].p + m_vertices[0].map(du, dv);
			const Point p1 = m_vertices[1].p + m_vertices[1].map(du, dv);

			ray = Ray(p0, normalize(p1 - p0), m_time);
			vertex.p = ray.o;
			continue;
		} else if (vertex.type == EMovable) {
			const Float dp = dot(ray.d, vertex.shFrame.n);
			if (std::abs(dp) < Epsilon)
				return false;

			const Float t = dot(vertex.p - ray.o, vertex.shFrame.n) / dp;
			vertex.p = ray(t);
			break;
		} else if (vertex.type == EReflection || vertex.type == ERefraction) {
			if (!m_scene->rayIntersect(ray, its))
				return false;
			const BSDF *bsdf = its.shape->getBSDF();
			if (vertex.object != bsdf)
				return false;

			bsdf->getFrame(its, vertex.shFrame);
			Vector m = vertex.shFrame.toWorld(vertex.m);

			const Vector scattered = (vertex.type == EReflection) ?
				reflect(-ray.d, m) :
				refract(-ray.d, m, bsdf->getEta());

			if (scattered.isZero())
				return false;

			vertex.p = its.p;
			vertex.dpdu = its.dpdu;
			vertex.dpdv = its.dpdv;
			bsdf->getFrameDerivative(its, vertex.shFrameDu, vertex.shFrameDv);
			ray.setOrigin(its.p);
			ray.setDirection(scattered);
		} else if (vertex.type == EMedium) {
			Float length = (m_vertices[i].p - m_vertices[i-1].p).length(),
				  invLength = 1.0f / length;

			/* Check for occlusion */
			if (m_scene->rayIntersect(Ray(ray, Epsilon, length)))
				return false;

			Vector wi = -ray.d;
			vertex.p = ray(length);
			vertex.shFrame.n = wi;

			coordinateSystem(wi, vertex.dpdu, vertex.dpdv);

			vertex.shFrame.s = vertex.dpdu;
			vertex.shFrame.t = vertex.dpdv;
			vertex.shFrameDu.n = vertex.dpdu * invLength;
			vertex.shFrameDv.n = vertex.dpdv * invLength;

			ray.setOrigin(vertex.p);
			const Vector m = vertex.shFrame.toWorld(vertex.m);
			ray.setDirection(m);
		} else {
			Log(EError, "Unsupported vertex type!");
		}
	}
	return true;
}

bool PathManifold::move(const Point &target, const Normal &n) {
	SimpleVertex &last = m_vertices[m_vertices.size()-1];

	#if MTS_MANIFOLD_DEBUG == 1
		cout << "moveTo(" << last.p.toString() << " => " << target.toString() << ", n=" << n.toString() << ")" << endl;
	#endif

	if (m_vertices.size() == 2 && m_vertices[0].type == EPinnedPosition) {
		/* Nothing to do */
		return true;
	}

	bool medium = false;
	for (size_t i=0; i<m_vertices.size(); ++i) {
		if (m_vertices[i].type == EMedium)
			medium = true;
	}

	if (medium)
		statsMediumSuccess.incrementBase();

	statsAvgManifoldSize.incrementBase();
	statsAvgManifoldSize += m_vertices.size();
	statsMaxManifold.recordMaximum(m_vertices.size());

	statsSuccessfulWalks.incrementBase();

	Float invScale = 1.0f / std::max(std::max(std::abs(target.x),
			std::abs(target.y)), std::abs(target.z));
	Float stepSize = 1;

	/* Replace the derivative at the last vertex with the virtual plane connecting source and target */
	BDAssert(last.type == EMovable);
	last.shFrame = Frame(n);
	last.dpdu = last.shFrame.s;
	last.dpdv = last.shFrame.t;

	m_proposal.reserve(m_vertices.size());
	m_iterations = 0;
	statsAvgIterations.incrementBase();
	while (m_iterations < m_maxIterations) {
		Vector rel = target - m_vertices[m_vertices.size()-1].p;
		Float dist = rel.length(), newDist;
		if (dist * invScale < MTS_MANIFOLD_EPSILON) {
			/* Check for an annoying corner-case where the last
			   two vertices converge to the same point (this can
			   happen e.g. on rough planar reflectors) */
			dist = (m_vertices[m_vertices.size()-1].p
				  - m_vertices[m_vertices.size()-2].p).length();
			if (dist * invScale < Epsilon) {
				return false;
			}

			/* The manifold walk converged. */
			++statsSuccessfulWalks;
			statsAvgIterationsSuccess.incrementBase();
			statsAvgIterationsSuccess += m_iterations;
			statsMaxIterationsSuccess.recordMaximum(m_iterations);
			if (medium)
				++statsMediumSuccess;
			#if MTS_MANIFOLD_DEBUG == 1
				cout << "move(): converged after " << m_iterations << " iterations" << endl;
				cout << "Final configuration:" << toString() << endl;
			#endif
			return true;
		}
		m_iterations++;
		++statsAvgIterations;

		/* Compute the tangent vectors for the current path */
		statsStepTooFar.incrementBase();
		statsStepFailed.incrementBase();
		statsNonManifold.incrementBase();
		statsStepSuccess.incrementBase();
		if (!computeDerivatives() || !computeTransferMatrices()) {
			++statsNonManifold;
			#if MTS_MANIFOLD_DEBUG == 1
				cout << "move(): unable to compute tangents!" << endl;
			#endif
			return false;
		}

		/* Take a step using the computed tangents and project
		   back on the manifold */
		#if MTS_MANIFOLD_DEBUG == 1
			const SimpleVertex &last = m_vertices[m_vertices.size()-1];
			Float du = dot(rel, last.dpdu), dv = dot(rel, last.dpdv);
			cout << "project(du=" << du << ", dv=" << dv << ", stepSize=" << stepSize << ")" << endl;
		#endif

		if (!project(rel * stepSize)) {
			#if MTS_MANIFOLD_DEBUG == 1
				cout << "project failed!" << endl;
			#endif
			++statsStepFailed;
			goto failure;
		}

		/* Reject if the step increased the distance */
		newDist = (target - m_proposal[m_proposal.size()-1].p).length();
		#if MTS_MANIFOLD_DEBUG == 1
			cout << "Distance: " << dist << " -> " << newDist << endl;
		#endif
		if (newDist > dist) {
			++statsStepTooFar;
			#if MTS_MANIFOLD_DEBUG == 1
				cout << "-> Rejecting!" << endl;
			#endif
			goto failure;
		}
		#if MTS_MANIFOLD_DEBUG == 1
			cout << "-> Accepting!" << endl;
		#endif
		++statsStepSuccess;

		m_proposal.swap(m_vertices);

		/* Increase the step size */
		stepSize = std::min((Float) 1.0f, stepSize * 2.0f);
		continue;
	failure:
		/* Reduce the step size */
		stepSize /= 2.0f;
	}
	#if MTS_MANIFOLD_DEBUG == 1
		cout << "Exceeded the max. iteration count!" << endl;
	#endif

	return false;
}

bool PathManifold::halfvectorsToPositions(const vector_hv& h_tg, const Float stepSize) {
	// custom LU decomposition based matrix solve
	const int n = (int) m_vertices.size()-1;

	// solve for delta x
	Matrix2x2* Li = (Matrix2x2*)alloca((n+1)*sizeof(Matrix2x2));
	Matrix2x2* A = (Matrix2x2*)alloca((n+1)*sizeof(Matrix2x2));
	if (!m_vertices[0].b.invert2x2(Li[0]))
		return false;
	// invert only up to vertex n-1, the last one is fixed and doesn't have a half vector constraint.
	for (int k=1; k<n; k++) {
		A[k] = m_vertices[k].a * Li[k-1];
		// L = b - au
		const Matrix2x2 t = m_vertices[k].b - A[k]*m_vertices[k-1].c;
		if (!t.invert2x2(Li[k]))
			return false;
	}

	Vector2* H = (Vector2*)alloca((n+1)*sizeof(Vector2));
	H[0] = Vector2(0.0f, 0.0f);
	for (int k = 1; k<n; k++) {
		// create tangent space half vector delta
		const SimpleVertex& v = m_vertices[k];
		const Vector2 dh = h_tg[k-1] - Vector2(v.m.x, v.m.y) / v.m.z;
		// Involve it into LU
		H[k] = dh - A[k] * H[k-1];
	}

	Vector2 x = Li[n-1] * H[n-1]; // x[n-1]
	memcpy(m_proposal.data(), m_vertices.data(), sizeof(m_proposal[0]) * m_vertices.size());
	SimpleVertex& v = m_proposal[n-1];
	const SimpleVertex& vo = m_vertices[n-1];
	v.p = vo.p + (vo.dpdu * x.x + vo.dpdv * x.y) * stepSize;
	for (int k = n-2; k>0; k--) {
		x = Li[k] * (H[k] - m_vertices[k].c * x); // x[k] = (.. x[k+1])
		SimpleVertex& v = m_proposal[k];
		const SimpleVertex& vo = m_vertices[k];
		v.p = vo.p + (vo.dpdu * x.x + vo.dpdv * x.y) * stepSize;
	}
	return true;
}

/**
 * \brief Computes the error between two directions in the projected
 * solid angle domain.
 *
 * \param h0
 *     Direction represented using 2D Euclidean coordinates
 *     for the intersection with the plane at z=1 (similar to
 *     stereographic projection)
 * \param h1
 *     A unit vector
 */
static inline Float angError(const Vector2 &h0, const Vector &h1) {
	const Vector dh = normalize(Vector3(h0.x, h0.y, 1.0f)) - h1;
	return std::sqrt(dh.x*dh.x + dh.y*dh.y);
}

bool PathManifold::find(const vector_hv& h_tg) {
	const int k = (int) m_vertices.size() + 1;
	/* Nothing to do */
	if (k < 4)
		return true;
	const int numConstraints = k-3;
	int b, a; // outside declaration to not jump over it with the label.

	/* Statistics */
	statsAvgManifoldSize.incrementBase();
	statsAvgManifoldSize += m_vertices.size();
	statsMaxManifold.recordMaximum(m_vertices.size());
	statsSuccessfulWalks.incrementBase();
	statsAvgIterations.incrementBase();

	/* Initial error */
	Float lastError = 0.f;
	for (int j = 0; j<numConstraints; ++j) {
		/* Compute projected half-vector difference */
		const SimpleVertex& v = m_vertices[j+1];
		const Float currentErr = angError(h_tg[j], v.m);
		lastError = std::max(lastError, currentErr);
	}
	/* Nothing to do */
	// TODO: return false to avoid drifting?
	if (lastError < MTS_MANIFOLD_HV_EPSILON) {
		++statsSuccessfulWalks;
		return true;
	}

	m_proposal.resize(m_vertices.size());

	/* Decide on the projection direction */
	int step=1; // always trace from light
	a = step < 0 ? numConstraints-1 : 0; b = numConstraints-1 - a + step;

	/* Iterate predictor-corrector scheme */
	m_iterations = 0;
	Float stepSize = 1;
	while(m_iterations < m_maxIterations) {
		++m_iterations;
		++statsAvgIterations;
		Float maxError = 0;
		bool doneTg;

		statsNonManifold.incrementBase();
		statsStepTooFar.incrementBase();
		statsStepFailed.incrementBase();
		statsStepSuccess.incrementBase();

		/* Try to convert a set of half-vectors to positions */
		if (!halfvectorsToPositions(h_tg, stepSize))
			goto failure;

		/* Project vertices to surfaces */
		for (int j = a; j!=b; j += step) {
			const SimpleVertex& vp = m_proposal[j+1-step];
			SimpleVertex& v = m_proposal[j+1];

			if (v.type == EReflection || v.type == ERefraction) {
				/* Try to project it to surface along the old incident direction */
				/* Reject in a patalogic case of collapsing vertices */
				if (v.p == vp.p)
					return false;
				const Vector dir = normalize(v.p - vp.p);
				Ray r(vp.p, dir, m_time);
				Intersection its;
				if (!m_scene->rayIntersect(r, its) || its.getBSDF() != v.object) {
					++statsStepFailed;
					goto failure;
				}

				/* Update vertex intersection and local geometric derivatives */
				v.p = its.p;
				its.getBSDF()->getFrame(its, v.shFrame);
				its.getBSDF()->getFrameDerivative(its, v.shFrameDu, v.shFrameDv);
				v.dpdu = its.dpdu;
				v.dpdv = its.dpdv;
			} else {
				Log(EError, "Participating media is not supported");
				return false;
			}
		}

		/* Update half-vectors of the proposal and its manifold derivatives */
		// TODO: update derivatives only for succeeded walks, separate half vectors update and do it here
		m_proposal.swap(m_vertices);
		doneTg = computeDerivatives();
		m_proposal.swap(m_vertices);
		if (!doneTg) {
			++statsNonManifold;
			goto failure;
		}

		/* Compute maximum error */
		maxError = 0;
		for (int j = 0; j<numConstraints; ++j) {
			/* Compute projected half-vector difference */
			const SimpleVertex& v = m_proposal[j+1];
			maxError = std::max(maxError, angError(h_tg[j], v.m));
		}

		if (maxError > lastError) {
			++statsStepTooFar;
			goto failure;
		}

		/* Accept the try */
		m_proposal.swap(m_vertices);
		++statsStepSuccess;

		if (maxError < MTS_MANIFOLD_HV_EPSILON) {
			/* The manifold walk converged. */
			++statsSuccessfulWalks;
			statsAvgIterationsSuccess.incrementBase();
			statsAvgIterationsSuccess += m_iterations;
			statsMaxIterationsSuccess.recordMaximum(m_iterations);
			return true;
		}

		lastError = maxError;
		/* Increase the step size */
		stepSize = std::min((Float)1.0f, stepSize * (Float)2.0f);
		continue;
failure:
		/* Reduce the step size */
		stepSize *= (Float)0.5f;
	}

	return false;
}

bool PathManifold::update(Path &path, int start, int end) const {
	int step;
	ETransportMode mode;

	if (start < end) {
		step = 1; mode = EImportance;
	} else {
		step = -1; mode = ERadiance;
	}

	int last = (int) m_vertices.size() - 2;
	if (m_vertices[0].type == EPinnedDirection)
		last = std::max(last, 1);

	statsUpdateFailed.incrementBase();

	for (int j=0, i=start; j < last; ++j, i += step) {
		const SimpleVertex
			&v = m_vertices[j],
			&vn = m_vertices[j+1];

		PathVertex
			*pred   = path.vertexOrNull(i-step),
			*vertex = path.vertex(i),
			*succ   = path.vertex(i+step);

		int predEdgeIdx = (mode == EImportance) ? i-step : i-step-1;
		PathEdge *predEdge = path.edgeOrNull(predEdgeIdx),
				 *succEdge = path.edge(predEdgeIdx + step);

		Vector d = vn.p - v.p;
		Float length = d.length();
		d /= length;
		PathVertex::EVertexType desiredType = vn.type == EMedium ?
			PathVertex::EMediumInteraction : PathVertex::ESurfaceInteraction;

		if (v.type == EPinnedDirection) {
			/* Create a fake vertex and use it to call sampleDirect(). This is
			   kind of terrible -- a nicer API is needed to cleanly support this */
			PathVertex temp;
			temp.type = PathVertex::EMediumInteraction;
			temp.degenerate = false;
			temp.measure = EArea;
			MediumSamplingRecord &mRec = temp.getMediumSamplingRecord();
			mRec.time = m_time;
			mRec.p = vn.p;

			if (temp.sampleDirect(m_scene, NULL, vertex, succEdge, succ, mode).isZero()) {
				#if MTS_MANIFOLD_DEBUG == 1
					cout << "update(): failed in sampleDirect()!" << endl;
				#endif
				++statsUpdateFailed;
				return false;
			}

			if (m_vertices.size() >= 3) {
				PathVertex *succ2 = path.vertex(i+2*step);
				PathEdge *succ2Edge = path.edge(predEdgeIdx + 2*step);
				if (!succ->sampleNext(m_scene, NULL, vertex, succEdge, succ2Edge, succ2, mode)) {
					#if MTS_MANIFOLD_DEBUG == 1
						cout << "update(): failed in sampleNext() / pinned direction!" << endl;
					#endif
					++statsUpdateFailed;
					return false;
				}
			}
			i += step;
		} else if (!v.degenerate) {
			if (!vertex->perturbDirection(m_scene,
					pred, predEdge, succEdge, succ, d,
					length, desiredType, mode)) {
				#if MTS_MANIFOLD_DEBUG == 1
					cout << "update(): failed in perturbDirection()" << endl;
				#endif
				++statsUpdateFailed;
				return false;
			}

			Float relerr = (vn.p - succ->getPosition()).length() /
				std::max(std::max(std::abs(vn.p.x),
					std::abs(vn.p.y)), std::abs(vn.p.z));

			if (relerr > 1e-3f) {
				// be extra-cautious
				#if MTS_MANIFOLD_DEBUG == 1
					cout << "update(): failed, relative error of perturbDirection() too high:" << relerr << endl;
				#endif
				++statsUpdateFailed;
				return false;
			}
		} else {
			unsigned int compType;
			if (v.type == ERefraction)
				compType = v.eta != 1 ? BSDF::EDeltaTransmission : (BSDF::ENull | BSDF::EDeltaTransmission);
			else
				compType = BSDF::EDeltaReflection;

			if (!vertex->propagatePerturbation(m_scene,
					pred, predEdge, succEdge, succ, compType,
					length, desiredType, mode)) {
				#if MTS_MANIFOLD_DEBUG == 1
					cout << "update(): failed in propagatePerturbation()" << endl;
				#endif
				++statsUpdateFailed;
				return false;
			}

			Float relerr = (vn.p - succ->getPosition()).length() /
				std::max(std::max(std::abs(vn.p.x),
					std::abs(vn.p.y)), std::abs(vn.p.z));
			if (relerr > 1e-3f) {
				// be extra-cautious
				#if MTS_MANIFOLD_DEBUG == 1
					cout << "update(): failed, relative error of propagatePerturbation() too high:" << relerr << endl;
				#endif
				++statsUpdateFailed;
				return false;
			}
		}
	}

	return true;
}

Float PathManifold::det(const Path &path, int a, int b, int c) {
	int k = path.length();

	if (a == 0 || a == k)
		std::swap(a, c);

	int step = b > a ? 1 : -1, nGlossy = 0, nSpecular = 0;

	for (int i=a + step; i != c; i += step) {
		if (path.vertex(i)->isConnectable())
			++nGlossy;
		else
			++nSpecular;
	}

	if (nGlossy <= 1) /* No glossy materials -- we don't need this derivative */
		return 1.0f;

	bool success = init(path, a, c);
	BDAssert(success);

	int b_idx = std::abs(b-a);
	SimpleVertex &vb = m_vertices[b_idx];
	const PathVertex *pb = path.vertex(b);

	vb.shFrame = Frame(pb->isOnSurface() ?
		pb->getShadingNormal() : Normal(path.edge(a < b ? (b-1) : b)->d));

	vb.dpdu = vb.shFrame.s;
	vb.dpdv = vb.shFrame.n;

	if (!computeDerivatives() || !computeTransferMatrices()) {
		Log(EWarn, "Could not compute tangents!");
		return 0.0f;
	}

	m_vertices[b_idx].a.setZero();
	m_vertices[b_idx].b.setIdentity();
	m_vertices[b_idx].c.setZero();

	statsFullMxInversion.incrementBase();

	if (nSpecular == 0) {
		/* The chain only consists of glossy vertices -- simply compute the
		   determinant of the block tridiagonal matrix A.

		   See D.K. Salkuyeh, Comments on "A note on a three-term recurrence for a
		   tridiagonal matrix", Appl. Math. Comput. 176 (2006) 442-444. */

		Matrix2x2 Di(0.0f), D = m_vertices[1].b;

		Float det = D.det();
		Float area = cross(m_vertices[1].dpdu, m_vertices[1].dpdv).length();
		for (size_t i=2; i<m_vertices.size()-1; ++i) {
			if (!D.invert2x2(Di)) {
				Log(EWarn, "Could not invert matrix!");
				return 0.0f;
			}

			D = m_vertices[i].b - m_vertices[i].a * Di * m_vertices[i-1].c;
			det *= D.det();
			area *= cross(m_vertices[i].dpdu, m_vertices[i].dpdv).length();
		}

		return std::abs(area / det);
	} else {
		++statsFullMxInversion;
		/* The chain contains both glossy and specular materials. Compute the
		   determinant of A^-1, where rows corresponding to specular vertices
		   have been crossed out. The performance of the following is probably
		   terrible (lots of dynamic memory allocation), but it works and
		   this case happens rarely enough .. */

		Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> A(2*(nGlossy + nSpecular), 2*(nGlossy + nSpecular));
		A.setZero();

		for (int j=0, i=0; j<nGlossy+nSpecular; ++j) {
			if (j-1 >= 0) {
				A(2*i,   2*(j-1))   = m_vertices[j+1].a(0,0);
				A(2*i,   2*(j-1)+1) = m_vertices[j+1].a(0,1);
				A(2*i+1, 2*(j-1))   = m_vertices[j+1].a(1,0);
				A(2*i+1, 2*(j-1)+1) = m_vertices[j+1].a(1,1);
			}
			A(2*i,   2*j)   = m_vertices[j+1].b(0,0);
			A(2*i,   2*j+1) = m_vertices[j+1].b(0,1);
			A(2*i+1, 2*j)   = m_vertices[j+1].b(1,0);
			A(2*i+1, 2*j+1) = m_vertices[j+1].b(1,1);

			if (j+1 < nGlossy + nSpecular) {
				A(2*i,   2*(j+1))   = m_vertices[j+1].c(0,0);
				A(2*i,   2*(j+1)+1) = m_vertices[j+1].c(0,1);
				A(2*i+1, 2*(j+1))   = m_vertices[j+1].c(1,0);
				A(2*i+1, 2*(j+1)+1) = m_vertices[j+1].c(1,1);
			}
			++i;
		}

		/* Compute the inverse and "cross out" irrelevant columns and rows */
		Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> Ai = A.inverse();

		Float area = 1.f;
		
		for (int i=0; i<nGlossy+nSpecular; ++i) {
			
			area *= cross(m_vertices[i+1].dpdu, m_vertices[i+1].dpdv).length();
			
			if (!m_vertices[i+1].degenerate)
				continue;

			Ai.row(2*i).setZero();
			Ai.col(2*i).setZero();
			Ai.row(2*i+1).setZero();
			Ai.col(2*i+1).setZero();

			Ai.block<2,2>(2*i, 2*i).setIdentity();
		}

		return std::abs(Ai.determinant()) * area;
	}
}

Float PathManifold::fullG(const Path &path) {
	/* The the full-path-length generalized geometric term (transfer matrix + the first geo term) */
	const Vector d = path.edge(1)->d;
	const Float len = path.edge(1)->length;

	/* Transfer matrix determinant + the inverse square distance of the first edge */
	Float det = cross(m_vertices[1].map(1, 0), m_vertices[1].map(0, 1)).length() / (len*len);

	/* The remaining cosines of the geometric term in case of surface interaction */
	if (path.vertex(1)->isOnSurface())
		det *= dot(d, path.vertex(1)->getShadingNormal());

	if (path.vertex(2)->isOnSurface())
		det *= dot(d, path.vertex(2)->getShadingNormal());

	/* Return the determinant */
	return std::abs(det);
}

Float PathManifold::multiG(const Path &path, int a, int b) {
	if (a == 0)
		++a;
	else if (a == path.length())
		--a;
	if (b == 0)
		++b;
	else if (b == path.length())
		--b;

	int step = b > a ? 1 : -1;
	while (!path.vertex(b)->isConnectable())
		b -= step;
	while (!path.vertex(a)->isConnectable())
		a += step;

	Float result = 1;

	BDAssert(path.vertex(a)->isConnectable() && path.vertex(b)->isConnectable());
	for (int i = a + step, start = a; i != b + step; i += step) {
		if (path.vertex(i)->isConnectable()) {
			result *= G(path, start, i);
			start = i;
		}
	}

	return result;
}

Float PathManifold::G(const Path &path, int a, int b) {
	if (std::abs(a-b) == 1) {
		if (a > b)
			std::swap(a, b);
		return path.edge(a)->evalCached(path.vertex(a),
			path.vertex(b), PathEdge::EGeometricTerm)[0];
	}

	Assert(path.vertex(a)->isConnectable());
	Assert(path.vertex(b)->isConnectable());
	int step = b > a ? 1 : -1;

	bool success = init(path, a, b);
	BDAssert(success);

	SimpleVertex &last = m_vertices[m_vertices.size()-1];
	const PathVertex *vb = path.vertex(b);

	last.shFrame = Frame(vb->isOnSurface() ?
		vb->getShadingNormal() : Normal(path.edge(a < b ? (b-1) : b)->d));

	last.dpdu = last.shFrame.s;
	last.dpdv = last.shFrame.t;

	statsNonManifold.incrementBase();
	if (!computeDerivatives() || !computeTransferMatrices()) {
		++statsNonManifold;
		Log(EWarn, "SpecularPathManifold::evalG(): non-manifold configuration!");
		return 0;
	}

	Float result;
	if (m_vertices[0].type == EPinnedDirection) {
		result = cross(m_vertices[0].map(1, 0), m_vertices[0].map(0, 1)).length();
	} else if (m_vertices[0].type == EPinnedPosition) {
		Vector d = m_vertices[1].p - m_vertices[0].p;
		Float lengthSqr = d.lengthSquared(), invLength = 1/std::sqrt(lengthSqr);

		result = cross(m_vertices[1].map(1, 0), m_vertices[1].map(0, 1)).length() / lengthSqr;

		if (path.vertex(a)->isOnSurface())
			result *= absDot(d, path.vertex(a)->getShadingNormal()) * invLength;

		if (path.vertex(a+step)->isOnSurface())
			result *= absDot(d, path.vertex(a+step)->getShadingNormal()) * invLength;
	} else {
		Log(EError, "Invalid vertex type!");
		return 0;
	}

	return result;
}

/* Take the mean roughness of the surface BSDF as its Beckmann equivalent. */
Float PathManifold::getEffectiveRoughness(const PathVertex *vertex) {
	/// TODO: rewrite when BSDF::getEffectiveAlbedo(int component) exists
	if (!vertex->isConnectable())
		return 0;
	const Intersection &its = vertex->getIntersection();
	const BSDF *bsdf = its.getBSDF();

	Float roughness = 0;
	int roughnessSamples = 0;
	for (int k=0; k<bsdf->getComponentCount(); ++k) {
		const int bsdfType = bsdf->getType(k);
		if ((bsdfType & vertex->getComponentType()) == 0)
			continue;
		/* For diffuse take the roughness as 1 (instead of infinity) */
		if ((bsdfType & BSDF::EDiffuse)) {
			if (!bsdf->getDiffuseReflectance(its).isZero()) {
				/* Heuristic roughness value that will cause diffuse materials to be explored well */
				roughness += 1000.f; /* Roughness of one for diffuse */
				roughnessSamples++;
			}
		} else if (bsdfType & BSDF::EGlossy) {
			/* For rough/glossy materials just query its roughness */
			roughness += bsdf->getRoughness(its, k);
			roughnessSamples++;
		} else {
			/* Unhandled case */
			BDAssert(bsdf->getComponentCount() > 1);
		}
	}
	/* Average the accumulated roughness */
	BDAssert(roughnessSamples > 0);
	if (roughnessSamples > 1)
		roughness /= roughnessSamples;
	BDAssert(roughness > 1e-30f);
	return roughness;
}

std::string PathManifold::SimpleVertex::toString() const {
	std::ostringstream oss;

	oss << "SimpleVertex[" << endl
		<< "  type = ";

	switch (type) {
		case EPinnedPosition: oss << "pinnedPosition"; break;
		case EPinnedDirection: oss << "pinnedDirection"; break;
		case EReflection: oss << "reflection"; break;
		case ERefraction: oss << "refraction"; break;
		case EMedium: oss << "medium"; break;
		case EMovable: oss << "movable"; break;
		default: SLog(EError, "Unknown vertex type!");
	}

	oss << "," << endl
		<< "  p = " << p.toString() << "," << endl
		<< "  n = " << shFrame.n.toString() << "," << endl
		<< "  m = " << m.toString() << "," << endl
		<< "  roughness = " << roughness << "," << endl
		<< "  dpdu = " << dpdu.toString() << "," << endl
		<< "  dpdv = " << dpdv.toString() << "," << endl
		<< "  shFrame = " << shFrame.toString() << "," << endl
		<< "  shFrameDu = " << shFrameDu.toString() << "," << endl
		<< "  shFrameDv = " << shFrameDv.toString() << "," << endl
		<< "  eta = " << eta << "," << endl
		<< "  object = " << (object ? indent(object->toString()).c_str() : "null") << endl
		<< "]";

	return oss.str();
}

std::string PathManifold::toString() const {
	std::ostringstream oss;

	oss << "SpecularManifold[" << endl;
	for (size_t i=0; i<m_vertices.size(); ++i) {
		oss << "  " << i << " => " << indent(m_vertices[i].toString());
		if (i+1 < m_vertices.size())
			oss << ",";
		oss << endl;
	}
	oss << "]";

	return oss.str();
}

MTS_IMPLEMENT_CLASS(PathManifold, false, Object)

MTS_NAMESPACE_END
