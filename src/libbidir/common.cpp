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

#include <mitsuba/bidir/common.h>
#include <mitsuba/bidir/mutator.h>

#define MTS_BD_MEDIUM_PERTURBATION_MONOCHROMATIC 1

MTS_NAMESPACE_BEGIN

std::string EndpointRecord::toString() const {
	std::ostringstream oss;
	oss << "EndpointRecord[time=" << time << "]";
	return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Mutator::EMutationType &type) {
	switch (type) {
		case Mutator::EBidirectionalMutation: os << "bidir"; break;
		case Mutator::ELensPerturbation: os << "lens"; break;
		case Mutator::ELensSubpathMutation: os << "lensSubpath"; break;
		case Mutator::ECausticPerturbation: os << "caustic"; break;
		case Mutator::EIndependentMutation: os << "indep"; break;
		case Mutator::EMultiChainPerturbation: os << "multiChain"; break;
		case Mutator::EManifoldPerturbation: os << "manifold"; break;
		case Mutator::EHalfvectorPerturbation: os << "halfVector"; break;
		default: os << "invalid"; break;
	};
	return os;
}

std::string MutationRecord::toString() const {
	std::ostringstream oss;
	oss << "MutationRecord["
		<< "type=" << type
		<< ", l=" << l
		<< ", m=" << m
		<< ", kd=" << m-l
		<< ", ka=" << ka
		<< ", weight=" << weight.toString()
		<< "]";
	return oss.str();
}

MutatorBase::MutatorBase(const Scene *scene, Sampler *sampler, MemoryPool &pool)
		: m_scene(scene), m_sampler(sampler), m_pool(pool) {
	m_mediumDensityMultiplier = 100.0f;
}


/// Perturb a direction at vertex a and trace a perturbed subpath until a new vertex b
bool MutatorBase::perturbSubpath(const Path& source, Path& proposal, int a, int b, int perturbFlags) {
	const int step = a < b ? 1 : -1;
	const ETransportMode mode = (step == 1) ? EImportance : ERadiance;
	
	/* Generate subsequent vertices between a .. b deterministically */
	for (int i = a + step; i != b; i += step) {
		const PathVertex
		*pred_old     = source.vertex(i-step),
		*vertex_old   = source.vertex(i),
		*succ_old     = source.vertex(i+step);
		const PathEdge
		*succEdge_old = source.edge(mode == EImportance ? i : i-1);
		PathVertex
		*pred         = proposal.vertex(i-step),
		*vertex       = proposal.vertex(i),
		*succ         = proposal.vertex(i+step);
		PathEdge
		*predEdge     = proposal.edge(mode == EImportance ? i-step : i-1-step),
		*succEdge     = proposal.edge(mode == EImportance ? i : i-1);
		
		if (vertex_old->isSurfaceInteraction()) {
			const Intersection
			&its_old = vertex_old->getIntersection(),
			&its_new = vertex->getIntersection();
			
			Vector
			wi_old = its_old.toLocal(normalize(pred_old->getPosition() - its_old.p)),
			wo_old = its_old.toLocal(normalize(succ_old->getPosition() - its_old.p));
			
			bool reflection = Frame::cosTheta(wi_old) * Frame::cosTheta(wo_old) > 0;
			Float eta = vertex_old->getIntersection().getBSDF()->getEta();
			Vector wi_world = normalize(pred->getPosition() - vertex->getPosition()),
			wo_world(0.0f);
			
			/// todo: this is perhaps a bit drastic
			if ((perturbFlags & EPreserveManifold) && its_old.getBSDF() != its_new.getBSDF())
				return false;
			
			if (vertex_old->isConnectable()) {
				Vector m(0.0f);
				if (reflection)
					m = normalize(wi_old + wo_old);
				else if (eta != 1)
					m = normalize(wi_old.z < 0 ? (wi_old*eta + wo_old)
								  : (wi_old + wo_old*eta));
				
				/* Compensate for the occasional rotation of tangent frame if hit different shape or no uv mapping present */
				if(its_old.dpdu != its_new.dpdu) {
					/* Check if it's a mesh and we have no uv mapping */
					const TriMesh* mesh = NULL;
					if(its_old.shape && its_old.shape->getClass()->derivesFrom(MTS_CLASS(TriMesh)))
					    mesh = static_cast<const TriMesh*>(its_old.shape);
					if(its_old.shape != its_new.shape || (mesh && !mesh->hasUVTangents())) {
						/* Take the line from one point to another as an invariant direction, make sure its orientation is stable */
						const Vector dt = Vector(its_old.p).length() < Vector(its_new.p).length() ? (its_old.p - its_new.p) : (its_new.p - its_old.p);
						
						/* Project the line onto both tangent frames */
						const Vector2 dt_old = normalize(Vector2(dot(dt, its_old.shFrame.s), dot(dt, its_old.shFrame.t)));
						const Vector2 dt_new = normalize(Vector2(dot(dt, its_new.shFrame.s), dot(dt, its_new.shFrame.t)));
						
						/* Compute the rotation angle */
						const Float cosPhi = dot(dt_old, dt_new);
						const Float sinPhi = dt_old.x*dt_new.y - dt_old.y*dt_new.x;
						
						/* Rotate the half vector */
						const Float x_new = m.x * cosPhi - m.y*sinPhi,
						y_new = m.x * sinPhi + m.y*cosPhi;
						m.x = x_new;
						m.y = y_new;
					}
				}
				
				m = its_new.toWorld(m.z > 0 ? m : -m);
				
				if (reflection) {
					wo_world = reflect(wi_world, m);
				} else {
					if (eta != 1) {
						wo_world = refract(wi_world, m, eta);
						if (wo_world.isZero())
							return false;
					} else {
						wo_world = -wi_world;
					}
				}
				
				Float dist = succEdge_old->length;
				if (i+step == b || (perturbFlags & EPerturbAllDistances))
					dist += perturbMediumDistance(m_sampler, succ_old);
				
				if (!vertex->perturbDirection(m_scene,
											  pred, predEdge, succEdge, succ, wo_world,
											  dist, succ_old->getType(), mode)) {
					return false;
				}
			} else {
				int component = reflection ? BSDF::EDeltaReflection :
				(BSDF::EDeltaTransmission | BSDF::ENull);
				
				Float dist = succEdge_old->length;
				if (i+step == b || (perturbFlags & EPerturbAllDistances))
					dist += perturbMediumDistance(m_sampler, succ_old);
				
				if (!vertex->propagatePerturbation(m_scene,
												   pred, predEdge, succEdge, succ, component, dist,
												   succ_old->getType(), mode)) {
					return false;
				}
			}
		} else if (vertex_old->isMediumInteraction()) {
			Point p_old = vertex_old->getPosition(),
			p_new = vertex->getPosition();
			
			Normal
			n_old(normalize(p_old - pred_old->getPosition())),
			n_new(normalize(p_new - pred->getPosition()));
			
			Vector
			dpdu_old = Vector(p_old) - dot(Vector(p_old), n_old) * n_old,
			dpdu_new = Vector(p_new) - dot(Vector(p_new), n_new) * n_new;
			
			Vector dpdv_old, dpdv_new, wo_old, wo_new;
			Float cosTheta, cosPhi, sinPhi;
			
			if (dpdu_old.isZero() || dpdu_new.isZero())
				return false;
			
			dpdu_old = normalize(dpdu_old);
			dpdu_new = normalize(dpdu_new);
			dpdv_old = cross(n_old, dpdu_old);
			dpdv_new = cross(n_new, dpdu_new);
			
			wo_old = normalize(succ_old->getPosition() - p_old);
			
			cosTheta = dot(wo_old, n_old);
			
			Float dTheta = warp::squareToStdNormal(m_sampler->next2D()).x
			* 0.5f * M_PI / m_mediumDensityMultiplier;
			math::sincos(dTheta, &sinPhi, &cosPhi);
			
			Float x = dot(wo_old, dpdu_old), y = dot(wo_old, dpdv_old);
			Float x_new = x * cosPhi - y*sinPhi,
			y_new = x * sinPhi + y*cosPhi;
			
			wo_new = dpdu_new * x_new + dpdv_new * y_new + n_new * cosTheta;
			
			Float dist = succEdge_old->length;
			if (i+step == b || (perturbFlags & EPerturbAllDistances))
				dist += perturbMediumDistance(m_sampler, succ_old);
			
			if (!vertex->perturbDirection(m_scene,
										  pred, predEdge, succEdge, succ, wo_new,
										  dist, succ_old->getType(), mode))
				return false;
		} else {
			Log(EError, "Unsupported vertex type!");
		}
	}
	
	
	return true;
}

// Throughput of a subpath from a to b
Spectrum MutatorBase::subpathThroughput(const Path& source, const Path& proposal, int a, int b, int perturbFlags) const {
	Spectrum thp(1.f);
	const int step = a < b ? 1 : -1;
	const ETransportMode mode = (step == 1) ? EImportance : ERadiance;

	for (int i = a; i != b; i += step) {
		int l = std::min(i, i+step),
		r = std::max(i, i+step);
		const PathVertex *v0 = proposal.vertex(l),
						 *v1 = proposal.vertex(r);
		const PathEdge *edge = proposal.edge(l);
		
		if (!proposal.vertex(i)->isConnectable() || !(perturbFlags&EPreserveManifold))
			thp *= edge->evalCached(v0, v1, PathEdge::ETransmittance | ((mode == EImportance)
																		? PathEdge::EValueCosineImp : PathEdge::EValueCosineRad));
		else
			thp *= edge->evalCached(v0, v1, PathEdge::ETransmittance | ((mode == EImportance)
																		? PathEdge::EValueImp : PathEdge::EValueRad));
		
		if ((perturbFlags & EPerturbAllDistances) && proposal.vertex(i+step)->isMediumInteraction())
			thp /= pdfMediumPerturbation(source.vertex(i+step),
											source.edge(l), edge);
	}
	
	return thp;
}


Float MutatorBase::perturbMediumDistance(Sampler *sampler, const PathVertex *vertex) {
	if (vertex->isMediumInteraction()) {
#if MTS_BD_MEDIUM_PERTURBATION_MONOCHROMATIC == 1
		/* Monochromatic version */
		const MediumSamplingRecord &mRec = vertex->getMediumSamplingRecord();
		Float sigma = (mRec.sigmaA + mRec.sigmaS).average() * m_mediumDensityMultiplier;
#else
		const MediumSamplingRecord &mRec = vertex->getMediumSamplingRecord();
		Spectrum sigmaT = (mRec.sigmaA + mRec.sigmaS) * m_mediumDensityMultiplier;
		Float sigma = sigmaT[
			std::min((int) (sampler->next1D() * SPECTRUM_SAMPLES), SPECTRUM_SAMPLES-1)];
#endif
		return (sampler->next1D() > .5 ? -1.0f : 1.0f) *
			math::fastlog(1-sampler->next1D()) / sigma;
	} else {
		return 0.0f;
	}
}

Float MutatorBase::pdfMediumPerturbation(const PathVertex *oldVertex,
		const PathEdge *oldEdge, const PathEdge *newEdge) const {
	BDAssert(oldEdge->medium && newEdge->medium);
	const MediumSamplingRecord &mRec = oldVertex->getMediumSamplingRecord();
#if MTS_BD_MEDIUM_PERTURBATION_MONOCHROMATIC == 1
	Float sigmaT = (mRec.sigmaA + mRec.sigmaS).average() * m_mediumDensityMultiplier;
	Float diff = std::abs(oldEdge->length - newEdge->length);
	return 0.5f * sigmaT*math::fastexp(-sigmaT*diff);
#else
	Spectrum sigmaT = (mRec.sigmaA + mRec.sigmaS) * m_mediumDensityMultiplier;
	Float diff = std::abs(oldEdge->length - newEdge->length);
	Float sum = 0.0f;
	for (int i=0; i<SPECTRUM_SAMPLES; ++i)
		sum += sigmaT[i]*math::fastexp(-sigmaT[i]*diff);
	return sum * (0.5f / SPECTRUM_SAMPLES);
#endif
}

MTS_IMPLEMENT_CLASS(Mutator, true, Object)
MTS_IMPLEMENT_CLASS(MutatorBase, true, Mutator)
MTS_NAMESPACE_END
