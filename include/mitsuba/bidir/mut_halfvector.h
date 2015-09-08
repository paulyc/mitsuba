/*
	This file is part of Mitsuba, a physically based rendering system.

	Copyright (c) 2007-2014 Wenzel Jakob and others.

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

#pragma once
#if !defined(__MITSUBA_BIDIR_MUT_HALFVECTOR_H_)
#define __MITSUBA_BIDIR_MUT_HALFVECTOR_H_

#include <mitsuba/bidir/mutator.h>
#include <mitsuba/bidir/manifold.h>

MTS_NAMESPACE_BEGIN

struct CachedTransitionPdf
{
	Float breakupPdf;
	Float transferMx;
	Float totalRoughness;
	std::vector<Float> vtxRoughtness;
};

/**
 * \brief Glossy half-vector space perturbation strategy
 *
 * \ingroup libbidir
 */
class MTS_EXPORT_BIDIR HalfvectorPerturbation : public MutatorBase {
public:
	/**
	 * \brief Construct a new half-vector space perturbation strategy
	 *
	 * \param scene
	 *     A pointer to the underlying scene
	 *
	 * \param sampler
	 *     A sample generator
	 *
	 * \param pool
	 *     A memory pool used to allocate new path vertices and edges
	 */
	HalfvectorPerturbation(const Scene *scene, Sampler *sampler, MemoryPool &pool, const Float probFactor);

	// =============================================================
	//! @{ \name Implementation of the Mutator interface

	EMutationType getType() const;
	Float suitability(const Path &path) const;
	bool sampleMutation(Path &source, Path &proposal,
			MutationRecord &muRec, const MutationRecord& sourceMuRec);
	Float Q(const Path &source, const Path &proposal,
			const MutationRecord &muRec) const;
	void accept(const MutationRecord &muRec);

	//! @}
	// =============================================================

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~HalfvectorPerturbation();

	// Strategy deciding-routines
	bool computeBreakupProbabilities(const Path& path) const;
	bool sampleSubpath(const Path& source, int& a, int& b, int& c);
	
	// Perturbations
	bool perturbEndpoints(Path& path);
	bool perturbLensSubpath(const Path& source, Path& proposal, const int a, const int b);
	bool perturbHalfvectors(vector_hv& hs, const Path& source, const int b, const int c);
public:
	struct HalfVectorDifferential {
		Matrix2x2 R; // rotation matrix (det = 1) in plane/plane space
		Vector2   v; // eigenvalues
	};

	CachedTransitionPdf m_fwPdf, m_bwPdf;
  
	const Float	m_probFactor;	// Used for perturbing end points
	mutable ref<PathManifold> m_manifold;
	mutable DiscreteDistribution m_breakupPmf;
	mutable RayDifferential m_rayDifferential;
	mutable std::vector<HalfVectorDifferential> m_halfvectorDifferentials;
	vector_hv m_h_orig;
	vector_hv m_h_perturbed;
	Path *m_current;
};

MTS_NAMESPACE_END

#endif /*__MITSUBA_BIDIR_MUT_HALFVECTOR_H_ */
