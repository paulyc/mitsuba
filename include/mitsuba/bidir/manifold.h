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

#pragma once
#if !defined(__MITSUBA_BIDIR_MANIFOLD_H_)
#define __MITSUBA_BIDIR_MANIFOLD_H_

#include <mitsuba/bidir/vertex.h>

#define MTS_MANIFOLD_DEBUG          0
#define MTS_MANIFOLD_MAX_ITERATIONS 20

MTS_NAMESPACE_BEGIN

typedef std::vector<Vector2> vector_hv;

/**
 * \brief Utility class for perturbing paths located on a
 * high-dimensional path manifold.
 *
 * Used for both Manifold Exploration and Halfvector Space Light Transport (HSLT).
 *
 * \author Wenzel Jakob
 * \author Anton Kaplanyan
 * \author Johannes Hanika
 */
class MTS_EXPORT_BIDIR PathManifold : public Object {
public:
	/// Construct an uninitialized path manifold data structure
	PathManifold(const Scene *scene, int maxIterations = -1);

	/**
	 * \brief Initialize the start and end vertices of a path manifold with the specified
	 * path vertices. Internally used by \ref init()
	 */
	void initEndpoints(const PathVertex *vs, const PathVertex *vpred_s,
					   const PathVertex *vpred_e, const PathVertex *ve);

	/**
	 * \brief Initialize the path manifold with the specified
	 * path segment
	 */
	bool init(const Path &path, int start, int end);

	/**
	 * \brief Update the provided path segment based on the stored
	 * path manifold configuration
	 */
	bool update(Path &path, int start, int end) const;

	/// Attempt to move the movable endpoint vertex to position \c target
	bool move(const Point &target, const Normal &normal);

	/**
	 * \brief Attempt to find the path corresponding to the set of proposed
	 * half-vectors
	 *
	 * This is used to implement HSLT (Halfvector Space Light Transport)
	 */
	bool find(const vector_hv& h_tg);

	/**
	 * \brief Compute the generalized geometric term between 'a' and 'b'
	 */
	Float G(const Path &path, int a, int b);

	/**
	 * \brief Compute a product of standard and generalized geometric
	 * terms between 'a' and 'b' depending on whether vertices are
	 * specular or non-specular.
	 */
	Float multiG(const Path &path, int a, int b);

	Float fullG(const Path &path);

	Float det(const Path &path, int a, int b, int c);

	/// Return the number of iterations used by \ref move()
	inline int getIterationCount() const { return m_iterations; }

	/// Return the position of a vertex
	inline const Point &getPosition(int i) { return m_vertices[i].p; }

	/// Return a string representation
	std::string toString() const;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~PathManifold() { }

public:
	enum EType {
		EPinnedPosition = 0,
		EPinnedDirection,
		EReflection,
		ERefraction,
		EMedium,
		EMovable
	};

	/// Describes a single interaction on the path
	struct SimpleVertex {
		bool degenerate : 1;
		EType type : 31;

		/* Position and partials */
		Point p;
		Vector dpdu, dpdv;

		/* Shading frame and U/V partial derivatives */
		Frame shFrame;
		Frame shFrameDu;
		Frame shFrameDv;

		/* Half direction vector expressed in local coordinates (wrt. shading frame) */
		Normal m;

		/* Roughness of the vertex (for acceptance) */
		Float roughness;

		/* Further information about the vertex */
		Float eta;
		const Object *object;

		/* Scratch space for matrix assembly */
		Matrix2x2 a, b, c;

		/* Manifold tangent space projected onto this vertex */
		Matrix2x2 Tp;

		/// Initialize certain fields to zero by default
		inline SimpleVertex(EType type, const Point &p) :
			degenerate(false), type(type), p(p), dpdu(0.0f),
			dpdv(0.0f), m(0.0f), roughness(0.f), eta(1.0f), object(NULL),
			shFrame(Frame(Vector(0.f), Vector(0.f), Normal(0.f))),
			shFrameDu(Frame(Vector(0.f), Vector(0.f), Normal(0.f))), 
			shFrameDv(Frame(Vector(0.f), Vector(0.f), Normal(0.f)))
		{
		}

		inline SimpleVertex() { }

		/// Map a tangent space displacement into world space
		inline Vector map(const Float u, const Float v) const {
			const Vector2 T = Tp * Vector2(u, v);
			return dpdu * T.x + dpdv * T.y;
		}

		std::string toString() const;
	};

	/**
	 * \brief Compute the tangent derivatives matrices a, b, and c
	 * at each vertex of the path
	 */
	bool computeDerivatives();

	/**
	 * \brief Compute the transfer matrices with the specified
	 * components when projected onto the movable endpoint vertex
	 */
	bool computeTransferMatrices();
	
	Float fullDet() const { return std::abs(m_det); }

	FINLINE const SimpleVertex& vertex(const size_t i) const { return m_vertices[i]; }
	FINLINE const size_t size() const { return m_vertices.size(); }

	/// Compute the effective surface roughness of a path vertex (for HSLT)
	static Float getEffectiveRoughness(const PathVertex *vertex);

private:

	typedef std::vector<SimpleVertex> vector_path;

	/**
	 * Tries to find world-space path on the given manifold.
	 * One iteration of find() method.
	 */
	bool halfvectorsToPositions(const vector_hv& h_tg, const Float stepSize);

	/// Take the specified step and project back onto the manifold
	bool project(const Vector &d);

protected:
	const Scene *m_scene;
	Float m_time;
	int m_iterations, m_maxIterations;
	
	Float m_det;

	vector_path m_vertices, m_proposal;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_BIDIR_MANIFOLD_H_ */
