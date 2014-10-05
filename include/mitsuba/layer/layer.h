/* This file is based on work conducted at Weta Digital */

#pragma once
#if !defined(__MITSUBA_LAYER_LAYER_H_)
#define __MITSUBA_LAYER_LAYER_H_

#include <mitsuba/layer/common.h>

MTS_NAMESPACE_BEGIN

/**
 * \brief Helper class, which stores one Fourier order of a
 * layer scattering function
 *
 * This class also allows convenient access to the different matrix
 * blocks (without creating costly temporaries).
 */
class MTS_EXPORT_LAYER LayerMode {
public:
	/// Create storage for a new layer mode
	LayerMode(size_t size = 0) {
		size_t hSize = size/2;
		m_reflectionTop = SMatrix(hSize, hSize);
		m_reflectionBottom = SMatrix(hSize, hSize);
		m_transmissionTopBottom = SMatrix(hSize, hSize);
		m_transmissionBottomTop = SMatrix(hSize, hSize);
	}

	/// Return the matrix describing reflection from the top face
	inline const SMatrix &reflectionTop() const {
		return m_reflectionTop;
	}

	/// Return the matrix describing reflection from the top face
	inline SMatrix &reflectionTop() {
		return m_reflectionTop;
	}

	/// Return the matrix describing reflection from the bottom face
	inline const SMatrix &reflectionBottom() const {
		return m_reflectionBottom;
	}

	/// Return the matrix describing reflection from the bottom face
	inline SMatrix &reflectionBottom() {
		return m_reflectionBottom;
	}

	/// Return the matrix describing transmission from the top->bottom face
	inline const SMatrix &transmissionTopBottom() const {
		return m_transmissionTopBottom;
	}

	/// Return the matrix describing transmission from the top->bottom face
	inline SMatrix &transmissionTopBottom() {
		return m_transmissionTopBottom;
	}

	/// Return the matrix describing transmission from the bottom->top face
	inline const SMatrix &transmissionBottomTop() const {
		return m_transmissionBottomTop;
	}

	/// Return the matrix describing transmission from the bottom->top face
	inline SMatrix &transmissionBottomTop() {
		return m_transmissionBottomTop;
	}

	/// Reverse the layer
	inline void reverse() {
		m_reflectionTop.swap(m_reflectionBottom);
		m_transmissionTopBottom.swap(m_transmissionBottomTop);
	}

	inline void insert(int i, int j, Float value) {
		if (value == 0)
			return;
		int n = m_reflectionTop.rows();

		if (i < n && j < n)
			m_transmissionBottomTop.insert(i, j) = value;
		else if (i >= n && j >= n)
			m_transmissionTopBottom.insert(i-n, j-n) = value;
		else if (i < n && j >= n)
			m_reflectionTop.insert(i, j-n) = value;
		else if (i >= n && j < n)
			m_reflectionBottom.insert(i-n, j) = value;
		else
			SLog(EError, "LayerMode::put(): Internal error!");
	}

	inline Float coeff(int i, int j) const {
		int n = m_reflectionTop.rows();

		if (i < n && j < n)
			return m_transmissionBottomTop.coeff(i, j);
		else if (i >= n && j >= n)
			return m_transmissionTopBottom.coeff(i-n, j-n);
		else if (i < n && j >= n)
			return m_reflectionTop.coeff(i, j-n);
		else if (i >= n && j < n)
			return m_reflectionBottom.coeff(i-n, j);
		else
			SLog(EError, "LayerMode::get(): Internal error!");
		return -1;
	}

	void scale(Float f) {
		m_reflectionBottom *= f;
		m_reflectionTop *= f;
		m_transmissionTopBottom *= f;
		m_transmissionBottomTop *= f;
	}

	void scaleColumns(const DVector &d) {
		int n = m_reflectionTop.rows();
		SAssert(d.size() == 2*n);
		SMatrix scale;
		sparseDiag(d.head(n), scale);
		m_transmissionBottomTop = m_transmissionBottomTop * scale;
		m_reflectionBottom = m_reflectionBottom * scale;
		sparseDiag(d.tail(n), scale);
		m_transmissionTopBottom = m_transmissionTopBottom * scale;
		m_reflectionTop = m_reflectionTop * scale;
	}

	/// Compress the sparse matrices to compressed format
	void makeCompressed() {
		m_reflectionTop.makeCompressed();
		m_reflectionBottom.makeCompressed();
		m_transmissionTopBottom.makeCompressed();
		m_transmissionBottomTop.makeCompressed();
	}

	/// Clear the layer
	inline void clear() {
		m_reflectionTop.setZero();
		m_reflectionBottom.setZero();
		m_transmissionTopBottom.setIdentity();
		m_transmissionBottomTop.setIdentity();
	}

	/// Is this layer represented by a diagonal matrix?
	inline bool isDiagonal() const {
		int n = m_reflectionTop.rows();
		return
			m_reflectionTop.nonZeros() == 0 &&
			m_reflectionBottom.nonZeros() == 0 &&
			m_transmissionTopBottom.nonZeros() == n &&
			m_transmissionBottomTop.nonZeros() == n;
	}

	/// Return the number of nonzero coefficients
	inline size_t nonZeros() const {
		return
			m_reflectionTop.nonZeros() +
			m_reflectionBottom.nonZeros() +
			m_transmissionTopBottom.nonZeros() +
			m_transmissionBottomTop.nonZeros();
	}

    void addScaled(const LayerMode &layer, Float scale) {
        m_reflectionTop         += layer.m_reflectionTop         * scale;
        m_reflectionBottom      += layer.m_reflectionBottom      * scale;
        m_transmissionTopBottom += layer.m_transmissionTopBottom * scale;
        m_transmissionBottomTop += layer.m_transmissionBottomTop * scale;
    }
protected:
	SMatrix m_reflectionTop;
	SMatrix m_reflectionBottom;
	SMatrix m_transmissionTopBottom;
	SMatrix m_transmissionBottomTop;
};

/**
 * \brief Discretized layer reflection model
 *
 * Describes the linear response to illumination that is incident along a
 * chosen set of zenith angles. The azimuthal dependence is modeled as
 * an even real Fourier transform. Each Fourier order is stored in a special
 * \c LayerMode data structure.
 */
class MTS_EXPORT_LAYER Layer {
public:
	/**
	 * \brief Create a new layer with the given discretization in zenith angles
	 * \param nodes
	 *    A vector with the zenith angle cosines of the chosen discretization
	 * \param weights
	 *    Associated weights for each angle. Usually, the 'nodes' and 'weights'
	 *    are generated using some kind of quadrature rule (e.g. Gauss-Legendre,
	 *    Gauss-Lobatto, etc.)
	 * \param nFourierOrders
	 *    Specifies the number of Fourier orders
	 */
	Layer(const DVector &nodes, const DVector &weights, size_t nFourierOrders);

	/// Copy constructor (needs to be overwritten due to reference-valued attributes)
	Layer(const Layer &l) : m_modes(l.m_modes),
		m_tau(l.m_tau), m_nodes(l.m_nodes),
		m_weights(l.m_weights) { }

	virtual ~Layer() { }

	/// Resize the number of layer modes
	void resize(int i) { m_modes.resize(i); }

	/// Initialize the layer with a HG scattering model
	void setHenyeyGreenstein(Float albedo, Float g);

	/// Initialize the layer with a vMF scattering model
	void setVMF(Float albedo, Float kappa);

	/// Initialize the layer with a microfacet model (dielectric or conductor)
	void setMicrofacet(Float eta, Float k, Float alpha, bool conserveEnergy = true, int fourierOrders = -1);

	/// Initialize the layer with a diffuse base layer
	void setDiffuse(Float albedo);

	/// Initialize the layer with a Matusik-style BRDF data file
	void setMatusik(const fs::path &path, int channel, int fourierOrders = -1);

	/// Set the matrix for a layer with only absorption
	void setAbsorbing(Float tau);

	/// Solve for the transport matrix of a layer with the given width (using Adding-Doubling)
	void solveAddingDoubling(Float tau);

	/// Solve for the transport matrix of a layer with the given width (using Discrete Ordinates)
	void solveDiscreteOrdinates(Float tau);

	/// Solve for the transport matrix of a layer with the given width (choose automatically)
	void solve(Float tau);

	/// Scale scattering matrices by the given scalar
	void scale(Float scale);

	/// Scale scattering matrices by the given scalar
	void addScaled(const Layer &layer, Float scale);

	/// Return the number of Fourier orders
	inline int getFourierOrders() const { return (int) m_modes.size(); }

	/// Return the number of nodes (i.e. the number of discretizations in \mu)
	inline int getNodeCount() const { return (int) m_nodes.size(); }

	/// Return positions of the underlying integrations nodes in the \mu parameter
	inline const DVector &getNodes() const { return m_nodes; }

	/// Return values of the underlying integrations weights in the \mu parameter
	inline const DVector &getWeights() const { return m_weights; }

	/// Look up a Fourier mode (const version)
	inline const LayerMode &operator[](int i) const { return m_modes[i]; }

	/// Look up a Fourier mode
	inline LayerMode &operator[](int i) { return m_modes[i]; }

	/// Combine two layers using the adding equations
	static void add(const Layer &l1, const Layer &l2, Layer &output, bool homogeneous);

	/// Convenience shortcut -- add a layer on top of the current one
	inline void addToTop(const Layer &l, bool homogeneous = false) {
		Layer::add(l, *this, *this, homogeneous);
	}

	/// Convenience shortcut -- add a layer at the bottom of the current one
	inline void addToBottom(const Layer &l, bool homogeneous = false) {
		Layer::add(*this, l, *this, homogeneous);
	}

	/// Expand the thickness of a layer to the desired amount
	void expand(Float desiredSize, bool homogeneous);

	/// Assignment operator (needs to be overwritten due to references)
	inline void operator=(const Layer &l) {
		SAssert(getFourierOrders() == l.getFourierOrders() &&
			getNodeCount() == l.getNodeCount());
		for (size_t i=0; i<m_modes.size(); ++i)
			m_modes[i] = l.m_modes[i];
	}

	/// Create a clear layer
	inline void clear() {
		for (size_t i=0; i<m_modes.size(); ++i)
			m_modes[i].clear();
		m_tau = 0;
	}

	/// Reverse the layer
	inline void reverse() {
		for (size_t i=0; i<m_modes.size(); ++i)
			m_modes[i].reverse();
	}

	/// Return the width of the layer
	inline Float tau() const { return m_tau; }

	/// Return the albedo of the top layer
	inline Float albedoTop() const {
		const SMatrix &Rt = m_modes[0].reflectionTop();
		ssize_t n = Rt.cols();
		DVector R = Rt * DVector::Constant(n, 1);
		return (R.array() * m_nodes.tail(n).array() * m_weights.tail(n).array()).matrix().sum() /
			m_nodes.tail(n).dot(m_weights.tail(n));
	}

	/// Return the albedo of the bottom layer
	inline Float albedoBottom() const {
		const SMatrix &Rb = m_modes[0].reflectionBottom();
		ssize_t n = Rb.cols();
		DVector R = Rb * DVector::Constant(n, 1);
		return (R.array() * m_nodes.tail(n).array() * m_weights.tail(n).array()).matrix().sum() /
			m_nodes.tail(n).dot(m_weights.tail(n));
	}

	/// Clear the transmission component (this can save space when creating opaque BRDFs)
	void clearBottomAndTransmission();

	void initializeFromQuartets(std::vector<Quartet> &quartets);
protected:
	/// Disk-backed storage (optional)
	ref<MemoryMappedFile> m_mmap;

	/// Storage for all of the Fourier modes
	std::vector<LayerMode> m_modes;

	/// Thickness of the layer
	Float m_tau;

	/// Reference to the integration nodes (for convenience)
	DVector m_nodes;

	/// Reference to the integration nodes (for convenience)
	DVector m_weights;

	/// Caches the single scattering albedo (used by solveAddingDoubling)
	Float m_albedo;
};

/**
 * Initialize the nodes and weights of a Gauss-Lobatto quadrature scheme
 * for a layer with the given relative index of refraction compared to its
 * surroundings
 */
extern MTS_EXPORT_LAYER int initializeQuadrature(int n, DVector &nodes, DVector &weights);

/// Compute the vMF kappa parameter based on g
extern MTS_EXPORT_LAYER Float vmfKappa(Float g);

/// Heuristic to guess a suitable number of parameters (Microfacet model)
extern MTS_EXPORT_LAYER std::pair<int, int> parameterHeuristicMicrofacet(Float alpha, Float eta, Float k);

/// Heuristic to guess a suitable number of parameters (HG model)
extern MTS_EXPORT_LAYER std::pair<int, int> parameterHeuristicHG(Float g);

/// Heuristic to guess a suitable number of parameters (vMF model)
extern MTS_EXPORT_LAYER std::pair<int, int> parameterHeuristicVMF(Float kappa);

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_LAYER_H_ */
