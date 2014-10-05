/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2012 by Wenzel Jakob and others.

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
#if !defined(__MITSUBA_LAYER_STORAGE__H_)
#define __MITSUBA_LAYER_STORAGE__H_

#include <mitsuba/core/mmap.h>
#include <mitsuba/core/serialization.h>

MTS_NAMESPACE_BEGIN

class Layer;

/*
 * \brief Storage class for isotropic BSDFs
 *
 * This class implements sparse storage support for isotropic BSDFs which
 * are point-sampled as a function of the incident and exitant
 * zenith angles and expanded into Fourier coefficients as a
 * function of the azimuthal difference angle.
 */
class MTS_EXPORT_LAYER BSDFStorage : public SerializableObject {
public:
	typedef uint32_t OffsetType;

	/// Create a new BSDF storage file for the given amount of coefficients etc
	BSDFStorage(const fs::path &filename, size_t nNodes, size_t nChannels,
			size_t nMaxOrder, size_t nCoeffs, size_t nBases = 1,
			size_t nParameters = 0, const size_t *paramSampleCounts = NULL,
			const float **paramSamplePositions = NULL, bool extrapolate = false,
			bool isBSDF = true, const std::string &metadata = "");

	/// Map an existing BSDF storage file into memory
	BSDFStorage(const fs::path &filename, bool readOnly = true);

	/// Unserialize a BSDF storage object from a binary data stream
	BSDFStorage(Stream *stream, InstanceManager *manager);

	/// Serialize a BSDF storage object to a binary data stream
	void serialize(Stream *stream, InstanceManager *manager) const;

	/// Return the number of Fourier coefficients
	inline size_t getMaxOrder() const { return (size_t) m_header->nMaxOrder; }

	/// Return the number of color channels
	inline size_t getChannelCount() const { return (size_t) m_header->nChannels; }

	/// Return the resolution of the discretization in \mu_i and \mu_o
	inline size_t getNodeCount() const { return (size_t) m_header->nNodes; }

	/// Return the number of basis functions stored in this file (usually just 1)
	inline size_t getBasisCount() const { return (size_t) m_header->nBases; }

	/// Return the number of model parameters
	inline size_t getParameterCount() const { return (size_t) m_header->nParameters; }

	/// Return the number of samples associated with parameter \c i
	inline size_t getParameterSampleCount(size_t i) const { return (size_t) m_paramSampleCounts[i]; }

	/// Return the sample positions associated with parameter \c i
	inline const float *getParameterSamplePositions(size_t i) const { return m_paramSamplePositionsNested[i]; }

	/// Does this file store coefficients for the harmonic extrapolation-based model?
	bool isExtrapolated() const;

	/// Does this file store a BSDF (vs a phase function)
	bool isBSDF() const;

	/// Return the size of the underlying representation in bytes
	size_t size() const;

	/// Return metadata attached to the BSDF file (if any)
	inline const std::string &getMetadata() const { return m_metadata; }

	/// Return the relative index of refraction
	inline float getEta() const { return m_header->eta; }

	/// Set the relative index of refraction
	inline void setEta(float eta) { m_header->eta = eta; }

	/// Return the Beckmann-equivalent roughness (0: bottom, 1: top surface)
	inline float getAlpha(int index) const { SAssert(index>=0&&index<=1); return m_header->alpha[index]; }

	/// Return the Beckmann-equivalent roughness (0: bottom, 1: top surface)
	inline void setAlpha(int index, float alpha) { SAssert(index>=0&&index<=1); m_header->alpha[index] = alpha; }

	/// Return the nodes of the underlying discretization in \mu_i and \mu_o
	inline const float *getNodes() const { return m_nodes; }

	/// Return a pointer to the coefficients of the CDF associated with the incident angle \c i
	inline float *getCDF(size_t i) { return m_cdfMu + i*getNodeCount()*getBasisCount(); }

	/// Return a pointer to the coefficients of the CDF associated with the incident angle \c i
	inline const float *getCDF(int i) const { return m_cdfMu + i*getNodeCount()*getBasisCount(); }

	/// Evaluate the model for the given values of \mu_i, \mu_o, and \phi_d
	Spectrum eval(Float mu_i, Float mu_o, Float phi_d, const float *basisCoeffs = NULL) const;

	/// Evaluate the model for the given values of \mu_i, \mu_o, and \phi_d
	Float pdf(Float mu_i, Float mu_o, Float phi_d, const float *basisCoeffs = NULL) const;

	/// Importance sample the model
	Spectrum sample(Float mu_i, Float &mu_o, Float &phi_d, Float &pdf, const Point2 &sample, const float *basisCoeffs = NULL) const;

	/// For debugging: return a Fourier series for the given parameters
	void interpolateSeries(Float mu_i, Float mu_o, int basis, int channel, float *coeffs) const;

	/// Forcefully release all resources
	inline void close() { m_mmap = NULL; m_header = NULL; m_coeffs = NULL; m_cdfMu = NULL; m_nodes = NULL; }

	/// Return a string representation
	std::string toString() const;

	/// Read metadata from a BSDF storage file
	static std::string readMetadata(const fs::path &path);

	/// Create a BSDF storage file from a Layer data structure (monochromatic)
	inline static ref<BSDFStorage> fromLayer(const fs::path &filename, const Layer *layer,
			bool extrapolate = false, bool isBSDF = true, const std::string &metadata = "") {
		const Layer *layers[1] = { layer };
		return BSDFStorage::fromLayerGeneral(filename, layers, 1, 1, 0, NULL, NULL, extrapolate, isBSDF, metadata);
	}

	/// Create a BSDF storage file from three Layer data structures (RGB)
	inline static ref<BSDFStorage> fromLayerRGB(const fs::path &filename, const Layer *layerR,
			const Layer *layerG, const Layer *layerB, bool extrapolate = false, bool isBSDF = true, const std::string &metadata = "") {
		const Layer *layers[3] = { layerR, layerG, layerB };
		return BSDFStorage::fromLayerGeneral(filename, layers, 3, 1, 0, NULL, NULL, extrapolate, isBSDF, metadata);
	}

	/// Create a BSDF storage file from three Layer data structures (most general interface)
	static ref<BSDFStorage> fromLayerGeneral(const fs::path &filename,
			const Layer **layers, size_t nChannels, size_t nBases = 1, size_t nParameters = 0,
			const size_t *paramSampleCounts = NULL, const float **paramSamplePositions = NULL,
			bool extrapolate = false, bool isBSDF = true, const std::string &metadata = "");

	/// Create a new BSDF storage file by merging two existing files (to create multiple basis functions, e.g. for texturing)
	static ref<BSDFStorage> merge(const fs::path &outputFile, const BSDFStorage *s0, const BSDFStorage *s1);

	std::string stats() const;

	MTS_DECLARE_CLASS()
protected:
	struct Header {
		uint8_t identifier[7];     // Set to 'SCATFUN'
		uint8_t version;           // Currently version is 1
		uint32_t flags;            // 0x01: file contains a BSDF, 0x02: uses harmonic extrapolation
		uint32_t nNodes;           // Number of samples in the elevational discretization

		uint32_t nCoeffs;          // Total number of Fourier series coefficients stored in the file
		uint32_t nMaxOrder;        // Coeff. count for the longest series occuring in the file
		uint32_t nChannels;        // Number of color channels (usually 1 or 3)
		uint32_t nBases;           // Number of BSDF basis functions (relevant for texturing)

		uint32_t nMetadataBytes;   // Size of descriptive metadata that follows the BSDF data
		uint32_t nParameters;      // Number of textured material parameters
		uint32_t nParameterValues; // Total number of BSDF samples for all textured parameters
		float eta;                 // Relative IOR through the material (eta(bottom) / eta(top))

		float alpha[2];            // Beckmann-equiv. roughness on the top (0) and bottom (1) side
		float unused[2];           // Unused fields to pad the header to 64 bytes

		float data[0];             // BSDF data starts here
	};

	/// Return a posize_ter to the underlying sparse offset table
	inline OffsetType *getOffsetTable(size_t o = 0, size_t i = 0)
		{ return m_offsetTable + 2*(o + i * getNodeCount()); }

	/// Return a posize_ter to the underlying sparse offset table (const version)
	inline const OffsetType *getOffsetTable(size_t o = 0, size_t i = 0) const
		{ return m_offsetTable + 2*(o + i * getNodeCount()); }

	/// Return the sparse data offset of the given incident and exitant angle pair
	inline const float *getCoefficients(size_t o, size_t i) const { return m_coeffs + getOffsetTable(o, i)[0]; }

	/// Return the sparse data offset of the given incident and exitant angle pair
	inline float *getCoefficients(size_t o, size_t i) { return m_coeffs + getOffsetTable(o, i)[0]; }

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline const float *getCoefficients(size_t o, size_t i, size_t basis, size_t channel) const {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return m_coeffs + offset + basis * size + getBasisCount()*size*channel;
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline float *getCoefficients(size_t o, size_t i, size_t basis, size_t channel) {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return m_coeffs + offset + basis * size + getBasisCount()*size*channel;
	}

	/// Return the sparse data size of the given incident and exitant angle pair
	inline OffsetType getCoefficientsCount(size_t o, size_t i) const {
		return getOffsetTable(o, i)[1];
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<const float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i) const {
		const OffsetType *offset = getOffsetTable(o, i);
		return std::make_pair(m_coeffs + offset[0], offset[1]);
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i) {
		const OffsetType *offset = getOffsetTable(o, i);
		return std::make_pair(m_coeffs + offset[0], offset[1]);
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<const float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i, size_t basis) const {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return std::make_pair(m_coeffs + offset + basis * size, size);
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i, size_t basis) {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return std::make_pair(m_coeffs + offset + basis * size, size);
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<const float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i, size_t basis, size_t channel) const {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return std::make_pair(m_coeffs + offset + basis * size + getBasisCount()*size*channel, size);
	}

	/// Return the sparse data offset and size of the given incident and exitant angle pair
	inline std::pair<float *, OffsetType> getCoefficientsAndCount(size_t o, size_t i, size_t basis, size_t channel) {
		const OffsetType *offsetPtr = getOffsetTable(o, i);
		OffsetType offset = offsetPtr[0], size = offsetPtr[1];
		return std::make_pair(m_coeffs + offset + basis * size + getBasisCount()*size*channel, size);
	}
	/// Look up spline size_terpolation weights for the given zenith angle cosine
	void computeInterpolationWeights(float mu, size_t &offset, float (&weights)[4]) const;

	/// Evaluate the discrete CDF that is used to sample a zenith angle spline segment
	inline float evalLatitudinalCDF(size_t knotOffset, float *knotWeights, size_t index, const float *basisCoeffs) const {
		const size_t n = getNodeCount(), m = getBasisCount();
		const float *cdf = m_cdfMu + (knotOffset*n + index) * m;
		float result = 0;

		for (size_t i=0; i<4; ++i) {
			float weight = knotWeights[i];
			if (weight != 0)
				for (size_t basis=0; basis<m; ++basis)
					result += cdf[i*n*m+basis] * weight * basisCoeffs[basis];
		}

		return result;
	}

	/// Evaluate the zeroeth-order Fourier coefficient given size_terpolation weights
	float evalLatitudinalAverage(size_t knotOffset, float *knotWeights, size_t index, const float *basisCoeffs) const {
		float result = 0.0f;
		for (size_t i=0; i<4; ++i) {
			float weight = knotWeights[i];
			if (weight == 0)
				continue;
			std::pair<const float *, OffsetType> coeffsAndCount = getCoefficientsAndCount(index, knotOffset + i);
			const OffsetType count = coeffsAndCount.second;
			if (!count)
				continue;

			const float *coeffs = coeffsAndCount.first;
			for (size_t basis=0; basis<getBasisCount(); ++basis)
				result += weight * basisCoeffs[basis] * coeffs[basis*count];
		}

		return result;
	}

	/// Virtual destructor
	virtual ~BSDFStorage();
protected:
	ref<MemoryMappedFile> m_mmap;
	Header *m_header;
	float *m_nodes;
	float *m_cdfMu;
	OffsetType *m_offsetTable;
	float *m_coeffs;
	double *m_reciprocals;
	uint32_t *m_paramSampleCounts;
	fs::path m_filename;
	float *m_paramSamplePositions;
	float **m_paramSamplePositionsNested;
	std::string m_metadata;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_STORAGE__H_ */
