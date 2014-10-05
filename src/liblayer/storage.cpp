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

#include <mitsuba/layer/storage.h>
#include <mitsuba/layer/fourier.h>
#include <mitsuba/layer/layer.h>
#include <mitsuba/layer/hrex.h>
#include <mitsuba/core/spline.h>
#include <mitsuba/core/atomic.h>

#if defined(__OSX__)
	#include <dispatch/dispatch.h>
	#define MTS_GCD 1
#endif

#if defined(__MSVC__)
# include <intrin.h>
#else
# include <immintrin.h>
#endif

MTS_NAMESPACE_BEGIN

#define BSDF_STORAGE_HEADER_ID          "SCATFUN"
#define BSDF_STORAGE_VERSION            1
#define BSDF_STORAGE_FLAGS_EXTRAPOLATED 2
#define BSDF_STORAGE_FLAGS_BSDF         1
#define BSDF_STORAGE_HEADER_SIZE        64

static const float __basisCoeffsDefault[3] = { 1.0, 1.0, 1.0 };

BSDFStorage::BSDFStorage(const fs::path &filename, size_t nNodes, size_t nChannels,
			size_t nMaxOrder, size_t nCoeffs, size_t nBases, size_t nParameters,
			const size_t *paramSampleCounts, const float **paramSamplePositions, bool extrapolate,
			bool isBSDF, const std::string &metadata) : m_header(NULL), m_reciprocals(NULL),
			m_filename(filename), m_paramSamplePositionsNested(NULL) {

	if (nChannels != 1 && nChannels != 3)
		Log(EError, "Only 1 and 3-channel files are supported!");

	if (extrapolate && nMaxOrder != 3)
		Log(EError, "Only three Fourier orders should be specified "
			"for the extrapolated storage format!");

	size_t nBasesPred = 1, nParameterValues = 0;

	for (size_t i=0; i<nParameters; ++i) {
		nParameterValues += paramSampleCounts[i];
		nBasesPred *= paramSampleCounts[i];
	}

	if (nBasesPred != nBases)
		Log(EError, "BSDFStorage::BSDFStorage(): provided an invalid number of basis functions");

	size_t size = BSDF_STORAGE_HEADER_SIZE + // Header
		sizeof(float)*nNodes +               // Node locations
		sizeof(uint32_t)*nParameters +       // Parameter sample counts
		sizeof(float)*nParameterValues +     // Parameter sample positions
		sizeof(float)*nNodes*nNodes*nBases + // CDF in \mu
		sizeof(OffsetType)*nNodes*nNodes*2 + // Offset + size table
		sizeof(float)*nCoeffs +              // Fourier coefficients
		metadata.size();                     // Metadata

	size_t uncompressedSize = size - sizeof(float)*nCoeffs
		+ nNodes*nNodes*nChannels*nBases*nMaxOrder*sizeof(float);

	Log(EInfo, "Creating sparse BSDF storage file \"%s\":", filename.filename().string().c_str());
	Log(EInfo, "  Discretizations in mu  : " SIZE_T_FMT, nNodes);
	if (!extrapolate)
		Log(EInfo, "  Max. Fourier orders    : " SIZE_T_FMT, nMaxOrder);
	else
		Log(EInfo, "  Harmonic extrapolation : yes");
	Log(EInfo, "  Color channels         : " SIZE_T_FMT, nChannels);
	Log(EInfo, "  Textured parameters    : " SIZE_T_FMT, nParameters);
	Log(EInfo, "  Basis functions        : " SIZE_T_FMT, nBases);
	Log(EInfo, "  Uncompressed size      : %s", memString(uncompressedSize).c_str());
	Log(EInfo, "  Actual size            : %s (reduced to %.2f%%)", memString(size).c_str(),
			100 * size / (Float) uncompressedSize);

	m_mmap = new MemoryMappedFile(filename, size);
	m_header = (Header *) m_mmap->getData();

	const char *id = BSDF_STORAGE_HEADER_ID;

	const int len = strlen(BSDF_STORAGE_HEADER_ID);
	for (int i=0; i<len; ++i)
		m_header->identifier[i] = id[i];

	m_header->version = BSDF_STORAGE_VERSION;
	m_header->flags = 0;
	if (isBSDF)
		m_header->flags |= BSDF_STORAGE_FLAGS_BSDF;
	if (extrapolate)
		m_header->flags |= BSDF_STORAGE_FLAGS_EXTRAPOLATED;
	m_header->nNodes = (uint32_t) nNodes;
	m_header->nParameters = (uint16_t) nParameters;
	m_header->nMaxOrder = (uint32_t) nMaxOrder;
	m_header->nChannels = (uint32_t) nChannels;
	m_header->nBases = (uint32_t) nBases;
	m_header->nParameterValues = (uint16_t) nParameterValues;
	m_header->nCoeffs = (uint32_t) nCoeffs;
	m_header->nMetadataBytes = (uint32_t) metadata.size();
	m_header->eta = 1.0f; // default

	m_nodes = m_header->data;
	m_paramSampleCounts = (uint32_t *) (m_nodes + nNodes);
	m_paramSamplePositions = (float *) (m_paramSampleCounts + nParameters);
	m_cdfMu = m_paramSamplePositions + nParameterValues;
	m_offsetTable = (OffsetType *) (m_cdfMu + nNodes*nNodes*nBases);
	m_coeffs = (float *) (m_offsetTable + nNodes*nNodes*2);

	size_t idx = 0;
	m_paramSamplePositionsNested = new float*[nParameters];
	for (size_t i=0; i<nParameters; ++i) {
		m_paramSampleCounts[i] = (uint32_t) paramSampleCounts[i];
		m_paramSamplePositionsNested[i] = m_paramSamplePositions + idx;
		for (size_t j = 0; j<m_paramSampleCounts[i]; ++j)
			m_paramSamplePositions[idx++] = (float) paramSamplePositions[i][j];
	}

	memcpy(m_coeffs + nCoeffs, metadata.c_str(), metadata.size());
	m_metadata = metadata;

	int extra = LANE_WIDTH + LANE_WIDTH-1;
	m_reciprocals = (double *) allocAligned((nMaxOrder+extra) * sizeof(double));
	memset(m_reciprocals, 0, sizeof(double) * (nMaxOrder+extra));
	m_reciprocals += LANE_WIDTH-1;
	for (uint32_t i=0; i<nMaxOrder+LANE_WIDTH; ++i)
		m_reciprocals[i] = 1.0 / (double) i;
}

BSDFStorage::BSDFStorage(const fs::path &filename, bool readOnly)
		: m_header(NULL), m_reciprocals(NULL), m_filename(filename),
		  m_paramSamplePositionsNested(NULL) {
	BOOST_STATIC_ASSERT(sizeof(Header) == BSDF_STORAGE_HEADER_SIZE);

	m_mmap = new MemoryMappedFile(filename, readOnly);
	if (m_mmap->getSize() < sizeof(Header))
		Log(EError, "BSDF storage file \"%s\" has a truncated header!",
				filename.string().c_str());

	m_header = (Header *) m_mmap->getData();
	const char *id = BSDF_STORAGE_HEADER_ID;
	const int len = strlen(BSDF_STORAGE_HEADER_ID);
	if (memcmp(id, m_header->identifier, len) != 0)
		Log(EError, "BSDF storage file \"%s\" has a corrupt header!",
				filename.string().c_str());

	size_t
		nNodes = m_header->nNodes,
		nMaxOrder = m_header->nMaxOrder,
		nChannels = m_header->nChannels,
		nBases = m_header->nBases,
		nParameters = m_header->nParameters,
		nCoeffs = m_header->nCoeffs,
		nParameterValues = m_header->nParameterValues,
		nMetadataBytes = m_header->nMetadataBytes;

	size_t size = BSDF_STORAGE_HEADER_SIZE + // Header
		sizeof(float)*nNodes +               // Node locations
		sizeof(uint32_t)*nParameters +       // Parameter sample counts
		sizeof(float)*nParameterValues +     // Parameter sample positions
		sizeof(float)*nNodes*nNodes*nBases + // CDF in \mu
		sizeof(OffsetType)*nNodes*nNodes*2 + // Offset + size table
		sizeof(float)*nCoeffs +              // Fourier coefficients
		nMetadataBytes;                      // Metadata

	size_t uncompressedSize = size - sizeof(float)*nCoeffs
		+ nNodes*nNodes*nChannels*nBases*nMaxOrder*sizeof(float);

	if (m_mmap->getSize() != size)
		Log(EError, "BSDF storage file \"%s\" has an invalid size! (it"
			" is potentially truncated)", filename.string().c_str());

	Log(EInfo, "Mapped sparse BSDF storage file \"%s\" into memory:", filename.filename().string().c_str());
	Log(EInfo, "  Discretizations in mu  : " SIZE_T_FMT, nNodes);
	if (!isExtrapolated())
		Log(EInfo, "  Max. Fourier orders    : " SIZE_T_FMT, nMaxOrder);
	else
		Log(EInfo, "  Harmonic extrapolation : yes");
	Log(EInfo, "  Color channels         : " SIZE_T_FMT, nChannels);
	Log(EInfo, "  Textured parameters    : " SIZE_T_FMT, nParameters);
	Log(EInfo, "  Basis functions        : " SIZE_T_FMT, nBases);
	Log(EInfo, "  Uncompressed size      : %s", memString(uncompressedSize).c_str());
	Log(EInfo, "  Actual size            : %s (reduced to %.2f%%)", memString(size).c_str(),
			100 * size / (Float) uncompressedSize);

	m_nodes = m_header->data;
	m_paramSampleCounts = (uint32_t *) (m_nodes + nNodes);
	m_paramSamplePositions = (float *) (m_paramSampleCounts + nParameters);
	m_cdfMu = m_paramSamplePositions + nParameterValues;
	m_offsetTable = (OffsetType *) (m_cdfMu + nNodes*nNodes*nBases);
	m_coeffs = (float *) (m_offsetTable + nNodes*nNodes*2);

	size_t idx = 0;
	m_paramSamplePositionsNested = new float*[nParameters];
	for (size_t i=0; i<nParameters; ++i) {
		m_paramSamplePositionsNested[i] = m_paramSamplePositions + idx;
		idx += m_paramSampleCounts[i];
	}

	m_metadata.resize(nMetadataBytes);
	memcpy(&m_metadata[0], m_coeffs + nCoeffs, nMetadataBytes);

	int extra = LANE_WIDTH + LANE_WIDTH-1;
	m_reciprocals = (double *) allocAligned((nMaxOrder+extra) * sizeof(double));
	memset(m_reciprocals, 0, sizeof(double) * (nMaxOrder+extra));
	m_reciprocals += LANE_WIDTH-1;
	for (uint32_t i=0; i<nMaxOrder+LANE_WIDTH; ++i)
		m_reciprocals[i] = 1.0 / (double) i;
}

BSDFStorage::BSDFStorage(Stream *stream, InstanceManager *manager)
		: SerializableObject(stream, manager), m_header(NULL),
		  m_reciprocals(NULL), m_paramSamplePositionsNested(NULL) {
	size_t size = stream->readSize();
	uint8_t *data = (uint8_t *) allocAligned(size);

	stream->read(data, size);
	m_header = (Header *) data;

	size_t
		nNodes = m_header->nNodes,
		nMaxOrder = m_header->nMaxOrder,
		nChannels = m_header->nChannels,
		nBases = m_header->nBases,
		nParameters = m_header->nParameters,
		nCoeffs = m_header->nCoeffs,
		nParameterValues = m_header->nParameterValues,
		nMetadataBytes = m_header->nMetadataBytes;

	size_t expSize = BSDF_STORAGE_HEADER_SIZE + // Header
		sizeof(float)*nNodes +               // Node locations
		sizeof(uint32_t)*nParameters +       // Parameter sample counts
		sizeof(float)*nParameterValues +     // Parameter sample positions
		sizeof(float)*nNodes*nNodes*nBases + // CDF in \mu
		sizeof(OffsetType)*nNodes*nNodes*2 + // Offset + size table
		sizeof(float)*nCoeffs +              // Fourier coefficients
		nMetadataBytes;                      // Metadata

	if (expSize != size)
		Log(EError, "Internal error while unserializing BSDF storage file");

	size_t uncompressedSize = size - sizeof(float)*nCoeffs
		+ nNodes*nNodes*nChannels*nBases*nMaxOrder*sizeof(float);

	Log(EInfo, "Unserialized sparse BSDF model:");
	Log(EInfo, "  Discretizations in mu  : " SIZE_T_FMT, nNodes);
	if (!isExtrapolated())
		Log(EInfo, "  Max. Fourier orders    : " SIZE_T_FMT, nMaxOrder);
	else
		Log(EInfo, "  Harmonic extrapolation : yes");
	Log(EInfo, "  Color channels         : " SIZE_T_FMT, nChannels);
	Log(EInfo, "  Textured parameters    : " SIZE_T_FMT, nParameters);
	Log(EInfo, "  Basis functions        : " SIZE_T_FMT, nBases);
	Log(EInfo, "  Uncompressed size      : %s", memString(uncompressedSize).c_str());
	Log(EInfo, "  Actual size            : %s (reduced to %.2f%%)", memString(size).c_str(),
			100 * size / (Float) uncompressedSize);

	m_nodes = m_header->data;
	m_paramSampleCounts = (uint32_t *) (m_nodes + nNodes);
	m_paramSamplePositions = (float *) (m_paramSampleCounts + nParameters);
	m_cdfMu = m_paramSamplePositions + nParameterValues;
	m_offsetTable = (OffsetType *) (m_cdfMu + nNodes*nNodes*nBases);
	m_coeffs = (float *) (m_offsetTable + nNodes*nNodes*2);

	size_t idx = 0;
	m_paramSamplePositionsNested = new float*[nParameters];
	for (size_t i=0; i<nParameters; ++i) {
		m_paramSamplePositionsNested[i] = m_paramSamplePositions + idx;
		idx += m_paramSampleCounts[i];
	}

	m_metadata.resize(nMetadataBytes);
	memcpy(&m_metadata[0], m_coeffs + nCoeffs, nMetadataBytes);

	int extra = LANE_WIDTH + LANE_WIDTH-1;
	m_reciprocals = (double *) allocAligned((nMaxOrder+extra) * sizeof(double));
	memset(m_reciprocals, 0, sizeof(double) * (nMaxOrder+extra));
	m_reciprocals += LANE_WIDTH-1;
	for (uint32_t i=0; i<nMaxOrder+LANE_WIDTH; ++i)
		m_reciprocals[i] = 1.0 / (double) i;

}

BSDFStorage::~BSDFStorage() {
	if (m_reciprocals)
		freeAligned(m_reciprocals - (LANE_WIDTH-1));
	if (m_mmap.get() == NULL && m_header != NULL)
		freeAligned(m_header);
	if (m_paramSamplePositionsNested)
		delete[] m_paramSamplePositionsNested;
}

void BSDFStorage::serialize(Stream *stream, InstanceManager *manager) const {
	stream->writeSize(m_mmap->getSize());
	stream->write(m_mmap->getData(), m_mmap->getSize());
}

std::string BSDFStorage::readMetadata(const fs::path &filename) {
	ref<FileStream> fs = new FileStream(filename, FileStream::EReadOnly);

	Header header;
	fs->read(&header, sizeof(Header));

	const char *id = BSDF_STORAGE_HEADER_ID;
	const int len = strlen(BSDF_STORAGE_HEADER_ID);
	if (memcmp(id, header.identifier, len) != 0)
		Log(EError, "BSDF storage file \"%s\" has a corrupt header!",
				filename.string().c_str());

	size_t
		nNodes = header.nNodes,
		nBases = header.nBases,
		nParameters = header.nParameters,
		nCoeffs = header.nCoeffs,
		nParameterValues = header.nParameterValues,
		nMetadataBytes = header.nMetadataBytes,
		fileSize = fs->getSize();

	size_t expSize = BSDF_STORAGE_HEADER_SIZE + // Header
		sizeof(float)*nNodes +               // Node locations
		sizeof(uint32_t)*nParameters +       // Parameter sample counts
		sizeof(float)*nParameterValues +     // Parameter sample positions
		sizeof(float)*nNodes*nNodes*nBases + // CDF in \mu
		sizeof(OffsetType)*nNodes*nNodes*2 + // Offset + size table
		sizeof(float)*nCoeffs +              // Fourier coefficients
		nMetadataBytes;                      // Metadata

	if (fileSize != expSize)
		Log(EError, "BSDF storage file \"%s\" has an invalid size! (it"
			" is potentially truncated)", filename.string().c_str());

	std::string metadata;
	metadata.resize(nMetadataBytes);
	fs->seek(fileSize - nMetadataBytes);
	fs->read(&metadata[0], nMetadataBytes);
	return metadata;
}


ref<BSDFStorage> BSDFStorage::fromLayerGeneral(const fs::path &filename,
			const Layer **layers, size_t nChannels, size_t nBases, size_t nParameters,
			const size_t *paramSampleCounts, const float **paramSamplePositions,
			bool extrapolate, bool isBSDF, const std::string &metadata) {
	const Layer &layer = *layers[0];
	size_t n = layer.getNodeCount(), h = n/2;

	/* Insert an explicit mu=0 node to simplify the evaluation / sampling code */
	DVector nodes;

	if (isBSDF)
		nodes = (DVector(n+2) << layer.getNodes().head(h).reverse(), 0, 0, layer.getNodes().tail(h)).finished();
	else
		nodes = (DVector(n) << layer.getNodes().head(h).reverse(), layer.getNodes().tail(h)).finished();

	Log(EInfo, "BSDFStorage::fromLayerGeneral(): merging %u layers into \"%s\" "
			"- analyzing sparsity pattern..", nBases*nChannels, filename.filename().string().c_str());

	/* Do a huge matrix transpose */
	size_t maxCoeffs = nodes.size()*nodes.size()*nBases*nChannels*
		(extrapolate ? 3 : layer.getFourierOrders());
	size_t nNodes = (size_t) nodes.size();
	#if defined(MTS_GCD)
		__block int64_t __m = 0;
	#else
		int64_t __m = 0;
	#endif

	OffsetType *offsetTable = new OffsetType[nodes.size()*nodes.size()*2];

	#if defined(MTS_GCD)
		dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
		dispatch_apply(nNodes, queue, ^(size_t i) {
	#elif defined(MTS_OPENMP)
		#pragma omp parallel for schedule(dynamic)
		for (size_t i=0; i<nNodes; ++i) {
	#else
		for (size_t i=0; i<nNodes; ++i) {
	#endif
	{
		/* Parallel loop over 'i' */

		for (size_t o=0; o<nNodes; ++o) {
			size_t ip, op;
			size_t offset = (o + i * nNodes) * 2;

			if (EXPECT_TAKEN(isBSDF)) {
				if (i == h || i == h+1 || o == h || o == h+1) {
					offsetTable[offset + 0] = 0;
					offsetTable[offset + 1] = 0;
					continue;
				}

				ip = i < h ? (h-i-1) : (i-2);
				op = o < h ? (h-o-1) : (o-2);
			} else {
				ip = i < h ? h-i-1 : i;
				op = o < h ? h-o-1 : o;
			}

			size_t nCoeffs = 0;
			for (size_t basis=0; basis<nBases; ++basis) {
				for (size_t ch=0; ch<nChannels; ++ch) {
					size_t sparseSize = 0;
					float ref = std::abs((float) (*layers[basis*nChannels+ch])[0].coeff(op, ip));
					float partialSum = 0;
					if (ref != 0) {
						sparseSize = (size_t) layer.getFourierOrders();
						for (size_t j=(size_t) layer.getFourierOrders()-1; j>=1; --j) {
							float value = (float) (*layers[basis*nChannels+ch])[j].coeff(op, ip);
							partialSum += std::abs(value);
							if (partialSum <= ref * ERROR_GOAL || value == 0)
								sparseSize = j;
						}
						nCoeffs = std::max(nCoeffs, sparseSize);
					}
				}
			}
			if (extrapolate && nCoeffs > 0)
				nCoeffs = 3;

			atomicMaximum(&__m, nCoeffs);

			offsetTable[offset + 0] = 0;
			offsetTable[offset + 1] = (OffsetType) nCoeffs;
		}
	}
	#if defined(MTS_GCD)
		} );
	#else
		}
	#endif

	size_t m = (size_t) __m;
	if (extrapolate)
		m = 3;

	/* Compute the offset table */
	size_t totalCoeffs = 0;
	for (size_t i=0; i<nNodes*nNodes; ++i) {
		offsetTable[2*i] = (OffsetType) totalCoeffs;
		totalCoeffs += offsetTable[2*i + 1] * nBases*nChannels;
	}

	Log(EInfo, "Done. Number of coefficients: " SIZE_T_FMT " / " SIZE_T_FMT ", sparsity=%.2f%%",
		totalCoeffs, maxCoeffs, 100 * (Float) totalCoeffs / (Float) maxCoeffs);

	ref<BSDFStorage> storage = new BSDFStorage(filename, nNodes, nChannels, m,
		totalCoeffs, nBases, nParameters, paramSampleCounts, paramSamplePositions,
		extrapolate, isBSDF, metadata);

	Log(EInfo, "Copying data into sparse BSDF file ..");
	for (size_t i=0; i<nNodes; ++i)
		storage->m_nodes[i] = (float) nodes[i];

	memcpy(storage->getOffsetTable(), offsetTable,
		nNodes*nNodes*2*sizeof(OffsetType));

	/* Do a huge matrix transpose */
	for (size_t i=0; i<nNodes; ++i) {
		for (size_t o=0; o<nNodes; ++o) {
			std::pair<float *, OffsetType> coeffsAndCount = storage->getCoefficientsAndCount(o, i);
			float *coeffs = coeffsAndCount.first;
			OffsetType size = coeffsAndCount.second;

			size_t ip, op;

			if (EXPECT_TAKEN(isBSDF)) {
				if (i == h || o == h) {
					Assert(size == 0);
					continue;
				}

				ip = i < h ? (h-i-1) : (i-2);
				op = o < h ? (h-o-1) : (o-2);
			} else {
				ip = i < h ? h-i-1 : i;
				op = o < h ? h-o-1 : o;
			}

			float weight;
			if (isBSDF)
				weight = (float) std::abs(nodes[o] / (M_PI * nodes[i] * layer.getWeights()[ip]));
			else
				weight = (float) (1 / (M_PI * layer.getWeights()[ip]));


			if (nChannels == 1) {
				for (size_t basis=0; basis<nBases; ++basis) {
					for (OffsetType j=0; j<size; ++j) {
						float value = (float) (*layers[basis])[j].coeff(op, ip) *
							weight * (j == 0 ? 0.5f : 1.0f);
						if (!std::isfinite(value))
							Log(EWarn, "Encountered invalid data: %f", value);

						*coeffs++ = value;
					}
				}
			} else if (nChannels == 3) {
				float *coeffsY = coeffs;
				float *coeffsR = coeffsY + size*nBases;
				float *coeffsB = coeffsR + size*nBases;

				for (size_t basis=0; basis<nBases; ++basis) {
					for (OffsetType j=0; j<size; ++j) {
						float weight2 = weight * (j == 0 ? 0.5f : 1.0f);
						float R = (float) (*layers[basis*nChannels+0])[j].coeff(op, ip) * weight2;
						float G = (float) (*layers[basis*nChannels+1])[j].coeff(op, ip) * weight2;
						float B = (float) (*layers[basis*nChannels+2])[j].coeff(op, ip) * weight2;

						float Y = R * 0.212671f + G * 0.715160f + B * 0.072169f;
						if (!std::isfinite(Y))
							Log(EWarn, "Encountered invalid data: %f", Y);

						*coeffsY++ = Y; *coeffsR++ = R; *coeffsB++ = B;
					}
				}
			}
		}
	}

	Log(EInfo, "Computing cumulative distributions for importance sampling ..");

	/* Create an importance sampling CDF */
	Float *splineValues = new Float[nNodes];
	for (size_t i=0; i<nNodes; ++i) {
		for (size_t basis=0; basis<nBases; ++basis) {
			for (size_t o=0; o<nNodes; ++o) {
				std::pair<const float *, OffsetType> coeffAndCount = storage->getCoefficientsAndCount(o, i, basis, 0);
				splineValues[o] = coeffAndCount.second ? *coeffAndCount.first : 0;
			}

			float *cdf = storage->getCDF(i) + basis;
			size_t idx = 0;

			cdf[(idx++) * nBases] = 0;
			if (isBSDF) {
				for (size_t j=0; j<h; ++j)
					cdf[(idx++) * nBases] = (float) integrateCubicInterp1DN(j, nodes.data(), splineValues, h+1);
				for (size_t j=0; j<h; ++j)
					cdf[(idx++) * nBases] = (float) integrateCubicInterp1DN(j, nodes.data()+h, splineValues+h, h+1);
			} else {
				for (size_t j=0; j<nNodes-1; ++j)
					cdf[(idx++) * nBases] = (float) integrateCubicInterp1DN(j, nodes.data(), splineValues, nNodes);
			}

			for (size_t j=1; j<nNodes; ++j)
				cdf[j*nBases] += cdf[(j-1)*nBases];
		}
	}
	delete[] splineValues;

	if (extrapolate) {
		Log(EInfo, "Performing harmonic extrapolation ..");
		SAssert(totalCoeffs % 3 == 0);
		for (size_t i=0; i<totalCoeffs; i += 3)
			HarmonicExtrapolation::transform(storage->m_coeffs + i, storage->m_coeffs + i);
	}
	Log(EInfo, "BSDFStorage::fromLayerGeneral(): Done.");

	return storage;
}

ref<BSDFStorage> BSDFStorage::merge(const fs::path &filename, const BSDFStorage *s0, const BSDFStorage *s1) {
	if (!s0->m_header || !s1->m_header || !s0->m_mmap || !s1->m_mmap)
		Log(EError, "BSDFStorage::merge(): given invalid/unitialized input instances!");
	if (s0->m_header->flags != s1->m_header->flags)
		Log(EError, "BSDFStorage::merge(): file flags mismatch!");
	if (s0->getNodeCount() != s1->getNodeCount())
		Log(EError, "BSDFStorage::merge(): node count mismatch!");
	if (s0->getChannelCount() != s1->getChannelCount())
		Log(EError, "BSDFStorage::merge(): channel count mismatch!");
	if (s0->getEta() != s1->getEta())
		Log(EWarn, "BSDFStorage::merge(): eta mismatch (%f vs %f)!", s0->getEta(), s1->getEta());
	if (s0->getBasisCount() != 1 || s1->getBasisCount() != 1)
		Log(EWarn, "BSDFStorage::merge(): unsupported (multiple basis functions).");

	Log(EInfo, "Merging sparse BSDF storage files \"%s\" and \"%s\" into \"%s\"",
		s0->m_filename.string().c_str(),
		s1->m_filename.string().c_str(),
		filename.string().c_str());

	size_t
		nMaxOrder = std::max(s0->getMaxOrder(), s1->getMaxOrder()),
		nNodes = s0->getNodeCount(),
		nChannels = s0->getChannelCount(),
		nBases = s0->getBasisCount() + s1->getBasisCount(),
		nParameters = 1;

	OffsetType *offsetTable = new OffsetType[nNodes*nNodes*2];

	size_t totalCoeffs = 0;
	for (size_t i=0; i<nNodes; ++i) {
		for (size_t o=0; o<nNodes; ++o) {
			size_t index = (o + i * nNodes) * 2;
			OffsetType maxSize = std::max(s0->m_offsetTable[index+1], s1->m_offsetTable[index+1]);

			offsetTable[index+0] = (OffsetType) totalCoeffs;
			offsetTable[index+1] = maxSize;
			totalCoeffs += maxSize * nBases * nChannels;
		}
	}

	const size_t paramSampleCounts[1] = { 2 };
	float paramSamplePositions0[2] = { 0.0f, 1.0f };
	const float *paramSamplePositions = { paramSamplePositions0 };

	ref<BSDFStorage> storage = new BSDFStorage(filename, nNodes, nChannels, nMaxOrder,
			totalCoeffs, nBases, nParameters, paramSampleCounts, &paramSamplePositions, s0->isExtrapolated(),
			s0->isBSDF(), "");

	memcpy(storage->m_nodes, s0->m_nodes, nNodes * sizeof(float));
	memcpy(storage->getOffsetTable(), offsetTable,
		nNodes*nNodes*2*sizeof(OffsetType));
	storage->m_header->eta = (s0->getEta()*s0->getBasisCount() + s1->getEta()*s1->getBasisCount())
		/ (s0->getBasisCount() + s1->getBasisCount());

	for (size_t i=0; i<nNodes; ++i) {
		for (size_t o=0; o<nNodes; ++o) {
			std::pair<const float *, OffsetType> coeffsAndCount0 = s0->getCoefficientsAndCount(o, i);
			std::pair<const float *, OffsetType> coeffsAndCount1 = s1->getCoefficientsAndCount(o, i);
			std::pair<float *, OffsetType> coeffsAndCountO = storage->getCoefficientsAndCount(o, i);

			const float *coeffs0 = coeffsAndCount0.first, *coeffs1 = coeffsAndCount1.first;
			OffsetType count0 = coeffsAndCount0.second, count1 = coeffsAndCount1.second;
			OffsetType count = std::max(count0, count1);

			float *target = coeffsAndCountO.first;
			Assert(count == coeffsAndCountO.second);

			for (size_t channel=0; channel<nChannels; ++channel) {
				for (size_t basis=0; basis<s0->getBasisCount(); ++basis) {
					memcpy(target, coeffs0, sizeof(float) * count0);
					memset(target+count0, 0, (count-count0)*sizeof(float));
					coeffs0 += count0;
					target  += count;
				}
				for (size_t basis=0; basis<s1->getBasisCount(); ++basis) {
					memcpy(target, coeffs1, sizeof(float) * count1);
					memset(target+count1, 0, (count-count1)*sizeof(float));
					coeffs1 += count1;
					target  += count;
				}
			}
		}
	}

	const float *cdf0 = s0->m_cdfMu, *cdf1 = s1->m_cdfMu;
	float *target = storage->m_cdfMu;

	size_t size = nNodes*nNodes;
	for (size_t i=0; i<size; ++i) {
		for (size_t basis=0; basis<s0->getBasisCount(); ++basis)
			*target++ = *cdf0++;
		for (size_t basis=0; basis<s1->getBasisCount(); ++basis)
			*target++ = *cdf1++;
	}

	return storage;
}

size_t BSDFStorage::size() const {
	if (!m_mmap)
		return 0;
	return m_mmap->getSize();
}

bool BSDFStorage::isExtrapolated() const {
	return m_header->flags & BSDF_STORAGE_FLAGS_EXTRAPOLATED;
}

bool BSDFStorage::isBSDF() const {
	return m_header->flags & BSDF_STORAGE_FLAGS_BSDF;
}

Spectrum BSDFStorage::eval(Float mu_i, Float mu_o, Float phi_d, const float *basisCoeffs) const {
	if (!basisCoeffs) {
		SAssert(getBasisCount() == 1);
		basisCoeffs = __basisCoeffsDefault;
	}

	size_t knotOffsetO, knotOffsetI;
	float knotWeightsO[4], knotWeightsI[4];

	computeInterpolationWeights((float) mu_o, knotOffsetO, knotWeightsO);
	computeInterpolationWeights((float) mu_i, knotOffsetI, knotWeightsI);

	size_t nChannels = getChannelCount(), nBases = getBasisCount();
	OffsetType nCoeffs = 0;

	float *coeffs[3];
	for (size_t i=0; i<nChannels; ++i)
		coeffs[i] = fourier_aligned_alloca(getMaxOrder() * sizeof(float));

	for (int i=0; i<4; ++i) {
		for (int o=0; o<4; ++o) {
			float weight = knotWeightsO[o] * knotWeightsI[i];
			if (weight == 0)
				continue;

			std::pair<const float *, OffsetType> coeffAndCount = getCoefficientsAndCount(knotOffsetO + o, knotOffsetI + i);

			const float *source = coeffAndCount.first;
			OffsetType count = coeffAndCount.second;

			if (count == 0)
				continue;

			nCoeffs = std::max(nCoeffs, count);

			for (size_t channel=0; channel<nChannels; ++channel) {
				for (size_t basis=0; basis<nBases; ++basis) {
					float interpWeight = weight * basisCoeffs[channel*nBases+basis];
					if (interpWeight == 0) {
						source += count;
						continue;
					}
					float *target = coeffs[channel];
					OffsetType remainder = count;

					#if MTS_FOURIER_VECTORIZED == 1
						/* Copy first (unaligned) element using scalar arithmetic */
						*target++ += *source++ * interpWeight; --remainder;

						/* Copy as many elements as possible using AVX */
						__m256 weight_vec = _mm256_set1_ps(interpWeight);
						OffsetType simdCount = remainder & ~7;
						for (OffsetType k=0; k<simdCount; k += 8)
							_mm256_store_ps(target+k, _mm256_add_ps(_mm256_load_ps(target+k),
								_mm256_mul_ps(_mm256_loadu_ps(source+k), weight_vec)));

						source += simdCount; target += simdCount; remainder -= simdCount;
					#endif

					for (OffsetType k=0; k<remainder; ++k)
						*target++ += *source++ * interpWeight;
				}
			}
		}
	}

	Spectrum result;
	if (nCoeffs == 0 || coeffs[0][0] == 0.0f) {
		result = Spectrum(0.0f);
	} else if (m_header->flags & BSDF_STORAGE_FLAGS_EXTRAPOLATED) {
		float phi_d_sp = (float) phi_d;

		for (size_t ch=0; ch<nChannels; ++ch) {
			coeffs[ch][0] = std::max(0.0f, coeffs[ch][0]);
			coeffs[ch][1] = std::max(0.0f, std::min(1.0f, coeffs[ch][1]));
			coeffs[ch][2] = std::max(1e-6f,coeffs[ch][2]);
		}

		if (nChannels == 1) {
			result = Spectrum((Float) HarmonicExtrapolation::eval(coeffs[0], phi_d_sp));
		} else {
			Float Y = HarmonicExtrapolation::eval(coeffs[0], phi_d_sp);
			Float R = HarmonicExtrapolation::eval(coeffs[1], phi_d_sp);
			Float B = HarmonicExtrapolation::eval(coeffs[2], phi_d_sp);
			Float G = 1.39829f*Y - 0.100913f*B - 0.297375f*R;
			result.fromLinearRGB(R, G, B);
		}
	} else if (nChannels == 1) {
		result = Spectrum(std::max((Float) 0, (Float) evalFourier(coeffs[0], nCoeffs, phi_d)));
	} else {
		result = evalFourier3(coeffs, nCoeffs, phi_d);
	}
	result.clampNegative();

	return result;
}

Float BSDFStorage::pdf(Float mu_i, Float mu_o, Float phi_d, const float *basisCoeffs) const {
	if (!basisCoeffs) {
		SAssert(getBasisCount() == 1);
		basisCoeffs = __basisCoeffsDefault;
	}

	size_t knotOffsetO, knotOffsetI;
	float knotWeightsO[4], knotWeightsI[4];

	computeInterpolationWeights((float) mu_o, knotOffsetO, knotWeightsO);
	computeInterpolationWeights((float) mu_i, knotOffsetI, knotWeightsI);

	size_t nBases = getBasisCount();
	OffsetType nCoeffs = 0;

	float *coeffs = fourier_aligned_alloca(getMaxOrder() * sizeof(float));

	for (int i=0; i<4; ++i) {
		for (int o=0; o<4; ++o) {
			float weight = knotWeightsO[o] * knotWeightsI[i];
			if (weight == 0)
				continue;

			std::pair<const float *, OffsetType> coeffAndCount = getCoefficientsAndCount(knotOffsetO + o, knotOffsetI + i);

			const float *source = coeffAndCount.first;
			OffsetType count = coeffAndCount.second;

			if (count == 0)
				continue;

			nCoeffs = std::max(nCoeffs, count);

			for (size_t basis=0; basis<nBases; ++basis) {
				float interpWeight = weight * basisCoeffs[basis];
				if (interpWeight == 0) {
					source += count;
					continue;
				}
				float *target = coeffs;
				OffsetType remainder = count;

				#if MTS_FOURIER_VECTORIZED == 1
					/* Copy first (unaligned) element using scalar arithmetic */
					*target++ += *source++ * interpWeight; --remainder;

					/* Copy as many elements as possible using AVX */
					__m256 weight_vec = _mm256_set1_ps(interpWeight);
					OffsetType simdCount = remainder & ~7;
					for (OffsetType k=0; k<simdCount; k += 8)
						_mm256_store_ps(target+k, _mm256_add_ps(_mm256_load_ps(target+k),
							_mm256_mul_ps(_mm256_loadu_ps(source+k), weight_vec)));

					source += simdCount; target += simdCount; remainder -= simdCount;
				#endif

				for (OffsetType k=0; k<remainder; ++k)
					*target++ += *source++ * interpWeight;
			}
		}
	}

	Float pdfMu = coeffs[0] / evalLatitudinalCDF(knotOffsetI,
			knotWeightsI, getNodeCount()-1, basisCoeffs);

	if (nCoeffs == 0 || coeffs[0] == 0.0f) {
		return 0.0f;
	} else if (m_header->flags & BSDF_STORAGE_FLAGS_EXTRAPOLATED) {
		coeffs[0] = std::max(0.0f, coeffs[0]);
		coeffs[1] = std::max(0.0f, std::min(1.0f, coeffs[1]));
		coeffs[2] = std::max(1e-6f,coeffs[2]);

		return HarmonicExtrapolation::pdf(coeffs, (float) phi_d) * pdfMu;
	} else {
		return std::max((Float) 0, pdfFourier(coeffs, nCoeffs, phi_d) * pdfMu);
	}
}

Spectrum BSDFStorage::sample(Float mu_i, Float &mu_o, Float &phi_d,
		Float &pdf, const Point2 &sample, const float *basisCoeffs) const {
	if (!basisCoeffs) {
		SAssert(getBasisCount() == 1);
		basisCoeffs = __basisCoeffsDefault;
	}

	size_t knotOffsetI, n = getNodeCount();
	float knotWeightsI[4];

	/* Lookup spline nodes and weights for mu_i */
	computeInterpolationWeights((float) mu_i, knotOffsetI, knotWeightsI);

	/* Account for energy loss */
	float normalization = evalLatitudinalCDF(knotOffsetI, knotWeightsI, n-1, basisCoeffs);
	float sample_y = (float) sample.y * normalization;

	/* Binary search for the spline segment containing the outgoing angle */
	size_t first = 0, len = n;
	while (len > 0) {
		ssize_t half = len >> 1, middle = first + half;
		if (evalLatitudinalCDF(knotOffsetI, knotWeightsI, middle, basisCoeffs) < sample_y) {
			first = middle + 1;
			len -= half + 1;
		} else {
			len = half;
		}
	}

	size_t index = std::min(n-2, std::max((size_t) 0, first-1));

	/* The spline segment to be sampled has been chosen. Determine the
	   values of its nodes and then use the inversion method to sample
	   an exact position within the segment */
	float cdf0  = evalLatitudinalCDF(knotOffsetI, knotWeightsI, index, basisCoeffs),
	      cdf1  = evalLatitudinalCDF(knotOffsetI, knotWeightsI, index+1, basisCoeffs),
	      f0    = evalLatitudinalAverage(knotOffsetI, knotWeightsI, index, basisCoeffs),
	      f1    = evalLatitudinalAverage(knotOffsetI, knotWeightsI, index+1, basisCoeffs),
	      width = m_nodes[index+1] - m_nodes[index],
	      d0, d1;

	/* Catmull-Rom spline: approximate the derivatives at the endpoints
	   using finite differences */
	if (index > 0) {
		d0 = width / (m_nodes[index+1] - m_nodes[index-1]) *
			(f1 - evalLatitudinalAverage(knotOffsetI, knotWeightsI, index-1, basisCoeffs));
	} else {
		d0 = f1 - f0;
	}

	if (index + 2 < n) {
		d1 = width / (m_nodes[index+2] - m_nodes[index]) *
			(evalLatitudinalAverage(knotOffsetI, knotWeightsI, index+2, basisCoeffs) - f0);
	} else {
		d1 = f1 - f0;
	}

	/* Bracketing interval and starting guess */
	float a = 0, c = 1, b;

	b          = (sample_y-cdf0) / (cdf1-cdf0);
	sample_y   = (sample_y-cdf0) / width;

	if (f0 != f1) /* Importance sample linear interpolant */
		b = (f0-math::safe_sqrt(f0*f0 + b * (f1*f1-f0*f0))) / (f0-f1);

	/* Invert CDF using Newton-Bisection */
	int it = 0;
	while (true) {
		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		/* CDF and PDF in Horner form */
		float value = b*(f0 + b*(.5f*d0 + b*((1.0f/3.0f) * (-2*d0-d1)
			+ f1 - f0 + b*(0.25f*(d0 + d1) + 0.5f * (f0 - f1))))) - sample_y;
		float deriv = f0 + b*(d0 + b*(-2*d0 - d1 + 3*(f1-f0) + b*(d0 + d1 + 2*(f0 - f1))));

		if (std::abs(value) < 1e-6f * deriv || ++it > 10) {
			mu_o = m_nodes[index] + width*b;
			break;
		}

		if (value > 0)
			c = b;
		else
			a = b;

		b -= value / deriv;
	}
	/* Outgoing zenith angle has been sampled -- interpolate
	   Fourier coefficients and sample the series */
	size_t knotOffsetO;
	float knotWeightsO[4];
	computeInterpolationWeights((float) mu_o, knotOffsetO, knotWeightsO);

	size_t nChannels = getChannelCount(), nBases = getBasisCount();
	OffsetType nCoeffs = 0;

	float *coeffs[3];
	for (size_t i=0; i<nChannels; ++i)
		coeffs[i] = fourier_aligned_alloca(getMaxOrder() * sizeof(float));

	for (int i=0; i<4; ++i) {
		for (int o=0; o<4; ++o) {
			float weight = knotWeightsO[o] * knotWeightsI[i];
			if (weight == 0)
				continue;

			std::pair<const float *, OffsetType> coeffAndCount = getCoefficientsAndCount(knotOffsetO + o, knotOffsetI + i);

			const float *source = coeffAndCount.first;
			OffsetType count = coeffAndCount.second;

			if (count == 0)
				continue;

			nCoeffs = std::max(nCoeffs, count);

			for (size_t channel=0; channel<nChannels; ++channel) {
				for (size_t basis=0; basis<nBases; ++basis) {
					float interpWeight = weight * basisCoeffs[channel*nBases+basis];
					if (interpWeight == 0) {
						source += count;
						continue;
					}
					float *target = coeffs[channel];
					OffsetType remainder = count;

					#if MTS_FOURIER_VECTORIZED == 1
						/* Copy first (unaligned) element using scalar arithmetic */
						*target++ += *source++ * interpWeight; --remainder;

						/* Copy as many elements as possible using AVX */
						__m256 weight_vec = _mm256_set1_ps(interpWeight);
						OffsetType simdCount = remainder & ~7;
						for (OffsetType k=0; k<simdCount; k += 8)
							_mm256_store_ps(target+k, _mm256_add_ps(_mm256_load_ps(target+k),
								_mm256_mul_ps(_mm256_loadu_ps(source+k), weight_vec)));

						source += simdCount; target += simdCount; remainder -= simdCount;
					#endif

					for (OffsetType k=0; k<remainder; ++k)
						*target++ += *source++ * interpWeight;
				}
			}
		}
	}

	Float pdfMu = coeffs[0][0] / normalization;
	Float pdfPhi = 0;
	Spectrum weight;

	if (coeffs[0][0] == 0) {
		weight = Spectrum(0.0f);
	} else if (m_header->flags & BSDF_STORAGE_FLAGS_EXTRAPOLATED) {
		for (size_t ch=0; ch<nChannels; ++ch) {
			coeffs[ch][0] = std::max(0.0f, coeffs[ch][0]);
			coeffs[ch][1] = std::max(0.0f, std::min(1.0f, coeffs[ch][1]));
			coeffs[ch][2] = std::max(1e-6f,coeffs[ch][2]);
		}

		Float phiWeight = HarmonicExtrapolation::sample(coeffs[0], phi_d, sample.x);
		float phi_d_sp = (float) phi_d;
		pdfPhi = HarmonicExtrapolation::pdf(coeffs[0], phi_d_sp);

		if (nChannels == 1) {
			weight = Spectrum(phiWeight * 2 * M_PI / pdfMu);
		} else {
			Float Y = HarmonicExtrapolation::eval(coeffs[0], phi_d_sp);
			Float R = HarmonicExtrapolation::eval(coeffs[1], phi_d_sp);
			Float B = HarmonicExtrapolation::eval(coeffs[2], phi_d_sp);
			Float G = 1.39829f*Y - 0.100913f*B - 0.297375f*R;
			weight.fromLinearRGB(R, G, B);
			weight /= pdfPhi * pdfMu;
		}
	} else if (nChannels == 1) {
		weight = Spectrum(std::max((Float) 0.0f, sampleFourier(coeffs[0], m_reciprocals,
			nCoeffs, (float) sample.x, pdfPhi, phi_d) / pdfMu));
	} else {
		weight = sampleFourier3(coeffs, m_reciprocals, nCoeffs,
			(float) sample.x, pdfPhi, phi_d) / pdfMu;
	}
	weight.clampNegative();

	pdf = std::max((Float) 0, pdfPhi * pdfMu);

	return weight;
}

void BSDFStorage::computeInterpolationWeights(float mu, size_t &offset, float (&weights)[4]) const {
	ssize_t size = m_header->nNodes;
	const float *nodes = m_nodes;

	/* Find the index of the left knot in the queried subinterval, be
	   robust to cases where 't' lies exactly on the right endpoint */
	ssize_t k = std::max((ssize_t) 0, std::min(size - 2,
		(ssize_t) (std::lower_bound(nodes, nodes+size, mu) - nodes - 1)));

	float width = nodes[k+1] - nodes[k];

	/* Compute the relative position within the interval */
	float t = (mu - nodes[k]) / width,
		  t2 = t*t, t3 = t2*t;

	/* Compute node weights */
	weights[0] = 0.0f;
	weights[1] = 2*t3 - 3*t2 + 1;
	weights[2] = -2*t3 + 3*t2;
	weights[3] = 0.0f;
	offset = k-1 + (nodes - m_nodes);

	/* Derivative weights */
	float d0 = t3 - 2*t2 + t, d1 = t3 - t2;

	/* Turn derivative weights into node weights using
	   an appropriate chosen finite differences stencil */
	if (k > 0) {
		float factor = width / (nodes[k+1]-nodes[k-1]);
		weights[2] += d0 * factor;
		weights[0] -= d0 * factor;
	} else {
		weights[2] += d0;
		weights[1] -= d0;
	}

	if (k + 2 < size) {
		float factor = width / (nodes[k+2]-nodes[k]);
		weights[3] += d1 * factor;
		weights[1] -= d1 * factor;
	} else {
		weights[2] += d1;
		weights[1] -= d1;
	}
}

void BSDFStorage::interpolateSeries(Float mu_i, Float mu_o, int basis, int channel, float *coeffs) const {
	size_t knotOffsetO, knotOffsetI;
	float knotWeightsO[4], knotWeightsI[4];

	computeInterpolationWeights((float) mu_o, knotOffsetO, knotWeightsO);
	computeInterpolationWeights((float) mu_i, knotOffsetI, knotWeightsI);

	memset(coeffs, 0, sizeof(float) * getMaxOrder());

	for (int i=0; i<4; ++i) {
		for (int o=0; o<4; ++o) {
			float weight = knotWeightsO[o] * knotWeightsI[i];
			if (weight == 0)
				continue;

			std::pair<const float *, OffsetType> coeffAndCount =
				getCoefficientsAndCount(knotOffsetO + o, knotOffsetI + i, basis, channel);

			const float *source = coeffAndCount.first;
			OffsetType count = coeffAndCount.second;
			float *target = coeffs;

			for (OffsetType k=0; k<count; ++k)
				*target++ += *source++ * weight;
		}
	}
}

std::string BSDFStorage::stats() const {
	std::ostringstream oss;

	size_t
		nNodes = m_header->nNodes,
		nMaxOrder = m_header->nMaxOrder,
		nChannels = m_header->nChannels,
		nBases = m_header->nBases,
		nParameters = m_header->nParameters,
		nCoeffs = m_header->nCoeffs,
		nParameterValues = m_header->nParameterValues,
		nMetadataBytes = m_header->nMetadataBytes;

	size_t size = BSDF_STORAGE_HEADER_SIZE + // Header
		sizeof(float)*nNodes +               // Node locations
		sizeof(uint32_t)*nParameters +       // Parameter sample counts
		sizeof(float)*nParameterValues +     // Parameter sample positions
		sizeof(float)*nNodes*nNodes*nBases + // CDF in \mu
		sizeof(OffsetType)*nNodes*nNodes*2 + // Offset + size table
		sizeof(float)*nCoeffs +              // Fourier coefficients
		nMetadataBytes;                      // Metadata

	size_t uncompressedSize = size - sizeof(float)*nCoeffs
		+ nNodes*nNodes*nChannels*nBases*nMaxOrder*sizeof(float);

	oss.precision(2);
	oss << " Discretizations in mu  : " << nNodes << endl;
	if (!isExtrapolated())
		oss << " Max. Fourier orders    : " << nMaxOrder << endl;
	else
		oss <<  " Harmonic extrapolation : yes" << endl;
	oss << " Color channels         : " << nChannels << endl;
	oss << " Textured parameters    : " << nParameters << endl;
	oss << " Basis functions        : " << nBases << endl;
	oss << " Uncompressed size      : " << memString(uncompressedSize) << endl;
	oss << " Actual size            : " << memString(size).c_str();
	oss << " (reduced to " << (100 * size / (Float) uncompressedSize) << "%)";
	return oss.str();
}

std::string BSDFStorage::toString() const {
	std::ostringstream oss;
	oss << "BSDFStorage[" << endl
		<< "  mmap = " << m_mmap.toString() << "," << endl;
	if (m_header) {
		oss << "  nNodes = " << m_header->nNodes << "," << endl
		    << "  nMaxOrder = " << m_header->nMaxOrder << "," << endl
		    << "  nChannels = " << m_header->nChannels << "," << endl
		    << "  nBases = " << m_header->nBases << "," << endl
		    << "  eta = " << m_header->eta << endl;
	}
	oss << "]";
	return oss.str();
}

MTS_IMPLEMENT_CLASS_S(BSDFStorage, false, SerializableObject)
MTS_NAMESPACE_END
