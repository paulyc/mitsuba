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
#if !defined(__MITSUBA_LAYER_FOURIER_H_)
#define __MITSUBA_LAYER_FOURIER_H_

#include <mitsuba/mitsuba.h>
#include <boost/function.hpp>

#if defined(__AVX__)
#define MTS_FOURIER_SCALAR     0
#define MTS_FOURIER_VECTORIZED 1
#define LANE_WIDTH             8
#else
#define MTS_FOURIER_SCALAR     1
#define MTS_FOURIER_VECTORIZED 0
#define LANE_WIDTH             1
#endif

#define MTS_LANE_SIZE_BYTES (LANE_WIDTH * 4)

MTS_NAMESPACE_BEGIN

/**
 * \brief Evaluate an even Fourier series (i.e. containing
 * only cosine terms).
 *
 * \param coeffs
 *     Coefficient storage
 * \param nCoeffs
 *     Denotes the size of \c coeffs
 * \param phi
 *     Angle for which the series should be evaluated
 * \return
 *     The value of the series for this angle
 */
extern MTS_EXPORT_LAYER Float evalFourier(const float *coeffs,
		int nCoeffs, Float phi);

/**
 * \brief Simultaneously evaluate <em>three</em> even Fourier series
 * corresponding to the color channels (Y, R, B) and return
 * a spectral power distribution
 *
 * \param coeffs
 *     Coefficient storage
 * \param nCoeffs
 *     Denotes the size of \c coeffs
 * \param phi
 *     Angle for which the series should be evaluated
 * \return
 *     The value of the series for this angle
 */
extern MTS_EXPORT_LAYER Spectrum evalFourier3(float * const coeffs[3],
		int nCoeffs, Float phi);

/**
 * \brief Sample a angle from an even Fourier series
 * (i.e. containing only cosine terms).
 *
 * This is done by importance sampling a uniform piecewise constant
 * approximation with 2^nRecursions elements, which is constructed
 * on the fly in a recursive manner.
 *
 * \param coeffs
 *     Coefficient storage
 * \param nCoeffs
 *     Denotes the size of \c coeffs
 * \param recip
 *     Precomputed array of integer reciprocals, i.e. <tt>inf, 1, 1/2, 1/3,
 *     1/4, ..</tt> of size <tt>nCoeffs-1</tt>. This is used to reduce
 *     integer-to-FP pipeline stalls and division latencies at runtime.
 * \param sample
 *     A uniformly distributed random sample in the interval <tt>[0,1]</tt>
 * \param[out] pdf
 *     This parameter is used to return the probability density of
 *     the sampling scheme evaluated at the generated point on the
 *     underlying piecewise constant approximation (on [0, \pi])
 * \param[out] phi
 *     Used to return the sampled angle (on [0, \pi])
 * \return
 *     The importance weight (i.e. the value of the Fourier series
 *     divided by \c pdf)
 */
extern MTS_EXPORT_LAYER Float sampleFourier(const float *coeffs, const double *recip,
		int nCoeffs, Float sample, Float &pdf, Float &phi);

/**
 * \brief Sample a angle from <em>three</tt> Fourier series
 * corresponding to the color channels (Y, R, B)
 *
 * This is done by importance sampling a uniform piecewise constant
 * approximation with respect to the luminance, which is constructed
 * on the fly in a recursive manner.
 *
 * \param coeffs
 *     Coefficient storage
 * \param nCoeffs
 *     Denotes the size of \c coeffs
 * \param recip
 *     Precomputed array of integer reciprocals, i.e. <tt>inf, 1, 1/2, 1/3,
 *     1/4, ..</tt> of size <tt>nCoeffs-1</tt>. This is used to reduce
 *     integer-to-FP pipeline stalls and division latencies at runtime.
 * \param sample
 *     A uniformly distributed random sample in the interval <tt>[0,1]</tt>
 * \param[out] pdf
 *     This parameter is used to return the probability density of
 *     the sampling scheme evaluated at the generated point on the
 *     underlying piecewise constant approximation (on [0, \pi])
 * \param[out] phi
 *     Used to return the sampled angle (on [0, \pi])
 * \return
 *     The importance weight (i.e. the value of the Fourier series
 *     divided by \c pdf)
 */
extern MTS_EXPORT_LAYER Spectrum sampleFourier3(float * const coeffs[3], const double *recip, int nCoeffs,
		Float sample, Float &pdf, Float &phi);

/**
 * \brief Evaluate the probability density of the sampling scheme
 * implemented by \ref sampleFourier() at the position \c phi.
 *
 * \param coeffs
 *     Coefficient storage
 * \param nCoeffs
 *     Denotes the size of \c coeffs
 * \return
 *     The continuous probability density on [0, \pi]
 */
inline Float pdfFourier(const float *coeffs, int nCoeffs, Float phi) {
	return evalFourier(coeffs, nCoeffs, phi) * INV_TWOPI / (Float) coeffs[0];
}

/**
 * \brief Computes the Fourier series of a product of even Fourier series
 * using discrete convolution.
 *
 * The input series are assumed to have \c ka and \c kb coefficients, and the
 * output must have room for <tt>ka+kb-1</tt> coefficients.
 */
extern MTS_EXPORT_LAYER void convolveFourier(const Float *a,
		int ka, const Float *b, int kb, Float *c);

/**
 * \brief Compute a Fourier series of the given even function
 * by integrating it against the basis functions using Filon
 * quadrature
 *
 * Filon quadrature works by constructing a piecewise quadratic
 * interpolant of the original function. The Fourier basis functions
 * are then integrated against this representation, which has an
 * analytic solution. This avoids all of the problems of traditional
 * quadrature schemes involving highly oscillatory integrals. It
 * is thus possible to obtain accurate coefficients even for high
 * orders.
 *
 * \param[out] coeffs
 *    Output buffer used to store the computed coefficients
 * \param nCoeffs
 *    Desired number of coefficients
 * \param nEvals
 *    Desired resolution of the piecewise quadratic interpolant
 * \param a
 *    Start of the integration, can optionally be set to values
 *    other than zero. Note that the Fourier basis functions
 *    are not orthogonal anymore in this case.
 * \param b
 *    End of the integration, can be set to values other
 *    than pi. Note that the Fourier basis functions
 *    are not orthogonal anymore in this case.
 */
extern MTS_EXPORT_LAYER void filonIntegrate(
		const boost::function<Float (Float)> &eval,
		Float *coeffs, int nCoeffs, int nEvals,
		Float a = 0, Float b = M_PI);

#if MTS_FOURIER_SCALAR == 1

#define fourier_aligned_alloca(size) \
	__align_helper(alloca(size), size)

static inline float *__align_helper(void *ptr, size_t size) {
	memset(ptr, 0, size);
	return (float *) ptr;
}

#else

/**
 * \brief Helper functions for allocating temporary stack memory for
 * Fourier coefficients.
 *
 * This macro works like alloca(), except that it also ensures that
 * the resulting buffer is properly aligned so that it can be used
 * with the SSE/AVX-vectorized evaluation and sampling routines.
 *
 * Given an allocation, we require memory for
 *
 *  1 float at alignment -4
 *  N quadruplets at alignment 0
 *
 * SSE: 12 bytes in addition to make sure that
 *      this alignment can be established
 * AVX: 28 bytes in addition to make sure that
 *      this alignment can be established
 *
 * i.e.
 *
 * SSE:  size = 4 + 16*((size-4 + 12) / 16) + 12;
 * SSE:  size = 4 + 32*((size-4 + 28) / 32) + 28;
 *
 * and to align:
 *
 * SSE:  buffer += 12 - buffer mod 16
 * AVX:  buffer += 28 - buffer mod 32
 */

#define fourier_aligned_alloca(size) \
	__align_helper(alloca(MTS_LANE_SIZE_BYTES*(((MTS_LANE_SIZE_BYTES-2*sizeof(float))+size) / MTS_LANE_SIZE_BYTES) + MTS_LANE_SIZE_BYTES), \
	                      MTS_LANE_SIZE_BYTES*(((MTS_LANE_SIZE_BYTES-2*sizeof(float))+size) / MTS_LANE_SIZE_BYTES) + MTS_LANE_SIZE_BYTES)

namespace {
	static inline float *__align_helper(void *ptr, size_t size) {
		memset(ptr, 0, size);
		return (float *) ((uint8_t *) ptr + (MTS_LANE_SIZE_BYTES - sizeof(float) - ((uintptr_t) ptr) % MTS_LANE_SIZE_BYTES));
	}
};

#endif

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_FOURIER_H_ */
