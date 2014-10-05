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

#include <mitsuba/layer/fourier.h>
#include <mitsuba/core/warp.h>

#if defined(__MSVC__)
# include <intrin.h>
#else
# include <immintrin.h>
#endif

MTS_NAMESPACE_BEGIN

#if MTS_FOURIER_VECTORIZED == 1

#define _mm256_set_ss(value)    _mm256_insertf128_ps(_mm256_setzero_ps(), _mm_set_ss(value), 0)
#define _mm256_set_sd(value)    _mm256_insertf128_pd(_mm256_setzero_pd(), _mm_set_sd(value), 0)
#define _mm256_splat0_pd(value) _mm256_permute2f128_pd(_mm256_shuffle_pd(value, value, 0x0), value, 0x00)
#define _mm256_splat1_pd(value) _mm256_permute2f128_pd(_mm256_shuffle_pd(value, value, 0x3), value, 0x00)
#define _mm256_splat2_pd(value) _mm256_permute2f128_pd(_mm256_shuffle_pd(value, value, 0x0), value, 0x11)
#define _mm256_splat3_pd(value) _mm256_permute2f128_pd(_mm256_shuffle_pd(value, value, 0xC), value, 0x11)


#if !defined(MM_ALIGN16)
# if defined(__GNUC__)
#  define MM_ALIGN16 __attribute__ ((aligned (16)))
# else
#  define MM_ALIGN16 __declspec(align(16))
# endif
#endif

#if !defined(MM_ALIGN32)
# if defined(__GNUC__)
#  define MM_ALIGN32 __attribute__ ((aligned (32)))
# else
#  define MM_ALIGN32 __declspec(align(32))
# endif
#endif

namespace simd {
	/* How to generate:
		c[0] = 1;
		c[1] = c;
		c[n_] := Expand[2 c c[n - 1] - c[n - 2]]
		last = 2 c;
		For[i = 3, i <= 9, ++i,
			last = Expand[(c[i] + last)/c];
			Print[last]
		]
	*/

	void initializeRecurrence(double c, __m256d &factor_prev, __m256d &factor_cur) {
		double c2 = c*c,
			  temp1 = 2.0*c,
			  temp2 = -1.0+4.0*c2,
			  temp3 = c*(-4.0+8.0*c2),
			  temp4 = 1.0+c2*(-12.0+16.0*c2);

		factor_prev = _mm256_set_pd(-temp3, -temp2, -temp1, -1.0f);
		factor_cur  = _mm256_set_pd( temp4,  temp3,  temp2, temp1);
	}

	inline void hadd(__m256d a, __m256d b, __m256d c, __m256d d, double *target) {
		/* See http://stackoverflow.com/questions/10833234/4-horizontal-double-precision-sums-in-one-go-with-avx */
		__m256d sumab = _mm256_hadd_pd(a, b);
		__m256d sumcd = _mm256_hadd_pd(c, d);
		__m256d blend = _mm256_blend_pd(sumab, sumcd, 0x0C);
		__m256d perm = _mm256_permute2f128_pd(sumab, sumcd, 0x21);
		__m256d result =  _mm256_add_pd(perm, blend);
		_mm256_store_pd(target, result);
	}

	inline void hadd(__m256d a, __m256d b, double *target) {
		__m256d sum = _mm256_hadd_pd(a, b);
		__m128d sum_high = _mm256_extractf128_pd(sum, 1);
		__m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
		_mm_store_pd(target, result);
	}

	inline double hadd(__m256d x) {
		double MM_ALIGN16 result[2];
		hadd(x, x, result);
		return result[0];
	}
}

#endif

Float evalFourier(const float *coeffs, int nCoeffs, Float phi) {
	#if MTS_FOURIER_SCALAR == 1
		double cosPhi      = std::cos((double) phi),
		       cosPhi_prev = cosPhi,
		       cosPhi_cur  = 1.0,
		       value       = 0.0;

		for (int i=0; i<nCoeffs; ++i) {
			value += coeffs[i] * cosPhi_cur;

			double cosPhi_next = 2.0*cosPhi*cosPhi_cur - cosPhi_prev;
			cosPhi_prev = cosPhi_cur; cosPhi_cur = cosPhi_next;
		}

		return (Float) value;
	#else
		double cosPhi = std::cos((double) phi);

		__m256d
			cosPhi_prev = _mm256_set1_pd(cosPhi),
			cosPhi_cur  = _mm256_set1_pd(1.0),
			value       = _mm256_set_sd((double) coeffs[0]),
			factorPhi_prev, factorPhi_cur;

		simd::initializeRecurrence(cosPhi, factorPhi_prev, factorPhi_cur);

		for (int i=1; i<nCoeffs; i+=4) {
			__m256d coeff = _mm256_cvtps_pd(_mm_load_ps(coeffs+i));

			__m256d cosPhi_next = _mm256_add_pd(_mm256_mul_pd(factorPhi_prev, cosPhi_prev),
					_mm256_mul_pd(factorPhi_cur,  cosPhi_cur));
			value = _mm256_add_pd(value, _mm256_mul_pd(cosPhi_next, coeff));
			cosPhi_prev = _mm256_splat2_pd(cosPhi_next);
			cosPhi_cur = _mm256_splat3_pd(cosPhi_next);
		}

		return (Float) simd::hadd(value);
	#endif
}

Spectrum evalFourier3(float * const coeffs[3], int nCoeffs, Float phi) {
	#if MTS_FOURIER_SCALAR == 1
		Float cosPhi      = std::cos(phi),
			  cosPhi_prev = cosPhi,
			  cosPhi_cur  = 1.0f;

		Float Y = 0, R = 0, B = 0;

		for (int i=0; i<nCoeffs; ++i) {
			Y += coeffs[0][i] * cosPhi_cur;
			R += coeffs[1][i] * cosPhi_cur;
			B += coeffs[2][i] * cosPhi_cur;

			Float cosPhi_next = 2*cosPhi*cosPhi_cur - cosPhi_prev;
			cosPhi_prev = cosPhi_cur; cosPhi_cur = cosPhi_next;
		}

		Float G = 1.39829f*Y -0.100913f*B - 0.297375f*R;

		Spectrum result;
		result.fromLinearRGB(R, G, B);

		return result;
	#else
		double cosPhi = std::cos((double) phi);

		__m256d
			cosPhi_prev = _mm256_set1_pd(cosPhi),
			cosPhi_cur  = _mm256_set1_pd(1.0),
			Y           = _mm256_set_sd((double) coeffs[0][0]),
			R           = _mm256_set_sd((double) coeffs[1][0]),
			B           = _mm256_set_sd((double) coeffs[2][0]),
			factorPhi_prev, factorPhi_cur;

		simd::initializeRecurrence(cosPhi, factorPhi_prev, factorPhi_cur);

		for (int i=1; i<nCoeffs; i+=4) {
			__m256d cosPhi_next = _mm256_add_pd(_mm256_mul_pd(factorPhi_prev, cosPhi_prev),
					_mm256_mul_pd(factorPhi_cur,  cosPhi_cur));

			Y = _mm256_add_pd(Y, _mm256_mul_pd(cosPhi_next, _mm256_cvtps_pd(_mm_load_ps(coeffs[0]+i))));
			R = _mm256_add_pd(R, _mm256_mul_pd(cosPhi_next, _mm256_cvtps_pd(_mm_load_ps(coeffs[1]+i))));
			B = _mm256_add_pd(B, _mm256_mul_pd(cosPhi_next, _mm256_cvtps_pd(_mm_load_ps(coeffs[2]+i))));

			cosPhi_prev = _mm256_splat2_pd(cosPhi_next);
			cosPhi_cur = _mm256_splat3_pd(cosPhi_next);
		}

		MM_ALIGN32 struct {
			double Y;
			double R;
			double B;
			double unused;
		} tmp;

		simd::hadd(Y, R, B, _mm256_setzero_pd(), (double *) &tmp);

		Float G = 1.39829f*tmp.Y -0.100913f*tmp.B - 0.297375f*tmp.R;

		Spectrum result;
		result.fromLinearRGB(tmp.R, G, tmp.B);
		return result;
	#endif
}

Float sampleFourier(const float *coeffs, const double *recip, int nCoeffs,
		Float sample, Float &pdf, Float &phi) {

	bool flip = false;
	if (sample < 0.5f) {
		sample *= 2.0f;
	} else {
		sample = 1.0f - 2.0f * (sample - 0.5f);
		flip = true;
	}

	int iterations = 0;

	double a = 0.0,
		   c = M_PI_DBL,
	       coeff0 = coeffs[0],
		   y = coeff0*M_PI_DBL*sample,
		   deriv = 0.0,
		   b = 0.5 * M_PI_DBL,
		   sinB = 1,
		   cosB = 0;

	if (nCoeffs > 10 && sample != 0 && sample != 1) {
		float stddev = std::sqrt(2.0f/3.0f * math::fastlog(coeffs[1]/coeffs[2]));
		if (std::isfinite(stddev)) {
			b = std::min(c, (double) warp::intervalToStdNormal(0.5f + sample/2) * stddev);
			cosB = std::cos(b);
			sinB = std::sqrt(1-cosB*cosB);
		}
	}

	while (true) {
		#if MTS_FOURIER_SCALAR == 1
			double cosB_prev = cosB,
			       sinB_prev = -sinB,
			       sinB_cur  = 0.0,
			       cosB_cur  = 1.0,
			       value     = coeff0 * b;

			deriv = coeff0;

			for (int j=1; j<nCoeffs; ++j) {
				double sinB_next = 2.0*cosB*sinB_cur - sinB_prev,
				       cosB_next = 2.0*cosB*cosB_cur - cosB_prev,
				       coeff     = (double) coeffs[j];

				value += coeff * recip[j] * sinB_next;
				deriv += coeff * cosB_next;

				sinB_prev = sinB_cur; sinB_cur = sinB_next;
				cosB_prev = cosB_cur; cosB_cur = cosB_next;
			}
		#else
			__m256d factorB_prev, factorB_cur;
			simd::initializeRecurrence(cosB, factorB_prev, factorB_cur);

			__m256d
				sinB_prev  = _mm256_set1_pd(-sinB),
				sinB_cur   = _mm256_set1_pd(0.0),
				cosB_prev  = _mm256_set1_pd(cosB),
				cosB_cur   = _mm256_set1_pd(1.0),
				value_vec  = _mm256_set_sd(coeff0 * b),
				deriv_vec  = _mm256_set_sd(coeff0);

			for (int j=1; j<nCoeffs; j+=4) {
				__m128 coeff_vec_f = _mm_load_ps(coeffs+j);
				__m256d recip_vec  = _mm256_load_pd(recip+j);
				__m256d coeff_vec  = _mm256_cvtps_pd(coeff_vec_f);

				__m256d sinB_next = _mm256_add_pd(
						_mm256_mul_pd(factorB_prev, sinB_prev),
						_mm256_mul_pd(factorB_cur, sinB_cur));

				__m256d cosB_next = _mm256_add_pd(
						_mm256_mul_pd(factorB_prev, cosB_prev),
						_mm256_mul_pd(factorB_cur, cosB_cur));

				value_vec = _mm256_add_pd(value_vec, _mm256_mul_pd(
					_mm256_mul_pd(recip_vec, coeff_vec), sinB_next));
				deriv_vec = _mm256_add_pd(deriv_vec, _mm256_mul_pd(coeff_vec, cosB_next));

				sinB_prev = _mm256_splat2_pd(sinB_next);
				cosB_prev = _mm256_splat2_pd(cosB_next);
				sinB_cur  = _mm256_splat3_pd(sinB_next);
				cosB_cur  = _mm256_splat3_pd(cosB_next);
			}

			double value = simd::hadd(value_vec);
			deriv = simd::hadd(deriv_vec);
		#endif

		value -= y;

		if (std::abs(value) <= 1e-5 * coeff0 || ++iterations > 20)
			break;
		else if (value > 0.0)
			c = b;
		else
			a = b;

		b -= value / deriv;

		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		cosB = std::cos(b);
		sinB = std::sqrt(1-cosB*cosB);
	}

	if (flip)
		b = 2.0*M_PI_DBL - b;

	pdf = (Float) (INV_TWOPI_DBL * deriv / coeff0);
	phi = (Float) b;
	return (Float) (coeff0*(2*M_PI_DBL));
}

Spectrum sampleFourier3(float * const coeffs[3], const double *recip, int nCoeffs,
		Float sample, Float &pdf, Float &phi) {
	bool flip = false;
	if (sample < 0.5f) {
		sample *= 2.0f;
	} else {
		sample = 1.0f - 2.0f * (sample - 0.5f);
		flip = true;
	}

	int iterations = 0;

	double a = 0.0,
		   c = M_PI_DBL,
	       coeff0 = coeffs[0][0],
		   y = coeff0*M_PI_DBL*sample,
		   deriv = 0.0,
		   b = 0.5 * M_PI_DBL,
		   cosB = 0,
		   sinB = 1;

	if (nCoeffs > 10 && sample != 0 && sample != 1) {
		float stddev = std::sqrt(2.0f/3.0f * math::fastlog(coeffs[0][1]/coeffs[0][2]));
		if (std::isfinite(stddev)) {
			b = std::min(c, (double) warp::intervalToStdNormal(0.5f + sample/2) * stddev);
			cosB = std::cos(b);
			sinB = std::sqrt(1-cosB*cosB);
		}
	}

	#if MTS_FOURIER_SCALAR != 1
		__m256d factorB_prev, factorB_cur;
	#endif

	while (true) {
		#if MTS_FOURIER_SCALAR == 1
			double cosB_prev = cosB,
			       sinB_prev = -sinB,
			       sinB_cur  = 0.0,
			       cosB_cur  = 1.0,
			       value     = coeff0 * b;

			deriv = coeff0;

			for (int j=1; j<nCoeffs; ++j) {
				double sinB_next = 2.0*cosB*sinB_cur - sinB_prev,
				       cosB_next = 2.0*cosB*cosB_cur - cosB_prev,
				       coeff     = (double) coeffs[0][j];

				value += coeff * recip[j] * sinB_next;
				deriv += coeff * cosB_next;

				sinB_prev = sinB_cur; sinB_cur = sinB_next;
				cosB_prev = cosB_cur; cosB_cur = cosB_next;
			}
		#else
			simd::initializeRecurrence(cosB, factorB_prev, factorB_cur);

			__m256d
				sinB_prev  = _mm256_set1_pd(-sinB),
				sinB_cur   = _mm256_set1_pd(0.0),
				cosB_prev  = _mm256_set1_pd(cosB),
				cosB_cur   = _mm256_set1_pd(1.0),
				value_vec  = _mm256_set_sd(coeff0 * b),
				deriv_vec  = _mm256_set_sd(coeff0);

			for (int j=1; j<nCoeffs; j+=4) {
				__m128 coeff_vec_f = _mm_load_ps(coeffs[0]+j);
				__m256d recip_vec  = _mm256_load_pd(recip+j);
				__m256d coeff_vec  = _mm256_cvtps_pd(coeff_vec_f);

				__m256d sinB_next = _mm256_add_pd(
						_mm256_mul_pd(factorB_prev, sinB_prev),
						_mm256_mul_pd(factorB_cur, sinB_cur));

				__m256d cosB_next = _mm256_add_pd(
						_mm256_mul_pd(factorB_prev, cosB_prev),
						_mm256_mul_pd(factorB_cur, cosB_cur));

				value_vec = _mm256_add_pd(value_vec, _mm256_mul_pd(
					_mm256_mul_pd(recip_vec, coeff_vec), sinB_next));
				deriv_vec = _mm256_add_pd(deriv_vec, _mm256_mul_pd(coeff_vec, cosB_next));

				sinB_prev = _mm256_splat2_pd(sinB_next);
				cosB_prev = _mm256_splat2_pd(cosB_next);
				sinB_cur  = _mm256_splat3_pd(sinB_next);
				cosB_cur  = _mm256_splat3_pd(cosB_next);
			}

			double value = simd::hadd(value_vec);
			deriv = simd::hadd(deriv_vec);
		#endif

		value -= y;

		if (std::abs(value) <= 1e-5 * coeff0 || ++iterations > 20)
			break;
		else if (value > 0.0)
			c = b;
		else
			a = b;

		b -= value / deriv;

		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		cosB = std::cos(b);
		sinB = std::sqrt(1-cosB*cosB);
	}

	double Y = deriv;
	if (flip)
		b = 2.0*M_PI_DBL - b;

	pdf = (Float) (INV_TWOPI_DBL * Y / coeff0);
	phi = (Float) b;

	#if MTS_FOURIER_SCALAR == 1
		double cosB_prev = cosB,
			   cosB_cur  = 1.0;

		double R = coeffs[1][0];
		double B = coeffs[2][0];

		for (int j=1; j<nCoeffs; ++j) {
			double cosB_next = 2.0*cosB*cosB_cur - cosB_prev,
				   coeffR    = (double) coeffs[1][j],
				   coeffB    = (double) coeffs[2][j];

			R += coeffR * cosB_next;
			B += coeffB * cosB_next;

			cosB_prev = cosB_cur; cosB_cur = cosB_next;
		}
	#else
		__m256d
			cosB_prev  = _mm256_set1_pd(cosB),
			cosB_cur   = _mm256_set1_pd(1.0),
			R_vec  = _mm256_set_sd(coeffs[1][0]),
			B_vec  = _mm256_set_sd(coeffs[2][0]);

		for (int j=1; j<nCoeffs; j+=4) {
			__m128 coeff_R_vec_f = _mm_load_ps(coeffs[1]+j);
			__m128 coeff_B_vec_f = _mm_load_ps(coeffs[2]+j);
			__m256d coeff_R_vec  = _mm256_cvtps_pd(coeff_R_vec_f);
			__m256d coeff_B_vec  = _mm256_cvtps_pd(coeff_B_vec_f);

			__m256d cosB_next = _mm256_add_pd(
					_mm256_mul_pd(factorB_prev, cosB_prev),
					_mm256_mul_pd(factorB_cur, cosB_cur));

			R_vec = _mm256_add_pd(R_vec, _mm256_mul_pd(coeff_R_vec, cosB_next));
			B_vec = _mm256_add_pd(B_vec, _mm256_mul_pd(coeff_B_vec, cosB_next));

			cosB_prev = _mm256_splat2_pd(cosB_next);
			cosB_cur  = _mm256_splat3_pd(cosB_next);
		}

		double R = simd::hadd(R_vec);
		double B = simd::hadd(B_vec);
	#endif

	Float G = 1.39829*Y -0.100913*B - 0.297375*R;

	Spectrum result;
	result.fromLinearRGB((Float) R, (Float) G, (Float) B);

	return result * (2*M_PI)*(Float) (coeff0 / Y);
}

namespace {
	/// Filon integration over a single spline segment (used by filonIntegrate)
	inline void filon(Float phi[2], Float f[3], int lmax, Float *output) {
		Float h = phi[1] - phi[0], invH = 1/h;

		output[0] += (INV_PI / 6.0) * h * (f[0]+4*f[1]+f[2]);

		Float cosPhi0Prev =  std::cos(phi[0]), cosPhi0Cur = 1.0,
		      cosPhi1Prev =  std::cos(phi[1]), cosPhi1Cur = 1.0,
		      sinPhi0Prev = -std::sin(phi[0]), sinPhi0Cur = 0.0,
		      sinPhi1Prev = -std::sin(phi[1]), sinPhi1Cur = 0.0,
		      twoCosPhi0  =  2.0f * cosPhi0Prev,
		      twoCosPhi1  =  2.0f * cosPhi1Prev;

		const Float term0 = 3*f[0]-4*f[1]+f[2],
		            term1 = f[0]-4*f[1]+3*f[2],
		            term2 = 4*(f[0]-2*f[1]+f[2]);

		for (int l=1; l<lmax; ++l) {
			Float cosPhi0Next = twoCosPhi0*cosPhi0Cur - cosPhi0Prev,
			      cosPhi1Next = twoCosPhi1*cosPhi1Cur - cosPhi1Prev,
			      sinPhi0Next = twoCosPhi0*sinPhi0Cur - sinPhi0Prev,
			      sinPhi1Next = twoCosPhi1*sinPhi1Cur - sinPhi1Prev;

			Float invL    = 1/(Float) l,
			      invL2H  = invH*invL*invL,
			      invL3H2 = invL2H*invL*invH;

			output[l] += (2 * INV_PI) *
			   ((invL2H * (term0 * cosPhi0Next + term1 * cosPhi1Next) +
			     invL3H2 * term2 * (sinPhi0Next - sinPhi1Next) +
			     invL * (f[2] * sinPhi1Next - f[0] * sinPhi0Next)));

			cosPhi0Prev = cosPhi0Cur; cosPhi0Cur = cosPhi0Next;
			cosPhi1Prev = cosPhi1Cur; cosPhi1Cur = cosPhi1Next;
			sinPhi0Prev = sinPhi0Cur; sinPhi0Cur = sinPhi0Next;
			sinPhi1Prev = sinPhi1Cur; sinPhi1Cur = sinPhi1Next;
		}
	}
};

void filonIntegrate(const boost::function<Float (Float)> &eval,
		Float *coeffs, int nCoeffs, int nEvals, Float a, Float b) {

	/* Avoid numerical overflow issues for extremely small intervals */
	#if defined(SINGLE_PRECISION)
		if (std::abs(b-a) < 1e-6)
			return;
	#else
		if (std::abs(b-a) < 1e-15)
			return;
	#endif

	if (nEvals % 2 == 0)
		++nEvals;

	Float value[3], phi[2], delta = (b-a) / (nEvals - 1);
	phi[0] = a; value[0] = eval(a);
	for (int i=0; i<(nEvals-1)/2; ++i) {
		phi[1]   = phi[0] + 2*delta;
		value[1] = eval(phi[0] + delta);
		value[2] = eval(phi[1]);

		filon(phi, value, nCoeffs, coeffs);

		value[0] = value[2];
		phi[0]   = phi[1];
	}
}

void convolveFourier(const Float *a, int ka, const Float *b, int kb, Float *c) {
	for (int i=0; i<ka+kb-1; ++i) {
		Float sum = 0;

		for (int j=0; j<std::min(kb, ka-i); ++j)
			sum += b[j]*a[i+j];

		for (int j=std::max(0, i-ka+1); j<std::min(kb, i+ka); ++j)
			sum += b[j]*a[std::abs(i-j)];

		if (i < kb)
			sum += b[i]*a[0];

		if (i == 0)
			sum = .5f * (sum + a[0]*b[0]);

		c[i] = .5f * sum;
	}
}

MTS_NAMESPACE_END
