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

#include <mitsuba/layer/microfacet.h>
#include <mitsuba/layer/fourier.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/layer/common.h>
#include <boost/bind.hpp>
#include <Eigen/SVD>

#if defined(MTS_HAS_LAPACK)
# if defined(__OSX__)
#  include <clapack.h>
#  define LAPACK(name) name ##_
# else
#  include <lapacke.h>
#  define LAPACK(name) LAPACKE_ ##name
#  define MTS_HAS_LAPACKE
# endif
#endif

MTS_NAMESPACE_BEGIN

/* Copied from the Cephes math library: I0E -- Exponentially scaled
   modified Bessel function of the first kind, zeroeth order. */
namespace cephes {
	/* Chebyshev coefficients for exp(-x) I0(x)
	 * in the interval [0,8].
	 *
	 * lim(x->0){ exp(-x) I0(x) } = 1.
	 */
	static double A[] = {
		-4.41534164647933937950E-18, 3.33079451882223809783E-17,
		-2.43127984654795469359E-16, 1.71539128555513303061E-15,
		-1.16853328779934516808E-14, 7.67618549860493561688E-14,
		-4.85644678311192946090E-13, 2.95505266312963983461E-12,
		-1.72682629144155570723E-11, 9.67580903537323691224E-11,
		-5.18979560163526290666E-10, 2.65982372468238665035E-9,
		-1.30002500998624804212E-8,  6.04699502254191894932E-8,
		-2.67079385394061173391E-7,  1.11738753912010371815E-6,
		-4.41673835845875056359E-6,  1.64484480707288970893E-5,
		-5.75419501008210370398E-5,  1.88502885095841655729E-4,
		-5.76375574538582365885E-4,  1.63947561694133579842E-3,
		-4.32430999505057594430E-3,  1.05464603945949983183E-2,
		-2.37374148058994688156E-2,  4.93052842396707084878E-2,
		-9.49010970480476444210E-2,  1.71620901522208775349E-1,
		-3.04682672343198398683E-1,  6.76795274409476084995E-1
	};

	/* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
	 * in the inverted interval [8,infinity].
	 *
	 * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
	 */
	static double B[] = {
		-7.23318048787475395456E-18, -4.83050448594418207126E-18,
		 4.46562142029675999901E-17,  3.46122286769746109310E-17,
		-2.82762398051658348494E-16, -3.42548561967721913462E-16,
		 1.77256013305652638360E-15,  3.81168066935262242075E-15,
		-9.55484669882830764870E-15, -4.15056934728722208663E-14,
		 1.54008621752140982691E-14,  3.85277838274214270114E-13,
		 7.18012445138366623367E-13, -1.79417853150680611778E-12,
		-1.32158118404477131188E-11, -3.14991652796324136454E-11,
		 1.18891471078464383424E-11,  4.94060238822496958910E-10,
		 3.39623202570838634515E-9,   2.26666899049817806459E-8,
		 2.04891858946906374183E-7,   2.89137052083475648297E-6,
		 6.88975834691682398426E-5,   3.36911647825569408990E-3,
		 8.04490411014108831608E-1
	};

	double chbevl(double x, double *array, int n) {
		double *p = array;
		double b0 = *p++;
		double b1 = 0.0;
		int i = n - 1;
		double b2;

		do {
			b2 = b1;
			b1 = b0;
			b0 = x * b1  -  b2  + *p++;
		} while (--i);

		return 0.5 * (b0-b2);
	}

	double i0e(double x) {
		if (x < 0)
			x = -x;

		if (x <= 8.0) {
			double y = (x/2.0) - 2.0;
			return chbevl(y, A, 30);
		}

		return chbevl(32.0/x - 2.0, B, 25) / sqrt(x);
	}
};

#if defined(MTS_HAS_LAPACK)
/* Helper functions to expose faster SVD implementations when LAPACK is available */
void fast_svd_recursive(DMatrix &A, DVector &S, DMatrix &U, DMatrix &V) {
	int m = A.rows(), n = A.cols(), info = 0;
	int lda = m, ldu = m, ldv = m;
	const char *jobz = "A";

	U.resize(m, m);
	V.resize(n, n);
	S.resize(std::min(m, n));

	#if defined(MTS_HAS_LAPACKE)
		#if defined(SINGLE_PRECISION)
			info = LAPACK(sgesdd)(LAPACK_COL_MAJOR, *jobz, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv);
		#else
			info = LAPACK(dgesdd)(LAPACK_COL_MAJOR, *jobz, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv);
		#endif
	#else
		Float work_size = 0;
		int *iwork = NULL;
		Float *work = NULL;
		int lwork = -1;

		/* Run a workspace query */
		#if defined(SINGLE_PRECISION)
			LAPACK(sgesdd)((char *) jobz, &m, &n, NULL, &lda, NULL, NULL, &ldu, NULL, &ldv,
				&work_size, &lwork, NULL, &info);
		#else
			LAPACK(dgesdd)((char *) jobz, &m, &n, NULL, &lda, NULL, NULL, &ldu, NULL, &ldv,
				&work_size, &lwork, NULL, &info);
		#endif

		if (info == 0) {
			lwork = ((int) work_size) + std::min(m, n)*64;
			work = new Float[lwork];
			iwork = new int[8*std::min(m, n)];

			#if defined(SINGLE_PRECISION)
				LAPACK(sgesdd)((char *) jobz, &m, &n, A.data(), &lda, S.data(), U.data(),
					&ldu, V.data(), &ldv, work, &lwork, iwork, &info);
			#else
				LAPACK(dgesdd)((char *) jobz, &m, &n, A.data(), &lda, S.data(), U.data(),
					&ldu, V.data(), &ldv, work, &lwork, iwork, &info);
			#endif

			delete[] iwork;
			delete[] work;
		} else {
			SLog(EError, "[S/D]GESDD: returned info=%i during workspace query!", info);
		}
	#endif

	if (info != 0)
		SLog(EWarn, "[S/D]GESDD: returned info=%i!", info);

	V.transposeInPlace();
}


void fast_svd(DMatrix &A, DVector &S, DMatrix &U, DMatrix &V) {
	int m = A.rows(), n = A.cols(), info = 0;
	int lda = m, ldu = m, ldv = m;
	const char *jobu = "A", *jobv = "A";

	U.resize(m, m);
	V.resize(n, n);
	S.resize(std::min(m, n));

	#if defined(MTS_HAS_LAPACKE)
		Float *superb = new Float[std::min(m, n)];
		#if defined(SINGLE_PRECISION)
			info = LAPACK(sgesvd)(LAPACK_COL_MAJOR, *jobu, *jobv, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv, superb);
		#else
			info = LAPACK(dgesvd)(LAPACK_COL_MAJOR, *jobu, *jobv, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv, superb);
		#endif
		delete[] superb;
	#else
		Float work_size = 0;
		Float *work = NULL;
		int lwork = -1;

		/* Run a workspace query */
		#if defined(SINGLE_PRECISION)
			LAPACK(sgesvd)((char *) jobu, (char *) jobv, &m, &n, NULL, &lda, NULL, NULL, &ldu, NULL, &ldv,
				&work_size, &lwork, &info);
		#else
			LAPACK(dgesvd)((char *) jobu, (char *) jobv, &m, &n, NULL, &lda, NULL, NULL, &ldu, NULL, &ldv,
				&work_size, &lwork, &info);
		#endif

		if (info == 0) {
			lwork = ((int) work_size) + std::min(m, n)*64;
			work = new Float[lwork];

			#if defined(SINGLE_PRECISION)
				LAPACK(sgesvd)((char *) jobu, (char *) jobv, &m, &n, A.data(), &lda, S.data(), U.data(),
					&ldu, V.data(), &ldv, work, &lwork, &info);
			#else
				LAPACK(dgesvd)((char *) jobv, (char *) jobv, &m, &n, A.data(), &lda, S.data(), U.data(),
					&ldu, V.data(), &ldv, work, &lwork, &info);

			#endif

			delete[] work;
		} else {
			SLog(EError, "[S/D]GESVD: returned info=%i during workspace query!", info);
		}
	#endif

	if (info != 0)
		SLog(EWarn, "[S/D]GESVD: returned info=%i!", info);

	V.transposeInPlace();
}
#endif

static int expcosCoefficientCount(Float B, Float relerr) {
	Float prod = 1, invB = 1.0f / B;
	if (B == 0)
		return 1;

	for (int i=0; ; ++i) {
		prod /= 1 + i * invB;

		if (prod < relerr)
			return i+1;
	}
}

static Float modBesselRatio(Float B, Float k) {
	const Float eps = std::numeric_limits<Float>::epsilon(),
	            invTwoB = 2.0f / B;

	Float i = (Float) k,
	       D = 1 / (invTwoB * i++),
	       Cd = D, C = Cd;

	while (std::abs(Cd) > eps * std::abs(C)) {
		Float coeff = invTwoB * i++;
		D = 1 / (D + coeff);
		Cd *= coeff*D - 1;
		C += Cd;
	}

	return C;
}

void expcosFourierSeries(Float A, Float B, Float relerr, std::vector<Float> &coeffs) {
	/* Determine the required number of coefficients and allocate memory */
	int n = expcosCoefficientCount(B, relerr);
	coeffs.resize(n);

	/* Determine the last ratio and work downwards */
	coeffs[n-1] = modBesselRatio(B, n - 1);
	for (int i=n-2; i>0; --i)
		coeffs[i] = B / (2*i + B*coeffs[i+1]);

	/* Evaluate the exponentially scaled I0 and correct scaling */
	coeffs[0] = (Float) cephes::i0e((double) B) * math::fastexp(A+B);

	/* Apply the ratios & factor of two upwards */
	Float prod = 2*coeffs[0];
	for (int i=1; i<n; ++i) {
		prod *= coeffs[i];
		if (std::abs(prod) < coeffs[0] * relerr) {
			coeffs.erase(coeffs.begin() + i, coeffs.end());
			break;
		}
		coeffs[i] = prod;
	}
}

/// Smith's 1D shadowing masking term for the Beckmann microfacet distribution
Float smithG1(const Vector &v, const Vector &m, Float alpha) {
	const Float tanTheta = std::abs(Frame::tanTheta(v));

	/* Can't see the back side from the front and vice versa */
	if (dot(v, m) * Frame::cosTheta(v) <= 0)
		return 0.0f;

	Float a = 1.0f / (alpha * tanTheta);
	if (a < 1.6f) {
		/* Use a fast and accurate (<0.35% rel. error) rational
		   approximation to the shadowing-masking function */
		const Float aSqr = a * a;
		return (3.535f * a + 2.181f * aSqr)
			 / (1.0f + 2.276f * a + 2.577f * aSqr);
	}

	return 1.0f;
}

/**
 * \brief Implements Bruce Walter's microfacet model specialized to the
 * Beckmann distribution. The exponential term is intentionally excluded.
 */
Float microfacetNoExp(Float mu_o, Float mu_i, Float _eta, Float k, Float alpha, Float phi_d) {
	Float sinThetaI = std::sqrt(1-mu_i*mu_i),
	      sinThetaO = std::sqrt(1-mu_o*mu_o),
		  cosPhi = std::cos(phi_d),
		  sinPhi = std::sin(phi_d);

	Vector wi(-sinThetaI, 0, -mu_i);
	Vector wo(sinThetaO*cosPhi, sinThetaO*sinPhi, mu_o);
	bool reflect = -mu_i*mu_o > 0;

	if (k != 0 && !reflect)
		return 0.0f;

	Float eta = (-mu_i > 0 || k != 0) ? _eta : 1/_eta;

	Vector H = normalize(wi + wo * (reflect ? 1.0f : eta));
	H *= math::signum(Frame::cosTheta(H));

	Float cosThetaH2 = Frame::cosTheta2(H),
	      D = (Float) 1.0f / (M_PI * alpha*alpha * cosThetaH2*cosThetaH2),
	      F = (k == 0) ? fresnelDielectricExt(dot(wi, H), _eta)
	                   : fresnelConductorExact(std::abs(dot(wi, H)), eta, k),
	      G = smithG1(wi, H, alpha) * smithG1(wo, H, alpha);

	if (reflect) {
		return F * D * G / (4.0f * std::abs(mu_i*mu_o));
	} else {
		Float sqrtDenom = dot(wi, H) + eta * dot(wo, H);

		return std::abs(((1 - F) * D * G * eta * eta * dot(wi, H)
			* dot(wo, H)) / (mu_i*mu_o * sqrtDenom * sqrtDenom));
	}
}

/**
 * \brief Implements Bruce Walter's microfacet model specialized to the
 * Beckmann distribution. The exponential term is included.
 */
Float microfacet(Float mu_o, Float mu_i, Float _eta, Float k, Float alpha, Float phi_d) {
	Float sinThetaI = std::sqrt(1-mu_i*mu_i),
	      sinThetaO = std::sqrt(1-mu_o*mu_o),
		  cosPhi = std::cos(phi_d),
		  sinPhi = std::sin(phi_d);

	Vector wi(-sinThetaI, 0, -mu_i);
	Vector wo(sinThetaO*cosPhi, sinThetaO*sinPhi, mu_o);
	bool reflect = -mu_i*mu_o > 0;

	if (k != 0 && !reflect)
		return 0.0f;

	Float eta = (-mu_i > 0 || k != 0) ? _eta : 1/_eta;

	Vector H = normalize(wi + wo * (reflect ? 1.0f : eta));
	H *= math::signum(Frame::cosTheta(H));

	Float cosThetaH2 = Frame::cosTheta2(H),
	      exponent = -Frame::tanTheta2(H) / (alpha*alpha),
	      D = math::fastexp(exponent) / (M_PI * alpha*alpha * cosThetaH2*cosThetaH2),
	      F = (k == 0) ? fresnelDielectricExt(dot(wi, H), _eta)
	                   : fresnelConductorExact(std::abs(dot(wi, H)), eta, k),
	      G = smithG1(wi, H, alpha) * smithG1(wo, H, alpha);

	if (reflect) {
		return F * D * G / (4.0f * std::abs(mu_i*mu_o));
	} else {
		Float sqrtDenom = dot(wi, H) + eta * dot(wo, H);

		return std::abs(((1 - F) * D * G * eta * eta * dot(wi, H)
			* dot(wo, H)) / (mu_i*mu_o * sqrtDenom * sqrtDenom));
	}
}

static Float Bmax(int kmax, Float relerr) {
	if (relerr == 1e-6f)
		return 0.0299f*std::pow((Float) kmax, (Float) 2.02628);
	else if (relerr ==  1e-5f)
		return 0.0337f*std::pow((Float) kmax, (Float) 2.03865);
	else if (relerr ==  1e-4f)
		return 0.0406f*std::pow((Float) kmax, (Float) 2.04686);
	else if (relerr ==  1e-3f)
		return 0.0538f*std::pow((Float) kmax, (Float) 2.05001);
	else if (relerr ==  1e-2f)
		return 0.0818f*std::pow((Float) kmax, (Float) 2.04982);
	else if (relerr ==  1e-1f)
		return 0.1662f*std::pow((Float) kmax, (Float) 2.05039);
	else {
		SLog(EError, "Bmax(): unknown relative error bound!");
		return 0.0f;
	}
}

static void lowfreqFourierSeries(Float mu_o, Float mu_i, Float _eta, Float k,
		Float alpha, Float relerr, int kmax, std::vector<Float> &result) {
	const int nEvals = 100;

	bool reflect = -mu_i*mu_o > 0;
	bool conductor = (k != 0.0f);

	Float sinMu2 = math::safe_sqrt((1-mu_i*mu_i)*(1-mu_o*mu_o));
	Float eta = (-mu_i > 0 || conductor) ? _eta : 1/_eta;
	Float phiCritical = 0.0f, B = 0;
	int n = (int) result.size();

	if (reflect) {
		if (!conductor)
			phiCritical = math::safe_acos((2*eta*eta-mu_i*mu_o-1)/sinMu2);
		Float temp = 1.0f / (alpha*(mu_i-mu_o));
		B = 2*sinMu2*temp*temp;
	} else if (!reflect) {
		if (conductor)
			SLog(EError, "lowfreqFourierSeries(): internal error: encountered refraction case for a conductor");
		Float etaDenser = (_eta>1 ? _eta : 1/_eta);
		phiCritical = math::safe_acos((1-etaDenser*mu_i*mu_o) / (etaDenser*sinMu2));
		Float temp = 1.0f / (alpha*(mu_i-eta*mu_o));
		B = 2*eta*sinMu2*temp*temp;
	}

	Float B_max = Bmax(kmax, relerr);
	if (B > B_max)
		B = B_max;

	/* Only fit in the region where the result is actually
	   going to make some sort of difference */
	Float phiMax = math::safe_acos(1 + math::fastlog(relerr) / B);

	if (!conductor && phiCritical > Epsilon && phiCritical < M_PI-Epsilon && phiCritical < phiMax-Epsilon) {
		/* Rarely, some high frequency content leaks into the overall low frequency
		   portion. Increase the number of coefficients so that we can capture it. */
		n = 100;
		result.resize(n);
	}

	DVector coeffs(n);
	coeffs.setZero();
	boost::function<Float (Float)> integrand = boost::bind(&microfacetNoExp, mu_o, mu_i, _eta, k, alpha, _1);

	if (reflect) {
		if (phiCritical > Epsilon && phiCritical < phiMax-Epsilon) {
			filonIntegrate(integrand, (Float *) coeffs.data(), n, nEvals/2, 0, phiCritical);
			filonIntegrate(integrand, (Float *) coeffs.data(), n, nEvals/2, phiCritical, phiMax);
		} else {
			filonIntegrate(integrand, (Float *) coeffs.data(), n, nEvals, 0, phiMax);
		}
	} else {
		filonIntegrate(integrand, (Float *) coeffs.data(), n, nEvals, 0, std::min(phiCritical, phiMax));
	}

	coeffs[0] *= M_PI;
	for (int i=1; i<n; ++i)
		coeffs[i] *= 0.5 * M_PI;

	if (phiMax < M_PI - Epsilon) {
		/* Precompute some sines and cosines */
		DVector cosPhi(n), sinPhi(n);
		for (int i=0; i<n; ++i) {
			sinPhi[i] = std::sin(i*phiMax);
			cosPhi[i] = std::cos(i*phiMax);
		}

		/* The fit only occurs on a subset [0, phiMax], where the Fourier
		   Fourier basis functions are not orthogonal anymore. The following
		   then does a change of basis to proper Fourier coefficients. */
		DMatrix A(n, n);

		for (int i=0; i<n; ++i) {
			for (int j=0; j<=i; ++j) {
				if (i != j) {
					A(i, j) = A(j, i) =
						(i*cosPhi[j]*sinPhi[i]-j*cosPhi[i]*sinPhi[j]) / (i*i-j*j);
				} else if (i != 0) {
					A(i, i) = (std::sin(2*i*phiMax) + 2*i*phiMax) / (4*i);
				} else {
					A(i, i) = phiMax;
				}
			}
		}

		#if 0 /* Solve using a rank-estimating QR factorization with column pivoting */
			coeffs = A.colPivHouseholderQr().solve(coeffs).eval();
		#else
			#if defined(MTS_HAS_LAPACK)
				DMatrix U, V;
				DVector sigma;
				fast_svd(A, sigma, U, V);
			#else /* Solve using Eigen's Jacobi SVD */
				Eigen::JacobiSVD<DMatrix> svd = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

				const DMatrix &U = svd.matrixU();
				const DMatrix &V = svd.matrixV();
				const DVector &sigma = svd.singularValues();
			#endif

			if (sigma[0] == 0) {
				result.clear();
				result.push_back(0);
				return;
			}

			DVector result = DVector::Zero(n);
			for (int i=0; i<n; ++i) {
				if (sigma[i] < 1e-9f * sigma[0])
					break;
				result += V.col(i) * U.col(i).dot(coeffs) / sigma[i];
			}
			coeffs = result;
		#endif
	} else {
		coeffs[0] *= INV_PI;
		for (int i=1; i<n; ++i)
			coeffs[i] *= 2 * INV_PI;
	}
	for (int i=0; i<n; ++i)
		result[i] = coeffs[i];
}

void microfacetFourierSeries(Float mu_o, Float mu_i, Float _eta, Float k,
		Float alpha, Float relerr, int kmax, std::vector<Float> &result) {
	bool reflect = -mu_i*mu_o > 0, conductor = (k != 0.0f);

	/* Compute the 'A' and 'B' constants, as well as the critical azimuth */
	Float A, B;
	Float sinMu2 = math::safe_sqrt((1-mu_i*mu_i)*(1-mu_o*mu_o));
	Float eta = -mu_i > 0 ? _eta : 1/_eta;

	if (reflect) {
		Float temp = 1.0f / (alpha*(mu_i-mu_o));
		A = (mu_i*mu_i + mu_o*mu_o - 2) * temp * temp;
		B = 2*sinMu2*temp*temp;
	} else {
		if (conductor) {
			/* No refraction in conductors */
			result.clear();
			result.push_back(0.0f);
			return;
		} else {
			Float temp = 1.0f / (alpha*(mu_i-eta*mu_o));
			A = (mu_i*mu_i - 1 + eta*eta*(mu_o*mu_o - 1))*temp*temp;
			B = 2*eta*sinMu2*temp*temp;
		}
	}

	/* Minor optimization: don't even bother computing the Fourier series
	   if the contribution to the scattering model is miniscule */
	if ((Float) cephes::i0e((double) B) * math::fastexp(A+B) < 1e-10) {
		result.clear();
		result.push_back(0.0f);
		return;
	}

	Float B_max = Bmax(kmax, relerr);
	if (B > B_max) {
		A = A + B - B_max + (Float) math::fastlog(
			cephes::i0e((double) B)/cephes::i0e((double) B_max));
		B = B_max;
	}

	std::vector<Float> lowfreq_coeffs(12), expcos_coeffs;

	/* Compute Fourier coefficients of the exponential term */
	expcosFourierSeries(A, B, relerr, expcos_coeffs);

	/* Compute Fourier coefficients of the low-frequency term */
	lowfreqFourierSeries(mu_o, mu_i, _eta, k, alpha, relerr, kmax, lowfreq_coeffs);

	/* Perform discrete circular convolution of the two series */
	result.resize(lowfreq_coeffs.size() + expcos_coeffs.size() - 1);

	convolveFourier(lowfreq_coeffs.data(), lowfreq_coeffs.size(),
		expcos_coeffs.data(), expcos_coeffs.size(), result.data());

	/* Truncate the series if error bounds are satisfied */
	for (size_t i=0; i<result.size(); ++i) {
		SAssert(std::isfinite(result[i]));
		if (result[i] == 0 || std::abs(result[i]) < result[0] * relerr) {
			result.erase(result.begin() + i, result.end());
			break;
		}
	}
}

MTS_NAMESPACE_END
