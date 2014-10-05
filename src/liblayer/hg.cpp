/* This file is based on work conducted at Weta Digital */

#include <mitsuba/layer/hg.h>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>

MTS_NAMESPACE_BEGIN

Float hg(Float mu_o, Float mu_i, Float g, Float phi_d) {
	Float cosTheta = mu_i * mu_o + std::cos(phi_d) *
		std::sqrt((1-mu_i*mu_i) * (1-mu_o*mu_o));

	Float temp = 1.0f + g*g - 2.0f * g * cosTheta;

	return INV_FOURPI * (1 - g*g) / (temp * std::sqrt(temp));
}

/**
 * This function first finds the 0-th and 1st-order Fourier coefficients
 * using elliptic integrals (using the implementation in boost::math).
 *
 * The other coefficients can then be determined much more efficiently;
 * the approach here is based on the idea that the ratio of adjacent
 * coefficients eventually converges to a constant value. Using a
 * 2nd-order Taylor expansion, we can obtain a fairly accurate estimate
 * of this ratio somewhere "in the middle" (i.e. for large $n$, but
 * well before the aforementioned convergence).
 *
 * Using a backwards recurrence scheme, we can then determine all
 * previous ratios and, thereby (using the explicitly computed
 * first values), all Fourier coefficients.
 *
 * Based on
 *
 * "A Recurrence Formula For Computing Fourier Components of the
 *  Henyey-Greenstein Phase Function" by E.G. Yanovitskij
 *
 * Journal of Quantitative Spectroscopy & Radiative Transfer, 57, no 1. 1977
 */
void hgFourierSeries(Float mu_o, Float mu_i, Float g, int kmax, Float relerr, std::vector<Float> &result) {
	result.clear();
	if (g == 0) {
		result.push_back(INV_FOURPI);
		return;
	}

	/* Compute A and B coefficients */
	Float a = 1 + g*g - 2*g*mu_i*mu_o;
	Float b = -2 * g * math::safe_sqrt((1-mu_i*mu_i) * (1-mu_o*mu_o));

	/* Find the first two Fourier coefficients using elliptic integrals */
	Float absB = std::abs(b), arg = std::sqrt(2*absB / (a+absB));
	Float K = boost::math::ellint_1(arg), E = boost::math::ellint_2(arg);
	Float sqrtAB = std::sqrt(a+absB), temp = (1-g*g) * (0.5f * INV_PI * INV_PI);

	Float coeff0 = (E * temp * sqrtAB) / (a*a - b*b);
	Float coeff1 = b == 0 ? 0 :
		(math::signum(b) * temp / (absB * sqrtAB) * (K - a / (a-absB) * E));

	int m = std::max(kmax * 2, 500);
	Float *s = (Float *) alloca(sizeof(Float) * (m + 1));

	/* Compute the ratio between the $m$-th and $m+1$-th term
	   using a second-order Taylor expansion */
	Float z     =  a / math::safe_sqrt(a*a - b*b),
	      delta =  z / math::safe_sqrt(z*z-1);
	s[m] = (1 + 1 / (Float) (2*m) - (1+3*z) / (Float) (8*m*m)) *
		   std::sqrt((z-1) / (z+1));

	do {
		/* Work backwards using a recurrence scheme */
		--m;
		s[m] = (2*m+3) / (4*(m+1) * delta - (2*m+1) * s[m+1]);
	} while (m != 0);

	/* Simple to get a bit of extra accuracy here: apply a correction
	   in case s[0] does not quite match the known reference value */
	Float C = 0.0f;
	if (s[0] != 0)
		C = coeff1 / (coeff0 * s[0]);

	/* Now, multiply all ratios together to get the desired value */
	result.push_back(coeff0);

	Float prod = coeff0 * C * 2;
	for (int j=0; j<kmax-1; ++j) {
		if (prod == 0 || std::abs(prod) < coeff0 * relerr)
			break;
		prod *= s[j];
		if (j % 2 == 0)
			result.push_back(prod);
		else
			result.push_back(prod * math::signum(g));
	}
}

MTS_NAMESPACE_END
