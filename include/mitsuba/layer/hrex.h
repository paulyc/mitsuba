#if !defined(__HREX_H)
#define __HREX_H

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

class MTS_EXPORT_LAYER HarmonicExtrapolation {
public:
	/// Transform Fourier series coefficients into the extrapolation coefficients
	static void transform(const float *coeff_in, float *coeff_out);

	/// Evaluate extrapolation coefficients
	static Float eval(const float *coeff, float phi);

	/// Evaluate the density function of \ref sample()
	static Float pdf(const float *coeff, float phi);

	/// Importance sample the extrapolated distribution
	static Float sample(const float *coeff, Float &phi, Float sample);
};

MTS_NAMESPACE_END

#endif /* __HREX_H */
