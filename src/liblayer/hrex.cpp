#include <mitsuba/layer/hrex.h>
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

namespace {
	// normalized 1D gaussian with variance var
	float gauss1D(float r, float var) {
		return math::fastexp(-r*r/(2.0f * var)) / std::sqrt(2 * M_PI_FLT * var);
	}

	// elliptic theta function of variance v (normalized over [0,2pi])
	float wrappedGaussian(float phi, float var) {
		if(var < 5.0f) {
			phi = std::fmod(phi, 2*M_PI_FLT);
			return gauss1D(phi - 2*M_PI_FLT, var) + gauss1D(phi, var) + gauss1D(phi + 2 * M_PI_FLT, var);
		} else {
			return (1.0f + 2.0f * std::cos(phi) * math::fastexp(-0.5f * var)) / (2 * M_PI_FLT);
		}
	}
};

void HarmonicExtrapolation::transform(const float *coeff_in, float *coeff_out) {
	if (coeff_in[0] == 0) {
		coeff_out[0] = coeff_out[1] = coeff_out[2] = 0;
		return;
	}

	if (coeff_in[1] <= 0 || coeff_in[2] <= 0 || coeff_in[2] >= coeff_in[1]) {
		/* Unsupported case -- put all energy into the uniform distribution */
		cout << coeff_in[0] << ", " << coeff_in[1] << ", " << coeff_in[2] << " => unsupported!" << endl;
		coeff_out[0] = coeff_in[0];
		coeff_out[1] = 1.0f;
		coeff_out[2] = 0.0f;
		return;
	}

	float cbrt_1   = cbrtf(coeff_in[1]),
	      cbrt_2   = cbrtf(coeff_in[2]),
	      cbrt_1_2 = cbrt_1*cbrt_1,
	      cbrt_2_2 = cbrt_2*cbrt_2,
	      temp     = cbrt_1_2*cbrt_1_2 / cbrt_2,
	      temp2    = cbrt_1_2 / cbrt_2_2;

	float alpha = M_PI_FLT * (2*coeff_in[0] - temp);
	float beta  = M_PI_FLT * temp;

	/* Can't mix negative distributions -- but be tolerant towards roundoff errors */
	if (alpha < 0 && alpha > -1e-3)
		alpha = 0;
	if (beta < 0 && beta > -1e-3)
		beta = 0;

	if (alpha < 0 || beta < 0 || temp2 <= 1) {
		/* Unsupported case -- put all energy into the uniform distribution */
		cout << coeff_in[0] << ", " << coeff_in[1] << ", " << coeff_in[2] << " => "
			<< "alpha=" << alpha << ", beta=" << beta << ", temp2=" << temp2 << ", var=" << math::fastlog(temp2) << endl;
		coeff_out[0] = std::max(alpha + beta, 0.0f);
		coeff_out[1] = 1.0f;
		coeff_out[2] = 0.0f;
		return;
	}

	float var = math::fastlog(temp2);

	coeff_out[0] = (alpha + beta) * INV_TWOPI_FLT;
	coeff_out[1] = alpha / (alpha + beta);
	coeff_out[2] = std::sqrt(var);
}

Float HarmonicExtrapolation::eval(const float *coeff, float phi) {
	float result = coeff[1];

	if (coeff[1] < 1)
		result += 2 * M_PI_FLT * (1-coeff[1]) * wrappedGaussian(phi, coeff[2]*coeff[2]);

	return (Float) coeff[0] * result;
}

Float HarmonicExtrapolation::pdf(const float *coeff, float phi) {
	float result = INV_TWOPI_FLT * coeff[1];

	if (coeff[1] < 1)
		result += (1-coeff[1]) * wrappedGaussian(phi, coeff[2]*coeff[2]);

	return (Float) result;
}

Float HarmonicExtrapolation::sample(const float *coeff, Float &phi, Float sample) {
	if (coeff[0] == 0.0f) {
		phi = 0.0f;
		return 0.0f;
	}

	if (sample < coeff[1]) {
		phi = 2 * M_PI * (sample / coeff[1]) - M_PI;
	} else {
		sample = (sample - coeff[1]) / (1 - coeff[1]);
		sample = std::max(std::min(sample, ONE_MINUS_EPS),
			std::numeric_limits<Float>::epsilon());

		phi = warp::intervalToStdNormal(sample) * coeff[2];
	}

	return (Float) coeff[0];
}

MTS_NAMESPACE_END
