/* This file is based on work conducted at Weta Digital */

#pragma once
#if !defined(__MITSUBA_LAYER_HG_H_)
#define __MITSUBA_LAYER_HG_H_

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

/**
 * \brief Evaluate the HG model using the mu_i, mu_o,
 * phi_d parameterization
 *
 * \param mu_i
 *    Incident zenith angle cosine
 * \param mu_o
 *    Exitant zenith angle cosine
 * \param phi_d
 *    Azimuthal difference angle
 * \param g
 *    Anisotropy parameter
 */
extern MTS_EXPORT_LAYER Float hg(Float mu_o, Float mu_i,
		Float g, Float phi_d);

/**
 * \brief Compute a Fourier series of the HG model
 *
 * \param mu_i
 *    Incident zenith angle cosine
 * \param mu_o
 *    Exitant zenith angle cosine
 * \param g
 *    Anisotropy parameter
 * \param kmax
 *    Indicates a desired maximum number of Fourier coefficients. The
 *    implementation will blur out higher Frequency content to try to
 *    achieve this number.
 * \param result
 *    Storage for the generated Fourier coefficients
 */
extern MTS_EXPORT_LAYER void hgFourierSeries(Float mu_o,
		Float mu_i, Float g, int kmax, Float relerr, std::vector<Float> &result);

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_HG_H_ */
