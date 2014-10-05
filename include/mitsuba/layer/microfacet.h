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
#if !defined(__MITSUBA_LAYER_MICROFACET_H_)
#define __MITSUBA_LAYER_MICROFACET_H_

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

/**
 * \brief Evaluate the Beckmann distribution-based microfacet BSDF by
 * Walter et al. using the mu_i, mu_o, phi_d parameterization
 * (dielectric version)
 *
 * \param mu_i
 *    Incident zenith angle cosine
 * \param mu_o
 *    Exitant zenith angle cosine
 * \param phi_d
 *    Azimuthal difference angle
 * \param eta
 *    Relative index of refraction
 * \param k
 *    Absorption coefficient
 * \param alpha
 *    Beckmann roughness parameter
 */
extern MTS_EXPORT_LAYER Float microfacet(Float mu_o, Float mu_i,
		Float eta, Float k, Float alpha, Float phi_d);

/**
 * \brief Compute a Fourier series of the Beckmann-distribution based
 * microfacet BSDF by Walter et al. (covers both the dielectric and
 * conducting case)
 *
 * \param mu_i
 *    Incident zenith angle cosine
 * \param mu_o
 *    Exitant zenith angle cosine
 * \param eta
 *    Relative index of refraction (real component)
 * \param k
 *    Relative index of refraction (imaginary component)
 * \param alpha
 *    Beckmann roughness parameter
 * \param relerr
 *    A relative error threshold after which series terms can safely
 *    be truncated
 * \param kmax
 *    Indicates a desired maximum number of Fourier coefficients. The
 *    implementation will blur out higher Frequency content to try to
 *    achieve this number.
 * \param result
 *    Storage for the generated Fourier coefficients
 */
extern MTS_EXPORT_LAYER void microfacetFourierSeries(Float mu_o,
		Float mu_i, Float eta, Float k, Float alpha, Float relerr,
		int kmax, std::vector<Float> &result);

extern MTS_EXPORT_LAYER void expcosFourierSeries(Float A, Float B,
	Float relerr, std::vector<Float> &coeffs);

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_MICROFACET_H_ */
