/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

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

#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

namespace warp {

Vector squareToUniformSphere(const Point2 &sample) {
	Float z = 1.0f - 2.0f * sample.y;
	Float r = math::safe_sqrt(1.0f - z*z);
	Float sinPhi, cosPhi;
	math::sincos(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
	return Vector(r * cosPhi, r * sinPhi, z);
}

Vector squareToUniformHemisphere(const Point2 &sample) {
	Float z = sample.x;
	Float tmp = math::safe_sqrt(1.0f - z*z);

	Float sinPhi, cosPhi;
	math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

	return Vector(cosPhi * tmp, sinPhi * tmp, z);
}

Vector squareToCosineHemisphere(const Point2 &sample) {
	Point2 p = squareToUniformDiskConcentric(sample);
	Float z = math::safe_sqrt(1.0f - p.x*p.x - p.y*p.y);

	/* Guard against numerical imprecisions */
	if (EXPECT_NOT_TAKEN(z == 0))
		z = 1e-10f;

	return Vector(p.x, p.y, z);
}

Vector squareToUniformCone(Float cosCutoff, const Point2 &sample) {
	Float cosTheta = (1-sample.x) + sample.x * cosCutoff;
	Float sinTheta = math::safe_sqrt(1.0f - cosTheta * cosTheta);

	Float sinPhi, cosPhi;
	math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

	return Vector(cosPhi * sinTheta,
		sinPhi * sinTheta, cosTheta);
}

Point2 squareToUniformDisk(const Point2 &sample) {
	Float r = std::sqrt(sample.x);
	Float sinPhi, cosPhi;
	math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

	return Point2(
		cosPhi * r,
		sinPhi * r
	);
}

Point2 squareToUniformTriangle(const Point2 &sample) {
	Float a = math::safe_sqrt(1.0f - sample.x);
	return Point2(1 - a, a * sample.y);
}

Point2 squareToUniformDiskConcentric(const Point2 &sample) {
	Float r1 = 2.0f*sample.x - 1.0f;
	Float r2 = 2.0f*sample.y - 1.0f;

	/* Modified concencric map code with less branching (by Dave Cline), see
	   http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
	Float phi, r;
	if (r1 == 0 && r2 == 0) {
		r = phi = 0;
	} else if (r1*r1 > r2*r2) {
		r = r1;
		phi = (M_PI/4.0f) * (r2/r1);
	} else {
		r = r2;
		phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
	}

	Float cosPhi, sinPhi;
	math::sincos(phi, &sinPhi, &cosPhi);

	return Point2(r * cosPhi, r * sinPhi);
}

Point2 uniformDiskToSquareConcentric(const Point2 &p) {
	Float r   = std::sqrt(p.x * p.x + p.y * p.y),
		  phi = std::atan2(p.y, p.x),
		  a, b;

	if (phi < -M_PI/4) {
  		/* in range [-pi/4,7pi/4] */
		phi += 2*M_PI;
	}

	if (phi < M_PI/4) { /* region 1 */
		a = r;
		b = phi * a / (M_PI/4);
	} else if (phi < 3*M_PI/4) { /* region 2 */
		b = r;
		a = -(phi - M_PI/2) * b / (M_PI/4);
	} else if (phi < 5*M_PI/4) { /* region 3 */
		a = -r;
		b = (phi - M_PI) * a / (M_PI/4);
	} else { /* region 4 */
		b = -r;
		a = -(phi - 3*M_PI/2) * b / (M_PI/4);
	}

	return Point2(0.5f * (a+1), 0.5f * (b+1));
}

Point2 squareToStdNormal(const Point2 &sample) {
	Float r   = std::sqrt(-2 * math::fastlog(1-sample.x)),
		  phi = 2 * M_PI * sample.y;
	Point2 result;
	math::sincos(phi, &result.y, &result.x);
	return result * r;
}

Float squareToStdNormalPdf(const Point2 &pos) {
	return INV_TWOPI * math::fastexp(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
}

#if !defined(SQRT_TWOPI)
#define SQRT_TWOPI 2.50662827463100050242
#endif

#if !defined(M_1_SQRTPI)
#define M_1_SQRTPI 0.564189583547756286948
#endif

#if !defined(M_SQRTPI)
#define M_SQRTPI 1.77245385090551602792981
#endif

#if !defined(M_1_SQRT2PI)
#define M_1_SQRT2PI (M_SQRT_2*M_1_SQRTPI)
#endif

/*
 * The standard normal CDF, for one random variable.
 *
 *   Author:  W. J. Cody
 *   URL:   http://www.netlib.org/specfun/erf
 *
 * This is the erfc() routine only, adapted by the
 * transform stdnormal_cdf(u)=(erfc(-u/sqrt(2))/2;
 */
static Float stdnormal_cdf(Float u) {
	const Float a[5] = {
		1.161110663653770e-002,3.951404679838207e-001,2.846603853776254e+001,
		1.887426188426510e+002,3.209377589138469e+003
	};
	const Float b[5] = {
		1.767766952966369e-001,8.344316438579620e+000,1.725514762600375e+002,
		1.813893686502485e+003,8.044716608901563e+003
	};
	const Float c[9] = {
		2.15311535474403846e-8,5.64188496988670089e-1,8.88314979438837594e00,
		6.61191906371416295e01,2.98635138197400131e02,8.81952221241769090e02,
		1.71204761263407058e03,2.05107837782607147e03,1.23033935479799725E03
	};
	const Float d[9] = {
		1.00000000000000000e00,1.57449261107098347e01,1.17693950891312499e02,
		5.37181101862009858e02,1.62138957456669019e03,3.29079923573345963e03,
		4.36261909014324716e03,3.43936767414372164e03,1.23033935480374942e03
	};
	const Float p[6] = {
		1.63153871373020978e-2,3.05326634961232344e-1,3.60344899949804439e-1,
		1.25781726111229246e-1,1.60837851487422766e-2,6.58749161529837803e-4
	};
	const Float q[6] = {
		1.00000000000000000e00,2.56852019228982242e00,1.87295284992346047e00,
		5.27905102951428412e-1,6.05183413124413191e-2,2.33520497626869185e-3
	};
	Float y, z;

	if (!std::isfinite(u))
		return (u < 0 ? 0 : 1);
	y = std::abs(u);

	if (y <= 0.46875*SQRT_TWO) {
		/* evaluate erf() for |u| <= sqrt(2)*0.46875 */
		z = y*y;
		y = u*((((a[0]*z+a[1])*z+a[2])*z+a[3])*z+a[4])
		     /((((b[0]*z+b[1])*z+b[2])*z+b[3])*z+b[4]);
		return (Float) 0.5 + y;
	}

	z = math::fastexp(-y*y/2)/2;
	if (y <= 4.0) {
		/* evaluate erfc() for sqrt(2)*0.46875 <= |u| <= sqrt(2)*4.0 */
		y = y/SQRT_TWO;
		y = ((((((((c[0]*y+c[1])*y+c[2])*y+c[3])*y+c[4])*y+c[5])*y+c[6])*y+c[7])*y+c[8])
		   /((((((((d[0]*y+d[1])*y+d[2])*y+d[3])*y+d[4])*y+d[5])*y+d[6])*y+d[7])*y+d[8]);
		y = z*y;
	} else {
		/* evaluate erfc() for |u| > sqrt(2)*4.0 */
		z = z*SQRT_TWO/y;
		y = 2/(y*y);
		y = y*(((((p[0]*y+p[1])*y+p[2])*y+p[3])*y+p[4])*y+p[5])
		     /(((((q[0]*y+q[1])*y+q[2])*y+q[3])*y+q[4])*y+q[5]);
		y = z*(M_1_SQRTPI-y);
	}
	return (u < 0.0 ? y : 1-y);
}

/*
 * The inverse standard normal distribution.
 *
 *   Author:      Peter John Acklam <pjacklam@online.no>
 *   URL:         http://home.online.no/~pjacklam
 *
 * This function is based on the MATLAB code from the address above,
 * translated to C, and adapted for our purposes.
 */
Float intervalToStdNormal(Float p) {
	const Float a[6] = {
		-3.969683028665376e+01,  2.209460984245205e+02,
		-2.759285104469687e+02,  1.383577518672690e+02,
		-3.066479806614716e+01,  2.506628277459239e+00
	};
	const Float b[5] = {
		-5.447609879822406e+01,  1.615858368580409e+02,
		-1.556989798598866e+02,  6.680131188771972e+01,
		-1.328068155288572e+01
	};
	const Float c[6] = {
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00,  2.938163982698783e+00
	};
	const Float d[4] = {
		7.784695709041462e-03,  3.224671290700398e-01,
		2.445134137142996e+00,  3.754408661907416e+00
	};

	if (p <= 0)
		return -std::numeric_limits<Float>::infinity();
	else if (p >= 1)
		return -std::numeric_limits<Float>::infinity();

	Float q = std::min(p,1-p);
	Float t, u;
	if (q > (Float) 0.02425) {
		/* Rational approximation for central region. */
		u = q-(Float) 0.5;
		t = u*u;
		u = u*(((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])
		     /(((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1);
	} else {
		/* Rational approximation for tail region. */
		t = std::sqrt(-2*log(q));
		u = (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5])
		    /((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1);
	}

	/* The relative error of the approximation has absolute value less
	   than 1.15e-9.  One iteration of Halley's rational method (third
	   order) gives full machine precision... */
	t = stdnormal_cdf(u)-q;    /* error */
	t = t*SQRT_TWOPI*math::fastexp(u*u/2);   /* f(u)/df(u) */
	u = u-t/(1+u*t/2);     /* Halley's method */

	return p > (Float) 0.5 ? -u : u;
}

static Float intervalToTent(Float sample) {
	Float sign;

	if (sample < 0.5f) {
		sign = 1;
		sample *= 2;
	} else {
		sign = -1;
		sample = 2 * (sample - 0.5f);
	}

	return sign * (1 - std::sqrt(sample));
}

Point2 squareToTent(const Point2 &sample) {
	return Point2(
		intervalToTent(sample.x),
		intervalToTent(sample.y)
	);
}

Float intervalToNonuniformTent(Float a, Float b, Float c, Float sample) {
	Float factor;

	if (sample * (c-a) < b-a) {
		factor = a-b;
		sample *= (a-c)/(a-b);
	} else {
		factor = c-b;
		sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
	}

	return b + factor * (1-math::safe_sqrt(sample));
}

};

MTS_NAMESPACE_END
