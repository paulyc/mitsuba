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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/layer/storage.h>

MTS_NAMESPACE_BEGIN

class TabulatedBSDF : public BSDF {
public:
	TabulatedBSDF(const Properties &props)
		: BSDF(props) {
		fs::path filename = Thread::getThread()->getFileResolver()->resolve(
			props.getString("filename"));
		m_storage = new BSDFStorage(filename);
	}

	TabulatedBSDF(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_storage = static_cast<BSDFStorage *>(manager->getInstance(stream));
		size_t count = stream->readSize();
		m_textures.reserve(count);
		for (size_t i=0; i<count; ++i)
			m_textures.push_back(static_cast<Texture *>(manager->getInstance(stream)));
		configure();
	}

	void configure() {
		m_components.clear();
		m_components.push_back(EGlossyReflection | EGlossyTransmission | EFrontSide | EBackSide);
		m_usesRayDifferentials = false;
		for (size_t i=0; i<m_textures.size(); ++i)
			m_usesRayDifferentials |= m_textures[i]->usesRayDifferentials();
		BSDF::configure();

		if (m_textures.size() != m_storage->getParameterCount())
			Log(EError, "The number of provided textures does not match the "
				"parameter count of the BRDF data file");

		m_coeffCount = m_storage->getBasisCount() * m_storage->getChannelCount();
		m_homogeneous = m_storage->getBasisCount() == 1 && m_storage->getParameterCount() == 0;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
		manager->serialize(stream, m_storage.get());
		stream->writeSize(m_textures.size());
		for (size_t i=0; i<m_textures.size(); ++i)
			manager->serialize(stream, m_textures[i].get());
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.02f;
	}

	Spectrum getDiffuseReflectance(const Intersection &its) const {
		return Spectrum(0.0f); // not implemented
	}

	void computeBasisCoeffs(const Intersection &its, float *coeffs) const {
		if (EXPECT_TAKEN(m_homogeneous)) { /* Fast path */
			for (size_t i=0; i<m_coeffCount; ++i)
				coeffs[i] = 1.0f;
			return;
		} else {
			typedef TSpectrum<float, 3> Color3f;

			Color3f *values = (Color3f *) alloca(sizeof(Color3f) * m_textures.size());
			Color3f **weights = (Color3f **) alloca(sizeof(Color3f *) * m_storage->getParameterCount());
			for (size_t i=0; i<m_storage->getParameterCount(); ++i)
				weights[i] = (Color3f *) alloca(sizeof(Color3f) * m_storage->getParameterSampleCount(i));

			for (size_t i=0; i<m_textures.size(); ++i) {
				Float R, G, B;
				m_textures[i].get()->eval(its).toLinearRGB(R, G, B);
				Float Y = R * 0.212671f + G * 0.715160f + B * 0.072169f;
				values[i][0] = (float) Y; values[i][1] = (float) R; values[i][2] = (float) B;
			}

			for (size_t param=0; param<m_storage->getParameterCount(); ++param) {
				size_t count = m_storage->getParameterSampleCount(param);
				const float *pos = m_storage->getParameterSamplePositions(param);
				const Color3f &value = values[param];

				for (int ch=0; ch<3; ++ch) {
					for (size_t i = 0; i<count; ++i) {
						float prod = 1.0f;
						for (size_t k = 0; k<count; ++k)
							prod *= (i != k) ? ((value[ch] - pos[k]) / (pos[i]-pos[k])) : 1.0f;
						weights[param][i][ch] = prod;
					}
				}
			}

			for (size_t channel=0; channel<m_storage->getChannelCount(); ++channel) {
				for (size_t basis=0; basis<m_storage->getBasisCount(); ++basis) {
					float weight = 1;
					uint32_t index = (uint32_t) basis;
					for (size_t param=0; param<m_storage->getParameterCount(); ++param) {
						uint32_t count = (uint32_t) m_storage->getParameterSampleCount(param);
						uint32_t weightOffset = index % count;
						index /= count;

						weight *= weights[param][weightOffset][channel];
					}
					*coeffs++ = weight;
				}
			}
		}
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle)
			return Spectrum(0.0f);

		const Vector wi = bRec.wi, wo = -bRec.wo;

		Float phi_d = math::safe_acos(
			(wi.x * wo.x + wi.y * wo.y) / std::sqrt(
			(wi.x * wi.x + wi.y * wi.y) *
			(wo.x * wo.x + wo.y * wo.y)));

		float *basisCoeffs = (float *) alloca(m_coeffCount * sizeof(float));
		computeBasisCoeffs(bRec.its, basisCoeffs);

		return m_storage->eval(Frame::cosTheta(wi),
			Frame::cosTheta(wo), phi_d, basisCoeffs);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle)
			return 0.0f;

		const Vector wi = bRec.wi, wo = -bRec.wo;

		Float phi_d = math::safe_acos(
			(wi.x * wo.x + wi.y * wo.y) / std::sqrt(
			(wi.x * wi.x + wi.y * wi.y) *
			(wo.x * wo.x + wo.y * wo.y)));

		float *basisCoeffs = (float *) alloca(m_coeffCount * sizeof(float));
		computeBasisCoeffs(bRec.its, basisCoeffs);

		return m_storage->pdf(Frame::cosTheta(wi),
			Frame::cosTheta(wo), phi_d, basisCoeffs);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float mu_i = Frame::cosTheta(bRec.wi);
		Float mu_o, phi_d, pdf;

		float *basisCoeffs = (float *) alloca(m_coeffCount * sizeof(float));
		computeBasisCoeffs(bRec.its, basisCoeffs);
		Spectrum weight = m_storage->sample(mu_i, mu_o, phi_d, pdf, sample, basisCoeffs);

		Float sinPhi, cosPhi;
		math::sincos(phi_d, &sinPhi, &cosPhi);

		Float sin_o = std::sqrt(1-mu_o*mu_o);
		Vector2 d = normalize(Vector2(bRec.wi.x, bRec.wi.y));

		bRec.wo = -Vector(
			sin_o * (cosPhi*d.x - sinPhi*d.y),
			sin_o * (sinPhi*d.x + cosPhi*d.y),
			mu_o
		);

		if (Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo) > 0) {
			bRec.eta = 1.0f;
		} else {
			bRec.eta = Frame::cosTheta(bRec.wo) < 0
				? m_storage->getEta() : (1/m_storage->getEta());

			if (bRec.mode == ERadiance) {
				Float factor = 1/bRec.eta;
				weight *= factor*factor;
			}
		}

		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection | EGlossyTransmission;
		return weight;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		Float mu_i = Frame::cosTheta(bRec.wi);
		Float mu_o, phi_d;
		float *basisCoeffs = (float *) alloca(m_storage->getBasisCount() * sizeof(float));
		computeBasisCoeffs(bRec.its, basisCoeffs);
		Spectrum weight = m_storage->sample(mu_i, mu_o, phi_d, pdf, sample, basisCoeffs);

		Float sinPhi, cosPhi;
		math::sincos(phi_d, &sinPhi, &cosPhi);

		Float sin_o = std::sqrt(1-mu_o*mu_o);
		Vector2 d = normalize(Vector2(bRec.wi.x, bRec.wi.y));

		bRec.wo = -Vector(
			sin_o * (cosPhi*d.x - sinPhi*d.y),
			sin_o * (sinPhi*d.x + cosPhi*d.y),
			mu_o
		);

		if (Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo) > 0) {
			bRec.eta = 1.0f;
		} else {
			bRec.eta = Frame::cosTheta(bRec.wo) < 0
				? m_storage->getEta() : (1/m_storage->getEta());

			if (bRec.mode == ERadiance) {
				Float factor = 1/bRec.eta;
				weight *= factor*factor;
			}
		}

		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection | EGlossyTransmission;

		return weight;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			m_textures.push_back(static_cast<Texture *>(child));
		} else {
			BSDF::addChild(name, child);
		}
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "TabulatedBSDF[" << endl
			<< "  id = \"" << getID() << "\"" << endl
			<< "  storage = " << indent(m_storage.toString()) << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	ref<BSDFStorage> m_storage;
	ref_vector<Texture> m_textures;
	size_t m_coeffCount;
	bool m_homogeneous;
};

MTS_IMPLEMENT_CLASS_S(TabulatedBSDF, false, BSDF)
MTS_EXPORT_PLUGIN(TabulatedBSDF, "Tabulated BSDF")
MTS_NAMESPACE_END
