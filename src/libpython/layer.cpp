#include "base.h"
#include <mitsuba/layer/layer.h>
#include <mitsuba/layer/hg.h>
#include <mitsuba/layer/microfacet.h>
#include <mitsuba/layer/storage.h>
#include <mitsuba/render/scene.h>

using namespace mitsuba;

class DVectorHelper {
public:
	static std::string toString(const DVector &value) { return matToString(value); }
	static Float get(const DVector &value, int i) { return value[i]; }
	static Float dot(const DVector &v0, const DVector &v1) { return v0.dot(v1); }
	static DVector mul(const DVector &v, Float value) { return v*value; }
	static void set(DVector &value, int i, Float arg) { value[i] = arg; }
	static int len(const DVector &value) { return value.size(); }
	static void setZero(DVector &value) { value.setZero(); }
	static void setConstant(DVector &value, Float scalar) { value.setConstant(scalar); }
	static DVector head(DVector &value, int n) { return DVector(value.head(n)); }
	static DVector tail(DVector &value, int n) { return DVector(value.tail(n)); }
	static DVector *fromList(bp::list list) {
		int length = bp::len(list);
		DVector *result = new DVector(length);

		for (int i=0; i<length; ++i)
			(*result)[i] = bp::extract<Float>(list[i]);

		return result;
	}
};

class DMatrixHelper {
public:
	static std::string toString(const DMatrix &value) { return matToString(value); }
	static Float get(const DMatrix &value, bp::tuple pos) {
		if (bp::len(pos) != 2)
			SLog(EError, "Invalid matrix indexing operation, required a tuple of length 2");
		return value(bp::extract<int>(pos[0]), bp::extract<int>(pos[1]));
	}
	static void set(DMatrix &value, bp::tuple pos, Float arg) {
		if (bp::len(pos) != 2)
			SLog(EError, "Invalid matrix indexing operation, required a tuple of length 2");
		value(bp::extract<int>(pos[0]), bp::extract<int>(pos[1])) = arg;
	}
	static void setZero(DMatrix &value) { value.setZero(); }
	static void setConstant(DMatrix &value, Float scalar) { value.setConstant(scalar); }
	static void setIdentity(DMatrix &value) { value.setIdentity(); }


	static DVector mul(const DMatrix &mat, const DVector &vec) {
		return mat*vec;
	}
};

DMatrix LayerMode_reflectionTop(const LayerMode &value) { return value.reflectionTop(); }
DMatrix LayerMode_reflectionBottom(const LayerMode &value) { return value.reflectionBottom(); }
DMatrix LayerMode_transmissionBottomTop(const LayerMode &value) { return value.transmissionBottomTop(); }
DMatrix LayerMode_transmissionTopBottom(const LayerMode &value) { return value.transmissionTopBottom(); }
//DMatrix LayerMode_matrix(const LayerMode &value) { return value.matrix(); }
LayerMode Layer_getitem(const Layer &layer, int i) { return layer[i]; }

bp::tuple initializeQuadrature_py(int n) {
	DVector nodes, weights;
	initializeQuadrature(n, nodes, weights);
	return bp::make_tuple(nodes, weights);
}

Spectrum BSDFStorage_eval1(const BSDFStorage *storage, Float mu_i, Float mu_o, Float phi_d, bp::list basisCoeffs) {
	SAssert((size_t) bp::len(basisCoeffs) == storage->getBasisCount());

	float *coeffs = (float *) alloca(sizeof(float) * storage->getBasisCount());
	for (size_t i=0; i<storage->getBasisCount(); ++i)
		coeffs[i] = bp::extract<float>(basisCoeffs[i]);

	return storage->eval(mu_i, mu_o, phi_d, coeffs);
}

Spectrum BSDFStorage_eval2(const BSDFStorage *storage, Float mu_i, Float mu_o, Float phi_d) {
	return storage->eval(mu_i, mu_o, phi_d);
}

Float BSDFStorage_pdf1(const BSDFStorage *storage, Float mu_i, Float mu_o, Float phi_d, bp::list basisCoeffs) {
	SAssert((size_t) bp::len(basisCoeffs) == storage->getBasisCount());

	float *coeffs = (float *) alloca(sizeof(float) * storage->getBasisCount());
	for (size_t i=0; i<storage->getBasisCount(); ++i)
		coeffs[i] = bp::extract<float>(basisCoeffs[i]);

	return storage->pdf(mu_i, mu_o, phi_d, coeffs);
}

Float BSDFStorage_pdf2(const BSDFStorage *storage, Float mu_i, Float mu_o, Float phi_d) {
	return storage->pdf(mu_i, mu_o, phi_d);
}

bp::tuple BSDFStorage_sample1(const BSDFStorage *storage, Float mu_i, Point2 sample, bp::list basisCoeffs) {
	SAssert((size_t) bp::len(basisCoeffs) == storage->getBasisCount());

	float *coeffs = (float *) alloca(sizeof(float) * storage->getBasisCount());
	for (size_t i=0; i<storage->getBasisCount(); ++i)
		coeffs[i] = bp::extract<float>(basisCoeffs[i]);

	Float mu_o, phi_d, pdf;
	Spectrum value = storage->sample(mu_i, mu_o, phi_d, pdf, sample, coeffs);
	return bp::make_tuple(value, mu_o, phi_d, pdf);
}

bp::tuple BSDFStorage_sample2(const BSDFStorage *storage, Float mu_i, Point2 sample) {
	Float mu_o, phi_d, pdf;
	Spectrum value = storage->sample(mu_i, mu_o, phi_d, pdf, sample);
	return bp::make_tuple(value, mu_o, phi_d, pdf);
}

ref<BSDFStorage> BSDFStorage_fromLayerGeneral(const fs::path &outputFile,
		bp::list layers, size_t nChannels, size_t nBases = 1,
		size_t nParameters = 0, bp::object paramSampleCounts = bp::object(),
		bp::object paramSamplePositions = bp::object(),
		bool extrapolate = false, bool isBSDF = true, const std::string &metadata = "") {
	SAssert((size_t) bp::len(layers) == nChannels * nBases);
	SAssert(nParameters == 0 || (size_t) bp::len(paramSampleCounts) == nParameters);
	SAssert(nParameters == 0 || (size_t) bp::len(paramSamplePositions) == nParameters);

	const Layer **layers_arg = (const Layer **) alloca(sizeof(const Layer **) * bp::len(layers));
	size_t *paramSampleCounts_arg = NULL;
	float **paramSamplePositions_arg = NULL;

	for (int i=0; i<bp::len(layers); ++i)
		layers_arg[i] = bp::extract<const Layer*>(layers[i]);

    if (!paramSampleCounts.is_none()) {
    	paramSampleCounts_arg = (size_t *) alloca(sizeof(size_t) * bp::len(paramSampleCounts));
        for (int i=0; i<bp::len(paramSampleCounts); ++i)
            paramSampleCounts_arg[i] = bp::extract<size_t>(paramSampleCounts[i]);
    }

    if (!paramSamplePositions.is_none()) {
    	paramSamplePositions_arg = (float **) alloca(sizeof(float*) * bp::len(paramSamplePositions));
        for (int i=0; i<bp::len(paramSamplePositions); ++i) {
            bp::list list = bp::extract<bp::list>(paramSamplePositions[i]);
			SAssert((size_t) bp::len(list) == paramSampleCounts_arg[i]);
            paramSamplePositions_arg[i] = (float *) alloca(sizeof(float) * bp::len(list));
            for (int j=0; j<bp::len(list); ++j)
                paramSamplePositions_arg[i][j] = bp::extract<float>(list[j]);
        }
    }

	ref<BSDFStorage> result = BSDFStorage::fromLayerGeneral(outputFile, layers_arg, nChannels, nBases,
			nParameters, (const size_t *) paramSampleCounts_arg,
			(const float **) paramSamplePositions_arg, extrapolate, isBSDF, metadata);

	return result;
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setMicrofacet_overloads, Layer::setMicrofacet, 3, 5);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setMatusik_overloads, Layer::setMatusik, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(fromLayer_overloads, BSDFStorage::fromLayer, 2, 5);
BOOST_PYTHON_FUNCTION_OVERLOADS(fromLayerRGB_overloads, BSDFStorage::fromLayerRGB, 4, 7);
BOOST_PYTHON_FUNCTION_OVERLOADS(fromLayerGeneral_overloads, BSDFStorage_fromLayerGeneral, 3, 10);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addToTop_overloads, Layer::addToTop, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(addToBottom_overloads, Layer::addToBottom, 1, 2);


bp::tuple parameterHeuristicMicrofacet2(Float alpha, Float eta, Float k) {
	std::pair<int, int> result = parameterHeuristicMicrofacet(alpha, eta, k);
	return bp::make_tuple(result.first, result.second);
}

bp::tuple parameterHeuristicHG2(Float g) {
	std::pair<int, int> result = parameterHeuristicHG(g);
	return bp::make_tuple(result.first, result.second);
}

bp::tuple parameterHeuristicVMF2(Float kappa) {
	std::pair<int, int> result = parameterHeuristicVMF(kappa);
	return bp::make_tuple(result.first, result.second);
}

void export_layer() {
	bp::object layerModule(
		bp::handle<>(bp::borrowed(PyImport_AddModule("mitsuba.layer"))));
	bp::scope().attr("layer") = layerModule;
	PyObject *oldScope = bp::detail::current_scope;

	BP_SETSCOPE(layerModule);
	layerModule.attr("__path__") = "mitsuba.layer";

	bp::def("initializeQuadrature", &initializeQuadrature_py, BP_RETURN_VALUE);
	bp::def("parameterHeuristicMicrofacet", &parameterHeuristicMicrofacet2);
	bp::def("parameterHeuristicHG", &parameterHeuristicHG2);
	bp::def("parameterHeuristicVMF", &parameterHeuristicVMF2);
	bp::def("microfacet", &microfacet);
	bp::def("hg", &hg);
	bp::def("vmfKappa", &vmfKappa);

	BP_STRUCT(DVector, (bp::init<>()))
		.def(bp::init<int>())
		.def(bp::init<const DVector &>())
		.def(bp::self != bp::self)
		.def(bp::self == bp::self)
		.def(-bp::self)
		.def(bp::self + bp::self)
		.def(bp::self += bp::self)
		.def(bp::self - bp::self)
		.def(bp::self -= bp::self)
		.def("__mul__", &DVectorHelper::mul)
		.def(Float() * bp::self)
		.def(bp::self / Float())
		.def(bp::self /= Float())
		.def("__init__", bp::make_constructor(&DVectorHelper::fromList))
		.def("dot", &DVectorHelper::dot)
		.def("head", &DVectorHelper::head)
		.def("tail", &DVectorHelper::tail)
		.def("setZero", &DVectorHelper::setZero)
		.def("setConstant", &DVectorHelper::setConstant)
		.def("__repr__", &DVectorHelper::toString)
		.def("__len__", &DVectorHelper::len)
		.def("__getitem__", &DVectorHelper::get)
		.def("__setitem__", &DVectorHelper::set);

	BP_STRUCT(DMatrix, (bp::init<>()))
		.def(bp::init<int, int>())
		.def(bp::init<const DMatrix &>())
		.def(bp::self != bp::self)
		.def(bp::self == bp::self)
		.def(-bp::self)
		.def(bp::self + bp::self)
		.def(bp::self += bp::self)
		.def(bp::self - bp::self)
		.def(bp::self -= bp::self)
		.def(bp::self *= Float())
		.def(bp::self * Float())
		.def(Float() * bp::self)
		.def(bp::self / Float())
		.def(bp::self /= Float())
		.def("__mul__", &DMatrixHelper::mul, BP_RETURN_VALUE)
		.def("rows", &DMatrix::rows)
		.def("cols", &DMatrix::cols)
		.def("setZero", &DMatrixHelper::setZero)
		.def("setConstant", &DMatrixHelper::setConstant)
		.def("setIdentity", &DMatrixHelper::setIdentity)
		.def("__repr__", &DMatrixHelper::toString)
		.def("__getitem__", &DMatrixHelper::get)
		.def("__setitem__", &DMatrixHelper::set);

	BP_STRUCT(LayerMode, (bp::init<int>()))
		.def(bp::init<const LayerMode &>())
		.def("clear", &LayerMode::clear)
		.def("reverse", &LayerMode::reverse)
//		.def("matrix", &LayerMode_matrix, BP_RETURN_VALUE)
		.def("reflectionTop", &LayerMode_reflectionTop, BP_RETURN_VALUE)
		.def("reflectionBottom", &LayerMode_reflectionBottom, BP_RETURN_VALUE)
		.def("transmissionTopBottom", &LayerMode_transmissionTopBottom, BP_RETURN_VALUE)
		.def("transmissionBottomTop", &LayerMode_transmissionBottomTop, BP_RETURN_VALUE);

	BP_STRUCT(Layer, (bp::init<DVector, DVector, int>()))
		.def(bp::init<const Layer &>())
		.def("resize", &Layer::resize)
		.def("clear", &Layer::clear)
		.def("add", &Layer::add)
		.def("addToTop", &Layer::addToTop, addToTop_overloads())
		.def("addToBottom", &Layer::addToBottom, addToBottom_overloads())
		.def("expand", &Layer::expand)
		.def("reverse", &Layer::reverse)
		.def("addScaled", &Layer::addScaled)
		.def("tau", &Layer::tau)
		.def("scale", &Layer::scale)
		.def("albedoTop", &Layer::albedoTop)
		.def("albedoBottom", &Layer::albedoBottom)
		.def("setHenyeyGreenstein", &Layer::setHenyeyGreenstein)
		.def("setVMF", &Layer::setVMF)
		.def("setMicrofacet", &Layer::setMicrofacet, setMicrofacet_overloads())
		.def("setMatusik", &Layer::setMatusik, setMatusik_overloads())
		.def("setAbsorbing", &Layer::setAbsorbing)
		.def("setDiffuse", &Layer::setDiffuse)
		.def("solveAddingDoubling", &Layer::solveAddingDoubling)
		.def("solveDiscreteOrdinates", &Layer::solveDiscreteOrdinates)
		.def("solve", &Layer::solve)
		.def("clearBottomAndTransmission", &Layer::clearBottomAndTransmission)
		.def("__getitem__", &Layer_getitem, BP_RETURN_VALUE)
		.staticmethod("add");

	BP_CLASS(BSDFStorage, Object, (bp::init<fs::path, bp::optional<bool> >()))
		.def("close", &BSDFStorage::close)
		.def("getNodeCount", &BSDFStorage::getNodeCount)
		.def("getChannelCount", &BSDFStorage::getChannelCount)
		.def("getParameterCount", &BSDFStorage::getParameterCount)
		.def("getMaxOrder", &BSDFStorage::getMaxOrder)
		.def("getBasisCount", &BSDFStorage::getBasisCount)
		.def("getMetadata", &BSDFStorage::getMetadata, BP_RETURN_VALUE)
		.def("getEta", &BSDFStorage::getEta)
		.def("setEta", &BSDFStorage::setEta)
		.def("getAlpha", &BSDFStorage::getAlpha)
		.def("setAlpha", &BSDFStorage::setAlpha)
		.def("eval", &BSDFStorage_eval1)
		.def("eval", &BSDFStorage_eval2)
		.def("pdf", &BSDFStorage_pdf1)
		.def("pdf", &BSDFStorage_pdf2)
		.def("sample", &BSDFStorage_sample1)
		.def("sample", &BSDFStorage_sample2)
		.def("size", &BSDFStorage::size)
		.def("stats", &BSDFStorage::stats)
		.def("readMetadata", &BSDFStorage::readMetadata, BP_RETURN_VALUE)
		.def("merge", &BSDFStorage::merge)
		.def("fromLayer", &BSDFStorage::fromLayer, fromLayer_overloads())
		.def("fromLayerRGB", &BSDFStorage::fromLayerRGB, fromLayerRGB_overloads())
		.def("fromLayerGeneral", &BSDFStorage_fromLayerGeneral, fromLayerGeneral_overloads())
		.staticmethod("fromLayer")
		.staticmethod("fromLayerRGB")
		.staticmethod("fromLayerGeneral")
		.staticmethod("merge")
		.staticmethod("readMetadata");

	bp::detail::current_scope = oldScope;
}
