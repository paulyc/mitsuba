#pragma once
#if !defined(__MITSUBA_LAYER_COMMON_H_)
#define __MITSUBA_LAYER_COMMON_H_

#include <mitsuba/mitsuba.h>

#ifndef NDEBUG
# define NDEBUG
#endif
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Core>
#include <Eigen/SparseCore>

#define ERROR_GOAL 1e-3f

MTS_NAMESPACE_BEGIN

/* For convenience: turn Eigen matrices/vectors into MATLAB-style strings */
template <int N, int M> inline std::string matToString(const Eigen::Matrix<Float, N, M> &m) {
	std::ostringstream oss;
	if (m.cols() == 1)
		oss << m.transpose().format(Eigen::IOFormat(7, 1, ", ", ";\n", "", "", "[", "]"));
	else
		oss << m.format(Eigen::IOFormat(7, 0, ", ", ";\n", "", "", "[", "]"));
	return oss.str();
}

/* For convenience: turn Eigen matrices/vectors into MATLAB-style strings */
inline std::string matToString(const Eigen::SparseMatrix<Float> &m) {
	std::ostringstream oss;
	oss << m << endl;
	return oss.str();
}

/* Define a few convenient shortcuts for dense Eigen matrices */
typedef Eigen::Matrix<Float, Eigen::Dynamic, 1>                               DVector;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>                  DMatrix;
typedef Eigen::Map<DMatrix, Eigen::Aligned>                                   DMap;
typedef Eigen::Block<DMap>                                                    Block;
typedef Eigen::Block<const DMap>                                              ConstBlock;
typedef Eigen::SparseMatrix<Float, Eigen::ColMajor>                           SMatrix;
typedef Eigen::Triplet<Float>                                                 Triplet;

struct Quartet {
	int l, o, i;
	Float value;

	inline Quartet(int l, int o, int i, Float value)
	 : l(l), o(o), i(i), value(value) { }
};


template <typename VectorType> void sparseDiag(const VectorType &vec, SMatrix &scale) {
	scale = SMatrix(vec.size(), vec.size());
	for (int i=0; i<vec.size(); ++i)
		scale.insert(i, i) = vec[i];
	scale.makeCompressed();
}

inline void sparsify(const DMatrix &dense, SMatrix &sparse) {
	sparse.setZero();

	for (int j=0; j<dense.cols(); ++j) {
		for (int i=0; i<dense.rows(); ++i) {
			Float value = dense.coeff(i, j);
			if (value != 0)
				sparse.insert(i, j) = value;
		}
	}
	sparse.makeCompressed();
}

MTS_NAMESPACE_END

#endif /* __MITSUBA_LAYER_COMMON_H_ */
