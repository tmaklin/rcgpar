// Header-only matrix wrapper for std::vector<std::vector<>>
//
// Functions in this file run either single-threaded or parallellized
// with OpenMP.

#ifndef RCGPAR_MATRIX_CPP
#define RCGPAR_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>

#include "openmp_config.hpp"

namespace rcgpar {
// Parameter Constructor
template<typename T>
Matrix<T>::Matrix(uint32_t _rows, uint32_t _cols, const T& _initial) {
    mat.resize(_rows*_cols, _initial);
    rows = _rows;
    cols = _cols;
}
// Copy constructor from 2D vector
template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &rhs) {
    rows = rhs.size();
    cols = rhs.at(0).size();
    mat.resize(rows*cols);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < rows; ++i) {
	for (uint32_t j = 0; j < cols; ++j) {
	    this->operator()(i, j) = rhs[i][j];
	}
    }
}

// Resize a matrix
template<typename T>
void Matrix<T>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    if (new_rows != rows || new_cols != cols) {
	mat.resize(new_rows*new_cols, initial);
	rows = new_rows;
	cols = new_cols;
    }
}

// Assignment Operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
    if (&rhs == this)
	return *this;

    uint32_t new_rows = rhs.get_rows();
    uint32_t new_cols = rhs.get_cols();
    if (new_rows != rows || new_cols != cols) {
	resize(new_rows, new_cols, (T)0);
    }
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < new_rows; i++) {
	for (uint32_t j = 0; j < new_cols; j++) {
	    this->operator()(i, j) = rhs(i, j); 
	}
    }
    return *this;
}

// Matrix-matrix addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(i, j) + rhs(i,j);
	}
    }

    return result;
}

// In-place matrix-matrix addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) += rhs(i, j);
	}
    }

    return *this;
}

// Fill matrix with sum of two matrices
template <typename T>
void Matrix<T>::sum_fill(const Matrix<T>& rhs1, const Matrix<T>& rhs2) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) = rhs1(i, j) + rhs2(i, j);
	}
    }
}

// Matrix-matrix subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(i, j) - rhs(i, j);
	}
    }

    return result;
}

// In-place matrix-matrix subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) -= rhs(i, j);
	}
    }

    return *this;
}

// Matrix-matrix left multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    for (uint32_t k = 0; k < this->rows; k++) {
		result(i, j) += this->operator()(i, k) * rhs(k, j);
	    }
	}
    }

    return result;
}

// Transpose matrix
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(j, i);
	}
    }

    return result;
}

// In-place matrix-scalar addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) += rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j=0; j < this->cols; j++) {
	    this->operator()(i, j) -= rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar multiplication
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) *= rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar division
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) /= rhs;
	}
    }

    return *this;
}

// Matrix-vector right multiplication
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& rhs) const {
    std::vector<T> result(rhs.size(), 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < rows; i++) {
	for (uint32_t j = 0; j < cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }

    return result;
}

// Matrix-vector right multiplication, store result in arg
template<typename T>
void Matrix<T>::right_multiply(const std::vector<long unsigned>& rhs, std::vector<T>& result) const {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	result[i] = 0.0;
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }
}

// log-space Matrix-vector right multiplication, store result in arg
template<typename T>
void Matrix<T>::exp_right_multiply(const std::vector<T>& rhs, std::vector<T>& result) const {
    std::fill(result.begin(), result.end(), 0.0);
#pragma omp parallel for schedule(static) reduction(vec_double_plus:result)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += std::exp(this->operator()(i, j) + rhs[j]);
	}
    }
}

// Specialized matrix-vector right multiplication
template<typename T>
std::vector<double> Matrix<T>::operator*(const std::vector<long unsigned>& rhs) const {
    std::vector<double> result(this->rows, 0.0);
  
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }

    return result;
}

// // Access cols
// template<typename T>
// const std::vector<T>& Matrix<T>::get_col(unsigned col_id) const {
//     std::vector<T> col(this->rows);
// #pragma omp parallel for schedule(static)
//     for (unsigned i = 0; i < this->rows; ++i) {
// 	col[i] = this->mat(i, col_id);
//     }
//     return col;
// }
}

#endif
