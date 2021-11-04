#ifndef RCGMPI_MATRIX_HPP
#define RCGMPI_MATRIX_HPP

#include <vector>
#include <cmath>

// Basic matrix structure and operations
// Implementation was done following the instructions at
// https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Header-File
//
// **None of the operations validate the matrix sizes**
//
// This file provides implementations for the functions that are not
// parallellized.  Implementations for the functions that are
// parallellized should be provided in the .cpp file that is included
// at the very end of this file.

template <typename T> class Matrix {
private:
    std::vector<std::vector<T> > mat;
    unsigned rows;
    unsigned cols;

public:
    Matrix() = default;
    // Parameter constructor
    Matrix(unsigned _rows, unsigned _cols, const T& _initial);
    // Copy constructor
    Matrix(const Matrix<T>& rhs) {
	mat = rhs.mat;
	rows = rhs.get_rows();
	cols = rhs.get_cols();
    }
    // (Virtual) Destructor
    virtual ~Matrix() {};

    // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial);

    // Operator overloading
    Matrix<T>& operator=(const Matrix<T>& rhs);

    // Mathematical operators
    // Matrix-matrix
    Matrix<T> operator+(const Matrix<T>& rhs) const;
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T> operator-(const Matrix<T>& rhs) const;
    Matrix<T>& operator-=(const Matrix<T>& rhs);
    Matrix<T> operator*(const Matrix<T>& rhs) const;
    // In-place matrix-matrix left multiplication
    Matrix<T>& operator*=(const Matrix<T>& rhs) {
	Matrix result = (*this) * rhs;
	(*this) = result;
	return *this;
    }

    // Matrix-scalar, only in-place
    Matrix<T>& operator+=(const T& rhs);
    Matrix<T>& operator-=(const T& rhs);
    Matrix<T>& operator*=(const T& rhs);
    Matrix<T>& operator/=(const T& rhs);

    // Matrix-vector
    std::vector<T> operator*(const std::vector<T>& rhs) const;
    std::vector<double> operator*(const std::vector<long unsigned>& rhs) const;

    // Matrix-vector right multiplication, store result in arg
    void right_multiply(const std::vector<long unsigned>& rhs, std::vector<T>& result) const;
    void exp_right_multiply(const std::vector<T>& rhs, std::vector<T>& result) const;

    // Access individual elements
    T& operator()(unsigned row, unsigned col) {
	return this->mat[row][col];
    }
    // Access individual elements (const)
    const T& operator()(unsigned row, unsigned col) const {
	return this->mat[row][col];
    }

    // Access rows
    const std::vector<T>& get_row(unsigned row_id) const {
	return this->mat[row_id];
    }
    // Access columns
    const std::vector<T>& get_col(unsigned col_id) const;

    // LogSumExp a Matrix column
    T log_sum_exp_col(unsigned col_id) const {
	// Note: this function accesses the elements rather inefficiently so
	// it shouldn't be parallellised here. However, the caller can
	// parallellize logsumexping multiple cols.
	T max_elem = 0;
	T sum = 0;
	for (unsigned i = 0; i < this->rows; ++i) {
	    max_elem = (this->mat[i][col_id] > max_elem ? this->mat[i][col_id] : max_elem);
	}
	
	for (unsigned i = 0; i < this->rows; ++i) {
	    sum += std::exp(this->mat[i][col_id] - max_elem);
	}
	return max_elem + std::log(sum);
    }

    // Fill a matrix with the sum of two matrices
    void sum_fill(const Matrix<T>& rhs1, const Matrix<T>& rhs2);

    // Transpose
    Matrix<T> transpose() const;

    // Get the number of rows of the matrix
    unsigned get_rows() const { return this->rows; }
    // Get the number of columns of the matrix
    unsigned get_cols() const { return this->cols; }
};

#include "../src/Matrix_omp.cpp"

#endif
