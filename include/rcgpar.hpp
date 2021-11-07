#ifndef RCGPAR_RCGPAR_HPP
#define RCGPAR_RCGPAR_HPP

#include <vector>

#include "Matrix.hpp"

Matrix<double> rcg_optl_omp(const Matrix<double> &logl, const std::vector<double> &log_times_observed,
			    const std::vector<double> &alpha0, const double &tol, uint16_t maxiters);
Matrix<double> rcg_optl_mpi(Matrix<double> &logl, const std::vector<double> &log_times_observed,
			    const std::vector<double> &alpha0, const double &tol, uint16_t maxiters);

#endif
