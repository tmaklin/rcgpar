#ifndef RCGMPI_RCG_HPP
#define RCGMPI_RCG_HPP

#include <vector>

#include "Matrix.hpp"

Matrix<double> rcg_optl_mat(const Matrix<double> &logl, const std::vector<double> &log_times_observed,
			    const std::vector<double> &alpha0, const double &tol, uint16_t maxiters);

#endif
