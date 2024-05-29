// rcgpar: parallel estimation of mixture model components
// https://github.com/tmaklin/rcgpar
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
// USA
//
#include "rcgpar.hpp"

// Riemannian conjugate gradient for parameter estimation.

#include <exception>
#include <string>
#include <cmath>
#include <iostream>

#include "rcg.hpp"

namespace rcgpar {
void check_input(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters) {
    uint16_t n_groups = logl.get_rows();
    uint32_t n_obs = logl.get_cols();
    if (tol < 0) {
	throw std::invalid_argument("Tolerance cannot be negative (type: double).");
    }
    if (max_iters == 0) {
	throw std::invalid_argument("Max iters cannot be 0 (type: uint16_t).");
    }
    if (n_groups != alpha0.size()) {
	throw std::domain_error("Number of components (rows) in logl differs from the number of values in alpha0.");
    }
    if (n_obs != log_times_observed.size()) {
	throw std::domain_error("Number of observations (columns) in logl differs from the number of observations in log_times_observed.");
    }
}

seamat::DenseMatrix<double> rcg_optl_omp(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    // Validate input data
    check_input(logl, log_times_observed, alpha0, tol, max_iters);

    // where gamma_Z is init at 1.0
    seamat::DenseMatrix<double> gamma_Z(logl.get_rows(), log_times_observed.size(), std::log(1.0/(double)logl.get_rows()));
    double bound_const = calc_bound_const(log_times_observed, alpha0);

    // Estimate gamma_Z
    rcg_optl_mat(logl, log_times_observed, alpha0,
		 bound_const, tol, max_iters,
		 false, gamma_Z, log);

    return(gamma_Z);
}

seamat::DenseMatrix<double> rcg_optl_mpi(const seamat::Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    throw std::runtime_error("`rcg_optl_mpi` is no longer supported by rcgpar, switch to rcg_optl_omp instead. `rcg_optl_mpi`  will be removed in the next major release."); 
}
}
