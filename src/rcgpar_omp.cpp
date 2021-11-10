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
//
// This file contains the rcg_optl_mat function for OpenMP calls.

#include <cmath>
#include <iostream>

#include "rcg.hpp"
#include "openmp_config.hpp"

namespace rcgpar {
Matrix<double> rcg_optl_omp(const Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    Matrix<double> gamma_Z(logl.get_rows(), log_times_observed.size(), std::log(1.0/(double)logl.get_rows())); // where gamma_Z is init at 1.0
    double bound_const = calc_bound_const(log_times_observed, alpha0);

    // Estimate gamma_Z
    rcg_optl_mat(logl, log_times_observed, alpha0,
		 bound_const, tol, max_iters,
		 false, gamma_Z, log);

    return(gamma_Z);
}
}
