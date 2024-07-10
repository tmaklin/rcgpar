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
#ifndef RCGPAR_RCGPAR_HPP
#define RCGPAR_RCGPAR_HPP

#include <vector>
#include <cstddef>
#include <ostream>

#include "Matrix.hpp"

namespace rcgpar {
seamat::DenseMatrix<double> rcg_optl_omp(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed,
			    const std::vector<double> &alpha0, const double &tol, uint16_t maxiters,
			    std::ostream &log);

seamat::DenseMatrix<double> rcg_optl_mpi(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed,
			    const std::vector<double> &alpha0, const double &tol, uint16_t maxiters,
			    std::ostream &log);

std::vector<double> mixture_components(const seamat::Matrix<double> &probs, const std::vector<double> &log_times_observed);

seamat::DenseMatrix<double> rcg_optl_torch(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, 
				const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, 
				std::ostream &log);

std::vector<double> mixture_components_torch(const seamat::Matrix<double> &probs, const std::vector<double> &log_times_observed);

std::vector<double> em_torch(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, 
				const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, 
				std::ostream &log, std::string precision)

}

#endif
