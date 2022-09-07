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
#ifndef RCGPAR_RCG_HPP
#define RCGPAR_RCG_HPP

#include <vector>
#include <cstddef>
#include <fstream>

#include "Matrix.hpp"

#include <omp.h>
#include <algorithm>
#pragma omp declare reduction(vec_double_plus : std::vector<double> :	\
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

namespace rcgpar {
double digamma(double in);
void logsumexp(seamat::Matrix<double> &gamma_Z);
void logsumexp(seamat::Matrix<double> &gamma_Z, std::vector<double> &m);

double mixt_negnatgrad(const seamat::Matrix<double> &gamma_Z,
		       const std::vector<double> &N_k,
		       const seamat::Matrix<double> &logl, seamat::Matrix<double> &dL_dphi, bool mpi_mode = false);

void update_N_k(const seamat::Matrix<double> &gamma_Z, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, std::vector<double> &N_k, bool mpi_mode = false);

long double ELBO_rcg_mat(const seamat::Matrix<double> &logl, const seamat::Matrix<double> &gamma_Z,
			 const std::vector<double> &counts, const std::vector<double> &N_k,
			 const double bound_const, bool mpi_mode = false);

void revert_step(seamat::Matrix<double> &gamma_Z, const std::vector<double> &oldm);

double calc_bound_const(const std::vector<double> &log_times_observed,
			const std::vector<double> &alpha0);

void add_alpha0_to_Nk(const std::vector<double> &alpha0,
		      std::vector<double> &N_k);
void rcg_optl_mat(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed,
		  const std::vector<double> &alpha0,
		  const long double bound_const, const double tol, const uint16_t max_iters,
		  const bool mpi_mode, seamat::Matrix<double> &gamma_Z, std::ostream &log);
}

#endif
