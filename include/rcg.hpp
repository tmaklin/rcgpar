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

#include "Matrix.hpp"

namespace rcgpar {
void logsumexp(Matrix<double> &gamma_Z);
void logsumexp(Matrix<double> &gamma_Z, std::vector<double> &m);

double mixt_negnatgrad(const Matrix<double> &gamma_Z,
		       const std::vector<double> &N_k,
		       const Matrix<double> &logl, Matrix<double> &dL_dphi);

void ELBO_rcg_mat(const Matrix<double> &logl, const Matrix<double> &gamma_Z,
		  const std::vector<double> &counts,
		  const std::vector<double> &alpha0,
		  const std::vector<double> &N_k, long double &bound);

void revert_step(Matrix<double> &gamma_Z, const std::vector<double> &oldm);

double calc_bound_const(const std::vector<double> &log_times_observed,
			const std::vector<double> &alpha0);

void add_alpha0_to_Nk(const std::vector<double> &alpha0,
		      std::vector<double> &N_k);
}

#endif
