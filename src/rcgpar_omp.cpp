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
#include <algorithm>
#include <numeric>
#include <iostream>

#include "rcg.hpp"
#include "openmp_config.hpp"

namespace rcgpar {
Matrix<double> rcg_optl_omp(const Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters, std::ostream &log) {
    uint16_t n_rows = logl.get_rows();
    uint32_t n_cols = log_times_observed.size();
    Matrix<double> gamma_Z(n_rows, n_cols, std::log(1.0/(double)n_rows)); // where gamma_Z is init at 1.0
    Matrix<double> oldstep(n_rows, n_cols, 0.0);
    Matrix<double> step(n_rows, n_cols, 0.0);
    std::vector<double> oldm(n_cols, 0.0);
    double oldnorm = 1.0;
    long double bound = -100000.0;
    bool didreset = false;

    double bound_const = calc_bound_const(log_times_observed, alpha0);

    std::vector<double> N_k(alpha0.size());
    gamma_Z.exp_right_multiply(log_times_observed, N_k);
    std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());
  
    for (uint16_t k = 0; k < maxiters; ++k) {
	double newnorm = mixt_negnatgrad(gamma_Z, N_k, logl, step);
	double beta_FR = newnorm/oldnorm;
	oldnorm = newnorm;

	if (didreset) {
	    oldstep *= 0.0;
	} else if (beta_FR > 0) {
	    oldstep *= beta_FR;
	    step += oldstep;
	}
	didreset = false;

	gamma_Z += step;
	logsumexp(gamma_Z, oldm);
	gamma_Z.exp_right_multiply(log_times_observed, N_k);
	std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());

	long double oldbound = bound;
	bound = 0.0;
	ELBO_rcg_mat(logl, gamma_Z, log_times_observed, alpha0, N_k, bound);
	bound += std::accumulate(N_k.begin(), N_k.end(), (double)0.0, [](double acc, double elem){ return acc + std::lgamma(elem); });
	bound += bound_const;

	if (bound < oldbound) {
	    didreset = true;
	    revert_step(gamma_Z, oldm);
	    if (beta_FR > 0) {
		gamma_Z -= oldstep;
	    }
	    logsumexp(gamma_Z);
	    gamma_Z.exp_right_multiply(log_times_observed, N_k);
	    std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());

	    bound = bound_const;
	    ELBO_rcg_mat(logl, gamma_Z, log_times_observed, alpha0, N_k, bound);
	} else {
	    oldstep = step;
	}
	if (k % 5 == 0) {
	    log << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
	}
	if (bound - oldbound < tol && !didreset) {
	    logsumexp(gamma_Z);
	    log << std::endl;
	    return(gamma_Z);
	}
    }
    logsumexp(gamma_Z);
    log << std::endl;
    return(gamma_Z);
}
}
