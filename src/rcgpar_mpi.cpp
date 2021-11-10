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
// This file contains the rcg_optl_mat function for MPI calls.

#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "rcg.hpp"
#include "MpiHandler.hpp"

const uint16_t M_NUM_MAX_PROCESSES = 1024;

namespace rcgpar {
Matrix<double> rcg_optl_mpi(Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters, std::ostream &log) {
    // MPI handler
    MpiHandler handler;
    const int rank = handler.get_rank();

    // Input data dimensions
    const uint16_t n_groups = alpha0.size();
    uint32_t n_obs = log_times_observed_full.size();

    // Initialize variables for MPI
    MPI_Bcast(&n_obs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    handler.initialize(n_obs);
    const uint32_t n_obs_per_task = handler.obs_per_task(n_obs);
    const int* displs = handler.get_displacements();
    const int* sendcounts = handler.get_bufcounts();

    std::vector<double> log_times_observed(n_obs_per_task);
    MPI_Scatterv(&log_times_observed_full.front(), sendcounts, displs, MPI_DOUBLE, &log_times_observed.front(), n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // log likelihoods
    Matrix<double> logl_partial(n_groups, n_obs_per_task, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Scatterv(&logl_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, &logl_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Initialize variables.
    Matrix<double> gamma_Z_partial = Matrix<double>(n_groups, n_obs_per_task, std::log(1.0/(double)n_groups));
    Matrix<double> step_partial(n_groups, n_obs_per_task, 0.0);

    // Oldstep, oldm, and oldnorm are needed to revert the gradient descent step in some special cases.
    Matrix<double> oldstep_partial(n_groups, n_obs_per_task, 0.0);
    std::vector<double> oldm_partial(n_obs_per_task, 0.0);
    double oldnorm = 1.0;

    // ELBO variables
    long double bound = -100000.0;
    double bound_const = 0.0;
    if (rank == 0) {
	bound_const = calc_bound_const(log_times_observed_full, alpha0);
    }
    MPI_Bcast(&bound_const, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bool didreset = false;

    // // gamma_Z %*% exp(log_times_observed), store result in N_k.
    std::vector<double> N_k_partial(n_groups);
    gamma_Z_partial.exp_right_multiply(log_times_observed, N_k_partial);
    std::vector<double> N_k(n_groups);
    MPI_Allreduce(&N_k_partial.front(), &N_k.front(), n_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());

    for (uint16_t k = 0; k < maxiters; ++k) {
	double newnorm;
	double newnorm_partial = mixt_negnatgrad(gamma_Z_partial, N_k, logl_partial, step_partial);
	MPI_Allreduce(&newnorm_partial, &newnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double beta_FR = newnorm/oldnorm;
	oldnorm = newnorm;
    
	if (didreset) {
	    oldstep_partial *= 0.0;
	} else if (beta_FR > 0) {
	    oldstep_partial *= beta_FR;
	    step_partial += oldstep_partial;
	}
	didreset = false;

	gamma_Z_partial += step_partial;

	// Logsumexp 1
	logsumexp(gamma_Z_partial, oldm_partial);

	gamma_Z_partial.exp_right_multiply(log_times_observed, N_k_partial);
	MPI_Allreduce(&N_k_partial.front(), &N_k.front(), n_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());

	long double oldbound = bound;
	long double bound_partial = 0.0;
	bound = 0.0;
  	ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, bound_partial);
	MPI_Allreduce(&bound_partial, &bound, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        bound += std::accumulate(N_k.begin(), N_k.end(), (double)0.0, [](double acc, double elem){ return acc + std::lgamma(elem); });
	bound += bound_const;

	if (bound < oldbound) {
	    didreset = true;
	    revert_step(gamma_Z_partial, oldm_partial);
	    if (beta_FR > 0) {
		gamma_Z_partial -= oldstep_partial;
	    }

	    // Logsumexp 2
	    logsumexp(gamma_Z_partial, oldm_partial);

	    gamma_Z_partial.exp_right_multiply(log_times_observed, N_k_partial);
	    MPI_Allreduce(&N_k_partial.front(), &N_k.front(), n_groups, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	    std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());
	
	    bound_partial = 0.0;
	    ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, bound_partial);
	    MPI_Allreduce(&bound_partial, &bound, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	    bound += bound_const;
	} else {
	    oldstep_partial = step_partial;
	}
	if (k % 5 == 0 && rank == 0) {
	    log << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
	}
	if (bound - oldbound < tol && !didreset) {
	    // Logsumexp 3
	    logsumexp(gamma_Z_partial, oldm_partial);
	    log << std::endl;
	    return(gamma_Z_partial);
	}
    }
    // Logsumexp 3
    logsumexp(gamma_Z_partial, oldm_partial);
    log << std::endl;
    return(gamma_Z_partial);
}
}
