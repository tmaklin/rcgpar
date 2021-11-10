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
#include <iostream>

#include "rcg.hpp"
#include "MpiHandler.hpp"

namespace rcgpar {
Matrix<double> rcg_optl_mpi(const Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters, std::ostream &log) {
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
    std::vector<double> N_k(n_groups);
    update_N_k(gamma_Z_partial, log_times_observed, alpha0, N_k, true);

    for (uint16_t k = 0; k < maxiters; ++k) {
	double newnorm = mixt_negnatgrad(gamma_Z_partial, N_k, logl_partial, step_partial);
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

	update_N_k(gamma_Z_partial, log_times_observed, alpha0, N_k, true);

	long double oldbound = bound;
	bound = ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, N_k, bound_const, true);

	if (bound < oldbound) {
	    didreset = true;
	    revert_step(gamma_Z_partial, oldm_partial);
	    if (beta_FR > 0) {
		gamma_Z_partial -= oldstep_partial;
	    }

	    // Logsumexp 2
	    logsumexp(gamma_Z_partial, oldm_partial);

	    update_N_k(gamma_Z_partial, log_times_observed, alpha0, N_k, true);

	    bound = ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, N_k, bound_const, true);
	} else {
	    oldstep_partial = step_partial;
	}
	if (k % 5 == 0 && rank == 0) {
	    log << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
	}
	if (bound - oldbound < tol && !didreset) {
	    // Logsumexp 3
	    logsumexp(gamma_Z_partial, oldm_partial);

	    // Construct gamma_Z from the partials
	    Matrix<double> gamma_Z_full(n_groups, n_obs, 0.0);
	    for (uint16_t i = 0; i < n_groups; ++i) {
		MPI_Gatherv(&gamma_Z_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, &gamma_Z_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    }
	    MPI_Bcast(&gamma_Z_full.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	    log << std::endl;
	    return(gamma_Z_full);
	}
    }
    // Logsumexp 3
    logsumexp(gamma_Z_partial, oldm_partial);

    // Construct gamma_Z from the partials
    Matrix<double> gamma_Z_full(n_groups, n_obs, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Gatherv(&gamma_Z_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, &gamma_Z_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&gamma_Z_full.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    log << std::endl;
    return(gamma_Z_full);
}
}
