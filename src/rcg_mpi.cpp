// Riemannian conjugate gradient for parameter estimation.
//
// This file contains the rcg_optl_mat function for MPI calls.

#include "rcg.hpp"

#include <mpi.h>

#include <cmath>
#include <algorithm>
#include <iostream>

#include "rcg_util.hpp"

Matrix<double> rcg_optl_mpi(Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters) {
    int ntasks,rank;
    int rc = MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint16_t n_groups = alpha0.size();
    uint32_t n_obs;
    if (rank == 0) {
	n_obs = log_times_observed_full.size();
    }
    MPI_Bcast(&n_obs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Subdimensions for the processes
    uint32_t n_obs_per_task = n_obs/ntasks;
    uint32_t n_values_per_task = n_obs_per_task*n_groups;

    // Scatter the log likelihoods and log counts
    // log counts
    std::vector<double> log_times_observed(n_obs_per_task);
    MPI_Scatter(&log_times_observed_full.front(), n_obs_per_task, MPI_DOUBLE, &log_times_observed.front(), n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // log likelihoods
    Matrix<double> logl_partial(n_groups, n_obs_per_task, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Scatter(&logl_full.front() + i*n_obs, n_obs_per_task, MPI_DOUBLE, &logl_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
  	ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, alpha0, N_k, bound_partial);
	MPI_Allreduce(&bound_partial, &bound, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
	    ELBO_rcg_mat(logl_partial, gamma_Z_partial, log_times_observed, alpha0, N_k, bound_partial);
	    MPI_Allreduce(&bound_partial, &bound, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	    bound += bound_const;
	} else {
	    oldstep_partial = step_partial;
	}
	if (k % 5 == 0 && rank == 0) {
	    std::cerr << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
	}
	if (bound - oldbound < tol && !didreset) {
	    // Logsumexp 3
	    logsumexp(gamma_Z_partial, oldm_partial);
	    std::cerr << std::endl;
	    return(gamma_Z_partial);
	}
    }
    // Logsumexp 3
    logsumexp(gamma_Z_partial, oldm_partial);
    std::cerr << std::endl;
    return(gamma_Z_partial);
}
