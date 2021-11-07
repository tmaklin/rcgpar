// Riemannian conjugate gradient for parameter estimation.
//
// This file contains the rcg_optl_mat function for OpenMP calls.

#include "rcgpar.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

#include "rcg.hpp"
#include "openmp_config.hpp"

Matrix<double> rcg_optl_omp(const Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters) {
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
	bound = bound_const;
	ELBO_rcg_mat(logl, gamma_Z, log_times_observed, alpha0, N_k, bound);

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
	    std::cerr << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
	}
	if (bound - oldbound < tol && !didreset) {
	    logsumexp(gamma_Z);
	    std::cerr << std::endl;
	    return(gamma_Z);
	}
    }
    logsumexp(gamma_Z);
    std::cerr << std::endl;
    return(gamma_Z);
}