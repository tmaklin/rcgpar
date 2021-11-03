// Riemannian conjugate gradient for parameter estimation.
//
// This file contains an implementation that runs either
// single-threaded or parallellized with OpenMP.

#include "rcg.hpp"

#include <assert.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "openmp_config.hpp"

double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    assert(x > 0);
    for ( ; x < 7; ++x)
	result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += std::log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
}

void logsumexp(Matrix<double> &gamma_Z) {
    uint32_t n_cols = gamma_Z.get_cols();
    uint16_t n_rows = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n_cols; ++i) {
	double m = gamma_Z.log_sum_exp_col(i);
	for (uint16_t j = 0; j < n_rows; ++j) {
	    gamma_Z(j, i) -= m;
	}
    }
}

void logsumexp(Matrix<double> &gamma_Z, std::vector<double> &m) {
    uint32_t n_cols = gamma_Z.get_cols();
    uint16_t n_rows = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n_cols; ++i) {
	m[i] = gamma_Z.log_sum_exp_col(i);
    }

#pragma omp parallel for schedule(static)
    for (uint16_t i = 0; i < n_rows; ++i) {
	for (uint32_t j = 0; j < n_cols; ++j) {
	    gamma_Z(i, j) -= m[j];
	}
    }
}

double mixt_negnatgrad(const Matrix<double> &gamma_Z, const std::vector<double> &N_k, const Matrix<double> &logl, Matrix<double> &dL_dphi) {
    uint32_t n_cols = gamma_Z.get_cols();
    uint16_t n_rows = gamma_Z.get_rows();

    std::vector<double> colsums(n_cols, 0.0);
#pragma omp parallel for schedule(static) reduction(vec_double_plus:colsums)
    for (uint16_t i = 0; i < n_rows; ++i) {
	double digamma_N_k = digamma(N_k[i]) - 1.0;
	for (uint32_t j = 0; j < n_cols; ++j) {
	    dL_dphi(i, j) = logl(i, j);
	    dL_dphi(i, j) += digamma_N_k - gamma_Z(i, j);
	    colsums[j] += dL_dphi(i, j) * std::exp(gamma_Z(i, j));
	}
    }
  
    double newnorm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:newnorm)
    for (uint16_t i = 0; i < n_rows; ++i) {
	for (uint32_t j = 0; j < n_cols; ++j) {
	    // dL_dgamma(i, j) would be q_Z(i, j) * (dL_dphi(i, j) - colsums[j])
	    newnorm += std::exp(gamma_Z(i, j)) * (dL_dphi(i, j) - colsums[j]) * dL_dphi(i, j);
	}
    }
    return newnorm;
}

void ELBO_rcg_mat(const Matrix<double> &logl, const Matrix<double> &gamma_Z, const std::vector<double> &counts, const std::vector<double> &alpha0, const std::vector<double> &N_k, long double &bound) {
    uint16_t n_rows = gamma_Z.get_rows();
    uint32_t n_cols = gamma_Z.get_cols();
#pragma omp parallel for schedule(static) reduction(+:bound)
    for (uint16_t i = 0; i < n_rows; ++i) {
	for (uint32_t j = 0; j < n_cols; ++j) {
	    bound += std::exp(gamma_Z(i, j) + counts[j])*(logl(i, j) - gamma_Z(i, j));
	}
	bound -= std::lgamma(alpha0[i]) - std::lgamma(N_k[i]);
    }
}

void revert_step(Matrix<double> &gamma_Z, const std::vector<double> &oldm) {
    uint16_t n_rows = gamma_Z.get_rows();
    uint32_t n_cols = gamma_Z.get_cols();
#pragma omp parallel for schedule(static)
    for (uint16_t i = 0; i < n_rows; ++i) {
	for (uint32_t j = 0; j < n_cols; ++j) {
	    gamma_Z(i, j) += oldm[j];
	}
    }
}

Matrix<double> rcg_optl_mat(const Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t maxiters) {
    uint16_t n_rows = logl.get_rows();
    uint32_t n_cols = log_times_observed.size();
    Matrix<double> gamma_Z(n_rows, n_cols, std::log(1.0/(double)n_rows)); // where gamma_Z is init at 1.0
    Matrix<double> oldstep(n_rows, n_cols, 0.0);
    Matrix<double> step(n_rows, n_cols, 0.0);
    std::vector<double> oldm(n_cols, 0.0);
    double oldnorm = 1.0;
    long double bound = -100000.0;
    bool didreset = false;

    double bound_const = 0.0;
#pragma omp parallel for schedule(static) reduction(+:bound_const)
    for (uint32_t i = 0; i < n_cols; ++i) {
	bound_const += std::exp(log_times_observed[i]);
    }
  
#pragma omp parallel for schedule(static) reduction(+:bound_const)
    for (uint16_t i = 0; i < n_rows; ++i) {
	bound_const += alpha0[i];
	bound_const += std::lgamma(alpha0[i]);
    }
  
    bound_const = -std::lgamma(bound_const);
    std::vector<double> N_k(alpha0.size());
    gamma_Z.exp_right_multiply(log_times_observed, N_k);
  
#pragma omp parallel for schedule(static)
    for (uint16_t i = 0; i < n_rows; ++i) {
	N_k[i] += alpha0[i];
    }
  
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

#pragma omp parallel for schedule(static)
	for (uint16_t i = 0; i < n_rows; ++i) {
	    N_k[i] += alpha0[i];
	}
    
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
      
#pragma omp parallel for schedule(static)
	    for (uint16_t i = 0; i < n_rows; ++i) {
		N_k[i] += alpha0[i];
	    }

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
