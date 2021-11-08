// Riemannian conjugate gradient for parameter estimation.
//
// This file contains implementations for the rcg.cpp functions that
// run either single-threaded or parallellized with OpenMP.

#include "rcg.hpp"

#include <assert.h>

#include <cmath>

#include "openmp_config.hpp"

namespace rcgpar {
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
    uint32_t n_obs = gamma_Z.get_cols();
    uint16_t n_groups = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n_obs; ++i) {
	double m = gamma_Z.log_sum_exp_col(i);
	for (uint16_t j = 0; j < n_groups; ++j) {
	    gamma_Z(j, i) -= m;
	}
    }
}

void logsumexp(Matrix<double> &gamma_Z, std::vector<double> &m) {
    uint32_t n_obs = gamma_Z.get_cols();
    uint16_t n_groups = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < n_obs; ++i) {
	m[i] = gamma_Z.log_sum_exp_col(i);
    }

#pragma omp parallel for schedule(static)
    for (uint16_t i = 0; i < n_groups; ++i) {
	for (uint32_t j = 0; j < n_obs; ++j) {
	    gamma_Z(i, j) -= m[j];
	}
    }
}

double mixt_negnatgrad(const Matrix<double> &gamma_Z, const std::vector<double> &N_k, const Matrix<double> &logl, Matrix<double> &dL_dphi) {
    uint32_t n_obs = gamma_Z.get_cols();
    uint16_t n_groups = gamma_Z.get_rows();

    std::vector<double> colsums(n_obs, 0.0);
#pragma omp parallel for schedule(static) reduction(vec_double_plus:colsums)
    for (uint16_t i = 0; i < n_groups; ++i) {
	double digamma_N_k = digamma(N_k[i]) - 1.0;
	for (uint32_t j = 0; j < n_obs; ++j) {
	    dL_dphi(i, j) = logl(i, j);
	    dL_dphi(i, j) += digamma_N_k - gamma_Z(i, j);
	    colsums[j] += dL_dphi(i, j) * std::exp(gamma_Z(i, j));
	}
    }

    double newnorm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:newnorm)
    for (uint16_t i = 0; i < n_groups; ++i) {
	for (uint32_t j = 0; j < n_obs; ++j) {
	    // dL_dgamma(i, j) would be q_Z(i, j) * (dL_dphi(i, j) - colsums[j])
	    newnorm += std::exp(gamma_Z(i, j)) * (dL_dphi(i, j) - colsums[j]) * dL_dphi(i, j);
	}
    }
    return newnorm;
}

void ELBO_rcg_mat(const Matrix<double> &logl, const Matrix<double> &gamma_Z, const std::vector<double> &counts, const std::vector<double> &alpha0, const std::vector<double> &N_k, long double &bound) {
    uint16_t n_groups = gamma_Z.get_rows();
    uint32_t n_obs = gamma_Z.get_cols();
#pragma omp parallel for schedule(static) reduction(+:bound)
    for (uint16_t i = 0; i < n_groups; ++i) {
	for (uint32_t j = 0; j < n_obs; ++j) {
	    bound += std::exp(gamma_Z(i, j) + counts[j])*(logl(i, j) - gamma_Z(i, j));
	}
    }
}

void revert_step(Matrix<double> &gamma_Z, const std::vector<double> &oldm) {
    uint16_t n_groups = gamma_Z.get_rows();
    uint32_t n_obs = gamma_Z.get_cols();
#pragma omp parallel for schedule(static)
    for (uint16_t i = 0; i < n_groups; ++i) {
	for (uint32_t j = 0; j < n_obs; ++j) {
	    gamma_Z(i, j) += oldm[j];
	}
    }
}

double calc_bound_const(const std::vector<double> &log_times_observed, const std::vector<double> &alpha0) {
    double counts_sum = 0.0;
    uint32_t n_obs = log_times_observed.size();
#pragma omp parallel for schedule(static) reduction(+:counts_sum)
    for (uint32_t i = 0; i < n_obs; ++i) {
	counts_sum += std::exp(log_times_observed[i]);
    }

    double alpha0_sum = 0.0;
    double lgamma_alpha0_sum = 0.0;
    uint16_t n_groups = alpha0.size();
#pragma omp parallel for schedule(static) reduction(+:alpha0_sum) reduction(+:lgamma_alpha0_sum)
    for (uint32_t i = 0; i < n_groups; ++i) {
	alpha0_sum += alpha0[i];
	lgamma_alpha0_sum += std::lgamma(alpha0[i]);
    }
    double bound_const = std::lgamma(alpha0_sum);
    bound_const -= std::lgamma(alpha0_sum + counts_sum);
    bound_const -= lgamma_alpha0_sum;
    return bound_const;
}
}
