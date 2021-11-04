#ifndef RCGMPI_RCG_UTIL_HPP
#define RCGMPI_RCG_UTIL_HPP

#include <vector>

#include "Matrix.hpp"

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

#endif
