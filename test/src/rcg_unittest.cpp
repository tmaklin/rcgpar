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
#include "rcg_unittest.hpp"

#include "gtest/gtest.h"

#include "rcg.hpp"
#include "openmp_config.hpp"

TEST_F(DigammaTest, ResultCorrect) {
    for (uint16_t i = 0; i < tests.size(); ++i) {
	SCOPED_TRACE(i);
	double got = rcgpar::digamma(tests[i]);
	EXPECT_NEAR(expects[i], got, 1e-6);
    }
}

TEST_F(Rcg, mixt_negnatgradNewnormCorrect) {
    rcgpar::Matrix<double> step_got(n_groups, n_obs, 0.0);
    double got = mixt_negnatgrad(gamma_Z, N_k, logl, step_got);
    EXPECT_NEAR(expected_newnorm, got, 1e-4);
}

TEST_F(Rcg, mixt_negnatgradStepUpdated) {
    rcgpar::Matrix<double> step_got(n_groups, n_obs, 0.0);
    mixt_negnatgrad(gamma_Z, N_k, logl, step_got);
    EXPECT_EQ(expected_step, step_got);
}

TEST_F(Rcg, logsumexp_returnmGammaZUpdated) {
    std::vector<double> oldm_got(n_obs, 0.0);
    rcgpar::Matrix<double> gamma_Z_got(gamma_Z);
    gamma_Z_got += oldstep_x_betaFR;
    gamma_Z_got += expected_step;
    logsumexp(gamma_Z_got, oldm_got);
    EXPECT_EQ(expected_gamma_Z, gamma_Z_got);
}

TEST_F(Rcg, logsumexpGammaZUpdated) {
    rcgpar::Matrix<double> gamma_Z_got(gamma_Z);
    gamma_Z_got += oldstep_x_betaFR;
    gamma_Z_got += expected_step;
    logsumexp(gamma_Z_got);
    EXPECT_EQ(expected_gamma_Z, gamma_Z_got);
}

TEST_F(Rcg, logsumexpOldmCorrect) {
    std::vector<double> oldm_got(n_obs, 0.0);
    rcgpar::Matrix<double> gamma_Z_got(gamma_Z);
    gamma_Z_got += oldstep_x_betaFR;
    gamma_Z_got += expected_step;
    logsumexp(gamma_Z_got, oldm_got);
    for (uint32_t i = 0; i < n_obs; ++i) {
 	SCOPED_TRACE(i);
	EXPECT_NEAR(expected_oldm[i], oldm_got[i], 1e-4);
    }
}

TEST_F(Rcg, ELBO_rcg_matBoundCorrect) {
    long double bound_got = 0.0;
    std::vector<double> N_k_new(n_groups, 0.0);
    expected_gamma_Z.exp_right_multiply(log_times_observed, N_k_new);
    std::transform(N_k_new.begin(), N_k_new.end(), alpha0.begin(), N_k_new.begin(), std::plus<double>());
    ELBO_rcg_mat(logl, expected_gamma_Z, log_times_observed, alpha0, N_k_new, bound_got);
    EXPECT_NEAR(expected_bound, bound_got, 1e-3);
}

TEST_F(Rcg, revert_stepGammaZCorrect) {
    rcgpar::Matrix<double> gamma_Z_got(expected_gamma_Z);
    revert_step(gamma_Z_got, expected_oldm);
    EXPECT_EQ(expected_reverted_gamma_Z, gamma_Z_got);
}

TEST_F(Rcg, calc_bound_constBoundConstCorrect) {
    double bound_const_got = rcgpar::calc_bound_const(log_times_observed, alpha0);
    EXPECT_NEAR(expected_bound_const, bound_const_got, 1e-2);
}
