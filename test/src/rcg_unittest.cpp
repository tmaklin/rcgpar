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

TEST(rcg, digammaResultCorrect) {
    double expects[100] = { 3.666681,4.49313,4.998763,5.521391,5.890845,6.155232,6.349544,6.67523,6.790242,6.789716,6.948787,7.083463,7.10944,7.125307,7.245922,7.342723,7.424796,7.429644,7.454423,7.538381,7.566032,7.682448,7.682166,7.760658,7.756377,7.842937,7.8882,7.88087,7.904453,8.003412,7.984717,8.056644,8.061121,8.074652,8.117342,8.141963,8.185475,8.240546,8.216422,8.252252,8.308739,8.3296,8.337507,8.371247,8.367322,8.407229,8.416926,8.463271,8.456402,8.513184,8.515145,8.525034,8.57059,8.585595,8.606247,8.610966,8.645924,8.665612,8.656205,8.681637,8.714146,8.723054,8.729884,8.760805,8.749339,8.790658,8.809692,8.807674,8.822722,8.842974,8.852751,8.861037,8.871359,8.896253,8.897765,8.920215,8.925264,8.945872,8.970063,8.963644,8.99025,9.007091,9.022007,9.017325,9.024177,9.040782,9.067534,9.068988,9.079973,9.092887,9.111634,9.116272,9.128577,9.133953,9.145141,9.16206,9.166411,9.170322,9.190482,9.206706 };
    double tests[100] = { 39.62078,89.90038,148.7294,250.4825,362.2105,471.6761,572.7317,793.0293,889.6282,889.1613,1042.386,1192.589,1223.963,1243.529,1402.874,1545.413,1677.557,1685.707,1727.987,1879.286,1931.96,2170.426,2169.814,2346.949,2336.924,2548.176,2666.142,2646.675,2709.821,2991.645,2936.246,3155.184,3169.341,3212.508,3352.599,3436.154,3588.948,3792.111,3701.735,3836.754,4059.692,4145.261,4178.161,4321.523,4304.596,4479.831,4523.478,4738.029,4705.598,4980.493,4990.27,5039.86,5274.738,5354.475,5466.198,5492.048,5687.417,5800.496,5746.186,5894.186,6088.933,6143.412,6185.509,6379.746,6307.021,6573.056,6699.358,6685.849,6787.211,6926.06,6994.103,7052.29,7125.454,7305.05,7316.105,7482.198,7520.071,7676.638,7864.595,7814.278,8024.961,8161.245,8283.882,8245.193,8301.877,8440.874,8669.721,8682.334,8778.232,8892.319,9060.59,9102.708,9215.396,9265.076,9369.309,9529.166,9570.716,9608.22,9803.874,9964.227 };

    for (uint16_t i = 0; i < 100; ++i) {
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
