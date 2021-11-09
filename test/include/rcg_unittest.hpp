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
#ifndef RCGPAR_RCGPAR_UNITTEST_HPP
#define RCGPAR_RCGPAR_UNITTEST_HPP

#include <vector>
#include <cstddef>

#include "gtest/gtest.h"

#include "Matrix.hpp"
#include "rcgpar.hpp"

// Test digamma()
class DigammaTest : public ::testing::Test {
protected:
    void SetUp() override {
	this->expects = { 3.666681,4.49313,4.998763,5.521391,5.890845,6.155232,6.349544,6.67523,6.790242,6.789716,6.948787,7.083463,7.10944,7.125307,7.245922,7.342723,7.424796,7.429644,7.454423,7.538381,7.566032,7.682448,7.682166,7.760658,7.756377,7.842937,7.8882,7.88087,7.904453,8.003412,7.984717,8.056644,8.061121,8.074652,8.117342,8.141963,8.185475,8.240546,8.216422,8.252252,8.308739,8.3296,8.337507,8.371247,8.367322,8.407229,8.416926,8.463271,8.456402,8.513184,8.515145,8.525034,8.57059,8.585595,8.606247,8.610966,8.645924,8.665612,8.656205,8.681637,8.714146,8.723054,8.729884,8.760805,8.749339,8.790658,8.809692,8.807674,8.822722,8.842974,8.852751,8.861037,8.871359,8.896253,8.897765,8.920215,8.925264,8.945872,8.970063,8.963644,8.99025,9.007091,9.022007,9.017325,9.024177,9.040782,9.067534,9.068988,9.079973,9.092887,9.111634,9.116272,9.128577,9.133953,9.145141,9.16206,9.166411,9.170322,9.190482,9.206706 };
	this->tests = { 39.62078,89.90038,148.7294,250.4825,362.2105,471.6761,572.7317,793.0293,889.6282,889.1613,1042.386,1192.589,1223.963,1243.529,1402.874,1545.413,1677.557,1685.707,1727.987,1879.286,1931.96,2170.426,2169.814,2346.949,2336.924,2548.176,2666.142,2646.675,2709.821,2991.645,2936.246,3155.184,3169.341,3212.508,3352.599,3436.154,3588.948,3792.111,3701.735,3836.754,4059.692,4145.261,4178.161,4321.523,4304.596,4479.831,4523.478,4738.029,4705.598,4980.493,4990.27,5039.86,5274.738,5354.475,5466.198,5492.048,5687.417,5800.496,5746.186,5894.186,6088.933,6143.412,6185.509,6379.746,6307.021,6573.056,6699.358,6685.849,6787.211,6926.06,6994.103,7052.29,7125.454,7305.05,7316.105,7482.198,7520.071,7676.638,7864.595,7814.278,8024.961,8161.245,8283.882,8245.193,8301.877,8440.874,8669.721,8682.334,8778.232,8892.319,9060.59,9102.708,9215.396,9265.076,9369.309,9529.166,9570.716,9608.22,9803.874,9964.227 };
    }
    void TearDown() override {
	tests.clear();
	expects.clear();
	tests.shrink_to_fit();
	expects.shrink_to_fit();
    }
    // Test values
    std::vector<double> tests;
    // Expected values
    std::vector<double> expects;
};

class Rcg : public ::testing::Test {
protected:
    static void SetUpTestSuite() {}
    // Sizes
    static const uint16_t n_groups;
    static const uint32_t n_obs;

    // Params
    static const rcgpar::Matrix<double> logl;
    static const std::vector<double> log_times_observed;

    // Intermediate values from iteration k == 3
    static const rcgpar::Matrix<double> gamma_Z;
    static const rcgpar::Matrix<double> oldstep_x_betaFR;
    static const std::vector<double> N_k;

    // Parameters
    static const std::vector<double> alpha0;

    // Expected values
    static const double expected_bound_const;
    static const double expected_newnorm;
    static const long double expected_bound;

    static const rcgpar::Matrix<double> expected_step;
    static const rcgpar::Matrix<double> expected_gamma_Z;
    static const rcgpar::Matrix<double> expected_reverted_gamma_Z;

    static const std::vector<double> expected_oldm;
};

// Sizes
const uint16_t Rcg::n_groups = 4;
const uint32_t Rcg::n_obs = 10;

// Initial values
const rcgpar::Matrix<double> Rcg::logl(std::vector<double>(std::initializer_list<double>(
    { -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503,
      -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713,
      -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503,
      -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 }
											 )), Rcg::n_groups, Rcg::n_obs);
const std::vector<double> Rcg::log_times_observed(std::initializer_list<double>(
    { 7.681099, 7.04316, 6.849066, 5.278115, 5.164786, 5.062595, 6.947937, 6.863803, 7.277248, 7.666222 }));

// Intermediate values
const rcgpar::Matrix<double> Rcg::gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.861124, -0.824187, -0.737067, -0.830991, -0.792902, -0.702885, -0.76075,  -0.719832, -0.622649, -0.742541,
      -1.01295,  -0.976009, -0.888889, -0.982813, -0.944725, -0.854708, -0.912572, -0.871654, -0.774472, -1.26242,
      -2.33926,  -2.30233,  -2.21521,  -2.67719,  -2.6391,   -2.54908,  -6.91527,  -6.87435,  -6.77717,  -2.22068,
      -2.13905,  -2.47017,  -6.69137,  -2.10891,  -2.43888,  -6.65719,  -2.03867,  -2.36581,  -6.57695,  -2.02046 }
											 )), Rcg::n_groups, Rcg::n_obs);
const rcgpar::Matrix<double> Rcg::oldstep_x_betaFR(std::vector<double>(std::initializer_list<double>(
    { 7.49005, 7.42931, 7.2764,  7.43193, 7.36678, 7.20105, 7.28606, 7.2082,  7.00366, 7.41032,
      7.43893, 7.37819, 7.22528, 7.3808,  7.31566, 7.14993, 7.23494, 7.15708, 6.95253, 7.3608,
      7.01656, 6.95583, 6.80291, 6.96004, 6.89489, 6.72916, 6.83289, 6.75504, 6.55049, 6.93683,
      7.07923, 7.0201,  6.8859,  7.02111, 6.95756, 6.81056, 6.87524, 6.79898, 6.61316, 6.9995 }
											 )), Rcg::n_groups, Rcg::n_obs);
const std::vector<double> Rcg::N_k(std::initializer_list<double>({ 4857.97, 3905.03, 701.053, 903.946 }));

// Parameters
const std::vector<double> Rcg::alpha0(n_groups, 1.0);

// Expected values
const double Rcg::expected_bound_const = -85494;
const double Rcg::expected_newnorm = 0.193162;
const long double Rcg::expected_bound = 9161.51;

const rcgpar::Matrix<double> Rcg::expected_step(std::vector<double>(std::initializer_list<double>(
    { 8.33935, 8.30241, 8.21529, 8.30921, 8.27113, 8.18111, 8.23897, 8.19806, 8.10087, 8.22076,
      8.27279, 8.23585, 8.14873, 8.24266, 8.20457, 8.11455, 8.17241, 8.1315,  8.03431, 8.1606,
      7.88108, 7.84415, 7.75703, 7.85735, 7.81926, 7.72924, 7.86197, 7.82105, 7.72387, 7.7625,
      7.93521, 7.90467, 7.89242, 7.90508, 7.87339, 7.85823, 7.83484, 7.80032, 7.778,   7.81663 }
											 )), Rcg::n_groups, Rcg::n_obs);
const rcgpar::Matrix<double> Rcg::expected_gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.681538, -0.662494, -0.617806, -0.667704, -0.648392, -0.603055, -0.635526, -0.615577, -0.568692, -0.557316,
      -0.951042, -0.931998, -0.887311, -0.937208, -0.917896, -0.872559, -0.905031, -0.885081, -0.838196, -1.18688,
      -3.09143,  -3.07238,  -3.0277,   -3.43766,  -3.41835,  -3.37301,  -7.62022,  -7.60027,  -7.55338,  -2.96721,
      -2.77441,  -3.11543,  -7.28548,  -2.76058,  -3.10133,  -7.27073,  -2.7284,   -3.06852,  -7.23637,  -2.65019 }
											 )), Rcg::n_groups, Rcg::n_obs);
const rcgpar::Matrix<double> Rcg::expected_reverted_gamma_Z(std::vector<double>(std::initializer_list<double>(
    { 14.9683, 14.9075, 14.7546, 14.9101, 14.845,  14.6793, 14.7643, 14.6864, 14.4819, 14.8885,
      14.6988, 14.638,  14.4851, 14.6406, 14.5755, 14.4098, 14.4948, 14.4169, 14.2124, 14.259,
      12.5584, 12.4976, 12.3447, 12.1402, 12.0751, 11.9093, 7.77959, 7.70174, 7.49719, 12.4787,
      12.8754, 12.4546, 8.08695, 12.8173, 12.3921, 8.0116,  12.6714, 12.2335, 7.81421, 12.7957 }
											 )), Rcg::n_groups, Rcg::n_obs);
const std::vector<double> Rcg::expected_oldm(std::initializer_list<double>(
    { 15.6498, 15.57, 15.3724, 15.5779, 15.4934, 15.2823, 15.3998, 15.302, 15.0506, 15.4459 }));


#endif
