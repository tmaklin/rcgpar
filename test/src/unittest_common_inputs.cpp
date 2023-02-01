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
#include "unittest_common_inputs.hpp"

// Input data
const seamat::DenseMatrix<double> LogLikelihoodTest::logl(std::vector<double>(std::initializer_list<double>(
    { -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503,
      -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713,
      -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503,
      -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 }
											 )), TEST_N_GROUPS, TEST_N_OBS);
const std::vector<double> LogCountsTest::log_times_observed(std::initializer_list<double>(
    { 7.681099, 7.04316, 6.849066, 5.278115, 5.164786, 5.062595, 6.947937, 6.863803, 7.277248, 7.666222 }));
const std::vector<double> Alpha0Test::alpha0(TEST_N_GROUPS, 1.0);

// Intermediate values from iteration k == 3
const seamat::DenseMatrix<double> GammaZTest::gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.861124, -0.824187, -0.737067, -0.830991, -0.792902, -0.702885, -0.76075,  -0.719832, -0.622649, -0.742541,
      -1.01295,  -0.976009, -0.888889, -0.982813, -0.944725, -0.854708, -0.912572, -0.871654, -0.774472, -1.26242,
      -2.33926,  -2.30233,  -2.21521,  -2.67719,  -2.6391,   -2.54908,  -6.91527,  -6.87435,  -6.77717,  -2.22068,
      -2.13905,  -2.47017,  -6.69137,  -2.10891,  -2.43888,  -6.65719,  -2.03867,  -2.36581,  -6.57695,  -2.02046 }
											 )), TEST_N_GROUPS, TEST_N_OBS);
const seamat::DenseMatrix<double> ExpectedStepTest::expected_step(std::vector<double>(std::initializer_list<double>(
    { 8.33935, 8.30241, 8.21529, 8.30921, 8.27113, 8.18111, 8.23897, 8.19806, 8.10087, 8.22076,
      8.27279, 8.23585, 8.14873, 8.24266, 8.20457, 8.11455, 8.17241, 8.1315,  8.03431, 8.1606,
      7.88108, 7.84415, 7.75703, 7.85735, 7.81926, 7.72924, 7.86197, 7.82105, 7.72387, 7.7625,
      7.93521, 7.90467, 7.89242, 7.90508, 7.87339, 7.85823, 7.83484, 7.80032, 7.778,   7.81663 }
											 )), TEST_N_GROUPS, TEST_N_OBS);
const seamat::DenseMatrix<double> ExpectedGammaZTest::expected_gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.681538, -0.662494, -0.617806, -0.667704, -0.648392, -0.603055, -0.635526, -0.615577, -0.568692, -0.557316,
      -0.951042, -0.931998, -0.887311, -0.937208, -0.917896, -0.872559, -0.905031, -0.885081, -0.838196, -1.18688,
      -3.09143,  -3.07238,  -3.0277,   -3.43766,  -3.41835,  -3.37301,  -7.62022,  -7.60027,  -7.55338,  -2.96721,
      -2.77441,  -3.11543,  -7.28548,  -2.76058,  -3.10133,  -7.27073,  -2.7284,   -3.06852,  -7.23637,  -2.65019 }
											 )), TEST_N_GROUPS, TEST_N_OBS);
const std::vector<double> ExpectedNkTest::expected_N_k(std::initializer_list<double>(
    { 5585.01,3983.44,327.192,472.355 }));

const long double ExpectedBoundConstTest::expected_bound_const = -85494;

const std::vector<double> ExpectedOldMTest::expected_oldm(std::initializer_list<double>(
    { 15.6498, 15.57, 15.3724, 15.5779, 15.4934, 15.2823, 15.3998, 15.302, 15.0506, 15.4459 }));

// Final output from rcg_optl_omp or rcg_optl_mpi
const seamat::DenseMatrix<double> FinalGammaZTest::final_gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.0010899, -0.00104044, -0.000928571, -0.00104519, -0.000995734, -0.000883857, -0.000944069, -0.000894604, -0.000782716, -0.000853449,
      -7.15745,   -7.1574,     -7.15729,     -7.15741,    -7.15736,     -7.15725,     -7.15731,     -7.15726,     -7.15715,     -7.51888,
      -8.82298,   -8.82293,    -8.82282,     -9.1846,     -9.18455,     -9.18444,     -13.418,      -13.4179,     -13.4178,     -8.82274,
      -8.72199,   -9.0836,     -13.3169,     -8.72195,    -9.08356,     -13.3169,     -8.72184,     -9.08346,     -13.3168,     -8.72175 }
											 )), TEST_N_GROUPS, TEST_N_OBS);
