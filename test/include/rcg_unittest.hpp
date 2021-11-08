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

class Rcg : public ::testing::Test {
protected:
    void SetUp() override {
	// Inputs
	std::vector<double> gamma_Z_vec = { -0.861124,-0.824187,-0.737067,-0.830991,-0.792902,-0.702885,-0.76075,-0.719832,-0.622649,-0.742541,
	                                    -1.01295,-0.976009,-0.888889,-0.982813,-0.944725,-0.854708,-0.912572,-0.871654,-0.774472,-1.26242,
					    -2.33926,-2.30233,-2.21521,-2.67719,-2.6391,-2.54908,-6.91527,-6.87435,-6.77717,-2.22068,
					    -2.13905,-2.47017,-6.69137,-2.10891,-2.43888,-6.65719,-2.03867,-2.36581,-6.57695,-2.02046 };
	std::vector<double> logl_vec = { -0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,
	                                 -0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.371713,
					 -0.0100503,-0.0100503,-0.0100503,-0.371713,-0.371713,-0.371713,-4.60517,-4.60517,-4.60517,-0.0100503,
					 -0.0100503,-0.371713,-4.60517,-0.0100503,-0.371713,-4.60517,-0.0100503,-0.371713,-4.60517,-0.0100503 };
	std::vector<double> step_vec = { 8.33935,8.30241,8.21529,8.30921,8.27113,8.18111,8.23897,8.19806,8.10087,8.22076,
                                         8.27279,8.23585,8.14873,8.24266,8.20457,8.11455,8.17241,8.1315,8.03431,8.1606,
					 7.88108,7.84415,7.75703,7.85735,7.81926,7.72924,7.86197,7.82105,7.72387,7.7625,
					 7.93521,7.90467,7.89242,7.90508,7.87339,7.85823,7.83484,7.80032,7.778,7.81663 };
	std::vector<double> expc_gamma_Z_vec = { -0.681538,-0.662494,-0.617806,-0.667704,-0.648392,-0.603055,-0.635526,-0.615577,-0.568692,-0.557316,
-0.951042,-0.931998,-0.887311,-0.937208,-0.917896,-0.872559,-0.905031,-0.885081,-0.838196,-1.18688,
-3.09143,-3.07238,-3.0277,-3.43766,-3.41835,-3.37301,-7.62022,-7.60027,-7.55338,-2.96721,
-2.77441,-3.11543,-7.28548,-2.76058,-3.10133,-7.27073,-2.7284,-3.06852,-7.23637,-2.65019 };
	std::vector<double> oldstep_x_betaFR_vec = { 7.49005,7.42931,7.2764,7.43193,7.36678,7.20105,7.28606,7.2082,7.00366,7.41032,
7.43893,7.37819,7.22528,7.3808,7.31566,7.14993,7.23494,7.15708,6.95253,7.3608,
7.01656,6.95583,6.80291,6.96004,6.89489,6.72916,6.83289,6.75504,6.55049,6.93683,
7.07923,7.0201,6.8859,7.02111,6.95756,6.81056,6.87524,6.79898,6.61316,6.9995 };
	std::vector<double> expc_rev_gamma_Z_vec = { 14.9683,14.9075,14.7546,14.9101,14.845,14.6793,14.7643,14.6864,14.4819,14.8885,
14.6988,14.638,14.4851,14.6406,14.5755,14.4098,14.4948,14.4169,14.2124,14.259,
12.5584,12.4976,12.3447,12.1402,12.0751,11.9093,7.77959,7.70174,7.49719,12.4787,
12.8754,12.4546,8.08695,12.8173,12.3921,8.0116,12.6714,12.2335,7.81421,12.7957 };
	// Sizes
	this->n_groups = 4;
	this->n_obs = 10;

	// Expecteds
	this->expected_newnorm = 0.193162;
	this->expected_step = rcgpar::Matrix<double>(step_vec, this->n_groups, this->n_obs);
	this->expected_gamma_Z = rcgpar::Matrix<double>(expc_gamma_Z_vec, this->n_groups, this->n_obs);
	this->expected_reverted_gamma_Z = rcgpar::Matrix<double>(expc_rev_gamma_Z_vec, this->n_groups, this->n_obs);
	this->expected_oldm = { 15.6498,15.57,15.3724,15.5779,15.4934,15.2823,15.3998,15.302,15.0506,15.4459 };
	this->expected_bound = 9161.51;
	this->expected_bound_const = -85494;

	// Params
	this->logl = rcgpar::Matrix<double>(logl_vec, this->n_groups, this->n_obs);
	this->gamma_Z = rcgpar::Matrix<double>(gamma_Z_vec, this->n_groups, this->n_obs);
	this->oldstep_x_betaFR = rcgpar::Matrix<double>(oldstep_x_betaFR_vec, this->n_groups, this->n_obs);
	this->log_times_observed = { 7.681099,7.04316,6.849066,5.278115,5.164786,5.062595,6.947937,6.863803,7.277248,7.666222 };
	this->N_k = { 4857.97,3905.03,701.053,903.946 };
	this->alpha0 = std::vector<double>(this->n_groups, 1.0);
    }
    // Result
    double expected_newnorm;
    rcgpar::Matrix<double> expected_step;
    rcgpar::Matrix<double> expected_gamma_Z;
    rcgpar::Matrix<double> expected_reverted_gamma_Z;
    std::vector<double> expected_oldm;
    long double expected_bound;
    double expected_bound_const;

    // Sizes
    uint16_t n_groups;
    uint32_t n_obs;

    // Params
    rcgpar::Matrix<double> logl;
    rcgpar::Matrix<double> gamma_Z;
    rcgpar::Matrix<double> oldstep_x_betaFR;
    std::vector<double> N_k;
    std::vector<double> alpha0;
    std::vector<double> log_times_observed;
};

#endif
