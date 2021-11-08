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

class Rcgpar : public ::testing::Test {
protected:
    void SetUp() override {
	std::vector<double> expc_vec = { -0.0010899,-0.00104044,-0.000928571,-0.00104519,-0.000995734,-0.000883857,-0.000944069,-0.000894604,-0.000782716,-0.000853449,
	                                 -7.15745,-7.1574,-7.15729,-7.15741,-7.15736,-7.15725,-7.15731,-7.15726,-7.15715,-7.51888,
					 -8.82298,-8.82293,-8.82282,-9.1846,-9.18455,-9.18444,-13.418,-13.4179,-13.4178,-8.82274,
					 -8.72199,-9.0836,-13.3169,-8.72195,-9.08356,-13.3169,-8.72184,-9.08346,-13.3168,-8.72175 };
	std::vector<double> logl_vec = { -0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,
	                                 -0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.0100503,-0.371713,
					 -0.0100503,-0.0100503,-0.0100503,-0.371713,-0.371713,-0.371713,-4.60517,-4.60517,-4.60517,-0.0100503,
					 -0.0100503,-0.371713,-4.60517,-0.0100503,-0.371713,-4.60517,-0.0100503,-0.371713,-4.60517,-0.0100503 };
	this->n_groups = 4;
	this->n_obs = 10;

	this->expected = rcgpar::Matrix<double>(expc_vec, this->n_groups, this->n_obs);


	this->logl = rcgpar::Matrix<double>(logl_vec, this->n_groups, this->n_obs);
	this->log_times_observed = { 7.681099,7.04316,6.849066,5.278115,5.164786,5.062595,6.947937,6.863803,7.277248,7.666222 };
	this->alpha0 = std::vector<double>(this->n_groups, 1.0);
	this->tol = 1e-8;
	this->max_iters = 5000;
    }
    // Result
    rcgpar::Matrix<double> expected;

    // Sizes
    uint16_t n_groups;
    uint32_t n_obs;

    // Params
    rcgpar::Matrix<double> logl;
    std::vector<double> log_times_observed;
    std::vector<double> alpha0;
    double tol;
    uint16_t max_iters;
};

#endif
