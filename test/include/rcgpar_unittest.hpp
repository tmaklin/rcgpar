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

#include <cstddef>

#include "gtest/gtest.h"

#include "Matrix.hpp"
#include "util_unittest.hpp"
#include "rcg_unittest.hpp"

class RcgOptlMatTest : public ::testing::Test, protected FinalGammaZTest, protected LogLikelihoodTest, protected LogCountsTest, protected Alpha0Test {
protected:
    void SetUp() override {
	n_groups = logl.get_rows();
	n_obs = logl.get_cols();
    }
    void TearDown() override {
	got = rcgpar::Matrix<double>();
	n_groups = 0;
	n_obs = 0;
    }

    // Params
    const double tol = 1e-8;
    const uint32_t max_iters = 5000;

    // Inputs
    uint16_t n_groups;
    uint32_t n_obs;

    // Test output
    rcgpar::Matrix<double> got;
};

#endif
