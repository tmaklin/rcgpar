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
#ifndef RCGPAR_UTIL_UNITTEST_HPP
#define RCGPAR_UTIL_UNITTEST_HPP

#include <vector>
#include <initializer_list>

#include "gtest/gtest.h"

#include "Matrix.hpp"
#include "unittest_common_inputs.hpp"

class MixtureComponentsTest : public ::testing::Test, protected LogCountsTest, protected FinalGammaZTest {
protected:
    void SetUp() override {
	expected_thetas = { 0.999543, 0.00073079, 9.66135e-05, 0.000112505 };
    }
    void TearDown() override {
	expected_thetas.clear();
	expected_thetas.shrink_to_fit();
    }

    // Expecteds
    std::vector<double> expected_thetas;

    // Test output
    std::vector<double> got;
};

#endif
