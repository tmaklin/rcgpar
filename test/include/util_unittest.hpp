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

#include "rcg_unittest.hpp"
#include "Matrix.hpp"

class FinalGammaZTest {
protected:
    static const rcgpar::Matrix<double> final_gamma_Z;
};
const rcgpar::Matrix<double> FinalGammaZTest::final_gamma_Z(std::vector<double>(std::initializer_list<double>(
    { -0.0010899, -0.00104044, -0.000928571, -0.00104519, -0.000995734, -0.000883857, -0.000944069, -0.000894604, -0.000782716, -0.000853449,
      -7.15745,   -7.1574,     -7.15729,     -7.15741,    -7.15736,     -7.15725,     -7.15731,     -7.15726,     -7.15715,     -7.51888,
      -8.82298,   -8.82293,    -8.82282,     -9.1846,     -9.18455,     -9.18444,     -13.418,      -13.4179,     -13.4178,     -8.82274,
      -8.72199,   -9.0836,     -13.3169,     -8.72195,    -9.08356,     -13.3169,     -8.72184,     -9.08346,     -13.3168,     -8.72175 }
											 )), TEST_N_GROUPS, TEST_N_OBS);

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
