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
#ifndef RCGPAR_UNITTEST_COMMON_INPUTS_HPP
#define RCGPAR_UNITTEST_COMMON_INPUTS_HPP

#include <cstddef>
#include <vector>
#include <initializer_list>
#include <algorithm>

#include "gtest/gtest.h"

#include "Matrix.hpp"

// Constant input data that is reused in several tests

const uint16_t TEST_N_GROUPS = 4;
const uint32_t TEST_N_OBS = 10;

class LogLikelihoodTest {
protected:
    static const rcgpar::Matrix<double> logl;
};

class LogCountsTest {
protected:
    static const std::vector<double> log_times_observed;
};

class GammaZTest {
protected:
    // Intermediate values from iteration k == 3
    static const rcgpar::Matrix<double> gamma_Z;
};

class ExpectedStepTest {
protected:
    static const rcgpar::Matrix<double> expected_step;
};

class ExpectedGammaZTest {
protected:
    static const rcgpar::Matrix<double> expected_gamma_Z;
};

class ExpectedOldMTest {
protected:
    static const std::vector<double> expected_oldm;
};

class Alpha0Test {
protected:
    static const std::vector<double> alpha0;
};

class FinalGammaZTest {
protected:
    static const rcgpar::Matrix<double> final_gamma_Z;
};

#endif
