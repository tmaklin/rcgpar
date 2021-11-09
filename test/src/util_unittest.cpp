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
#include "util_unittest.hpp"

#include <numeric>
#include <cmath>
#include <iostream>

#include "util.hpp"

TEST_F(MixtureComponentsTest, ThetasCorrect) {
    // Transform probs into thetas
    got = rcgpar::mixture_components(final_gamma_Z, log_times_observed);

    for (uint16_t i = 0; i < got.size(); ++i) {
	SCOPED_TRACE(i);
	EXPECT_NEAR(expected_thetas[i], got[i], 1e-4);
    }
}
