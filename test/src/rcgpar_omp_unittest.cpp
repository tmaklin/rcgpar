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
#include "rcgpar_unittest.hpp"

#include <fstream>

#include "rcgpar.hpp"
#include "openmp_config.hpp"

// Test rcg_otpl_mat_omp()
TEST_F(RcgOptlTest, FinalGammaZCorrect_OMP) {
#if defined(RCGPAR_OPENMP_SUPPORT) && (RCGPAR_OPENMP_SUPPORT) == 1
    omp_set_num_threads(2);
#endif
    // Estimate gamma_Z
    std::ofstream empty;
    got = rcg_optl_omp(logl, log_times_observed, alpha0, tol, max_iters, empty);

    EXPECT_EQ(final_gamma_Z, got);
}
