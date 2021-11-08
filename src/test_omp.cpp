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
#include <vector>
#include <iostream>

#include "Matrix.hpp"
#include "rcgpar.hpp"
#include "test_util.hpp"

#include "version.h"
#include "openmp_config.hpp"

int main() {
    std::cerr << "rcgpar-OpenMP-test" << RCGPAR_BUILD_VERSION << std::endl;
#if defined(RCGPAR_OPENMP_SUPPORT) && (RCGPAR_OPENMP_SUPPORT) == 1
    omp_set_num_threads(4);
#endif
    uint16_t n_groups = 4;
    uint32_t n_times_total = 0;

    std::vector<double> log_times_observed;
    rcgpar::Matrix<double> log_lls;
    read_test_data(log_lls, log_times_observed, n_times_total);

    std::vector<double> alpha0(n_groups, 1.0);
    double tol = 1e-8;
    uint16_t max_iters = 5000;

    const rcgpar::Matrix<double> &res = rcgpar::rcg_optl_omp(log_lls, log_times_observed, alpha0, tol, max_iters);
    const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

    for (uint16_t i = 0; i < n_groups; ++i) {
	std::cerr << i << '\t' << thetas.at(i) << '\n';
    }
    std::cerr << std::endl;

    return 0;
}
