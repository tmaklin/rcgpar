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
#ifndef RCGPAR_TEST_UTIL_HPP
#define RCGPAR_TEST_UTIL_HPP

#include <vector>

#include "Matrix.hpp"

std::vector<double> mixture_components(const rcgpar::Matrix<double> &probs,
				       const std::vector<double> &log_times_observed,
				       const uint32_t n_times_total);

void read_test_data(rcgpar::Matrix<double> &log_lls,
		    std::vector<double> &log_times_observed, uint32_t &n_times_total);

#endif
