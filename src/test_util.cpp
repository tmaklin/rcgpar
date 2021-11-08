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
#include "test_util.hpp"

#include <fstream>
#include <string>
#include <sstream>

std::vector<double> mixture_components(const rcgpar::Matrix<double> &probs, const std::vector<double> &log_times_observed, const uint32_t n_times_total) {
  std::vector<double> thetas(probs.get_rows(), 0.0);
  for (uint32_t i = 0; i < probs.get_rows(); ++i) {
    for (uint32_t j = 0; j < probs.get_cols(); ++j) {
      thetas[i] += std::exp(probs(i, j) + log_times_observed[j]);
    }
    thetas[i] /= n_times_total;
  }
  return thetas;
}

void read_test_data(rcgpar::Matrix<double> &log_lls, std::vector<double> &log_times_observed, uint32_t &n_times_total) {
    std::ifstream stream("../test_times_observed.txt");
    std::string line;
    while(std::getline(stream, line)) {
	uint32_t obs_count = std::stoul(line);
	log_times_observed.emplace_back(std::log(obs_count));
	n_times_total += obs_count;
    }
    std::ifstream stream2("../test_likelihoods.tsv");
    log_lls = rcgpar::Matrix<double>(4, 78, 0.0);
    uint16_t i = 0;
    while(std::getline(stream2, line)) {
	uint16_t j = 0;
	std::string part;
	std::stringstream parts(line);
	while(std::getline(parts, part, '\t')) {
	    log_lls(j, i) = std::stod(part);
	    ++j;
	}
	++i;
    }
}

