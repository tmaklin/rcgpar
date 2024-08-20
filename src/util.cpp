// rcgpar: parallel estimation of mixture model components
// https://github.com/tmaklin/rcgpar
//
// Copyright (C) 2024 rcgpar contributors (tommi@maklin.fi)
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
#include "rcgpar.hpp"

#include "rcgpar_torch_config.hpp"

#include <vector>

namespace rcgpar {
std::vector<double> mixture_components(const seamat::Matrix<double> &probs, const std::vector<double> &log_times_observed) {
  std::vector<double> thetas(probs.get_rows(), 0.0);
  for (size_t i = 0; i < probs.get_rows(); ++i) {
    size_t n_times_total = 0.0;
    for (size_t j = 0; j < probs.get_cols(); ++j) {
      n_times_total += std::exp(log_times_observed[j]);
      thetas[i] += std::exp(probs(i, j) + log_times_observed[j]);
    }
    thetas[i] /= n_times_total;
  }
  return thetas;
}

std::vector<double> mixture_components_torch(const seamat::Matrix<double> &probs, const std::vector<double> &log_times_observed) {
#if defined(RCGPAR_TORCH_SUPPORT) && (RCGPAR_TORCH_SUPPORT) == 1
  std::vector<double> probs_vec = probs.get_data();
  
  // Choose the device
  auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::Dtype precision = torch::kFloat64;

  int64_t rows = probs.get_rows();
  int64_t cols = probs.get_cols();
  
  torch::Tensor probs_ten = torch::from_blob((double*)probs_vec.data(), {rows, cols}, precision).clone().to(device);
  torch::Tensor log_times_observed_ten = torch::from_blob((double*)log_times_observed.data(), {cols}, precision).clone().to(device);
  torch::Tensor n_times_total = torch::sum(torch::exp(log_times_observed_ten));
  torch::Tensor thetas = torch::sum(torch::exp(probs_ten + log_times_observed_ten), 1) / n_times_total;
  thetas = thetas.to(torch::kCPU);
  std::vector<double> thetas_vec(thetas.data_ptr<double>(), thetas.data_ptr<double>() + thetas.numel());
  return thetas_vec;
#else
  throw std::runtime_error("rcgpar was not compiled with torch support.");
  return std::vector<double>();
#endif
}
}
