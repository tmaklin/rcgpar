// rcgpar: parallel estimation of mixture model components
// https://github.com/tmaklin/rcgpar
//
// Copyright (C) 2024 rcgpar contributors
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
#ifndef RCGPAR_EM_GPU_HPP
#define RCGPAR_EM_GPU_HPP

#include "em_gpu.hpp"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

namespace rcgpar {

torch::Tensor em_algorithm(torch::Tensor &log_likelihoods, torch::Tensor &loglik_counts_tensor, double threshold, size_t max_iters, std::ostream &log, torch::ScalarType dtype);

} // namespace rcgpar

#endif
