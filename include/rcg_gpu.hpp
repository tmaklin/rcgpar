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
#ifndef RCGPAR_RCG_GPU_HPP
#define RCGPAR_RCG_GPU_HPP

#include "rcg_gpu.hpp"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

namespace rcgpar {

void logsumexp(torch::Tensor &gamma_Z, torch::Tensor &m);

torch::Tensor mixt_negnatgrad(torch::Tensor &gamma_Z, torch::Tensor &N_k, torch::Tensor &logl, torch::Tensor &dL_dphi);

torch::Tensor update_N_k(torch::Tensor &gamma_Z, torch::Tensor &log_times_observed, torch::Tensor &alpha0);

torch::Tensor ELBO_rcg_mat(torch::Tensor &logl, torch::Tensor &gamma_Z, torch::Tensor &counts, torch::Tensor &N_k, torch::Tensor &bound_const);

torch::Tensor calc_bound_const(torch::Tensor &log_times_observed, torch::Tensor &alpha0);

void rcg_optl_mat_gpu(torch::Tensor &logl, torch::Tensor &log_times_observed, torch::Tensor &alpha0, double tol, size_t max_iters, torch::Tensor &gamma_Z, torch::TensorOptions options, std::ostream &log);

} // namespace rcgpar

#endif
