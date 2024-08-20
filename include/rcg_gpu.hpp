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
