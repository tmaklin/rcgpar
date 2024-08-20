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
