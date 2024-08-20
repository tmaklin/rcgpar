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
#include "em_gpu.hpp"

#include <fstream>
#include <torch/torch.h>

namespace rcgpar {
torch::Tensor em_algorithm(torch::Tensor &log_likelihoods, torch::Tensor &loglik_counts_tensor, double threshold, size_t max_iters, std::ostream &log, torch::ScalarType dtype) {

    torch::Device device = log_likelihoods.device();

    int64_t num_rows = log_likelihoods.size(0);
    int64_t num_cols = log_likelihoods.size(1);

    // pre allocate log_weighted_likelihoods
    torch::Tensor log_weighted_likelihoods = torch::empty({num_rows, num_cols}, dtype).to(device);

    torch::Tensor theta = torch::ones({num_cols}, dtype).to(device) / num_cols;
    torch::Tensor prev_loss = torch::tensor(1000000, dtype).to(device);
    torch::Tensor threshold_tensor = torch::tensor(threshold, dtype).to(device);

    for (size_t iteration = 0; iteration < max_iters; ++iteration) {
        // E-step: Compute responsibilities
        log_weighted_likelihoods.copy_(log_likelihoods + torch::log(theta));
        torch::Tensor log_sum_exp = torch::logsumexp(log_weighted_likelihoods, 1, true);
        //torch::Tensor log_responsibilities = log_weighted_likelihoods.sub_(log_sum_exp);
        log_weighted_likelihoods.sub_(log_sum_exp);

        // M-step: Update theta values weighted by log counts
        //torch::Tensor log_weighted_responsibilities = log_responsibilities.add_(loglik_counts_tensor.unsqueeze(1));
        log_weighted_likelihoods.add_(loglik_counts_tensor.unsqueeze(1));
        //log_weighted_responsibilities.exp_();
        log_weighted_likelihoods.exp_();
        torch::Tensor new_theta = log_weighted_likelihoods.sum(0) / torch::exp(loglik_counts_tensor).sum();

        // Compute the log likelihood
        torch::Tensor log_likelihood = torch::sum((log_sum_exp.add_(loglik_counts_tensor.unsqueeze(1))).exp_());

        // Check for convergence
        torch::Tensor loss = -log_likelihood;
        if (torch::abs(prev_loss - loss).lt(threshold_tensor).item<bool>()) {
            break;
        }
        prev_loss.copy_(loss);
        
        if (iteration % 5 == 0) {
            log << "iter: " << iteration << " loss: " << loss.item<double>() << '\n';
        }
        
        // Update theta
        theta.copy_(new_theta);
    }

    log_weighted_likelihoods.copy_(log_likelihoods + torch::log(theta));
    torch::Tensor log_sum_exp = torch::logsumexp(log_weighted_likelihoods, 1, true);
    log_weighted_likelihoods.sub_(log_sum_exp);
    return log_weighted_likelihoods;
}
} // namespace rcgpar
