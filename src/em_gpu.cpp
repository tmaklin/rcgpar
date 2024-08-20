#include "em_gpu.hpp"

#include <fstream>
#include <torch/torch.h>

namespace rcgpar {
torch::Tensor em_algorithm(torch::Tensor &log_likelihoods, torch::Tensor &loglik_counts_tensor, double threshold, uint16_t max_iters, std::ostream &log, torch::ScalarType dtype) {

    torch::Device device = log_likelihoods.device();

    int num_rows = log_likelihoods.size(0);
    int num_cols = log_likelihoods.size(1);

    // pre allocate log_weighted_likelihoods
    torch::Tensor log_weighted_likelihoods = torch::empty({num_rows, num_cols}, dtype).to(device);

    torch::Tensor theta = torch::ones({num_cols}, dtype).to(device) / num_cols;
    torch::Tensor prev_loss = torch::tensor(1000000, dtype).to(device);
    torch::Tensor threshold_tensor = torch::tensor(threshold, dtype).to(device);

    for (uint16_t iteration = 0; iteration < max_iters; ++iteration) {
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
