#include "rcg_gpu.hpp"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

namespace rcgpar {
void logsumexp(torch::Tensor &gamma_Z, torch::Tensor &m) {
    m = torch::logsumexp(gamma_Z, 0);
    gamma_Z -= m;
}

torch::Tensor mixt_negnatgrad(torch::Tensor &gamma_Z, torch::Tensor &N_k, torch::Tensor &logl, torch::Tensor &dL_dphi) {
    torch::Tensor digamma_N_k = torch::digamma(N_k) - 1.0;
    dL_dphi.copy_((logl + digamma_N_k.unsqueeze(1)).sub_(gamma_Z));
    torch::Tensor temp = torch::exp(gamma_Z).mul_(dL_dphi);
    torch::Tensor colsums = torch::sum(temp, 0);
    torch::Tensor newnorm = torch::sum(temp.mul_(dL_dphi - colsums));
    return newnorm;
}

torch::Tensor update_N_k(torch::Tensor &gamma_Z, torch::Tensor &log_times_observed, torch::Tensor &alpha0) {
    torch::Tensor N_k = torch::sum((gamma_Z + log_times_observed).exp_(), 1);
    N_k += alpha0;
    return N_k;
}

torch::Tensor ELBO_rcg_mat(torch::Tensor &logl, torch::Tensor &gamma_Z, torch::Tensor &counts, torch::Tensor &N_k, torch::Tensor &bound_const) {
    torch::Tensor bound = torch::sum((gamma_Z + counts).exp_().mul_(logl - gamma_Z));
    bound += torch::sum(torch::lgamma(N_k));
    bound += bound_const;
    return bound;
}

torch::Tensor calc_bound_const(torch::Tensor &log_times_observed, torch::Tensor &alpha0) {
    torch::Tensor counts_sum = torch::sum(torch::exp(log_times_observed));
    torch::Tensor alpha0_sum = torch::sum(alpha0);
    torch::Tensor lgamma_alpha0_sum = torch::sum(torch::lgamma(alpha0));
    torch::Tensor bound_const = torch::lgamma(alpha0_sum) - torch::lgamma(alpha0_sum + counts_sum) - lgamma_alpha0_sum;
    return bound_const;
}

void rcg_optl_mat_gpu(torch::Tensor &logl, torch::Tensor &log_times_observed, torch::Tensor &alpha0, double tol, uint16_t max_iters, torch::Tensor &gamma_Z, torch::TensorOptions options, std::ostream &log) {

    int n_obs = logl.size(1);
    
    torch::Tensor step = torch::zeros_like(logl, options);
    torch::Tensor oldstep = torch::zeros_like(logl, options);
    torch::Tensor oldm = torch::zeros({n_obs}, options);
    torch::Tensor oldnorm = torch::tensor(1.0, options);
    torch::Tensor bound = torch::tensor(-100000.0, options);
    torch::Tensor oldbound = torch::tensor(-100000.0, options);

    torch::Tensor bound_const = calc_bound_const(log_times_observed, alpha0);
    torch::Tensor N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

    bool didreset = false;
    for (uint16_t k = 0; k < max_iters; ++k) {
        torch::Tensor newnorm = mixt_negnatgrad(gamma_Z, N_k, logl, step);
        torch::Tensor beta_FR = newnorm / oldnorm;
        oldnorm.copy_(newnorm);

        if (didreset) {
            oldstep.mul_(0.0);
        } else if (beta_FR.item<double>() > 0) {
            oldstep.mul_(beta_FR);
            step.add_(oldstep);
        }
        didreset = false;

        gamma_Z.add_(step);

        logsumexp(gamma_Z, oldm);
        N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

        oldbound.copy_(bound);
        bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const);

        if (bound.item<double>() < oldbound.item<double>()) {
            didreset = true;
            gamma_Z.add_(oldm); // revert step
            if (beta_FR.item<double>() > 0) {
                gamma_Z.sub_(oldstep);
            }

            logsumexp(gamma_Z, oldm);
            N_k = update_N_k(gamma_Z, log_times_observed, alpha0);

            bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const);
        } else {
            oldstep.copy_(step);
        }

        if (k % 5 == 0) {
            log << "iter: " << k << ", bound: " << bound.item<double>() << ", |g|: " << newnorm.item<double>() << '\n';
        }

        if (bound.item<double>() - oldbound.item<double>() < tol && !didreset) {
            logsumexp(gamma_Z, oldm);
            log << std::endl;
	        return;
        }

        if (newnorm.item<double>() < 0) {
            tol *= 10;
        }
    }

    logsumexp(gamma_Z, oldm);
    log << std::endl;
	return;
}
} // namespace rcgpar