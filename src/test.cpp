#include "rcg.hpp"

#include <vector>
#include <cmath>
#include <iostream>

#include "version.h"
#include "Matrix.hpp"

std::vector<double> mixture_components(const Matrix<double> &probs, const std::vector<double> &log_times_observed, const uint32_t n_times_total) {
  std::vector<double> thetas(probs.get_rows(), 0.0);
  for (uint32_t i = 0; i < probs.get_rows(); ++i) {
    for (uint32_t j = 0; j < probs.get_cols(); ++j) {
      thetas[i] += std::exp(probs(i, j) + log_times_observed[j]);
    }
    thetas[i] /= n_times_total;
  }
  return thetas;
}

int main() {
    std::cerr << "rcg-MPI-" << RCGMPI_BUILD_VERSION << std::endl;

#if defined(RCGMPI_OPENMP_SUPPORT) && (RCGMPI_OPENMP_SUPPORT) == 1
  omp_set_num_threads(4);
#endif

    uint32_t n_rows = 10000;
    uint16_t n_cols = 50;
    Matrix<double> invals(n_cols, n_rows, 0.0);
    for (uint16_t i = 0; i < n_cols; ++i) {
	for (uint32_t j = 0; j < n_rows; ++j) {
	    invals(i, j) = (double)(i + j)/(n_rows + n_cols);
	}
    }

    std::vector<double> log_times_observed(n_rows, 0.0);
    std::vector<double> alpha0(n_cols, 1.0);
    double tol = 1e-8;
    uint16_t max_iters = 5000;

    const Matrix<double> &res = rcg_optl_mat(invals, log_times_observed, alpha0, tol, max_iters);
    const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_rows);

    for (uint16_t i = 0; i < n_cols; ++i) {
	std::cerr << i << '\t' << thetas.at(i) << '\n';
    }
    std::cerr << std::endl;

    return 0;
}
