#include "rcg.hpp"

#include <vector>
#include <cmath>
#include <iostream>

#include <fstream>
#include <cmath>
#include <sstream>

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

void read_test_data(Matrix<double> &log_lls, std::vector<double> &log_times_observed, uint32_t &n_times_total) {
    std::ifstream stream("../test_times_observed.txt");
    std::string line;
    while(std::getline(stream, line)) {
	uint32_t obs_count = std::stoul(line);
	log_times_observed.emplace_back(std::log(obs_count));
	n_times_total += obs_count;
    }
    std::ifstream stream2("../test_likelihoods.tsv");
    log_lls = Matrix<double>(4, 78, 0.0);
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

int main() {
    std::cerr << "rcg-MPI-" << RCGMPI_BUILD_VERSION << std::endl;

#if defined(RCGMPI_OPENMP_SUPPORT) && (RCGMPI_OPENMP_SUPPORT) == 1
    omp_set_num_threads(4);
#endif


    uint16_t n_rows = 78;
    uint16_t n_cols = 4;
    uint32_t n_times_total = 0;

    std::vector<double> log_times_observed;
    Matrix<double> log_lls;
    read_test_data(log_lls, log_times_observed, n_times_total);

    std::vector<double> alpha0(n_cols, 1.0);
    double tol = 1e-8;
    uint16_t max_iters = 5000;
    std::cerr << log_times_observed.size() << std::endl;
    std::cerr << log_lls.get_rows() << 'x' << log_lls.get_cols() << std::endl;

    const Matrix<double> &res = rcg_optl_mat(log_lls, log_times_observed, alpha0, tol, max_iters);
    const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

    for (uint16_t i = 0; i < n_cols; ++i) {
	std::cerr << i << '\t' << thetas.at(i) << '\n';
    }
    std::cerr << std::endl;

    return 0;
}
