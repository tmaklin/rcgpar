#include <vector>
#include <iostream>

#include "Matrix.hpp"
#include "rcg.hpp"
#include "test_util.hpp"

#include "version.h"
#include "openmp_config.hpp"

int main() {
    std::cerr << "rcg-OMP-" << RCGMPI_BUILD_VERSION << std::endl;
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

    const Matrix<double> &res = rcg_optl_mat(log_lls, log_times_observed, alpha0, tol, max_iters);
    const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

    for (uint16_t i = 0; i < n_cols; ++i) {
	std::cerr << i << '\t' << thetas.at(i) << '\n';
    }
    std::cerr << std::endl;

    return 0;
}
