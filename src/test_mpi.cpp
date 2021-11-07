#include <mpi.h>

#include <vector>
#include <iostream>

#include "Matrix.hpp"
#include "rcg.hpp"
#include "test_util.hpp"

#include "version.h"

int main(int argc, char* argv[]) {

    int rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
	std::cerr << "MPI initialization failed: " << rc << std::endl;
	return 1;
    }
    // "Command-line arguments"
    double tol = 1e-8;
    uint16_t max_iters = 5000;

    int ntasks,rank;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix<double> log_lls;
    std::vector<double> log_times_observed;
    uint32_t n_times_total = 0;

    // Dimensions of input data
    uint16_t n_cols;

    if (rank == 0) {
	// Only root reads in data.
	std::cerr << "rcg-MPI" << RCGMPI_BUILD_VERSION << std::endl;
	read_test_data(log_lls, log_times_observed, n_times_total);
	n_cols = log_lls.get_rows();
    }
    MPI_Bcast(&n_times_total, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_cols, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);

    // Hyperparams
    std::vector<double> alpha0(n_cols, 1.0);

    // Optimize with rcg MPI call
    const Matrix<double> &res = rcg_optl_mpi(log_lls, log_times_observed, alpha0, tol, max_iters);
    rc = MPI_Finalize();
    if (rank == 0) {
	const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

	for (uint16_t i = 0; i < n_cols; ++i) {
	    std::cerr << i << '\t' << thetas.at(i) << '\n';
	}
	std::cerr << std::endl;
    }
    return 0;
}
