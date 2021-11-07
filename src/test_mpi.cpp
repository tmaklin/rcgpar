#include <mpi.h>

#include <vector>
#include <iostream>

#include "Matrix.hpp"
#include "rcg.hpp"
#include "test_util.hpp"

#include "version.h"
#include "openmp_config.hpp"

int main(int argc, char* argv[]) {
#if defined(RCGPAR_OPENMP_SUPPORT) && (RCGPAR_OPENMP_SUPPORT) == 1
    omp_set_num_threads(2);
#endif

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
    uint16_t n_groups;
    uint64_t n_obs;

    if (rank == 0) {
	// Only root reads in data.
	std::cerr << "rcgpar-MPI-test" << RCGPAR_BUILD_VERSION << std::endl;
	read_test_data(log_lls, log_times_observed, n_times_total);
	n_groups = log_lls.get_rows();
	n_obs = log_times_observed.size();
    }
    MPI_Bcast(&n_times_total, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_groups, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_obs, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Hyperparams
    std::vector<double> alpha0(n_groups, 1.0);

    // Optimize with rcg MPI call
    const Matrix<double> &res_partial = rcg_optl_mpi(log_lls, log_times_observed, alpha0, tol, max_iters);

    Matrix<double> res;
    if (rank == 0) {
	res = Matrix<double>(n_groups, n_obs, std::log(1.0/(double)n_groups)); // where gamma_Z is init at 1.0
    }

    // Construct gamma_Z from the partials
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Gather(&res_partial.front() + i*n_obs/ntasks, n_obs/ntasks, MPI_DOUBLE, &res.front() + i*n_obs, n_obs/ntasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    rc = MPI_Finalize();
    if (rank == 0) {
	const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

	for (uint16_t i = 0; i < n_groups; ++i) {
	    std::cerr << i << '\t' << thetas.at(i) << '\n';
	}
	std::cerr << std::endl;
    }
    return 0;
}
