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

    int ntasks,rank;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix<double> log_lls;
    std::vector<double> log_times_observed;
    uint16_t n_rows = 78;
    uint16_t n_cols = 4;
    uint32_t n_times_total = 0;
    uint64_t n_items;

    if (rank == 0) {
	std::cerr << "rcg-MPI" << RCGMPI_BUILD_VERSION << std::endl;
	read_test_data(log_lls, log_times_observed, n_times_total);
	n_items = log_lls.get_rows()*log_lls.get_cols();
    }
    if (rank > 0) {
	log_times_observed.resize(n_rows);
    }
    std::vector<double> alpha0(n_cols, 1.0);
    double tol = 1e-8;
    uint16_t max_iters = 5000;

    MPI_Bcast(&n_items, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_times_total, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iters, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&log_times_observed.front(), n_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int nItems = 3;
    const int blocklengths[nItems] = {n_items, 1, 1};
    MPI_Datatype types[nItems] = {MPI_DOUBLE, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Datatype MPI_MatrixDouble_Type;
    MPI_Aint offsets[nItems];
    offsets[0] = Matrix<double>::mat_offset();
    offsets[1] = Matrix<double>::rows_offset();
    offsets[2] = Matrix<double>::cols_offset();

    MPI_Type_create_struct(nItems, blocklengths, offsets, types, &MPI_MatrixDouble_Type);
    MPI_Type_commit(&MPI_MatrixDouble_Type);

    if (rank > 0) {
	log_lls.resize(n_cols, n_rows, 0.0);
    }

    MPI_Bcast(&log_lls.front(), 1, MPI_MatrixDouble_Type, 0, MPI_COMM_WORLD);

    const Matrix<double> &res = rcg_optl_mpi(log_lls, log_times_observed, alpha0, tol, max_iters);
    const std::vector<double> &thetas = mixture_components(res, log_times_observed, n_times_total);

    for (uint16_t i = 0; i < n_cols; ++i) {
	std::cerr << i << '\t' << thetas.at(i) << '\n';
    }
    std::cerr << std::endl;

    std::cerr << "Starting process " << rank + 1 << "/" << ntasks << "." << std::endl;
    rc = MPI_Finalize();
    return 0;
}
