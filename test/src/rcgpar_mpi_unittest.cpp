// rcgpar: parallel estimation of mixture model components
// https://github.com/tmaklin/rcgpar
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
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
#include "rcgpar_unittest.hpp"

#include <mpi.h>

#include <fstream>

#include "rcgpar.hpp"
#include "openmp_config.hpp"

const uint16_t M_NUM_MAX_PROCESSES = 1024;

// Test rcg_optl_mat_mpi()
TEST_F(RcgOptlMatTest, FinalGammaZCorrect_MPI) {
    // Init MPI
    MPI_Init(NULL, NULL);
    int ntasks,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if defined(RCGPAR_OPENMP_SUPPORT) && (RCGPAR_OPENMP_SUPPORT) == 1
    omp_set_num_threads(2);
#endif

    // Estimate gamma_Z
    rcgpar::Matrix<double> my_logl(logl);
    std::ofstream empty;
    got = rcg_optl_mpi(my_logl, log_times_observed, alpha0, tol, max_iters, empty);

    // Construct gamma_Z from the partials
    rcgpar::Matrix<double> got_all = rcgpar::Matrix<double>(n_groups, n_obs, 0.0);

    uint32_t n_obs_per_task = std::floor(n_obs/ntasks);
    if (rank == (ntasks - 1)) {
	n_obs_per_task += n_obs - n_obs_per_task*ntasks;
    }
    std::cerr << n_obs_per_task << std::endl;
    int displs[M_NUM_MAX_PROCESSES];
    int recvcounts[M_NUM_MAX_PROCESSES] = { 0 };
    int recvsum = 0;
    for (uint16_t i = 0; i < ntasks - 1; ++i) {
	displs[i] = recvsum;
	recvcounts[i] = std::floor(n_obs/ntasks);
	recvsum += recvcounts[i];
    }
    displs[ntasks - 1] = recvsum;
    recvcounts[ntasks - 1] = std::floor(n_obs/ntasks) + n_obs - ntasks*std::floor(n_obs/ntasks);

    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Gatherv(&got.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, &got_all.front() + i*n_obs, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank == (ntasks - 1)) {
	    std::cerr << i*n_obs_per_task << ',';
	    std::cerr << i*n_obs << ',';
	    std::cerr << n_obs_per_task << std::endl;
	}
    }
    MPI_Bcast(&got_all.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    EXPECT_EQ(final_gamma_Z, got_all);
    MPI_Finalize();
}
