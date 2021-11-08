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

#include "openmp_config.hpp"

TEST_F(Rcgpar, rcg_optl_mpi) {
    // Init MPI
    MPI_Init(NULL, NULL);
    int ntasks,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if defined(RCGPAR_OPENMP_SUPPORT) && (RCGPAR_OPENMP_SUPPORT) == 1
    omp_set_num_threads(2);
#endif
    // Estimate gamma_Z
    rcgpar::Matrix<double> got_partial = rcg_optl_mpi(logl, log_times_observed, alpha0, tol, max_iters);

    // Construct gamma_Z from the partials
    rcgpar::Matrix<double> got = rcgpar::Matrix<double>(n_groups, n_obs, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	uint32_t n_obs_per_task = std::floor(n_obs/ntasks);
	if (rank == ntasks) {
	    n_obs_per_task += n_obs - n_obs_per_task*ntasks;
	}
	MPI_Gather(&got_partial.front() + i*n_obs/ntasks, n_obs/ntasks, MPI_DOUBLE, &got.front() + i*n_obs, n_obs/ntasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&got.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    EXPECT_EQ(expected_gamma_Z, got);
    MPI_Finalize();
}
