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
#include "rcgpar.hpp"

// Riemannian conjugate gradient for parameter estimation.
//
// This file contains the rcg_optl_mat function for MPI calls.

#include <mpi.h>

#include <cmath>
#include <iostream>

#include "rcg.hpp"
#include "MpiHandler.hpp"

namespace rcgpar {
Matrix<double> rcg_optl_mpi(const Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    // Input data dimensions
    const uint16_t n_groups = alpha0.size();
    uint32_t n_obs = log_times_observed_full.size();

    // MPI handler
    MpiHandler handler;
    const int rank = handler.get_rank();

    // Initialize variables for MPI
    MPI_Bcast(&n_obs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    handler.initialize(n_obs);
    const uint32_t n_obs_per_task = handler.obs_per_task(n_obs);
    const int* displs = handler.get_displacements();
    const int* sendcounts = handler.get_bufcounts();

    std::vector<double> log_times_observed(n_obs_per_task);
    MPI_Scatterv(&log_times_observed_full.front(), sendcounts, displs, MPI_DOUBLE, &log_times_observed.front(), n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // log likelihoods
    Matrix<double> logl_partial(n_groups, n_obs_per_task, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Scatterv(&logl_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, &logl_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Initialize partial gamma_Z
    Matrix<double> gamma_Z_partial = Matrix<double>(n_groups, n_obs_per_task, std::log(1.0/(double)n_groups));

    // ELBO constant
    double bound_const = 0.0;
    if (rank == 0) {
	bound_const = calc_bound_const(log_times_observed_full, alpha0);
    }
    MPI_Bcast(&bound_const, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Estimate gamma_Z partials
    rcg_optl_mat(logl_partial, log_times_observed, alpha0,
		 bound_const, tol, max_iters,
		 true, gamma_Z_partial, log);

    // Construct gamma_Z from the partials
    Matrix<double> gamma_Z_full(n_groups, n_obs, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Gatherv(&gamma_Z_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, &gamma_Z_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&gamma_Z_full.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return(gamma_Z_full);
}
}
