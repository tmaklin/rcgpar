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

#include "rcgpar_mpi_config.hpp"

#include <exception>
#include <string>
#include <cmath>
#include <iostream>

#include <vector>
#include <torch/torch.h>

#include "rcg.hpp"
#include "rcg_gpu.hpp"

#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
#include "MpiHandler.hpp"
#endif

namespace rcgpar {
void check_input(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters) {
    uint16_t n_groups = logl.get_rows();
    uint32_t n_obs = logl.get_cols();
    if (tol < 0) {
	throw std::invalid_argument("Tolerance cannot be negative (type: double).");
    }
    if (max_iters == 0) {
	throw std::invalid_argument("Max iters cannot be 0 (type: uint16_t).");
    }
    if (n_groups != alpha0.size()) {
	throw std::domain_error("Number of components (rows) in logl differs from the number of values in alpha0.");
    }
    if (n_obs != log_times_observed.size()) {
	throw std::domain_error("Number of observations (columns) in logl differs from the number of observations in log_times_observed.");
    }
}

#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
void check_mpi(const MpiHandler &handler) {
    if (handler.get_status() != MPI_SUCCESS) {
	int len,eclass;
	char mpi_error[MPI_MAX_ERROR_STRING];
	MPI_Error_class(handler.get_status(), &eclass);
	MPI_Error_string(handler.get_status(), mpi_error, &len);
	throw std::runtime_error("MPI returned an error: " + std::string(mpi_error));
    }
    if (handler.get_n_tasks() > RCGPAR_MPI_MAX_PROCESSES) {
	std::string msg("Number of MPI tasks is greater than the allowed " + std::to_string(RCGPAR_MPI_MAX_PROCESSES) +
				 ".\nRecompile rcgpar with '-DCMAKE_MPI_MAX_PROCESSES=<big number>' to support more tasks.");
	throw std::runtime_error(msg);
    }
}
#endif

seamat::DenseMatrix<double> rcg_optl_omp(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    // Validate input data
    check_input(logl, log_times_observed, alpha0, tol, max_iters);

    // where gamma_Z is init at 1.0
    seamat::DenseMatrix<double> gamma_Z(logl.get_rows(), log_times_observed.size(), std::log(1.0/(double)logl.get_rows()));
    double bound_const = calc_bound_const(log_times_observed, alpha0);

    // Estimate gamma_Z
    rcg_optl_mat(logl, log_times_observed, alpha0,
		 bound_const, tol, max_iters,
		 false, gamma_Z, log);

    return(gamma_Z);
}

seamat::DenseMatrix<double> rcg_optl_gpu(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    // Validate input data
    check_input(logl, log_times_observed, alpha0, tol, max_iters);

    uint16_t n_groups = alpha0.size();
    uint32_t n_obs = log_times_observed.size();

    std::vector<double> logl_vec = logl.get_data();

    // Choose the device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device == torch::kCUDA) {
        log << "Using GPU" << '\n';
    }
    torch::Dtype precision = torch::kFloat64;
    torch::TensorOptions options(precision);
    options = options.device(device);

    torch::Tensor logl_ten = torch::from_blob((double*)logl_vec.data(), {n_groups, n_obs}, precision).clone().to(device);
    torch::Tensor log_times_observed_ten = torch::from_blob((double*)log_times_observed.data(), {n_obs}, precision).clone().to(device);
    torch::Tensor alpha0_ten = torch::from_blob((double*)alpha0.data(), {n_groups}, precision).clone().to(device);

    // where gamma_Z is init at 1.0
    torch::Tensor gamma_Z = torch::full({n_groups, n_obs}, std::log(1.0 / n_groups), options);

    // Estimate gamma_Z
    rcg_optl_mat_gpu(logl_ten, log_times_observed_ten, alpha0_ten, tol, max_iters, gamma_Z, options, log);

    gamma_Z = gamma_Z.to(torch::kCPU);

    // Convert gamma_Z to a DenseMatrix
    std::vector<double> gamma_Z_vec(gamma_Z.data_ptr<double>(), gamma_Z.data_ptr<double>() + gamma_Z.numel());
    seamat::DenseMatrix<double> gamma_Z_mat(gamma_Z_vec, n_groups, n_obs);

    return(gamma_Z_mat);
}

#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
seamat::DenseMatrix<double> rcg_optl_mpi(const seamat::Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    // Input data dimensions
    const uint16_t n_groups = alpha0.size();
    uint32_t n_obs = log_times_observed_full.size();

    // MPI handler
    MpiHandler handler;
    const int rank = handler.get_rank();

    // Validate input data on root process and abort if incorrect
    if (rank == 0) {
      check_input(logl_full, log_times_observed_full, alpha0, tol, max_iters);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Check that MPI is running and setup correctly
    check_mpi(handler);

    // Initialize variables for MPI
    MPI_Bcast(&n_obs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    handler.initialize(n_obs);
    const uint32_t n_obs_per_task = handler.obs_per_task(n_obs);
    const int* displs = handler.get_displacements();
    const int* sendcounts = handler.get_bufcounts();

    std::vector<double> log_times_observed(n_obs_per_task);
    MPI_Scatterv(&log_times_observed_full.front(), sendcounts, displs, MPI_DOUBLE, &log_times_observed.front(), n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // log likelihoods
    seamat::DenseMatrix<double> logl_partial(n_groups, n_obs_per_task, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Scatterv(&logl_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, &logl_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Validate input data again on all processes to check that the scatters went through OK.
    check_input(logl_partial, log_times_observed, alpha0, tol, max_iters);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize partial gamma_Z
    seamat::DenseMatrix<double> gamma_Z_partial = seamat::DenseMatrix<double>(n_groups, n_obs_per_task, std::log(1.0/(double)n_groups));

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
    seamat::DenseMatrix<double> gamma_Z_full(n_groups, n_obs, 0.0);
    for (uint16_t i = 0; i < n_groups; ++i) {
	MPI_Gatherv(&gamma_Z_partial.front() + i*n_obs_per_task, n_obs_per_task, MPI_DOUBLE, &gamma_Z_full.front() + i*n_obs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPIX_Bcast_x(&gamma_Z_full.front(), n_groups*n_obs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return(gamma_Z_full);
}
#endif
}
