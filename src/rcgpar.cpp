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

#include <exception>
#include <string>
#include <cmath>
#include <iostream>

#include <vector>

#include "rcgpar_torch_config.hpp"

#include "rcg.hpp"

#if defined(RCGPAR_TORCH_SUPPORT) && (RCGPAR_TORCH_SUPPORT) == 1
#include "rcg_gpu.hpp"
#include "em_gpu.hpp"
#endif

namespace rcgpar {
void check_input(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, size_t max_iters) {
    size_t n_groups = logl.get_rows();
    size_t n_obs = logl.get_cols();
    if (tol < 0) {
	throw std::invalid_argument("Tolerance cannot be negative (type: double).");
    }
    if (max_iters == 0) {
	throw std::invalid_argument("Max iters cannot be 0 (type: size_t).");
    }
    if (n_groups != alpha0.size()) {
	throw std::domain_error("Number of components (rows) in logl differs from the number of values in alpha0.");
    }
    if (n_obs != log_times_observed.size()) {
	throw std::domain_error("Number of observations (columns) in logl differs from the number of observations in log_times_observed.");
    }
}

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

seamat::DenseMatrix<double> rcg_optl_mpi(const seamat::Matrix<double> &logl_full, const std::vector<double> &log_times_observed_full, const std::vector<double> &alpha0, const double &tol, uint16_t max_iters, std::ostream &log) {
    throw std::runtime_error("`rcg_optl_mpi` is no longer supported by rcgpar, switch to rcg_optl_omp or rcg_optl_torch instead. `rcg_optl_mpi`  will be removed in the next major release.");
}

seamat::DenseMatrix<double> rcg_optl_torch(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, size_t max_iters, std::ostream &log) {
#if defined(RCGPAR_TORCH_SUPPORT) && (RCGPAR_TORCH_SUPPORT) == 1
    // Validate input data
    check_input(logl, log_times_observed, alpha0, tol, max_iters);

    int64_t n_groups = alpha0.size();
    int64_t n_obs = log_times_observed.size();

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
#else
    throw std::runtime_error("rcgpar was not compiled with torch support.");
    return seamat::DenseMatrix<double>();
#endif
}

seamat::DenseMatrix<double> em_torch(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, const double &tol, size_t max_iters, std::ostream &log, std::string precision) {
#if defined(RCGPAR_TORCH_SUPPORT) && (RCGPAR_TORCH_SUPPORT) == 1
    // Validate input data
    check_input(logl, log_times_observed, alpha0, tol, max_iters);

    int64_t n_groups = alpha0.size();
    int64_t n_obs = log_times_observed.size();

    std::vector<double> logl_vec = logl.get_data();

    // Choose the device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device == torch::kCUDA) {
        log << "Using GPU" << '\n';
    }

    torch::Tensor gamma_Z;

    torch::ScalarType dtype;
    if (precision == "double") {
        dtype = torch::kFloat64;
        torch::Tensor logl_ten = torch::from_blob((double*)logl_vec.data(), {n_groups, n_obs}, dtype).clone().to(device).t().contiguous();
        torch::Tensor log_times_observed_ten = torch::from_blob((double*)log_times_observed.data(), {n_obs}, dtype).clone().to(device);

        gamma_Z = em_algorithm(logl_ten, log_times_observed_ten, tol, max_iters, log, dtype);
        gamma_Z = gamma_Z.to(torch::kCPU).t().contiguous();
        std::vector<double> gamma_Z_vec(gamma_Z.data_ptr<double>(), gamma_Z.data_ptr<double>() + gamma_Z.numel());
        seamat::DenseMatrix<double> gamma_Z_mat(gamma_Z_vec, n_groups, n_obs);
        return(gamma_Z_mat);
    } else {
        dtype = torch::kFloat32;
        std::vector<float> logl_vec_float(logl_vec.begin(), logl_vec.end());
        std::vector<float> log_times_observed_float(log_times_observed.begin(), log_times_observed.end());
        torch::Tensor logl_ten = torch::from_blob((float*)logl_vec_float.data(), {n_groups, n_obs}, dtype).clone().to(device).t().contiguous();
        torch::Tensor log_times_observed_ten = torch::from_blob((float*)log_times_observed_float.data(), {n_obs}, dtype).clone().to(device);

        gamma_Z = em_algorithm(logl_ten, log_times_observed_ten, tol, max_iters, log, dtype);
        gamma_Z = gamma_Z.to(torch::kCPU).t().contiguous();
        std::vector<double> gamma_Z_vec(gamma_Z.data_ptr<float>(), gamma_Z.data_ptr<float>() + gamma_Z.numel());
        seamat::DenseMatrix<double> gamma_Z_mat(gamma_Z_vec, n_groups, n_obs);
        return(gamma_Z_mat);
    }
#else
    throw std::runtime_error("rcgpar was not compiled with torch support.");
    return seamat::DenseMatrix<double>();
#endif
}
}
