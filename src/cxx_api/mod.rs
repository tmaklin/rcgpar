// rcgpar: Riemannian conjugate gradient descent for estimating mixture model weights.
//
// Copyright 2025 rcgpar contributors [https://github.com/tmaklin/rcgpar]
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

//! C++ API for compatibility with rcgpar v1

use burn::backend::ndarray::NdArray;
use burn_tensor::{Shape, Tensor};

#[cxx::bridge(namespace = "rcgpar")]
mod ffi {

    extern "Rust" {
        fn rcg_optl_cpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> &CxxVector<f64>;

        fn rcg_optl_gpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> &CxxVector<f64>;

        fn em_cpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> &CxxVector<f64>;

        fn em_gpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> &CxxVector<f64>;

        fn mixture_components(
            probs: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
        ) -> &CxxVector<f64>;

    }
}

fn rcg_optl_cpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tol: f64,
    max_iters: usize,
) -> &CxxVector<f64> {
    let mut options: rcgpar::OptimizerOpts = Default::default();
    options.tolerance = tol;
    options.max_iters = max_iters;
    options.device = crate::CPU64;
    options.algorithm = "rcg";

    crate::optimize_mat(logl, log_times_observed, alpha0, Some(options)).unwrap()
}

fn rcg_optl_gpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tol: f64,
    max_iters: usize,
) -> &CxxVector<f64> {
    let mut options: rcgpar::OptimizerOpts = Default::default();
    options.tolerance = tol;
    options.max_iters = max_iters;
    options.device = crate::GPU32;
    options.algorithm = "rcg";

    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    return crate::optimize_mat(logl, log_times_observed, alpha0, Some(options)).unwrap();

    #[cfg(not(any(feature = "wgpu", feature = "webgpu", feature = "vulkan")))]
    panic!("rcgpar: rcgpar was not compiled with GPU support.")
}

fn em_cpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tol: f64,
    max_iters: usize,
) -> &CxxVector<f64> {
    let mut options: rcgpar::OptimizerOpts = Default::default();
    options.tolerance = tol;
    options.max_iters = max_iters;
    options.device = crate::CPU64;
    options.algorithm = "em";

    crate::optimize_mat(logl, log_times_observed, alpha0, Some(options)).unwrap()
}

fn em_gpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tol: f64,
    max_iters: usize,
) -> &CxxVector<f64> {
    let mut options: rcgpar::OptimizerOpts = Default::default();
    options.tolerance = tol;
    options.max_iters = max_iters;
    options.device = crate::GPU32;
    options.algorithm = "em";

    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    return crate::optimize_mat(logl, log_times_observed, alpha0, Some(options)).unwrap();

    #[cfg(not(any(feature = "wgpu", feature = "webgpu", feature = "vulkan")))]
    panic!("rcgpar: rcgpar was not compiled with GPU support.")
}

fn mixture_components(
    probs: &cxx::CxxVector<f64>,
    log_times_observed: &cxx::CxxVector<f64>,
) -> &CxxVector<f64> {

    let n_obs = log_times_observed.len();
    let n_targets = probs.len()/n_obs;

    let device = Default::default();
    type Backend = NdArray<f32>;

    let probs_t = Tensor::<Backend, 2>::from_data(probs, &device).reshape(Shape::new([n_targets, n_obs]));
    let log_counts_t = Tensor::<Backend, 1>::from_data(log_times_observed, &device);
    let thetas_t = crate::optimizer::mixture_components(probs_t, log_counts_t).unwrap();

    thetas_t.into_data().iter().collect::<CxxVector<f64>>()
}
