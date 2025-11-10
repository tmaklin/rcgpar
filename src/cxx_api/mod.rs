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

use crate::OptimizerOpts;
use crate::BurnBackend::CPU64;
use crate::BurnBackend::GPU32;
use crate::optimize;

use crate::optimizer::Algorithm;

use burn::backend::ndarray::NdArray;
use burn_tensor::{Shape, Tensor};
use cxx::CxxVector;

#[cxx::bridge(namespace = "rcgpar")]
mod ffi {

    extern "Rust" {
        fn rcg_optl_cpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> Vec<f64>;

        fn rcg_optl_gpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> Vec<f64>;

        fn em_cpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> Vec<f64>;

        fn em_gpu(
            logl: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
            alpha0: &CxxVector<f64>,
            tol: f64,
            max_iters: usize,
        ) -> Vec<f64>;

        fn mixture_components(
            probs: &CxxVector<f64>,
            log_times_observed: &CxxVector<f64>,
        ) -> Vec<f64>;

    }
}

pub fn run_optimizer(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    options: OptimizerOpts,
) -> Vec<f64> {
    let n_targets = alpha0.len();
    let logl_r: Vec<Vec<f64>> = logl.iter().cloned().collect::<Vec<f64>>().chunks(n_targets).map(|x| x.to_vec()).collect::<Vec<Vec<f64>>>();
    let log_counts_r: Vec<u64> = log_times_observed.iter().map(|x| *x as u64).collect::<Vec<u64>>();
    let alpha0_r: Vec<f64> = alpha0.iter().cloned().collect::<Vec<f64>>();
    let (_, probs) = optimize(&logl_r, &log_counts_r, &alpha0_r, Some(options)).unwrap();
    probs
}

pub fn rcg_optl_cpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tolerance: f64,
    max_iters: usize,
) -> Vec<f64> {
    let options = OptimizerOpts { tolerance, max_iters, device: CPU64, algorithm: Algorithm::RCG };
    run_optimizer(logl, log_times_observed, alpha0, options)
}

#[allow(unused_variables)]
pub fn rcg_optl_gpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tolerance: f64,
    max_iters: usize,
) -> Vec<f64> {
    let options = OptimizerOpts { tolerance, max_iters, device: GPU32, algorithm: Algorithm::RCG };

    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    return run_optimizer(logl, log_times_observed, alpha0, options);

    #[cfg(not(any(feature = "wgpu", feature = "webgpu", feature = "vulkan")))]
    panic!("rcgpar: rcgpar was not compiled with GPU support.")
}

pub fn em_cpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tolerance: f64,
    max_iters: usize,
) -> Vec<f64> {
    let options = OptimizerOpts { tolerance, max_iters, device: CPU64, algorithm: Algorithm::EM };
    run_optimizer(logl, log_times_observed, alpha0, options)
}

#[allow(unused_variables)]
pub fn em_gpu(
    logl: &CxxVector<f64>,
    log_times_observed: &CxxVector<f64>,
    alpha0: &CxxVector<f64>,
    tolerance: f64,
    max_iters: usize,
) -> Vec<f64> {
    let options = OptimizerOpts { tolerance, max_iters, device: GPU32, algorithm: Algorithm::EM };

    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    return run_optimizer(logl, log_times_observed, alpha0, options);

    #[cfg(not(any(feature = "wgpu", feature = "webgpu", feature = "vulkan")))]
    panic!("rcgpar: rcgpar was not compiled with GPU support.")
}

pub fn mixture_components(
    probs: &cxx::CxxVector<f64>,
    log_times_observed: &cxx::CxxVector<f64>,
) -> Vec<f64> {

    let n_obs = log_times_observed.len();
    let n_targets = probs.len()/n_obs;

    let device = Default::default();
    type Backend = NdArray<f32>;

    let probs_r: Vec<u64> = probs.iter().map(|x| *x as u64).collect::<Vec<u64>>();
    let log_counts_r: Vec<u64> = log_times_observed.iter().map(|x| *x as u64).collect::<Vec<u64>>();

    let probs_t = Tensor::<Backend, 2>::from_data(probs_r.as_slice(), &device).reshape(Shape::new([n_targets, n_obs]));
    let log_counts_t = Tensor::<Backend, 1>::from_data(log_counts_r.as_slice(), &device);

    let thetas_t = crate::optimizer::mixture_components(probs_t, log_counts_t);

    let thetas: Vec<f64> = thetas_t.into_data().iter().collect::<Vec<f64>>();
    thetas
}
