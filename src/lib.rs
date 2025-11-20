// rcgpar: Riemannian conjugate gradient descent for estimating mixture model weights.
//
// Copyright 2025 Tommi Mäklin [tommi@maklin.fi].
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

//! rcgpar provides implementations for several optimization algorithms that
//! infer the `K` mixture model weights for a `N x K` log-likelihood matrix.
//!
//! ## Features
//! - GPU support
//! - Rust library
//! - C++ bindings
//! - Minimal CLI
//!
//! ## Installation
//!
//! By default, rcgpar is available as a Rust library with support for the Wgpu
//! and NdArray backends for [burn]. Several other options are available.
//!
//! ### Command-line client
//!
//! A minimal CLI for testing purposes can be compiled with `--feature cli`.
//!
//! ### The burn backend
//!
//! The Wgpu backend can be changed to use the alternative vulkan or webgpu
//! implementations by adding `--feature vulkan` or `--feature webgpu`.
//!
//! More backends from burn can be implemented if needed, make a request in the
//! source repository or implement it yourself in [BurnBackend] and
//! [optimize_flat], and by adding the appropriate feature to Cargo.toml. The
//! rest of the code is designed in a backend-agnostic manner.
//!
//! ### C++ bindings
//!
//! rcgpar provides bindings for running inference with 32-bit floating point
//! inputs. These can be compiled by adding `--feature cxxbridge`.
//!
//! The C++ bindings create the `librcgpar.a`, `rcgpar_cxx.cpp`, and
//! `rcgpar_cxx.h` files that can be included in a C++ project to call the
//! rcgpar C++ API.
//!
//! A CMake file is provided to configure the flags passed to cargo when
//! building the bindings.
//!
//! ## API
//!
//! The library provides several high-level functions to run on 2D vector
//! inputs, flattened vectors, or tensor data.
//!
//! Low-level functions are available in the [optimizer] module.
//!
//! Returns the mixing proportions that best fit the model corresponding to
//! `log_likelihood` with integer weights for each column given in `counts`.
//! Typically, `counts` is the number of times the likelihood vector in each
//! column was observed but can be any weight vector.
//!
//! ## C++ API
//! The C++ API provides four functions to peform inference:
//! - `rcg_optl_cpu`: run [rcg](optimizer::rcg) with the NdArray backend.
//! - `rcg_optl_gpu`: run [rcg](optimizer::rcg) with the Wgpu backend.
//! - `em_optl_cpu`: run [em](optimizer::em) with the NdArray backend.
//! - `em_optl_gpu`: run [em](optimizer::em) with the Wgpu backend.
//!
//! An additional convenience function `mixture_components` is provided to
//! convert the inference results to mixing proportions.
//!
//! ### Inputs
//!
//! The C++ API main functions all take the following inputs:
//! - `logl`: flattened column-major `n_cols x n_rows` log-likelihood matrix.
//! - `log_times_observed`: `n_rows` vector of natural logarithm of the weights for `logl_f` rows.
//! - `alpha0`: `n_cols` vector of prior counts for the Dirichlet model.
//! - `tol`: optimizer tolerance for convergence checking.
//! - `max_iters`: maximum number of iterations to run the optimizer for.
//!
//! The first 3 arguments expect a `std::vector<float>`, the tolerance is given
//! as a `double` and maximum iterations as a `size_t`
//!
//! ### Outputs
//!
//! All four functions return a single Rust vector that contains the flattened
//! `n_cols x n_rows` column-major matrix containing inferred probabilities that
//! the row `i` was generated from cluster `j`.
//!
//! The output can be converted to a `std::vector` by using for example the following code
//! ```c++
//! auto probs_rs = rcgpar::rcg_optl_gpu(loglls, log_counts, alpha0, (double)0.00001, (size_t)1000);
//! probs_cpp.reserve((uint64_t)((uint64_t)n_groups * (uint64_t)n_obs));
//! for (auto &val : probs_rs) {
//!     probs_cpp.push_back(val);
//! }
//! ```
//!
//! ## Using the optimizers
//!
//! The high-level API can be customized with several options and prior counts,
//! detailed below.
//!
//! ### Options
//!
//! Use [OptimizerOpts] to change the following:
//! - Tolerance for convergence checking via `opts.tolerance`.
//! - Maximum number of iterations via `opts.max_iters`.
//! - Run on CPU or GPU using `opts.device` (see [BurnBackend] for details).
//! - Set floating point precision to 32 or 64 bits via `opts.device`.
//!
//! See [OptimizerOpts] for more details.
//!
//! ### Prior
//!
//! Prior for the Dirichlet model mixing proportions is given via `prior`.
//! Values in `prior` can be interpreted as the observation counts from each
//! category that were observed before generating the log likelihood matrix
//! `logl` for the current data.
//!
//! Assumes a conjugate Dirichlet model, meaning that the mixing proportions
//! from a previously fitted model (weighted by the total observation count) can
//! be used as a prior when estimating a new dataset.
//!
//! ## Reading
//!
//! The rcgpar variational inference algorithm [rcg](optimizer::rcg) was originally a part of the
//! [mSWEEP](https://github.com/PROBIC/mSWEEP) software described in:
//! - M&auml;klin et al. (2020) "High-resolution sweep metagenomics using fast probabilistic
//!   inference", _Wellcome open research_. doi:
//!   [10.12688/wellcomeopenres.15639.2](https://doi.org/10.12688/wellcomeopenres.15639.2).
//! - M&auml;klin (2022) "Probabilistic methods for high-resolution
//!   metagenomics" chapter 2.3.7, Series of publications A / Department of
//!   Computer Science, University of Helsinki. ISBN:
//!   [978-951-51-8695-9](http://urn.fi/URN:ISBN:978-951-51-8695-9).
//!
//! The expectation-maximization algorithm [em](optimizer::em) and the original
//! rcg GPU implementations are described in
//! - Pietil&auml;inen (2025) "Accelerating mixture model inference for
//!   bacterial community estimation using GPU computing", University of Helsinki. urn:
//!   [hulib-202501301212](http://urn.fi/URN:NBN:fi:hulib-202501301212).
//!

// Backend support
#[cfg(feature = "ndarray")]
use burn::backend::ndarray::NdArray;
#[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
use burn::backend::wgpu::Wgpu;

use burn_tensor::{Device, Tensor};
use burn_tensor::backend::Backend;
use num::traits::{Float, PrimInt};
use num::FromPrimitive;

#[cfg(feature = "cxxbridge")]
pub mod cxx_api;

pub mod math;
pub mod optimizer;

use optimizer::Algorithm;

type E = Box<dyn std::error::Error>;

/// Backend type for [burn](https://docs.rs/burn)
///
/// Number after enum name denotes floating point width.
///
/// 64-bit floats may require extra compilation flags for some devices. The
/// optimizer code is designed to work with 32-bit floats, these should be
/// preferred.
///
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BurnBackend {
    /// [WebGPU](https://www.w3.org/TR/webgpu/), best cross-platform GPU support, 32 bit precision.
    Wgpu32,
    /// [WebGPU](https://www.w3.org/TR/webgpu/), best cross-platform GPU support, 64 bit precision.
    Wgpu64,

    /// [NdArray](https://docs.rs/ndarray), runs on most CPU architectures, 32 bit precision.
    #[default]
    NdArray32,
    /// [NdArray](https://docs.rs/ndarray), runs on most CPU architectures, 64 bit precision.
    NdArray64,
}

impl std::str::FromStr for BurnBackend {
    type Err = String; // Define an error type for parsing failures

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu32" => Ok(BurnBackend::NdArray32),
            "cpu64" => Ok(BurnBackend::NdArray64),
            "gpu32" => Ok(BurnBackend::Wgpu32),
            "gpu64" => Ok(BurnBackend::Wgpu64),
            _ => Err(format!("'{}' is not a valid BurnBackend variant", s)),
        }
    }
}

/// Options for [optimizer] algorithms.
///
/// This struct is
/// [non_exhaustive](https://doc.rust-lang.org/reference/attributes/type_system.html).
/// The struct is expected to be stabilized at some point.
///
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OptimizerOpts {
    /// - Terminate optimization if values change by less than `tolerance`.
    pub tolerance: f64,
    /// - Maximum number of iterations to run optimizer for.
    pub max_iters: usize,
    /// - Device to run on.
    pub device: BurnBackend,
    /// - Optimizer algorithm to use.
    pub algorithm: Algorithm,
}

impl Default for OptimizerOpts {
    /// Default to these values:
    /// ```rust
    /// let mut opts = rcgpar::OptimizerOpts::default();
    /// opts.tolerance = 1e-7_f64;
    /// opts.max_iters = 5000_usize;
    /// opts.device = rcgpar::BurnBackend::NdArray32;
    /// opts.algorithm = rcgpar::optimizer::Algorithm::RCG;
    /// # let expected = rcgpar::OptimizerOpts::default();
    /// # assert_eq!(opts.tolerance, expected.tolerance);
    /// # assert_eq!(opts.max_iters, expected.max_iters);
    /// # assert_eq!(opts.device, expected.device);
    /// # assert_eq!(opts.algorithm, expected.algorithm);
    /// ```
    ///
    fn default() -> OptimizerOpts {
        OptimizerOpts {
            tolerance: 1e-7_f64,
            max_iters: 5000_usize,
            device: BurnBackend::NdArray32,
            algorithm: Algorithm::RCG,
        }
    }
}

/// Helper function to run on a generic backend
///
/// Allocates the input data on the [BurnBackend] and calls [optimize_tensor](optimize_tensor<B: Backend>) to
/// run inference.
///
/// burn [Backend](https://docs.rs/burn/latest/src/burn/backend.rs.html#1-72)
/// must be given as the generic <B: Backend>.
/// Does *not* verify the input dimensions.
///
/// ## Inputs
/// - `logl_f`: flattened column-major `n_cols x n_rows` log-likelihood matrix.
/// - `log_counts_f`: `n_rows` vector of natural logarithm of the weights for `logl_f` rows.
/// - `alpha0_f`: `n_cols` vector of prior counts for the Dirichlet model.
/// - `options`: [OptimizerOpts]
/// - `device`: burn [Device](https://docs.rs/burn-tensor/0.19.1/src/burn_tensor/tensor/ops/alias.rs.html#7) wrapping the [Backend](https://docs.rs/burn/latest/src/burn/backend.rs.html#1-72).
///
/// ## Outputs
/// - `thetas`: `n_cols` vector of inferred mixing proportions.
/// - `gamma_Z`: flattened `n_cols x n_rows` column-major matrix containing inferred probabilities that the row `i` was generated from cluster `j`.
///
/// ## Errors
/// Propagates errors, does not error on its own.
///
pub fn run_optimizer<B: Backend>(
    logl_f: &[f32],
    log_counts_f: &[f32],
    alpha0_f: &[f32],
    options: &OptimizerOpts,
    device: &Device<B>,
) -> Result<(Vec<f32>, Vec<f32>), E> {

    let n_rows = log_counts_f.len();
    let n_cols = alpha0_f.len();

    let logl_flat = Tensor::<B, 1>::from_data(logl_f, device);
    let log_counts = Tensor::<B, 1>::from_data(log_counts_f, device);
    let alpha0 = Tensor::<B, 1>::from_data(alpha0_f, device);

    let logl = logl_flat.reshape([n_cols, n_rows]);

    let (alpha0, logl) = optimize_tensor::<B>(logl, log_counts, alpha0, options)?;

    Ok((alpha0.into_data().to_vec().unwrap(), logl.into_data().to_vec().unwrap()))
}

/// Run on [Tensor] inputs
///
/// Preferred function when efficiency is required but you don't want to call
/// [optimizer] directly.
///
/// burn [Backend](https://docs.rs/burn/latest/src/burn/backend.rs.html#1-72)
/// must be given as the generic <B: Backend>.
///
/// ## Inputs
/// - `log_likelihood`: column-major `n_cols x n_rows` log-likelihood matrix.
/// - `log_counts`: `n_rows` vector of natural logarithms of the weights for `log_likelihood` rows.
/// - `alpha0`: `n_cols` vector of prior counts for the Dirichlet model.
/// - `options`: [OptimizerOpts].
///
/// ## Outputs:
/// - `thetas`: `n_cols` vector of inferred mixing proportions.
/// - `gamma_Z`: flattened `n_cols x n_rows` column-major matrix containing inferred probabilities that the row `i` was generated from cluster `j`.
///
pub fn optimize_tensor<B: Backend>(
    log_likelihood: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
    options: &OptimizerOpts,
) -> Result<(Tensor::<B, 1>, Tensor::<B, 2>), E> {

    let probs = match options.algorithm {
        optimizer::Algorithm::RCG => optimizer::rcg::rcg_optl_mat(log_likelihood, log_counts.clone(), alpha0, options.tolerance, options.max_iters)?,
        optimizer::Algorithm::EM => optimizer::em::em_algorithm(log_likelihood, log_counts.clone(), options.tolerance, options.max_iters, &alpha0.device())?,
    };

    let proportions = optimizer::mixture_components(probs.clone(), log_counts);

    Ok((proportions, probs))
}

/// Run on flattened f32 vector inputs
///
/// Wrapper around [run_optimizer] & [optimize_tensor](optimize_tensor<B:
/// Backend>) to run inference.
///
/// Preferred for running on any backend supported by [BurnBackend] and given via [OptimizerOpts].
///
/// ## Inputs
/// - `log_likelihood`: column-major `n_cols x n_rows` log-likelihood matrix.
/// - `log_counts`: `n_rows` vector of natural logarithms of the weights for `log_likelihood` rows.
/// - `prior`: `n_cols` vector of prior counts for the Dirichlet model.
/// - `opts`: [OptimizerOpts].
///
/// ## Outputs:
/// - `thetas`: `n_cols` vector of inferred mixing proportions.
/// - `gamma_Z`: flattened `n_cols x n_rows` column-major matrix containing inferred probabilities that the row `i` was generated from cluster `j`.
///
pub fn optimize_flat(
    log_likelihood: &[f32],
    log_counts: &[f32],
    prior: &[f32],
    opts: Option<OptimizerOpts>,
) -> Result<(Vec<f32>, Vec<f32>), E> {
    assert_eq!(log_likelihood.len() as u64, (log_counts.len() as u64) * (prior.len() as u64));

    let options = opts.unwrap_or_default();

    let (proportions, probs_mat) = match options.device {
        #[cfg(feature = "ndarray")]
        BurnBackend::NdArray32 => {
            let device = burn::backend::ndarray::NdArrayDevice::default();
            type Backend = NdArray<f32>;
            run_optimizer::<Backend>(log_likelihood, log_counts, prior, &options, &device)?
        },
        #[cfg(feature = "ndarray")]
        BurnBackend::NdArray64 => {
            let device = burn::backend::ndarray::NdArrayDevice::default();
            type Backend = NdArray<f64>;
            run_optimizer::<Backend>(log_likelihood, log_counts, prior, &options, &device)?
        },
        #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
        BurnBackend::Wgpu32 => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            type Backend = Wgpu<f32>;
            run_optimizer::<Backend>(log_likelihood, log_counts, prior, &options, &device)?
        },
        #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
        BurnBackend::Wgpu64 => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            type Backend = Wgpu<f64>;
            run_optimizer::<Backend>(log_likelihood, log_counts, prior, &options, &device)?
        },
        // TODO Return error instead of panic when requesting a backend that is not supported
        #[cfg(not(any(feature = "wgpu", feature = "webgpu", feature = "vulkan")))]
        BurnBackend::Wgpu32 | BurnBackend::Wgpu64 => panic!("rcgpar was not compiled with WGPU support, recompile with `--features wgpu` to enable."),
        #[cfg(not(feature = "ndarray"))]
        BurnBackend::NdArray32 | BurnBackend::NdArray64 => panic!("rcgpar was not compiled with NdArray support, recompile with `--features ndarray` to enable."),
    };

    Ok((proportions, probs_mat))
}

/// Run optimizer on 2D f32 log-likelihoods and integer weights.
///
/// Wrapper around [optimize_flat] that flattens the input and computes the log
/// counts.
///
/// This function uses extra memory to handle generic floating point and integer
/// types. [optimize_flat] or [optimize_tensor](optimize_tensor<B: Backend>)
/// should be preferred if memory usage is a concern.
///
/// ## Inputs
/// - `log_likelihood`: 2D vector with `n_cols x n_rows` log-likelihood matrix.
/// - `counts`: `n_rows` vector of integer weights for `log_likelihood` rows.
/// - `prior`: `n_cols` vector of prior counts for the Dirichlet model.
/// - `opts`: [OptimizerOpts].
///
/// ## Outputs:
/// - `thetas`: `n_cols` vector of inferred mixing proportions.
/// - `gamma_Z`: flattened `n_cols x n_rows` column-major matrix containing inferred probabilities that the row `i` was generated from cluster `j`.
///
/// ## Floating point width
/// Values will be converted to and returned as 32-bit floats regardless of input width.
///
/// Computation is performed in 32-bit space by default. If you want to perform
/// computation in 64-bit space, specify a 64-bit device via `opts`.
///
/// If you want to supply *log likelihoods* using non-32-bit floating point
/// numbers, call [optimize_tensor](optimize_tensor<B: Backend>) with the
/// appropriate tensor data.
///
pub fn optimize<F: Float + FromPrimitive, U: PrimInt>(
    log_likelihood: &[Vec<F>],
    counts: &[U],
    prior: &[F],
    opts: Option<OptimizerOpts>,
) -> Result<(Vec<f32>, Vec<f32>), E> {
    let logl_flat = log_likelihood.iter().flatten().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>();
    let log_counts = counts.iter().map(|x| x.to_f32().unwrap().ln()).collect::<Vec<f32>>();
    let alpha0 = prior.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>();
    optimize_flat(&logl_flat, &log_counts, &alpha0, opts)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn optimize() {
        use super::BurnBackend;
        use super::OptimizerOpts;
        use super::optimize;
        use super::optimizer::Algorithm;

        let log_likelihood: Vec<Vec<f32>> =
            vec![
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503 ],
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713 ],
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503 ],
                vec![ -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 ],
            ];
        let counts: Vec<u32> = vec![2167, 1145, 943, 196, 175, 158, 1041, 957, 1447, 2135];
        let prior_counts: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let expected: Vec<f32> = vec![0.9990609232614853, 0.0007300889486079688, 9.656361438673255e-5, 0.00011242417552052694];

        let opts = OptimizerOpts { tolerance: 1e-7_f64, max_iters: 100, device: BurnBackend::NdArray32, algorithm: Algorithm::RCG };
        let (got, _) = optimize(&log_likelihood, &counts, &prior_counts, Some(opts)).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-4) });
    }
}
