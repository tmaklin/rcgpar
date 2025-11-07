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

// Backend support
use burn::backend::ndarray::NdArray;
#[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
use burn::backend::wgpu::Wgpu;

use burn_tensor::{Device, Tensor};
use burn_tensor::backend::Backend;
use num::traits::{Float, PrimInt};
use num::FromPrimitive;

pub mod rcg;

type E = Box<dyn std::error::Error>;

/// Backend type for [burn](https://docs.rs/burn)
///
/// Number after enum name denotes floating point width.
///
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BurnBackend {
    /// [WebGPU](https://www.w3.org/TR/webgpu/), best cross-platform GPU support, 32 bit precision.
    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    GPU32,
    /// [WebGPU](https://www.w3.org/TR/webgpu/), best cross-platform GPU support, 64 bit precision.
    #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
    GPU64,

    /// [NdArray](https://docs.rs/ndarray), runs on most CPU architectures, 32 bit precision.
    CPU32,
    /// [NdArray](https://docs.rs/ndarray), runs on most CPU architectures, 64 bit precision.
    #[default]
    CPU64,
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OptimizerOpts {
    /// - Terminate optimization if values change by less than `tolerance`
    pub tolerance: f64,
    /// - Maximum number of iterations to run optimizer for.
    pub max_iters: usize,
    /// - Run on CPU or GPU.
    pub device: BurnBackend,
}

impl Default for OptimizerOpts {
    /// Default to these values:
    /// ```rust
    /// let mut opts = rcgpar::OptimizerOpts::default();
    /// opts.tolerance = 1e-7_f64;
    /// opts.max_iters = 5000_usize;
    /// opts.device = rcgpar::BurnBackend::CPU64;
    /// # let expected = rcgpar::OptimizerOpts::default();
    /// # assert_eq!(opts.tolerance, expected.tolerance);
    /// # assert_eq!(opts.max_iters, expected.max_iters);
    /// # assert_eq!(opts.device, expected.device);
    /// ```
    ///
    fn default() -> OptimizerOpts {
        OptimizerOpts {
            tolerance: 1e-7_f64,
            max_iters: 5000_usize,
            device: BurnBackend::CPU64,
        }
    }
}

/// Helper function to run on a generic backend
fn run_optimizer<B: Backend, F: Float + FromPrimitive, U: PrimInt>(
    logl_f: &[Vec<F>],
    counts_i: &[U],
    alpha0_f: &[F],
    options: &OptimizerOpts,
    device: &Device<B>,
) -> Result<Vec<F>, E> {

    let n_rows = counts_i.len();
    let n_cols = logl_f.len();

    let logl_floats: Vec<f32> = logl_f.iter().flat_map(|x| x.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>()).collect::<Vec<f32>>();
    let logl_flat = Tensor::<B, 1>::from_data(logl_floats.as_slice(), device);
    let logl = logl_flat.reshape([n_cols, n_rows]);

    let log_counts_floats: Vec<f32> = counts_i.iter().map(|x| x.to_f32().unwrap().ln()).collect();
    let log_counts = Tensor::<B, 1>::from_data(log_counts_floats.as_slice(), device);

    let alpha0_floats: Vec<f32> = alpha0_f.iter().map(|x| x.to_f32().unwrap()).collect();
    let alpha0 = Tensor::<B, 1>::from_data(alpha0_floats.as_slice(), device);

    let probs = rcg::rcg_optl_mat(logl, log_counts.clone(), alpha0, options.tolerance, options.max_iters, device)?;
    Ok(rcg::mixture_components(probs, log_counts)?.into_data().iter().map(|x: f64| FromPrimitive::from_f64(x).unwrap()).collect::<Vec<F>>())
}

/// Infer mixing proportions for a weighted log-likelihood matrix
///
/// Returns the mixing proportions that best fit the model corresponding to
/// `log_likelihood` with integer weights for each column given in `counts`.
/// Typically, `counts` is the number of times the likelihood vector in each
/// column was observed but can be any weight vector.
///
/// ## Options
/// Use `opts` to change the following:
/// - Modify optimizer tolerance via `opts.tolerance`.
/// - Modify maximum number of iterations via `opts.max_iters`.
/// - Run on CPU or GPU using `opts.device` (see [BurnBackend] for details).
/// - Set floating point precision to 32 or 64 bits via `opts.device`.
///
/// See [OptimizerOpts] for more details.
///
/// ## Prior
/// Prior for the mixing proportions is given via `prior`. Values in `prior` can
/// be interpreted as the observation counts from each category that were
/// observed before generating the log likelihood matrix `logl` for the current data.
///
/// Assumes a conjugate Dirichlet model, meaning that the mixing proportions
/// from a previously fitted model (weighted by the total observation count) can
/// be used as a prior when estimating a new dataset.
///
pub fn optimize<F: Float + FromPrimitive, U: PrimInt>(
    log_likelihood: &[Vec<F>],
    counts: &[U],
    prior: &[F],
    opts: Option<OptimizerOpts>,
) -> Result<Vec<F>, E> {
    assert_eq!(log_likelihood[0].len(), counts.len());
    assert_eq!(log_likelihood.len(), prior.len());

    let options = opts.unwrap_or_default();

    let proportions = match options.device {
        BurnBackend::CPU32 => {
            let device = burn::backend::ndarray::NdArrayDevice::default();
            type Backend = NdArray<f32>;
            run_optimizer::<Backend, F, U>(log_likelihood, counts, prior, &options, &device)?
        },
        BurnBackend::CPU64 => {
            let device = burn::backend::ndarray::NdArrayDevice::default();
            type Backend = NdArray<f64>;
            run_optimizer::<Backend, F, U>(log_likelihood, counts, prior, &options, &device)?
        },
        #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
        BurnBackend::GPU32 => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            type Backend = Wgpu<f32>;
            run_optimizer::<Backend, F, U>(&log_likelihood, counts, prior, &options, &device)?
        },
        #[cfg(any(feature = "wgpu", feature = "webgpu", feature = "vulkan"))]
        BurnBackend::GPU64 => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            type Backend = Wgpu<f64>;
            run_optimizer::<Backend, F, U>(&log_likelihood, counts, prior, &options, &device)?
        },
    };

    Ok(proportions)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn optimize_f64() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::BurnBackend;
        use super::OptimizerOpts;
        use super::optimize;

        let log_likelihood: Vec<Vec<f64>> =
            vec![
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503 ],
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713 ],
                vec![ -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503 ],
                vec![ -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 ],
            ];
        let counts: Vec<u32> = vec![2167, 1145, 943, 196, 175, 158, 1041, 957, 1447, 2135];
        let prior_counts: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];

        let expected: Vec<f64> = vec![0.9990609231670258, 0.0007300890279000023, 9.656363112888921e-5, 0.00011242417394518503];

        let opts = OptimizerOpts { tolerance: 1e-7_f64, max_iters: 100, device: BurnBackend::CPU64 };
        let got = optimize(&log_likelihood, &counts, &prior_counts, Some(opts)).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-17) });
    }

    #[test]
    fn optimize_f32() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::BurnBackend;
        use super::OptimizerOpts;
        use super::optimize;

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

        let opts = OptimizerOpts { tolerance: 1e-7_f64, max_iters: 100, device: BurnBackend::CPU32 };
        let got = optimize(&log_likelihood, &counts, &prior_counts, Some(opts)).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-4); assert!((x - y).abs() > 1e-8) });
    }
}
