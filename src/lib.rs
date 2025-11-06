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

use burn::backend::ndarray::NdArray;
use burn_tensor::Tensor;
use num::traits::{Float, PrimInt};
use num::FromPrimitive;

pub mod rcg;

type E = Box<dyn std::error::Error>;

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerOpts {
    /// - Terminate optimization if values change by less than `tolerance`
    pub tolerance: f64,
    /// - Maximum number of iterations to run optimizer for.
    pub max_iters: usize,
    /// - Use 64-bit floating point numbers instead of 32-bit.
    pub use_f64: bool,
}

impl Default for OptimizerOpts {
    /// Default to these values:
    /// ```rust
    /// let mut opts = rcgpar::OptimizerOpts::default();
    /// opts.tolerance = 1e-7_f64;
    /// opts.max_iters = 5000_usize;
    /// opts.use_f64 = true;
    /// # let expected = rcgpar::OptimizerOpts::default();
    /// # assert_eq!(opts.tolerance, expected.tolerance);
    /// # assert_eq!(opts.max_iters, expected.max_iters);
    /// # assert_eq!(opts.use_f64, expected.use_f64);
    /// ```
    ///
    fn default() -> OptimizerOpts {
        OptimizerOpts {
            tolerance: 1e-7_f64,
            max_iters: 5000_usize,
            use_f64: true,
        }
    }
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
/// - Floating point precision can be set to 64 bits via `opts.use_f64`.
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

    let n_rows = log_likelihood[0].len();
    let n_cols = log_likelihood.len();

    // TODO Cleaner way to write selecting f32 vs. f64 precision in optimize().
    //
    let proportions = if !options.use_f64 {
        let logl_floats = log_likelihood.iter().flat_map(|x| x.iter().map(|y| y.to_f32().unwrap()).collect::<Vec<f32>>()).collect::<Vec<f32>>();
        let log_counts_floats = counts.iter().map(|x| x.to_f32().unwrap().ln()).collect::<Vec<f32>>();
        let alpha0_floats = prior.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<f32>>();

        let device = Default::default();
        type Backend = NdArray<f32>;

        let logl_flat = Tensor::<Backend, 1>::from_data(logl_floats.as_slice(), &device);
        let logl = logl_flat.reshape([n_cols, n_rows]);

        let log_counts = Tensor::<Backend, 1>::from_data(log_counts_floats.as_slice(), &device);
        let alpha0 = Tensor::<Backend, 1>::from_data(alpha0_floats.as_slice(), &device);

        let probs = rcg::rcg_optl_mat(logl, log_counts.clone(), alpha0)?;
        rcg::mixture_components(probs, log_counts)?.into_data().iter().map(|x| FromPrimitive::from_f32(x).unwrap()).collect::<Vec<F>>()
    } else {
        let logl_floats = log_likelihood.iter().flat_map(|x| x.iter().map(|y| y.to_f64().unwrap()).collect::<Vec<f64>>()).collect::<Vec<f64>>();
        let log_counts_floats = counts.iter().map(|x| x.to_f64().unwrap().ln()).collect::<Vec<f64>>();
        let alpha0_floats = prior.iter().map(|x| x.to_f64().unwrap()).collect::<Vec<f64>>();

        let device = Default::default();
        type Backend = NdArray<f64>;

        let logl_flat = Tensor::<Backend, 1>::from_data(logl_floats.as_slice(), &device);
        let logl = logl_flat.reshape([n_cols, n_rows]);

        let log_counts = Tensor::<Backend, 1>::from_data(log_counts_floats.as_slice(), &device);
        let alpha0 = Tensor::<Backend, 1>::from_data(alpha0_floats.as_slice(), &device);

        let probs = rcg::rcg_optl_mat(logl, log_counts.clone(), alpha0)?;
        rcg::mixture_components(probs, log_counts)?.into_data().iter().map(|x| FromPrimitive::from_f64(x).unwrap()).collect::<Vec<F>>()
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

        let expected: Vec<f64> = vec![0.9990609232614853, 0.0007300889486079688, 9.656361438673255e-5, 0.00011242417552052694];

        let opts = OptimizerOpts { tolerance: 1e-7_f64, max_iters: 100, use_f64: true };
        let got = optimize(&log_likelihood, &counts, &prior_counts, Some(opts)).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-17) });
    }

    #[test]
    fn optimize_f32() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

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

        let opts = OptimizerOpts { tolerance: 1e-7_f64, max_iters: 100, use_f64: false };
        let got = optimize(&log_likelihood, &counts, &prior_counts, Some(opts)).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-4); assert!((x - y).abs() > 1e-8) });
    }
}
