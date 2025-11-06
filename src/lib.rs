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

/// Optimize model weights (placeholder)
pub fn optimize<F: Float + FromPrimitive, U: PrimInt>(
    log_likelihood: &[Vec<F>],
    counts: &[U],
    prior: &[F],
    use_f32: bool,
) -> Result<Vec<F>, E> {
    assert_eq!(log_likelihood[0].len(), counts.len());
    assert_eq!(log_likelihood.len(), prior.len());

    let n_rows = log_likelihood[0].len();
    let n_cols = log_likelihood.len();

    // TODO Figure out a cleaner way to write selecting f32 vs. f64 precision.
    let proportions = if use_f32 {
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
        let got = optimize(&log_likelihood, &counts, &prior_counts, false).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-17) });
    }

    #[test]
    fn optimize_f32() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

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
        let got = optimize(&log_likelihood, &counts, &prior_counts, true).unwrap();

        got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-4) });
    }
}
