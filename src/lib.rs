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
use burn_tensor::{Shape, Tensor};
use num::traits::{Float, Unsigned};

pub mod rcg;

type E = Box<dyn std::error::Error>;

/// Optimize model weights (placeholder)
pub fn optimize<F: Float, U: Unsigned>(
    log_likelihood: &[Vec<F>],
    counts: &[U],
    prior_counts: &[F],
) -> Result<(), E> {
    let n_rows = log_likelihood[0].len();
    let n_cols = log_likelihood.len();

    let logl_floats = log_likelihood.iter().flat_map(|x| x.iter().map(|y| y.to_f32().unwrap()).collect::<Vec<f32>>()).collect::<Vec<f32>>();
    // let log_counts_floats = counts.iter().map(|x| *x as usize).collect::<Vec<u32>>();

    let device = Default::default();
    type Backend = NdArray<f32>;

    let logl_flat = Tensor::<Backend, 1>::from_data(logl_floats.as_slice(), &device);
    let logl = logl_flat.reshape([n_rows, n_cols]);

    todo!("Implement rcg_optl");
    Ok(())
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn optimize() {
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
        let prior_counts: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let expected: Vec<f64> = vec![0.999543, 0.00073079, 9.66135e-05, 0.000112505];
        let got = optimize(&log_likelihood, &counts, &prior_counts).unwrap();

        // got.iter().zip(expected.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 1e-16) });
        assert_eq!(0, 1);
    }
}
