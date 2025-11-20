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

//! Implementation of the expectation maximization algorithm used
//! in Pietiläinen 2024.
//!
//! This implementation is based on the derivation by Jarno and Joel & the gpu
//! implementation by Joel.
//!

use crate::math::logsumexp;
use crate::math::logsumexp_mat;

use burn_tensor::backend::Backend;
use burn_tensor::{Shape, Tensor};

type E = Box<dyn std::error::Error>;

/// Expectation maximization algorithm
pub fn em_algorithm<B: Backend>(
    logl: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    tolerance: f64,
    max_iters: usize,
) -> Result<Tensor::<B, 2>, E> {
    let log_counts = log_counts.unsqueeze();
    let lse2 = logsumexp_mat(log_counts.clone());
    let lse2 = lse2.unsqueeze();

    let mut thetas = Tensor::<B, 2>::zeros(Shape::new([logl.dims()[0], 1]), &logl.device());
    thetas = thetas.sub_scalar((logl.dims()[0] as f64).ln());

    let mut iter = 0;
    let mut logl_weighted = logl.clone().add(thetas.clone());
    let mut prev_loss = Tensor::<B, 1>::from_data([f64::MAX], &logl.device());
    while iter < max_iters {
        let lse = logsumexp(logl_weighted.clone(), 0);
        logl_weighted = logl_weighted.sub(lse.clone()).add(log_counts.clone());
        logl_weighted = logsumexp(logl_weighted, 1);

        thetas = logl_weighted.sub(lse2.clone());

        let loss = -logsumexp_mat(lse.add(log_counts.clone()));
        let diff: f64 = loss.clone().sub(prev_loss).into_data().iter().next().unwrap();

        if diff.abs() < tolerance {
            logl_weighted = logl.clone().add(thetas.clone());
            break;
        }
        logl_weighted = logl.clone().add(thetas.clone());
        prev_loss = loss;
        iter += 1;
    }
    let lse = logsumexp(logl_weighted.clone(), 0);
    let gamma_z = logl_weighted.sub(lse);

    Ok(gamma_z)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn em_optl_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

        use super::em_algorithm;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let logl = Tensor::<Backend, 2>::from_data(
            [
                [ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503 ],
                [ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713 ],
                [ -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503 ],
                [ -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 ],
            ],
            &device,
        );

        let log_counts = Tensor::<Backend, 1>::from_data(
            [
                7.681099, 7.04316, 6.849066, 5.278115, 5.164786, 5.062595, 6.947937, 6.863803, 7.277248, 7.666222
            ],
            &device,
        );

        let expected = Tensor::<Backend, 2>::from_data(
            [
                [-0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0013959194, -0.0009725131],
                [-6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.5748825, -6.936122],
                [-42.378433, -42.378433, -42.378433, -42.740093, -42.740093, -42.740093, -46.973553, -46.973553, -46.973553, -42.37801],
                [-37.232487, -37.594147, -41.827606, -37.232487, -37.594147, -41.827606, -37.232487, -37.594147, -41.827606, -37.232063]
            ],
            &device,
        );

        let got = em_algorithm::<Backend>(logl, log_counts.clone(), 1e-7_f64, 100_usize).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-5_f32) });
    }
}
