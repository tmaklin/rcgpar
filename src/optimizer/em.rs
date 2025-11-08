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

use burn_tensor::Device;
use burn_tensor::backend::Backend;
use burn_tensor::{Shape, Tensor};

type E = Box<dyn std::error::Error>;

pub fn logsumexp<B: Backend>(
    input: Tensor::<B, 2>,
    dim: usize,
) -> Result<Tensor<B, 1>, E> {
    let max = input.clone().max_dim(dim).unsqueeze();
    let res = (input - max.clone()).exp().sum_dim(dim).log();
    let res = res + max;
    let res = res.squeeze();
    Ok(res)
}

pub fn em_algorithm<B: Backend>(
    logl: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    tolerance: f64,
    max_iters: usize,
    device: &Device<B>,
) -> Result<Tensor::<B, 2>, E> {
    let n_targets = logl.clone().dims()[0];
    let n_obs = logl.clone().dims()[1];
    assert_eq!(n_obs, log_counts.clone().dims()[0]);

    let mut prev_loss = Tensor::<B, 1>::from_data([100000.0], device);
    let tol = Tensor::<B, 1>::from_data([tolerance], device);

    let mut logl_weighted;
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.clone().reshape(Shape::new([1, n_obs]));
    let mut thetas = Tensor::<B, 1>::zeros(Shape::new([n_targets]), device);
    thetas = thetas.sub_scalar((n_targets as f64).ln()).exp();

    let mut iter = 0;
    while iter < max_iters {
        // E step
        let thetas_squeezed: Tensor::<B, 2> = thetas.clone().log().reshape(Shape::new([n_targets, 1]));
        logl_weighted = logl.clone().add(thetas_squeezed);
        let lse = logsumexp(logl_weighted.clone(), 0)?;
        let lse_squeezed : Tensor::<B, 2> = lse.clone().reshape(Shape::new([1, n_obs]));
        logl_weighted = logl_weighted.sub(lse_squeezed);

        // M step
        logl_weighted = logl_weighted.add(log_counts_squeezed.clone());
        logl_weighted = logl_weighted.exp();

        thetas = logl_weighted.clone().sum_dim(1).reshape(Shape::new([n_targets])).div_scalar(log_counts.clone().exp().sum().into_scalar());

        let loss = -lse.add(log_counts.clone()).exp().sum();

        if loss.clone().sub(prev_loss.clone()).abs().lower(tol.clone()).all().into_data().iter().next().unwrap() {
            break;
        }
        prev_loss = loss;
        iter += 1;
    }
    let thetas_squeezed: Tensor::<B, 2> = thetas.clone().log().reshape(Shape::new([n_targets, 1]));
    logl_weighted = logl.clone().add(thetas_squeezed);
    let lse = logsumexp(logl_weighted.clone(), 0)?;
    let lse_squeezed : Tensor::<B, 2> = lse.clone().reshape(Shape::new([1, n_obs]));
    let gamma_z = logl_weighted.sub(lse_squeezed);

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

        let got = em_algorithm::<Backend>(logl, log_counts.clone(), 1e-7_f64, 100_usize, &device).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-7_f32) });
    }
}
