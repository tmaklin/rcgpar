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

    let mut prev_loss = Tensor::<B, 1>::from_data([0.0], device);
    let tol = Tensor::<B, 1>::from_data([tolerance], device);

    let mut logl_weighted = logl.zeros_like();
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.clone().reshape(Shape::new([1, n_obs]));
    let mut thetas = log_counts.zeros_like();
    thetas = thetas.sub_scalar((n_targets as f64).ln());

    let mut iter = 0;
    while iter < max_iters {
        // E step
        let thetas_squeezed: Tensor::<B, 2> = thetas.clone().reshape(Shape::new([1, n_obs]));
        logl_weighted = logl_weighted.add(thetas_squeezed);
        let lse = logsumexp(logl_weighted.clone(), 0)?;
        let lse_squeezed : Tensor::<B, 2> = lse.clone().reshape(Shape::new([1, n_obs]));
        logl_weighted = logl_weighted.sub(lse_squeezed);

        // M step
        logl_weighted = logl_weighted.add(log_counts_squeezed.clone());
        logl_weighted = logl_weighted.exp();

        thetas = logl_weighted.clone().sum_dim(0).reshape(Shape::new([1, n_obs])).div_scalar(log_counts.clone().exp().sum().into_scalar());

        let loss = -lse.add(log_counts.clone()).exp().sum();

        if loss.clone().sub(prev_loss.clone()).abs().lower(tol.clone()).all().into_data().iter().next().unwrap() {
            break;
        }
        prev_loss = loss;
        iter += 1;
    }
    let thetas_squeezed: Tensor::<B, 2> = thetas.clone().reshape(Shape::new([1, n_obs]));
    let logl_weighted = logl.clone().add(thetas_squeezed);
    let lse = logsumexp(logl_weighted.clone(), 0)?;
    let lse_squeezed : Tensor::<B, 2> = lse.reshape(Shape::new([1, n_obs]));
    let gamma_Z = logl_weighted.sub(lse_squeezed);

    Ok(gamma_Z)
}

pub fn mixture_components<B: Backend>(
    gamma_Z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
) -> Result<Tensor::<B, 1>, E> {
    let n_times_total = log_counts.clone().exp().sum().log().into_scalar();
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.clone().reshape(Shape::new([1, gamma_Z.clone().dims()[1]]));
    let thetas = gamma_Z.clone().add(log_counts_squeezed).exp().sum_dim(1).log().sub_scalar(n_times_total).exp().reshape(Shape::new([gamma_Z.clone().dims()[0], 1]));
    Ok(thetas)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn em_optl_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::{Shape, Tensor};
        use burn_tensor::Int;

        use super::em_algorithm;
        use super::mixture_components;

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

        let alpha0 = Tensor::<Backend, 1>::from_data(
            [
                1.0, 1.0, 1.0, 1.0
            ],
            &device,
        );

        let expected = Tensor::<Backend, 2>::from_data(
            [
                [ -0.0010899, -0.00104044, -0.000928571, -0.00104519, -0.000995734, -0.000883857, -0.000944069, -0.000894604, -0.000782716, -0.000853449 ],
                [ -7.15745,   -7.1574,     -7.15729,     -7.15741,    -7.15736,     -7.15725,     -7.15731,     -7.15726,     -7.15715,     -7.51888 ],
                [ -8.82298,   -8.82293,    -8.82282,     -9.1846,     -9.18455,     -9.18444,     -13.418,      -13.4179,     -13.4178,     -8.82274 ],
                [ -8.72199,   -9.0836,     -13.3169,     -8.72195,    -9.08356,     -13.3169,     -8.72184,     -9.08346,     -13.3168,     -8.72175 ],
            ],
            &device,
        );

        let got = em_algorithm::<Backend>(logl, log_counts.clone(), 1e-7_f64, 100_usize, &device).unwrap();
        let components = mixture_components(got.clone(), log_counts).unwrap();
        eprintln!("{:?}", components);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }
}
