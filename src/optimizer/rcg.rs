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

//! Implementation of the Riemannian conjugate gradient descent algorithm used
//! in Mäklin et al. 2020 in Wellcome Open Research.
//!
//! This implementation is based on the rcg_gpu Torch code written by Joel.
//!

use burn_tensor::backend::Backend;
use burn_tensor::{Shape, Tensor};
use statrs::function::gamma::digamma;
use statrs::function::gamma::ln_gamma;

type E = Box<dyn std::error::Error>;

pub fn logsumexp<B: Backend>(
    input: Tensor::<B, 2>,
    dim: usize,
) -> Result<Tensor<B, 2>, E> {
    let max = input.clone().max_dim(dim);
    let res = (input - max.clone()).exp().sum_dim(dim).log();
    let res = res + max;
    Ok(res)
}

pub fn compute_norm<B: Backend>(
    gamma_z: Tensor::<B, 2>,
    dl_dphi: Tensor::<B, 2>,
) -> Result<f64, E> {
    let temp = gamma_z.exp().mul(dl_dphi.clone());
    let colsums = temp.clone().sum_dim(0).reshape(Shape::new([1, dl_dphi.dims()[1]]));
    let newnorm = temp.mul(dl_dphi.sub(colsums)).sum().into_data().iter::<f64>().next().unwrap();
    Ok(newnorm)
}

pub fn mixt_negnatgrad<B: Backend>(
    logl: Tensor::<B, 2>,
    gamma_z: Tensor::<B, 2>,
    n_k: Tensor::<B, 1>,
) -> Result<Tensor::<B, 2>, E> {
    let n_k_data = n_k.clone().into_data();
    let digamma_n_k_vals = n_k_data.iter().map(|x: f32| (digamma(x as f64) - 1_f64) as f32).collect::<Vec<f32>>();
    let digamma_n_k = Tensor::<B, 1>::from_data(digamma_n_k_vals.as_slice(), &n_k.device()).reshape(Shape::new([logl.dims()[0], 1]));
    let dl_dphi: Tensor::<B, 2> = logl.add(digamma_n_k).sub(gamma_z);
    Ok(dl_dphi)
}

pub fn update_n_k<B: Backend>(
    gamma_z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
) -> Result<Tensor::<B, 1>, E> {
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.reshape(Shape::new([1, gamma_z.dims()[1]]));
    let n_k: Tensor::<B, 1> = gamma_z.add(log_counts_squeezed).exp().sum_dim(1).reshape(Shape::new([alpha0.dims()[0]])).add(alpha0);
    Ok(n_k)
}

pub fn elbo_rcg_mat<B: Backend>(
    logl: Tensor::<B, 2>,
    gamma_z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    n_k: Tensor::<B, 1>,
) -> Result<f64, E> {
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.reshape(Shape::new([1, gamma_z.dims()[1]]));

    let n_k_data = n_k.into_data();
    let lgamma_n_k_sum = n_k_data.iter().map(|x: f64| ln_gamma(x)).sum::<f64>();

    let bound = gamma_z.clone().add(log_counts_squeezed).exp().mul(logl.sub(gamma_z)).sum();
    let newbound: f64 = bound.add_scalar(lgamma_n_k_sum).into_data().iter().next().unwrap();

    Ok(newbound)
}

pub fn calc_bound_const<B: Backend>(
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
) -> Result<f64, E> {
    let alpha0_sum = alpha0.clone().sum();
    let counts_sum = log_counts.exp().sum();
    let alpha0_data = alpha0.clone().into_data();
    let lgamma_alpha0_sum = alpha0_data.iter().map(|x: f64| ln_gamma(x)).sum::<f64>();

    let alpha0_sum_f: f64 = alpha0_sum.into_data().iter().next().unwrap();
    let counts_sum_f: f64 = counts_sum.into_data().iter().next().unwrap();

    let bound_const = ln_gamma(alpha0_sum_f) + ln_gamma(alpha0_sum_f + counts_sum_f) - lgamma_alpha0_sum;

    Ok(bound_const)
}

pub fn rcg_optl_mat<B: Backend>(
    logl: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
    tolerance: f64,
    max_iters: usize,
) -> Result<Tensor::<B, 2>, E> {
    let n_targets = logl.dims()[0];
    let n_obs = logl.dims()[1];

    let mut gamma_z = logl.zeros_like() + (1_f64 / (n_targets as f64)).ln();
    let mut oldstep = logl.zeros_like();

    let mut iter = 0;
    let mut tol: f64 = tolerance;

    let mut bound = -10000_f64;

    let bound_const: f64 = calc_bound_const(log_counts.clone(), alpha0.clone())?;
    let mut n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone())?;

    let mut oldnorm: f64 = 1_f64;
    let mut didreset = false;

    while iter < max_iters {
        let mut step = mixt_negnatgrad(logl.clone(), gamma_z.clone(), n_k.clone())?;
        let newnorm: f64 = compute_norm(gamma_z.clone(), step.clone())?.abs().max(1e-7);
        let beta_fr: f64 = (newnorm.ln() - oldnorm.ln()).exp();
        oldnorm = newnorm;

        if didreset {
            oldstep = logl.zeros_like();
        } else {
            oldstep = oldstep.mul_scalar(beta_fr);
            step = step.add(oldstep.clone());
        }
        didreset = false;

        gamma_z = gamma_z.add(step.clone());

        let mut oldm = logsumexp(gamma_z.clone(), 0)?;
        gamma_z = gamma_z.sub(oldm.clone());

        n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone())?;
        let oldbound = bound;
        bound = bound_const + elbo_rcg_mat(logl.clone(), gamma_z.clone(), log_counts.clone(), n_k.clone())?;

        if bound < oldbound {
            didreset = true;
            gamma_z = gamma_z.add(oldm); // revert step
            if beta_fr > 0_f64 {
                gamma_z = gamma_z.sub(oldstep.clone());
            }

            oldm = logsumexp(gamma_z.clone(), 0)?;
            gamma_z = gamma_z.sub(oldm);
            n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone())?;

            bound = bound_const + elbo_rcg_mat(logl.clone(), gamma_z.clone(), log_counts.clone(), n_k.clone())?;
        } else {
            oldstep = step;
        }

        // if iter % 5 == 0 {
        //     eprintln!("\titer: {iter}, bound: {bound}, |g|: {newnorm}");
        // }

        if (bound - oldbound).abs() < tol && !didreset {
            oldm = logsumexp(gamma_z.clone(), 0)?;
            gamma_z = gamma_z.sub(oldm);
            break;
        }

        if newnorm < oldnorm {
            tol *= 10_f64;
        }

        iter += 1;
    }

    let m = logsumexp(gamma_z.clone(), 0)?;
    gamma_z = gamma_z.sub(m);

    Ok(gamma_z)
}

pub fn mixture_components<B: Backend>(
    gamma_z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
) -> Result<Tensor::<B, 1>, E> {
    let n_times_total = log_counts.clone().exp().sum().log().into_scalar();
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.reshape(Shape::new([1, gamma_z.dims()[1]]));
    let thetas = gamma_z.clone().add(log_counts_squeezed).exp().sum_dim(1).log().sub_scalar(n_times_total).exp().reshape(Shape::new([gamma_z.dims()[0], 1]));
    Ok(thetas)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn mixt_negnatgrad() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::mixt_negnatgrad;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.861124, -0.824187, -0.737067, -0.830991, -0.792902, -0.702885, -0.76075,  -0.719832, -0.622649, -0.742541 ],
                [ -1.01295,  -0.976009, -0.888889, -0.982813, -0.944725, -0.854708, -0.912572, -0.871654, -0.774472, -1.26242 ],
                [ -2.33926,  -2.30233,  -2.21521,  -2.67719,  -2.6391,   -2.54908,  -6.91527,  -6.87435,  -6.77717,  -2.22068 ],
                [ -2.13905,  -2.47017,  -6.69137,  -2.10891,  -2.43888,  -6.65719,  -2.03867,  -2.36581,  -6.57695,  -2.02046 ],
            ],
            &device,
        );

        let n_k = Tensor::<Backend, 1>::from_data(
            [
                4857.97, 3905.03, 701.053, 903.946,
            ],
            &device,
        );

        let logl = Tensor::<Backend, 2>::from_data(
            [
                [ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503 ],
                [ -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.0100503, -0.371713 ],
                [ -0.0100503, -0.0100503, -0.0100503, -0.371713,  -0.371713,  -0.371713,  -4.60517,   -4.60517,   -4.60517,   -0.0100503 ],
                [ -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503, -0.371713,  -4.60517,   -0.0100503 ],
            ],
            &device,
        );

        // mixt_negnatgrad should return the next value for `step`
        let expected = Tensor::<Backend, 2>::from_data(
            [
                [ 8.33935, 8.30241, 8.21529, 8.30921, 8.27113, 8.18111, 8.23897, 8.19806, 8.10087, 8.22076 ],
                [ 8.27279, 8.23585, 8.14873, 8.24266, 8.20457, 8.11455, 8.17241, 8.1315,  8.03431, 8.1606 ],
                [ 7.88108, 7.84415, 7.75703, 7.85735, 7.81926, 7.72924, 7.86197, 7.82105, 7.72387, 7.7625 ],
                [ 7.93521, 7.90467, 7.89242, 7.90508, 7.87339, 7.85823, 7.83484, 7.80032, 7.778,   7.81663 ],
            ],
            &device,
        );

        let got = mixt_negnatgrad::<Backend>(logl, gamma_z, n_k).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-5) });
    }

    #[test]
    fn compute_norm() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::compute_norm;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.861124, -0.824187, -0.737067, -0.830991, -0.792902, -0.702885, -0.76075,  -0.719832, -0.622649, -0.742541 ],
                [ -1.01295,  -0.976009, -0.888889, -0.982813, -0.944725, -0.854708, -0.912572, -0.871654, -0.774472, -1.26242 ],
                [ -2.33926,  -2.30233,  -2.21521,  -2.67719,  -2.6391,   -2.54908,  -6.91527,  -6.87435,  -6.77717,  -2.22068 ],
                [ -2.13905,  -2.47017,  -6.69137,  -2.10891,  -2.43888,  -6.65719,  -2.03867,  -2.36581,  -6.57695,  -2.02046 ],
            ],
            &device,
        );

        let dl_dphi = Tensor::<Backend, 2>::from_data(
            [
                [ 8.33935, 8.30241, 8.21529, 8.30921, 8.27113, 8.18111, 8.23897, 8.19806, 8.10087, 8.22076 ],
                [ 8.27279, 8.23585, 8.14873, 8.24266, 8.20457, 8.11455, 8.17241, 8.1315,  8.03431, 8.1606 ],
                [ 7.88108, 7.84415, 7.75703, 7.85735, 7.81926, 7.72924, 7.86197, 7.82105, 7.72387, 7.7625 ],
                [ 7.93521, 7.90467, 7.89242, 7.90508, 7.87339, 7.85823, 7.83484, 7.80032, 7.778,   7.81663 ],
            ],
            &device,
        );

        let expected: f64 = 0.193162;
        let got = compute_norm::<Backend>(gamma_z, dl_dphi).unwrap();

        assert_approx_eq!(expected, got, 1e-4);
    }

    #[test]
    fn update_n_k() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::update_n_k;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.681538, -0.662494, -0.617806, -0.667704, -0.648392, -0.603055, -0.635526, -0.615577, -0.568692, -0.557316 ],
                [ -0.951042, -0.931998, -0.887311, -0.937208, -0.917896, -0.872559, -0.905031, -0.885081, -0.838196, -1.18688 ],
                [ -3.09143,  -3.07238,  -3.0277,   -3.43766,  -3.41835,  -3.37301,  -7.62022,  -7.60027,  -7.55338,  -2.96721 ],
                [ -2.77441,  -3.11543,  -7.28548,  -2.76058,  -3.10133,  -7.27073,  -2.7284,   -3.06852,  -7.23637,  -2.65019 ],
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

        let expected = Tensor::<Backend, 1>::from_data(
            [
                5585.01, 3983.44, 327.192, 472.355
            ],
            &device,
        );

        let got = update_n_k::<Backend>(gamma_z, log_counts, alpha0).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-2) });
    }

    #[test]
    fn elbo_rcg_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::elbo_rcg_mat;

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

        let gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.681538, -0.662494, -0.617806, -0.667704, -0.648392, -0.603055, -0.635526, -0.615577, -0.568692, -0.557316 ],
                [ -0.951042, -0.931998, -0.887311, -0.937208, -0.917896, -0.872559, -0.905031, -0.885081, -0.838196, -1.18688 ],
                [ -3.09143,  -3.07238,  -3.0277,   -3.43766,  -3.41835,  -3.37301,  -7.62022,  -7.60027,  -7.55338,  -2.96721 ],
                [ -2.77441,  -3.11543,  -7.28548,  -2.76058,  -3.10133,  -7.27073,  -2.7284,   -3.06852,  -7.23637,  -2.65019 ],
            ],
            &device,
        );

        let n_k = Tensor::<Backend, 1>::from_data(
            [
                5585.01, 3983.44, 327.192, 472.355
            ],
            &device,
        );

        let bound_const = 85494_f64;
        let expected: f64 = -699.064 + bound_const;

        let got = elbo_rcg_mat::<Backend>(logl, gamma_z, log_counts, n_k).unwrap();

        assert_approx_eq!(expected, got, 1e-1);
    }

    #[test]
    fn calc_bound_const() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::calc_bound_const;

        let device = Default::default();
        type Backend = NdArray<f32>;

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

        let expected = 85494_f64;

        let got = calc_bound_const::<Backend>(log_counts, alpha0).unwrap();

        assert_approx_eq!(expected, got, 4_f64);
    }

    #[test]
    fn logsumexp() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::{Shape, Tensor};
        use burn_tensor::Int;

        use super::logsumexp;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let old_gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.861124, -0.824187, -0.737067, -0.830991, -0.792902, -0.702885, -0.76075,  -0.719832, -0.622649, -0.742541 ],
                [ -1.01295,  -0.976009, -0.888889, -0.982813, -0.944725, -0.854708, -0.912572, -0.871654, -0.774472, -1.26242 ],
                [ -2.33926,  -2.30233,  -2.21521,  -2.67719,  -2.6391,   -2.54908,  -6.91527,  -6.87435,  -6.77717,  -2.22068 ],
                [ -2.13905,  -2.47017,  -6.69137,  -2.10891,  -2.43888,  -6.65719,  -2.03867,  -2.36581,  -6.57695,  -2.02046 ],
            ],
            &device,
        );

        let expected = Tensor::<Backend, 2>::from_data(
            [
                [ -0.681538, -0.662494, -0.617806, -0.667704, -0.648392, -0.603055, -0.635526, -0.615577, -0.568692, -0.557316 ],
                [ -0.951042, -0.931998, -0.887311, -0.937208, -0.917896, -0.872559, -0.905031, -0.885081, -0.838196, -1.18688 ],
                [ -3.09143,  -3.07238,  -3.0277,   -3.43766,  -3.41835,  -3.37301,  -7.62022,  -7.60027,  -7.55338,  -2.96721 ],
                [ -2.77441,  -3.11543,  -7.28548,  -2.76058,  -3.10133,  -7.27073,  -2.7284,   -3.06852,  -7.23637,  -2.65019 ],
            ],
            &device,
        );

        let m = logsumexp::<Backend>(old_gamma_z.clone(), 0).unwrap();

        let got = old_gamma_z.sub(m);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }

    #[test]
    fn rcg_optl_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::{Shape, Tensor};
        use burn_tensor::Int;

        use super::rcg_optl_mat;

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

        let got = rcg_optl_mat::<Backend>(logl, log_counts, alpha0, 1e-7_f64, 100_usize).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }
}
