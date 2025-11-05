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
use burn_tensor::backend::Device;
use burn_tensor::{Shape, Tensor};
use statrs::function::gamma::digamma;
use statrs::function::gamma::ln_gamma;

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

pub fn newnorm<B: Backend<FloatElem = f32>>(
    gamma_Z: Tensor::<B, 2>,
    dl_dphi: Tensor::<B, 2>,
) -> Result<f32, E> {
    let temp = gamma_Z.exp().mul(dl_dphi.clone());

    let colsums = temp.clone().sum_dim(0);
    let colsums_squeezed: Tensor::<B, 2> = colsums.clone().reshape(Shape::new([1, dl_dphi.clone().dims()[1]]));

    let newnorm = temp.mul(dl_dphi.sub(colsums_squeezed)).sum().into_scalar();

    Ok(newnorm)
}

pub fn mixt_negnatgrad<B: Backend, D: Device>(
    logl: Tensor::<B, 2>,
    gamma_Z: Tensor::<B, 2>,
    n_k: Tensor::<B, 1>,
) -> Result<Tensor::<B, 2>, E> {

    let n_k_data = n_k.clone().into_data();
    let digamma_n_k_vals = n_k_data.iter().map(|x: f32| (digamma(x as f64) - 1_f64) as f32).collect::<Vec<f32>>();
    let digamma_n_k = Tensor::<B, 1>::from_data(digamma_n_k_vals.as_slice(), &n_k.device());
    let digamma_n_k_squeezed: Tensor::<B, 2> = digamma_n_k.reshape(Shape::new([logl.dims()[0], 1]));

    let dl_dphi: Tensor::<B, 2> = logl.add(digamma_n_k_squeezed).sub(gamma_Z.clone());

    Ok(dl_dphi)
}

pub fn update_n_k<B: Backend>(
    gamma_Z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
) -> Result<Tensor::<B, 1>, E> {
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.reshape(Shape::new([1, gamma_Z.clone().dims()[1]]));
    let n_k: Tensor::<B, 1> = gamma_Z.add(log_counts_squeezed).exp().sum_dim(1).reshape(Shape::new([alpha0.clone().dims()[0]])).add(alpha0);
    Ok(n_k)
}

pub fn elbo_rcg_mat<B: Backend<FloatElem = f32>>(
    logl: Tensor::<B, 2>,
    gamma_Z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    n_k: Tensor::<B, 1>,
) -> Result<f32, E> {
    let log_counts_squeezed: Tensor::<B, 2> = log_counts.reshape(Shape::new([1, gamma_Z.clone().dims()[1]]));
    let n_k_data = n_k.into_data();
    let lgamma_n_k_vals = n_k_data.iter().map(|x: f32| (ln_gamma(x as f64)) as f32).collect::<Vec<f32>>();

    let bound = gamma_Z.clone().add(log_counts_squeezed).exp().mul(logl.sub(gamma_Z)).sum();
    let lgamma_sum = lgamma_n_k_vals.into_iter().sum::<f32>();

    let newbound: f32 = bound.into_scalar() + lgamma_sum;

    Ok(newbound)
}

pub fn calc_bound_const<B: Backend<FloatElem = f32>>(
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
) -> Result<f32, E> {
    let counts_sum: f64 = log_counts.exp().sum().into_scalar() as f64;
    let alpha0_sum: f64 = alpha0.clone().sum().into_scalar() as f64;
    let alpha0_data = alpha0.into_data();
    let lgamma_alpha0_sum = alpha0_data.iter().map(|x: f32| (ln_gamma(x as f64))).sum::<f64>();

    let bound_const = ln_gamma(alpha0_sum) + ln_gamma(alpha0_sum + counts_sum) - lgamma_alpha0_sum;
    Ok(bound_const as f32)
}

pub fn rcg_optl_mat(

) -> Result<(), E> {
    todo!("Implement rcg_optl_mat");
    Ok(())
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn mixt_negnatgrad() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::mixt_negnatgrad;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_Z = Tensor::<Backend, 2>::from_data(
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

        let got = mixt_negnatgrad::<Backend, NdArrayDevice>(logl, gamma_Z, n_k).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-5) });
    }

    #[test]
    fn newnorm() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::Tensor;
        use burn_tensor::Int;

        use super::newnorm;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_Z = Tensor::<Backend, 2>::from_data(
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

        let expected: f32 = 0.193162;
        let got = newnorm::<Backend>(gamma_Z, dl_dphi).unwrap();

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

        let gamma_Z = Tensor::<Backend, 2>::from_data(
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

        let got = update_n_k::<Backend>(gamma_Z, log_counts, alpha0).unwrap();

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

        let gamma_Z = Tensor::<Backend, 2>::from_data(
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

        let bound_const = 85494_f32;
        let expected: f32 = -699.064 + bound_const;

        let got = elbo_rcg_mat::<Backend>(logl, gamma_Z, log_counts, n_k).unwrap();

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

        let expected = 85494_f32;

        let got = calc_bound_const::<Backend>(log_counts, alpha0).unwrap();

        assert_approx_eq!(expected, got, 4_f32);
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

        let old_gamma_Z = Tensor::<Backend, 2>::from_data(
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

        let m = logsumexp::<Backend>(old_gamma_Z.clone(), 0).unwrap();

        let m_squeezed: Tensor::<Backend, 2> = m.reshape(Shape::new([1, old_gamma_Z.clone().dims()[1]]));
        let got = old_gamma_Z.sub(m_squeezed);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-7) });
    }
}
