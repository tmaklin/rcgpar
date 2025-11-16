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

use crate::math::digamma_tensor;
use crate::math::ln_gamma_tensor;
use crate::math::logsumexp;

use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

type E = Box<dyn std::error::Error>;


pub fn compute_norm<B: Backend>(
    gamma_z: Tensor::<B, 2>,
    dl_dphi: Tensor::<B, 2>,
) -> Tensor::<B, 1> {
    let temp = gamma_z.exp().mul(dl_dphi.clone());
    let colsums = temp.clone().sum_dim(0);
    temp.mul(dl_dphi.sub(colsums.unsqueeze())).sum()
}

pub fn mixt_negnatgrad<B: Backend>(
    logl: Tensor::<B, 2>,
    gamma_z: Tensor::<B, 2>,
    n_k: Tensor::<B, 1>,
) -> Tensor::<B, 2> {
    let digamma_n_k = digamma_tensor(n_k).sub_scalar(1.0);
    logl.add(digamma_n_k.unsqueeze().swap_dims(0, 1)).sub(gamma_z)
}

pub fn update_n_k<B: Backend>(
    gamma_z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
) -> Tensor::<B, 1> {
    gamma_z.add(log_counts.unsqueeze()).exp().sum_dim(1).squeeze().add(alpha0)
}

pub fn elbo_rcg_mat<B: Backend>(
    logl: Tensor::<B, 2>,
    gamma_z: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    n_k: Tensor::<B, 1>,
) -> Tensor::<B, 1> {
    let lgamma_n_k_sum = ln_gamma_tensor(n_k).sum();

    let bound = gamma_z.clone().add(log_counts.unsqueeze()).exp().mul(logl.sub(gamma_z)).sum();
    bound.add(lgamma_n_k_sum)
}

pub fn revert_step<B: Backend>(
    mut gamma_z: Tensor::<B, 2>,
    oldstep: Tensor::<B, 2>,
    mut oldm: Tensor::<B, 2>,
) -> Tensor::<B, 2> {
    gamma_z = gamma_z.add(oldm).sub(oldstep);
    oldm = logsumexp(gamma_z.clone(), 0);
    gamma_z.sub(oldm)
}

pub fn rcg_optl_mat<B: Backend>(
    logl: Tensor::<B, 2>,
    log_counts: Tensor::<B, 1>,
    alpha0: Tensor::<B, 1>,
    tolerance: f64,
    max_iters: usize,
) -> Result<Tensor::<B, 2>, E> {
    let mut gamma_z = logl.zeros_like() + (1_f64 / (logl.dims()[0] as f64)).ln();
    let mut oldstep = logl.zeros_like();

    let mut iter = 0;

    let mut oldbound = Tensor::<B, 1>::from_data([f64::MIN], &logl.device());

    let mut n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone());

    let mut oldnorm_t = Tensor::<B, 1>::from_data([1_f64], &logl.device());

    let mut diff: f64 = 1000_f64;

    while iter < max_iters {
        let mut step = mixt_negnatgrad(logl.clone(), gamma_z.clone(), n_k.clone());
        let newnorm_t = compute_norm(gamma_z.clone(), step.clone()).abs();

        if diff < 0_f64 {
            oldstep = logl.zeros_like();
        } else {
            let beta_fr_t = newnorm_t.clone().log().sub(oldnorm_t.log()).exp();
            oldstep = oldstep.mul(beta_fr_t.unsqueeze());
            step = step.add(oldstep.clone());
        }
        oldnorm_t = newnorm_t;

        gamma_z = gamma_z.add(step.clone());

        let mut oldm = logsumexp(gamma_z.clone(), 0);
        gamma_z = gamma_z.sub(oldm.clone());

        n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone());
        let bound = elbo_rcg_mat(logl.clone(), gamma_z.clone(), log_counts.clone(), n_k.clone());

        diff = bound.clone().sub(oldbound.clone()).into_data().iter().next().unwrap();
        if diff < 0_f64 {
            gamma_z = revert_step(gamma_z, oldstep.clone(), oldm);
            n_k = update_n_k(gamma_z.clone(), log_counts.clone(), alpha0.clone());
            oldbound = elbo_rcg_mat(logl.clone(), gamma_z.clone(), log_counts.clone(), n_k.clone());
        } else {
            oldstep = step;
            oldbound = bound;
        }

        // if iter % 5 == 0 {
        //     eprintln!("\titer: {iter}, bound: {bound}, |g|: {newnorm}");
        // }

        if diff >= 0_f64 && diff < tolerance {
            oldm = logsumexp(gamma_z.clone(), 0);
            gamma_z = gamma_z.sub(oldm);
            break;
        }

        iter += 1;
    }

    let m = logsumexp(gamma_z.clone(), 0);
    gamma_z = gamma_z.sub(m);

    Ok(gamma_z)
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn mixt_negnatgrad() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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

        let got = mixt_negnatgrad::<Backend>(logl, gamma_z, n_k);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-5) });
    }

    #[test]
    fn compute_norm() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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
        let got: f64 = compute_norm::<Backend>(gamma_z, dl_dphi).into_data().iter().next().unwrap();

        assert_approx_eq!(expected, got, 1e-4);
    }

    #[test]
    fn update_n_k() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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

        let got = update_n_k::<Backend>(gamma_z, log_counts, alpha0);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-2) });
    }

    #[test]
    fn elbo_rcg_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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

        let expected = -699.064_f64 + 85494_f64;
        let got: f64 = elbo_rcg_mat::<Backend>(logl, gamma_z, log_counts, n_k).into_data().iter().next().unwrap();
        assert_approx_eq!(expected, got, 1e-1);
    }

    #[test]
    fn revert_step() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

        use super::revert_step;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_z = Tensor::<Backend, 2>::from_data(
            [
                [-0.0011119843, -0.0010623932, -0.0009498596, -0.0010662079, -0.0010166168, -0.0009059906, -0.0009651184, -0.00091552734, -0.0008049011, -0.0008678436],
                [-7.130493, -7.1304436, -7.130331, -7.130451, -7.1303997, -7.130287, -7.1303463, -7.1302986, -7.130188, -7.491913],
                [-8.822966, -8.822918, -8.8228035, -9.184584, -9.184534, -9.184422, -13.417935, -13.417886, -13.417775, -8.8227215],
                [-8.722033, -9.083645, -13.316987, -8.721989, -9.083599, -13.316945, -8.721882, -9.083494, -13.316844, -8.721788],
            ],
            &device,
        );

        let oldstep = Tensor::<Backend, 2>::from_data(
            [
                [20.177214, 20.177094, 20.176817, 20.177105, 20.176983, 20.176708, 20.176857, 20.176735, 20.17646, 20.17661],
                [20.171112, 20.17099, 20.170715, 20.171001, 20.17088, 20.170609, 20.170753, 20.170631, 20.170357, 20.170507],
                [20.177217, 20.177097, 20.176823, 20.17711, 20.176989, 20.176714, 20.176863, 20.17674, 20.176468, 20.176615],
                [20.177156, 20.177038, 20.176762, 20.177048, 20.176928, 20.176651, 20.1768, 20.17668, 20.176403, 20.176554],
            ],
            &device,
        );

        let oldm = Tensor::<Backend, 2>::from_data(
            [
                [28.41345, 28.41328, 28.412891, 28.413298, 28.413126, 28.412739, 28.412945, 28.412773, 28.41239, 28.412603],
            ],
            &device,
        );

        let expected = Tensor::<Backend, 2>::from_data(
            [
                [-0.001115799, -0.0010662079, -0.000954628, -0.0010719299, -0.0010223389, -0.0009098053, -0.0009698868, -0.0009202957, -0.0008087158, -0.0008716583],
                [-7.1243954, -7.124344, -7.124234, -7.1243534, -7.124302, -7.1241913, -7.1242476, -7.1242, -7.1240883, -7.485813],
                [-8.822973, -8.822926, -8.822814, -9.184595, -9.1845455, -9.184431, -13.417946, -13.417896, -13.417787, -8.822729],
                [-8.721979, -9.083593, -13.3169365, -8.721937, -9.0835495, -13.316892, -8.721829, -9.083444, -13.316791, -8.721735],
            ],
                &device,
        );

        let got = revert_step::<Backend>(gamma_z, oldstep, oldm);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }

    #[test]
    fn rcg_optl_mat() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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
