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

//! Tensor math used in [optimizer](crate::optimizer) algorithms

use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// Approximate the derivative of the log gamma function (digamma)
///
/// Based on the
/// [statrs::function::gamma::digamma](https://docs.rs/statrs/0.18.0/src/statrs/function/gamma.rs.html#373-412)
/// source code which uses "Algorithm AS 103" from Jose Bernardo, Applied
/// Statistics, Volume 25, Number 3, 1976, pages 315 - 317.
///
/// ## Notes
///
/// Does not work for negative inputs or very small (<1e-6) inputs.
///
/// It is possible to extend the code to work on these inputs, see the statrs
/// code.
///
pub fn digamma_tensor<B: Backend>(
    tensor: Tensor::<B, 1>,
) -> Tensor::<B, 1> {
    const S3: f64 = 1.0 / 12.0;
    const S4: f64 = 1.0 / 120.0;
    const S5: f64 = 1.0 / 252.0;
    const S6: f64 = 1.0 / 240.0;
    const S7: f64 = 1.0 / 132.0;

    let mut result = tensor.zeros_like();
    let mut z = tensor.clone();
    let mut mask = tensor.clone().lower_elem(12.0);
    for _ in 0..12 {
        result = result.clone().mask_where(mask.clone(), result.sub(z.clone().recip()));
        z = z.clone().mask_where(mask, z.add_scalar(1.0));
        mask = tensor.clone().lower_elem(12.0);
    }

    mask = z.clone().greater_equal_elem(12.0);
    let mut r = z.clone().mask_where(mask.clone(), z.clone().recip());
    result = result.clone().mask_where(mask.clone(), result.add(z.log()).sub(r.clone().mul_scalar(0.5)));
    r = r.clone().mask_where(mask.clone(), r.square().mul_scalar(-1.0));

    result.clone().mask_where(mask, result.sub(
        r.clone().mul_scalar(S7).add_scalar(S6).mul(r.clone()).add_scalar(S5).mul(r.clone()).add_scalar(S4).mul(r.clone()).add_scalar(S3).mul(r).mul_scalar(-1.0)))
}

/// Approximate the log-gamma function
///
/// Based on the
/// [statrs::function::gamma::ln_gamma](https://docs.rs/statrs/0.18.0/src/statrs/function/gamma.rs.html#54-78)
/// source code which is derived from "An Analysis of the Lanczos Gamma
/// Approximation", Glendon Ralph Pugh, 2004 p. 116
///
/// ## Notes
///
/// Computes the logarithm of the gamma function with an accuracy of 16 floating
/// point digits.
///
pub fn ln_gamma_tensor<B: Backend>(
    tensor: Tensor::<B, 1>,
) -> Tensor::<B, 1> {
    // Constants
    const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452;
    const GAMMA_R: f64  = 10.900511;

    // Polynomial coefficients for approximating the `gamma_ln` function
    const C0: f64 = 2.4857408913875356e-5;
    const C1: f64 = 1.0514237858172197;
    const C2: f64 = -3.4568709722201623;
    const C3: f64 = 4.512277094668948;
    const C4: f64 = -2.9828522532357665;
    const C5: f64 = 1.056397115771267;
    const C6: f64 = -1.9542877319164586e-1;
    const C7: f64 = 1.709705434044412e-2;
    const C8: f64 = -5.719261174043057e-4;
    const C9: f64 = 4.633994733599056e-6;
    const C10: f64 = -2.719949084886077e-9;

    // Compute for elements < 0.5
    let mask = tensor.clone().lower_elem(0.5);
    let tensor_neg = tensor.clone().neg();
    let s1 = tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(1.0).recip().mul_scalar(C1)
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(2.0).recip().mul_scalar(C2)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(3.0).recip().mul_scalar(C3)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(4.0).recip().mul_scalar(C4)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(5.0).recip().mul_scalar(C5)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(6.0).recip().mul_scalar(C6)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(7.0).recip().mul_scalar(C7)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(8.0).recip().mul_scalar(C8)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(9.0).recip().mul_scalar(C9)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(10.0).recip().mul_scalar(C10)))
                          .add_scalar(C0)
                          .log()
                          .add_scalar(LN_2_SQRT_E_OVER_PI)
                          .sub_scalar(std::f64::consts::PI.ln())
    );
    let s1 = s1.clone().zeros_like()
                       .mask_where(mask.clone(), tensor.clone()
                       .mul_scalar(std::f64::consts::PI)
                       .sin()
                       .log()
                       .neg()
                       .sub(s1)
                       );
    let temp = tensor_neg.clone().mask_where(mask.clone(), tensor_neg.clone().add_scalar(0.5));
    let temp = tensor_neg.clone().mask_where(mask.clone(), tensor_neg.add_scalar(0.5 + GAMMA_R).div_scalar(std::f64::consts::E).log().mul(temp));
    let result = s1.clone().mask_where(mask.clone(),
                                      s1.sub(temp));

    // Compute for elements >= 0.5
    let mask = mask.bool_not();
    let s2 = tensor.clone().mask_where(mask.clone(), tensor.clone().recip().mul_scalar(C1)
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(1.0).recip().mul_scalar(C2)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(2.0).recip().mul_scalar(C3)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(3.0).recip().mul_scalar(C4)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(4.0).recip().mul_scalar(C5)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(5.0).recip().mul_scalar(C6)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(6.0).recip().mul_scalar(C7)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(7.0).recip().mul_scalar(C8)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(8.0).recip().mul_scalar(C9)))
                          .add(tensor.clone().mask_where(mask.clone(), tensor.clone().add_scalar(9.0).recip().mul_scalar(C10)))
                          .add_scalar(C0)
                          .log()
                          .add_scalar(LN_2_SQRT_E_OVER_PI));
    let temp = tensor.clone().mask_where(mask.clone(), tensor.clone().sub_scalar(0.5));
    let temp = tensor.clone().mask_where(mask.clone(), tensor.add_scalar(GAMMA_R - 0.5).div_scalar(std::f64::consts::E).log().mul(temp));

    result.mask_where(mask, s2.add(temp))
}

/// Log of the sum of exponentials over a dimension
pub fn logsumexp<B: Backend>(
    input: Tensor::<B, 2>,
    dim: usize,
) -> Tensor<B, 2> {
    let max = input.clone().max_dim(dim);
    let res = (input - max.clone()).exp().sum_dim(dim).log();
    res + max
}

/// Log of the sum of exponentials over a dimension
pub fn logsumexp_mask<B: Backend>(
    input: Tensor::<B, 2>,
    max_mask: Tensor::<B, 2, burn_tensor::Bool>,
) -> Tensor<B, 1> {
    let max = input.clone().mask_where(max_mask, input.clone()).max();
    let res = (input - max.clone().unsqueeze()).exp().sum().log();
    res + max
}

// Tests
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn digamma_tensor() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;
        use statrs::function::gamma::digamma;

        use super::digamma_tensor;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let n_k_data = vec![4857.97, 3905.03, 701.053, 903.946];
        let n_k = Tensor::<Backend, 1>::from_data(
            n_k_data.as_slice(),
            &device,
        );

        // mixt_negnatgrad should return the next value for `step`
        let expected = Tensor::<Backend, 1>::from_data(
            n_k_data.iter().map(|x| digamma(*x as f64)).collect::<Vec<f64>>().as_slice(),
            &device,
        );

        let got = digamma_tensor::<Backend>(n_k);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-16) });
    }


    #[test]
    fn ln_gamma_tensor_small() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;
        use statrs::function::gamma::ln_gamma;

        use super::ln_gamma_tensor;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let input_data = vec![0.33440901, 0.41306856, 0.06975968, 0.36323925, 0.18675239];
        let input = Tensor::<Backend, 1>::from_data(
            input_data.as_slice(),
            &device,
        );

        let expected = Tensor::<Backend, 1>::from_data(
            input_data.iter().map(|x| ln_gamma(*x as f64)).collect::<Vec<f64>>().as_slice(),
            &device,
        );

        let got = ln_gamma_tensor::<Backend>(input);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-6) });
    }

    #[test]
    fn ln_gamma_tensor_large() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;
        use statrs::function::gamma::ln_gamma;

        use super::ln_gamma_tensor;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let input_data = vec![1.33440901, 10.41306856, 100.06975968, 1000.36323925, 10000.18675239];
        let input = Tensor::<Backend, 1>::from_data(
            input_data.as_slice(),
            &device,
        );

        let expected = Tensor::<Backend, 1>::from_data(
            input_data.iter().map(|x| ln_gamma(*x as f64)).collect::<Vec<f64>>().as_slice(),
            &device,
        );

        let got = ln_gamma_tensor::<Backend>(input);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-2) });
    }

    #[test]
    fn ln_gamma_tensor_mixed() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;
        use statrs::function::gamma::ln_gamma;

        use super::ln_gamma_tensor;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let input_data = vec![100.33440901, 0.41306856, 1.06975968, 1.36323925, 0.5];
        let input = Tensor::<Backend, 1>::from_data(
            input_data.as_slice(),
            &device,
        );

        let expected = Tensor::<Backend, 1>::from_data(
            input_data.iter().map(|x| ln_gamma(*x as f64)).collect::<Vec<f64>>().as_slice(),
            &device,
        );

        let got = ln_gamma_tensor::<Backend>(input);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-4) });
    }

    #[test]
    fn logsumexp() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::Tensor;

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

        let m = logsumexp::<Backend>(old_gamma_z.clone(), 0);

        let got = old_gamma_z.sub(m);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }
}
