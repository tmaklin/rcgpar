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

type E = Box<dyn std::error::Error>;

/// Approximate the derivative of the log gamma function (digamma)
///
/// Based on the
/// [statrs::function::gamma::digamma](https://docs.rs/statrs/0.18.0/src/statrs/function/gamma.rs.html#373-412)
/// source code at: which uses "Algorithm AS 103" from Jose Bernardo, Applied
/// Statistics, Volume 25, Number 3, 1976, pages 315 - 317.
///
/// ## Notes
/// Does not work for negative inputs or very small (<1e-6) inputs.
///
/// It is possible to extend the code to work on these inputs, see the statrs
/// code.
///
pub fn digamma_tensor<B: Backend>(
    tensor: Tensor::<B, 1>,
) -> Result<Tensor::<B, 1>, E> {
    let c = 12.0;
    let s3 = 1.0 / 12.0;
    let s4 = 1.0 / 120.0;
    let s5 = 1.0 / 252.0;
    let s6 = 1.0 / 240.0;
    let s7 = 1.0 / 132.0;

    let mut result = tensor.zeros_like();
    let mut z = tensor.clone();
    let mut mask = tensor.clone().lower_elem(c);
    for _ in 0..12 {
        result = result.clone().mask_where(mask.clone(), result.sub(z.clone().recip()));
        z = z.clone().mask_where(mask, z.add_scalar(1.0));
        mask = tensor.clone().lower_elem(c);
    }

    mask = z.clone().greater_equal_elem(c);
    let mut r = z.clone().mask_where(mask.clone(), z.clone().recip());
    result = result.clone().mask_where(mask.clone(), result.add(z.log()).sub(r.clone().mul_scalar(0.5)));
    r = r.clone().mask_where(mask.clone(), r.square().mul_scalar(-1.0));

    result = result.clone().mask_where(mask, result.sub(
            r.clone().mul_scalar(s7).add_scalar(s6).mul(r.clone()).add_scalar(s5).mul(r.clone()).add_scalar(s4).mul(r.clone()).add_scalar(s3).mul(r).mul_scalar(-1.0)));

    Ok(result)
}

/// Log of the sum of exponentials over a dimension
pub fn logsumexp<B: Backend>(
    input: Tensor::<B, 2>,
    dim: usize,
) -> Result<Tensor<B, 2>, E> {
    let max = input.clone().max_dim(dim);
    let res = (input - max.clone()).exp().sum_dim(dim).log();
    let res = res + max;
    Ok(res)
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

        let got = digamma_tensor::<Backend>(n_k).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-16) });
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

        let m = logsumexp::<Backend>(old_gamma_z.clone(), 0).unwrap();

        let got = old_gamma_z.sub(m);

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1_f32) });
    }
}
