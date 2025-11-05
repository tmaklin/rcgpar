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

type E = Box<dyn std::error::Error>;

pub fn logsumexp(

) -> Result<(), E> {
    todo!("Implement logsumexp");
    Ok(())
}

pub fn newnorm<B: Backend<FloatElem = f32>>(
    gamma_Z: Tensor::<B, 2>,
    dl_dphi: Tensor::<B, 2>,
) -> Result<(), E> {
    todo!("Implement newnorm");

    Ok(())
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

pub fn update_N_k(

) -> Result<(), E> {
    todo!("Implement update_N_k");
    Ok(())
}

pub fn elbo_rcg_mat(

) -> Result<(), E> {
    todo!("Implement ELBO_rcg_mat");
    Ok(())
}

pub fn calc_bound_const(

) -> Result<(), E> {
    todo!("Implement calc_bound_const");
    Ok(())
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
}
