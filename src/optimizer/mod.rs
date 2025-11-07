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

//! Algorithm and helper function implementations for rcgpar

pub mod em;
pub mod rcg;

use burn_tensor::{Shape, Tensor};
use burn_tensor::backend::Backend;

type E = Box<dyn std::error::Error>;

/// Compute mixture components from a fitted probability matrix
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
    fn mixture_components() {
        use burn::backend::ndarray::NdArray;
        use burn_tensor::backend::Device;
        use burn::backend::ndarray::NdArrayDevice;
        use burn_tensor::{Shape, Tensor};
        use burn_tensor::Int;

        use super::mixture_components;

        let device = Default::default();
        type Backend = NdArray<f32>;

        let gamma_Z = Tensor::<Backend, 2>::from_data(
            [
                [ -0.0010899, -0.00104044, -0.000928571, -0.00104519, -0.000995734, -0.000883857, -0.000944069, -0.000894604, -0.000782716, -0.000853449 ],
                [ -7.15745,   -7.1574,     -7.15729,     -7.15741,    -7.15736,     -7.15725,     -7.15731,     -7.15726,     -7.15715,     -7.51888 ],
                [ -8.82298,   -8.82293,    -8.82282,     -9.1846,     -9.18455,     -9.18444,     -13.418,      -13.4179,     -13.4178,     -8.82274 ],
                [ -8.72199,   -9.0836,     -13.3169,     -8.72195,    -9.08356,     -13.3169,     -8.72184,     -9.08346,     -13.3168,     -8.72175 ],
            ],
            &device,
        );

        let log_counts = Tensor::<Backend, 1>::from_data(
            [
                7.681099, 7.04316, 6.849066, 5.278115, 5.164786, 5.062595, 6.947937, 6.863803, 7.277248, 7.666222
            ],
            &device,
        );

        let expected = Tensor::<Backend, 1>::from_data(
            [
                0.999543, 0.00073079, 9.66135e-05, 0.000112505
            ],
            &device,
        );

        let got = mixture_components::<Backend>(gamma_Z, log_counts).unwrap();

        let got_data = got.into_data();
        let expected_data = expected.into_data();

        got_data.iter().zip(expected_data.iter()).for_each(|(x, y): (f32, f32)| { assert_approx_eq!(x, y, 1e-2_f32) });
    }
}
