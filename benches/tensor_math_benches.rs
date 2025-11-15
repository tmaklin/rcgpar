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

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

use burn::backend::ndarray::NdArray;
use burn_tensor::Tensor;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn digamma_tensor_bench(c: &mut Criterion) {
    use rcgpar::math::digamma_tensor;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 41;

    let n_k: Vec<f64> = sample_n_gamma(4000_f64, 1_f64, k, &mut rng).iter().map(|x| x.ln()).collect();

    let device = Default::default();
    type Backend = NdArray<f64>;

    let n_k = Tensor::<Backend, 1>::from_data(n_k.as_slice(), &device);

    c.bench_function("digamma_tensor", |b|
                     b.iter(||
                            digamma_tensor(black_box(n_k.clone()))
                     ));
}

fn ln_gamma_tensor_bench(c: &mut Criterion) {
    use rcgpar::math::ln_gamma_tensor;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 41;

    let n_k: Vec<f64> = sample_n_gamma(4000_f64, 1_f64, k, &mut rng).iter().map(|x| x.ln()).collect();

    let device = Default::default();
    type Backend = NdArray<f64>;

    let n_k = Tensor::<Backend, 1>::from_data(n_k.as_slice(), &device);

    c.bench_function("ln_gamma_tensor", |b|
                     b.iter(||
                            ln_gamma_tensor(black_box(n_k.clone()))
                     ));
}

criterion_group!(tensor_math_benches,
                 digamma_tensor_bench,
                 ln_gamma_tensor_bench,
);
criterion_main!(tensor_math_benches);

// util

use rand::RngCore;

use rand_distr::Distribution;
use rand_distr::Gamma;

/// Sample n values from the gamma distribution
fn sample_n_gamma(
    shape: f64,
    scale: f64,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let gamma = Gamma::new(shape, scale).unwrap();
    (0..n).map(|_| gamma.sample(rng)).collect::<Vec<f64>>()
}
