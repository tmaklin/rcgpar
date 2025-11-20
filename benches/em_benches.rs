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

fn em_algorithm_bench(c: &mut Criterion) {
    use rcgpar::optimizer::em::em_algorithm;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 41;
    let n: usize = 1109;

    let (log_lls, _) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f32> = sample_n_poisson(100_f32, n, &mut rng).iter().map(|x| x.ln()).collect();

    let device = Default::default();
    type Backend = NdArray<f32>;

    let logl = Tensor::<Backend, 1>::from_data(log_lls.as_slice(), &device);
    let logl = logl.reshape([k, n]);
    let log_counts = Tensor::<Backend, 1>::from_data(log_counts.as_slice(), &device);

    c.bench_function("em_algorithm", |b|
                     b.iter(||
                            em_algorithm(black_box(logl.clone()), log_counts.clone(), 1e-7_f64, 5000_usize)
                     ));
}

criterion_group!(em_benches,
                 em_algorithm_bench,
);
criterion_main!(em_benches);

// util

use rand::RngCore;

use rand_distr::Distribution;
use rand_distr::{Gamma, Normal, Poisson, Uniform};
use rand_distr::weighted::WeightedIndex;

use statrs::distribution::Continuous;

/// Sample a single value from the Dirichlet distribution
fn sample_dirichlet(
    alphas: &[f32],
    rng: &mut dyn RngCore,
) -> Vec<f32> {
    let k = alphas.len();

    let mut y_sum: f32 = 0.0;
    let ys: Vec<f32> = (0..k).map(|idx| {
        let gamma = Gamma::new(alphas[idx], 1.0).unwrap();
        let y = gamma.sample(rng);
        y_sum += y;
        y
    }).collect();

    ys.iter().map(|y| y/y_sum).collect::<Vec<f32>>()
}

/// Sample n values from the normal distribution
fn sample_n_normal(
    mean: f32,
    sd: f32,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f32> {
    let normal = Normal::new(mean, sd).unwrap();
    (0..n).map(|_| normal.sample(rng)).collect::<Vec<f32>>()
}

/// Sample n values from the gamma distribution
fn sample_n_gamma(
    shape: f32,
    scale: f32,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f32> {
    let gamma = Gamma::new(shape, scale).unwrap();
    (0..n).map(|_| gamma.sample(rng)).collect::<Vec<f32>>()
}

/// Sample n values from the uniform distribution
fn sample_n_uniform(
    min: f32,
    max: f32,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f32> {
    let uniform = Uniform::new(min, max).unwrap();
    (0..n).map(|_| uniform.sample(rng)).collect::<Vec<f32>>()
}

/// Sample n values from the poission
fn sample_n_poisson(
    rate: f32,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f32> {
    let poisson = Poisson::new(rate).unwrap();
    (0..n).map(|_| poisson.sample(rng)).collect::<Vec<f32>>()
}

fn random_loglls(
    k: usize,
    n: usize,
    rng: &mut dyn RngCore,
) -> (Vec<f32>, Vec<f32>) {
    // Normal distribution parameters to generate observations
    let means: Vec<f32> = sample_n_normal(0_f32, 10_f32, k, rng);
    let sds: Vec<f32> = sample_n_gamma(1_f32, 2_f32, k, rng).iter().map(|x| x.sqrt()).collect();

    let normals: Vec<_> = means.iter().zip(sds.iter()).map(|(mu, sigma)| {
        statrs::distribution::Normal::new(*mu as f64, *sigma as f64).unwrap()
    }).collect();

    // Generate random thetas ~ Dirichlet(alpha_1, ..., alpha_k) by sampling from
    // Gamma(alpha_i, 1) distributions, where alpha_1 ~ Unif(0, 1)
    //
    // This tends to produce thetas that are concentrated around a few values
    let alphas: Vec<f32> = sample_n_uniform(0_f32, 1_f32, k, rng);
    let thetas: Vec<f32> = sample_dirichlet(&alphas, rng);

    // Generate log likelihoods for a mixture of `k` normal distributions
    let dist = WeightedIndex::new(&thetas).unwrap();
    let mut log_lls: Vec<Vec<f32>> = vec![vec![0_f32; n]; k];
    for i in 0..n {
        let cluster: usize = dist.sample(rng);
        let obs: f32 = sample_n_normal(means[cluster], sds[cluster], 1, rng)[0];
        for j in 0..k {
            log_lls[j][i] = normals[j].ln_pdf(obs as f64) as f32;
        }
    }
    let log_lls: Vec<f32> = log_lls.iter().cloned().flatten().collect();

    (log_lls, thetas)
}
