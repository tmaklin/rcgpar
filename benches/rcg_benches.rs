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

fn compute_norm_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::compute_norm;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 5;
    let n: usize = 10;

    let (gamma_z, _) = random_loglls(k, n, &mut rng);
    let dl_dphi = sample_n_uniform(-1_f64, 1_f64, n * k, &mut rng);

    let device = Default::default();
    type Backend = NdArray<f64>;

    let gamma_z = Tensor::<Backend, 1>::from_data(gamma_z.as_slice(), &device);
    let gamma_z = gamma_z.reshape([k, n]);
    let dl_dphi = Tensor::<Backend, 1>::from_data(dl_dphi.as_slice(), &device);
    let dl_dphi = dl_dphi.reshape([k, n]);

    c.bench_function("compute_norm 5x10", |b|
                     b.iter(||
                            compute_norm(black_box(gamma_z.clone()), dl_dphi.clone())
                     ));
}

fn mixt_negnatgrad_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::mixt_negnatgrad;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 5;
    let n: usize = 10;

    let (log_lls, _) = random_loglls(k, n, &mut rng);
    let (gamma_z, _) = random_loglls(k, n, &mut rng);
    let n_k: Vec<f64> = sample_n_gamma(4000_f64, 1_f64, k, &mut rng).iter().map(|x| x.ln()).collect();

    let device = Default::default();
    type Backend = NdArray<f64>;

    let logl = Tensor::<Backend, 1>::from_data(log_lls.as_slice(), &device);
    let logl = logl.reshape([k, n]);
    let gamma_z = Tensor::<Backend, 1>::from_data(gamma_z.as_slice(), &device);
    let gamma_z = gamma_z.reshape([k, n]);
    let n_k = Tensor::<Backend, 1>::from_data(n_k.as_slice(), &device);

    c.bench_function("mixt_negnatgrad 5x10", |b|
                     b.iter(||
                            mixt_negnatgrad(black_box(logl.clone()), gamma_z.clone(), n_k.clone())
                     ));
}

fn update_n_k_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::update_n_k;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 5;
    let n: usize = 10;

    let (gamma_z, _) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f64> = sample_n_poisson(100_f64, n, &mut rng).iter().map(|x| x.ln()).collect();
    let alpha0: Vec<f64> = vec![1.0; k];

    let device = Default::default();
    type Backend = NdArray<f64>;

    let gamma_z = Tensor::<Backend, 1>::from_data(gamma_z.as_slice(), &device);
    let gamma_z = gamma_z.reshape([k, n]);
    let log_counts = Tensor::<Backend, 1>::from_data(log_counts.as_slice(), &device);
    let alpha0 = Tensor::<Backend, 1>::from_data(alpha0.as_slice(), &device);

    c.bench_function("update_n_k 5x10", |b|
                     b.iter(||
                            update_n_k(black_box(gamma_z.clone()), log_counts.clone(), alpha0.clone())
                     ));
}

fn elbo_rcg_mat_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::elbo_rcg_mat;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 5;
    let n: usize = 10;

    let (log_lls, _) = random_loglls(k, n, &mut rng);
    let (gamma_z, _) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f64> = sample_n_poisson(100_f64, n, &mut rng).iter().map(|x| x.ln()).collect();
    let n_k: Vec<f64> = sample_n_gamma(4000_f64, 1_f64, k, &mut rng).iter().map(|x| x.ln()).collect();

    let device = Default::default();
    type Backend = NdArray<f64>;

    let logl = Tensor::<Backend, 1>::from_data(log_lls.as_slice(), &device);
    let logl = logl.reshape([k, n]);
    let gamma_z = Tensor::<Backend, 1>::from_data(gamma_z.as_slice(), &device);
    let gamma_z = gamma_z.reshape([k, n]);
    let log_counts = Tensor::<Backend, 1>::from_data(log_counts.as_slice(), &device);
    let n_k = Tensor::<Backend, 1>::from_data(n_k.as_slice(), &device);

    c.bench_function("elbo_rcg_mat 5x10", |b|
                     b.iter(||
                            elbo_rcg_mat(black_box(logl.clone()), gamma_z.clone(), log_counts.clone(), n_k.clone())
                     ));
}

fn rcg_optl_mat_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::rcg_optl_mat;

    let mut rng = ChaCha8Rng::seed_from_u64(20251115_u64);

    let k: usize = 5;
    let n: usize = 10;

    let (log_lls, _) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f64> = sample_n_poisson(100_f64, n, &mut rng).iter().map(|x| x.ln()).collect();
    let alpha0: Vec<f64> = vec![1.0; k];

    let device = Default::default();
    type Backend = NdArray<f64>;

    let logl = Tensor::<Backend, 1>::from_data(log_lls.as_slice(), &device);
    let logl = logl.reshape([k, n]);
    let log_counts = Tensor::<Backend, 1>::from_data(log_counts.as_slice(), &device);
    let alpha0 = Tensor::<Backend, 1>::from_data(alpha0.as_slice(), &device);

    c.bench_function("rcg_optl_mat 5x10", |b|
                     b.iter(||
                            rcg_optl_mat(black_box(logl.clone()), log_counts.clone(), alpha0.clone(), 1e-7_f64, 5000_usize)
                     ));
}

criterion_group!(rcg_benches,
                 rcg_optl_mat_bench,
                 elbo_rcg_mat_bench,
                 update_n_k_bench,
                 mixt_negnatgrad_bench,
                 compute_norm_bench,
);
criterion_main!(rcg_benches);

// util

use rand::RngCore;

use rand_distr::Distribution;
use rand_distr::{Gamma, Normal, Poisson, Uniform};
use rand_distr::weighted::WeightedIndex;

use statrs::distribution::Continuous;

/// Sample a single value from the Dirichlet distribution
fn sample_dirichlet(
    alphas: &[f64],
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let k = alphas.len();

    let mut y_sum: f64 = 0.0;
    let ys: Vec<f64> = (0..k).map(|idx| {
        let gamma = Gamma::new(alphas[idx], 1.0).unwrap();
        let y = gamma.sample(rng);
        y_sum += y;
        y
    }).collect();

    ys.iter().map(|y| y/y_sum).collect::<Vec<f64>>()
}

/// Sample n values from the normal distribution
fn sample_n_normal(
    mean: f64,
    sd: f64,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let normal = Normal::new(mean, sd).unwrap();
    (0..n).map(|_| normal.sample(rng)).collect::<Vec<f64>>()
}

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

/// Sample n values from the uniform distribution
fn sample_n_uniform(
    min: f64,
    max: f64,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let uniform = Uniform::new(min, max).unwrap();
    (0..n).map(|_| uniform.sample(rng)).collect::<Vec<f64>>()
}

/// Sample n values from the poission
fn sample_n_poisson(
    rate: f64,
    n: usize,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let poisson = Poisson::new(rate).unwrap();
    (0..n).map(|_| poisson.sample(rng)).collect::<Vec<f64>>()
}

fn random_loglls(
    k: usize,
    n: usize,
    rng: &mut dyn RngCore,
) -> (Vec<f64>, Vec<f64>) {
    // Normal distribution parameters to generate observations
    let means: Vec<f64> = sample_n_normal(0_f64, 10_f64, k, rng);
    let sds: Vec<f64> = sample_n_gamma(1_f64, 2_f64, k, rng).iter().map(|x| x.sqrt()).collect();

    let normals: Vec<_> = means.iter().zip(sds.iter()).map(|(mu, sigma)| {
        statrs::distribution::Normal::new(*mu, *sigma).unwrap()
    }).collect();

    // Generate random thetas ~ Dirichlet(alpha_1, ..., alpha_k) by sampling from
    // Gamma(alpha_i, 1) distributions, where alpha_1 ~ Unif(0, 1)
    //
    // This tends to produce thetas that are concentrated around a few values
    let alphas: Vec<f64> = sample_n_uniform(0_f64, 1_f64, k, rng);
    let thetas: Vec<f64> = sample_dirichlet(&alphas, rng);

    // Generate log likelihoods for a mixture of `k` normal distributions
    let dist = WeightedIndex::new(&thetas).unwrap();
    let mut log_lls: Vec<Vec<f64>> = vec![vec![0_f64; n]; k];
    for i in 0..n {
        let cluster: usize = dist.sample(rng);
        let obs: f64 = sample_n_normal(means[cluster], sds[cluster], 1, rng)[0];
        for j in 0..k {
            log_lls[j][i] = normals[j].ln_pdf(obs)
        }
    }
    let log_lls: Vec<f64> = log_lls.iter().cloned().flatten().collect();

    (log_lls, thetas)
}
