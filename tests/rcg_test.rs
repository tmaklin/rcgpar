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

use assert_approx_eq::assert_approx_eq;
use rcgpar::optimize_flat;
use rcgpar::BurnBackend;
use rcgpar::OptimizerOpts;
use rcgpar::optimizer::Algorithm;

use rand::rngs::ThreadRng;

use rand_distr::Distribution;
use rand_distr::{Gamma, Normal, Poisson, Uniform};
use rand_distr::weighted::WeightedIndex;

use statrs::distribution::Continuous;

/// Sample a single value from the Dirichlet distribution
fn sample_dirichlet(
    alphas: &[f64],
    rng: &mut ThreadRng,
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
    rng: &mut ThreadRng,
) -> Vec<f64> {
    let normal = Normal::new(mean, sd).unwrap();
    (0..n).map(|_| normal.sample(rng)).collect::<Vec<f64>>()
}

/// Sample n values from the gamma distribution
fn sample_n_gamma(
    shape: f64,
    scale: f64,
    n: usize,
    rng: &mut ThreadRng,
) -> Vec<f64> {
    let gamma = Gamma::new(shape, scale).unwrap();
    (0..n).map(|_| gamma.sample(rng)).collect::<Vec<f64>>()
}

/// Sample n values from the uniform distribution
fn sample_n_uniform(
    min: f64,
    max: f64,
    n: usize,
    rng: &mut ThreadRng,
) -> Vec<f64> {
    let uniform = Uniform::new(min, max).unwrap();
    (0..n).map(|_| uniform.sample(rng)).collect::<Vec<f64>>()
}

/// Sample n values from the poission
fn sample_n_poisson(
    rate: f64,
    n: usize,
    rng: &mut ThreadRng,
) -> Vec<f64> {
    let poisson = Poisson::new(rate).unwrap();
    (0..n).map(|_| poisson.sample(rng)).collect::<Vec<f64>>()
}

fn random_loglls(
    k: usize,
    n: usize,
    rng: &mut ThreadRng,
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

#[test]
fn rcg32_random() {
    let mut rng = rand::rng();

    let k: usize = 2;
    let n: usize = 1000;

    let (log_lls, thetas) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f64> = sample_n_poisson(100_f64, n, &mut rng).iter().map(|x| x.ln()).collect();
    let alphas: Vec<f64> = vec![1.0; k];

    let mut opts: OptimizerOpts = Default::default();
    opts.tolerance = 1e-16_f64;
    opts.max_iters = 1000;
    opts.device = BurnBackend::NdArray32;
    opts.algorithm = Algorithm::RCG;

    let (got, _) = optimize_flat(&log_lls, &log_counts, &alphas, Some(opts)).unwrap();

    got.iter().zip(thetas.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 4e-2) });

}

#[test]
fn em32_random() {
    let mut rng = rand::rng();

    let k: usize = 2;
    let n: usize = 1000;

    let (log_lls, thetas) = random_loglls(k, n, &mut rng);
    let log_counts: Vec<f64> = sample_n_poisson(100_f64, n, &mut rng).iter().map(|x| x.ln()).collect();
    let alphas: Vec<f64> = vec![1.0; k];

    let mut opts: OptimizerOpts = Default::default();
    opts.tolerance = 1e-16_f64;
    opts.max_iters = 1000;
    opts.device = BurnBackend::NdArray32;
    opts.algorithm = Algorithm::EM;

    let (got, _) = optimize_flat(&log_lls, &log_counts, &alphas, Some(opts)).unwrap();

    got.iter().zip(thetas.iter()).for_each(|(x, y)| { assert_approx_eq!(x, y, 4e-2) });

}
