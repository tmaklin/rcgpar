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

use assert_approx_eq::assert_approx_eq;
use rcgpar::optimize;
use rcgpar::optimize_flat;
use rcgpar::BurnBackend;
use rcgpar::OptimizerOpts;
use rcgpar::optimizer::Algorithm;

use rand_distr::{Gamma, Dirichlet};
use rand_distr::Distribution;

fn rcg_random_data(c: &mut Criterion) {

    let mut rng = rand::rng();
    let gamma = Gamma::new(1.0, 1.0).unwrap();

    const k: usize = 50;
    let n: usize = 1000;

    let alphas_real: Vec<f64> = (0..k).map(|_| gamma.sample(&mut rng)).collect();
    let dirichlet = Dirichlet::<_, k>::new(alphas_real.try_into().unwrap()).unwrap();

    let logl_mat: Vec<Vec<f64>> = (0..n).map(|_| dirichlet.sample(&mut rng).iter().map(|x: &f64| x.ln()).collect::<Vec<f64>>()).collect();

    let mut transposed: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        transposed.push(Vec::with_capacity(n));
    }
    for i in 0..n {
        for j in 0..k {
            transposed[j].push(logl_mat[i][j].clone());
        }
    }
    let log_likelihoods = transposed.iter().cloned().flatten().collect::<Vec<f64>>();

    let mut col_sums: Vec<f64> = vec![0_f64; k];
    for i in 0..n {
        for j in 0..k {
            col_sums[j] += logl_mat[i][j].exp();
        }
    }
    let col_sums = col_sums.iter().map(|x| x/(n as f64)).collect::<Vec<f64>>();

    let log_counts: Vec<f64> = (0..n).map(|(_)| 0.0).collect();
    let prior_counts: Vec<f64> = vec![1.0; k];

    let expected: Vec<f64> = vec![0.9990609231670258, 0.0007300890279000023, 9.656363112888921e-5, 0.00011242417394518503, 0.0];

    let mut opts: OptimizerOpts = Default::default();
    opts.tolerance = 1e-16_f64;
    opts.max_iters = 5000;
    opts.device = BurnBackend::Wgpu32;
    opts.algorithm = Algorithm::RCG;

    c.bench_function("rcg 5x10", |b|
                     b.iter(||
                            optimize_flat(black_box(&log_likelihoods), &log_counts, &prior_counts, Some(opts.clone())).unwrap()
                     ));
}

criterion_group!(benches, rcg_random_data);
criterion_main!(benches);
