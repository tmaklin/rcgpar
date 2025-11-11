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

fn compute_norm_bench(c: &mut Criterion) {
    use rcgpar::optimizer::rcg::compute_norm;

    let mut rng = rand::rng();

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

    let mut rng = rand::rng();

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

    let mut rng = rand::rng();

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

    let mut rng = rand::rng();

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

    let mut rng = rand::rng();

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

criterion_group!(benches,
                 rcg_optl_mat_bench,
                 elbo_rcg_mat_bench,
                 update_n_k_bench,
                 mixt_negnatgrad_bench,
                 compute_norm_bench,
);
criterion_main!(benches);
