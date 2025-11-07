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
use std::path::PathBuf;

use clap::Parser;

// Command-line interface
mod cli;

fn read_counts(
    path: &PathBuf,
    delimiter: u8
) -> Vec<u32> {
    let fs = match std::fs::File::open(path) {
        Ok(fs) => fs,
        Err(e) => panic!("  Error in reading --input-list: {}", e),
    };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .from_reader(fs);

    let counts = reader.records().map(|line| {
        if let Ok(record) = line {
            let val: u32 = record.iter().next().unwrap().parse().unwrap();
            val
        } else {
            panic!("  Error in reading --weights: {}", path.clone().into_os_string().into_string().unwrap());
        }
    }).collect::<Vec<u32>>();

    counts
}

fn read_log_likelihoods(
    path: &PathBuf,
    delimiter: u8
) -> Vec<Vec<f32>> {
    let fs = match std::fs::File::open(path) {
        Ok(fs) => fs,
        Err(e) => panic!("  Error in reading --input-list: {}", e),
    };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .from_reader(fs);

    let mut logl: Vec<Vec<f32>> = Vec::new();
    reader.records().for_each(|line| {
        if let Ok(record) = line {
            logl.push(record.iter().map(|x| { let val: f32 = x.parse().unwrap(); val } ).collect::<Vec<f32>>());
        } else {
            panic!("  Error in reading --log-likelihood: {}", path.clone().into_os_string().into_string().unwrap());
        }
    });

    logl
}

/// Initializes the logger with verbosity given in `log_max_level`.
fn init_log(log_max_level: usize) {
    stderrlog::new()
        .module(module_path!())
        .quiet(false)
        .verbosity(log_max_level)
        .timestamp(stderrlog::Timestamp::Off)
        .init()
        .unwrap();
}

fn main() {
    let cli = cli::Cli::parse();
    match &cli.command {
        Some(cli::Commands::Fit {
            logl_path,
            weights_path,
            device,
            tolerance,
            max_iters,
            num_threads,
            verbose,
        }) => {
            init_log(if *verbose { 2 } else { 1 });

            let logl = read_log_likelihoods(logl_path, b'\t');
            let weights = read_counts(weights_path, b'\t');
            let prior: Vec<f32> = vec![1.0; logl.len()];

            let mut options: rcgpar::OptimizerOpts = Default::default();
            options.tolerance = *tolerance;
            options.max_iters = *max_iters;
            options.device = device.clone().unwrap_or_default();

            let proportions = rcgpar::optimize(&logl, &weights, &prior, Some(options)).unwrap();

            proportions.iter().enumerate().for_each(|(idx, theta)| {
                eprintln!("{idx}\t{theta}");
            });

        },
        None => {},
    }
}
