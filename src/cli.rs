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
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    Fit {
        // Input log-likelihood matrix
        #[arg(long = "log-likelihoods", required = true, help_heading = "Input", help = "Tab-separated `n_observations x n_categories` log-likelihood matrix file.")]
        logl_path: PathBuf,

        // Weights/observation counts for each row of --log-likelihood
        #[arg(long = "weights", required = true, help_heading = "Input", help = "File with `n_observations` lines containing weights for each row of `--log-likelihood`.")]
        weights_path: PathBuf,

        // Device
        #[arg(long = "device", required = false, help = "Which backend to run on (default: NdArray with 64 bit floats on CPU).")]
        device: Option<rcgpar::BurnBackend>,

        // RCG parameters
        // // Tolerance when checking for convergence
        #[arg(long = "tolerance", default_value_t = 1e-7, help_heading = "RCG parameters", help = "Tolerance when checking for convergence in RCG.")]
        tolerance: f64,
        // // Maximum number of iterations
        #[arg(long = "max-iters", default_value_t = 100, help_heading = "RCG parameters", help = "Maximum number of iterations to run RCG for.")]
        max_iters: usize,

        // Resources
        // // Threads
        #[arg(short = 't', long = "threads", default_value_t = 1)]
        num_threads: usize,

        // Verbosity
        #[arg(long = "verbose", default_value_t = false)]
        verbose: bool,
    },
}
