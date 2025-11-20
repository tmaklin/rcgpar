# rcgpar - Infer mixture model parameters

rcgpar provides algorithms for estimating the mixing proportions for a mixture
model with a fixed log-likelihood matrix.

Documentation is available at [https://docs.rs/rcgpar](https://docs.rs/rcgpar).

## Usage
rcgpar supports three main use cases:
- Rust library API.
- C++ API.
- Minimal CLI.

Both the Rust and C++ API support GPU acceleration with the Wgpu backend from
[burn](https://docs.rs/burn/latest/burn/).

## About
rcgpar v2 onwards is a rewrite of the previous C++ implementation in Rust. You
can find the original C++ code in versions preceding v2.

The C++ and Rust code have roughly equal CPU performance, but the Rust code
implements multiple numerical stability tricks that enable running on 32-bit
floating point numbers, allowing for better GPU utilization.

[burn](https://docs.rs/burn) allows compiling rcgpar for many different GPU
architectures, whereas the C++ implementation only supports
[torch](https://pytorch.org/).

## License
The source code from this project is subject to the terms of the
LGPL-2.1 license. A copy of the LGPL-2.1 license is supplied with the
project, or can be obtained at
https://opensource.org/licenses/LGPL-2.1.
