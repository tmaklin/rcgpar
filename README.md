# rcgpar-rs - Infer mixture model parameters

rcgpar-rs provides algorithms for estimating the mixing proportions for a mixture
model with a fixed log-likelihood matrix.

Documentation is available at [https://docs.rs/rcgpar-rs](https://docs.rs/rcgpar-rs).

## Usage
rcgpar-rs supports three main use cases:
- Rust library API.
- C++ API.
- Minimal CLI.

Both the Rust and C++ API support GPU acceleration with the Wgpu backend from
[burn](https://docs.rs/burn/latest/burn/).

## About
rcgpar-rs is the successor to a previous C++ implementation which you can find
at [tmaklin/rcgpar](https://github.com/tmaklin/rcgpar).

The C++ and Rust code have roughly equal CPU performance, but the Rust code
implements multiple numerical stability tricks that enable running on 32-bit
floating point numbers, allowing for better GPU utilization.

[burn](https://docs.rs/burn) allows compiling rcgpar-rs for many different GPU
architectures, whereas the C++ implementation only supports
[torch](https://pytorch.org/).

## License
The source code from this project is subject to the terms of the
LGPL-2.1 license. A copy of the LGPL-2.1 license is supplied with the
project, or can be obtained at
https://opensource.org/licenses/LGPL-2.1.
