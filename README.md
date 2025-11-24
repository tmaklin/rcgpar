# mixt - Infer mixture model parameters

mixt provides algorithms for estimating the mixing proportions for a mixture
model with a fixed log-likelihood matrix, ie. a model where the hyperparameters
of the emission distributions are known but the weights are not.

Documentation is available at [https://docs.rs/mixt](https://docs.rs/mixt).

## Usage
mixt supports three main use cases:
- Rust library API.
- C++ API.
- Minimal CLI.

Both the Rust and C++ API support GPU acceleration with the Wgpu backend from
[burn](https://docs.rs/burn/latest/burn/).

## About
mixt is a Rust rewrite of a previous C++ implementation. You can find the C++
code at [tmaklin/mixt](https://github.com/tmaklin/mixt).

The C++ and Rust code have equal CPU performance, but the Rust code implements
multiple numerical stability tricks that enable running on 32-bit floating point
numbers, allowing for better GPU utilization.

[burn](https://docs.rs/burn) allows compiling mixt for many different GPU
architectures, whereas the C++ implementation only supports
[torch](https://pytorch.org/).

## License
The source code from this project is subject to the terms of the
LGPL-2.1 license. A copy of the LGPL-2.1 license is supplied with the
project, or can be obtained at
https://opensource.org/licenses/LGPL-2.1.
