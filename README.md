# rcgpar-gpu
Adds onto rcgpar by adding two new implementations of inference algorithms for estimating mixture model components from a
likelihood matrix using PyTorch for GPU acceleration.

# rcgpar - Fit mixture models in HPC environments
rcgpar provides MPI and OpenMP implementations of a variational
inference algorithm for estimating mixture model components from a
likelihood matrix in parallel.

## Installation
### Requirements
- C++17 compliant compiler.
- cmake
- [LibTorch](https://pytorch.org/get-started/locally/)

#### Optional
- CUDA Toolkit (if using LibTorch with CUDA support; version depending on downloaded LibTorch) or ROCm (if using LibTorch with ROCm support; version depending on downloaded LibTorch)
- Compiler with OpenMP support.

Without a CUDA or ROCm supported LibTorch and CUDA Toolkit or ROCm there will be no GPU acceleration.

### Compiling from source
Clone the repository to a suitable folder, enter the directory and run
```
mkdir build
cd build
```

... and follow the instructions below.

#### OpenMP
in the `build/` directory, run
```
cmake -DCMAKE_LIBTORCH_PATH=/absolute/path/to/libtorch ..
cmake --build .
```
where `/absolute/path/to/libtorch` should be the absolute (!) path to the unzipped LibTorch distribution.

This creates the `librcgomp` library in `build/lib/`.

#### MPI
You will need to use the appropriate platform-specifc commands
to set up your MPI environment. For example, to set up rcgpar using
[Open MPI](https://www.open-mpi.org/) enter the `build/` directory and run
```
module load mpi/openmp
cmake -DCMAKE_ENABLE_MPI_SUPPORT=1 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx ..
make
```

creating the `librcgmpi` library in `build/lib/`. If OpenMP is also
supported, the `librcgomp` library will also be created.

librcgmpi is compiled by default to support up to 1024 processes. If
you need more, recompile the project with
`-DCMAKE_MPI_MAX_PROCESSES=<big number>` added to the cmake command.

#### Hybrid OpenMP + MPI
`librcgmpi` automatically provides hybrid OpenMP + MPI
parallelization when the library is compiled on a system that
supports both protocols.

### Compiling and running tests
rcgpar uses the [googletest](https://github.com/google/googletest)
framework to test the libraries. Tests can be built by compiling the
program in debug mode by appending the `-DCMAKE_BUILD_TESTS=1` flag to
the cmake call. Tests will be created in `build/bin/` and all tests
(except the MPI test) can be run from the runUnitTests executable.

Note: you will need to use mpirun (or some other appropriate call) to
run the MPI test from the executable runMpiTest.

## Usage
Simply include the `rcgpar.hpp` header in your project. This header
provides four functions: 'rcgpar::rcg\_optl\_omp' for OpenMP parallelization, 
'rcgpar::rcg\_optl\_mpi' for MPI (+OpenMP, if enabled) parallelization,
'rcgpar::rcg\_optl\_torch' for PyTorch acceleration, and
'rcgpar::em\_torch' a different algorithm with PyTorch acceleration.

### rcg\_optl\_omp, rcg\_optl\_mpi, rcg\_optl\_torch, and em\_torch
These four functions perform the actual model fitting. All have to be called with the following
arguments:
```
const rcgpar::Matrix<double> &logl:
    KxN row-major order matrix containing the log-likelihoods for theobservations,
    where K is the number of components and N is the number of observations.
const std::vector<double> &log_times_observed:
    N-dimensional vector which contains the natural logarithm of the number
	of times that the N:th row in `logl` should be counted. Useful if many
	rows in the log-likelihood matrix are identical - they can be compressed
	by counting them several times via this argument.
const std::vector<double> &alpha0:
    N-dimensional vector containing the prior parameters of the Dirichlet
	distribution that is used as a conjugate prior in the model. Good
	default choice is to set all entries to 1.
const double &tol:
    The estimation process will terminate once the evidence lower bound
	ELBO changes by less than this value from one iteration to the next.
	Good choices are around 1e-6 and 1e-8, adjust according to your needs.
const uint16_t maxiters:
    Maximum number of iterations to run the optimizer for if the tolerance
	criterion is not fulfilled.
std::ostream &log:
    Print status messages here. Silence the messages by supplying a
	std::ofstream that has not been assigned to any file.
```
'em\_torch' requires the extra argument:
```
std::string precision:
    Either "float" or "double", which determines the precision of the algorithm.
```

The optimizers (except for em\_torch) return a KxN `rcgpar::Matrix<double>` type row-major order
matrix, where each row is a probability vector assigning the row to
the mixture components.

'em\_torch' returns immediately an N-dimensional probability vector
containing the mixture component proportions (no need for 'mixture\_components')

Note: rcg\_optl\_mpi assumes that the root process holds the full
'logl' and 'log\_times\_observed values', which are then distributed
from the root process to other processes. Contrary to this, 'alpha0',
'tol', and 'maxiters' are assumed to be present on all processes when
calling rcg\_optl\_mpi.

### mixture\_components and mixture\_components\_torch
Use 'rcgpar::mixture\_components\(_torch)' to transform the matrix from
rcg\_optl\_omp/mpi/torch into a probability vector containing the relative
contributions of each mixture component. 'mixture\_components\(_torch)' takes
the following input arguments:
```
const rcgpar::Matrix<double> &probs:
    The matrix returned from either rcg_optl_omp or rcg_optl_mpi.
const std::vector<double> &log_times_observed:
    The N-dimensional vector of log times observed that was used
	as input to the call to rcg_optl_omp or rcg_optl_mpi.
```

'mixture\_components\(_torch)' will return a N-dimensional probability vector
containing the mixture component proportions.

### Creating the input matrix
rcgpar requires the input log-likelihood matrix formatted with the
internal rcgpar::Matrix class. If your input log-likelihoods are
stored in a flattened vector, you can construct the input object to
rcg\_optl\_omp/mpi with the constructor:
```
Matrix<double>(std::vector<double> &flattened_logl,
               uint16_t n_mixture_components, uint32_t n_observations)
```

If your data is stored in a 2D vector, use the following constructor:
```
Matrix<double>(std::vector<std::vector<double>> &logl_2D)
```

Note that both constructors assume the data is stored in row-major
order.

## License
The source code from this project is subject to the terms of the
LGPL-2.1 license. A copy of the LGPL-2.1 license is supplied with the
project, or can be obtained at
https://opensource.org/licenses/LGPL-2.1.
