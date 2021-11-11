# rcgpar - Fit mixture models in HPC environments
rcgpar provides MPI and OpenMP implementations of a variational
inference algorithm for estimating mixture model components from a
likelihood matrix in parallel.

## Installation
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
cmake ..
make
```

creating the `librcgomp` library in `build/lib/`.

#### MPI
You will need to use the appropriate platform-specifc commands
to set up your MPI environment. For example, to set up rcgpar using
[Open MPI](https://www.open-mpi.org/) enter the `build/` directory and run
```
module load mpi/openmp
cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx ..
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
provides two functions: 'rcg_optl_omp' for OpenMP parallelization
and 'rcg_optl_mpi' for MPI (+OpenMP, if enabled) parallelization.

## License
The source code from this project is subject to the terms of the
LGPL-2.1 license. A copy of the LGPL-2.1 license is supplied with the
project, or can be obtained at
https://opensource.org/licenses/LGPL-2.1.
