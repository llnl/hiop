# Option 1: Use Combination of System Modules and ExaSGD-Built Modules

On Summit, a limited number of dependecies are available by default, and the modules
tend to change somewhat frequently. We have a script `./scripts/summitVariables.sh`
which loads a combination of system modules and modules built in the ExaSGD
shared project folder which enables a full build of HiOp.

```console
$ git clone git@github.com:LLNL/hiop.git
$ cd hiop
$ source ./scripts/summitVariables.sh
$ mkdir build install && cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../install
$ ccmake . # Optional, if you would like to customize the build
$ make -j 16 install
$ make test # Optional, if you would like to run tests. This command should be
$           # ran in a BSUB allocation.
```

Note that this script may become out of date, and is not officially supported.

## Notes on HiOp's Buildsystem and the `module` Utility

Different versions of modules may conflict with each other; in this case, one needs to inquire additional compatibility information using `module spider package_name`.

HiOp with GPU support can be configured, built, and tested using with the cmake
option `HIOP_USE_CUDA`. This automatically enables `HIOP_USE_GPU`. For example, you may invoke `cmake`
like so:

```
cd build/
cmake -DHIOP_USE_CUDA=ON ..
make -j 16
make test
make install
```

In some cases, `cmake` picks up a different gcc compiler than the one loaded via `module`. This situation can be remedied by preloading the `CC` and `CXX` for `cmake` as it is done above.

Developers are usually required to perform a more comprehensive build like so:
```
rm -rf *
CC=/sw/summit/gcc/9.2.0/bin/gcc CXX=/sw/summit/gcc/9.2.0/bin/g++ cmake -DHIOP_SPARSE=ON -DHIOP_USE_CUDA=ON -DHIOP_TEST_WITH_BSUB=ON -DHIOP_USE_MPI=ON -DMETIS_DIR=$INSTALL_DIR/Compiler/gcc-9.2.0/metis/5.1.0/ -DCOINHSL_DIR=/ccs/home/cpetra/work/installs/coinhsl-2015.06.23/_install -DCMAKE_BUILD_TYPE=DEBUG -DHIOP_USE_RAJA=ON -DHIOP_DEEPCHECKS=ON ..
make -j
```

Depending on the LAPACK/BLAS CPU library loaded by the `module` command, runtime crashes can occur (intermintently for different problems/linear systems sizes) if the `nvblas.conf` cannot be find or is not specifying the so-called NVBLAS CPU BLAS library required by CUDA. This can be remedied by specifying the location of the BLAS library in `NVBLAS_CPU_BLAS_LIB` in the `nvblas.conf` file; for example, this can be done by adding the following lines to `nvblas.conf` file 
```
NVBLAS_CPU_BLAS_LIB /gpfs/alpine/proj-shared/csc359/installs/ExaSGD/Compiler/gcc-9.2.0/openblas/0.3.10/lib/libopenblas.so
```
Be aware that this assumes that the openblas has been `module load`ed.

For non-standard locations of `nvblas.conf`, one can specify the path to `nvblas.conf` via the environment variable `NVBLAS_CONFIG_FILE`; for example, this can be achieved by exporting the variable in your submission `your_job_file.lsf` file:
```
# ##LSF directives  here
#  other commands here
# example assumes a bash shell
export NVBLAS_CONFIG_FILE=/ccs/home/cpetra/work/projects/hiop/runs/nvblas.conf 

# jsrun command here

```

# Option 2 - using Spack

## Using official Spack recipe

An Spack recipe is available in the builtin spack repository. To build HiOp
on Summit using Spack requires some experience with Spack. Users are referred to
[the Spack documentation here](https://spack.readthedocs.io/en/latest/).
