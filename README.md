# HiOp - HPC solver for optimization
![tests](https://github.com/LLNL/hiop/workflows/tests/badge.svg)

HiOp is an optimization solver for solving certain mathematical optimization problems expressed as nonlinear programming problems. HiOp is a lightweight HPC solver that leverages application's existing data parallelism to parallelize the optimization iterations by using specialized parallel linear algebra kernels.

Please cite the user manual whenever HiOp is used:
```
@TECHREPORT{hiop_techrep,
  title={{HiOp} -- {U}ser {G}uide},
  author={Petra, Cosmin G. and Chiang, NaiYuan and Jingyi Wang},
  year={2018},
  institution = {Center for Applied Scientific Computing, Lawrence Livermore National Laboratory},
  number = {LLNL-SM-743591}
}
```
In addition, when using the quasi-Newton solver please cite:
```
@ARTICLE{Petra_18_hiopdecomp,
title = {A memory-distributed quasi-Newton solver for nonlinear programming problems with a small number of general constraints},
journal = {Journal of Parallel and Distributed Computing},
volume = {133},
pages = {337-348},
year = {2019},
issn = {0743-7315},
doi = {https://doi.org/10.1016/j.jpdc.2018.10.009},
url = {https://www.sciencedirect.com/science/article/pii/S0743731518307731},
author = {Cosmin G. Petra},
}
```
and when using the the PriDec solver please cite:
```
@article{wang2023,
  author = {J. Wang and C. G. Petra},
  title = {A Sequential Quadratic Programming Algorithm for Nonsmooth Problems with Upper-$\mathcal{C}^2$ Objective},
  journal = {SIAM Journal on Optimization},
  volume = {33},
  number = {3},
  pages = {2379-2405},
  year = {2023},
  doi = {10.1137/22M1490995}
}
@INPROCEEDINGS{wang2021,
  author={J. Wang and N. Chiang and C. G. Petra},
  booktitle={2021 20th International Symposium on Parallel and Distributed Computing (ISPDC)}, 
  title={An asynchronous distributed-memory optimization solver for two-stage stochastic programming problems}, 
  year={2021},
  volume={},
  number={},
  pages={33-40},
  doi={10.1109/ISPDC52870.2021.9521613}}
 }
```

## Build/install instructions
HiOp uses a CMake-based build system. A standard build can be done by invoking in the 'build' directory the following 
```shell 
$> cmake ..
$> make 
$> make test
$> make install
```
This sequence will build HiOp, run integrity and correctness tests, and install the headers and the library in the directory '_dist-default-build' in HiOp's root directory. 

Command `make test` runs extensive tests of the various modules of HiOp to check integrity and correctness. The tests suite range from unit testing to solving concrete optimization problems and checking the performance of HiOp solvers on these problems against known solutions. By default `make test` runs `mpirun` locally, which may not work on some HPC machines. For these HiOp allows using `bsub` to schedule `make test` on the compute nodes; to enable this, the use should use *-DHIOP_TEST_WITH_BSUB=ON* with cmake when building and run `make test` in a bsub shell session, for example,
```
bsub -P your_proj_name -nnodes 1 -W 30
make test
CTRL+D
```

The installation can be customized using the standard CMake options. For example, one can provide an alternative installation directory for HiOp by using 
```sh
$> cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/hiop ..'
```


### Selected HiOp-specific build options
* Enable/disable MPI: *-DHIOP_USE_MPI=[ON/OFF]* (by default ON)
* GPU support: *-DHIOP_USE_GPU=ON*. MPI can be either off or on. For more build system options related to GPUs, see "Dependencies" section below.
* Enable/disable "developer mode" build that enforces more restrictive compiler rules and guidelines: *-DHIOP_DEVELOPER_MODE=ON*. This option is by default off.
* Additional checks and self-diagnostics inside HiOp meant to detect abnormalities and help to detect bugs and/or troubleshoot problematic instances: *-DHIOP_DEEPCHECKS=[ON/OFF]* (by default ON). Disabling HIOP_DEEPCHECKS usually provides 30-40% execution speedup in HiOp. For full strength, it is recommended to use HIOP_DEEPCHECKS with debug builds. With non-debug builds, in particular the ones that disable the assert macro, HIOP_DEEPCHECKS does not perform all checks and, thus, may overlook potential issues.

For example:
```shell 
$> cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON ..
$> make 
$> make test
$> make install
```


### Other useful options to use with CMake
* *-DCMAKE_BUILD_TYPE=Release* will build the code with the optimization flags on
* *-DCMAKE_CXX_FLAGS="-O3"* will enable a high level of compiler code optimization

### Dependencies

A complete list of dependencies is maintained [here](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/hiop/package.py).

For a minimal build, HiOp requires LAPACK and BLAS. These dependencies are automatically detected by the build system. MPI is optional and by default enabled. To disable use cmake option '-DHIOP_USE_MPI=OFF'.

Please note that HiOp has support for GPU computations, for both NVIDIA and AMD hardware, via RAJA and Umpire performance portability libraries.   To quickly enable the use of GPUs, use cmake with '-DHIOP_USE_GPU=ON'.

## Install with Spack

If you would like to use our public [`package.py`](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/hiop/package.py):

```bash
spack install hiop@develop
```

Additional information about using Spack can be found at [spack.readthedocs.io](https://spack.readthedocs.io/en/latest/) or within HiOp's [`package.py`]((https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/hiop/package.py)).

# Interfacing with HiOp 

HiOp supports three types of optimization problems, each with a separate input formats in the form of the C++ interfaces `hiopInterfaceDenseConstraints`,`hiopInterfaceSparse` and `hiopInterfaceMDS`. These interfaces are specified in [hiopInterface.hpp](src/Interface/hiopInterface.hpp) and documented and discussed as well in the [user manual](doc/hiop_usermanual.pdf).

*`hiopInterfaceDenseConstraints` interface* supports NLPs with **billions** of variables with and without bounds but only limited number (<100) of general, equality and inequality constraints. The underlying algorithm is a limited-memory quasi-Newton interior-point method and generally scales well computationally (but it may not algorithmically) on thousands of cores. This interface uses MPI for parallelization

*`hiopInterfaceSparse` interface* supports general sparse and large-scale NLPs. This functionality is similar to that of the state-of-the-art [Ipopt](https://github.com/coin-or/Ipopt) (without being as robust and flexible as Ipopt is). Acceleration for this class of problems can be achieved via OpenMP or CUDA, however, this is work in progress and you are encouraged to contact HiOp's developers for up-to-date information.

*`hiopInterfaceMDS` interface* supports mixed dense-sparse NLPs and achives parallelization using GPUs and RAJA portability abstraction layer. 

More information on the HiOp interfaces are [here](src/Interface/README.md).

# HiOpBBpy - Python based black-box optimization

HiOpBBpy is a black-box (derivative free) Bayesian optimization solver. HiOpBBpy primarily targets problems for which the objective is very expensive and derivative information is not available. HiOpBBpy leverages MPI parallelism for batched Bayesian optimization methods and other embarassingly parallel tasks in said methods.

## Build/install instructions

HiOpBBpy can be built via pip. A standard build can be done by invoking the following from the hiop home directory
```shell 
$> pip install .
```

## Usage

Various example scripts are contained in src/Drivers/hiopbbpy/ that can be run directly from said directory, e.g.,

```shell 
$> python BODriverEX.py
```

or to run the same script but utilizing MPI parallelism with 4 workers

```shell 
$> mpiexec -n 4 python -m mpi4py.futures BODriverEX.py
```

or to run the same script with MPI parallelism but on a slurm based scheduling system 

```shell 
$> srun -n 4 python -m mpi4py.futures BODriverEX.py
```

Further details on optional dependencies and parallel examples can be found in the HiOp user manual.

## Dependencies

HiOpBBpy dependencies are listed in the pyproject.toml file.

## Issues
Users are highly encouraged to report any issues they found when using HiOp.

## Acknowledgments

HiOp has been developed under the financial support of: 
- Department of Energy, Office of Advanced Scientific Computing Research (ASCR): Exascale Computing Program (ECP) and Applied Math Program.
- Department of Energy, Advanced Research Projects Agency-Energy (ARPA‑E)
- Lawrence Livermore National Laboratory Institutional Scientific Capability Portfolio (ISCP)
- Lawrence Livermore National Laboratory, through the LDRD program

# Contributors

HiOp is written by Cosmin G. Petra (petra1@llnl.gov), Nai-Yuan Chiang (chiang7@llnl.gov), Jingyi "Frank" Wang (wang125@llnl.gov), and Tucker Hartland (hartland1@llnl.gov) from LLNL and has received important contributions from Weslley da Silva Pereira (NREL), Asher Mancinelli (PNNL), Slaven Peles (ORNL), Cameron Rutherford (PNNL), Jake K. Ryan (PNNL), and Michel Schanen (ANL).

# Copyright

Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC. All rights reserved. Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-742473. HiOp is free software; you can modify it and/or redistribute it under the terms of the BSD 3-clause license. See [COPYRIGHT](/COPYRIGHT) and [LICENSE](/LICENSE) for complete copyright and license information.
 

