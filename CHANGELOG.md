# Change Log
All notable changes to HiOp are documented in this file.

## Version 1.2.0: Release of HiOpBBpy solver for derivative-free optimization
* HiOpBBpy Batched Bayesian Optimizer  @thartland in https://github.com/LLNL/hiop/pull/724
* HiOpBBpy Nonlinear constraints by @nychiang in https://github.com/LLNL/hiop/pull/726
* HiOpBBpy parallel MPI evaluation manager to utils by @weslleyspereira in https://github.com/LLNL/hiop/pull/733
* Derivative checker for quasi-Newton and Newton IPMs by @cnpetra in https://github.com/LLNL/hiop/pull/731
* HiOpBBpy parallel Evaluator Upgrade by @thartland in https://github.com/LLNL/hiop/pull/734
* Addressing misc compilation warnings by @cnpetra in https://github.com/LLNL/hiop/pull/738
* Logger for HiOpBBpy by @nychiang in https://github.com/LLNL/hiop/pull/739
* README update with HiOpBBpy information by @thartland in https://github.com/LLNL/hiop/pull/742
* Misc fixes for fixed variables, feasibility restoration, Krylov-based IR, and others by @cnpetra in https://github.com/LLNL/hiop/pull/710

## New Contributors
* @weslleyspereira made their first contribution in https://github.com/LLNL/hiop/pull/733

**Full Changelog**: https://github.com/LLNL/hiop/compare/v1.1.1...v1.2.0

## Version 1.1.1: Misc updates to quasi-Newton solver and build system, updates to recent RAJA, and preliminary black-box solver
* Consolidating qn solver by @cnpetra in https://github.com/LLNL/hiop/pull/697 and https://github.com/LLNL/hiop/pull/699
* Use clang-format in hiop to keep code style by @nychiang in https://github.com/LLNL/hiop/pull/700
* Cmake option HIOP_TEST_WITH_SLURM by @thartland in https://github.com/LLNL/hiop/pull/707
* Support for C++17 by @nychiang in https://github.com/LLNL/hiop/pull/703
* Improved log output for NLP scaling by @cnpetra in https://github.com/LLNL/hiop/pull/709
* Use the latest RAJA and UMPIRE by @nychiang in https://github.com/LLNL/hiop/pull/706
* Add Bayesian Optimization into HiOp by @nychiang in https://github.com/LLNL/hiop/pull/711
* Support for constrained problems for black-box solver by @nychiang in https://github.com/LLNL/hiop/pull/722

## Version 1.1.0: Checkpointing, advanced restarts, and misc fixes
* Checkpointing for quasi-Newton solver: new API for load and saving checkpoints, user options for rollover restarts, and scalable I/O via axom::sidre.
* New function for advanced warmstarts.
* Updated spack build with build cache.

## Version 1.0.3: Moving limits and misc fixes
* Update modules for CI tests on LLNL LC by @nychiang in #679
* Update cmake build system to require RAJA when GPU compute mode is used by @nychiang in #676
* Moving limits options for NLP IPM solvers by @cnpetra in #681

## Version 1.0.2: Misc fixes
* Removed deprecated ALG2 for cusparseCsr2cscEx2 by @cnpetra in #671
* Addressed fixed buffer size vulnerability for vsnprintf by @nychiang in #673
* Removed stringent -Wall and -Werror from release builds to avoid downstream compilation errors

## Version 1.0.1: C++17 compatible and misc fixes
Default C++ standard remains C++14

* Fix LLNL CI by @nychiang in #663
* C++17 support: compile -Wall/-Werror proof by @tepperly in #653
* Fix a bug in copying an empty matrix into a bigger matrix by @nychiang in #666
* Fix the approach used to update mu by @nychiang in #664

## Version 1.0.0: Mature solvers interfaces and execution backends
### Notable new features
Interfaces of various solvers reached an equilibrium point after HiOp was interfaced with multiple optimization front-ends (e.g., power grid ACOPF and SC-ACOPF problems and topology optimization) both on CPUs and GPUs. The PriDec solver reached exascale on Frontier after minor communication optimizations. The quasi-Newton interior-point solver received a couple of updates that increase robustness. The Newton interior-point solver can fully operate on GPUs with select GPU linear solvers (CUSOLVER-LU and Gingko).

* Instrumentation of RAJA sparse matrix class with execution spaces by @cnpetra in https://github.com/LLNL/hiop/pull/589
* Fix Assignment Typo in hiopMatrixSparseCsrCuda.cpp by @pate7 in https://github.com/LLNL/hiop/pull/612
* Use failure not failed in PNNL commit status posting by @cameronrutherford in https://github.com/LLNL/hiop/pull/609
* rebuild modules on quartz by @nychiang in https://github.com/LLNL/hiop/pull/619
* Use constraint violation in checkTermination by @nychiang in https://github.com/LLNL/hiop/pull/617
* MPI communication optimization by @rothpc in https://github.com/LLNL/hiop/pull/613
* fix memory leaks in inertia-free alg and condensed linsys by @nychiang in https://github.com/LLNL/hiop/pull/622
* Update IPM algorithm for the dense solver by @nychiang in https://github.com/LLNL/hiop/pull/616
* Use integer preprocessor macros for version information by @tepperly in https://github.com/LLNL/hiop/pull/627
* use compound vec in bicg IR by @nychiang in https://github.com/LLNL/hiop/pull/621
* Use bicg ir in the quasi-Newton solver by @nychiang in https://github.com/LLNL/hiop/pull/620
* Add support to MPI in C/Fortran examples  by @nychiang in https://github.com/LLNL/hiop/pull/633
* Refactor CUSOLVER-LU module and interface by @pelesh in https://github.com/LLNL/hiop/pull/634
* Add MPI unit test for DenseEx4 by @nychiang in https://github.com/LLNL/hiop/pull/644
* Add more options to control NLP scaling by @nychiang in https://github.com/LLNL/hiop/pull/649
* Development of the feasibility restoration in the quasi-Newton solver by @nychiang in https://github.com/LLNL/hiop/pull/647
* GPU linear solver interface by @pelesh in https://github.com/LLNL/hiop/pull/650


### New Contributors
* @pate7 made their first contribution in https://github.com/LLNL/hiop/pull/612
* @rothpc made their first contribution in https://github.com/LLNL/hiop/pull/613

## Version 0.7.2: Execution spaces abstractions and misc fixes
This release hosts a series of comprehensive internal developments and software re-engineering to improve the portability and performance on accelerators/GPU platforms. No changes to the user interface permeated under this release.

### Notable new features

A new execution space abstraction is introduced to allow multiple hardware backends to run concurrently. The proposed design differentiates between "memory backend" and "execution policies" to allow using RAJA with Umpire-managed memory, RAJA with Cuda- or Hip-managed memory, RAJA with std memory, Cuda/Hip kernels with Cuda-/Hip- or Umpire-managed memory, etc.

* Execution spaces: support for memory backends and execution policies by @cnpetra in https://github.com/LLNL/hiop/pull/543
* Build: Cuda without raja  by @cnpetra in https://github.com/LLNL/hiop/pull/579
* Update of RAJA-based dense matrix to support runtime execution spaces by @cnpetra in https://github.com/LLNL/hiop/pull/580
* Reorganization of device namespace  by @cnpetra in https://github.com/LLNL/hiop/pull/582
* RAJA Vector int with ExecSpace by @cnpetra in https://github.com/LLNL/hiop/pull/583
* Instrumentation of host vectors with execution spaces by @cnpetra in https://github.com/LLNL/hiop/pull/584
* Remove copy from/to device methods in vector classes by @cnpetra in https://github.com/LLNL/hiop/pull/587
* Add support for Raja with OpenMP into LLNL CI by @nychiang in https://github.com/LLNL/hiop/pull/566
 
New vector classes using vendor-provided API were introduced and documentation was updated/improved
* Development of `hiopVectorCuda` by @nychiang in https://github.com/LLNL/hiop/pull/572
* Implementation of `hiopVectorHip` by @nychiang in https://github.com/LLNL/hiop/pull/590
* Update user manual by @nychiang in https://github.com/LLNL/hiop/pull/591
* Update the code comments in `hiopVector` classes by @nychiang in https://github.com/LLNL/hiop/pull/592

Refinement of triangular solver implementation for Ginkgo by @fritzgoebel in https://github.com/LLNL/hiop/pull/585

### Bug fixes
* Refine the computation in normal equation system by @nychiang in https://github.com/LLNL/hiop/pull/530
* Fix static culibos issue #567 by @nychiang in https://github.com/LLNL/hiop/pull/568
* Fix segfault, remove nonsymmetric ginkgo solver by @fritzgoebel in https://github.com/LLNL/hiop/pull/548
* Calculate the inverse objective scale correctly. by @tepperly in https://github.com/LLNL/hiop/pull/570
* Fix `hiopVectorRajaPar::copyToStartingAt_w_pattern` by @nychiang in https://github.com/LLNL/hiop/pull/569
* Gitlab pipeline refactor by @CameronRutherford in https://github.com/LLNL/hiop/pull/597

### New Contributors
* @tepperly made their first contribution in https://github.com/LLNL/hiop/pull/570

**Full Changelog**: https://github.com/LLNL/hiop/compare/v0.7.1...v0.7.2

## Version 0.7.1: Miscellaneous fixes to build system
This minor release fixes a couple of issues found in the build system after the major release 0.7 of HiOp. 

## Version 0.7.0: Fortran interface and misc fixes and improvements:
- Fortran interface and examples
- Bug fixing for sparse device linear solvers
- Implementation of CUDA CSR matrices
- Iterative refinement within CUSOLVER linear solver class
- Improved robustness and performance of mixed dense-sparse solver for AMD/HIP

## Version 0.6.2:  Initial ginkgo solver integration and misc fixes
This tag provides an initial integration with ginko, fixes a couple of issues, and add options for (outer) iterative refinement.

## Version 0.6.1: HIP linear algebra workaround and update for RAJA > v0.14  (March 31, 2022)
This version/tag provides a workaround for an issue in the HIP BLAS and updates the RAJA code to better operate with the newer versions of RAJA.

## Version 0.6.0: Release of the PriDec optimization and improved GPU computations (March 31, 2022)
The salient features of v0.6.0 are

- the release of the primal decomposition (PriDec) solver for structured two-stage problems
- improved support for (NVIDIA) GPUs for solving sparse optimization problems via NVIDIA's cuSOLVER API and newly developed condensed optimization kernels.

Other notable capabilities include
 - improved accuracy in the computations of the search directions via Krylov-based iterative refinement
 - design of a matrix interface for sparse matrices in compressed sparse row format and (capable) CPU reference implementation

## Version 0.5.4: Elastic mode, Krylov solvers, and misc bug fixes (March 2, 2022)
New algorithmic features related to the NLP solver(s) and associated linear algebra KKT systems

 - soft feasibility restoration
 - Relaxer of equality constraints at the NLP formulation level
 - Krylov interfaces and implementation for CG and BiCGStab (ready for device computations)
 - protype of the condensed linear system and initial Krylov-based iterative refinement
 - update of the Magma solver class for the latest Magma API
 - elastic mode
 
This release also includes several bug fixes.

## Version 0.5.3: xSDK compliance (Dec 3, 2021)

 xSDK compliance 
 
## Version 0.5.2: xSDK compliance and misc bug fixes (Dec 2, 2021)

 - fixed bugs in the IPM solver: gradient scaling on CUDA, unscaled objective in the  user callbacks, lambda capture fix in axpy for ROCm
 - exported sparse config in cmake
 - added user options for the algorithm parameters in PriDec solver
 
## Version 0.5.1: Objective scaling factor fix (Oct 21, 2021)

Modified the computation of the scaling factor to use the user-specified initial point

## Version 0.5.0: MDS device computations, and porting of sparse kernels (Sep 30, 2021)

The salient features of this major release are

 - update of the interface to MAGMA and capability for running mixed dense-sparse (MDS) problems solely in the device memory space
 - added interface PARDISO linear solver
 - porting of the sparse linear algebra kernels to device via RAJA performance portability layer
 - various optimizations and bug fixes for the RAJA-based dense linear algebra kernels
 - Primal decomposition solver HiOp-PriDec available as a release candidate
