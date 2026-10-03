//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.


/**
 * @file hiopLinSolverSparseReSolve.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 *
 * @brief Sparse symmetric linear solver adapter for ReSolve's SystemSolver.
 *
 * The adapter converts HiOp's symmetric triplet matrix to CSR and drives
 * ReSolve::SystemSolver, which performs the initial KLU analysis and
 * factorization, sets up the selected host or accelerator refactorization
 * backend, and optionally wraps the triangular solve in FGMRES iterative
 * refinement.
 */

#include "hiopLinSolverSparseReSolve.hpp"

#include "LinAlgFactory.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopVector.hpp"

#include "hiopCppStdUtils.hpp"

#include <resolve/LinSolverDirect.hpp>
#include <resolve/LinSolverIterative.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

#ifdef HIOP_USE_CUDA
#include <cuda_runtime.h>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

#ifdef HIOP_USE_HIP
#include <hip/hip_runtime.h>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace hiop
{
namespace
{

/**
 * @brief Copy a device buffer to a host mirror during matrix setup.
 *
 * Runtime failures are reported through HiOp's logger and propagated to the
 * caller rather than being hidden behind debug-only assertions.
 * 
 * @warning We assume that eithr CUDA or HIP backend are enabled but not both.
 * So far, this has been a safe assumtion.
 */
#ifdef HIOP_USE_CUDA
bool copy_device_to_host(hiopNlpFormulation* nlp, void* destination, const void* source, size_t bytes, const char* operation)
{
  const cudaError_t status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);

  if(status == cudaSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "CUDA failure during %s: %s\n", operation, cudaGetErrorString(status));

  return false;
}
#endif

#ifdef HIOP_USE_HIP
bool copy_device_to_host(hiopNlpFormulation* nlp, void* destination, const void* source, size_t bytes, const char* operation)
{
  const hipError_t status = hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost);

  if(status == hipSuccess) {
    return true;
  }

  nlp->log->printf(hovError, "HIP failure during %s: %s\n", operation, hipGetErrorString(status));

  return false;
}
#endif

#ifdef HIOP_USE_GPU

/**
 * @brief Gather source values into a destination ordering.
 *
 * Equivalent host operation:
 * @code
 * for(I i = 0; i < n; ++i) {
 *   dst[i] = src[mapidx[i]];
 * }
 * @endcode
 */
template<typename T, typename I>
__global__ void mapArraysKernel(T* dst, const T* src, const I* mapidx, I n)
{
  I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    dst[tid] = src[mapidx[tid]];
  }
}

/**
 * @brief Add separately stored diagonal entries to mapped CSR values.
 *
 * Equivalent host operation:
 * @code
 * for(I i = 0; i < n; ++i) {
 *   if(mapidx[i] != -1) {
 *     dst[mapidx[i]] += src[nnz - n + i];
 *   }
 * }
 * @endcode
 */
template<typename T, typename I>
__global__ void addToArrayKernel(T* dst, const T* src, const I* mapidx, I n, I nnz)
{
  I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    if(mapidx[tid] != -1) {
      dst[mapidx[tid]] += src[nnz - n + tid];
    }
  }
}

#endif


}  // namespace

hiopLinSolverSymSparseReSolve::hiopLinSolverSymSparseReSolve(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      M_host_(nullptr),
      n_(n),
      nnz_(0),
      index_convert_CSR2Triplet_host_(nullptr),
      index_convert_extra_Diag2CSR_host_(nullptr),
      index_convert_CSR2Triplet_device_(nullptr),
      index_convert_extra_Diag2CSR_device_(nullptr),
      factorizationSetupSucc_(0),
      refactorization_setup_complete_(false),
      is_first_call_(true),
      use_ir_(false),
      solve_on_device_(false),
      refactorization_method_("klu"),
      matrix_(nullptr),
      rhs_(nullptr),
      solution_(nullptr),
      cpu_workspace_(nullptr),
#ifdef HIOP_USE_CUDA
      cuda_workspace_(nullptr),
#endif
#ifdef HIOP_USE_HIP
      hip_workspace_(nullptr),
#endif
      solver_(nullptr)
{
  const std::string mem_space = nlp_->options->GetString("mem_space");

  if(mem_space == "device") {
#ifdef HIOP_USE_GPU
    M_host_ = LinearAlgebraFactory::create_matrix_sparse("default", n, n, nnz);
#else
    nlp_->log->printf(hovError, "ReSolve device execution requires a CUDA or HIP build.\n");
    std::abort();
#endif
  } else if(mem_space != "host" && mem_space != "default") {
    nlp_->log->printf(hovError, "Memory space %s is not supported by ReSolve.\n", mem_space.c_str());
    std::abort();
  }

#ifdef HIOP_USE_GPU
  const std::string compute_mode = nlp_->options->GetString("compute_mode");

  if(mem_space == "device" && compute_mode == "cpu") {
    nlp_->log->printf(hovError, "ReSolve CPU execution does not support device-resident input.\n");
    std::abort();
  }

  // In auto mode, device-resident input selects device execution.
  // Explicit compute_mode values are not overridden.
  solve_on_device_ = compute_mode == "hybrid" || compute_mode == "gpu" ||
                     (compute_mode == "auto" && mem_space == "device");
#endif

  const std::string ordering = nlp_->options->GetString("linear_solver_sparse_ordering");

  int ordering_method = 1;

  if(ordering == "amd-ssparse") {
    ordering_method = 0;
  } else if(ordering != "colamd-ssparse") {
    nlp_->log->printf(hovWarning,
                      "Ordering %s is not supported by ReSolve; "
                      "using colamd-ssparse.\n",
                      ordering.c_str());
  }

  // Select the Re::Solve refactorization method and create the system solver
  // on the matching workspace. SystemSolver owns KLU, the refactorization
  // backend, and any iterative-refinement components.
  if(solve_on_device_) {
#ifdef HIOP_USE_GPU
    std::string refactorization = nlp_->options->GetString("resolve_refactorization");

#ifdef HIOP_USE_CUDA
    if (refactorization == "rf")
      refactorization = "cusolverrf";
    refactorization_method_ = refactorization;

    cuda_workspace_ = new ReSolve::LinAlgWorkspaceCUDA();
    cuda_workspace_->initializeHandles();

    solver_ = new ReSolve::SystemSolver(cuda_workspace_,
                                        "klu",
                                        refactorization_method_,
                                        refactorization_method_,
                                        "none",
                                        "none");
#elif defined(HIOP_USE_HIP)
    if(refactorization == "glu") {
      nlp_->log->printf(hovWarning, "GLU is unavailable with HIP; using rocSolverRf.\n");
      refactorization = "rocsolverrf";
    }
    if (refactorization == "rf")
      refactorization = "rocsolverrf";

    refactorization_method_ = refactorization;

    hip_workspace_ = new ReSolve::LinAlgWorkspaceHIP();
    hip_workspace_->initializeHandles();

    solver_ = new ReSolve::SystemSolver(hip_workspace_,
                                        "klu",
                                        refactorization_method_,
                                        refactorization_method_,
                                        "none",
                                        "none");
#endif
#endif
  } else {
    refactorization_method_ = "klu";

    cpu_workspace_ = new ReSolve::LinAlgWorkspaceCpu();
    cpu_workspace_->initializeHandles();

    solver_ = new ReSolve::SystemSolver(cpu_workspace_, "klu", "klu", "klu", "none", "none");
  }

  assert(solver_ != nullptr);

  // SystemSolver only exposes the KLU factorization solver through its
  // LinSolverDirect base, so KLU-specific options go through the CLI
  // parameter interface.
  ReSolve::LinSolverDirect& klu = solver_->getFactorizationSolver();

  klu.setCliParam("ordering", std::to_string(ordering_method));
  klu.setCliParam("halt_if_singular", "yes");

  rhs_ = new ReSolve::vector::Vector(n_);
  solution_ = new ReSolve::vector::Vector(n_);

  const auto vector_memory = solve_on_device_ ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  const int rhs_status = rhs_->allocate(vector_memory);

  const int solution_status = solution_->allocate(vector_memory);

  if(rhs_status != 0 || solution_status != 0) {
    nlp_->log->printf(hovError, "Failed to allocate ReSolve vectors.\n");
    std::abort();
  }

  setup_iterative_refinement();

  nlp_->log->printf(hovSummary, "Ordering: %d\n", ordering_method);

  nlp_->log->printf(hovSummary, "Factorization: klu\n");

  nlp_->log->printf(hovSummary, "Refactorization: %s\n", refactorization_method_.c_str());

  nlp_->log->printf(hovSummary, "Use IR: %s\n", use_ir_ ? "yes" : "no");
}

void hiopLinSolverSymSparseReSolve::setup_iterative_refinement()
{
  assert(solver_ != nullptr);

  const int ir_maxit = nlp_->options->GetInteger("ir_inner_maxit");

  // ReSolve does not support iterative refinement on the host, and the
  // adapter keeps GLU restricted to the direct solve as before.
  if(!solve_on_device_ || ir_maxit <= 0) {
    return;
  }

  if(refactorization_method_ == "glu") {
    nlp_->log->printf(hovWarning,
                      "ReSolve iterative refinement is supported only with RF; "
                      "disabling it for CUDA GLU.\n");
    return;
  }

  const std::string gs_method = nlp_->options->GetString("ir_inner_gs_scheme");

  solver_->setRefinementMethod("fgmres", gs_method);

  if(solver_->getRefinementMethod() != "fgmres") {
    nlp_->log->printf(hovWarning,
                      "ReSolve iterative refinement configuration failed; "
                      "using the direct solution only.\n");
    return;
  }

  ReSolve::LinSolverIterative& fgmres = solver_->getIterativeSolver();

  fgmres.setTol(nlp_->options->GetNumeric("ir_inner_tol"));
  fgmres.setMaxit(ir_maxit);
  fgmres.setCliParam("restart", std::to_string(nlp_->options->GetInteger("ir_inner_restart")));
  fgmres.setCliParam("conv_cond", std::to_string(nlp_->options->GetInteger("ir_inner_conv_cond")));

  use_ir_ = true;
}

hiopLinSolverSymSparseReSolve::~hiopLinSolverSymSparseReSolve()
{
  // The system solver references the workspace, so destroy it first.
  delete solver_;
  delete rhs_;
  delete solution_;
  delete matrix_;

  delete cpu_workspace_;
#ifdef HIOP_USE_CUDA
  delete cuda_workspace_;
#endif
#ifdef HIOP_USE_HIP
  delete hip_workspace_;
#endif
  delete M_host_;
  delete[] index_convert_CSR2Triplet_host_;
  delete[] index_convert_extra_Diag2CSR_host_;

#ifdef HIOP_USE_CUDA
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    cudaFree(index_convert_CSR2Triplet_device_);
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    cudaFree(index_convert_extra_Diag2CSR_device_);
  }
#endif
#ifdef HIOP_USE_HIP
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    const hipError_t status = hipFree(index_convert_CSR2Triplet_device_);
    (void)status;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    const hipError_t status = hipFree(index_convert_extra_Diag2CSR_device_);
    (void)status;
  }
#endif
}

int hiopLinSolverSymSparseReSolve::matrixChanged()
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(solver_ != nullptr);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(is_first_call_) {
    if(firstCall() != 0) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  } else {
    if(update_matrix_values() != 0) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  int status = 0;

  // KLU is used directly on the host and supplies the initial factors
  // required to set up an accelerator backend. Once that one-time setup is
  // complete, later matrix changes require only numerical refactorization.
  const bool needs_full_factorization = !solve_on_device_ || !refactorization_setup_complete_;

  if(factorizationSetupSucc_ == 0 && needs_full_factorization) {
    status = solver_->factorize();

    if(status != 0) {
      nlp_->log->printf(hovWarning, "ReSolve KLU factorization failed. Regularizing ...\n");

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

    if(solve_on_device_) {
      // Extracts the KLU factors, sets up the accelerator backend, and, when
      // enabled, attaches FGMRES refinement preconditioned by that backend.
      status = solver_->refactorizationSetup();

      if(status != 0) {
        nlp_->log->printf(hovWarning, "ReSolve refactorization solver setup failed.\n");

        nlp_->runStats.linsolv.tmFactTime.stop();
        return -1;
      }

      refactorization_setup_complete_ = true;

      // GLU setup already performs the first numerical factorization. The RF
      // backends only import and analyze the KLU factors, and SystemSolver
      // routes the triangular solve through host KLU until refactorize() has
      // run on the device. The vectors here are device-only, so the first
      // device refactorization must happen before the first solve.
      if(refactorization_method_ != "glu") {
        status = solver_->refactorize();
      }
    }

    if(status != 0) {
      nlp_->log->printf(hovWarning,
                        "ReSolve initial numerical refactorization failed. "
                        "Regularizing ...\n");

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

    factorizationSetupSucc_ = 1;

    nlp_->log->printf(hovScalars, "ReSolve factorization setup successful.\n");
  } else {
    status = solver_->refactorize();

    if(status != 0) {
      nlp_->log->printf(hovWarning, "ReSolve refactorization failed. Regularizing ...\n");

      factorizationSetupSucc_ = 0;

      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }

    factorizationSetupSucc_ = 1;
  }

  nlp_->runStats.linsolv.tmFactTime.stop();
  return 0;
}

bool hiopLinSolverSymSparseReSolve::solve(hiopVector& x)
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(x.get_size() == M_->n());
  assert(solver_ != nullptr);

  if(factorizationSetupSucc_ == 0) {
    nlp_->log->printf(hovError, "ReSolve solve requested without a valid factorization.\n");
    return false;
  }

  double* x_data = x.local_data();

  if(x_data == nullptr) {
    nlp_->log->printf(hovError, "Failed to access the HiOp solve vector.\n");
    return false;
  }

  nlp_->runStats.linsolv.tmTriuSolves.start();

  const std::string mem_space = nlp_->options->GetString("mem_space");

  // HiOp vector memory depends on mem_space; ReSolve vector memory depends on the selected backend.
  const auto external_memory = mem_space == "device" ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  const auto internal_memory = solve_on_device_ ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  if(rhs_->copyFromExternal(x_data, external_memory, internal_memory) != 0) {
    nlp_->log->printf(hovError, "Failed to copy the right-hand side into ReSolve.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  // SystemSolver runs the triangular solve and, when enabled, FGMRES
  // refinement starting from the direct solution.
  if(solver_->solve(rhs_, solution_) != 0) {
    nlp_->log->printf(hovError, "ReSolve solve failed.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  if(use_ir_) {
    ReSolve::LinSolverIterative& fgmres = solver_->getIterativeSolver();

    nlp_->log->printf(hovScalars,
                      "ReSolve IR iterations: %d, "
                      "final relative residual: %e\n",
                      static_cast<int>(fgmres.getNumIter()),
                      fgmres.getFinalResidualNorm());
  }

  if(solution_->copyToExternal(x_data, internal_memory, external_memory) != 0) {
    nlp_->log->printf(hovError, "Failed to copy the ReSolve solution into HiOp.\n");

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return true;
}

int hiopLinSolverSymSparseReSolve::firstCall()
{
  assert(M_ != nullptr);
  assert(n_ == M_->n());
  assert(M_->n() == M_->m());
  assert(n_ > 0);
  assert(solver_ != nullptr);

#ifdef HIOP_USE_GPU
  const std::string mem_space = nlp_->options->GetString("mem_space");

  if(mem_space == "device") {
    assert(M_host_ != nullptr);

    if(!copy_device_to_host(nlp_,
                            M_host_->M(),
                            M_->M(),
                            sizeof(double) * static_cast<size_t>(M_->numberOfNonzeros()),
                            "copying ReSolve values to the host")) {
      return -1;
    }

    if(!copy_device_to_host(nlp_,
                            M_host_->i_row(),
                            M_->i_row(),
                            sizeof(index_type) * static_cast<size_t>(M_->numberOfNonzeros()),
                            "copying ReSolve row indices to the host")) {
      return -1;
    }

    if(!copy_device_to_host(nlp_,
                            M_host_->j_col(),
                            M_->j_col(),
                            sizeof(index_type) * static_cast<size_t>(M_->numberOfNonzeros()),
                            "copying ReSolve column indices to the host")) {
      return -1;
    }
  }
#endif

  compute_nnz();

  // firstCall() may be re-entered after HiOp regularizes following a
  // setup failure. Clear partial matrix and mapping state before rebuilding.
  delete matrix_;
  matrix_ = nullptr;

  delete[] index_convert_CSR2Triplet_host_;
  index_convert_CSR2Triplet_host_ = nullptr;

  delete[] index_convert_extra_Diag2CSR_host_;
  index_convert_extra_Diag2CSR_host_ = nullptr;

#ifdef HIOP_USE_CUDA
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    const cudaError_t status = cudaFree(index_convert_CSR2Triplet_device_);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure freeing the CSR-to-triplet mapping: %s\n", cudaGetErrorString(status));
      return -1;
    }

    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    const cudaError_t status = cudaFree(index_convert_extra_Diag2CSR_device_);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure freeing the diagonal-to-CSR mapping: %s\n", cudaGetErrorString(status));
      return -1;
    }

    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#endif
#ifdef HIOP_USE_HIP
  if(index_convert_CSR2Triplet_device_ != nullptr) {
    const hipError_t status = hipFree(index_convert_CSR2Triplet_device_);

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure freeing the CSR-to-triplet mapping: %s\n", hipGetErrorString(status));
      return -1;
    }

    index_convert_CSR2Triplet_device_ = nullptr;
  }

  if(index_convert_extra_Diag2CSR_device_ != nullptr) {
    const hipError_t status = hipFree(index_convert_extra_Diag2CSR_device_);

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure freeing the diagonal-to-CSR mapping: %s\n", hipGetErrorString(status));
      return -1;
    }

    index_convert_extra_Diag2CSR_device_ = nullptr;
  }
#endif

  // ReSolve::Csr owns all CSR arrays, so allocate the complete structure
  // after the expanded symmetric nonzero count is known.
  matrix_ = new ReSolve::matrix::Csr(n_, n_, nnz_, true, true);

  if(matrix_->allocateMatrixData(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to allocate the ReSolve CSR matrix.\n");
    return -1;
  }

  if(set_csr_indices_values() != 0) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve CSR matrix.\n");
    return -1;
  }

  if(matrix_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to mark the ReSolve matrix as updated.\n");
    return -1;
  }

#ifdef HIOP_USE_GPU
  if(solve_on_device_) {
    if(matrix_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to allocate ReSolve matrix device storage.\n");
      return -1;
    }

    if(matrix_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to copy the ReSolve matrix to device memory.\n");
      return -1;
    }
  }
#endif

  // setMatrix() may be called again after a regularization retry rebuilds
  // matrix_; SystemSolver replaces its internal references accordingly.
  if(solver_->setMatrix(matrix_) != 0) {
    nlp_->log->printf(hovError, "ReSolve system solver matrix setup failed.\n");
    return -1;
  }

  if(solver_->analyze() != 0) {
    nlp_->log->printf(hovError, "ReSolve KLU symbolic analysis failed.\n");
    return -1;
  }

  is_first_call_ = false;
  return 0;
}

int hiopLinSolverSymSparseReSolve::update_matrix_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

#ifdef HIOP_USE_GPU
  const std::string mem_space = nlp_->options->GetString("mem_space");

  if(mem_space == "device") {
    double* values = matrix_->getValues(ReSolve::memory::DEVICE);

    const double* source_values = M_->M();

    if(values == nullptr || source_values == nullptr || index_convert_CSR2Triplet_device_ == nullptr ||
       index_convert_extra_Diag2CSR_device_ == nullptr) {
      nlp_->log->printf(hovError, "Failed to access ReSolve device matrix data.\n");
      return -1;
    }

    const uint32_t blocksize = 512;
    uint32_t gridsize = (static_cast<uint32_t>(nnz_) + blocksize - 1) / blocksize;

    mapArraysKernel<double, int>
      <<<gridsize, blocksize>>>(
          values,
          source_values,
          index_convert_CSR2Triplet_device_,
          nnz_);

#ifdef HIOP_USE_CUDA
    cudaError_t cuda_launch_status = cudaGetLastError();

    if(cuda_launch_status != cudaSuccess) {
      nlp_->log->printf(hovError,
                        "CUDA failure launching the CSR value-mapping kernel: %s\n",
                        cudaGetErrorString(cuda_launch_status));
      return -1;
    }
#endif
#ifdef HIOP_USE_HIP
    hipError_t hip_launch_status = hipGetLastError();

    if(hip_launch_status != hipSuccess) {
      nlp_->log->printf(hovError,
                        "HIP failure launching the CSR value-mapping kernel: %s\n",
                        hipGetErrorString(hip_launch_status));
      return -1;
    }
#endif

    gridsize = (static_cast<uint32_t>(n_) + blocksize - 1) / blocksize;

    addToArrayKernel<double, int>
        <<<gridsize, blocksize>>>(
            values, source_values, index_convert_extra_Diag2CSR_device_, n_, M_->numberOfNonzeros());

#ifdef HIOP_USE_CUDA
    cuda_launch_status = cudaGetLastError();

    if(cuda_launch_status != cudaSuccess) {
      nlp_->log->printf(hovError,
                        "CUDA failure launching the diagonal-update kernel: %s\n",
                        cudaGetErrorString(cuda_launch_status));
      return -1;
    }
#endif
#ifdef HIOP_USE_HIP
    hip_launch_status = hipGetLastError();

    if(hip_launch_status != hipSuccess) {
      nlp_->log->printf(hovError,
                        "HIP failure launching the diagonal-update kernel: %s\n",
                        hipGetErrorString(hip_launch_status));
      return -1;
    }
#endif

    if(matrix_->setUpdated(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to mark ReSolve device matrix values as updated.\n");
      return -1;
    }

    // KLU consumes host values during initial factorization. After
    // accelerator setup, numerical refactorization consumes device values
    // directly and no host synchronization is required.
    if(factorizationSetupSucc_ == 0 && !refactorization_setup_complete_) {
      if(matrix_->syncData(ReSolve::memory::HOST) != 0) {
        nlp_->log->printf(hovError, "Failed to synchronize ReSolve matrix values to the host.\n");
        return -1;
      }
    }

    return 0;
  }
#endif  // HIOP_USE_GPU

  // Update values on the host when device execution is unavailable
  // or not selected at runtime.
  double* values = matrix_->getValues(ReSolve::memory::HOST);

  if(values == nullptr) {
    nlp_->log->printf(hovError, "Failed to access ReSolve host matrix values.\n");
    return -1;
  }

  for(int k = 0; k < nnz_; ++k) {
    values[k] = M_->M()[index_convert_CSR2Triplet_host_[k]];
  }

  for(int i = 0; i < n_; ++i) {
    if(index_convert_extra_Diag2CSR_host_[i] != -1) {
      values[index_convert_extra_Diag2CSR_host_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
  }

  if(matrix_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to mark ReSolve host matrix values as updated.\n");
    return -1;
  }

#ifdef HIOP_USE_GPU
  if(solve_on_device_) {
    // Accelerator refactorization consumes device values.
    if(matrix_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to synchronize ReSolve matrix values to the device.\n");
      return -1;
    }
  }
#endif

  return 0;
}

hiopMatrixSparse* hiopLinSolverSymSparseReSolve::host_matrix() const
{
  return nlp_->options->GetString("mem_space") == "device" ? M_host_ : M_;
}

void hiopLinSolverSymSparseReSolve::compute_nnz()
{
  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  // Reserve one diagonal CSR entry per row.
  nnz_ = n_;

  // Count both symmetric CSR positions for each off-diagonal triplet.
  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    if(source->i_row()[k] != source->j_col()[k]) {
      nnz_ += 2;
    }
  }
}

int hiopLinSolverSymSparseReSolve::set_csr_indices_values()
{
  assert(M_ != nullptr);
  assert(matrix_ != nullptr);

  hiopMatrixSparse* source = host_matrix();
  assert(source != nullptr);

  // HiOp stores the structural sparse entries first, followed by n additional
  // diagonal entries. ReSolve maps the trailing diagonal entries separately
  // so their values can be updated independently.
  ReSolve::index_type* row_ptr = matrix_->getRowData(ReSolve::memory::HOST);

  ReSolve::index_type* col_idx = matrix_->getColData(ReSolve::memory::HOST);

  double* values = matrix_->getValues(ReSolve::memory::HOST);

  std::fill(row_ptr, row_ptr + n_ + 1, 0);

  // Build CSR row offsets by counting both symmetric positions for each
  // off-diagonal triplet, reserving one diagonal entry per row, and
  // prefix-summing the row counts.
  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    if(source->i_row()[k] != source->j_col()[k]) {
      row_ptr[source->i_row()[k] + 1]++;
      row_ptr[source->j_col()[k] + 1]++;
    }
  }

  for(int i = 0; i < n_; ++i) {
    row_ptr[i + 1]++;
  }

  for(int i = 1; i < n_ + 1; ++i) {
    row_ptr[i] += row_ptr[i - 1];
  }

  assert(nnz_ == row_ptr[n_]);

  index_convert_CSR2Triplet_host_ = new int[static_cast<size_t>(nnz_)];

  index_convert_extra_Diag2CSR_host_ = new int[static_cast<size_t>(n_)];

  int* nnz_each_row_tmp = new int[static_cast<size_t>(n_)]{0};

  int total_nnz_tmp{0};
  int nnz_tmp{0};
  int rowID_tmp{0};
  int colID_tmp{0};

  for(int i = 0; i < n_; ++i) {
    index_convert_extra_Diag2CSR_host_[i] = -1;
  }

  // Populate CSR values and mappings from the structural triplets,
  // expanding each off-diagonal entry into both symmetric positions.
  for(int k = 0; k < source->numberOfNonzeros() - n_; ++k) {
    rowID_tmp = source->i_row()[k];
    colID_tmp = source->j_col()[k];

    if(rowID_tmp == colID_tmp) {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      values[nnz_tmp] += source->M()[source->numberOfNonzeros() - n_ + rowID_tmp];

      index_convert_extra_Diag2CSR_host_[rowID_tmp] = nnz_tmp;

      nnz_each_row_tmp[rowID_tmp]++;
      total_nnz_tmp++;
    } else {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];

      col_idx[nnz_tmp] = colID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_tmp = nnz_each_row_tmp[colID_tmp] + row_ptr[colID_tmp];

      col_idx[nnz_tmp] = rowID_tmp;
      values[nnz_tmp] = source->M()[k];

      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_each_row_tmp[rowID_tmp]++;
      nnz_each_row_tmp[colID_tmp]++;
      total_nnz_tmp += 2;
    }
  }

  // Insert any diagonal missing from the structural triplets, then sort
  // the completed row and reorder its values and mappings consistently.
  for(int i = 0; i < n_; ++i) {
    if(nnz_each_row_tmp[i] != row_ptr[i + 1] - row_ptr[i]) {
      assert(nnz_each_row_tmp[i] == row_ptr[i + 1] - row_ptr[i] - 1);

      nnz_tmp = nnz_each_row_tmp[i] + row_ptr[i];

      col_idx[nnz_tmp] = i;

      values[nnz_tmp] = source->M()[source->numberOfNonzeros() - n_ + i];

      index_convert_CSR2Triplet_host_[nnz_tmp] = source->numberOfNonzeros() - n_ + i;

      total_nnz_tmp++;

      std::vector<int> permutation(static_cast<size_t>(row_ptr[i + 1] - row_ptr[i]));

      std::iota(permutation.begin(), permutation.end(), 0);

      std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        return col_idx[a + row_ptr[i]] < col_idx[b + row_ptr[i]];
      });

      reorder(values + row_ptr[i], permutation, row_ptr[i + 1] - row_ptr[i]);

      reorder(index_convert_CSR2Triplet_host_ + row_ptr[i], permutation, row_ptr[i + 1] - row_ptr[i]);

      std::sort(col_idx + row_ptr[i], col_idx + row_ptr[i + 1]);
    }
  }

  assert(total_nnz_tmp == nnz_);
  (void)total_nnz_tmp;

#ifdef HIOP_USE_GPU
  const std::string mem_space = nlp_->options->GetString("mem_space");
#endif

#ifdef HIOP_USE_CUDA
  if(mem_space == "device") {
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&index_convert_CSR2Triplet_device_), sizeof(int) * nnz_);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure allocating the CSR-to-triplet mapping: %s\n", cudaGetErrorString(status));

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = cudaMalloc(reinterpret_cast<void**>(&index_convert_extra_Diag2CSR_device_), sizeof(int) * n_);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure allocating the diagonal-to-CSR mapping: %s\n", cudaGetErrorString(status));

      cudaFree(index_convert_CSR2Triplet_device_);
      index_convert_CSR2Triplet_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = cudaMemcpy(index_convert_CSR2Triplet_device_,
                        index_convert_CSR2Triplet_host_,
                        sizeof(int) * nnz_,
                        cudaMemcpyHostToDevice);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure copying the CSR-to-triplet mapping: %s\n", cudaGetErrorString(status));

      cudaFree(index_convert_CSR2Triplet_device_);
      cudaFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = cudaMemcpy(index_convert_extra_Diag2CSR_device_,
                        index_convert_extra_Diag2CSR_host_,
                        sizeof(int) * n_,
                        cudaMemcpyHostToDevice);

    if(status != cudaSuccess) {
      nlp_->log->printf(hovError, "CUDA failure copying the diagonal-to-CSR mapping: %s\n", cudaGetErrorString(status));

      cudaFree(index_convert_CSR2Triplet_device_);
      cudaFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }
  }
#endif

#ifdef HIOP_USE_HIP
  if(mem_space == "device") {
    hipError_t status = hipMalloc(reinterpret_cast<void**>(&index_convert_CSR2Triplet_device_), sizeof(int) * static_cast<size_t>(nnz_));

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure allocating the CSR-to-triplet mapping: %s\n", hipGetErrorString(status));

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = hipMalloc(reinterpret_cast<void**>(&index_convert_extra_Diag2CSR_device_), sizeof(int) * static_cast<size_t>(n_));

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure allocating the diagonal-to-CSR mapping: %s\n", hipGetErrorString(status));

      status = hipFree(index_convert_CSR2Triplet_device_);
      index_convert_CSR2Triplet_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = hipMemcpy(index_convert_CSR2Triplet_device_,
                       index_convert_CSR2Triplet_host_,
                       sizeof(int) * static_cast<size_t>(nnz_),
                       hipMemcpyHostToDevice);

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure copying the CSR-to-triplet mapping: %s\n", hipGetErrorString(status));

      status = hipFree(index_convert_CSR2Triplet_device_);
      status = hipFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }

    status = hipMemcpy(index_convert_extra_Diag2CSR_device_,
                       index_convert_extra_Diag2CSR_host_,
                       sizeof(int) * static_cast<size_t>(n_),
                       hipMemcpyHostToDevice);

    if(status != hipSuccess) {
      nlp_->log->printf(hovError, "HIP failure copying the diagonal-to-CSR mapping: %s\n", hipGetErrorString(status));

      status = hipFree(index_convert_CSR2Triplet_device_);
      status = hipFree(index_convert_extra_Diag2CSR_device_);
      index_convert_CSR2Triplet_device_ = nullptr;
      index_convert_extra_Diag2CSR_device_ = nullptr;

      delete[] nnz_each_row_tmp;
      return -1;
    }
  }
#endif

  delete[] nnz_each_row_tmp;
  return 0;
}

}  // namespace hiop
