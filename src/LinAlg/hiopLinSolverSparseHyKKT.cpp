// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.


/**
 * @file hiopLinSolverSparseHyKKT.cpp
 *
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 *
 * @brief Sparse KKT solver adapter for ReSolve's HyKKT.
 */

#include "hiopLinSolverSparseHyKKT.hpp"

#include "hiopMatrixSparse.hpp"
#include "hiopVector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#ifdef HIOP_USE_RAJA
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#endif

#include <resolve/hykkt/HyKKTSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>

#ifdef HIOP_USE_CUDA
#include <cuda_runtime.h>
#include <resolve/workspace/LinAlgWorkspaceCUDA.hpp>
#endif

#ifdef HIOP_USE_HIP
#include <hip/hip_runtime.h>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#endif

namespace
{
#ifdef HIOP_USE_GPU

template<typename T, typename I>
__global__ void map_triplet_to_csr(T* dst, const T* src, const I* map, I n)
{
  const I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    const I triplet = map[tid];
    dst[tid] = triplet >= 0 ? src[triplet] : T{0};
  }
}

template<typename T, typename I>
__global__ void add_diagonal_to_csr(T* dst, const T* diagonal, const I* map, I n)
{
  const I tid = static_cast<I>(blockDim.x * blockIdx.x + threadIdx.x);

  if(tid < n) {
    dst[map[tid]] += diagonal[tid];
  }
}

bool copy_mapping_to_device(int*& dst, const int* src, size_t count)
{
  if(count == 0) {
    dst = nullptr;
    return true;
  }

  if(src == nullptr) {
    return false;
  }

#ifdef HIOP_USE_CUDA
  if(cudaMalloc(reinterpret_cast<void**>(&dst), count * sizeof(int)) != cudaSuccess) {
    return false;
  }

  if(cudaMemcpy(dst, src, count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dst);
    dst = nullptr;
    return false;
  }
#elif defined(HIOP_USE_HIP)
  if(hipMalloc(reinterpret_cast<void**>(&dst), count * sizeof(int)) != hipSuccess) {
    return false;
  }

  if(hipMemcpy(dst, src, count * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(dst);
    dst = nullptr;
    return false;
  }
#endif

  return true;
}

#endif

struct CsrEntry
{
  int row;
  int col;
  int triplet;
};

/**
 * Build a sorted ReSolve CSR structure and a mapping back to HiOp triplet entries.
 *
 * Symmetric matrices are expanded to full CSR. When ensure_diagonal is true,
 * missing diagonal entries are inserted with a triplet mapping of -1.
 */
bool build_csr_structure(int nrows,
                         int ncols,
                         const int* triplet_rows,
                         const int* triplet_cols,
                         int triplet_nnz,
                         bool symmetric,
                         bool ensure_diagonal,
                         ReSolve::matrix::Csr*& matrix,
                         int*& csr_to_triplet,
                         int*& diag_to_csr)
{
  std::vector<CsrEntry> entries;
  entries.reserve(static_cast<size_t>(symmetric ? 2 * triplet_nnz + nrows : triplet_nnz));

  std::vector<bool> has_diagonal(static_cast<size_t>(nrows), false);

  for(int k = 0; k < triplet_nnz; ++k) {
    const int row = triplet_rows[k];
    const int col = triplet_cols[k];

    entries.push_back({row, col, k});

    if(row == col) {
      has_diagonal[static_cast<size_t>(row)] = true;
    } else if(symmetric) {
      entries.push_back({col, row, k});
    }
  }

  if(ensure_diagonal) {
    for(int i = 0; i < nrows; ++i) {
      if(!has_diagonal[static_cast<size_t>(i)]) {
        entries.push_back({i, i, -1});
      }
    }
  }

  std::sort(entries.begin(), entries.end(), [](const CsrEntry& a, const CsrEntry& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  std::vector<ReSolve::index_type> row_ptr(static_cast<size_t>(nrows + 1), 0);
  std::vector<ReSolve::index_type> col_idx(entries.size());
  std::vector<double> values(entries.size(), 0.0);

  csr_to_triplet = new int[entries.size()];

  if(ensure_diagonal) {
    diag_to_csr = new int[static_cast<size_t>(nrows)];
    std::fill(diag_to_csr, diag_to_csr + nrows, -1);
  }

  for(size_t k = 0; k < entries.size(); ++k) {
    const auto& entry = entries[k];

    row_ptr[static_cast<size_t>(entry.row + 1)]++;
    col_idx[k] = entry.col;
    csr_to_triplet[k] = entry.triplet;

    if(ensure_diagonal && entry.row == entry.col) {
      diag_to_csr[entry.row] = static_cast<int>(k);
    }
  }

  std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

  matrix = new ReSolve::matrix::Csr(nrows, ncols, static_cast<ReSolve::index_type>(entries.size()), symmetric, true);

  if(matrix->allocateMatrixData(ReSolve::memory::HOST) != 0) {
    return false;
  }

  return matrix->copyFromExternal(row_ptr.data(),
                                  col_idx.data(),
                                  values.data(),
                                  ReSolve::memory::HOST,
                                  ReSolve::memory::HOST) == 0;
}

}  // namespace

namespace hiop
{

hiopLinSolverSparseHyKKT::hiopLinSolverSparseHyKKT(hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(nlp),
      HessSp_{nullptr},
      Jac_cSp_{nullptr},
      Jac_dSp_{nullptr},
      Hx_{nullptr},
      Hd_{nullptr},
      hykkt_solver_{nullptr},
      H_{nullptr},
      D_s_{nullptr},
      J_{nullptr},
      J_d_{nullptr},
      r_x_{nullptr},
      r_s_{nullptr},
      r_y_{nullptr},
      r_yd_{nullptr},
      x_{nullptr},
      s_{nullptr},
      y_{nullptr},
      y_d_{nullptr},
      cpu_workspace_{nullptr},
#ifdef HIOP_USE_CUDA
      cuda_workspace_{nullptr},
#endif
#ifdef HIOP_USE_HIP
      hip_workspace_{nullptr},
#endif
      matrix_handler_{nullptr},
      vector_handler_{nullptr},
      H_csr_to_triplet_host_{nullptr},
      H_diag_to_csr_host_{nullptr},
      J_csr_to_triplet_host_{nullptr},
      J_d_csr_to_triplet_host_{nullptr}
#ifdef HIOP_USE_GPU
      ,
      H_csr_to_triplet_device_{nullptr},
      H_diag_to_csr_device_{nullptr},
      J_csr_to_triplet_device_{nullptr},
      J_d_csr_to_triplet_device_{nullptr}
#endif
{
#ifdef HIOP_USE_GPU
  const std::string mem_space = nlp_->options->GetString("mem_space");
  const std::string compute_mode = nlp_->options->GetString("compute_mode");

  if(mem_space == "device" && compute_mode == "cpu") {
    nlp_->log->printf(hovError, "ReSolve HyKKT CPU execution does not support device-resident input.\n");
    std::abort();
  }
#endif
}

hiopLinSolverSparseHyKKT::~hiopLinSolverSparseHyKKT()
{
  delete hykkt_solver_;

  delete H_;
  delete D_s_;
  delete J_;
  delete J_d_;

  delete r_x_;
  delete r_s_;
  delete r_y_;
  delete r_yd_;

  delete x_;
  delete s_;
  delete y_;
  delete y_d_;

  delete[] H_csr_to_triplet_host_;
  delete[] H_diag_to_csr_host_;
  delete[] J_csr_to_triplet_host_;
  delete[] J_d_csr_to_triplet_host_;

#ifdef HIOP_USE_CUDA
  if(H_csr_to_triplet_device_) cudaFree(H_csr_to_triplet_device_);
  if(H_diag_to_csr_device_) cudaFree(H_diag_to_csr_device_);
  if(J_csr_to_triplet_device_) cudaFree(J_csr_to_triplet_device_);
  if(J_d_csr_to_triplet_device_) cudaFree(J_d_csr_to_triplet_device_);
#endif

#ifdef HIOP_USE_HIP
  if(H_csr_to_triplet_device_) (void)hipFree(H_csr_to_triplet_device_);
  if(H_diag_to_csr_device_) (void)hipFree(H_diag_to_csr_device_);
  if(J_csr_to_triplet_device_) (void)hipFree(J_csr_to_triplet_device_);
  if(J_d_csr_to_triplet_device_) (void)hipFree(J_d_csr_to_triplet_device_);
#endif

  delete vector_handler_;
  delete matrix_handler_;

  delete cpu_workspace_;
#ifdef HIOP_USE_CUDA
  delete cuda_workspace_;
#endif

#ifdef HIOP_USE_HIP
  delete hip_workspace_;
#endif
}

void hiopLinSolverSparseHyKKT::set_kkt_blocks(hiopMatrixSparse* Hess,
                                              const hiopMatrixSparse* Jac_c,
                                              const hiopMatrixSparse* Jac_d,
                                              const hiopVector* Hx,
                                              const hiopVector* Hd)
{
  HessSp_ = Hess;
  Jac_cSp_ = Jac_c;
  Jac_dSp_ = Jac_d;
  Hx_ = Hx;
  Hd_ = Hd;
}

bool hiopLinSolverSparseHyKKT::solve_on_device() const
{
  const std::string mem_space = nlp_->options->GetString("mem_space");
  const std::string compute_mode = nlp_->options->GetString("compute_mode");

  return compute_mode == "hybrid" || compute_mode == "gpu" || (compute_mode == "auto" && mem_space == "device");
}

bool hiopLinSolverSparseHyKKT::blocks_on_device() const { return nlp_->options->GetString("mem_space") == "device"; }

int hiopLinSolverSparseHyKKT::matrixChanged()
{
  if(!HessSp_ || !Jac_cSp_ || !Jac_dSp_ || !Hx_ || !Hd_) {
    nlp_->log->printf(hovError, "ReSolve HyKKT KKT blocks have not been set.\n");
    return -1;
  }

  nlp_->runStats.linsolv.tmFactTime.start();

  if(nullptr == H_) {
    if(!initialize_matrix_blocks()) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  if(nullptr == r_x_) {
    if(!initialize_vector_blocks()) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  if(nullptr == hykkt_solver_) {
    if(!initialize_solver()) {
      nlp_->runStats.linsolv.tmFactTime.stop();
      return -1;
    }
  }

  // HyKKT factorizes inside solve(); here only the numerical values are refreshed.
  const bool ok = update_matrix_blocks();

  nlp_->runStats.linsolv.tmFactTime.stop();

  return ok ? 0 : -1;
}

bool hiopLinSolverSparseHyKKT::solve(hiopVector& x)
{
  assert(hykkt_solver_);
  assert(r_x_);
  assert(r_s_);
  assert(r_y_);
  assert(r_yd_);
  assert(x_);
  assert(s_);
  assert(y_);
  assert(y_d_);

  if(nullptr == hykkt_solver_) {
    nlp_->log->printf(hovError, "ReSolve HyKKT solve requested before the solver was set up.\n");
    return false;
  }

  const int nx = HessSp_->n();
  const int neq = Jac_cSp_->m();
  const int nineq = Jac_dSp_->m();

  assert(x.get_size() == nx + nineq + neq + nineq);

  double* x_data = x.local_data();

  if(x_data == nullptr) {
    nlp_->log->printf(hovError, "Failed to access the HiOp solve vector.\n");
    return false;
  }

  const auto external_memory = blocks_on_device() ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;
  const auto internal_memory = solve_on_device() ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  nlp_->runStats.linsolv.tmTriuSolves.start();

  // x is the stacked right-hand side [rx; rd; ryc; ryd].
  const bool rhs_ok = r_x_->copyFromExternal(x_data, external_memory, internal_memory) == 0 &&
                      r_s_->copyFromExternal(x_data + nx, external_memory, internal_memory) == 0 &&
                      r_y_->copyFromExternal(x_data + nx + nineq, external_memory, internal_memory) == 0 &&
                      r_yd_->copyFromExternal(x_data + nx + nineq + neq, external_memory, internal_memory) == 0;

  if(!rhs_ok) {
    nlp_->log->printf(hovError, "Failed to copy HiOp RHS to ReSolve HyKKT.\n");
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  const ReSolve::real_type error = hykkt_solver_->solve();

  const ReSolve::real_type residual_tol = nlp_->options->GetNumeric("hykkt_residual_tol");
  if(!std::isfinite(error) || error >= residual_tol) {
    nlp_->log->printf(hovError, "ReSolve HyKKT solve failed with residual %e.\n", error);
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return false;
  }

  const bool solution_ok = x_->copyToExternal(x_data, internal_memory, external_memory) == 0 &&
                           s_->copyToExternal(x_data + nx, internal_memory, external_memory) == 0 &&
                           y_->copyToExternal(x_data + nx + nineq, internal_memory, external_memory) == 0 &&
                           y_d_->copyToExternal(x_data + nx + nineq + neq, internal_memory, external_memory) == 0;

  nlp_->runStats.linsolv.tmTriuSolves.stop();

  if(!solution_ok) {
    nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT solution to HiOp.\n");
    return false;
  }

  return true;
}

bool hiopLinSolverSparseHyKKT::initialize_matrix_blocks()
{
  assert(HessSp_);
  assert(Jac_cSp_);
  assert(Jac_dSp_);

  const int nx = HessSp_->n();
  const int neq = Jac_cSp_->m();
  const int nineq = Jac_dSp_->m();

#ifdef HIOP_USE_RAJA
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator host_alloc = resmgr.getAllocator("HOST");
#endif

  const size_t H_nnz = static_cast<size_t>(HessSp_->numberOfNonzeros());

#ifdef HIOP_USE_RAJA
  int* H_rows = H_nnz > 0 ? static_cast<int*>(host_alloc.allocate(H_nnz * sizeof(int))) : nullptr;
  int* H_cols = H_nnz > 0 ? static_cast<int*>(host_alloc.allocate(H_nnz * sizeof(int))) : nullptr;
  double* H_vals = H_nnz > 0 ? static_cast<double*>(host_alloc.allocate(H_nnz * sizeof(double))) : nullptr;
#else
  std::vector<int> H_rows_storage(H_nnz);
  std::vector<int> H_cols_storage(H_nnz);
  std::vector<double> H_vals_storage(H_nnz);

  int* H_rows = H_nnz > 0 ? H_rows_storage.data() : nullptr;
  int* H_cols = H_nnz > 0 ? H_cols_storage.data() : nullptr;
  double* H_vals = H_nnz > 0 ? H_vals_storage.data() : nullptr;
#endif

  if(H_nnz > 0) {
    HessSp_->copy_to(H_rows, H_cols, H_vals);
  }

  const bool H_ok = build_csr_structure(nx,
                                        nx,
                                        H_rows,
                                        H_cols,
                                        HessSp_->numberOfNonzeros(),
                                        true,
                                        true,
                                        H_,
                                        H_csr_to_triplet_host_,
                                        H_diag_to_csr_host_);

#ifdef HIOP_USE_RAJA
  if(H_rows != nullptr) {
    host_alloc.deallocate(H_rows);
  }
  if(H_cols != nullptr) {
    host_alloc.deallocate(H_cols);
  }
  if(H_vals != nullptr) {
    host_alloc.deallocate(H_vals);
  }
#endif

  if(!H_ok) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT Hessian block.\n");
    return false;
  }

  const size_t J_nnz = static_cast<size_t>(Jac_cSp_->numberOfNonzeros());

#ifdef HIOP_USE_RAJA
  int* J_rows = J_nnz > 0 ? static_cast<int*>(host_alloc.allocate(J_nnz * sizeof(int))) : nullptr;
  int* J_cols = J_nnz > 0 ? static_cast<int*>(host_alloc.allocate(J_nnz * sizeof(int))) : nullptr;
  double* J_vals = J_nnz > 0 ? static_cast<double*>(host_alloc.allocate(J_nnz * sizeof(double))) : nullptr;
#else
  std::vector<int> J_rows_storage(J_nnz);
  std::vector<int> J_cols_storage(J_nnz);
  std::vector<double> J_vals_storage(J_nnz);

  int* J_rows = J_nnz > 0 ? J_rows_storage.data() : nullptr;
  int* J_cols = J_nnz > 0 ? J_cols_storage.data() : nullptr;
  double* J_vals = J_nnz > 0 ? J_vals_storage.data() : nullptr;
#endif

  if(J_nnz > 0) {
    const_cast<hiopMatrixSparse*>(Jac_cSp_)->copy_to(J_rows, J_cols, J_vals);
  }

  int* unused_diag{nullptr};

  const bool J_ok = build_csr_structure(neq,
                                        nx,
                                        J_rows,
                                        J_cols,
                                        Jac_cSp_->numberOfNonzeros(),
                                        false,
                                        false,
                                        J_,
                                        J_csr_to_triplet_host_,
                                        unused_diag);

#ifdef HIOP_USE_RAJA
  if(J_rows != nullptr) {
    host_alloc.deallocate(J_rows);
  }
  if(J_cols != nullptr) {
    host_alloc.deallocate(J_cols);
  }
  if(J_vals != nullptr) {
    host_alloc.deallocate(J_vals);
  }
#endif

  if(!J_ok) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT equality Jacobian block.\n");
    return false;
  }

  const size_t J_d_nnz = static_cast<size_t>(Jac_dSp_->numberOfNonzeros());

#ifdef HIOP_USE_RAJA
  int* J_d_rows = J_d_nnz > 0 ? static_cast<int*>(host_alloc.allocate(J_d_nnz * sizeof(int))) : nullptr;
  int* J_d_cols = J_d_nnz > 0 ? static_cast<int*>(host_alloc.allocate(J_d_nnz * sizeof(int))) : nullptr;
  double* J_d_vals = J_d_nnz > 0 ? static_cast<double*>(host_alloc.allocate(J_d_nnz * sizeof(double))) : nullptr;
#else
  std::vector<int> J_d_rows_storage(J_d_nnz);
  std::vector<int> J_d_cols_storage(J_d_nnz);
  std::vector<double> J_d_vals_storage(J_d_nnz);

  int* J_d_rows = J_d_nnz > 0 ? J_d_rows_storage.data() : nullptr;
  int* J_d_cols = J_d_nnz > 0 ? J_d_cols_storage.data() : nullptr;
  double* J_d_vals = J_d_nnz > 0 ? J_d_vals_storage.data() : nullptr;
#endif

  if(J_d_nnz > 0) {
    const_cast<hiopMatrixSparse*>(Jac_dSp_)->copy_to(J_d_rows, J_d_cols, J_d_vals);
  }

  const bool J_d_ok = build_csr_structure(nineq,
                                          nx,
                                          J_d_rows,
                                          J_d_cols,
                                          Jac_dSp_->numberOfNonzeros(),
                                          false,
                                          false,
                                          J_d_,
                                          J_d_csr_to_triplet_host_,
                                          unused_diag);

#ifdef HIOP_USE_RAJA
  if(J_d_rows != nullptr) {
    host_alloc.deallocate(J_d_rows);
  }
  if(J_d_cols != nullptr) {
    host_alloc.deallocate(J_d_cols);
  }
  if(J_d_vals != nullptr) {
    host_alloc.deallocate(J_d_vals);
  }
#endif

  if(!J_d_ok) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT inequality Jacobian block.\n");
    return false;
  }

  std::vector<ReSolve::index_type> D_s_rows(static_cast<size_t>(nineq + 1));
  std::vector<ReSolve::index_type> D_s_cols(static_cast<size_t>(nineq));
  std::vector<double> D_s_vals(static_cast<size_t>(nineq), 0.0);

  std::iota(D_s_rows.begin(), D_s_rows.end(), 0);
  std::iota(D_s_cols.begin(), D_s_cols.end(), 0);

  D_s_ = new ReSolve::matrix::Csr(nineq, nineq, nineq, false, true);

  if(D_s_->allocateMatrixData(ReSolve::memory::HOST) != 0 || D_s_->copyFromExternal(D_s_rows.data(),
                                                                                    D_s_cols.data(),
                                                                                    D_s_vals.data(),
                                                                                    ReSolve::memory::HOST,
                                                                                    ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to construct the ReSolve HyKKT slack diagonal block.\n");
    return false;
  }

#ifdef HIOP_USE_GPU
  if(solve_on_device()) {
    if(H_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 || D_s_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 ||
       J_->allocateMatrixData(ReSolve::memory::DEVICE) != 0 || J_d_->allocateMatrixData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to allocate ReSolve HyKKT device matrix storage.\n");
      return false;
    }

    if(H_->syncData(ReSolve::memory::DEVICE) != 0 || D_s_->syncData(ReSolve::memory::DEVICE) != 0 ||
       J_->syncData(ReSolve::memory::DEVICE) != 0 || J_d_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT matrix structure to the device.\n");
      return false;
    }
  }

  if(blocks_on_device()) {
    if(!copy_mapping_to_device(H_csr_to_triplet_device_, H_csr_to_triplet_host_, static_cast<size_t>(H_->getNnz())) ||
       !copy_mapping_to_device(H_diag_to_csr_device_, H_diag_to_csr_host_, static_cast<size_t>(nx)) ||
       !copy_mapping_to_device(J_csr_to_triplet_device_, J_csr_to_triplet_host_, static_cast<size_t>(J_->getNnz())) ||
       !copy_mapping_to_device(J_d_csr_to_triplet_device_, J_d_csr_to_triplet_host_, static_cast<size_t>(J_d_->getNnz()))) {
      nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT matrix mappings to the device.\n");
      return false;
    }
  }
#endif

  return true;
}

bool hiopLinSolverSparseHyKKT::initialize_vector_blocks()
{
  assert(HessSp_);
  assert(Jac_cSp_);
  assert(Jac_dSp_);

  const int nx = HessSp_->n();
  const int neq = Jac_cSp_->m();
  const int nineq = Jac_dSp_->m();

  r_x_ = new ReSolve::vector::Vector(nx);
  r_s_ = new ReSolve::vector::Vector(nineq);
  r_y_ = new ReSolve::vector::Vector(neq);
  r_yd_ = new ReSolve::vector::Vector(nineq);

  x_ = new ReSolve::vector::Vector(nx);
  s_ = new ReSolve::vector::Vector(nineq);
  y_ = new ReSolve::vector::Vector(neq);
  y_d_ = new ReSolve::vector::Vector(nineq);

  const auto internal_memory = solve_on_device() ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  if(r_x_->allocate(internal_memory) != 0 || r_s_->allocate(internal_memory) != 0 || r_y_->allocate(internal_memory) != 0 ||
     r_yd_->allocate(internal_memory) != 0 || x_->allocateAll(internal_memory) != 0 ||
     s_->allocateAll(internal_memory) != 0 || y_->allocateAll(internal_memory) != 0 ||
     y_d_->allocateAll(internal_memory) != 0) {
    nlp_->log->printf(hovError, "Failed to allocate ReSolve HyKKT vector blocks.\n");
    return false;
  }

  return true;
}

bool hiopLinSolverSparseHyKKT::initialize_solver()
{
  assert(H_);
  assert(D_s_);
  assert(J_);
  assert(J_d_);

  assert(r_x_);
  assert(r_s_);
  assert(r_y_);
  assert(r_yd_);

  assert(x_);
  assert(s_);
  assert(y_);
  assert(y_d_);

  const auto internal_memory = solve_on_device() ? ReSolve::memory::DEVICE : ReSolve::memory::HOST;

  if(internal_memory == ReSolve::memory::DEVICE) {
#ifdef HIOP_USE_CUDA
    cuda_workspace_ = new ReSolve::LinAlgWorkspaceCUDA();
    cuda_workspace_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(cuda_workspace_);
    vector_handler_ = new ReSolve::VectorHandler(cuda_workspace_);
#elif defined(HIOP_USE_HIP)
    hip_workspace_ = new ReSolve::LinAlgWorkspaceHIP();
    hip_workspace_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(hip_workspace_);
    vector_handler_ = new ReSolve::VectorHandler(hip_workspace_);
#else
    nlp_->log->printf(hovError, "ReSolve HyKKT device backend is not available.\n");
    return false;
#endif
  } else {
    cpu_workspace_ = new ReSolve::LinAlgWorkspaceCpu();
    cpu_workspace_->initializeHandles();

    matrix_handler_ = new ReSolve::MatrixHandler(cpu_workspace_);
    vector_handler_ = new ReSolve::VectorHandler(cpu_workspace_);
  }

  hykkt_solver_ = new ReSolve::hykkt::HyKKTSolver(H_->getNumRows(), J_d_->getNumRows(), J_->getNumRows(), internal_memory);

  if(hykkt_solver_->setMatrixBlocks(H_, D_s_, J_, J_d_) != 0) {
    nlp_->log->printf(hovError, "Failed to set ReSolve HyKKT matrix blocks.\n");
    return false;
  }

  hykkt_solver_->setRHSBlocks(r_x_, r_s_, r_y_, r_yd_);
  hykkt_solver_->setLHSPointers(x_, s_, y_, y_d_);
  hykkt_solver_->setGamma(nlp_->options->GetNumeric("hykkt_gamma"));
  hykkt_solver_->addHandlers(matrix_handler_, vector_handler_);

  return true;
}

bool hiopLinSolverSparseHyKKT::update_matrix_blocks()
{
  assert(H_);
  assert(D_s_);
  assert(J_);
  assert(J_d_);
  assert(HessSp_);
  assert(Jac_cSp_);
  assert(Jac_dSp_);
  assert(Hx_);
  assert(Hd_);

  const ReSolve::index_type H_nnz = H_->getNnz();
  const ReSolve::index_type D_s_nnz = D_s_->getNnz();
  const ReSolve::index_type J_nnz = J_->getNnz();
  const ReSolve::index_type J_d_nnz = J_d_->getNnz();
  const index_type Hx_size = Hx_->get_size();

#ifdef HIOP_USE_GPU
  if(blocks_on_device()) {
    double* H_values = H_->getValues(ReSolve::memory::DEVICE);
    double* D_s_values = D_s_->getValues(ReSolve::memory::DEVICE);
    double* J_values = J_->getValues(ReSolve::memory::DEVICE);
    double* J_d_values = J_d_->getValues(ReSolve::memory::DEVICE);

    const double* Hess_values = HessSp_->M();
    const double* Jac_c_values = Jac_cSp_->M();
    const double* Jac_d_values = Jac_dSp_->M();
    const double* Hx_values = Hx_->local_data_const();
    const double* Hd_values = Hd_->local_data_const();

    if((H_nnz > 0 && (!H_values || !Hess_values || !H_csr_to_triplet_device_)) ||
       (Hx_size > 0 && (!H_values || !Hx_values || !H_diag_to_csr_device_)) ||
       (D_s_nnz > 0 && (!D_s_values || !Hd_values)) ||
       (J_nnz > 0 && (!J_values || !Jac_c_values || !J_csr_to_triplet_device_)) ||
       (J_d_nnz > 0 && (!J_d_values || !Jac_d_values || !J_d_csr_to_triplet_device_))) {
      nlp_->log->printf(hovError, "Failed to access HyKKT device matrix block values.\n");
      return false;
    }

    constexpr unsigned int blocksize = 512;

    if(H_nnz > 0) {
      const unsigned int gridsize = (static_cast<unsigned int>(H_nnz) + blocksize - 1) / blocksize;

      map_triplet_to_csr<double, int>
          <<<gridsize, blocksize>>>(H_values, Hess_values, H_csr_to_triplet_device_, static_cast<int>(H_nnz));
    }

    if(Hx_size > 0) {
      const unsigned int gridsize = (static_cast<unsigned int>(Hx_size) + blocksize - 1) / blocksize;

      add_diagonal_to_csr<double, int>
          <<<gridsize, blocksize>>>(H_values, Hx_values, H_diag_to_csr_device_, static_cast<int>(Hx_size));
    }

    if(J_nnz > 0) {
      const unsigned int gridsize = (static_cast<unsigned int>(J_nnz) + blocksize - 1) / blocksize;

      map_triplet_to_csr<double, int>
          <<<gridsize, blocksize>>>(J_values, Jac_c_values, J_csr_to_triplet_device_, static_cast<int>(J_nnz));
    }

    if(J_d_nnz > 0) {
      const unsigned int gridsize = (static_cast<unsigned int>(J_d_nnz) + blocksize - 1) / blocksize;

      map_triplet_to_csr<double, int>
          <<<gridsize, blocksize>>>(J_d_values, Jac_d_values, J_d_csr_to_triplet_device_, static_cast<int>(J_d_nnz));
    }

#ifdef HIOP_USE_CUDA
    const cudaError_t cuda_launch_status = cudaGetLastError();
    if(cuda_launch_status != cudaSuccess) {
      nlp_->log->printf(hovError,
                        "CUDA failure launching ReSolve HyKKT matrix update kernels: %s\n",
                        cudaGetErrorString(cuda_launch_status));
      return false;
    }

    if(D_s_nnz > 0) {
      const cudaError_t cuda_copy_status =
          cudaMemcpy(D_s_values, Hd_values, static_cast<size_t>(D_s_nnz) * sizeof(double), cudaMemcpyDeviceToDevice);
      if(cuda_copy_status != cudaSuccess) {
        nlp_->log->printf(hovError,
                          "CUDA failure copying ReSolve HyKKT slack diagonal values: %s\n",
                          cudaGetErrorString(cuda_copy_status));
        return false;
      }
    }
#elif defined(HIOP_USE_HIP)
    const hipError_t hip_launch_status = hipGetLastError();
    if(hip_launch_status != hipSuccess) {
      nlp_->log->printf(hovError,
                        "HIP failure launching ReSolve HyKKT matrix update kernels: %s\n",
                        hipGetErrorString(hip_launch_status));
      return false;
    }

    if(D_s_nnz > 0) {
      const hipError_t hip_copy_status =
          hipMemcpy(D_s_values, Hd_values, static_cast<size_t>(D_s_nnz) * sizeof(double), hipMemcpyDeviceToDevice);
      if(hip_copy_status != hipSuccess) {
        nlp_->log->printf(hovError,
                          "HIP failure copying ReSolve HyKKT slack diagonal values: %s\n",
                          hipGetErrorString(hip_copy_status));
        return false;
      }
    }
#endif

    if(H_->setUpdated(ReSolve::memory::DEVICE) != 0 || D_s_->setUpdated(ReSolve::memory::DEVICE) != 0 ||
       J_->setUpdated(ReSolve::memory::DEVICE) != 0 || J_d_->setUpdated(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to mark ReSolve HyKKT device matrix blocks as updated.\n");
      return false;
    }

    return true;
  }
#endif

  double* H_values = H_->getValues(ReSolve::memory::HOST);
  double* D_s_values = D_s_->getValues(ReSolve::memory::HOST);
  double* J_values = J_->getValues(ReSolve::memory::HOST);
  double* J_d_values = J_d_->getValues(ReSolve::memory::HOST);

  const double* Hess_values = HessSp_->M();
  const double* Jac_c_values = Jac_cSp_->M();
  const double* Jac_d_values = Jac_dSp_->M();
  const double* Hx_values = Hx_->local_data_const();
  const double* Hd_values = Hd_->local_data_const();

  if((H_nnz > 0 && (!H_values || !Hess_values || !H_csr_to_triplet_host_)) ||
     (Hx_size > 0 && (!H_values || !Hx_values || !H_diag_to_csr_host_)) || (D_s_nnz > 0 && (!D_s_values || !Hd_values)) ||
     (J_nnz > 0 && (!J_values || !Jac_c_values || !J_csr_to_triplet_host_)) ||
     (J_d_nnz > 0 && (!J_d_values || !Jac_d_values || !J_d_csr_to_triplet_host_))) {
    nlp_->log->printf(hovError, "Failed to access HyKKT matrix block values.\n");
    return false;
  }

  for(ReSolve::index_type k = 0; k < H_nnz; ++k) {
    const int triplet = H_csr_to_triplet_host_[k];
    H_values[k] = triplet >= 0 ? Hess_values[triplet] : 0.0;
  }

  for(index_type i = 0; i < Hx_size; ++i) {
    assert(H_diag_to_csr_host_[i] >= 0);
    H_values[H_diag_to_csr_host_[i]] += Hx_values[i];
  }

  for(ReSolve::index_type k = 0; k < J_nnz; ++k) {
    J_values[k] = Jac_c_values[J_csr_to_triplet_host_[k]];
  }

  for(ReSolve::index_type k = 0; k < J_d_nnz; ++k) {
    J_d_values[k] = Jac_d_values[J_d_csr_to_triplet_host_[k]];
  }

  for(ReSolve::index_type k = 0; k < D_s_nnz; ++k) {
    D_s_values[k] = Hd_values[k];
  }

  if(H_->setUpdated(ReSolve::memory::HOST) != 0 || D_s_->setUpdated(ReSolve::memory::HOST) != 0 ||
     J_->setUpdated(ReSolve::memory::HOST) != 0 || J_d_->setUpdated(ReSolve::memory::HOST) != 0) {
    nlp_->log->printf(hovError, "Failed to mark ReSolve HyKKT matrix blocks as updated.\n");
    return false;
  }

#ifdef HIOP_USE_GPU
  if(solve_on_device()) {
    if(H_->syncData(ReSolve::memory::DEVICE) != 0 || D_s_->syncData(ReSolve::memory::DEVICE) != 0 ||
       J_->syncData(ReSolve::memory::DEVICE) != 0 || J_d_->syncData(ReSolve::memory::DEVICE) != 0) {
      nlp_->log->printf(hovError, "Failed to copy ReSolve HyKKT matrix blocks to the device.\n");
      return false;
    }
  }
#endif

  return true;
}

}  // namespace hiop
