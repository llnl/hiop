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
 *
 */

#include "hiopLinSolverSparseReSolve.hpp"
#include <IterativeRefinement.hpp>
#include <RefactorizationSolver.hpp>
#include <MatrixCsr.hpp>

#include "hiop_blasdefs.hpp"
#include "KrylovSolverKernels.h"

#include "cusparse_v2.h"
#include <sstream>
#include <string>

#define checkCudaErrors(val) hiopCheckCudaError((val), __FILE__, __LINE__)

/**
 * @brief Map elements of one array to the other
 *
 *    for(int k = 0; k < nnz_; k++) {
 *      vals[k] = M_->M()[index_convert_CSR2Triplet_host_[k]];
 *    }
 *
 */
template<typename T, typename I>
__global__ void mapArraysKernel(T* dst, const T* src, const I* mapidx, I n)
{
  I tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < n) {
    dst[tid] = src[mapidx[tid]];
  }
}

/**
 * @brief Map elements of one array to the other
 *
 *  for(int i = 0; i < n_; i++) {
 *    if(index_convert_extra_Diag2CSR_host_[i] != -1)
 *      vals[index_convert_extra_Diag2CSR_host_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
 *  }
 *
 */
template<typename T, typename I>
__global__ void addToArrayKernel(T* dst, const T* src, const I* mapidx, I n, I nnz)
{
  I tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < n) {
    if(mapidx[tid] != -1) dst[mapidx[tid]] += src[nnz - n + tid];
  }
}

namespace hiop
{
hiopLinSolverSymSparseReSolve::hiopLinSolverSymSparseReSolve(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      index_convert_CSR2Triplet_host_{nullptr},
      index_convert_extra_Diag2CSR_host_{nullptr},
      index_convert_CSR2Triplet_device_{nullptr},
      index_convert_extra_Diag2CSR_device_{nullptr},
      n_{n},
      nnz_{0},
      factorizationSetupSucc_{0},
      is_first_call_{true}
{
  // Create ReSolve solver and allocate rhs temporary storage
  solver_ = new ReSolve::RefactorizationSolver(n);

  // If memory space is device, allocate host mirror for HiOp's KKT matrix in triplet format
  if(nlp_->options->GetString("mem_space") == "device") {
    M_host_ = LinearAlgebraFactory::create_matrix_sparse("default", n, n, nnz);
  }

  // Set verbosity of ReSolve based on HiOp verbosity
  if(nlp_->options->GetInteger("verbosity_level") >= 3) {
    solver_->set_silent_output(false);
  }

  // Select matrix ordering
  int ordering = 1;
  std::string ord = nlp_->options->GetString("linear_solver_sparse_ordering");
  if(ord == "amd_ssparse") {
    ordering = 0;
  } else if(ord == "colamd_ssparse") {
    ordering = 1;
  } else {
    nlp_->log->printf(hovWarning, "Ordering %s not compatible with cuSOLVER LU, using default ...\n", ord.c_str());
    ordering = 1;
  }
  solver_->ordering() = ordering;
  nlp_->log->printf(hovSummary, "Ordering: %d\n", solver_->ordering());

  // Select factorization
  std::string fact;
  fact = nlp_->options->GetString("resolve_factorization");
  if(fact != "klu") {
    nlp_->log->printf(hovWarning, "Factorization %s not compatible with cuSOLVER LU, using default ...\n", fact.c_str());
    fact = "klu";
  }
  solver_->fact() = fact;
  nlp_->log->printf(hovSummary, "Factorization: %s\n", solver_->fact().c_str());

  // Select refactorization
  std::string refact;
  refact = nlp_->options->GetString("resolve_refactorization");
  if(refact != "glu" && refact != "rf") {
    nlp_->log->printf(hovWarning, "Refactorization %s not compatible with cuSOLVER LU, using default ...\n", refact.c_str());
    refact = "glu";
  }
  solver_->refact() = refact;
  nlp_->log->printf(hovSummary, "Refactorization: %s\n", solver_->refact().c_str());

  // by default, dont use iterative refinement
  std::string use_ir;
  int maxit_test = nlp_->options->GetInteger("ir_inner_maxit");

  if((maxit_test < 0) || (maxit_test > 1000)) {
    nlp_->log->printf(hovWarning,
                      "Wrong maxit value: %d. Use int maxit value between 0 and 1000. Setting default (50)  ...\n",
                      maxit_test);
    maxit_test = 50;
  }
  use_ir = "no";
  if(maxit_test > 0) {
    use_ir = "yes";
    solver_->enable_iterative_refinement();
    solver_->ir()->maxit() = maxit_test;
  }
  if(use_ir == "yes") {
    if((refact == "rf")) {
      solver_->ir()->restart() = nlp_->options->GetInteger("ir_inner_restart");

      if((solver_->ir()->restart() < 0) || (solver_->ir()->restart() > 100)) {
        nlp_->log->printf(hovWarning,
                          "Wrong restart value: %d. Use int restart value between 1 and 100. Setting default (20)  ...\n",
                          solver_->ir()->restart());
        solver_->ir()->restart() = 20;
      }

      solver_->ir()->tol() = nlp_->options->GetNumeric("ir_inner_tol");
      if((solver_->ir()->tol() < 0) || (solver_->ir()->tol() > 1)) {
        nlp_->log->printf(hovWarning,
                          "Wrong tol value: %e. Use double tol value between 0 and 1. Setting default (1e-12)  ...\n",
                          solver_->ir()->tol());
        solver_->ir()->tol() = 1e-12;
      }
      solver_->ir()->orth_option() = nlp_->options->GetString("ir_inner_gs_scheme");

      /* 0) "Standard" GMRES and FGMRES (Saad and Schultz, 1986, Saad, 1992) use Modified Gram-Schmidt ("mgs") to keep the
       * Krylov vectors orthogonal. Modified Gram-Schmidt requires k synchronization (due to inner products) in iteration k
       * and this becomes a scaling bottleneck for GPU-accelerated implementation and it becomes even more pronouced for
       * MPI+GPU-acceleration. Modified Gram-Schidt can be replaced by a different scheme.
       *
       * 1) One can use Classical Gram-Schmidt ("cgs") which is numerically unstable or reorthogonalized Classical
       * Gram-Schmidt ("cgs2"), which is numerically stable and requires 3 synchrnozations and each iteration.
       * Reorthogonalized Classical Gram-Schmidt makes two passes of Classical Gram-Schmidt. And two passes are enough to get
       * vectors orthogonal to machine precision (Bjorck 1967).
       *
       * 2) An alternative is a low-sych version (Swirydowicz and Thomas, 2020), which reformulates Modified Gram-Schmidt to
       * be a (very small) triangular solve. It requires extra storage for the matrix used in triangular solve (kxk at
       * iteration k), but only two sycnhronizations are needed per iteration. The inner producats are performed in bulk,
       * which quarantees better GPU utilization. The second synchronization comes from normalizing the vector and can be
       * eliminated if the norm is postponed to the next iteration, but also makes code more complicated. This is why we use
       * two-synch method ("mgs_two_synch")
       *
       * 3) A recently submitted paper by Stephen Thomas (Thomas 202*) takes the triangular solve idea further and uses a
       * different approximation for the inverse of a triangular matrix. It requires two (very small) triangular solves and
       * two sychroniztions (if the norm is NOT delayed). It also guarantees that the vectors are orthogonal to the machine
       * epsilon, as in cgs2. Since Stephen's paper is named "post modern GMRES", we call this Gram-Schmidt scheme "mgs_pm".
       */
      if(solver_->ir()->orth_option() != "mgs" && solver_->ir()->orth_option() != "cgs2" &&
         solver_->ir()->orth_option() != "mgs_two_synch" && solver_->ir()->orth_option() != "mgs_pm") {
        nlp_->log->printf(
            hovWarning,
            "mgs option : %s is wrong. Use 'mgs', 'cgs2', 'mgs_two_synch' or 'mgs_pm'. Switching to default (mgs) ...\n",
            use_ir.c_str());
        solver_->ir()->orth_option() = "mgs";
      }

      solver_->ir()->conv_cond() = nlp_->options->GetInteger("ir_inner_conv_cond");

      if((solver_->ir()->conv_cond() < 0) || (solver_->ir()->conv_cond() > 2)) {
        nlp_->log->printf(hovWarning,
                          "Wrong IR convergence condition: %d. Use int value: 0, 1 or 2. Setting default (0)  ...\n",
                          solver_->ir()->conv_cond());
        solver_->ir()->conv_cond() = 0;
      }

    } else {
      nlp_->log->printf(hovWarning, "Currently, inner iterative refinement works ONLY with cuSolverRf ... \n");
      use_ir = "no";
    }
  }
  solver_->use_ir() = use_ir;
  nlp_->log->printf(hovSummary, "Use IR: %s\n", solver_->use_ir().c_str());
}  // constructor

hiopLinSolverSymSparseReSolve::~hiopLinSolverSymSparseReSolve()
{
  delete solver_;

  // If memory space is device, delete allocated host mirrors
  if(nlp_->options->GetString("mem_space") == "device") {
    delete M_host_;
  }

  // Delete CSR <--> triplet mappings
  delete[] index_convert_CSR2Triplet_host_;
  delete[] index_convert_extra_Diag2CSR_host_;
  checkCudaErrors(cudaFree(index_convert_CSR2Triplet_device_));
  checkCudaErrors(cudaFree(index_convert_extra_Diag2CSR_device_));
}

int hiopLinSolverSymSparseReSolve::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(is_first_call_) {
    firstCall();
  } else {
    update_matrix_values();
  }

  if(factorizationSetupSucc_ == 0) {
    int retval = solver_->factorize();
    if(retval == -1) {
      nlp_->log->printf(hovWarning, "Numeric klu factorization failed. Regularizing ...\n");
      // This is not a catastrophic failure
      // The matrix is singular so return -1 to regularaize!
      return -1;
    } else {  // Numeric was succesfull so now can set up
      solver_->setup_refactorization();
      factorizationSetupSucc_ = 1;
      nlp_->log->printf(hovScalars, "Numeric klu factorization succesful! \n");
    }
  } else {  // factorizationSetupSucc_ == 1
    solver_->refactorize();
  }

  nlp_->runStats.linsolv.tmFactTime.stop();
  return 0;
}

bool hiopLinSolverSymSparseReSolve::solve(hiopVector& x)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(x.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  // Set IR tolerance
  double ir_tol = nlp_->options->GetNumeric("ir_inner_tol");

  std::string mem_space = nlp_->options->GetString("mem_space");
  double* dx = x.local_data();

  bool retval = solver_->triangular_solve(dx, ir_tol, mem_space);
  if(!retval) {
    nlp_->log->printf(hovError,  // catastrophic failure
                      "ReSolve triangular solver failed\n");
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return true;
}

void hiopLinSolverSymSparseReSolve::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  // If the matrix is on device, copy it to the host mirror
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "device") {
    checkCudaErrors(cudaMemcpy(M_host_->M(), M_->M(), sizeof(double) * M_->numberOfNonzeros(), cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(M_host_->i_row(), M_->i_row(), sizeof(index_type) * M_->numberOfNonzeros(), cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(M_host_->j_col(), M_->j_col(), sizeof(index_type) * M_->numberOfNonzeros(), cudaMemcpyDeviceToHost));
  }

  // Transfer triplet to CSR form

  // Allocate row pointers and compute number of nonzeros.
  solver_->mat_A_csr()->allocate_size(n_);
  compute_nnz();
  solver_->set_nnz(nnz_);

  // Allocate column indices and matrix values
  solver_->mat_A_csr()->allocate_nnz(nnz_);

  // Set column indices and matrix values.
  set_csr_indices_values();

  // Copy matrix to device
  solver_->mat_A_csr()->update_from_host_mirror();

  if(solver_->use_ir() == "yes") {
    solver_->setup_iterative_refinement_matrix(n_, nnz_);
  }
  /*
   * initialize matrix factorization
   */
  if(solver_->setup_factorization() < 0) {
    nlp_->log->printf(hovError,  // catastrophic failure
                      "Symbolic factorization failed!\n");
    return;
  };
  is_first_call_ = false;
}

/// nnz_ is number of nonzeros in CSR matrix
/// M_->numberOfNonzeros() is number of zeros in symmetric triplet matrix
void hiopLinSolverSymSparseReSolve::update_matrix_values()
{
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "device") {
    double* csr_vals = solver_->mat_A_csr()->get_vals();
    double* coo_vals = M_->M();
    int coo_nnz = M_->numberOfNonzeros();

    const int blocksize = 512;
    int gridsize = (nnz_ + blocksize - 1) / blocksize;
    mapArraysKernel<double, int><<<gridsize, blocksize>>>(csr_vals, coo_vals, index_convert_CSR2Triplet_device_, nnz_);

    gridsize = (n_ + blocksize - 1) / blocksize;
    addToArrayKernel<double, int>
        <<<gridsize, blocksize>>>(csr_vals, coo_vals, index_convert_extra_Diag2CSR_device_, n_, coo_nnz);

    // If factorization was not successful, we need a copy of values on the host
    if(factorizationSetupSucc_ == 0)
      checkCudaErrors(cudaMemcpy(solver_->mat_A_csr()->get_vals_host(),
                                 solver_->mat_A_csr()->get_vals(),
                                 sizeof(double) * nnz_,
                                 cudaMemcpyDeviceToHost));

  } else {
    // KKT matrix is on the host
    double* vals = solver_->mat_A_csr()->get_vals_host();
    // update matrix
    for(int k = 0; k < nnz_; k++) {
      vals[k] = M_->M()[index_convert_CSR2Triplet_host_[k]];
    }
    for(int i = 0; i < n_; i++) {
      if(index_convert_extra_Diag2CSR_host_[i] != -1)
        vals[index_convert_extra_Diag2CSR_host_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
    checkCudaErrors(cudaMemcpy(solver_->mat_A_csr()->get_vals(),
                               solver_->mat_A_csr()->get_vals_host(),
                               sizeof(double) * nnz_,
                               cudaMemcpyHostToDevice));
  }
}

/// @pre Data is either on the host or the host mirror is synced with the device
void hiopLinSolverSymSparseReSolve::compute_nnz()
{
  //
  // compute nnz in each row
  //
  int* row_ptr = solver_->mat_A_csr()->get_irows_host();

  // If the data is on device, fetch it from the host mirror
  hiopMatrixSparse* M_host = nullptr;
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "host" || mem_space == "default") {
    M_host = M_;
  } else if(mem_space == "device") {
    M_host = M_host_;
  } else {
    nlp_->log->printf(hovError, "Memory space %s incompatible with ReSolve.\n", mem_space.c_str());
  }

  // off-diagonal part
  row_ptr[0] = 0;
  for(int k = 0; k < M_host->numberOfNonzeros() - n_; k++) {
    if(M_host->i_row()[k] != M_host->j_col()[k]) {
      row_ptr[M_host->i_row()[k] + 1]++;
      row_ptr[M_host->j_col()[k] + 1]++;
      nnz_ += 2;
    }
  }
  // diagonal part
  for(int i = 0; i < n_; i++) {
    row_ptr[i + 1]++;
    nnz_ += 1;
  }
  // get correct row ptr index
  for(int i = 1; i < n_ + 1; i++) {
    row_ptr[i] += row_ptr[i - 1];
  }
  assert(nnz_ == row_ptr[n_]);
}

/// @pre Data is either on the host or the host mirror is synced with the device
void hiopLinSolverSymSparseReSolve::set_csr_indices_values()
{
  // If the data is on device, fetch it from the host mirror
  hiopMatrixSparse* M_host = nullptr;
  std::string mem_space = nlp_->options->GetString("mem_space");
  if(mem_space == "host" || mem_space == "default") {
    M_host = M_;
  } else if(mem_space == "device") {
    M_host = M_host_;
  } else {
    nlp_->log->printf(hovError, "Memory space %s incompatible with ReSolve.\n", mem_space.c_str());
  }

  //
  // set correct col index and value
  //
  const int* row_ptr = solver_->mat_A_csr()->get_irows_host();
  int* col_idx = solver_->mat_A_csr()->get_jcols_host();
  double* vals = solver_->mat_A_csr()->get_vals_host();

  index_convert_CSR2Triplet_host_ = new int[nnz_];
  index_convert_extra_Diag2CSR_host_ = new int[n_];
  checkCudaErrors(cudaMalloc(&index_convert_CSR2Triplet_device_, nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&index_convert_extra_Diag2CSR_device_, n_ * sizeof(int)));

  int* nnz_each_row_tmp = new int[n_]{0};
  int total_nnz_tmp{0}, nnz_tmp{0}, rowID_tmp, colID_tmp;

  for(int k = 0; k < n_; k++) {
    index_convert_extra_Diag2CSR_host_[k] = -1;
  }

  for(int k = 0; k < M_host->numberOfNonzeros() - n_; k++) {
    rowID_tmp = M_host->i_row()[k];
    colID_tmp = M_host->j_col()[k];
    if(rowID_tmp == colID_tmp) {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];
      col_idx[nnz_tmp] = colID_tmp;
      vals[nnz_tmp] = M_host->M()[k];
      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      vals[nnz_tmp] += M_host->M()[M_host->numberOfNonzeros() - n_ + rowID_tmp];
      index_convert_extra_Diag2CSR_host_[rowID_tmp] = nnz_tmp;

      nnz_each_row_tmp[rowID_tmp]++;
      total_nnz_tmp++;
    } else {
      nnz_tmp = nnz_each_row_tmp[rowID_tmp] + row_ptr[rowID_tmp];
      col_idx[nnz_tmp] = colID_tmp;
      vals[nnz_tmp] = M_host->M()[k];
      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_tmp = nnz_each_row_tmp[colID_tmp] + row_ptr[colID_tmp];
      col_idx[nnz_tmp] = rowID_tmp;
      vals[nnz_tmp] = M_host->M()[k];
      index_convert_CSR2Triplet_host_[nnz_tmp] = k;

      nnz_each_row_tmp[rowID_tmp]++;
      nnz_each_row_tmp[colID_tmp]++;
      total_nnz_tmp += 2;
    }
  }
  // correct the missing dia_gonal term
  for(int i = 0; i < n_; i++) {
    if(nnz_each_row_tmp[i] != row_ptr[i + 1] - row_ptr[i]) {
      assert(nnz_each_row_tmp[i] == row_ptr[i + 1] - row_ptr[i] - 1);
      nnz_tmp = nnz_each_row_tmp[i] + row_ptr[i];
      col_idx[nnz_tmp] = i;
      vals[nnz_tmp] = M_host->M()[M_host->numberOfNonzeros() - n_ + i];
      index_convert_CSR2Triplet_host_[nnz_tmp] = M_host->numberOfNonzeros() - n_ + i;
      total_nnz_tmp += 1;

      std::vector<int> ind_temp(row_ptr[i + 1] - row_ptr[i]);
      std::iota(ind_temp.begin(), ind_temp.end(), 0);
      std::sort(ind_temp.begin(), ind_temp.end(), [&](int a, int b) {
        return col_idx[a + row_ptr[i]] < col_idx[b + row_ptr[i]];
      });

      reorder(vals + row_ptr[i], ind_temp, row_ptr[i + 1] - row_ptr[i]);
      reorder(index_convert_CSR2Triplet_host_ + row_ptr[i], ind_temp, row_ptr[i + 1] - row_ptr[i]);
      std::sort(col_idx + row_ptr[i], col_idx + row_ptr[i + 1]);
    }
  }
  checkCudaErrors(cudaMemcpy(index_convert_CSR2Triplet_device_,
                             index_convert_CSR2Triplet_host_,
                             nnz_ * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(index_convert_extra_Diag2CSR_device_,
                             index_convert_extra_Diag2CSR_host_,
                             n_ * sizeof(int),
                             cudaMemcpyHostToDevice));
  delete[] nnz_each_row_tmp;
}

// Error checking utility for CUDA
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void hiopLinSolverSymSparseReSolve::hiopCheckCudaError(T result, const char* const file, int const line)
{
  if(result) {
    nlp_->log->printf(hovError, "CUDA error at %s:%d, error# %d\n", file, line, result);
    assert(false);
  }
}

}  // namespace hiop
