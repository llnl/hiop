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
 * @file RefactorizationSolver.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#include "MatrixCsr.hpp"
#include "IterativeRefinement.hpp"
#include "RefactorizationSolver.hpp"

#include "klu.h"
#include "cusparse_v2.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#define checkCudaErrors(val) resolveCheckCudaError((val), __FILE__, __LINE__)

namespace ReSolve
{

RefactorizationSolver::RefactorizationSolver(int n)
    : n_(n)
{
  mat_A_csr_ = new MatrixCsr();

  // handles
  cusparseCreate(&handle_);
  cusolverSpCreate(&handle_cusolver_);
  cublasCreate(&handle_cublas_);

  // descriptors
  cusparseCreateMatDescr(&descr_A_);
  cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

  // Allocate host mirror for the solution vector
  hostx_ = new double[n_];

  // Allocate solution and rhs vectors
  checkCudaErrors(cudaMalloc(&devx_, n_ * sizeof(double)));
  checkCudaErrors(cudaMalloc(&devr_, n_ * sizeof(double)));
}

RefactorizationSolver::~RefactorizationSolver()
{
  if(iterative_refinement_enabled_) delete ir_;
  delete mat_A_csr_;

  // Delete workspaces and handles
  cudaFree(d_work_);
  cusparseDestroy(handle_);
  cusolverSpDestroy(handle_cusolver_);
  cublasDestroy(handle_cublas_);
  cusparseDestroyMatDescr(descr_A_);

  // Delete host mirror for the solution vector
  delete[] hostx_;

  // Delete residual and solution vectors
  cudaFree(devr_);
  cudaFree(devx_);

  // Delete matrix descriptor used in cuSolverGLU setup
  if(cusolver_glu_enabled_) {
    cusparseDestroyMatDescr(descr_M_);
    cusolverSpDestroyGluInfo(info_M_);
  }

  if(cusolver_rf_enabled_) {
    cudaFree(d_P_);
    cudaFree(d_Q_);
    cudaFree(d_T_);
  }

  klu_free_symbolic(&Symbolic_, &Common_);
  klu_free_numeric(&Numeric_, &Common_);
  delete[] mia_;
  delete[] mja_;
}

void RefactorizationSolver::enable_iterative_refinement()
{
  ir_ = new IterativeRefinement();
  if(ir_ != nullptr) iterative_refinement_enabled_ = true;
}

// TODO: Refactor to only pass mat_A_csr_ to setup_system_matrix; n and nnz can be read from mat_A_csr_
void RefactorizationSolver::setup_iterative_refinement_matrix(int n, int nnz)
{
  ir_->setup_system_matrix(n, nnz, mat_A_csr_->get_irows(), mat_A_csr_->get_jcols(), mat_A_csr_->get_vals());
}

// TODO: Can this function be merged with setup_iterative_refinement_matrix ?
void RefactorizationSolver::configure_iterative_refinement(cusparseHandle_t cusparse_handle,
                                                           cublasHandle_t cublas_handle,
                                                           cusolverRfHandle_t cusolverrf_handle,
                                                           int n,
                                                           double* d_T,
                                                           int* d_P,
                                                           int* d_Q,
                                                           double* devx,
                                                           double* devr)
{
  ir_->setup(cusparse_handle, cublas_handle, cusolverrf_handle, n, d_T, d_P, d_Q, devx, devr);
}

int RefactorizationSolver::setup_factorization()
{
  int* row_ptr = mat_A_csr_->get_irows_host();
  int* col_idx = mat_A_csr_->get_jcols_host();

  if(fact_ == "klu") {
    /* initialize KLU setup parameters, dont factorize yet */
    initializeKLU();

    /*perform KLU but only the symbolic analysis (important)   */
    klu_free_symbolic(&Symbolic_, &Common_);
    klu_free_numeric(&Numeric_, &Common_);
    Symbolic_ = klu_analyze(n_, row_ptr, col_idx, &Common_);

    if(Symbolic_ == nullptr) {
      return -1;
    }
  } else {  // for future
    assert(0 && "Only KLU is available for the first factorization.\n");
  }
  return 0;
}

int RefactorizationSolver::factorize()
{
  Numeric_ = klu_factor(mat_A_csr_->get_irows_host(),
                        mat_A_csr_->get_jcols_host(),
                        mat_A_csr_->get_vals_host(),
                        Symbolic_,
                        &Common_);
  return (Numeric_ == nullptr) ? -1 : 0;
}

void RefactorizationSolver::setup_refactorization()
{
  if(refact_ == "glu") {
    initializeCusolverGLU();
    refactorizationSetupCusolverGLU();
  } else if(refact_ == "rf") {
    initializeCusolverRf();
    refactorizationSetupCusolverRf();
    if(use_ir_ == "yes") {
      configure_iterative_refinement(handle_, handle_cublas_, handle_rf_, n_, d_T_, d_P_, d_Q_, devx_, devr_);
    }
  } else {  // for future -
    assert(0 && "Only glu and rf refactorizations available.\n");
  }
}

int RefactorizationSolver::refactorize()
{
  if(refact_ == "glu") {
    sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     mat_A_csr_->get_vals(),
                                     mat_A_csr_->get_irows(),
                                     mat_A_csr_->get_jcols(),
                                     info_M_);
    sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
  } else {
    if(refact_ == "rf") {
      sp_status_ = cusolverRfResetValues(n_,
                                         nnz_,
                                         mat_A_csr_->get_irows(),
                                         mat_A_csr_->get_jcols(),
                                         mat_A_csr_->get_vals(),
                                         d_P_,
                                         d_Q_,
                                         handle_rf_);
      cudaDeviceSynchronize();
      sp_status_ = cusolverRfRefactor(handle_rf_);
    }
  }
  return 0;
}

bool RefactorizationSolver::triangular_solve(double* dx, double tol, std::string memspace)
{
  if(refact_ == "glu") {
    double* devx = nullptr;
    if(memspace == "device") {
      checkCudaErrors(cudaMemcpy(devr_, dx, sizeof(double) * n_, cudaMemcpyDeviceToDevice));
      devx = dx;
    } else {
      checkCudaErrors(cudaMemcpy(devr_, dx, sizeof(double) * n_, cudaMemcpyHostToDevice));
      devx = devx_;
    }
    sp_status_ = cusolverSpDgluSolve(handle_cusolver_,
                                     n_,
                                     /* A is original matrix */
                                     nnz_,
                                     descr_A_,
                                     mat_A_csr_->get_vals(),
                                     mat_A_csr_->get_irows(),
                                     mat_A_csr_->get_jcols(),
                                     devr_, /* right hand side */
                                     devx,  /* left hand side, local pointer */
                                     &ite_refine_succ_,
                                     &r_nrminf_,
                                     info_M_,
                                     d_work_);
    if(sp_status_ != 0 && !silent_output_) {
      std::cout << "GLU solve failed with status: " << sp_status_ << "\n";
      return false;
    }
    if(memspace == "device") {
      // do nothing
    } else {
      checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    }
    return true;
  }

  if(refact_ == "rf") {
    // First solve is performed on CPU
    if(is_first_solve_) {
      double* hostx = nullptr;
      if(memspace == "device") {
        checkCudaErrors(cudaMemcpy(hostx_, dx, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        hostx = hostx_;
      } else {
        hostx = dx;
      }
      int ok = klu_solve(Symbolic_, Numeric_, n_, 1, hostx, &Common_);  // replace dx with hostx
      klu_free_numeric(&Numeric_, &Common_);
      klu_free_symbolic(&Symbolic_, &Common_);
      is_first_solve_ = false;
      if(memspace == "device") {
        checkCudaErrors(cudaMemcpy(dx, hostx, sizeof(double) * n_, cudaMemcpyHostToDevice));
      } else {
        // do nothing
      }
      return true;
    }

    double* devx = nullptr;
    if(memspace == "device") {
      devx = dx;
      checkCudaErrors(cudaMemcpy(devr_, dx, sizeof(double) * n_, cudaMemcpyDeviceToDevice));
    } else {
      checkCudaErrors(cudaMemcpy(devx_, dx, sizeof(double) * n_, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(devr_, devx_, sizeof(double) * n_, cudaMemcpyDeviceToDevice));
      devx = devx_;
    }

    // Each next solve is performed on GPU
    sp_status_ = cusolverRfSolve(handle_rf_,
                                 d_P_,
                                 d_Q_,
                                 1,
                                 d_T_,
                                 n_,
                                 devx,  // replace devx_ with local pointer devx
                                 n_);
    if(sp_status_ != 0) {
      if(!silent_output_) std::cout << "Rf solve failed with status: " << sp_status_ << "\n";
      return false;
    }

    if(use_ir_ == "yes") {
      // Set tolerance based on barrier parameter mu
      ir_->set_tol(tol);

      ir_->fgmres(devx, devr_);  // replace devx_ with local pointer devx
      if(!silent_output_ && (ir_->getFinalResidalNorm() > tol * ir_->getBNorm())) {
        std::cout << "[Warning] Iterative refinement did not converge!\n";
        std::cout << "\t Iterative refinement tolerance " << tol << "\n";
        std::cout << "\t Relative solution error        " << ir_->getFinalResidalNorm() / ir_->getBNorm() << "\n";
        std::cout << "\t fgmres: init residual norm: " << ir_->getInitialResidalNorm() << "\n"
                  << "\t final residual norm:        " << ir_->getFinalResidalNorm() << "\n"
                  << "\t number of iterations:       " << ir_->getFinalNumberOfIterations() << "\n";
      }
    }
    if(memspace == "device") {
      // do nothing
    } else {
      checkCudaErrors(cudaMemcpy(dx, devx_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    }
    return true;
  }

  if(!silent_output_) {
    std::cout << "Unknown refactorization " << refact_ << ", exiting\n";
  }
  return false;
}

// helper private function needed for format conversion
int RefactorizationSolver::createM(const int n,
                                   const int /* nnzL */,
                                   const int* Lp,
                                   const int* Li,
                                   const int /* nnzU */,
                                   const int* Up,
                                   const int* Ui)
{
  int row;
  for(int i = 0; i < n; ++i) {
    // go through EACH COLUMN OF L first
    for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
      row = Li[j];
      // BUT dont count diagonal twice, important
      if(row != i) {
        mia_[row + 1]++;
      }
    }
    // then each column of U
    for(int j = Up[i]; j < Up[i + 1]; ++j) {
      row = Ui[j];
      mia_[row + 1]++;
    }
  }
  // then organize mia_;
  mia_[0] = 0;
  for(int i = 1; i < n + 1; i++) {
    mia_[i] += mia_[i - 1];
  }

  std::vector<int> Mshifts(n, 0);
  for(int i = 0; i < n; ++i) {
    // go through EACH COLUMN OF L first
    for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
      row = Li[j];
      if(row != i) {
        // place (row, i) where it belongs!
        mja_[mia_[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    // each column of U next
    for(int j = Up[i]; j < Up[i + 1]; ++j) {
      row = Ui[j];
      mja_[mia_[row] + Mshifts[row]] = i;
      Mshifts[row]++;
    }
  }
  return 0;
}

int RefactorizationSolver::initializeKLU()
{
  klu_defaults(&Common_);

  // TODO: consider making this a part of setup options so that user can
  // set up these values. For now, we keep them hard-wired.
  Common_.btf = 0;
  Common_.ordering = ordering_;  // COLAMD=1; AMD=0
  Common_.tol = 0.1;
  Common_.scale = -1;
  Common_.halt_if_singular = 1;

  return 0;
}

int RefactorizationSolver::initializeCusolverGLU()
{
  cusparseCreateMatDescr(&descr_M_);
  cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);

  // info (data structure where factorization is stored)
  // this is done in the constructor - however, this function might be called more than once
  cusolverSpDestroyGluInfo(info_M_);
  cusolverSpCreateGluInfo(&info_M_);

  cusolver_glu_enabled_ = true;
  return 0;
}

int RefactorizationSolver::initializeCusolverRf()
{
  cusolverRfCreate(&handle_rf_);

  checkCudaErrors(cusolverRfSetAlgs(handle_rf_, CUSOLVERRF_FACTORIZATION_ALG2, CUSOLVERRF_TRIANGULAR_SOLVE_ALG2));

  checkCudaErrors(cusolverRfSetMatrixFormat(handle_rf_, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_STORED_L));

  cusolverRfSetResetValuesFastMode(handle_rf_, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);

  const double boost = 1e-12;
  const double zero = 1e-14;

  cusolverRfSetNumericProperties(handle_rf_, zero, boost);

  cusolver_rf_enabled_ = true;
  return 0;
}

// call if both the matrix and the nnz structure changed or if convergence is
// poor while using refactorization.
int RefactorizationSolver::refactorizationSetupCusolverGLU()
{
  // for now this ONLY WORKS if proceeded by KLU. Might be worth decoupling
  // later

  // get sizes
  const int nnzL = Numeric_->lnz;
  const int nnzU = Numeric_->unz;

  const int nnzM = (nnzL + nnzU - n_);

  /* parse the factorization */

  mia_ = new int[n_ + 1]{0};
  mja_ = new int[nnzM]{0};
  int* Lp = new int[n_ + 1];
  int* Li = new int[nnzL];
  // we can't use nullptr instead od Lx and Ux because it causes SEG FAULT. It
  // seems like a waste of memory though.

  double* Lx = new double[nnzL];
  int* Up = new int[n_ + 1];
  int* Ui = new int[nnzU];

  double* Ux = new double[nnzU];

  int ok = klu_extract(Numeric_,
                       Symbolic_,
                       Lp,
                       Li,
                       Lx,
                       Up,
                       Ui,
                       Ux,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       &Common_);
  createM(n_, nnzL, Lp, Li, nnzU, Up, Ui);

  delete[] Lp;
  delete[] Li;
  delete[] Lx;
  delete[] Up;
  delete[] Ui;
  delete[] Ux;

  /* setup GLU */
  sp_status_ = cusolverSpDgluSetup(handle_cusolver_,
                                   n_,
                                   nnz_,
                                   descr_A_,
                                   mat_A_csr_->get_irows_host(),  // kRowPtr_,
                                   mat_A_csr_->get_jcols_host(),  // jCol_,
                                   Numeric_->Pnum,                /* base-0 */
                                   Symbolic_->Q,                  /* base-0 */
                                   nnzM,                          /* nnzM */
                                   descr_M_,
                                   mia_,
                                   mja_,
                                   info_M_);

  sp_status_ = cusolverSpDgluBufferSize(handle_cusolver_, info_M_, &size_M_);
  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

  buffer_size_ = size_M_;
  checkCudaErrors(cudaMalloc((void**)&d_work_, buffer_size_));

  sp_status_ = cusolverSpDgluAnalysis(handle_cusolver_, info_M_, d_work_);
  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);

  // reset and refactor so factors are ON THE GPU

  sp_status_ = cusolverSpDgluReset(handle_cusolver_,
                                   n_,
                                   /* A is original matrix */
                                   nnz_,
                                   descr_A_,
                                   mat_A_csr_->get_vals(),
                                   mat_A_csr_->get_irows(),
                                   mat_A_csr_->get_jcols(),
                                   info_M_);

  assert(CUSOLVER_STATUS_SUCCESS == sp_status_);
  sp_status_ = cusolverSpDgluFactor(handle_cusolver_, info_M_, d_work_);
  return 0;
}

int RefactorizationSolver::refactorizationSetupCusolverRf()
{
  // for now this ONLY WORKS if preceeded by KLU. Might be worth decoupling
  // later
  const int nnzL = Numeric_->lnz;
  const int nnzU = Numeric_->unz;

  checkCudaErrors(cudaMalloc(&d_P_, (n_) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Q_, (n_) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_T_, (n_) * sizeof(double)));

  checkCudaErrors(cudaMemcpy(d_P_, Numeric_->Pnum, sizeof(int) * (n_), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Q_, Symbolic_->Q, sizeof(int) * (n_), cudaMemcpyHostToDevice));

  int* Lp = new int[n_ + 1];
  int* Li = new int[nnzL];
  double* Lx = new double[nnzL];
  int* Up = new int[n_ + 1];
  int* Ui = new int[nnzU];
  double* Ux = new double[nnzU];

  int ok = klu_extract(Numeric_,
                       Symbolic_,
                       Lp,
                       Li,
                       Lx,
                       Up,
                       Ui,
                       Ux,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       &Common_);

  /* CSC */
  int* d_Lp;
  int* d_Li;
  int* d_Up;
  int* d_Ui;
  double* d_Lx;
  double* d_Ux;
  /* CSR */
  int* d_Lp_csr;
  int* d_Li_csr;
  int* d_Up_csr;
  int* d_Ui_csr;
  double* d_Lx_csr;
  double* d_Ux_csr;

  /* allocate CSC */
  checkCudaErrors(cudaMalloc(&d_Lp, (n_ + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Li, nnzL * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Lx, nnzL * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Up, (n_ + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Ui, nnzU * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Ux, nnzU * sizeof(double)));

  /* allocate CSR */
  checkCudaErrors(cudaMalloc(&d_Lp_csr, (n_ + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Li_csr, nnzL * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Lx_csr, nnzL * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Up_csr, (n_ + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Ui_csr, nnzU * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_Ux_csr, nnzU * sizeof(double)));

  /* copy CSC to the GPU */
  checkCudaErrors(cudaMemcpy(d_Lp, Lp, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Li, Li, sizeof(int) * (nnzL), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Lx, Lx, sizeof(double) * (nnzL), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_Up, Up, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Ui, Ui, sizeof(int) * (nnzU), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Ux, Ux, sizeof(double) * (nnzU), cudaMemcpyHostToDevice));

  /* we dont need these any more */
  delete[] Lp;
  delete[] Li;
  delete[] Lx;
  delete[] Up;
  delete[] Ui;
  delete[] Ux;

  /* now CSC to CSR using the new cuda 11 awkward way */
  size_t bufferSizeL;
  size_t bufferSizeU;

  cusparseStatus_t csp = cusparseCsr2cscEx2_bufferSize(handle_,
                                                       n_,
                                                       n_,
                                                       nnzL,
                                                       d_Lx,
                                                       d_Lp,
                                                       d_Li,
                                                       d_Lx_csr,
                                                       d_Lp_csr,
                                                       d_Li_csr,
                                                       CUDA_R_64F,
                                                       CUSPARSE_ACTION_NUMERIC,
                                                       CUSPARSE_INDEX_BASE_ZERO,
                                                       CUSPARSE_CSR2CSC_ALG1,
                                                       &bufferSizeL);

  csp = cusparseCsr2cscEx2_bufferSize(handle_,
                                      n_,
                                      n_,
                                      nnzU,
                                      d_Ux,
                                      d_Up,
                                      d_Ui,
                                      d_Ux_csr,
                                      d_Up_csr,
                                      d_Ui_csr,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1,
                                      &bufferSizeU);
  /* allocate buffers */

  double* d_workL;
  double* d_workU;
  checkCudaErrors(cudaMalloc((void**)&d_workL, bufferSizeL));
  checkCudaErrors(cudaMalloc((void**)&d_workU, bufferSizeU));

  /* actual CSC to CSR */

  csp = cusparseCsr2cscEx2(handle_,
                           n_,
                           n_,
                           nnzL,
                           d_Lx,
                           d_Lp,
                           d_Li,
                           d_Lx_csr,
                           d_Lp_csr,
                           d_Li_csr,
                           CUDA_R_64F,
                           CUSPARSE_ACTION_NUMERIC,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUSPARSE_CSR2CSC_ALG1,
                           d_workL);

  csp = cusparseCsr2cscEx2(handle_,
                           n_,
                           n_,
                           nnzU,
                           d_Ux,
                           d_Up,
                           d_Ui,
                           d_Ux_csr,
                           d_Up_csr,
                           d_Ui_csr,
                           CUDA_R_64F,
                           CUSPARSE_ACTION_NUMERIC,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUSPARSE_CSR2CSC_ALG1,
                           d_workU);

  (void)csp;  // mute unused variable warnings

  /* CSC no longer needed, nor the work arrays! */

  cudaFree(d_Lp);
  cudaFree(d_Li);
  cudaFree(d_Lx);

  cudaFree(d_Up);
  cudaFree(d_Ui);
  cudaFree(d_Ux);

  cudaFree(d_workU);
  cudaFree(d_workL);

  /* actual setup */

  sp_status_ = cusolverRfSetupDevice(n_,
                                     nnz_,
                                     mat_A_csr_->get_irows(),  // dia_,
                                     mat_A_csr_->get_jcols(),  // dja_,
                                     mat_A_csr_->get_vals(),   // da_,
                                     nnzL,
                                     d_Lp_csr,
                                     d_Li_csr,
                                     d_Lx_csr,
                                     nnzU,
                                     d_Up_csr,
                                     d_Ui_csr,
                                     d_Ux_csr,
                                     d_P_,
                                     d_Q_,
                                     handle_rf_);
  cudaDeviceSynchronize();
  sp_status_ = cusolverRfAnalyze(handle_rf_);

  // clean up
  cudaFree(d_Lp_csr);
  cudaFree(d_Li_csr);
  cudaFree(d_Lx_csr);

  cudaFree(d_Up_csr);
  cudaFree(d_Ui_csr);
  cudaFree(d_Ux_csr);

  return 0;
}

// Error checking utility for CUDA
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void RefactorizationSolver::resolveCheckCudaError(T result, const char* const file, int const line)
{
  if(result) {
    fprintf(stdout, "CUDA error at %s:%d, error# %d\n", file, line, result);
    assert(false);
  }
}

}  // namespace ReSolve
