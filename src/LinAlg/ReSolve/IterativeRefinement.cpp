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
 * @file IterativeRefinement.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#include "IterativeRefinement.hpp"

#include "hiop_blasdefs.hpp"
#include "KrylovSolverKernels.h"

#include "klu.h"
#include "cusparse_v2.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#define checkCudaErrors(val) resolveCheckCudaError((val), __FILE__, __LINE__)

namespace ReSolve
{

// Default constructor
IterativeRefinement::IterativeRefinement() {}

// Parametrized constructor
IterativeRefinement::IterativeRefinement(int restart, double tol, int maxit)
    : restart_{restart},
      maxit_{maxit},
      tol_{tol}
{}

IterativeRefinement::~IterativeRefinement()
{
  cusparseDestroySpMat(mat_A_);
  // free GPU variables that belong to this class and are not shared with CUSOLVER class
  cudaFree(mv_buffer_);
  cudaFree(d_V_);
  cudaFree(d_Z_);
  cudaFree(d_rvGPU_);
  cudaFree(d_Hcolumn_);

  if(orth_option_ == "cgs2") {
    cudaFree(d_H_col_);
  }
  // delete all CPU GMRES variables
  delete[] h_H_;

  if(orth_option_ == "mgs_two_synch" || orth_option_ == "mgs_pm") {
    delete[] h_L_;
    delete[] h_rv_;
  }
  delete[] h_c_;
  delete[] h_s_;
  delete[] h_rs_;

  if(orth_option_ == "mgs_pm" || orth_option_ == "cgs2") {
    delete[] h_aux_;
  }
}

int IterativeRefinement::setup_system_matrix(int n, int nnz, int* dia, int* dja, double* da)
{
  dia_ = dia;
  dja_ = dja;
  da_ = da;
  n_ = n;
  nnz_ = nnz;
  checkCudaErrors(cusparseCreateCsr(&mat_A_,
                                    n,
                                    n,
                                    nnz,
                                    dia_,
                                    dja_,
                                    da_,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    CUDA_R_64F));
  return 0;
}

int IterativeRefinement::setup(cusparseHandle_t cusparse_handle,
                               cublasHandle_t cublas_handle,
                               cusolverRfHandle_t cusolverrf_handle,
                               int n,
                               double* d_T,
                               int* d_P,
                               int* d_Q,
                               double* devx,
                               double* devr)
{
  cusparse_handle_ = cusparse_handle;
  cublas_handle_ = cublas_handle;
  cusolverrf_handle_ = cusolverrf_handle;
  assert(n_ == n && "Size of the linear system incorrectly set in the iterative refinement class!\n");

  // only set pointers
  d_T_ = d_T;
  d_P_ = d_P;
  d_Q_ = d_Q;

  // setup matvec

  cusparseCreateDnVec(&vec_x_, n_, devx, CUDA_R_64F);
  cusparseCreateDnVec(&vec_Ax_, n_, devr, CUDA_R_64F);
  size_t buffer_size;
  checkCudaErrors(cusparseSpMV_bufferSize(cusparse_handle_,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &(minusone_),
                                          mat_A_,
                                          vec_x_,
                                          &(one_),
                                          vec_Ax_,
                                          CUDA_R_64F,
                                          CUSPARSE_SPMV_CSR_ALG2,
                                          &buffer_size));

  cudaDeviceSynchronize();
  checkCudaErrors(cudaMalloc(&mv_buffer_, buffer_size));

  // allocate space for the GPU

  checkCudaErrors(cudaMalloc(&(d_V_), n_ * (restart_ + 1) * sizeof(double)));
  checkCudaErrors(cudaMalloc(&(d_Z_), n_ * (restart_ + 1) * sizeof(double)));
  checkCudaErrors(cudaMalloc(&(d_rvGPU_), 2 * (restart_ + 1) * sizeof(double)));
  checkCudaErrors(cudaMalloc(&(d_Hcolumn_), 2 * (restart_ + 1) * (restart_ + 1) * sizeof(double)));

  // and for the CPU

  h_H_ = new double[restart_ * (restart_ + 1)];
  h_c_ = new double[restart_];       // needed for givens
  h_s_ = new double[restart_];       // same
  h_rs_ = new double[restart_ + 1];  // for residual norm history

  // for specific orthogonalization options, need a little more memory
  if(orth_option_ == "mgs_two_synch" || orth_option_ == "mgs_pm") {
    h_L_ = new double[restart_ * (restart_ + 1)];
    h_rv_ = new double[restart_ + 1];
  }

  if(orth_option_ == "cgs2") {
    h_aux_ = new double[restart_ + 1];
    checkCudaErrors(cudaMalloc(&(d_H_col_), (restart_ + 1) * sizeof(double)));
  }

  if(orth_option_ == "mgs_pm") {
    h_aux_ = new double[restart_ + 1];
  }
  return 0;
}

double IterativeRefinement::getFinalResidalNorm() { return final_residual_norm_; }

double IterativeRefinement::getInitialResidalNorm() { return initial_residual_norm_; }

double IterativeRefinement::getBNorm() { return bnorm_; }

int IterativeRefinement::getFinalNumberOfIterations() { return fgmres_iters_; }

double IterativeRefinement::matrixAInfNrm()
{
  double nrm;
  matrix_row_sums(n_, nnz_, dia_, da_, d_Z_);
  cusolverSpDnrminf(cusolver_handle_, n_, d_Z_, &nrm, mv_buffer_ /* at least 8192 bytes */);
  return nrm;
}

double IterativeRefinement::vectorInfNrm(int n, double* d_v)
{
  double nrm;

  cusolverSpDnrminf(cusolver_handle_, n, d_v, &nrm, mv_buffer_ /* at least 8192 bytes */);
  return nrm;
}

void IterativeRefinement::fgmres(double* d_x, double* d_b)
{
  int outer_flag = 1;
  int notconv = 1;
  int i = 0;
  int it = 0;
  int j;
  int k;
  int k1;

  double t;
  double rnorm;
  double bnorm;
  // double rnorm_aux;
  double tolrel;
  // V[0] = b-A*x_0
  cudaMemcpy(&(d_V_[0]), d_b, sizeof(double) * n_, cudaMemcpyDeviceToDevice);

  cudaMatvec(d_x, d_V_, "residual");

  rnorm = 0.0;
  cublasDdot(cublas_handle_, n_, d_b, 1, d_b, 1, &bnorm);
  cublasDdot(cublas_handle_, n_, d_V_, 1, d_V_, 1, &rnorm);
  // rnorm = ||V_1||
  rnorm = sqrt(rnorm);
  bnorm = sqrt(bnorm);
  bnorm_ = bnorm;
  while(outer_flag) {
    // check if maybe residual is already small enough?
    if(it == 0) {
      tolrel = tol_ * rnorm;
      if(fabs(tolrel) < 1e-16) {
        tolrel = 1e-16;
      }
    }
    int exit_cond = 0;
    if(conv_cond() == 0) {
      exit_cond = ((fabs(rnorm - ZERO) <= EPSILON));
    } else {
      if(conv_cond() == 1) {
        exit_cond = ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < tol_));
      } else {
        if(conv_cond() == 2) {
          exit_cond = ((fabs(rnorm - ZERO) <= EPSILON) || (rnorm < (tol_ * bnorm)));
        }
      }
    }
    if(exit_cond) {
      outer_flag = 0;
      final_residual_norm_ = rnorm;
      initial_residual_norm_ = rnorm;
      fgmres_iters_ = 0;
      break;
    }

    // normalize first vector
    t = 1.0 / rnorm;
    cublasDscal(cublas_handle_, n_, &t, d_V_, 1);

    // initialize norm history

    h_rs_[0] = rnorm;
    initial_residual_norm_ = rnorm;
    i = -1;
    notconv = 1;

    while((notconv) && (it < maxit_)) {
      i++;
      it++;
      // Z_i = (LU)^{-1}*V_i
      cudaMemcpy(&d_Z_[i * n_], &d_V_[i * n_], sizeof(double) * n_, cudaMemcpyDeviceToDevice);
      checkCudaErrors(cusolverRfSolve(cusolverrf_handle_, d_P_, d_Q_, 1, d_T_, n_, &d_Z_[i * n_], n_));
      cudaDeviceSynchronize();
      // V_{i+1}=A*Z_i
      cudaMatvec(&d_Z_[i * n_], &d_V_[(i + 1) * n_], "matvec");
      // orthogonalize V[i+1], form a column of h_L
      GramSchmidt(i);

      if(i != 0) {
        for(int k = 1; k <= i; k++) {
          k1 = k - 1;
          t = h_H_[i * (restart_ + 1) + k1];
          h_H_[i * (restart_ + 1) + k1] = h_c_[k1] * t + h_s_[k1] * h_H_[i * (restart_ + 1) + k];
          h_H_[i * (restart_ + 1) + k] = -h_s_[k1] * t + h_c_[k1] * h_H_[i * (restart_ + 1) + k];
        }
      }  // if i!=0

      double Hii = h_H_[i * (restart_ + 1) + i];
      double Hii1 = h_H_[(i) * (restart_ + 1) + i + 1];
      double gam = sqrt(Hii * Hii + Hii1 * Hii1);

      if(fabs(gam - ZERO) <= EPSILON) {
        gam = EPSMAC;
      }

      /* next Given's rotation */
      h_c_[i] = Hii / gam;
      h_s_[i] = Hii1 / gam;
      h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
      h_rs_[i] = h_c_[i] * h_rs_[i];

      h_H_[(i) * (restart_ + 1) + (i)] = h_c_[i] * Hii + h_s_[i] * Hii1;
      h_H_[(i) * (restart_ + 1) + (i + 1)] = h_c_[i] * Hii1 - h_s_[i] * Hii;

      // residual norm estimate
      rnorm = fabs(h_rs_[i + 1]);
      // check convergence
      if(i + 1 >= restart_ || rnorm <= tolrel || it >= maxit_) {
        notconv = 0;
      }
    }  // inner while

    // solve tri system
    h_rs_[i] = h_rs_[i] / h_H_[i * (restart_ + 1) + i];
    for(int ii = 2; ii <= i + 1; ii++) {
      k = i - ii + 1;
      k1 = k + 1;
      t = h_rs_[k];
      for(j = k1; j <= i; j++) {
        t -= h_H_[j * (restart_ + 1) + k] * h_rs_[j];
      }
      h_rs_[k] = t / h_H_[k * (restart_ + 1) + k];
    }

    // get solution
    for(j = 0; j <= i; j++) {
      cublasDaxpy(cublas_handle_, n_, &h_rs_[j], &d_Z_[j * n_], 1, d_x, 1);
    }

    /* test solution */

    if(rnorm <= tolrel || it >= maxit_) {
      // rnorm_aux = rnorm;
      outer_flag = 0;
    }

    cudaMemcpy(&d_V_[0], d_b, sizeof(double) * n_, cudaMemcpyDeviceToDevice);
    cudaMatvec(d_x, d_V_, "residual");

    rnorm = 0.0;
    cublasDdot(cublas_handle_, n_, d_V_, 1, d_V_, 1, &rnorm);
    // rnorm = ||V_1||
    rnorm = sqrt(rnorm);

    if(!outer_flag) {
      final_residual_norm_ = rnorm;
      fgmres_iters_ = it;
    }
  }  // outer while
}

// b-Ax
void IterativeRefinement::cudaMatvec(double* d_x, double* d_b, std::string option)
{
  cusparseCreateDnVec(&vec_x_, n_, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vec_Ax_, n_, d_b, CUDA_R_64F);
  if(option == "residual") {
    // b = b-Ax
    cusparseSpMV(cusparse_handle_,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &minusone_,
                 mat_A_,
                 vec_x_,
                 &one_,
                 vec_Ax_,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 mv_buffer_);
  } else {
    // just b = A*x
    cusparseSpMV(cusparse_handle_,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &one_,
                 mat_A_,
                 vec_x_,
                 &zero_,
                 vec_Ax_,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 mv_buffer_);
  }
  cusparseDestroyDnVec(vec_x_);
  cusparseDestroyDnVec(vec_Ax_);
}

void IterativeRefinement::GramSchmidt(int i)
{
  double t;
  const double one = 1.0;
  const double minusone = -1.0;
  const double zero = 0.0;
  double s;
  int sw = 0;
  if(orth_option_ == "mgs") {
    sw = 0;
  } else {
    if(orth_option_ == "cgs2") {
      sw = 1;
    } else {
      if(orth_option_ == "mgs_two_synch") {
        sw = 2;
      } else {
        if(orth_option_ == "mgs_pm") {
          sw = 3;
        } else {
          // display error message and set sw = 0;
          /*
             nlp_->log->printf(hovWarning,
             "Wrong Gram-Schmidt option. Setting default (modified Gram-Schmidt, mgs) ...\n");
             */
          sw = 0;
        }
      }
    }
  }

  switch(sw) {
    case 0:  // mgs

      for(int j = 0; j <= i; ++j) {
        t = 0.0;
        cublasDdot(cublas_handle_, n_, &d_V_[j * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

        h_H_[i * (restart_ + 1) + j] = t;
        t *= -1.0;

        cublasDaxpy(cublas_handle_, n_, &t, &d_V_[j * n_], 1, &d_V_[(i + 1) * n_], 1);
      }
      t = 0.0;
      cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

      // set the last entry in Hessenberg matrix
      t = sqrt(t);
      h_H_[(i) * (restart_ + 1) + i + 1] = t;
      if(t != 0.0) {
        t = 1.0 / t;
        cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
      } else {
        assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
      }
      break;

    case 1:  // cgs2
      // Hcol = V(:,1:i)^T *V(:,i+1);
      cublasDgemv(cublas_handle_, CUBLAS_OP_T, n_, i + 1, &one_, d_V_, n_, &d_V_[(i + 1) * n_], 1, &zero_, d_H_col_, 1);
      // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol
      cublasDgemv(cublas_handle_, CUBLAS_OP_N, n_, i + 1, &minusone_, d_V_, n_, d_H_col_, 1, &one_, &d_V_[n_ * (i + 1)], 1);
      // copy H_col to aux, we will need it later

      cudaMemcpy(h_aux_, d_H_col_, sizeof(double) * (i + 1), cudaMemcpyDeviceToHost);

      // Hcol = V(:,1:i)*V(:,i+1);
      cublasDgemv(cublas_handle_, CUBLAS_OP_T, n_, i + 1, &one_, d_V_, n_, &d_V_[(i + 1) * n_], 1, &zero_, d_H_col_, 1);
      // V(:,i+1) = V(:, i+1) -  V(:,1:i)*Hcol

      cublasDgemv(cublas_handle_, CUBLAS_OP_N, n_, i + 1, &minusone_, d_V_, n_, d_H_col_, 1, &one_, &d_V_[n_ * (i + 1)], 1);
      // copy H_col to H

      cudaMemcpy(&h_H_[i * (restart_ + 1)], d_H_col_, sizeof(double) * (i + 1), cudaMemcpyDeviceToHost);
      // add both pieces together (unstable otherwise, careful here!!)
      for(int j = 0; j <= i; ++j) {
        h_H_[i * (restart_ + 1) + j] += h_aux_[j];
      }
      t = 0.0;
      cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

      // set the last entry in Hessenberg matrix
      t = sqrt(t);
      h_H_[(i) * (restart_ + 1) + i + 1] = t;
      if(t != 0.0) {
        t = 1.0 / t;
        cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
      } else {
        assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
      }
      break;
      // the two low synch schemes
    case 2:
      // KS: the kernels are limited by the size of the shared memory on the GPU. If too many vectors in Krylov space, use
      // standard cublas routines. V[1:i]^T[V[i] w]
      if(i < 200) {
        mass_inner_product_two_vectors(n_, i, &d_V_[i * n_], &d_V_[(i + 1) * n_], d_V_, d_rvGPU_);
      } else {
        cublasDgemm(cublas_handle_,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i + 1,          // m
                    2,              // n
                    n_,             // k
                    &one,           // alpha
                    d_V_,           // A
                    n_,             // lda
                    &d_V_[i * n_],  // B
                    n_,             // ldb
                    &zero,
                    d_rvGPU_,  // c
                    i + 1);    // ldc
      }
      // copy rvGPU to L
      cudaMemcpy(&h_L_[(i) * (restart_ + 1)], d_rvGPU_, (i + 1) * sizeof(double), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_rv_, &d_rvGPU_[i + 1], (i + 1) * sizeof(double), cudaMemcpyDeviceToHost);

      for(int j = 0; j <= i; ++j) {
        h_H_[(i) * (restart_ + 1) + j] = 0.0;
      }
      // triangular solve
      for(int j = 0; j <= i; ++j) {
        h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
        s = 0.0;
        for(int k = 0; k < j; ++k) {
          s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
        }  // for k
        h_H_[(i) * (restart_ + 1) + j] -= s;
      }  // for j

      cudaMemcpy(d_Hcolumn_, &h_H_[(i) * (restart_ + 1)], (i + 1) * sizeof(double), cudaMemcpyHostToDevice);
      // again, use std cublas functions if Krylov space is too large
      if(i < 200) {
        mass_axpy(n_, i, d_V_, &d_V_[(i + 1) * n_], d_Hcolumn_);
      } else {
        cublasDgemm(cublas_handle_,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n_,          // m
                    1,           // n
                    i + 1,       // k
                    &minusone,   // alpha
                    d_V_,        // A
                    n_,          // lda
                    d_Hcolumn_,  // B
                    i + 1,       // ldb
                    &one,
                    &d_V_[(i + 1) * n_],  // c
                    n_);                  // ldc
      }
      // normalize (second synch)
      t = 0.0;
      cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

      // set the last entry in Hessenberg matrix
      t = sqrt(t);
      h_H_[(i) * (restart_ + 1) + i + 1] = t;
      if(t != 0.0) {
        t = 1.0 / t;
        cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
      } else {
        assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
      }
      break;

    case 3:  // two synch Gauss-Seidel mgs, SUPER STABLE
      // according to unpublisjed work by ST
      // L is where we keep the triangular matrix(L is ON THE CPU)
      // if Krylov space is too large, use std cublas (because out of shared mmory)
      if(i < 200) {
        mass_inner_product_two_vectors(n_, i, &d_V_[i * n_], &d_V_[(i + 1) * n_], d_V_, d_rvGPU_);
      } else {
        cublasDgemm(cublas_handle_,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    i + 1,          // m
                    2,              // n
                    n_,             // k
                    &one,           // alpha
                    d_V_,           // A
                    n_,             // lda
                    &d_V_[i * n_],  // B
                    n_,             // ldb
                    &zero,
                    d_rvGPU_,  // c
                    i + 1);    // ldc
      }
      // copy rvGPU to L
      cudaMemcpy(&h_L_[(i) * (restart_ + 1)], d_rvGPU_, (i + 1) * sizeof(double), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_rv_, &d_rvGPU_[i + 1], (i + 1) * sizeof(double), cudaMemcpyDeviceToHost);

      for(int j = 0; j <= i; ++j) {
        h_H_[(i) * (restart_ + 1) + j] = 0.0;
      }
      // triangular solve
      for(int j = 0; j <= i; ++j) {
        h_H_[(i) * (restart_ + 1) + j] = h_rv_[j];
        s = 0.0;
        for(int k = 0; k < j; ++k) {
          s += h_L_[j * (restart_ + 1) + k] * h_H_[(i) * (restart_ + 1) + k];
        }  // for k
        h_H_[(i) * (restart_ + 1) + j] -= s;
      }  // for j

      // now compute h_rv = L^T h_H
      double h;
      for(int j = 0; j <= i; ++j) {
        // go through COLUMN OF L
        h_rv_[j] = 0.0;
        for(int k = j + 1; k <= i; ++k) {
          h = h_L_[k * (restart_ + 1) + j];
          h_rv_[j] += h_H_[(i) * (restart_ + 1) + k] * h;
        }
      }

      // and do one more tri solve with L^T: h_aux = (I-L)^{-1}h_rv
      for(int j = 0; j <= i; ++j) {
        h_aux_[j] = h_rv_[j];
        s = 0.0;
        for(int k = 0; k < j; ++k) {
          s += h_L_[j * (restart_ + 1) + k] * h_aux_[k];
        }  // for k
        h_aux_[j] -= s;
      }  // for j

      // and now subtract that from h_H
      for(int j = 0; j <= i; ++j) {
        h_H_[(i) * (restart_ + 1) + j] -= h_aux_[j];
      }
      cudaMemcpy(d_Hcolumn_, &h_H_[(i) * (restart_ + 1)], (i + 1) * sizeof(double), cudaMemcpyHostToDevice);
      // if Krylov space too large, use std cublas routines
      if(i < 200) {
        mass_axpy(n_, i, d_V_, &d_V_[(i + 1) * n_], d_Hcolumn_);
      } else {
        cublasDgemm(cublas_handle_,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n_,          // m
                    1,           // n
                    i + 1,       // k
                    &minusone,   // alpha
                    d_V_,        // A
                    n_,          // lda
                    d_Hcolumn_,  // B
                    i + 1,       // ldb
                    &one,
                    &d_V_[(i + 1) * n_],  // c
                    n_);                  // ldc
      }
      // normalize (second synch)
      t = 0.0;
      cublasDdot(cublas_handle_, n_, &d_V_[(i + 1) * n_], 1, &d_V_[(i + 1) * n_], 1, &t);

      // set the last entry in Hessenberg matrix
      t = sqrt(t);
      h_H_[(i) * (restart_ + 1) + i + 1] = t;
      if(t != 0.0) {
        t = 1.0 / t;
        cublasDscal(cublas_handle_, n_, &t, &d_V_[(i + 1) * n_], 1);
      } else {
        assert(0 && "Iterative refinement failed, Krylov vector with zero norm\n");
      }
      break;

    default:
      assert(0 && "Iterative refinement failed, wrong orthogonalization.\n");
      break;
  }  // switch
}  // GramSchmidt

// Error checking utility for CUDA
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void IterativeRefinement::resolveCheckCudaError(T result, const char* const file, int const line)
{
#ifdef DEBUG
  if(result) {
    fprintf(stdout, "CUDA error at %s:%d, error# %d\n", file, line, result);
    assert(false);
  }
#endif
}

}  // namespace ReSolve
