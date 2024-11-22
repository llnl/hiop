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
 * @file RefactorizationSolver.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#pragma once

#include "klu.h"
#include "resolve_cusolver_defs.hpp"
#include <string>

namespace ReSolve
{

class MatrixCsr;
class IterativeRefinement;

/**
 * @brief Implements refactorization solvers using KLU and cuSOLVER libraries
 *
 */
class RefactorizationSolver
{
public:
  // constructor
  // RefactorizationSolver();
  RefactorizationSolver(int n);
  ~RefactorizationSolver();

  void enable_iterative_refinement();
  void setup_iterative_refinement_matrix(int n, int nnz);
  void configure_iterative_refinement(cusparseHandle_t cusparse_handle,
                                      cublasHandle_t cublas_handle,
                                      cusolverRfHandle_t cusolverrf_handle,
                                      int n,
                                      double* d_T,
                                      int* d_P,
                                      int* d_Q,
                                      double* devx,
                                      double* devr);

  /**
   * @brief Set the number of nonzeros in system matrix.
   *
   * @param nnz
   */
  void set_nnz(int nnz) { nnz_ = nnz; }

  IterativeRefinement* ir() { return ir_; }

  MatrixCsr* mat_A_csr() { return mat_A_csr_; }

  double* devr() { return devr_; }

  int& ordering() { return ordering_; }

  std::string& fact() { return fact_; }

  std::string& refact() { return refact_; }

  std::string& use_ir() { return use_ir_; }

  void set_silent_output(bool silent_output) { silent_output_ = silent_output; }

  /**
   * @brief Set up factorization of the first linear system.
   *
   * @return int
   */
  int setup_factorization();

  /**
   * @brief Factorize system matrix
   *
   * @return int - factorization status: success=0, failure=-1
   */
  int factorize();

  /**
   * @brief Set the up the refactorization
   *
   */
  void setup_refactorization();

  /**
   * @brief Refactorize system matrix
   *
   * @return int
   */
  int refactorize();

  /**
   * @brief Invokes triangular solver given matrix factors
   *
   * @param dx
   * @param tol
   * @return bool
   */
  bool triangular_solve(double* dx, double tol, std::string memspace);

private:
  int n_{0};    ///< Size of the linear system
  int nnz_{0};  ///< Number of nonzeros in the system's matrix

  MatrixCsr* mat_A_csr_{nullptr};     ///< System matrix in nonsymmetric CSR format
  IterativeRefinement* ir_{nullptr};  ///< Iterative refinement class

  bool cusolver_glu_enabled_{false};          ///< cusolverGLU on/off flag
  bool cusolver_rf_enabled_{false};           ///< cusolverRf on/off flag
  bool iterative_refinement_enabled_{false};  ///< Iterative refinement on/off flag
  bool is_first_solve_{true};                 ///< If it is first call to triangular solver

  // Options
  int ordering_{-1};
  std::string fact_;
  std::string refact_;
  std::string use_ir_;
  bool silent_output_{true};

  /** needed for cuSolver **/

  cusolverStatus_t sp_status_;
  cusparseHandle_t handle_ = 0;
  cusolverSpHandle_t handle_cusolver_ = nullptr;
  cublasHandle_t handle_cublas_;

  cusparseMatDescr_t descr_A_;
  cusparseMatDescr_t descr_M_;
  csrluInfoHost_t info_lu_ = nullptr;
  csrgluInfo_t info_M_ = nullptr;

  cusolverRfHandle_t handle_rf_ = nullptr;
  size_t buffer_size_;
  size_t size_M_;
  double* d_work_;
  int ite_refine_succ_ = 0;
  double r_nrminf_;

  // KLU stuff
  int klu_status_;
  klu_common Common_;
  klu_symbolic* Symbolic_ = nullptr;
  klu_numeric* Numeric_ = nullptr;
  /*pieces of M */
  int* mia_ = nullptr;
  int* mja_ = nullptr;

  /* CPU data */
  double* hostx_ = nullptr;

  /* for GPU data */
  double* devx_ = nullptr;
  double* devr_ = nullptr;

  /* needed for cuSolverRf */
  int* d_P_ = nullptr;
  int* d_Q_ = nullptr;  // permutation matrices
  double* d_T_ = nullptr;

  /**
   * @brief Function that computes M = (L-I) + U
   *
   * @param n
   * @param nnzL
   * @param Lp
   * @param Li
   * @param nnzU
   * @param Up
   * @param Ui
   * @return int
   */
  int createM(const int n, const int nnzL, const int* Lp, const int* Li, const int nnzU, const int* Up, const int* Ui);

  int initializeKLU();
  int initializeCusolverGLU();
  int initializeCusolverRf();

  int refactorizationSetupCusolverGLU();
  int refactorizationSetupCusolverRf();

  /**
   * @brief Check for CUDA errors.
   *
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template<typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);
};

}  // namespace ReSolve
