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
 * @file hiopLinSolverCholCuSparse.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_LINSOLVER_CHOL_CUSP
#define HIOP_LINSOLVER_CHOL_CUSP

#include "hiopLinSolver.hpp"

#ifdef HIOP_USE_CUDA

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "hiopMatrixSparseCsrCuda.hpp"
#include "hiopKKTLinSysSparseCondensed.hpp"
namespace hiop
{

/**
 * Wrapper class for cusolverSpXcsrchol Cholesky solver.
 */

class hiopLinSolverCholCuSparse : public hiopLinSolverSymSparse
{
public:
  hiopLinSolverCholCuSparse(hiopMatrixSparseCSR* M, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverCholCuSparse();

  /**
   * Triggers a refactorization of the matrix, if necessary.
   * Returns -1 if zero or negative pivots are encountered
   */
  int matrixChanged();

  using hiopLinSolverSymSparse::solve;

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).
   */
  bool solve(hiopVector& x_in);

protected:
  /// performs initial analysis, sparsity permutation, and rescaling
  bool initial_setup();

  /// computes the sparsity-promoting ordering based on the user options (device path)
  bool do_symb_analysis(const size_type n,
                        const size_type nnz,
                        const index_type* rowptr,
                        const index_type* colind,
                        const double* value,
                        index_type* perm);

  /**
   * Permutes an array accordingly to given permutation. All pointers are on device and
   * the method executes on device.
   */
  bool permute_vec(int n,
                   /*const*/ double* vec_in,
                   /*const*/ index_type* permutation,
                   double* vec_out);

protected:
  /// Internal handle required by cuSPARSE functions
  cusparseHandle_t h_cusparse_;

  /// Internal handle required by cusolverSpXcsrchol
  cusolverSpHandle_t h_cusolver_;

  /// Internal struct required by cusolverSpXcsrchol
  csrcholInfo_t info_;

  /// Number of nonzeros in the matrix sent to cuSOLVER
  size_type nnz_;

  /// Array with row pointers of the matrix (permuted based on ordering) to be factorized (on device)
  int* rowptr_;
  /// Array with column indexes of the matrix (permuted based on ordering) to be factorized (on device)
  int* colind_;
  /// Array with values of the matrix (permuted based on ordering) to be factorized (on device)
  double* values_;
  /// cuSPARSE matrix descriptor
  cusparseMatDescr_t mat_descr_;

  /// Buffer required by the cuSOLVER Chol factor (on device)
  unsigned char* buf_fact_;
  /// Size of the above array
  size_t buf_fact_size_;

  /// Reordering permutation to promote sparsity of the factor (on device)
  int* P_;
  /// Transpose or inverse of the above permutation (on device)
  int* PT_;
  /// Permutation map for nonzeros (on device)
  int* map_nnz_perm_;
  /// internal buffers in the size of the linear system (on device)
  double* rhs_buf1_;
  double* rhs_buf2_;

protected:
  inline hiopMatrixSparseCSRCUDA* sys_mat_csr() { return dynamic_cast<hiopMatrixSparseCSRCUDA*>(M_); }

private:
  hiopLinSolverCholCuSparse() = delete;
};

}  // namespace hiop

#endif  // HIOP_USE_CUDA
#endif  // HIOP_LINSOLVER_CHOL_CUSP
