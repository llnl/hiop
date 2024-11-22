// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC.
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
 * @file hiopMatrixSparseCSR.hpp
 *
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 *
 */
#ifndef HIOP_SPARSE_MATRIX_CSR
#define HIOP_SPARSE_MATRIX_CSR

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixSparse.hpp"
#include "hiopMatrixSparseTriplet.hpp"

#include <cassert>
#include <unordered_map>

namespace hiop
{

/**
 * @brief Abstract class for compressed sparse row storage for sparse matrices.
 */
class hiopMatrixSparseCSR : public hiopMatrixSparse
{
public:
  hiopMatrixSparseCSR(int num_rows, int num_cols, int nnz)
      : hiopMatrixSparse(num_rows, num_cols, nnz)
  {}

  hiopMatrixSparseCSR()
      : hiopMatrixSparseCSR(0, 0, 0)
  {}

  virtual ~hiopMatrixSparseCSR() {}

  /////////////////////////////////////////////////////////////////////
  // Below are  CSR-specific methods (addition to hiopMatrixSparse)
  /////////////////////////////////////////////////////////////////////

  /**
   * @brief Extracts the diagonal entries of `this` matrix into the vector passed as argument
   *
   * @pre `this` matrix needs to be symmetric and of same size(s) as `diag_out`
   */
  virtual void extract_diagonal(hiopVector& diag_out) const = 0;

  /**
   * Sets the diagonal of `this` to the constant `val`. If `val` is zero, the sparsity pattern
   * of `this` is not altered.
   *
   * @pre  `this` is expected to store the diagonal entries as nonzero elements.
   */
  virtual void set_diagonal(const double& val) = 0;

  /**
   * Allocates a CSR matrix capable of storing the multiplication result of M = X*Y, where X
   * is the calling matrix class (`this`) and Y is the `Y` argument of the method.
   *
   * @note Should be used in conjunction with `times_mat_symbolic` and `times_mat_numeric`
   *
   * @pre The dimensions of the matrices should be consistent with the multiplication.
   *
   */
  virtual hiopMatrixSparseCSR* times_mat_alloc(const hiopMatrixSparseCSR& Y) const = 0;

  /**
   * Computes sparsity pattern, meaning computes row pointers and column indexes of `M`,
   * of M = X*Y, where X is the calling matrix class (`this`) and Y is the second argument.
   *
   * @note The output matrix `M` will have unique and ordered column indexes (with the same
   * row)
   *
   * @note Specializations of this class may only be able to compute the sparsity pattern in
   * tandem with the numerical multiplications (for example, because of API limitations).
   * In this cases, the `times_mat_numeric` will take over sparsity computations and the
   * arrays with row pointers and column indexes may be uninitialized after this call.
   *
   * @pre The dimensions of the matrices should be consistent with the multiplication.
   *
   * @pre The column indexes within the same row must be unique and ordered for `Y`.
   *
   * @pre The internal arrays of `M` should have enough storage to hold the sparsity
   * pattern (row pointers and column indexes) and values of the multiplication result.
   * This preallocation can be done by calling `times_mat_alloc` prior to this method.
   *
   */
  virtual void times_mat_symbolic(hiopMatrixSparseCSR& M, const hiopMatrixSparseCSR& Y) const = 0;

  /**
   * Computes (numerical values of) M = beta*M + alpha*X*D*Y, where X is the calling matrix
   * class (`this`), beta and alpha are scalars passed as arguments, and M and Y are matrices
   * of appropriate sizes passed as arguments.
   *
   * @note Generally, only the nonzero values of the input/output argument `M` are updated
   * since the sparsity pattern (row pointers and column indexes) of `M` should have been
   * already computed by `times_mat_symbolic`. Some specializations of this method may be
   * restricted to performing both phases in inside this method.
   *
   * @pre The dimensions of the matrices should be consistent with the multiplication.
   *
   * @pre The column indexes within the same row must be unique and ordered both for input
   * matrices and result matrix `M`.
   *
   * @pre The indexes arrays of `this`, `Y`, and `M` should not have changed since the
   * last call to `times_diag_times_mat`.
   *
   * Example of usage:
   * //initially allocate and compute M
   * auto* M = X.times_mat_alloc(Y);
   * X.times_mat_symbolic(M, Y);
   * X.times_mat_numeric(0.0, M, 1.0, Y);
   * ... calculations ....
   * //if only nonzero entries of X and Y have changed, call the fast multiplication routine
   * X.times_mat_numeric(0.0, M, 1.0, Y);
   *
   */
  virtual void times_mat_numeric(double beta, hiopMatrixSparseCSR& M, double alpha, const hiopMatrixSparseCSR& Y) = 0;

  /// @brief Column scaling or right multiplication by a diagonal: `this`=`this`*D
  virtual void scale_cols(const hiopVector& D) = 0;

  /// @brief Row scaling or left multiplication by a diagonal: `this`=D*`this`
  virtual void scale_rows(const hiopVector& D) = 0;

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of the triplet matrix `M`.
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  virtual void form_from_symbolic(const hiopMatrixSparseTriplet& M) = 0;

  /**
   * Copies the numerical values of the triplet matrix M into the CSR matrix `this`
   *
   * @pre The sparsity pattern (row pointers and column indexes arrays) of `this` should be
   * allocated and populated, possibly by a previous call to `form_from_symbolic`
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  virtual void form_from_numeric(const hiopMatrixSparseTriplet& M) = 0;

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of transpose of the triplet matrix `M`.
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  virtual void form_transpose_from_symbolic(const hiopMatrixSparseTriplet& M) = 0;

  /**
   * Copies the numerical values of the transpose of the triplet matrix M into the
   * CSR matrix `this`
   *
   * @pre The sparsity pattern (row pointers and column indexes arrays) of `this` should be
   * allocated and populated, possibly by a previous call to `form_transpose_from_symbolic`
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  virtual void form_transpose_from_numeric(const hiopMatrixSparseTriplet& M) = 0;

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of transpose of the CSR matrix `M`.
   *
   * @pre The input argument should have the column indexes sorted and unique within a row.
   */
  virtual void form_transpose_from_symbolic(const hiopMatrixSparseCSR& M) = 0;

  /**
   * Copies the numerical values of the transpose of the CSR matrix M into the CSR matrix `this`.
   *
   * @pre The sparsity pattern (row pointers and column indexes arrays) of `this` should be
   * allocated and populated, possibly by a previous call to `form_transpose_from_symbolic`
   *
   * @pre The input argument should have the column indexes sorted and unique within a row.
   */
  virtual void form_transpose_from_numeric(const hiopMatrixSparseCSR& M) = 0;

  /**
   * (Re)Initializes `this` to a diagonal matrix with diagonal entries given by D.
   */
  virtual void form_diag_from_symbolic(const hiopVector& D) = 0;

  /**
   * Sets the diagonal entries of `this` equal to entries of D
   *
   * @pre Length of `D` should be equal to size(s) of `this`
   *
   * @pre `this` should be a diagonal matrix (in CSR format) with storage for
   * all the diagonal entries, which can be ensured by calling the sister method
   * `form_diag_from_symbolic`
   */
  virtual void form_diag_from_numeric(const hiopVector& D) = 0;

  /**
   * Allocates and returns CSR matrix `M` capable of holding M = X+Y, where X is
   * the calling matrix class (`this`) and Y is the argument passed to the method.
   */
  virtual hiopMatrixSparseCSR* add_matrix_alloc(const hiopMatrixSparseCSR& Y) const = 0;

  /**
   * Computes sparsity pattern of M = X+Y (i.e., populates the row pointers and
   * column indexes arrays) of `M`. `X` is `this`.
   *
   * @pre `this` and `Y` should hold matrices of identical dimensions.
   *
   */
  virtual void add_matrix_symbolic(hiopMatrixSparseCSR& M, const hiopMatrixSparseCSR& Y) const = 0;

  /**
   * Performs matrix addition M = alpha*X + beta*Y numerically, where
   * X is `this` and alpha and beta are scalars.
   *
   * @pre `M`, `this` and `Y` should hold matrices of identical dimensions.
   *
   * @pre `M` and `X+Y` should have identical sparsity pattern, namely the
   * `add_matrix_symbolic` should have been called previously.
   *
   */
  virtual void add_matrix_numeric(hiopMatrixSparseCSR& M, double alpha, const hiopMatrixSparseCSR& Y, double beta) const = 0;

  /** Performs a quick check and returns false if the CSR indexes are not ordered.
   *
   * Should be used with caution, for example only under HIOP_DEEPCHECKS or for debugging purposes
   * because it is a computationally intensive method for GPU implementations as transfers the
   * matrix data from device to host.
   */
  virtual bool check_csr_is_ordered() = 0;
  /////////////////////////////////////////////////////////////////////
  // end of new CSR-specific methods
  /////////////////////////////////////////////////////////////////////

protected:
  //// inherits nrows_, ncols_, and nnz_ from parent hiopSparseMatrix

private:
  hiopMatrixSparseCSR(const hiopMatrixSparseCSR&) = delete;
};

}  // namespace hiop

#endif
