// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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
 * @file hiopMatrixSparse.cpp
 *
 */
#pragma once

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"

#include <cassert>

namespace hiop
{

/**
 * @brief Sparse matrix of doubles in triplet format - it is not distributed
 * @note for now (i,j) are expected ordered: first on rows 'i' and then on cols 'j'
 */
class hiopMatrixSparse : public hiopMatrix
{
public:
  hiopMatrixSparse(int rows, int cols, int nnz)
      : nrows_(rows),
        ncols_(cols),
        nnz_(nnz)
  {}
  virtual ~hiopMatrixSparse() {}

  virtual void setToZero() = 0;
  virtual void setToConstant(double c) = 0;
  virtual void copyFrom(const hiopMatrixSparse& dm) = 0;

  /* @brief copy the nonzeros into 3 arrays, in their triplet form.
   * This function is not used right now.
   *
   * TODO: Unit test is missing.
   */
  virtual void copy_to(int* irow, int* jcol, double* val) = 0;

  /* @brief copy the matrix into a dense matrix
   * This function is not used right now. Unit test is missing.
   */
  virtual void copy_to(hiopMatrixDense& W) = 0;

  /* @brief copy `n_rows` rows from `src` into `this`, i.e., the ith row of this is copied from
   * the rows_idx[i]_th row in `src`.
   *
   *  @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows) = 0;

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;
  virtual void timesVec(double beta, double* y, double alpha, const double* x) const = 0;

  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;
  virtual void transTimesVec(double beta, double* y, double alpha, const double* x) const = 0;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  /**
   * Adds alpha times the i-th entry of `D` to the i-th diagonal of `this`.
   *
   * @pre Sparse matrices should have the diagonal entries allocated as nonzeros.
   * @pre `this` is expected to be symmetric.
   * @pre Size of `D` should match the size(s) of `M`.
   *
   */
  virtual void addDiagonal(const double& alpha, const hiopVector& D) = 0;
  virtual void addDiagonal(const double& value) = 0;
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& D) = 0;
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha,
                              const hiopVector& d_,
                              int start_on_src_vec,
                              int num_elems = -1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems', scaled by 'scal'
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                   const size_type& num_elems,
                                   const hiopVector& d_,
                                   const index_type& start_on_nnz_idx,
                                   double scal = 1.0) = 0;

  /*
   * @brief: add constant 'c' to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements.
   * The number of elements added is 'num_elems'
   *
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void setSubDiagonalTo(const index_type& start_on_dest_diag,
                                const size_type& num_elems,
                                const double& c,
                                const index_type& start_on_nnz_idx) = 0;

  /// @brief this += alpha*X
  virtual void addMatrix(double alpha, const hiopMatrix& X) = 0;

  /* block of W += alpha*transpose(this) */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const = 0;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const = 0;

  virtual void addUpperTriangleToSymSparseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixSparse& W) const
  {
    assert(false && "counterpart method of hiopMatrixSymSparse should be used");
  }

  /* diag block of W += alpha * M * D^{-1} * transpose(M), where M=this
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowCol_dest_start,
                                                       const double& alpha,
                                                       const hiopVector& D,
                                                       hiopMatrixDense& W) const = 0;

  /* block of W += alpha * M * D^{-1} * transpose(N), where M=this
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the
   * case, the method will assert(false) in debug; in release, the method will issue a
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(int row_dest_start,
                                            int col_dest_start,
                                            const double& alpha,
                                            const hiopVector& D,
                                            const hiopMatrixSparse& N,
                                            hiopMatrixDense& W) const = 0;

  /**
   * @brief Copy 'n_rows' rows from matrix 'src_gen', started from 'rows_src_idx_st', to the rows started from 'B_rows_st' in
   * 'this'. The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
   *
   * @pre 'src_gen' must have exactly, or more than 'n_rows' rows after row 'rows_src_idx_st'
   * @pre 'this' must have exactly, or more than 'n_rows' rows after row 'rows_dest_idx_st'
   * @pre 'dest_nnz_st' + the number of non-zeros in the copied the rows must be less or equal to this->numOfNumbers()
   * @pre User must know the nonzero pattern of src and dest matrices. Assume non-zero patterns of these two wont change, and
   * 'src_gen' is a submatrix of 'this'
   * @pre Otherwise, this function may replace the non-zero values and nonzero patterns for the undesired elements.
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copyRowsBlockFrom(const hiopMatrix& src_gen,
                                 const index_type& rows_src_idx_st,
                                 const size_type& n_rows,
                                 const index_type& rows_dest_idx_st,
                                 const size_type& dest_nnz_st) = 0;

  /**
   * @brief Copy matrix 'src_gen', into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
   * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
   * When `offdiag_only` is set to true, only the off-diagonal part of `src_gen` is copied.
   *
   * @pre 'this' must have enough rows and cols after row 'dest_row_st' and col 'dest_col_st'
   * @pre 'dest_nnz_st' + the number of non-zeros in the copied matrix must be less or equal to
   * this->numOfNumbers()
   * @pre User must know the nonzero pattern of src and dest matrices. The method assumes
   * that non-zero patterns does not change between calls and that 'src_gen' is a valid
   *  submatrix of 'this'
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copySubmatrixFrom(const hiopMatrix& src_gen,
                                 const index_type& dest_row_st,
                                 const index_type& dest_col_st,
                                 const size_type& dest_nnz_st,
                                 const bool offdiag_only = false) = 0;

  /**
   * @brief Copy the transpose of matrix 'src_gen', into 'this' as a submatrix from corner
   * 'dest_row_st' and 'dest_col_st'.
   * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
   * When `offdiag_only` is set to true, only the off-diagonal part of `src_gen` is copied.
   *
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                      const index_type& dest_row_st,
                                      const index_type& dest_col_st,
                                      const size_type& dest_nnz_st,
                                      const bool offdiag_only = false) = 0;

  /**
   * @brief Copy selected columns of a diagonal matrix (a constant 'scalar' times identity),
   * into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
   * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
   * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
   * @pre this function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const size_type& nnz_to_copy,
                                                       const hiopVector& ix) = 0;

  /**
   * @brief Copy selected rows of a diagonal matrix (a constant 'scalar' times identity),
   * into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
   * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
   * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
   * @pre this function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                                       const index_type& dest_row_st,
                                                       const index_type& dest_col_st,
                                                       const size_type& dest_nnz_st,
                                                       const size_type& nnz_to_copy,
                                                       const hiopVector& ix) = 0;

  /**
   * @brief Sets the diagonal of a subblock of `this` to a constant times identity matrix.
   *
   * This diagonal matrix is 'src_val'*identity matrix with size 'nnz_to_copy'x'nnz_to_copy'.
   * The destination is updated from the start row 'row_dest_st' and start column 'col_dest_st'.
   * At the destination, 'nnz_to_copy` nonzeros starting from index `dest_nnz_st` will be replased.
   * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   */
  virtual void copyDiagMatrixToSubblock(const double& src_val,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const size_type& nnz_to_copy) = 0;

  /**
   * @brief same as @copyDiagMatrixToSubblock, but copies only diagonal entries specified by `pattern`.
   * At the destination, 'nnz_to_copy` nonzeros starting from index `dest_nnz_st` will be replaced.
   * @pre The added entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   * @pre 'pattern' has same size as `x`.
   * @pre 'pattern` has exactly `nnz_to_copy` nonzeros.
   */
  virtual void copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                                  const index_type& dest_row_st,
                                                  const index_type& dest_col_st,
                                                  const size_type& dest_nnz_st,
                                                  const size_type& nnz_to_copy,
                                                  const hiopVector& pattern) = 0;

  virtual double max_abs_value() = 0;

  virtual void row_max_abs_value(hiopVector& ret_vec) = 0;

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale) = 0;

  virtual bool isfinite() const = 0;

  // virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const = 0;

  /* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
   * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
   * are available in 'vec_dest' starting at 'vec_start'
   */
  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start,
                                                    const double& alpha,
                                                    hiopVector& vec_dest,
                                                    int vec_start,
                                                    int num_elems = -1) const = 0;

  virtual hiopMatrixSparse* alloc_clone() const = 0;
  virtual hiopMatrixSparse* new_copy() const = 0;

  virtual index_type* i_row() = 0;
  virtual index_type* j_col() = 0;
  virtual double* M() = 0;
  virtual const index_type* i_row() const = 0;
  virtual const index_type* j_col() const = 0;
  virtual const double* M() const = 0;
  virtual size_type numberOfOffDiagNonzeros() const = 0;

  /// @brief build Jac for FR problem, from the base problem `Jac_c` and `Jac_d`. Set sparsity if `task`=0, otherwise set
  /// values
  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS) = 0;

  /// @brief build Hess for FR problem, from the base problem `Hess`.
  virtual void set_Hess_FR(const hiopMatrixSparse& Hess, int* iHSS, int* jHSS, double* MHSS, const hiopVector& add_diag) = 0;

  inline size_type m() const { return nrows_; }
  inline size_type n() const { return ncols_; }
  inline size_type numberOfNonzeros() const { return nnz_; }

  virtual bool is_diagonal() const = 0;

  virtual void extract_diagonal(hiopVector& diag_out) const = 0;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const { return false; }
  virtual bool checkIndexesAreOrdered() const = 0;
#endif
protected:
  size_type nrows_;  ///< number of rows
  size_type ncols_;  ///< number of columns
  size_type nnz_;    ///< number of nonzero entries
};

}  // namespace hiop
