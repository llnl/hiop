// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov and Juraj Kardos
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

#ifndef HIOP_MATRIX
#define HIOP_MATRIX

#include <cstdio>
#include "hiop_defs.hpp"

namespace hiop
{

class hiopVector;
class hiopVectorPar;
class hiopMatrixDense;

/* See readme.md for some conventions on matrices */
class hiopMatrix
{
public:
  hiopMatrix() {}
  virtual ~hiopMatrix() {}

  virtual hiopMatrix* alloc_clone() const = 0;
  virtual hiopMatrix* new_copy() const = 0;

  virtual void setToZero() = 0;
  virtual void setToConstant(double c) = 0;

  /// @brief y = beta * y + alpha * this * x
  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;

  /// @brief y = beta * y + alpha * this^T * x
  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const = 0;
  /// @brief W = beta*W + alpha*this*X
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  /// @brief W = beta*W + alpha*this^T*X
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  /// @brief W = beta*W + alpha*this*X^T
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  /// @brief this += alpha * (sub)diag
  virtual void addDiagonal(const double& alpha, const hiopVector& d_) = 0;
  virtual void addDiagonal(const double& value) = 0;

  /**
   * @brief subdigonal(this) += alpha*d
   *
   * Adds elements of 'd' to the diagonal of 'this' starting at 'start_on_dest_diag'.
   * Precondition:  start_on_dest_diag + length(d) <= n_local_
   *
   * @pre _this_ is local/non-distributed
   */
  virtual void addSubDiagonal(const double& alpha, index_type start_on_dest_diag, const hiopVector& d) = 0;

  /**
   * @brief subdigonal(this) += alpha*d *
   * Adds to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * if num_elems>=0, otherwise the remaining elems in 'd' starting at 'start_on_src_vec'.
   *
   * @pre _this_ is local/non-distributed
   */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha,
                              const hiopVector& d,
                              int start_on_src_vec,
                              int num_elems = -1) = 0;

  /**
   * @brief subdiagonal(this) += c
   *
   * Adds the constant @param c to the diagonal starting at @param start_on_dest_diag
   * and updating @param num_elems in the diagonal
   *
   * @pre _this_ is local/non-distributed
   */
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c) = 0;

  /// @brief this += alpha*X
  virtual void addMatrix(double alpha, const hiopMatrix& X) = 0;

  /**
   * @brief block of W += alpha*transpose(this)
   *
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   *
   * The functionality of this method is needed only for general (non-symmetric) matrices and, for this
   * reason, only general matrices classes implement/need to implement this method.
   *
   * @pre transpose of 'this' fits in the upper triangle of W
   * @pre W.n() == W.m()
   * @pre 'this' and W are local/non-distributed matrices
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const = 0;

  /**
   * @brief diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute to.
   *
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed.
   *
   * This functionality of this method is needed only for symmetric matrices and, for this reason,
   * only symmetric matrices classes implement/need to implement it.
   *
   * @pre this->n()==this->m()
   * @pre W.n() == W.m()
   * @pre 'this' and W are local/non-distributed matrices
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const = 0;

  /**
   * @brief Copy 'n_rows' rows specified by 'rows_idxs' (array of size 'n_rows') from 'src' to 'this'
   *
   * @pre 'this' has exactly 'n_rows' rows
   * @pre 'src' and 'this' must have same number of columns
   * @pre number of rows in 'src' must be at least the number of rows in 'this'
   */
  virtual void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows) = 0;

  virtual double max_abs_value() = 0;

  /**
   * @brief Find the maximum absolute value in each row of `this` matrix, and return them in `ret_vec`
   *
   * @pre 'ret_vec' has exactly same number of rows as `this` matrix
   */
  virtual void row_max_abs_value(hiopVector& ret_vec) = 0;

  /**
   * @brief Scale each row of `this` matrix, according to the component of `ret_vec`.
   *
   * if inv_scale=false:
   *   this[i] = ret_vec[i]*this[i]
   * else
   *   this[i] = (1/ret_vec[i])*this[i]
   *
   * @pre 'ret_vec' has exactly same number of rows as `this` matrix
   */
  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale) = 0;

  /** @brief return false is any of the entry is a nan, inf, or denormalized */
  virtual bool isfinite() const = 0;

  /**
   * @brief call with -1 to print all rows, all columns, or on all ranks; otherwise will
   *  will print the first rows and/or columns on the specified rank.
   *
   * If the underlying matrix is sparse, maxCols is ignored and a max number elements
   * given by the value of 'maxRows' will be printed. If this value is negative, all
   * elements will be printed.
   */
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const = 0;

  /// @brief number of rows
  virtual size_type m() const = 0;

  /// @brief number of columns
  virtual size_type n() const = 0;
#ifdef HIOP_DEEPCHECKS
  /**
   * @brief Checks symmetry for locally/non-distributed matrices: returns true if the absolute difference
   * (i,j) and  (j,i) entries is less than @param tol, otherwise return false and assert(false)
   *
   * For distributed matrices, this function returns false (and assert(false)).
   */
  virtual bool assertSymmetry(double tol = 1e-16) const = 0;
#endif
};

}  // namespace hiop
#endif
