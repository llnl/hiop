// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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
 * @file hiopMatrixDenseRowMajor.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#pragma once
#include "hiopMatrixDense.hpp"
#include <cstddef>
#include <cstdio>

namespace hiop
{

/**
 * @brief Dense matrix stored row-wise and distributed column-wise
 */
class hiopMatrixDenseRowMajor : public hiopMatrixDense
{
public:
  hiopMatrixDenseRowMajor(const size_type& m,
                          const size_type& glob_n,
                          index_type* col_part = NULL,
                          MPI_Comm comm = MPI_COMM_SELF,
                          const size_type& m_max_alloc = -1);
  virtual ~hiopMatrixDenseRowMajor();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixDense& dm);
  virtual void copyFrom(const double* buffer);
  virtual void copy_to(double* buffer);

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const;

  /* same as above for mostly internal use - avoid using it */
  virtual void timesVec(double beta, double* y, double alpha, const double* x) const;

  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const;

  /* same as above for mostly for internal use - avoid using it */
  virtual void transTimesVec(double beta, double* y, double alpha, const double* x) const;

  /**
   * @brief W = beta*W + alpha*this*X
   *
   * @pre W, 'this', and 'X' need to be local matrices (not distributed). All multiplications
   * of distributed matrices needed by HiOp internally can be done efficiently in parallel using the
   * 'timesMatTrans' and 'transTimesMat' methods below.
   */
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /**
   * @brief W = beta*W + alpha*this*X
   * Contains the implementation internals of the above; can be used on its own.
   */
  virtual void timesMat_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /**
   * @brief W = beta*W + alpha*this^T*X
   *
   * @pre 'this' should be local/non-distributed. 'X' (and 'W') can be distributed.
   *
   * Note: no inter-process communication occurs in the parallel case
   */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /**
   * @brief W = beta*W + alpha*this*X^T
   * @pre 'W' need to be local/non-distributed.
   *
   * 'this' and 'X' can be distributed, in which case communication will occur.
   */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  /* Contains dgemm wrapper needed by the above */
  virtual void timesMatTrans_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, index_type start_on_dest_diag, const hiopVector& d_);

  /**
   * @brief add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'.
   */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha,
                              const hiopVector& d_,
                              int start_on_src_vec,
                              int num_elems = -1);
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c);

  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /**
   * @brief block of W += alpha*transpose(this)
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   *
   * @pre transpose of 'this' has to fit in the upper triangle of W
   * @pre W.n() == W.m()
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const;

  /**
   * @brief diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   *
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   *
   * This functionality of this method is needed only for symmetric matrices and, for this reason,
   * only symmetric matrices classes implement/need to implement it.
   *
   * @pre this->n()==this->m()
   * @pre W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const;

  virtual double max_abs_value();

  virtual void row_max_abs_value(hiopVector& ret_vec);

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale = false);

  virtual bool isfinite() const;

  // virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const;

  virtual hiopMatrixDense* alloc_clone() const;
  virtual hiopMatrixDense* new_copy() const;

  void appendRow(const hiopVector& row);

  /// @brief copies the first 'num_rows' rows from 'src' to 'this' starting at 'row_dest'
  void copyRowsFrom(const hiopMatrixDense& src, int num_rows, int row_dest);

  /**
   * @brief Copy 'n_rows' rows specified by 'rows_idxs' (array of size 'n_rows') from 'src' to 'this'
   *
   * @pre 'this' has exactly 'n_rows' rows
   * @pre 'src' and 'this' must have same number of columns
   * @pre number of rows in 'src' must be at least the number of rows in 'this'
   */
  void copyRowsFrom(const hiopMatrix& src_gen, const index_type* rows_idxs, size_type n_rows);

  /// @brief copies 'src' into this as a block starting at (i_block_start,j_block_start)
  void copyBlockFromMatrix(const index_type i_block_start, const index_type j_block_start, const hiopMatrixDense& src);

  /**
   * @brief overwrites 'this' with 'src''s block that starts at (i_src_block_start,j_src_block_start)
   * and has dimensions of 'this'
   */
  void copyFromMatrixBlock(const hiopMatrixDense& src, const int i_src_block_start, const int j_src_block_start);
  /// @brief  shift<0 -> up; shift>0 -> down
  void shiftRows(size_type shift);
  void replaceRow(index_type row, const hiopVector& vec);
  /// @brief copies row 'irow' in the vector 'row_vec' (sizes should match)
  void getRow(index_type irow, hiopVector& row_vec);

  /// @brief build Hess for FR problem, from the base problem `Hess`.
  virtual void set_Hess_FR(const hiopMatrixDense& Hess, const hiopVector& add_diag_de);

  /// @brief build Jac for FR problem, from the base problem `Jac_c` and `Jac_d`.
  virtual void set_Jac_FR(const hiopMatrixDense& Jac_c, const hiopMatrixDense& Jac_d);

#ifdef HIOP_DEEPCHECKS
  void overwriteUpperTriangleWithLower();
  void overwriteLowerTriangleWithUpper();
#endif
  virtual size_type get_local_size_n() const { return n_local_; }
  virtual size_type get_local_size_m() const { return m_local_; }
  virtual MPI_Comm get_mpi_comm() const { return comm_; }

  double* local_data_const() const { return M_[0]; }
  double* local_data() { return M_[0]; }

protected:
  // do not use this unless you sure you know what you're doing
  inline double** get_M() { return M_; }

public:
  virtual size_type m() const { return m_local_; }
  virtual size_type n() const { return n_global_; }
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const;
#endif
  virtual bool symmetrize();

private:
  double** M_;   // local storage
  int n_local_;  // local number of rows and cols, respectively
  size_type glob_jl_, glob_ju_;

  mutable double* buff_mxnlocal_;

  // this is very private do not touch :)
  size_type max_rows_;

private:
  hiopMatrixDenseRowMajor() {};
  /** copy constructor, for internal/private use only (it doesn't copy the values) */
  hiopMatrixDenseRowMajor(const hiopMatrixDenseRowMajor&);

  inline double* new_mxnlocal_buff() const
  {
    if(buff_mxnlocal_ == NULL) {
      buff_mxnlocal_ = new double[max_rows_ * n_local_];
    }
    return buff_mxnlocal_;
  }
};

}  // namespace hiop
