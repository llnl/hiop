// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
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
 * @file hiopMatrixSparseCSRSeq.hpp
 *
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 *
 */
#ifndef HIOP_SPARSE_MATRIX_CSRSEQ
#define HIOP_SPARSE_MATRIX_CSRSEQ

#include "ExecSpace.hpp"

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixSparseCSR.hpp"
#include "hiopMatrixSparseTriplet.hpp"

#include <cassert>
#include <unordered_map>

namespace hiop
{

/**
 * @brief Sparse matrix of doubles in compressed row format for use on CPU/host. Data
 * is not (memory, MPI) distributed.
 *
 * @note The methods of this class expect and maintains unique and ordered column indexes
 * within the same row.
 *
 * Note: most of the methods are not implemented (TODO) as this is work in progress (wip).
 */
class hiopMatrixSparseCSRSeq : public hiopMatrixSparseCSR
{
public:
  hiopMatrixSparseCSRSeq(int num_rows, int num_cols, int nnz);
  hiopMatrixSparseCSRSeq();
  virtual ~hiopMatrixSparseCSRSeq();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixSparse& dm);
  virtual void copy_to(index_type* irow, index_type* jcol, double* val);
  virtual void copy_to(hiopMatrixDense& W);

  virtual void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows);

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta, double* y, double alpha, const double* x) const;

  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta, double* y, double alpha, const double* x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_);

  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(index_type start_on_dest_diag,
                              const double& alpha,
                              const hiopVector& d_,
                              index_type start_on_src_vec,
                              int num_elems = -1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(index_type start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  /* Add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems',
   * scaled by 'scal'
   */
  virtual void copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                   const size_type& num_elems,
                                   const hiopVector& d_,
                                   const index_type& start_on_nnz_idx,
                                   double scal = 1.0);

  /* add constant 'c' to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements.
   * The number of elements added is 'num_elems'
   */
  virtual void setSubDiagonalTo(const index_type& start_on_dest_diag,
                                const size_type& num_elems,
                                const double& c,
                                const index_type& start_on_nnz_idx);

  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /* block of W += alpha*transpose(this), where W is dense */
  virtual void transAddToSymDenseMatrixUpperTriangle(index_type row_dest_start,
                                                     index_type col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(index_type diag_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not yet implemented");
  }

  virtual void addUpperTriangleToSymSparseMatrixUpperTriangle(index_type diag_start, double alpha, hiopMatrixSparse& W) const
  {
    assert(false && "not yet implemented");
  }

  /* Diag block of W += alpha * M * D^{-1} * transpose(M), where M=this
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(index_type rowCol_dest_start,
                                                       const double& alpha,
                                                       const hiopVector& D,
                                                       hiopMatrixDense& W) const;

  /* Block of W += alpha * M * D^{-1} * transpose(N), where M=this
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the
   * case, the method will assert(false) in debug; in release, the method will issue a
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(index_type row_dest_start,
                                            index_type col_dest_start,
                                            const double& alpha,
                                            const hiopVector& D,
                                            const hiopMatrixSparse& N,
                                            hiopMatrixDense& W) const;

  virtual void copyRowsBlockFrom(const hiopMatrix& src_gen,
                                 const index_type& rows_src_idx_st,
                                 const size_type& n_rows,
                                 const index_type& rows_dest_idx_st,
                                 const size_type& dest_nnz_st);

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
                                 const bool offdiag_only = false);

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
                                      const bool offdiag_only = false);

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
                                                       const hiopVector& ix);

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
                                                       const hiopVector& ix);

  /**
   * @brief Copy a diagonal matrix to destination.
   * This diagonal matrix is 'src_val'*identity matrix with size 'nnz_to_copy'x'nnz_to_copy'.
   * The destination is updated from the start row 'row_dest_st' and start column 'col_dest_st'. USE WITH CAUTION!
   */
  virtual void copyDiagMatrixToSubblock(const double& src_val,
                                        const index_type& dest_row_st,
                                        const index_type& dest_col_st,
                                        const size_type& dest_nnz_st,
                                        const size_type& nnz_to_copy);

  /**
   * @brief same as @copyDiagMatrixToSubblock, but copies only diagonal entries specified by `pattern`.
   * At the destination, 'nnz_to_copy` nonzeros starting from index `dest_nnz_st` will be replaced.
   * @pre The added entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
   * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
   * @pre 'pattern' has same size as `x`.
   * @pre 'pattern` has exactly `nnz_to_copy` nonzeros.
   */
  virtual void copyDiagMatrixToSubblock_w_pattern(const hiopVector& x,
                                                  const index_type& dest_row_st,
                                                  const index_type& dest_col_st,
                                                  const size_type& dest_nnz_st,
                                                  const size_type& nnz_to_copy,
                                                  const hiopVector& pattern);

  virtual double max_abs_value();

  virtual void row_max_abs_value(hiopVector& ret_vec);

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale = false);

  virtual bool isfinite() const;

  virtual void print(FILE* f = nullptr, const char* msg = nullptr, int maxRows = -1, int maxCols = -1, int rank = -1) const;

  virtual void startingAtAddSubDiagonalToStartingAt(index_type diag_src_start,
                                                    const double& alpha,
                                                    hiopVector& vec_dest,
                                                    index_type vec_start,
                                                    size_type num_elems = -1) const
  {
    assert(0 && "not implemented; should be used only for symmetric matrices.");
  }

  virtual size_type numberOfOffDiagNonzeros() const
  {
    assert(false && "not yet implemented");
    return 0;
  }

  virtual bool is_diagonal() const
  {
    assert(false && "not yet implemented");
    return false;
  }

  /// @brief extend base problem Jac to the Jac in feasibility problem
  virtual void set_Jac_FR(const hiopMatrixSparse& Jac_c,
                          const hiopMatrixSparse& Jac_d,
                          index_type* iJacS,
                          index_type* jJacS,
                          double* MJacS)
  {
    assert(false && "not yet implemented");
  }

  /// @brief extend base problem Hess to the Hess in feasibility problem
  virtual void set_Hess_FR(const hiopMatrixSparse& Hess,
                           index_type* iHSS,
                           index_type* jHSS,
                           double* MHSS,
                           const hiopVector& add_diag)
  {
    assert(false && "not yet implemented");
  }

  virtual hiopMatrixSparse* alloc_clone() const;
  virtual hiopMatrixSparse* new_copy() const;

  inline index_type* i_row() { return irowptr_; }
  inline index_type* j_col() { return jcolind_; }
  inline double* M() { return values_; }
  inline const index_type* i_row() const { return irowptr_; }
  inline const index_type* j_col() const { return jcolind_; }
  inline const double* M() const { return values_; }

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const
  {
    assert(false && "not yet implemented");
    return false;
  }
  virtual bool checkIndexesAreOrdered() const { return true; }
#endif

  /**
   * @brief Extracts the diagonal entries of `this` matrix into the vector passed as argument
   *
   * @pre `this` matrix needs to be symmetric and of same size(s) as `diag_out`
   */
  virtual void extract_diagonal(hiopVector& diag_out) const { assert(false && "wip"); }

  /**
   * Sets the diagonal of `this` to the constant `val`. If `val` is zero, the sparsity pattern
   * of `this` is not altered.
   *
   * @pre  `this` is expected to store the diagonal entries as nonzero elements.
   */
  virtual void set_diagonal(const double& val);

  /**
   * Allocates a CSR matrix capable of storing the multiplication result of M = X*Y, where X
   * is the calling matrix class (`this`) and Y is the `Y` argument of the method.
   *
   * @note Should be used in conjunction with `times_mat_symbolic` and `times_mat_numeric`
   *
   * @pre The dimensions of the matrices should be consistent with the multiplication.
   *
   */
  hiopMatrixSparseCSR* times_mat_alloc(const hiopMatrixSparseCSR& Y) const;

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
  void times_mat_symbolic(hiopMatrixSparseCSR& M, const hiopMatrixSparseCSR& Y) const;

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
  void times_mat_numeric(double beta, hiopMatrixSparseCSR& M, double alpha, const hiopMatrixSparseCSR& Y);

  /// @brief Column scaling or right multiplication by a diagonal: `this`=`this`*D
  void scale_cols(const hiopVector& D);

  /// @brief Row scaling or left multiplication by a diagonal: `this`=D*`this`
  void scale_rows(const hiopVector& D);

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of the triplet matrix `M`.
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  //// note: cusparseXcoo2csr + on the device cudamemcpy
  void form_from_symbolic(const hiopMatrixSparseTriplet& M);

  /**
   * Copies the numerical values of the triplet matrix M into the CSR matrix `this`
   *
   * @pre The sparsity pattern (row pointers and column indexes arrays) of `this` should be
   * allocated and populated, possibly by a previous call to `form_from_symbolic`
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  //// note: only device cuda memcpy
  void form_from_numeric(const hiopMatrixSparseTriplet& M);

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of transpose of the triplet matrix `M`.
   *
   * @pre The input argument should have the nonzeros sorted by row and then by column
   * indexes.
   */
  //// note: cusparseCsr2cscEx2
  void form_transpose_from_symbolic(const hiopMatrixSparseTriplet& M);

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
  void form_transpose_from_numeric(const hiopMatrixSparseTriplet& M);

  /**
   * Allocates and populates the sparsity pattern of `this` as the CSR representation
   * of transpose of the CSR matrix `M`.
   *
   * @pre The input argument should have the column indexes sorted and unique within a row.
   */
  virtual void form_transpose_from_symbolic(const hiopMatrixSparseCSR& M);

  /**
   * Copies the numerical values of the transpose of the CSR matrix M into the CSR matrix `this`.
   *
   * @pre The sparsity pattern (row pointers and column indexes arrays) of `this` should be
   * allocated and populated, possibly by a previous call to `form_transpose_from_symbolic`
   *
   * @pre The input argument should have the column indexes sorted and unique within a row.
   */
  virtual void form_transpose_from_numeric(const hiopMatrixSparseCSR& M);

  /**
   * (Re)Initializes `this` to a diagonal matrix with diagonal entries given by D.
   */
  void form_diag_from_symbolic(const hiopVector& D);

  /**
   * Sets the diagonal entries of `this` equal to entries of D
   *
   * @pre Length of `D` should be equal to size(s) of `this`
   *
   * @pre `this` should be a diagonal matrix (in CSR format) with storage for
   * all the diagonal entries, which can be ensured by calling the sister method
   * `form_diag_from_symbolic`
   */
  void form_diag_from_numeric(const hiopVector& D);

  /**
   * Allocates and returns CSR matrix `M` capable of holding M = X+Y, where X is
   * the calling matrix class (`this`) and Y is the argument passed to the method.
   */
  hiopMatrixSparseCSR* add_matrix_alloc(const hiopMatrixSparseCSR& Y) const;

  /**
   * Computes sparsity pattern of M = X+Y (i.e., populates the row pointers and
   * column indexes arrays) of `M`. `X` is this.
   *
   * @pre `this` and `Y` should hold matrices of identical dimensions.
   *
   */
  void add_matrix_symbolic(hiopMatrixSparseCSR& M, const hiopMatrixSparseCSR& Y) const;

  /**
   * Performs matrix addition M = gamma*M + alpha*X + beta*Y numerically, where
   * X is `this` and alpha and beta are scalars.
   *
   * @pre `M`, `this` and `Y` should hold matrices of identical dimensions.
   *
   * @pre `M` and `X+Y` should have identical sparsity pattern, namely the
   * `add_matrix_symbolic` should have been called previously.
   *
   */
  void add_matrix_numeric(hiopMatrixSparseCSR& M, double alpha, const hiopMatrixSparseCSR& Y, double beta) const;

  /// @brief Performs a quick check and returns false if the CSR indexes are not ordered
  bool check_csr_is_ordered();
  /////////////////////////////////////////////////////////////////////
  // end of new CSR-specific methods
  /////////////////////////////////////////////////////////////////////

private:
  void alloc();
  void dealloc();

protected:
  friend class hiopMatrixSparseCSRCUDA;
  ExecSpace<MemBackendCpp, ExecPolicySeq> exec_space_;

  //// inherits nrows_, ncols_, and nnz_ from  hiopSparseMatrix

  /// Row pointers (starting indexes) in the column and values arrays
  index_type* irowptr_;

  /// Column indexes
  index_type* jcolind_;

  /// Nonzero values
  double* values_;

  /// Working buffer in the size of columns, allocated on demand and reused by some methods
  double* buf_col_;

  /**
   * Storage for the row starts used by `form_transpose_from_xxx` methods (allocated on
   * demand, only the above mentioned methods are called)
   */
  index_type* row_starts_;

private:
  hiopMatrixSparseCSRSeq(const hiopMatrixSparseCSRSeq&) = delete;
};

}  // namespace hiop

#endif
