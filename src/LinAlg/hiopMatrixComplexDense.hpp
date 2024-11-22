#ifndef HIOP_MATRIX_DENSE_COMPLEX
#define HIOP_MATRIX_DENSE_COMPLEX

#include <hiopMPI.hpp>
#include "hiopMatrix.hpp"

#include <cassert>
#include <complex>

#include "hiopMatrixComplexSparseTriplet.hpp"

namespace hiop
{
/*
Note: the following methods of hiopMatrix are NOT
implemented in this class:
  - timesVec (both overloads)
  - transTimesVec (both overloads)
  - timesMat
  - timesMat_local
  - transTimesMat
  - timesMatTran
  - timesMatTran_local
  - addDiagonal (both overloads)
  - addSubDiagonal (all three overloads)
  - transAddToSymDenseMatrixUpperTriangle
  - addUpperTriangleToSymDenseMatrixUpperTriangle
  - copyRowsFrom
  - copyBlockFromMatrix
  - copyFromMatrixBlock
*/
class hiopMatrixComplexDense : public hiopMatrix
{
public:
  hiopMatrixComplexDense(const size_type& m,
                         const size_type& glob_n,
                         index_type* col_part = NULL,
                         MPI_Comm comm = MPI_COMM_SELF,
                         const size_type& m_max_alloc = -1);
  virtual ~hiopMatrixComplexDense();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void setToConstant(std::complex<double>& c);
  virtual void copyFrom(const hiopMatrixComplexDense& dm);
  virtual void copyFrom(const std::complex<double>* buffer);

  virtual void negate();

  /* Copy 'n_rows' rows specified by 'rows_idxs' (array of size 'n_rows') from 'src' to 'this'
   *
   * Preconditions
   * 1. 'this' has exactly 'n_rows' rows
   * 2. 'src' and 'this' must have same number of columns
   * 3. number of rows in 'src' must be at least the number of rows in 'this'
   */
  void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows);

  virtual void timesVec(std::complex<double> beta,
                        std::complex<double>* y,
                        std::complex<double> alpha,
                        const std::complex<double>* x) const;

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(false && "not yet supported");
  }
  /* same as above for mostly for internal use - avoid using it */
  virtual void timesVec(double beta, double* y, double alpha, const double* x) const
  {
    assert(false && "not yet supported");
  }

  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(false && "not yet supported");
  }
  /* same as above for mostly for internal use - avoid using it */
  virtual void transTimesVec(double beta, double* y, double alpha, const double* x) const
  {
    assert(false && "not yet supported");
  }

  // All methods taking an arguments 'hiopMatrix' will dynamic_cast the argument to
  // 'complex' dense matrix (this class). Specialized multiplications with sparse matrices
  // are to be done by the sparse matrix classes. Multiplications with double dense matrices
  // are to be determined.

  /* W = beta*W + alpha*this*X */
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }
  virtual void timesMat_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  // to be used only locally
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }
  // to be used only locally
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }
  virtual void timesMatTrans_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }
  virtual void addDiagonal(const double& alpha, const hiopVector& d_) { assert(false && "not yet supported"); }
  virtual void addDiagonal(const double& value) { assert(false && "not yet supported"); }
  virtual void addSubDiagonal(const double& alpha, index_type start_on_dest_diag, const hiopVector& d_)
  {
    assert(false && "not yet supported");
  }
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha,
                              const hiopVector& d_,
                              int start_on_src_vec,
                              int num_elems = -1)
  {
    assert(false && "not yet supported");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not yet supported");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X);
  virtual void addMatrix(const std::complex<double>& alpha, const hiopMatrixComplexDense& X);

  /* this = this + alpha*X
   * X is a general sparse matrix in triplet format (rows and cols indexes are assumed to be ordered)
   */
  void addSparseMatrix(const std::complex<double>& alpha, const hiopMatrixComplexSparseTriplet& X);

  /* uppertriangle(this) += alpha*uppertriangle(X)
   * where X is a sparse matrix stored in triplet format holding only upper triangle elements */
  void addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(const std::complex<double>& alpha,
                                                              const hiopMatrixComplexSparseTriplet& X);

  /* block of W += alpha*transpose(this)
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions:
   *  1. transpose of 'this' has to fit in the upper triangle of W
   *  2. W.n() == W.m()
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const
  {
    assert(false && "not supported");
  }

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   *
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   *
   * Preconditions:
   *  1. this->n()==this->m()
   *  2. W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not supported");
  }

  virtual double max_abs_value();

  virtual void row_max_abs_value(hiopVector& ret_vec) { assert(0 && "not yet"); }

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale) { assert(0 && "not yet"); }

  virtual bool isfinite() const;

  // virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const;

  //
  // below are methods specific to this class
  //
  virtual hiopMatrixComplexDense* alloc_clone() const;
  virtual hiopMatrixComplexDense* new_copy() const;

  /* copy 'num_rows' rows from 'src' in this starting at 'row_dest' */
  void copyRowsFrom(const hiopMatrixComplexDense& src, int num_rows, int row_dest)
  {
    assert(false && "not yet implemented");
  }
  /* copies 'src' into this as a block starting at (i_block_start,j_block_start) */
  /* copyMatrixAsBlock */
  void copyBlockFromMatrix(const index_type i_block_start, const index_type j_block_start, const hiopMatrixComplexDense& src)
  {
    assert(false && "not yet implemented");
  }

  /* overwrites 'this' with 'src''s block starting at (i_src_block_start,j_src_block_start)
   * and dimensions of this */
  void copyFromMatrixBlock(const hiopMatrixComplexDense& src, const int i_src_block_start, const int j_src_block_start)
  {
    assert(false && "not yet implemented");
  }

  inline size_type get_local_size_n() const { return n_local_; }
  inline size_type get_local_size_m() const { return m_local_; }

  // TODO: this is not kosher!
  inline std::complex<double>** local_data() const { return M; }
  inline std::complex<double>* local_buffer() const { return M[0]; }
  // do not use this unless you sure you know what you're doing
  inline std::complex<double>** get_M() { return M; }

  virtual size_type m() const { return m_local_; }
  virtual size_type n() const { return n_global_; }
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const;
#endif
private:
  std::complex<double>** M;  // local storage
  size_type n_global_;       // total / global number of columns
  int m_local_, n_local_;    // local number of rows and cols, respectively
  size_type glob_jl_, glob_ju_;
  MPI_Comm comm_;
  int myrank_;

  mutable std::complex<double>* buff_mxnlocal_;

  // this is very private do not touch :)
  size_type max_rows_;

private:
  hiopMatrixComplexDense() {};
  /** copy constructor, for internal/private use only (it doesn't copy the values) */
  hiopMatrixComplexDense(const hiopMatrixComplexDense&);

  inline std::complex<double>* new_mxnlocal_buff() const
  {
    if(buff_mxnlocal_ == NULL) {
      buff_mxnlocal_ = new std::complex<double>[max_rows_ * n_local_];
    }
    return buff_mxnlocal_;
  }
};  // end class
}  // namespace hiop
#endif
