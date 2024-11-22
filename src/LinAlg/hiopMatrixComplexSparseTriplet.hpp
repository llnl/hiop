#ifndef HIOP_SPARSE_COMPLEX_MATRIX
#define HIOP_SPARSE_COMPLEX_MATRIX

#include "hiopMatrix.hpp"
#include "hiopMatrixSparseTripletStorage.hpp"

namespace hiop
{

/** Sparse matrix of complex numbers in triplet format - it is not distributed
 *
 * Note: most methods expect (i,j) ordered: first on rows 'i' and then on cols 'j'. The
 * class hiopMatrixSparseTripletStorage offers this functionality.
 *
 * Existing limitations: this class is mostly used as storage both for symmetric and
 * rectangular matrices. Some of the ("not yet implemented") methods are ambiguous
 * or simply cannot be implemented without i. having this class specialized for
 * rectangular matrices and ii. derive a new specialization for symmetric matrices.
 */
/*
  Note: the following methods of hiopMatrix are NOT
  implemented in this class:
  - copyRowsFrom
  - timesVec
  - transTimesVec
  - timesMat
  - transTimesMat
  - timesMatTrans
  - addDiagonal (both overloads)
  - addSubDiagonal (all three overloads)
  - addMatrix
  - addToSymDenseMatrixUpperTriangle
  - transAddToSymDenseMatrixUpperTriangle
  - addUpperTriangleToSymDenseMatrixUpperTriangle
  - isfinite
  - assertSymmetry
*/
class hiopMatrixComplexSparseTriplet : public hiopMatrix
{
public:
  hiopMatrixComplexSparseTriplet(int rows, int cols, int nnz);
  virtual ~hiopMatrixComplexSparseTriplet();

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void setToConstant(std::complex<double> c);

  void copyRowsFrom(const hiopMatrix& src, const index_type* rows_idxs, size_type n_rows)
  {
    assert(false && "not yet implemented");
  }

  /** y = beta * y + alpha * this * x */
  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(false && "not yet implemented");
  }
  /** y = beta * y + alpha * this * x
   */
  virtual void timesVec(double beta, std::complex<double>* y, double alpha, const std::complex<double>* x) const;

  /** y = beta * y + alpha * this^T * x */
  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(false && "not yet implemented");
  }

  /* W = beta*W + alpha*this*X */
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  /* W = beta*W + alpha*this^T*X
   *
   * Only supports W and X of the type 'hiopMatrixComplexDense'
   */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  /* W = beta*W + alpha*this*X^T */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  /* this += alpha * (sub)diag */
  virtual void addDiagonal(const double& alpha, const hiopVector& d_) { assert(false && "not yet implemented"); }

  virtual void addDiagonal(const double& value) { assert(false && "not yet implemented"); }

  virtual void addSubDiagonal(const double& alpha, index_type start_on_dest_diag, const hiopVector& d_)
  {
    assert(false && "not yet implemented");
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
    assert(false && "not yet implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not yet implemented");
  }

  /* this += alpha*X */
  virtual void addMatrix(double alpha, const hiopMatrix& X) { assert(false && "not yet implemented"); }

  /* block of W += alpha*this
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions:
   *  1. 'this' has to fit in the upper triangle of W
   *  2. W.n() == W.m()
   */
  virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                int col_dest_start,
                                                double alpha,
                                                hiopMatrixDense& W) const
  {
    assert(false && "not yet implemented");
  }

  /* block of W += alpha*transpose(this)
   * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK
   * Preconditions:
   *  1. transpose of 'this' has to fit in the upper triangle of W
   *  2. W.n() == W.m()
   *
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start,
                                                     int col_dest_start,
                                                     double alpha,
                                                     hiopMatrixDense& W) const
  {
    assert(false && "not yet implemented");
  }

  /* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
   * 'this' should start to contribute.
   *
   * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
   * and only the upper triangle of 'this' is accessed
   *
   * Preconditions:
   *  1. this->n()==this-m()
   *  2. W.n() == W.m()
   */
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not yet implemented");
  }

  virtual double max_abs_value();

  virtual void row_max_abs_value(hiopVector& ret_vec) { assert(0 && "not yet"); }

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale) { assert(0 && "not yet"); }

  /* return false is any of the entry is a nan, inf, or denormalized */
  virtual bool isfinite() const
  {
    assert(false && "not yet implemented");
    return false;
  }

  /* call with -1 to print all rows, all columns, or on all ranks; otherwise will
   *  will print the first rows and/or columns on the specified rank.
   *
   * If the underlying matrix is sparse, maxCols is ignored and a max number elements
   * given by the value of 'maxRows' will be printed. If this value is negative, all
   * elements will be printed.
   */
  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const;

  /* number of rows */
  virtual size_type m() const { return stM->m(); }
  /* number of columns */
  virtual size_type n() const { return stM->n(); }

#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  virtual bool assertSymmetry(double tol = 1e-16) const
  {
    assert(false && "not yet implemented");
    return false;
  }
#endif
  // these are not part of the hiopMatrix

  // Builds/extracts submatrix nrows x ncols with rows and cols specified by row_idxs and cols_idx
  // Assumes
  //  - 'this' is unsymmetric
  //  - 'row_idxs' and 'col_idxs' are ordered
  hiopMatrixComplexSparseTriplet* new_slice(const int* row_idxs, int nrows, const int* col_idxs, int ncols) const;

  // Builds/extracts submatrix nrows x ncols with rows and cols specified by row_idxs and cols_idx
  // Assumes
  //  - 'this' is symmetric (only upper triangle is stored)
  //  - 'row_idxs' and 'col_idxs' are ordered
  hiopMatrixComplexSparseTriplet* new_sliceFromSym(const int* row_idxs, int nrows, const int* col_idxs, int ncols) const;

  // Extracts a symmetric matrix (for which only the upper triangle is stored)
  // Assumes
  //  - 'this' is symmetric (only upper triangle is stored)
  //  - 'row_col_idxs' is ordered
  hiopMatrixComplexSparseTriplet* new_sliceFromSymToSym(const int* row_col_idxs, int ndim) const;

  inline void copyFrom(const int* irow_, const int* jcol_, const std::complex<double>* values_)
  {
    stM->copyFrom(irow_, jcol_, values_);
  }
  inline size_type numberOfNonzeros() const { return stM->numberOfNonzeros(); }
  inline hiopMatrixSparseTripletStorage<int, std::complex<double> >* storage() const { return stM; }

private:
  hiopMatrixSparseTripletStorage<int, std::complex<double> >* stM;
};
}  // namespace hiop
#endif
