#ifndef HIOP_SPARSE_MATRIX_MDS
#define HIOP_SPARSE_MATRIX_MDS

#include "hiopVectorPar.hpp"
#include "hiopMatrixDenseRowMajor.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "LinAlgFactory.hpp"

#include <algorithm>

#include <cassert>

namespace hiop
{
/** Mixed Sparse-Dense blocks matrix  - it is not distributed
 *  M = [S D] where S is sparse and D is dense
 *  Note: the following methods of hiopMatrix are NOT
 *  implemented in this class:
 *  - timesMat
 *  - transTimesMat
 *  - timesMatTran
 *  - addDiagonal (both overloads)
 *  - addSubDiagonal (all three overloads)
 *  - addUpperTriangleToSymDenseMatrixUpperTriangle
 */
class hiopMatrixMDS : public hiopMatrix
{
public:
  hiopMatrixMDS(int rows, int cols_sparse, int cols_dense, int nnz_sparse, const std::string& mem_space)
  {
    mSp = LinearAlgebraFactory::create_matrix_sparse(mem_space, rows, cols_sparse, nnz_sparse);
    mDe = LinearAlgebraFactory::create_matrix_dense(mem_space, rows, cols_dense);
  }
  virtual ~hiopMatrixMDS()
  {
    delete mDe;
    delete mSp;
  }

  virtual void setToZero()
  {
    mSp->setToZero();
    mDe->setToZero();
  }
  virtual void setToConstant(double c)
  {
    mSp->setToConstant(c);
    mDe->setToConstant(c);
  }

  /**
   * @note should this method be called, an assertion will be thrown in
   * hiopMatrixSparseTriplet if that is the relevant implementation.
   */
  virtual void copyFrom(const hiopMatrixMDS& m)
  {
    mSp->copyFrom(*m.mSp);
    mDe->copyFrom(*m.mDe);
  }

  virtual void copyRowsFrom(const hiopMatrix& src_in, const index_type* rows_idxs, size_type n_rows)
  {
    const hiopMatrixMDS& src = dynamic_cast<const hiopMatrixMDS&>(src_in);
    mSp->copyRowsFrom(*src.mSp, rows_idxs, n_rows);
    mDe->copyRowsFrom(*src.mDe, rows_idxs, n_rows);
  }

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(x.get_size() == mSp->n() + mDe->n());
    mSp->timesVec(beta, y.local_data(), alpha, x.local_data_const());
    mDe->timesVec(1., y.local_data(), alpha, x.local_data_const() + mSp->n());
  }
  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(y.get_size() == mSp->n() + mDe->n());
    mSp->transTimesVec(beta, y.local_data(), alpha, x.local_data_const());
    mDe->transTimesVec(beta, y.local_data() + mSp->n(), alpha, x.local_data_const());
  }

  /* W = beta*W + alpha*this*X */
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    const hiopMatrixMDS& X_mds = dynamic_cast<const hiopMatrixMDS&>(X);

    mDe->timesMatTrans(beta, W, 1.0, *X_mds.de_mat());
    mSp->timesMatTrans(1.0, W, 1.0, *X_mds.sp_mat());
  }

  virtual void addDiagonal(const double& alpha, const hiopVector& d_) { assert(false && "not supported"); }
  virtual void addDiagonal(const double& value) { assert(false && "not supported"); }
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
  {
    assert(false && "not supported");
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
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X)
  {
    const hiopMatrixMDS* pX = dynamic_cast<const hiopMatrixMDS*>(&X);
    if(pX == NULL) {
      assert(false && "operation only supported for hiopMatrixMDS left operand");
    }
    mSp->addMatrix(alpha, *pX->mSp);
    mDe->addMatrix(alpha, *pX->mDe);
  }

  /* block of W += alpha*this */
  // virtual void addToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  // {
  //   mSp->addToSymDenseMatrixUpperTriangle(row_start, col_start, alpha, W);
  //   mDe->addToSymDenseMatrixUpperTriangle(row_start, col_start+mSp->n(), alpha, W);
  // } aaa

  /* block of W += alpha*this' */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    mSp->transAddToSymDenseMatrixUpperTriangle(row_start, col_start, alpha, W);
    mDe->transAddToSymDenseMatrixUpperTriangle(row_start + mSp->n(), col_start, alpha, W);
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
    assert(false && "not needed for general/nonsymmetric matrices.");
  }

  virtual double max_abs_value() { return std::max(mSp->max_abs_value(), mDe->max_abs_value()); }

  virtual void row_max_abs_value(hiopVector& ret_vec)
  {
    auto ret_vec_dense = ret_vec.new_copy();

    mSp->row_max_abs_value(ret_vec);
    mDe->row_max_abs_value(*ret_vec_dense);

    ret_vec.component_max(*ret_vec_dense);

    delete ret_vec_dense;
  }

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale)
  {
    mSp->scale_row(vec_scal, inv_scale);
    mDe->scale_row(vec_scal, inv_scale);
  }

  virtual bool isfinite() const { return mSp->isfinite() && mDe->isfinite(); }

  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const
  {
    mSp->print(f, msg, maxRows, maxCols, rank);
    mDe->print(f, msg, maxRows, maxCols, rank);
  }

  virtual hiopMatrix* alloc_clone() const
  {
    hiopMatrixMDS* m = new hiopMatrixMDS();
    assert(m->mSp == NULL);
    assert(m->mDe == NULL);
    m->mSp = mSp->alloc_clone();
    m->mDe = mDe->alloc_clone();
    assert(m->mSp != NULL);
    assert(m->mDe != NULL);
    return m;
  }
  virtual hiopMatrix* new_copy() const
  {
    hiopMatrixMDS* m = new hiopMatrixMDS();
    assert(m->mSp == NULL);
    assert(m->mDe == NULL);
    m->mSp = mSp->new_copy();
    m->mDe = mDe->new_copy();
    assert(m->mSp != NULL);
    assert(m->mDe != NULL);
    return m;
  }

  virtual inline size_type m() const { return mSp->m(); }
  virtual inline size_type n() const { return mSp->n() + mDe->n(); }
  inline size_type n_sp() const { return mSp->n(); }
  inline size_type n_de() const { return mDe->n(); }

  inline const hiopMatrixSparse* sp_mat() const { return mSp; }
  inline const hiopMatrixDense* de_mat() const { return mDe; }

  inline int sp_nnz() const { return mSp->numberOfNonzeros(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double* de_local_data() { return mDe->local_data(); }

  /// @brief extend base problem Jac to the Jac in feasibility problem
  virtual void set_Jac_FR(const hiopMatrixMDS& Jac_c,
                          const hiopMatrixMDS& Jac_d,
                          int* iJacS,
                          int* jJacS,
                          double* MJacS,
                          double* JacD);

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const { return false; }
#endif
private:
  hiopMatrixSparse* mSp;
  hiopMatrixDense* mDe;

private:
  hiopMatrixMDS()
      : mSp(NULL),
        mDe(NULL) {};
  hiopMatrixMDS(const hiopMatrixMDS&) {};
};

/*
 * Note: the following methods of hiopMatrix are NOT
 * implemented in this class:
 * - timesMat
 * - transTimesMat
 * - timesMatTran
 * - addDiagonal (both overloads)
 * - addSubDiagonal (all three overloads)
 * - transAddToSymDenseMatrixUpperTriangle
 */
class hiopMatrixSymBlockDiagMDS : public hiopMatrix
{
public:
  hiopMatrixSymBlockDiagMDS(int n_sparse, int n_dense, int nnz_sparse, const std::string& mem_space)
  {
    mSp = LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, n_sparse, nnz_sparse);
    mDe = LinearAlgebraFactory::create_matrix_dense(mem_space, n_dense, n_dense);
  }
  virtual ~hiopMatrixSymBlockDiagMDS()
  {
    delete mDe;
    delete mSp;
  }

  virtual void setToZero()
  {
    mSp->setToZero();
    mDe->setToZero();
  }
  virtual void setToConstant(double c)
  {
    mSp->setToConstant(c);
    mDe->setToConstant(c);
  }
  virtual void copyFrom(const hiopMatrixSymBlockDiagMDS& m)
  {
    mSp->copyFrom(*m.mSp);
    mDe->copyFrom(*m.mDe);
  }

  virtual void copyRowsFrom(const hiopMatrix& src_in, const index_type* rows_idxs, size_type n_rows)
  {
    const hiopMatrixSymBlockDiagMDS& src = dynamic_cast<const hiopMatrixSymBlockDiagMDS&>(src_in);
    mSp->copyRowsFrom(src, rows_idxs, n_rows);
    mDe->copyRowsFrom(src, rows_idxs, n_rows);
  }

  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    assert(x.get_size() == mSp->n() + mDe->n());
    assert(y.get_size() == mSp->n() + mDe->n());

    mSp->timesVec(beta, y.local_data(), alpha, x.local_data_const());
    mDe->timesVec(beta, y.local_data() + mSp->n(), alpha, x.local_data_const() + mSp->n());
  }
  virtual void transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
  {
    timesVec(beta, y, alpha, x);
  }

  /* W = beta*W + alpha*this*X */
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not yet implemented");
  }

  virtual void addDiagonal(const double& alpha, const hiopVector& d_) { assert(false && "not supported"); }
  virtual void addDiagonal(const double& value) { assert(false && "not supported"); }
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
  {
    assert(false && "not supported");
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
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X_in)
  {
    const hiopMatrixSymBlockDiagMDS& X = dynamic_cast<const hiopMatrixSymBlockDiagMDS&>(X_in);
    mSp->addMatrix(alpha, *X.mSp);
    mDe->addMatrix(alpha, *X.mDe);
  }

  /**
   *  block of W += alpha*this
   *
   * @warning This method should never be called/is never needed for symmetric matrixes.
   * Use addUpperTriangleToSymDenseMatrixUpperTriangle instead.
   */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(0 && "This should not be called for MDS symmetric matrices.");
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
    assert(mSp->m() == mSp->n());
    mSp->addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start, alpha, W);
    mDe->addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start + mSp->m(), alpha, W);
  }

  virtual double max_abs_value() { return std::max(mSp->max_abs_value(), mDe->max_abs_value()); }

  virtual void row_max_abs_value(hiopVector& ret_vec)
  {
    auto ret_vec_dense = ret_vec.new_copy();

    mSp->row_max_abs_value(ret_vec);
    mDe->row_max_abs_value(*ret_vec_dense);

    ret_vec.component_max(*ret_vec_dense);

    delete ret_vec_dense;
  }

  virtual void scale_row(hiopVector& vec_scal, const bool inv_scale)
  {
    mSp->scale_row(vec_scal, inv_scale);
    mDe->scale_row(vec_scal, inv_scale);
  }

  virtual bool isfinite() const { return mSp->isfinite() && mDe->isfinite(); }

  virtual void print(FILE* f = NULL, const char* msg = NULL, int maxRows = -1, int maxCols = -1, int rank = -1) const
  {
    mSp->print(f, msg, maxRows, maxCols, rank);
    mDe->print(f, msg, maxRows, maxCols, rank);
  }

  virtual hiopMatrix* alloc_clone() const
  {
    hiopMatrixSymBlockDiagMDS* m = new hiopMatrixSymBlockDiagMDS();
    assert(m->mSp == NULL);
    assert(m->mDe == NULL);
    m->mSp = mSp->alloc_clone();
    m->mDe = mDe->alloc_clone();
    assert(m->mSp != NULL);
    assert(m->mDe != NULL);
    return m;
  }
  virtual hiopMatrix* new_copy() const
  {
    hiopMatrixSymBlockDiagMDS* m = new hiopMatrixSymBlockDiagMDS();
    assert(m->mSp == NULL);
    assert(m->mDe == NULL);
    m->mSp = mSp->new_copy();
    m->mDe = mDe->new_copy();
    assert(m->mSp != NULL);
    assert(m->mDe != NULL);
    return m;
  }

  virtual inline size_type m() const { return n(); }
  virtual inline size_type n() const { return mSp->n() + mDe->n(); }
  inline size_type n_sp() const { return mSp->n(); }
  inline size_type n_de() const { return mDe->n(); }

  inline const hiopMatrixSparse* sp_mat() const { return mSp; }
  inline const hiopMatrixDense* de_mat() const { return mDe; }

  inline int sp_nnz() const { return mSp->numberOfNonzeros(); }
  inline int* sp_irow() { return mSp->i_row(); }
  inline int* sp_jcol() { return mSp->j_col(); }
  inline double* sp_M() { return mSp->M(); }
  inline double* de_local_data() { return mDe->local_data(); }

  /// @brief build Hess for FR problem, from the base problem `Hess`.
  virtual void set_Hess_FR(const hiopMatrixSymBlockDiagMDS& Hess,
                           int* iHSS,
                           int* jHSS,
                           double* MHSS,
                           double* MHDD,
                           const hiopVector& add_diag_sp,
                           const hiopVector& add_diag_de);

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol = 1e-16) const
  {
    if(mSp->assertSymmetry(tol))
      return mDe->assertSymmetry(tol);
    else
      return false;
  }
#endif
private:
  hiopMatrixSparse* mSp;  ///< Symmetric sparse matrix
  hiopMatrixDense* mDe;   ///< Row-major dense matrix
private:
  hiopMatrixSymBlockDiagMDS()
      : mSp(NULL),
        mDe(NULL) {};
  hiopMatrixSymBlockDiagMDS(const hiopMatrixMDS&) {};
};

}  // namespace hiop

#endif
