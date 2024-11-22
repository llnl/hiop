// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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

#include "hiopMatrixDenseRowMajor.hpp"

#include <cstdio>
#include <cstring>  //for memcpy
#include <cmath>
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include "hiopVectorPar.hpp"

namespace hiop
{

hiopMatrixDenseRowMajor::hiopMatrixDenseRowMajor(const size_type& m,
                                                 const size_type& glob_n,
                                                 index_type* col_part /*=NULL*/,
                                                 MPI_Comm comm /*=MPI_COMM_SELF*/,
                                                 const size_type& m_max_alloc /*=-1*/)
    : hiopMatrixDense(m, glob_n, comm)
{
  int P = 0;
  comm_size_ = 1;
  if(col_part) {
#ifdef HIOP_USE_MPI
    int ierr = MPI_Comm_rank(comm_, &P);
    assert(ierr == MPI_SUCCESS);
    ierr = MPI_Comm_size(comm_, &comm_size_);
    assert(MPI_SUCCESS == ierr);
#endif
    glob_jl_ = col_part[P];
    glob_ju_ = col_part[P + 1];
  } else {
    glob_jl_ = 0;
    glob_ju_ = n_global_;
  }
  n_local_ = glob_ju_ - glob_jl_;

  myrank_ = P;

  max_rows_ = m_max_alloc;
  if(max_rows_ == -1) max_rows_ = m_local_;
  assert(max_rows_ >= m_local_ && "the requested extra allocation is smaller than the allocation needed by the matrix");

  M_ = new double*[max_rows_ == 0 ? 1 : max_rows_];
  M_[0] = max_rows_ == 0 ? NULL : new double[max_rows_ * n_local_];
  for(int i = 1; i < max_rows_; i++) M_[i] = M_[0] + i * n_local_;

  //! valgrind reports a shit load of errors without this; check this
  for(int i = 0; i < max_rows_ * n_local_; i++) M_[0][i] = 0.0;

  // internal buffers
  buff_mxnlocal_ = NULL;  // new double[max_rows_*n_local_];
}
hiopMatrixDenseRowMajor::~hiopMatrixDenseRowMajor()
{
  if(buff_mxnlocal_) delete[] buff_mxnlocal_;
  if(M_) {
    if(M_[0]) delete[] M_[0];
    delete[] M_;
  }
}

/// TODO: check again
hiopMatrixDenseRowMajor::hiopMatrixDenseRowMajor(const hiopMatrixDenseRowMajor& dm)
{
  n_local_ = dm.n_local_;
  m_local_ = dm.m_local_;
  n_global_ = dm.n_global_;
  glob_jl_ = dm.glob_jl_;
  glob_ju_ = dm.glob_ju_;
  comm_ = dm.comm_;
  myrank_ = dm.myrank_;

  // M=new double*[m_local_==0?1:m_local_];
  max_rows_ = dm.max_rows_;
  M_ = new double*[max_rows_ == 0 ? 1 : max_rows_];
  // M[0] = m_local_==0?NULL:new double[m_local_*n_local_];
  M_[0] = max_rows_ == 0 ? NULL : new double[max_rows_ * n_local_];
  // for(int i=1; i<m_local_; i++)
  for(int i = 1; i < max_rows_; i++) M_[i] = M_[0] + i * n_local_;

  buff_mxnlocal_ = NULL;
}

void hiopMatrixDenseRowMajor::appendRow(const hiopVector& row)
{
#ifdef HIOP_DEEPCHECKS
  assert(row.get_local_size() == n_local_);
  assert(m_local_ < max_rows_ && "no more space to append rows ... should have preallocated more rows.");
#endif
  memcpy(M_[m_local_], row.local_data_const(), n_local_ * sizeof(double));
  m_local_++;
}

void hiopMatrixDenseRowMajor::copyFrom(const hiopMatrixDense& dmmat)
{
  const auto& dm = dynamic_cast<const hiopMatrixDenseRowMajor&>(dmmat);
  assert(n_local_ == dm.n_local_);
  assert(m_local_ == dm.m_local_);
  assert(n_global_ == dm.n_global_);
  assert(glob_jl_ == dm.glob_jl_);
  assert(glob_ju_ == dm.glob_ju_);
  if(NULL == dm.M_[0]) {
    M_[0] = NULL;
  } else {
    memcpy(M_[0], dm.M_[0], m_local_ * n_local_ * sizeof(double));
  }
}

void hiopMatrixDenseRowMajor::copyFrom(const double* buffer)
{
  if(NULL == buffer) {
    M_[0] = NULL;
  } else {
    memcpy(M_[0], buffer, m_local_ * n_local_ * sizeof(double));
  }
}

void hiopMatrixDenseRowMajor::copy_to(double* buffer)
{
  if(NULL == buffer) {
    return;
  } else {
    memcpy(buffer, M_[0], m_local_ * n_local_ * sizeof(double));
  }
}

void hiopMatrixDenseRowMajor::copyRowsFrom(const hiopMatrixDense& srcmat, int num_rows, int row_dest)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRowMajor&>(srcmat);
#ifdef HIOP_DEEPCHECKS
  assert(row_dest >= 0);
  assert(n_local_ == src.n_local_);
  assert(row_dest + num_rows <= m_local_);
  assert(num_rows <= src.m_local_);
#endif
  if(num_rows > 0) memcpy(M_[row_dest], src.M_[0], n_local_ * num_rows * sizeof(double));
}

void hiopMatrixDenseRowMajor::copyRowsFrom(const hiopMatrix& src_gen, const index_type* rows_idxs, size_type n_rows)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRowMajor&>(src_gen);
  assert(n_global_ == src.n_global_);
  assert(n_local_ == src.n_local_);
  assert(n_rows <= src.m_local_);
  assert(n_rows == m_local_);

  // todo //! opt -> copy multiple (consecutive rows at the time -> maybe keep blocks of eq and ineq,
  // instead of indexes)

  // int i should suffice for dense matrices
  for(int i = 0; i < n_rows; ++i) {
    memcpy(M_[i], src.M_[rows_idxs[i]], n_local_ * sizeof(double));
  }
}

void hiopMatrixDenseRowMajor::copyBlockFromMatrix(const index_type i_start,
                                                  const index_type j_start,
                                                  const hiopMatrixDense& srcmat)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRowMajor&>(srcmat);
  assert(n_local_ == n_global_ && "this method should be used only in 'serial' mode");
  assert(src.n_local_ == src.n_global_ && "this method should be used only in 'serial' mode");
  assert(m_local_ >= i_start + src.m_local_ && "the matrix does not fit as a sublock in 'this' at specified coordinates");
  assert(n_local_ >= j_start + src.n_local_ && "the matrix does not fit as a sublock in 'this' at specified coordinates");

  // quick returns for empty source matrices
  if(src.n() == 0) return;
  if(src.m() == 0) return;
#ifdef HIOP_DEEPCHECKS
  assert(i_start < m_local_ || !m_local_);
  assert(j_start < n_local_ || !n_local_);
  assert(i_start >= 0);
  assert(j_start >= 0);
#endif
  const size_t buffsize = src.n_local_ * sizeof(double);
  for(index_type ii = 0; ii < src.m_local_; ii++) memcpy(M_[ii + i_start] + j_start, src.M_[ii], buffsize);
}

void hiopMatrixDenseRowMajor::copyFromMatrixBlock(const hiopMatrixDense& srcmat, const int i_block, const int j_block)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRowMajor&>(srcmat);
  assert(n_local_ == n_global_ && "this method should be used only in 'serial' mode");
  assert(src.n_local_ == src.n_global_ && "this method should be used only in 'serial' mode");
  assert(m_local_ + i_block <= src.m_local_ && "the source does not enough rows to fill 'this'");
  assert(n_local_ + j_block <= src.n_local_ && "the source does not enough cols to fill 'this'");

  if(n_local_ == src.n_local_)  // and j_block=0
    memcpy(M_[0], src.M_[i_block], n_local_ * m_local_ * sizeof(double));
  else {
    for(int i = 0; i < m_local_; i++) memcpy(M_[i], src.M_[i + i_block] + j_block, n_local_ * sizeof(double));
  }
}

void hiopMatrixDenseRowMajor::shiftRows(size_type shift)
{
  if(shift == 0) return;
  if(std::abs(shift) == m_local_) return;  // nothing to shift
  if(m_local_ <= 1) return;                // nothing to shift

  assert(std::abs(shift) < m_local_);

  // at this point m_local_ should be >=2
  assert(m_local_ >= 2);
  // and
  assert(m_local_ - std::abs(shift) >= 1);
#ifdef HIOP_DEEPCHECKS
  double test1 = 8.3, test2 = -98.3;
  if(n_local_ > 0) {
    // not sure if memcpy is copying sequentially on all systems. we check this.
    // let's at least check it
    test1 = shift < 0 ? M_[-shift][0] : M_[m_local_ - shift - 1][0];
    test2 = shift < 0 ? M_[-shift][n_local_ - 1] : M_[m_local_ - shift - 1][n_local_ - 1];
  }
#endif

  // shift < 0 -> up; shift > 0 -> down
  // if(shift<0) memcpy(M[0], M[-shift], n_local_*(m_local_+shift)*sizeof(double));
  // else        memcpy(M[shift], M[0],  n_local_*(m_local_-shift)*sizeof(double));
  if(shift < 0) {
    for(int row = 0; row < m_local_ + shift; row++) memcpy(M_[row], M_[row - shift], n_local_ * sizeof(double));
  } else {
    for(int row = m_local_ - 1; row >= shift; row--) {
      memcpy(M_[row], M_[row - shift], n_local_ * sizeof(double));
    }
  }

#ifdef HIOP_DEEPCHECKS
  if(n_local_ > 0) {
    assert(test1 == M_[shift < 0 ? 0 : m_local_ - 1][0] &&
           "a different copy technique than memcpy is needed on this system");
    assert(test2 == M_[shift < 0 ? 0 : m_local_ - 1][n_local_ - 1] &&
           "a different copy technique than memcpy is needed on this system");
  }
#endif
}
void hiopMatrixDenseRowMajor::replaceRow(index_type row, const hiopVector& vec)
{
  assert(row >= 0);
  assert(row < m_local_);
  size_type vec_size = vec.get_local_size();
  memcpy(M_[row], vec.local_data_const(), (vec_size >= n_local_ ? n_local_ : vec_size) * sizeof(double));
}

void hiopMatrixDenseRowMajor::getRow(index_type irow, hiopVector& row_vec)
{
  assert(irow >= 0);
  assert(irow < m_local_);
  hiopVectorPar& vec = dynamic_cast<hiopVectorPar&>(row_vec);
  assert(n_local_ == vec.get_local_size());
  memcpy(vec.local_data(), M_[irow], n_local_ * sizeof(double));
}

void hiopMatrixDenseRowMajor::set_Hess_FR(const hiopMatrixDense& Hess, const hiopVector& add_diag_de)
{
  double one{1.0};
  copyFrom(Hess);
  addDiagonal(one, add_diag_de);
}

void hiopMatrixDenseRowMajor::set_Jac_FR(const hiopMatrixDense& Jac_c, const hiopMatrixDense& Jac_d)
{
  assert(Jac_c.n() == Jac_d.n());
  assert(Jac_c.get_local_size_n() == Jac_d.get_local_size_n());
  assert(Jac_c.m() + Jac_d.m() == this->m());
  assert(Jac_c.n() <= this->n());  // we will add variable p, n on the top of base prob

  const auto& Jeq = dynamic_cast<const hiopMatrixDenseRowMajor&>(Jac_c);
  const auto& Jin = dynamic_cast<const hiopMatrixDenseRowMajor&>(Jac_d);

  this->setToZero();

  if(myrank_ < comm_size_ - 1) {
    this->copyRowsFrom(Jeq, Jeq.m(), 0);
    this->copyRowsFrom(Jin, Jin.m(), Jeq.m());
  } else {
    // this is the last rank
    assert(Jeq.get_local_size_n() + 2 * Jeq.m() + 2 * Jin.m() == n_local_);

    // Jac for c(x) - p + n
    size_type m_eq_local_base = Jeq.m();
    size_type n_local_base = Jeq.get_local_size_n();

    for(int i = 0; i < m_eq_local_base; ++i) {
      memcpy(M_[i], Jeq.M_[i], n_local_base * sizeof(double));
      M_[i][n_local_base + i] = -1.0;
      M_[i][n_local_base + m_eq_local_base + i] = 1.0;
    }

    // Jac for d(x) - p + n
    size_type m_ineq_local_base = Jin.m();
    for(int i = 0; i < m_ineq_local_base; ++i) {
      memcpy(M_[i + m_eq_local_base], Jin.M_[i], n_local_base * sizeof(double));
      M_[i + m_eq_local_base][n_local_base + 2 * m_eq_local_base + i] = -1.0;
      M_[i + m_eq_local_base][n_local_base + 2 * m_eq_local_base + m_ineq_local_base + i] = 1.0;
    }
  }
}

#ifdef HIOP_DEEPCHECKS
void hiopMatrixDenseRowMajor::overwriteUpperTriangleWithLower()
{
  assert(n_local_ == n_global_ && "Use only with local, non-distributed matrices");
  for(int i = 0; i < m_local_; i++)
    for(int j = i + 1; j < n_local_; j++) M_[i][j] = M_[j][i];
}
void hiopMatrixDenseRowMajor::overwriteLowerTriangleWithUpper()
{
  assert(n_local_ == n_global_ && "Use only with local, non-distributed matrices");
  for(int i = 1; i < m_local_; i++)
    for(int j = 0; j < i; j++) M_[i][j] = M_[j][i];
}
#endif

hiopMatrixDense* hiopMatrixDenseRowMajor::alloc_clone() const
{
  hiopMatrixDense* c = new hiopMatrixDenseRowMajor(*this);
  return c;
}

hiopMatrixDense* hiopMatrixDenseRowMajor::new_copy() const
{
  hiopMatrixDense* c = new hiopMatrixDenseRowMajor(*this);
  c->copyFrom(*this);
  return c;
}

void hiopMatrixDenseRowMajor::setToZero() { setToConstant(0.0); }
void hiopMatrixDenseRowMajor::setToConstant(double c)
{
  if(!M_[0]) {
    assert(m_local_ == 0);
    return;
  }
  double* buf = M_[0];
  for(int j = 0; j < n_local_; j++) *(buf++) = c;

  buf = M_[0];
  int inc = 1;
  for(int i = 1; i < m_local_; i++) DCOPY(&n_local_, buf, &inc, M_[i], &inc);

  // memcpy(M[i], buf, sizeof(double)*n_local_);
  // memcpy has similar performance as dcopy_; both faster than a loop
}

bool hiopMatrixDenseRowMajor::isfinite() const
{
  for(int i = 0; i < m_local_; i++)
    for(int j = 0; j < n_local_; j++)
      if(false == std::isfinite(M_[i][j])) return false;
  return true;
}

void hiopMatrixDenseRowMajor::print(FILE* f,
                                    const char* msg /*=NULL*/,
                                    int maxRows /*=-1*/,
                                    int maxCols /*=-1*/,
                                    int rank /*=-1*/) const
{
  if(myrank_ == rank || rank == -1) {
    if(NULL == f) f = stdout;
    if(maxRows > m_local_) maxRows = m_local_;
    if(maxCols > n_local_) maxCols = n_local_;

    if(msg) {
      fprintf(f, "%s (local_dims=[%d,%d])\n", msg, m_local_, n_local_);
    } else {
      fprintf(f,
              "hiopMatrixDenseRowMajor::printing max=[%d,%d] (local_dims=[%d,%d], on rank=%d)\n",
              maxRows,
              maxCols,
              m_local_,
              n_local_,
              myrank_);
    }
    maxRows = maxRows >= 0 ? maxRows : m_local_;
    maxCols = maxCols >= 0 ? maxCols : n_local_;
    fprintf(f, "[");
    for(int i = 0; i < maxRows; i++) {
      if(i > 0) fprintf(f, " ");
      for(int j = 0; j < maxCols; j++) fprintf(f, "%20.12e ", M_[i][j]);
      if(i < maxRows - 1)
        fprintf(f, "; ...\n");
      else
        fprintf(f, "];\n");
    }
  }
}

/*  y = beta * y + alpha * this * x
 *
 * Sizes: y is m_local_, x is n_local_, the matrix is m_local_ x n_global_, and the
 * local chunk is m_local_ x n_local_
 */
void hiopMatrixDenseRowMajor::timesVec(double beta, hiopVector& y_, double alpha, const hiopVector& x_) const
{
  hiopVectorPar& y = dynamic_cast<hiopVectorPar&>(y_);
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(y.get_local_size() == m_local_);
  assert(y.get_size() == m_local_);  // y should not be distributed
  assert(x.get_local_size() == n_local_);
  assert(x.get_size() == n_global_);

  if(beta != 0) assert(y.isfinite_local());
  assert(x.isfinite_local());
#endif

  timesVec(beta, y.local_data(), alpha, x.local_data_const());

#ifdef HIOP_DEEPCHECKS
  assert(y.isfinite_local());
#endif
}

void hiopMatrixDenseRowMajor::timesVec(double beta, double* ya, double alpha, const double* xa) const
{
  char fortranTrans = 'T';
  int MM = m_local_, NN = n_local_, incx_y = 1;

#ifdef HIOP_USE_MPI
  // only add beta*y on one processor (rank 0)
  if(myrank_ != 0) beta = 0.0;
#endif
  if(MM != 0 && NN != 0) {
    // the arguments seem reversed but so is trans='T'
    // required since we keep the matrix row-wise, while the Fortran/BLAS expects them column-wise
    DGEMV(&fortranTrans, &NN, &MM, &alpha, &M_[0][0], &NN, xa, &incx_y, &beta, ya, &incx_y);
  } else {
    if(MM != 0) {
      // y.scale( beta );
      if(beta != 1.) {
        int one = 1;
        DSCAL(&MM, &beta, ya, &one);
      }
    } else {
      assert(MM == 0);
      return;
    }
  }
#ifdef HIOP_USE_MPI
  // here m_local_ is > 0
  double yglob[m_local_];
  int ierr = MPI_Allreduce(ya, yglob, m_local_, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  memcpy(ya, yglob, m_local_ * sizeof(double));
#endif
}

/* y = beta * y + alpha * transpose(this) * x */
void hiopMatrixDenseRowMajor::transTimesVec(double beta, hiopVector& y_, double alpha, const hiopVector& x_) const
{
  hiopVectorPar& y = dynamic_cast<hiopVectorPar&>(y_);
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(x.get_local_size() == m_local_);
  assert(x.get_size() == m_local_);  // x should not be distributed
  assert(y.get_local_size() == n_local_);
  assert(y.get_size() == n_global_);
  assert(y.isfinite_local());
  assert(x.isfinite_local());
#endif
  transTimesVec(beta, y.local_data(), alpha, x.local_data_const());
}

void hiopMatrixDenseRowMajor::transTimesVec(double beta, double* ya, double alpha, const double* xa) const
{
  char fortranTrans = 'N';
  int MM = m_local_, NN = n_local_, incx_y = 1;

  if(MM != 0 && NN != 0) {
    // the arguments seem reversed but so is trans='T'
    // required since we keep the matrix row-wise, while the Fortran/BLAS expects them column-wise
    DGEMV(&fortranTrans, &NN, &MM, &alpha, &M_[0][0], &NN, xa, &incx_y, &beta, ya, &incx_y);
  } else {
    if(NN != 0) {
      // y.scale( beta );
      int one = 1;
      DSCAL(&NN, &beta, ya, &one);
    }
  }
}

/* W = beta*W + alpha*this*X
 * -- this is 'M' mxn, X is nxk, W is mxk
 *
 * Precondition:
 * - W, this, and X need to be local matrices (not distributed). All multiplications of distributed
 * matrices needed by HiOp can be done efficiently in parallel using 'transTimesMat'
 */
void hiopMatrixDenseRowMajor::timesMat(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
#ifndef HIOP_USE_MPI
  timesMat_local(beta, W_, alpha, X_);
#else
  auto& W = dynamic_cast<hiopMatrixDenseRowMajor&>(W_);
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);

  assert(W.m() == this->m());
  assert(X.m() == this->n());
  assert(W.n() == X.n());

  if(W.m() == 0 || X.m() == 0 || W.n() == 0) return;
#ifdef HIOP_DEEPCHECKS
  assert(W.isfinite());
  assert(X.isfinite());
#endif

  if(X.n_local_ != X.n_global_ || this->n_local_ != this->n_global_) {
    assert(false && "'timesMat' involving distributed matrices is not needed/supported" &&
           "also, it cannot be performed efficiently with the data distribution used by this class");
    W.setToConstant(beta);
    return;
  }

  timesMat_local(beta, W_, alpha, X_);
  // if(0==myrank_) timesMat_local(beta,W_,alpha,X_);
  // else          timesMat_local(0.,  W_,alpha,X_);

  // int n2Red=W.m()*W.n();
  // double* Wglob = new_mxnlocal_buff(); //[n2Red];
  // int ierr = MPI_Allreduce(WM[0], Wglob, n2Red, MPI_DOUBLE, MPI_SUM,comm); assert(ierr==MPI_SUCCESS);
  // memcpy(WM[0], Wglob, n2Red*sizeof(double));

#endif
}

/* W = beta*W + alpha*this*X
 * -- this is 'M' mxn, X is nxk, W is mxk
 */
void hiopMatrixDenseRowMajor::timesMat_local(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);
  auto& W = dynamic_cast<hiopMatrixDenseRowMajor&>(W_);
#ifdef HIOP_DEEPCHECKS
  assert(W.m() == this->m());
  assert(X.m() == this->n());
  assert(W.n() == X.n());
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  assert(W.n_local_ == W.n_global_ && "requested multiplication is not supported, see timesMat");

  /* C = alpha*op(A)*op(B) + beta*C in our case is
     Wt= alpha* Xt  *Mt    + beta*Wt */
  char trans = 'N';
  int M = X.n(), N = m_local_, K = X.m();
  int ldx = X.n(), ldm = n_local_, ldw = X.n();

  double* XM = X.local_data_const();
  double* WM = W.local_data();
  // DGEMM(&trans,&trans, &M,&N,&K, &alpha,XM[0],&ldx, this->M_[0],&ldm, &beta,WM[0],&ldw);

  DGEMM(&trans, &trans, &M, &N, &K, &alpha, XM, &ldx, this->M_[0], &ldm, &beta, WM, &ldw);

  /* C = alpha*op(A)*op(B) + beta*C in our case is
     Wt= alpha* Xt  *Mt    + beta*Wt */

  // char trans='T';
  // int lda=X.m(), ldb=n_local_, ldc=W.n();
  // int M=X.n(), N=this->m(), K=this->n_local_;

  // DGEMM(&trans,&trans, &M,&N,&K, &alpha,XM[0],&lda, this->M[0],&ldb, &beta,WM[0],&ldc);
}

/* W = beta*W + alpha*this^T*X
 * -- this is mxn, X is mxk, W is nxk
 */
void hiopMatrixDenseRowMajor::transTimesMat(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);
  auto& W = dynamic_cast<hiopMatrixDenseRowMajor&>(W_);

  assert(W.m() == n_local_);
  assert(X.m() == m_local_);
  assert(W.n() == X.n());
#ifdef HIOP_DEEPCHECKS
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  if(W.m() == 0) return;

  assert(this->n_global_ == this->n_local_ && "requested parallel multiplication is not supported");

  /* C = alpha*op(A)*op(B) + beta*C in our case is Wt= alpha* Xt  *M    + beta*Wt */
  char transX = 'N', transM = 'T';
  int ldx = X.n_local_, ldm = n_local_, ldw = W.n_local_;
  int M = X.n_local_, N = n_local_, K = X.m();
  double* XM = X.local_data_const();
  double* WM = W.local_data();

  // DGEMM(&transX, &transM, &M,&N,&K, &alpha,XM,&ldx, this->M_[0],&ldm, &beta,WM,&ldw);
  DGEMM(&transX, &transM, &M, &N, &K, &alpha, XM, &ldx, this->local_data_const(), &ldm, &beta, WM, &ldw);
}

/* W = beta*W + alpha*this*X^T
 * -- this is mxn, X is kxn, W is mxk
 */
void hiopMatrixDenseRowMajor::timesMatTrans_local(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);
  auto& W = dynamic_cast<hiopMatrixDenseRowMajor&>(W_);
#ifdef HIOP_DEEPCHECKS
  assert(W.m() == m_local_);
  // assert(X.n()==n_local_);
  assert(W.n() == X.m());
#endif
  assert(W.n_local_ == W.n_global_ && "not intended for the case when the result matrix is distributed.");
  if(W.m() == 0) return;
  if(W.n() == 0) return;
  if(n_local_ == 0) {
    if(beta != 1.0) {
      int one = 1;
      int mn = W.m() * W.n();
      DSCAL(&mn, &beta, W.M_[0], &one);
    }
    return;
  }

  /* C = alpha*op(A)*op(B) + beta*C in our case is Wt= alpha* X  *Mt    + beta*Wt */
  char transX = 'T', transM = 'N';
  int ldx = n_local_;  //=X.n(); (modified to support the parallel case)
  int ldm = n_local_, ldw = W.n();
  int M = X.m(), N = m_local_, K = n_local_;
  double* XM = X.local_data_const();
  double* WM = W.local_data();

  DGEMM(&transX, &transM, &M, &N, &K, &alpha, XM, &ldx, this->local_data_const(), &ldm, &beta, WM, &ldw);
}
/* W = beta*W + alpha*this*X^T */
void hiopMatrixDenseRowMajor::timesMatTrans(double beta, hiopMatrix& W_, double alpha, const hiopMatrix& X_) const
{
  auto& W = dynamic_cast<hiopMatrixDenseRowMajor&>(W_);
  assert(W.n_local_ == W.n_global_ && "not intended for the case when the result matrix is distributed.");
#ifdef HIOP_DEEPCHECKS
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);
  assert(W.isfinite());
  assert(X.isfinite());
  assert(this->n() == X.n());
  assert(this->m() == W.m());
  assert(X.m() == W.n());
#endif

  if(W.m() == 0) return;
  if(W.n() == 0) return;

  if(0 == myrank_)
    timesMatTrans_local(beta, W_, alpha, X_);
  else
    timesMatTrans_local(0., W_, alpha, X_);

#ifdef HIOP_USE_MPI
  int n2Red = W.m() * W.n();
  double* WM = W.local_data();
  double* Wglob = W.new_mxnlocal_buff();
  int ierr = MPI_Allreduce(WM, Wglob, n2Red, MPI_DOUBLE, MPI_SUM, comm_);
  assert(ierr == MPI_SUCCESS);
  memcpy(WM, Wglob, n2Red * sizeof(double));
#endif
}
void hiopMatrixDenseRowMajor::addDiagonal(const double& alpha, const hiopVector& d_)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
#ifdef HIOP_DEEPCHECKS
  assert(d.get_size() == n());
  assert(d.get_size() == m());
  assert(d.get_local_size() == m_local_);
  assert(d.get_local_size() == n_local_);
#endif
  const double* dd = d.local_data_const();
  for(int i = 0; i < n_local_; i++) M_[i][i] += alpha * dd[i];
}
void hiopMatrixDenseRowMajor::addDiagonal(const double& value)
{
  for(int i = 0; i < n_local_; i++) M_[i][i] += value;
}
void hiopMatrixDenseRowMajor::addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
  size_type dlen = d.get_size();
#ifdef HIOP_DEEPCHECKS
  assert(start >= 0);
  assert(start + dlen <= n_local_);
#endif

  const double* dd = d.local_data_const();
  for(int i = start; i < start + dlen; i++) M_[i][i] += alpha * dd[i - start];
}

/* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
 * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems'
 * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
void hiopMatrixDenseRowMajor::addSubDiagonal(int start_on_dest_diag,
                                             const double& alpha,
                                             const hiopVector& d_,
                                             int start_on_src_vec,
                                             int num_elems /*=-1*/)
{
  const hiopVectorPar& d = dynamic_cast<const hiopVectorPar&>(d_);
  if(num_elems < 0) num_elems = d.get_size() - start_on_src_vec;
  assert(num_elems <= d.get_size());
  assert(n_local_ == n_global_ && "method supported only for non-distributed matrices");
  assert(n_local_ == m_local_ && "method supported only for symmetric matrices");

  assert(start_on_dest_diag >= 0 && start_on_dest_diag < m_local_);
  num_elems = std::min(num_elems, m_local_ - start_on_dest_diag);

  const double* dd = d.local_data_const();
  for(int i = 0; i < num_elems; i++) M_[i + start_on_dest_diag][i + start_on_dest_diag] += alpha * dd[start_on_src_vec + i];
}

void hiopMatrixDenseRowMajor::addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
{
  assert(num_elems >= 0);
  assert(start_on_dest_diag >= 0 && start_on_dest_diag + num_elems <= n_local_);
  assert(n_local_ == n_global_ && "method supported only for non-distributed matrices");
  assert(n_local_ == m_local_ && "method supported only for symmetric matrices");

  for(int i = 0; i < num_elems; i++) M_[i + start_on_dest_diag][i + start_on_dest_diag] += c;
}

void hiopMatrixDenseRowMajor::addMatrix(double alpha, const hiopMatrix& X_)
{
  const auto& X = dynamic_cast<const hiopMatrixDenseRowMajor&>(X_);
#ifdef HIOP_DEEPCHECKS
  assert(m_local_ == X.m_local_);
  assert(n_local_ == X.n_local_);
#endif

  int N = m_local_ * n_local_, inc = 1;
  DAXPY(&N, &alpha, X.M_[0], &inc, M_[0], &inc);
}

/* block of W += alpha*this' */
void hiopMatrixDenseRowMajor::transAddToSymDenseMatrixUpperTriangle(int row_start,
                                                                    int col_start,
                                                                    double alpha,
                                                                    hiopMatrixDense& W) const
{
  assert(row_start >= 0 && n() + row_start <= W.m());
  assert(col_start >= 0 && m() + col_start <= W.n());
  assert(W.n() == W.m());

  int n_W = W.n();
  double* WM = W.local_data();
  for(int ir = 0; ir < m_local_; ir++) {
    const int jW = ir + col_start;
    for(int jc = 0; jc < n_local_; jc++) {
      const int iW = jc + row_start;
      assert(iW <= jW && "source entries need to map inside the upper triangular part of destination");
      // WM[iW][jW] += alpha*this->M_[ir][jc];
      WM[iW * n_W + jW] += alpha * this->M_[ir][jc];
    }
  }
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
void hiopMatrixDenseRowMajor::addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start,
                                                                            double alpha,
                                                                            hiopMatrixDense& W) const
{
  int n_W = W.n();
  assert(W.n() == W.m());
  assert(this->n() == this->m());
  assert(diag_start + this->n() <= W.n());
  double* WM = W.local_data();
  for(int i = 0; i < n_local_; i++) {
    const int iW = i + diag_start;
    for(int j = i; j < m_local_; j++) {
      const int jW = j + diag_start;
      assert(iW <= jW && "source entries need to map inside the upper triangular part of destination");
      assert(iW < W.n() && jW < W.m());
      // WM[iW][jW] += alpha*this->M_[i][j];
      WM[iW * n_W + jW] += alpha * this->M_[i][j];
    }
  }
}

double hiopMatrixDenseRowMajor::max_abs_value()
{
  char norm = 'M';
  double maxv = DLANGE(&norm, &n_local_, &m_local_, M_[0], &n_local_, NULL);
#ifdef HIOP_USE_MPI
  double maxvg;
  int ierr = MPI_Allreduce(&maxv, &maxvg, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(ierr == MPI_SUCCESS);
  return maxvg;
#endif
  return maxv;
}

void hiopMatrixDenseRowMajor::row_max_abs_value(hiopVector& ret_vec)
{
  char norm = 'M';
  int one = 1;
  double maxv;

  hiopVectorPar& vec = dynamic_cast<hiopVectorPar&>(ret_vec);
  assert(m_local_ == vec.get_local_size());

  for(int irow = 0; irow < m_local_; irow++) {
    maxv = DLANGE(&norm, &one, &n_local_, M_[0] + (irow * n_local_), &one, nullptr);
#ifdef HIOP_USE_MPI
    double maxvg;
    int ierr = MPI_Allreduce(&maxv, &maxvg, 1, MPI_DOUBLE, MPI_MAX, comm_);
    assert(ierr == MPI_SUCCESS);
    maxv = maxvg;
#endif
    vec.local_data()[irow] = maxv;
  }
}

void hiopMatrixDenseRowMajor::scale_row(hiopVector& vec_scal, const bool inv_scale)
{
  int one = 1;
  double scal;

  hiopVectorPar& vec = dynamic_cast<hiopVectorPar&>(vec_scal);
  assert(m_local_ == vec.get_local_size());
  double* vd = vec.local_data();

  for(int irow = 0; irow < m_local_; irow++) {
    if(inv_scale) {
      scal = 1. / vd[irow];
    } else {
      scal = vd[irow];
    }
    DSCAL(&n_local_, &scal, M_[0] + (irow * n_local_), &one);
  }
}

#ifdef HIOP_DEEPCHECKS
bool hiopMatrixDenseRowMajor::assertSymmetry(double tol) const
{
  if(n_local_ != n_global_) {
    assert(false && "should be used only for local matrices");
    return false;
  }
  // must be square
  if(m_local_ != n_global_) {
    assert(false);
    return false;
  }

  // symmetry
  for(int i = 0; i < n_local_; i++)
    for(int j = 0; j < n_local_; j++) {
      double ij = M_[i][j], ji = M_[j][i];
      double relerr = std::abs(ij - ji) / (1 + std::abs(ij));
      assert(relerr < tol);
      if(relerr >= tol) {
        return false;
      }
    }
  return true;
}
#endif

bool hiopMatrixDenseRowMajor::symmetrize()
{
  if(n_local_ != n_global_) {
    assert(false && "should be used only for local matrices");
    return false;
  }
  // must be square
  if(m_local_ != n_global_) {
    assert(false);
    return false;
  }

  // symmetrize --- copy the upper triangular part to lower tirangular part
  for(index_type i = 0; i < n_local_; i++) {
    for(index_type j = i + 1; j < n_local_; j++) {
      M_[j][i] = M_[i][j];
    }
  }
  return true;
}

};  // namespace hiop
