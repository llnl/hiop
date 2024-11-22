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
 * @file hiopMatrixSparseCsrCuda.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#include "hiopMatrixSparseCsrCuda.hpp"

#ifdef HIOP_USE_CUDA

#include "hiopVectorPar.hpp"
#include "hiopVectorCuda.hpp"

#include "MatrixSparseCsrCudaKernels.hpp"
#include "MemBackendCudaImpl.hpp"

#include <algorithm>  //for std::min
#include <cmath>      //for std::isfinite
#include <cstring>
#include <vector>
#include <numeric>
#include <cassert>
#include <sstream>
#include <iostream>  // std::cout, std::fixed
#include <iomanip>   // std::setprecision

#include "hiopCppStdUtils.hpp"
#include <set>
#include <map>

namespace hiop
{
hiopMatrixSparseCSRCUDA::hiopMatrixSparseCSRCUDA(size_type rows, size_type cols, size_type nnz)
    : hiopMatrixSparseCSR(rows, cols, nnz),
      irowptr_(nullptr),
      jcolind_(nullptr),
      values_(nullptr),
      buffer_csc2csr_(nullptr),
      buffer_geam2_(nullptr),
      buffer_gemm3_(nullptr),
      buffer_gemm4_(nullptr),
      buffer_gemm5_(nullptr),
      mat_sp_descr_(nullptr)
{
  cusparseStatus_t ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  // the call below initializes the fields of mat_descr_ to what we need
  //  - MatrixType -> CUSPARSE_MATRIX_TYPE_GENERAL
  //  - IndexBase  -> CUSPARSE_INDEX_BASE_ZERO
  //  - leaves other fields uninitialized
  cusparseStatus_t st = cusparseCreateMatDescr(&mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  alloc();

  st = cusparseSpGEMM_createDescr(&gemm_sp_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

hiopMatrixSparseCSRCUDA::hiopMatrixSparseCSRCUDA()
    : hiopMatrixSparseCSR(0, 0, 0),
      irowptr_(nullptr),
      jcolind_(nullptr),
      values_(nullptr),
      buffer_csc2csr_(nullptr),
      buffer_geam2_(nullptr),
      buffer_gemm3_(nullptr),
      buffer_gemm4_(nullptr),
      buffer_gemm5_(nullptr),
      mat_sp_descr_(nullptr)
{
  cusparseStatus_t ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  // the call below initializes the fields of mat_descr_ to what we need
  //  - MatrixType -> CUSPARSE_MATRIX_TYPE_GENERAL
  //  - IndexBase  -> CUSPARSE_INDEX_BASE_ZERO
  //  - leaves other fields uninitialized
  cusparseStatus_t st = cusparseCreateMatDescr(&mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  st = cusparseSpGEMM_createDescr(&gemm_sp_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

hiopMatrixSparseCSRCUDA::~hiopMatrixSparseCSRCUDA()
{
  dealloc();

  auto cret = cudaFree(buffer_gemm5_);
  assert(cudaSuccess == cret);
  cret = cudaFree(buffer_gemm4_);
  assert(cudaSuccess == cret);
  cret = cudaFree(buffer_gemm3_);
  assert(cudaSuccess == cret);

  cret = cudaFree(buffer_geam2_);
  assert(cudaSuccess == cret);
  cret = cudaFree(buffer_csc2csr_);
  assert(cudaSuccess == cret);

  cusparseDestroy(h_cusparse_);
  // cusolverSpDestroy(h_cusolver_);

  cusparseStatus_t st = cusparseDestroyMatDescr(mat_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  st = cusparseSpGEMM_destroyDescr(gemm_sp_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

void hiopMatrixSparseCSRCUDA::alloc()
{
  cudaError_t err;
  err = cudaMalloc(&irowptr_, (nrows_ + 1) * sizeof(index_type));
  assert(cudaSuccess == err && irowptr_);

  err = cudaMalloc(&jcolind_, nnz_ * sizeof(index_type));
  assert(cudaSuccess == err && jcolind_);

  err = cudaMalloc(&values_, nnz_ * sizeof(double));
  assert(cudaSuccess == err && values_);

  assert(nullptr == mat_sp_descr_);
  auto st = cusparseCreateCsr(&mat_sp_descr_,
                              nrows_,
                              ncols_,
                              nnz_,
                              irowptr_,
                              jcolind_,
                              values_,
                              CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

void hiopMatrixSparseCSRCUDA::dealloc()
{
  auto st = cusparseDestroySpMat(mat_sp_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
  mat_sp_descr_ = nullptr;

  cudaError_t err;
  err = cudaFree(values_);
  assert(cudaSuccess == err);
  values_ = nullptr;

  err = cudaFree(jcolind_);
  assert(cudaSuccess == err);
  jcolind_ = nullptr;

  err = cudaFree(irowptr_);
  assert(cudaSuccess == err);
  irowptr_ = nullptr;
}

void hiopMatrixSparseCSRCUDA::setToZero() { assert(false && "work in progress"); }
void hiopMatrixSparseCSRCUDA::setToConstant(double c) { assert(false && "work in progress"); }

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSRCUDA::timesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
{
  assert(false && "work in progress");
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this * x */
void hiopMatrixSparseCSRCUDA::timesVec(double beta, double* y, double alpha, const double* x) const
{
  assert(false && "not yet implemented");
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSRCUDA::transTimesVec(double beta, hiopVector& y, double alpha, const hiopVector& x) const
{
  assert(false && "work in progress");
  assert(x.get_size() == nrows_);
  assert(y.get_size() == ncols_);

  hiopVectorPar& yy = dynamic_cast<hiopVectorPar&>(y);
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  transTimesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this^T * x */
void hiopMatrixSparseCSRCUDA::transTimesVec(double beta, double* y, double alpha, const double* x) const
{
  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::timesMatTrans(double beta, hiopMatrix& Wmat, double alpha, const hiopMatrix& M2mat) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::addDiagonal(const double& alpha, const hiopVector& D)
{
  assert(nrows_ == D.get_size());
  assert(nrows_ == ncols_ && "Matrix must be square");
  assert(dynamic_cast<const hiopVectorCuda*>(&D) && "input vector must be CUDA");

  hiop::cuda::csr_add_diag_kernel(nrows_,
                                  nnz_,
                                  irowptr_,
                                  jcolind_,
                                  values_,
                                  alpha,
                                  D.local_data_const(),
                                  exec_space_.exec_policies().bl_sz_binary_search);
}

void hiopMatrixSparseCSRCUDA::addDiagonal(const double& val)
{
  assert(nrows_ == ncols_ && "Matrix must be square");
  hiop::cuda::csr_add_diag_kernel(nrows_,
                                  nnz_,
                                  irowptr_,
                                  jcolind_,
                                  values_,
                                  val,
                                  exec_space_.exec_policies().bl_sz_binary_search);
}
void hiopMatrixSparseCSRCUDA::addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                                  const size_type& num_elems,
                                                  const hiopVector& d_,
                                                  const index_type& start_on_nnz_idx,
                                                  double scal)
{
  assert(false && "not implemented");
}

void hiopMatrixSparseCSRCUDA::setSubDiagonalTo(const index_type& start_on_dest_diag,
                                               const size_type& num_elems,
                                               const double& c,
                                               const index_type& start_on_nnz_idx)
{
  assert(start_on_dest_diag >= 0 && start_on_dest_diag + num_elems <= this->nrows_);
  assert(false && "not implemented");
}

void hiopMatrixSparseCSRCUDA::addMatrix(double alpha, const hiopMatrix& X) { assert(false && "not needed"); }

/* block of W += alpha*transpose(this)
 * Note W; contains only the upper triangular entries */
void hiopMatrixSparseCSRCUDA::transAddToSymDenseMatrixUpperTriangle(index_type row_start,
                                                                    index_type col_start,
                                                                    double alpha,
                                                                    hiopMatrixDense& W) const
{
  assert(row_start >= 0 && row_start + ncols_ <= W.m());
  assert(col_start >= 0 && col_start + nrows_ <= W.n());
  assert(W.n() == W.m());

  assert(false && "not yet implemented");
}

double hiopMatrixSparseCSRCUDA::max_abs_value()
{
  assert(false && "work in progress");
  // char norm='M'; size_type one=1;
  // double maxv = DLANGE(&norm, &one, &nnz_, values_, &one, nullptr);
  // return maxv;
  return 0.0;
}

void hiopMatrixSparseCSRCUDA::row_max_abs_value(hiopVector& ret_vec)
{
  assert(ret_vec.get_local_size() == nrows_);
  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::scale_row(hiopVector& vec_scal, const bool inv_scale)
{
  assert(vec_scal.get_local_size() == nrows_);
  assert(false && "not yet implemented");
}

bool hiopMatrixSparseCSRCUDA::isfinite() const
{
  assert(false && "work in progress");
  for(index_type i = 0; i < nnz_; i++)
    if(false == std::isfinite(values_[i])) return false;
  return true;
}

hiopMatrixSparse* hiopMatrixSparseCSRCUDA::alloc_clone() const { return new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnz_); }

hiopMatrixSparse* hiopMatrixSparseCSRCUDA::new_copy() const
{
  hiopMatrixSparseCSRCUDA* copy = new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnz_);
  assert(false && "work in progress");
  return copy;
}
void hiopMatrixSparseCSRCUDA::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "to be implemented - method def too vague for now");
}

/// @brief copy to 3 arrays.
/// @pre these 3 arrays are not nullptr
void hiopMatrixSparseCSRCUDA::copy_to(index_type* irow, index_type* jcol, double* val)
{
  assert(irow && jcol && val);
  assert(false && "work in progress");
  // memcpy(irow, irowptr_, (1+nrows_)*sizeof(index_type));
  // memcpy(jcol, jcolind_, nnz_*sizeof(index_type));
  // memcpy(val, values_, nnz_*sizeof(double));
}

void hiopMatrixSparseCSRCUDA::copy_to(hiopMatrixDense& W)
{
  assert(false && "not needed");
  assert(W.m() == nrows_);
  assert(W.n() == ncols_);
}

void hiopMatrixSparseCSRCUDA::copy_to(hiopMatrixSparseCSRSeq& W)
{
  assert(W.m() == nrows_);
  assert(W.n() == ncols_);
  assert(W.numberOfNonzeros() == nnz_);

  W.exec_space_.copy(W.i_row(), this->i_row(), 1 + nrows_, exec_space_);
  W.exec_space_.copy(W.j_col(), this->j_col(), nnz_, exec_space_);
  W.exec_space_.copy(W.M(), this->M(), nnz_, exec_space_);
}

void hiopMatrixSparseCSRCUDA::addMDinvMtransToDiagBlockOfSymDeMatUTri(index_type rowAndCol_dest_start,
                                                                      const double& alpha,
                                                                      const hiopVector& D,
                                                                      hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

/*
 * block of W += alpha * M1 * D^{-1} * transpose(M2), where M1=this
 *  Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx)
 */
void hiopMatrixSparseCSRCUDA::addMDinvNtransToSymDeMatUTri(index_type row_dest_start,
                                                           index_type col_dest_start,
                                                           const double& alpha,
                                                           const hiopVector& D,
                                                           const hiopMatrixSparse& M2mat,
                                                           hiopMatrixDense& W) const
{
  assert(false && "not needed");
}

void hiopMatrixSparseCSRCUDA::copyRowsFrom(const hiopMatrix& src_gen, const index_type* rows_idxs, size_type n_rows)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  assert(false && "not yet implemented");
}

/**
 * @brief Copy 'n_rows' rows started from 'rows_src_idx_st' (array of size 'n_rows') from 'src' to the destination,
 * which starts from the 'rows_dest_idx_st'th row in 'this'
 *
 * @pre 'this' must have exactly, or more than 'n_rows' rows
 * @pre 'this' must have exactly, or more cols than 'src'
 */
void hiopMatrixSparseCSRCUDA::copyRowsBlockFrom(const hiopMatrix& src_gen,
                                                const index_type& rows_src_idx_st,
                                                const size_type& n_rows,
                                                const index_type& rows_dest_idx_st,
                                                const size_type& dest_nnz_st)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(this->n() >= src.n());
  assert(n_rows + rows_src_idx_st <= src.m());
  assert(n_rows + rows_dest_idx_st <= this->m());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copySubmatrixFrom(const hiopMatrix& src_gen,
                                                const index_type& dest_row_st,
                                                const index_type& dest_col_st,
                                                const size_type& dest_nnz_st,
                                                const bool offdiag_only)
{
  const hiopMatrixSparseCSRCUDA& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  auto m_rows = src.m();
  auto n_cols = src.n();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n());
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                                     const index_type& dest_row_st,
                                                     const index_type& dest_col_st,
                                                     const size_type& dest_nnz_st,
                                                     const bool offdiag_only)
{
  const auto& src = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(src_gen);
  auto m_rows = src.n();
  auto n_cols = src.m();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n());
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::setSubmatrixToConstantDiag_w_colpattern(const double& scalar,
                                                                      const index_type& dest_row_st,
                                                                      const index_type& dest_col_st,
                                                                      const size_type& dest_nnz_st,
                                                                      const size_type& nnz_to_copy,
                                                                      const hiopVector& ix)
{
  assert(ix.get_local_size() + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::setSubmatrixToConstantDiag_w_rowpattern(const double& scalar,
                                                                      const index_type& dest_row_st,
                                                                      const index_type& dest_col_st,
                                                                      const size_type& dest_nnz_st,
                                                                      const size_type& nnz_to_copy,
                                                                      const hiopVector& ix)
{
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(ix.get_local_size() + dest_col_st <= this->n());
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copyDiagMatrixToSubblock(const double& src_val,
                                                       const index_type& dest_row_st,
                                                       const index_type& col_dest_st,
                                                       const size_type& dest_nnz_st,
                                                       const size_type& nnz_to_copy)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy + dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + col_dest_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                                                 const index_type& dest_row_st,
                                                                 const index_type& dest_col_st,
                                                                 const size_type& dest_nnz_st,
                                                                 const size_type& nnz_to_copy,
                                                                 const hiopVector& ix)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy + dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());

  assert(false && "not yet implemented");
}

void hiopMatrixSparseCSRCUDA::print(FILE* file,
                                    const char* msg /*=nullptr*/,
                                    int maxRows /*=-1*/,
                                    int maxCols /*=-1*/,
                                    int rank /*=-1*/) const
{
  int myrank_ = 0, numranks = 1;  // this is a local object => always print

  if(file == nullptr) file = stdout;

  int max_elems = maxRows >= 0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);

  if(myrank_ == rank || rank == -1) {
    index_type* irowptr = new index_type[nrows_ + 1];
    index_type* jcolind = new index_type[nnz_];
    double* values = new double[nnz_];

    cudaMemcpy(irowptr, irowptr_, (nrows_ + 1) * sizeof(index_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(jcolind, jcolind_, nnz_ * sizeof(index_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, values_, nnz_ * sizeof(double), cudaMemcpyDeviceToHost);

    std::stringstream ss;
    if(nullptr == msg) {
      if(numranks > 1) {
        ss << "CSR CUDA matrix of size " << m() << " " << n() << " and nonzeros " << numberOfNonzeros() << ", printing "
           << max_elems << " elems (on rank=" << myrank_ << ")" << std::endl;
      } else {
        ss << "CSR CUDA matrix of size " << m() << " " << n() << " and nonzeros " << numberOfNonzeros() << ", printing "
           << max_elems << " elems" << std::endl;
      }
    } else {
      ss << msg << " ";
    }

    // using matlab indices (starting at 1)
    // fprintf(file, "iRow_=[");
    ss << "iRow_=[";

    for(index_type i = 0; i < nrows_; i++) {
      const index_type ip1 = i + 1;
      for(int p = irowptr[i]; p < irowptr[i + 1] && p < max_elems; ++p) {
        ss << ip1 << "; ";
      }
    }
    ss << "];" << std::endl;

    ss << "jCol_=[";
    for(index_type it = 0; it < max_elems; it++) {
      ss << (jcolind[it] + 1) << "; ";
    }
    ss << "];" << std::endl;

    ss << "v=[";
    ss << std::scientific << std::setprecision(16);
    for(index_type it = 0; it < max_elems; it++) {
      ss << values[it] << "; ";
    }
    // fprintf(file, "];\n");
    ss << "];" << std::endl;

    fprintf(file, "%s", ss.str().c_str());

    delete[] values;
    delete[] irowptr;
    delete[] jcolind;
  }
}

// M = X*Y, where X is `this`. M is mxn, X is mxK and Y is Kxn
hiopMatrixSparseCSR* hiopMatrixSparseCSRCUDA::times_mat_alloc(const hiopMatrixSparseCSR& Y_in) const
{
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  auto& X = *this;

  assert(ncols_ == Y.m());

  cusparseStatus_t st;
  cudaError_t cret;

  //
  // create a temporary matrix descriptor for M
  cusparseSpMatDescr_t mat_descrM;
  st = cusparseCreateCsr(&mat_descrM,
                         X.m(),
                         Y.n(),
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO,
                         CUDA_R_64F);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  cusparseSpGEMMDescr_t spgemmDesc;
  st = cusparseSpGEMM_createDescr(&spgemmDesc);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  cusparseOperation_t opX = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opY = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // inquire buffer size
  size_t buff_size = 0;
  st = cusparseSpGEMMreuse_workEstimation(h_cusparse_,
                                          opX,
                                          opY,
                                          X.mat_sp_descr_,
                                          Y.mat_sp_descr_,
                                          mat_descrM,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &buff_size,
                                          nullptr);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  // allocate buffer
  void* buff_gemm1 = nullptr;
  cret = cudaMalloc((void**)&buff_gemm1, buff_size);
  assert(cret == cudaSuccess);

  // inspect input matrices to determine memory requirements for the next steps
  st = cusparseSpGEMMreuse_workEstimation(h_cusparse_,
                                          opX,
                                          opY,
                                          X.mat_sp_descr_,
                                          Y.mat_sp_descr_,
                                          mat_descrM,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &buff_size,
                                          buff_gemm1);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  // inquire buffer size for nnz call
  size_t buff_size2 = 0;
  size_t buff_size3 = 0;
  size_t buff_size4 = 0;
  st = cusparseSpGEMMreuse_nnz(h_cusparse_,
                               opX,
                               opY,
                               X.mat_sp_descr_,
                               Y.mat_sp_descr_,
                               mat_descrM,
                               CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc,
                               &buff_size2,
                               nullptr,
                               &buff_size3,
                               nullptr,
                               &buff_size4,
                               nullptr);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  void* buff_gemm2 = nullptr;
  void* buff_gemm3 = nullptr;
  void* buff_gemm4 = nullptr;

  cret = cudaMalloc((void**)&buff_gemm2, buff_size2);
  assert(cret == cudaSuccess);
  cret = cudaMalloc((void**)&buff_gemm3, buff_size3);
  assert(cret == cudaSuccess);
  cret = cudaMalloc((void**)&buff_gemm4, buff_size4);
  assert(cret == cudaSuccess);

  st = cusparseSpGEMMreuse_nnz(h_cusparse_,
                               opX,
                               opY,
                               X.mat_sp_descr_,
                               Y.mat_sp_descr_,
                               mat_descrM,
                               CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc,
                               &buff_size2,
                               buff_gemm2,
                               &buff_size3,
                               buff_gemm3,
                               &buff_size4,
                               buff_gemm4);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  cret = cudaFree(buff_gemm1);
  assert(cret == cudaSuccess);

  cret = cudaFree(buff_gemm2);
  assert(cret == cudaSuccess);

  // get sizes of M
  int64_t M_m, M_n, M_nnz;
  st = cusparseSpMatGetSize(mat_descrM, &M_m, &M_n, &M_nnz);
  assert(st == CUSPARSE_STATUS_SUCCESS);
  assert(M_n == Y.n());
  assert(M_m == nrows_);

  hiopMatrixSparseCSRCUDA* M = new hiopMatrixSparseCSRCUDA(nrows_, Y.n(), M_nnz);

  M->set_gemm_buffer3(buff_gemm3);
  M->set_gemm_buffer4(buff_gemm4);
  M->use_sparse_gemm_descriptor(spgemmDesc);
  M->use_sparse_mat_descriptor(mat_descrM);

  return M;
}

// M = X*D*Y, where X is `this`. M is mxn, X is mxK and Y is Kxn
void hiopMatrixSparseCSRCUDA::times_mat_symbolic(hiopMatrixSparseCSR& M_in, const hiopMatrixSparseCSR& Y_in) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  auto& X = *this;

  auto cret = cudaMemset(M.values_, 0x0, M.nnz_ * sizeof(double));
  assert(cudaSuccess == cret);

  cusparseOperation_t opX = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opY = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // inquire size
  size_t buff_size5 = 0;
  auto st = cusparseSpGEMMreuse_copy(h_cusparse_,
                                     opX,
                                     opY,
                                     X.mat_sp_descr_,
                                     Y.mat_sp_descr_,
                                     M.mat_sp_descr_,
                                     CUSPARSE_SPGEMM_DEFAULT,
                                     M.gemm_sp_descr_,
                                     &buff_size5,
                                     nullptr);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  // allocate buffer5
  auto* buffer_gemm5 = M.alloc_gemm_buffer5(buff_size5);

  // the actual call
  st = cusparseSpGEMMreuse_copy(h_cusparse_,
                                opX,
                                opY,
                                X.mat_sp_descr_,
                                Y.mat_sp_descr_,
                                M.mat_sp_descr_,
                                CUSPARSE_SPGEMM_DEFAULT,
                                M.gemm_sp_descr_,
                                &buff_size5,
                                buffer_gemm5);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  // buffer3 not needed anymore
  M.dealloc_gemm_buffer3();
}

// M = beta*M + alpha*X*Y, where X is `this`. M is mxn, X is mxK and Y is Kxn
void hiopMatrixSparseCSRCUDA::times_mat_numeric(double beta,
                                                hiopMatrixSparseCSR& M_in,
                                                double alpha,
                                                const hiopMatrixSparseCSR& Y_in)
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  auto& X = *this;

  if(beta == 0.0) {
    auto cret = cudaMemset(M.values_, 0x0, M.nnz_ * sizeof(double));
    assert(cudaSuccess == cret);
  }

  cusparseOperation_t opX = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opY = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType compute_type = CUDA_R_64F;

  auto st = cusparseSpGEMMreuse_compute(h_cusparse_,
                                        opX,
                                        opY,
                                        &alpha,
                                        X.mat_sp_descr_,
                                        Y.mat_sp_descr_,
                                        &beta,
                                        M.mat_sp_descr_,
                                        compute_type,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        M.gemm_sp_descr_);
  assert(st == CUSPARSE_STATUS_SUCCESS);
}

void hiopMatrixSparseCSRCUDA::form_from_symbolic(const hiopMatrixSparseTriplet& M)
{
  if(M.m() != nrows_ || M.n() != ncols_ || M.numberOfNonzeros() != nnz_) {
    dealloc();

    nrows_ = M.m();
    ncols_ = M.n();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_ >= 0);
  if(nnz_ <= 0) {
    return;
  }

  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  // transfer coo/triplet to device
  int* d_rowind = nullptr;
  d_rowind = exec_space_.alloc_array<index_type>(nnz_);
  assert(d_rowind);
  exec_space_.copy(d_rowind, M.i_row(), nnz_, M.exec_space_);

  // use cuda API
  cusparseStatus_t st = cusparseXcoo2csr(h_cusparse_, d_rowind, nnz_, nrows_, irowptr_, CUSPARSE_INDEX_BASE_ZERO);
  assert(CUSPARSE_STATUS_SUCCESS == st);

  exec_space_.dealloc_array(d_rowind);

  // j indexes can be just transfered
  cudaMemcpy(jcolind_, M.j_col(), nnz_ * sizeof(index_type), cudaMemcpyHostToDevice);
}

void hiopMatrixSparseCSRCUDA::form_from_numeric(const hiopMatrixSparseTriplet& M)
{
  assert(irowptr_ && jcolind_ && values_);
  assert(nrows_ == M.m());
  assert(ncols_ == M.n());
  assert(nnz_ == M.numberOfNonzeros());

  cudaMemcpy(values_, M.M(), nnz_ * sizeof(double), cudaMemcpyHostToDevice);
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_symbolic(const hiopMatrixSparseTriplet& M)
{
  assert(false && "Method not implemented: more efficient to use overload for CSR CUDA matrices.");
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_numeric(const hiopMatrixSparseTriplet& M)
{
  assert(false && "Method not implemented: more efficient to use overload for CSR CUDA matrices.");
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_symbolic(const hiopMatrixSparseCSR& M)
{
  if(M.m() != ncols_ || M.n() != nrows_ || M.numberOfNonzeros() != nnz_) {
    dealloc();

    nrows_ = M.n();
    ncols_ = M.m();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_ >= 0);
  if(nnz_ <= 0) {
    return;
  }

  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  cusparseStatus_t st;
  size_t buffer_size;
  st = cusparseCsr2cscEx2_bufferSize(h_cusparse_,
                                     M.m(),
                                     M.n(),
                                     nnz_,
                                     M.M(),
                                     M.i_row(),
                                     M.j_col(),
                                     values_,
                                     irowptr_,
                                     jcolind_,
                                     CUDA_R_64F,
                                     CUSPARSE_ACTION_SYMBOLIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG1,
                                     &buffer_size);
  assert(CUSPARSE_STATUS_SUCCESS == st);
  cudaError_t ret = cudaMalloc(&buffer_csc2csr_, sizeof(char) * buffer_size);
  assert(cudaSuccess == ret);
}

void hiopMatrixSparseCSRCUDA::form_transpose_from_numeric(const hiopMatrixSparseCSR& M)
{
  assert(irowptr_ && jcolind_ && values_);
  assert(nrows_ == M.n());
  assert(ncols_ == M.m());
  assert(nnz_ == M.numberOfNonzeros());

  assert(buffer_csc2csr_);
  cusparseStatus_t st;
  st = cusparseCsr2cscEx2(h_cusparse_,
                          M.m(),
                          M.n(),
                          nnz_,
                          M.M(),
                          M.i_row(),
                          M.j_col(),
                          values_,
                          irowptr_,
                          jcolind_,
                          CUDA_R_64F,
                          CUSPARSE_ACTION_NUMERIC,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUSPARSE_CSR2CSC_ALG1,
                          buffer_csc2csr_);
  assert(CUSPARSE_STATUS_SUCCESS == st);
}

void hiopMatrixSparseCSRCUDA::form_diag_from_symbolic(const hiopVector& D)
{
  const int m = D.get_size();
  if(m != ncols_ || m != nrows_ || m != nnz_) {
    dealloc();

    nrows_ = m;
    ncols_ = m;
    nnz_ = m;

    alloc();
  }

  assert(irowptr_ && jcolind_ && values_);

  hiop::cuda::csr_form_diag_symbolic_kernel(nrows_, irowptr_, jcolind_, exec_space_.exec_policies().bl_sz_vector_loop);
}

void hiopMatrixSparseCSRCUDA::form_diag_from_numeric(const hiopVector& D)
{
  assert(D.get_size() == ncols_ && D.get_size() == nrows_ && D.get_size() == nnz_);
  assert(irowptr_ && jcolind_ && values_);

  assert(dynamic_cast<const hiopVectorCuda*>(&D) && "input vector must be CUDA");

  cudaError_t ret = cudaMemcpy(values_, D.local_data_const(), nrows_ * sizeof(double), cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == ret);
}

/// Column scaling or right multiplication by a diagonal: `this`=`this`*D
void hiopMatrixSparseCSRCUDA::scale_cols(const hiopVector& D)
{
  assert(false && "work in progress");
  assert(ncols_ == D.get_size());
}

/// @brief Row scaling or left multiplication by a diagonal: `this`=D*`this`
void hiopMatrixSparseCSRCUDA::scale_rows(const hiopVector& D)
{
  assert(nrows_ == D.get_size());

  assert(dynamic_cast<const hiopVectorCuda*>(&D) && "input vector must be CUDA");

  hiop::cuda::csr_scalerows_kernel(nrows_,
                                   ncols_,
                                   nnz_,
                                   irowptr_,
                                   jcolind_,
                                   values_,
                                   D.local_data_const(),
                                   exec_space_.exec_policies().bl_sz_vector_loop);
}

// sparsity pattern of M=X+Y, where X is `this`
hiopMatrixSparseCSR* hiopMatrixSparseCSRCUDA::add_matrix_alloc(const hiopMatrixSparseCSR& Y_in) const
{
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);
  auto& X = *this;

  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());

  cusparseStatus_t st;
  cudaError_t cret;

  double alpha = 1.0;  // dummy
  double beta = 1.0;
  size_t buffer_size;

  //
  // create a (dummy) math descriptor
  //
  // the call below initializes the fields of mat_descr_ to what we need
  //  - MatrixType -> CUSPARSE_MATRIX_TYPE_GENERAL
  //  - IndexBase  -> CUSPARSE_INDEX_BASE_ZERO
  //  - leaves other fields uninitialized
  cusparseMatDescr_t mat_descrM;
  st = cusparseCreateMatDescr(&mat_descrM);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  int* irowptrM = nullptr;
  cret = cudaMalloc((void**)&irowptrM, sizeof(int) * (nrows_ + 1));
  assert(cudaSuccess == cret);
  assert(irowptrM);

  // get size of buffer needed internally
  st = cusparseDcsrgeam2_bufferSizeExt(h_cusparse_,
                                       nrows_,
                                       ncols_,
                                       &alpha,
                                       X.mat_descr_,
                                       X.nnz_,
                                       X.values_,
                                       X.irowptr_,
                                       X.jcolind_,
                                       &beta,
                                       Y.mat_descr_,
                                       Y.nnz_,
                                       Y.values_,
                                       Y.irowptr_,
                                       Y.jcolind_,
                                       mat_descrM,
                                       NULL,  // valuesM,
                                       irowptrM,
                                       NULL,  // jcolindM,
                                       &buffer_size);
  assert(CUSPARSE_STATUS_SUCCESS == st);

  // prepare and allocate buffer
  void* buffer_geam2;
  cret = cudaMalloc((void**)&buffer_geam2, sizeof(char) * buffer_size);
  assert(cudaSuccess == cret);
  assert(buffer_geam2);

  int nnzM;
  st = cusparseXcsrgeam2Nnz(h_cusparse_,
                            nrows_,
                            ncols_,
                            X.mat_descr_,
                            X.nnz_,
                            X.irowptr_,
                            X.jcolind_,
                            Y.mat_descr_,
                            Y.nnz_,
                            Y.irowptr_,
                            Y.jcolind_,
                            mat_descrM,
                            irowptrM,
                            &nnzM,
                            buffer_geam2);

  assert(CUSPARSE_STATUS_SUCCESS == st);

  // mat descriptor not needed anymore
  st = cusparseDestroyMatDescr(mat_descrM);
  assert(st == CUSPARSE_STATUS_SUCCESS);

  hiopMatrixSparseCSRCUDA* M = new hiopMatrixSparseCSRCUDA(nrows_, ncols_, nnzM);

  // play it safe and copy (instead of switching pointers)
  cret = cudaMemcpy(M->irowptr_, (void*)irowptrM, (nrows_ + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
  assert(cudaSuccess == cret);

  cret = cudaFree(irowptrM);
  assert(cudaSuccess == cret);

  // have the buffer_geam2 stay with M
  assert(nullptr == M->buffer_geam2_);
  M->buffer_geam2_ = buffer_geam2;
  buffer_geam2 = nullptr;

  return M;
}

/**
 * Computes sparsity pattern of M = X+Y (i.e., populates the row pointers and
 * column indexes arrays) of `M`.
 *
 */
void hiopMatrixSparseCSRCUDA::add_matrix_symbolic(hiopMatrixSparseCSR& M_in, const hiopMatrixSparseCSR& Y_in) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);

  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());

  assert(M.n() == Y.n());
  assert(M.m() == Y.m());

  //
  // nothing to do for this CUDA, geam2-based implementation
  //
}

/**
 * Performs matrix addition M = alpha*X + beta*Y numerically
 */
void hiopMatrixSparseCSRCUDA::add_matrix_numeric(hiopMatrixSparseCSR& M_in,
                                                 double alpha,
                                                 const hiopMatrixSparseCSR& Y_in,
                                                 double beta) const
{
  auto& M = dynamic_cast<hiopMatrixSparseCSRCUDA&>(M_in);
  auto& Y = dynamic_cast<const hiopMatrixSparseCSRCUDA&>(Y_in);

  assert(nrows_ == Y.m());
  assert(ncols_ == Y.n());

  assert(M.n() == Y.n());
  assert(M.m() == Y.m());
  auto& X = *this;

  assert(M.buffer_geam2_);

  cusparseStatus_t st;
  st = cusparseDcsrgeam2(h_cusparse_,
                         nrows_,
                         ncols_,
                         &alpha,
                         X.mat_descr_,
                         X.nnz_,
                         X.values_,
                         X.irowptr_,
                         X.jcolind_,
                         &beta,
                         Y.mat_descr_,
                         Y.nnz_,
                         Y.values_,
                         Y.irowptr_,
                         Y.jcolind_,
                         M.mat_descr_,
                         M.values_,
                         M.irowptr_,
                         M.jcolind_,
                         M.buffer_geam2_);
  assert(CUSPARSE_STATUS_SUCCESS == st);
}

void hiopMatrixSparseCSRCUDA::set_diagonal(const double& val)
{
  assert(irowptr_ && jcolind_ && values_);
  hiop::cuda::csr_set_diag_kernel(nrows_,
                                  nnz_,
                                  irowptr_,
                                  jcolind_,
                                  values_,
                                  val,
                                  exec_space_.exec_policies().bl_sz_binary_search);
}

void hiopMatrixSparseCSRCUDA::extract_diagonal(hiopVector& diag_out) const
{
  assert(dynamic_cast<const hiopVectorCuda*>(&diag_out) && "input vector must be RAJA-CUDA");

  hiop::cuda::csr_get_diag_kernel(nrows_,
                                  nnz_,
                                  irowptr_,
                                  jcolind_,
                                  values_,
                                  diag_out.local_data(),
                                  exec_space_.exec_policies().bl_sz_binary_search);
}

bool hiopMatrixSparseCSRCUDA::check_csr_is_ordered()
{
  hiopMatrixSparseCSRSeq mat_h(nrows_, ncols_, nnz_);
  this->copy_to(mat_h);
  return mat_h.check_csr_is_ordered();
}

}  // namespace hiop

#endif  // #ifdef HIOP_USE_CUDA
