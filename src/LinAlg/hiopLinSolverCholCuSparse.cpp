// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
 * @file hiopLinSolverCholCuSparse.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#include "hiopLinSolverCholCuSparse.hpp"
#include <hiop_defs.hpp>

#ifdef HIOP_USE_CUDA

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include "hiopMatrixSparseCSRSeq.hpp"

#ifdef HIOP_USE_EIGEN
#include <Eigen/Core>
#include <Eigen/Sparse>

using Scalar = double;
// using SparseMatrixCSC = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::ColMajor>;
using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::RowMajor>;
// using Triplet = Eigen::Triplet<Scalar>;
using Ordering = Eigen::AMDOrdering<SparseMatrixCSR::StorageIndex>;
using PermutationMatrix = Ordering::PermutationType;
#endif

namespace hiop
{

hiopLinSolverCholCuSparse::hiopLinSolverCholCuSparse(hiopMatrixSparseCSR* M, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(M, nlp),
      buf_fact_(nullptr),
      rowptr_(nullptr),
      colind_(nullptr),
      values_(nullptr),
      P_(nullptr),
      PT_(nullptr),
      map_nnz_perm_(nullptr),
      rhs_buf1_(nullptr),
      rhs_buf2_(nullptr)
{
  nnz_ = M->numberOfNonzeros();

  cusolverStatus_t ret;
  cusparseStatus_t ret_sp;

  ret_sp = cusparseCreate(&h_cusparse_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);

  ret = cusolverSpCreate(&h_cusolver_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  ret = cusolverSpCreateCsrcholInfo(&info_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  // matrix description
  ret_sp = cusparseCreateMatDescr(&mat_descr_);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatType(mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatIndexBase(mat_descr_, CUSPARSE_INDEX_BASE_ZERO);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
  ret_sp = cusparseSetMatDiagType(mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
  assert(ret_sp == CUSPARSE_STATUS_SUCCESS);
}

hiopLinSolverCholCuSparse::~hiopLinSolverCholCuSparse()
{
  cudaFree(rhs_buf1_);
  rhs_buf1_ = nullptr;

  cudaFree(rhs_buf2_);
  rhs_buf2_ = nullptr;

  cudaFree(map_nnz_perm_);
  map_nnz_perm_ = nullptr;

  cudaFree(buf_fact_);
  buf_fact_ = nullptr;

  cudaFree(P_);
  P_ = nullptr;
  cudaFree(PT_);
  PT_ = nullptr;

  cudaFree(rowptr_);
  cudaFree(colind_);
  cudaFree(values_);

  cusparseDestroyMatDescr(mat_descr_);
  cusolverSpDestroyCsrcholInfo(info_);
  cusolverSpDestroy(h_cusolver_);
  cusparseDestroy(h_cusparse_);
}

bool hiopLinSolverCholCuSparse::do_symb_analysis(const size_type n,
                                                 const size_type nnz,
                                                 const index_type* rowptr,
                                                 const index_type* colind,
                                                 const double* value,
                                                 index_type* perm)
{
  auto ordering = nlp_->options->GetString("linear_solver_sparse_ordering");
  cusolverStatus_t ret;

  nlp_->log->printf(hovScalars, "Chol CuSolver: using '%s' as ordering strategy.\n", ordering.c_str());

  if("metis" == ordering) {
    const int64_t* options = nullptr;  // use default METIS options
    ret = cusolverSpXcsrmetisndHost(h_cusolver_, n, nnz, mat_descr_, rowptr, colind, options, perm);
    assert(ret == CUSOLVER_STATUS_SUCCESS);
  } else if("symamd-cuda" == ordering) {
    ret = cusolverSpXcsrsymamdHost(h_cusolver_, n, nnz, mat_descr_, rowptr, colind, perm);
    assert(ret == CUSOLVER_STATUS_SUCCESS);
  } else if("symamd-eigen" == ordering) {
#ifdef HIOP_USE_EIGEN
    Eigen::Map<SparseMatrixCSR> M(n, n, nnz, const_cast<int*>(rowptr), const_cast<int*>(colind), const_cast<double*>(value));

    PermutationMatrix P;
    Ordering ordering;
    ordering(M.selfadjointView<Eigen::Upper>(), P);
    memcpy(perm, P.indices().data(), n * sizeof(int));
#else
    assert(false &&
           "user option linear_solver_sparse_ordering=symamd-eigen is inconsistent (HiOp was not build with EIGEN)");
    nlp_->log->printf(
        hovError,
        "option linear_solver_sparse_ordering=symamd-eigen is inconsistent (HiOp was not build with EIGEN).\n");
#endif
  } else {
    assert("symrcm" == ordering && "unrecognized option for sparse solver ordering");
    ret = cusolverSpXcsrsymrcmHost(h_cusolver_, n, nnz, mat_descr_, rowptr, colind, perm);
  }
  return (ret == CUSOLVER_STATUS_SUCCESS);
}

bool hiopLinSolverCholCuSparse::initial_setup()
{
  auto* mat_csr = this->sys_mat_csr();
  cusolverStatus_t ret;
  assert(nullptr == buf_fact_);
  size_type m = mat_csr->m();
  assert(m == mat_csr->n());
  assert(nnz_ == mat_csr->numberOfNonzeros());

  //
  // allocate device CSR arrays; then
  // copy row and col arrays to the device
  //
  assert(nullptr == rowptr_);
  cudaMalloc(&rowptr_, (m + 1) * sizeof(int));

  assert(nullptr == colind_);
  cudaMalloc(&colind_, nnz_ * sizeof(int));

  assert(nullptr == values_);
  cudaMalloc(&values_, nnz_ * sizeof(double));

  assert(rowptr_);
  assert(colind_);
  assert(values_);

  hiopTimer t;
  std::stringstream ss_log;

  //
  // compute permutation to promote sparsity in the factors (on CPU/host)
  //
  t.reset();
  t.start();

  auto* P_h = new index_type[m];

  hiopMatrixSparseCSRSeq mat_csr_h(mat_csr->m(), mat_csr->m(), mat_csr->numberOfNonzeros());
  mat_csr->copy_to(mat_csr_h);

  do_symb_analysis(mat_csr_h.m(), mat_csr_h.numberOfNonzeros(), mat_csr_h.i_row(), mat_csr_h.j_col(), mat_csr_h.M(), P_h);
  ss_log << "\tOrdering: '" << nlp_->options->GetString("linear_solver_sparse_ordering") << "': ";

  t.stop();
  ss_log << std::fixed << std::setprecision(4) << t.getElapsedTime() << " sec\n";

  // compute transpose/inverse permutation
  index_type* PT_h = new index_type[m];
  for(index_type i = 0; i < m; i++) {
    PT_h[P_h[i]] = i;
  }

  // transfer permutation and its transpose to the device
  assert(nullptr == P_);
  cudaMalloc(&P_, m * sizeof(index_type));
  cudaMemcpy(P_, P_h, m * sizeof(index_type), cudaMemcpyHostToDevice);

  assert(nullptr == PT_);
  cudaMalloc(&PT_, m * sizeof(index_type));
  cudaMemcpy(PT_, PT_h, m * sizeof(index_type), cudaMemcpyHostToDevice);
  delete[] PT_h;

  // get permutation buffer size
  size_t buf_size;
  ret = cusolverSpXcsrperm_bufferSizeHost(h_cusolver_,
                                          m,
                                          m,
                                          nnz_,
                                          mat_descr_,
                                          mat_csr_h.i_row(),
                                          mat_csr_h.j_col(),
                                          P_h,
                                          P_h,
                                          &buf_size);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  // temporary buffer needed for permutation purposes (on host)
  unsigned char* buf_perm_h = new unsigned char[buf_size];

  // compute permuted CSR arrays (on host)
  int* rowptr_perm_h = new int[m + 1];
  int* colind_perm_h = new int[nnz_];
  assert(rowptr_perm_h);
  assert(colind_perm_h);
  memcpy(rowptr_perm_h, mat_csr_h.i_row(), (m + 1) * sizeof(int));
  memcpy(colind_perm_h, mat_csr_h.j_col(), nnz_ * sizeof(int));

  // mapping (on host)
  int* map_h = new int[nnz_];
  for(index_type i = 0; i < nnz_; i++) {
    map_h[i] = i;
  }

  t.reset();
  t.start();
  ret =
      cusolverSpXcsrpermHost(h_cusolver_, m, m, nnz_, mat_descr_, rowptr_perm_h, colind_perm_h, P_h, P_h, map_h, buf_perm_h);
  assert(ret == CUSOLVER_STATUS_SUCCESS);
  t.stop();
  ss_log << "\tcsrpermHost: " << t.getElapsedTime() << " sec" << std::endl;
  delete[] P_h;
  delete[] buf_perm_h;

  assert(nullptr == map_nnz_perm_);
  cudaMalloc(&map_nnz_perm_, nnz_ * sizeof(int));
  // transfer the permutation map for nonzeros on device
  cudaMemcpy(map_nnz_perm_, map_h, nnz_ * sizeof(int), cudaMemcpyHostToDevice);
  delete[] map_h;
  // transfer the CSR index arrays on device
  //
  // values_ not needed here and will be updated in matrixChanged()
  cudaMemcpy(rowptr_, rowptr_perm_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(colind_, colind_perm_h, nnz_ * sizeof(int), cudaMemcpyHostToDevice);

  delete[] colind_perm_h;
  delete[] rowptr_perm_h;
  // end of ordering permutation to promote sparsity in the factors

#if 0
  // the code below skips ordering permutation computation that is computed above. It uses
  // the trivial permuation/no factor sparsity promoting permutation. It results in poor
  // sparsity pattern and hence, in very long factorization times. It is intended for debugging
  // purposes only
  {
    cudaMemcpy(rowptr_, mat_csr->i_row(), (m+1)*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(colind_, mat_csr->j_col(), nnz_*sizeof(int), cudaMemcpyDeviceToDevice);
    
    index_type map_h[nnz_];
    for(index_type i=0; i<nnz_; i++) {
      map_h[i] = i;
    }
    assert(nullptr == map_nnz_perm_);
    cudaMalloc(&map_nnz_perm_, nnz_*sizeof(int));
    cudaMemcpy(map_nnz_perm_, map_h, nnz_*sizeof(int), cudaMemcpyHostToDevice);


    index_type PT_h[m];
    index_type P_h[m];
    for(index_type i=0; i<m; i++) {
      PT_h[i] = i;
      P_h[i] = i;
    }
  
    assert(nullptr == P_);
    cudaMalloc(&P_, m*sizeof(index_type));
    cudaMemcpy(P_, P_h, m*sizeof(index_type), cudaMemcpyHostToDevice);
    
    assert(nullptr == PT_);
    cudaMalloc(&PT_, m*sizeof(index_type));
    cudaMemcpy(PT_, PT_h, m*sizeof(index_type), cudaMemcpyHostToDevice);
#endif  // end of trivial permutation

  t.reset();
  t.start();
  //
  // analysis -> pattern of L
  //
  ret = cusolverSpXcsrcholAnalysis(h_cusolver_, m, nnz_, mat_descr_, rowptr_, colind_, info_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);
  t.stop();
  ss_log << "\tcsrcholAnalysis: " << t.getElapsedTime() << " sec" << std::endl;

  // buffer size
  size_t internalData;  // in BYTEs
  ret = cusolverSpDcsrcholBufferInfo(h_cusolver_,
                                     m,
                                     nnz_,
                                     mat_descr_,
                                     mat_csr->M(),  //! don't we need to pass the permuted values?
                                     rowptr_,
                                     colind_,
                                     info_,
                                     &internalData,
                                     &buf_fact_size_);
  assert(ret == CUSOLVER_STATUS_SUCCESS);

  if(perf_report_) {
    nlp_->log->printf(hovSummary, "CholCuSolver: initial setup times: \n%s", ss_log.str().c_str());
  }

  cudaError_t ret_cu = cudaMalloc(&buf_fact_, sizeof(unsigned char) * buf_fact_size_);
  assert(ret_cu == cudaSuccess);

  return true;
}

/* returns -1 if zero or negative pivots are encountered */
int hiopLinSolverCholCuSparse::matrixChanged()
{
  auto* mat_csr = this->sys_mat_csr();
  size_type m = mat_csr->m();
  assert(m == mat_csr->n());
  assert(nnz_ == mat_csr->numberOfNonzeros());
  cusolverStatus_t ret;

  hiopTimer t;
  if(nullptr == buf_fact_) {
    t.start();
    nlp_->runStats.linsolv.tmFactTime.start();
    if(!initial_setup()) {
      nlp_->log->printf(hovError, "hiopLinSolverCholCuSparse: initial setup failed.\n");
      return -1;
    }
    nlp_->runStats.linsolv.tmFactTime.stop();
    t.stop();
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
                        "CholCuSolver: initial setup total %.4f sec (includes device transfer)\n",
                        t.getElapsedTime());
    }
  }

  nlp_->runStats.linsolv.tmFactTime.start();
  //
  // permute nonzeros in values_buf_ into values_ accordingly to map_nnz_perm_
  //
  permute_vec(nnz_, mat_csr->M(), map_nnz_perm_, values_);

  //
  // cuSOLVER factorization
  //
  ret = cusolverSpDcsrcholFactor(h_cusolver_, m, nnz_, mat_descr_, values_, rowptr_, colind_, info_, buf_fact_);
  if(ret != CUSOLVER_STATUS_SUCCESS) {
    // this does not return error when the factorization fails numerically
    nlp_->log->printf(hovWarning, "hiopLinSolverCholCuSparse: factorization failed: CUSOLVER_STATUS=%d.\n", ret);
    return -1;
  }
  nlp_->runStats.linsolv.tmFactTime.stop();

  //
  // check for zero or negative pivots
  //
  nlp_->runStats.linsolv.tmInertiaComp.start();
  const double zero_piv_tol = 1e-24;
  int position = -1;
  ret = cusolverSpDcsrcholZeroPivot(h_cusolver_, info_, zero_piv_tol, &position);
  nlp_->runStats.linsolv.tmInertiaComp.stop();

  if(position >= 0) {
    nlp_->log->printf(hovWarning, "hiopLinSolverCholCuSparse: the %dth pivot is <=%.5e\n", position, zero_piv_tol);
    return -1;
  }
  return 0;
}

bool hiopLinSolverCholCuSparse::solve(hiopVector& x_in)
{
  hiopTimer t;
  cusolverStatus_t ret;

  size_type m = M_->m();
  assert(m == x_in.get_size());

  if(!rhs_buf1_) {
    cudaMalloc(&rhs_buf1_, m * sizeof(double));
  }
  if(!rhs_buf2_) {
    cudaMalloc(&rhs_buf2_, m * sizeof(double));
  }

  nlp_->runStats.linsolv.tmDeviceTransfer.start();
  cudaMemcpy(rhs_buf1_, x_in.local_data(), m * sizeof(double), cudaMemcpyHostToDevice);
  nlp_->runStats.linsolv.tmDeviceTransfer.stop();

  nlp_->runStats.linsolv.tmTriuSolves.start();
  // b = P*b
  permute_vec(m, rhs_buf1_, P_, rhs_buf2_);

  //
  // solve -> two triangular solves
  //
  ret = cusolverSpDcsrcholSolve(h_cusolver_, m, rhs_buf2_, rhs_buf1_, info_, buf_fact_);

  // x = P'*x
  permute_vec(m, rhs_buf1_, PT_, rhs_buf2_);
  nlp_->runStats.linsolv.tmTriuSolves.stop();

  // transfer to host
  nlp_->runStats.linsolv.tmDeviceTransfer.start();
  cudaMemcpy(x_in.local_data(), rhs_buf2_, m * sizeof(double), cudaMemcpyDeviceToHost);
  nlp_->runStats.linsolv.tmDeviceTransfer.stop();

  if(ret != CUSOLVER_STATUS_SUCCESS) {
    nlp_->log->printf(hovWarning, "hiopLinSolverCholCuSparse: solve failed: CUSOLVER_STATUS=%d.\n", ret);
    return false;
  }

  return true;
}

bool hiopLinSolverCholCuSparse::permute_vec(int n, double* vec_in, index_type* perm, double* vec_out)
{
  cusparseStatus_t ret;
#if CUSPARSE_VERSION >= 11400
  // the descr of the array going to be permuted
  cusparseSpVecDescr_t v_out;
  // original nonzeros
  cusparseDnVecDescr_t v_in;

  // Create sparse vector (output)
  ret = cusparseCreateSpVec(&v_out, n, n, perm, vec_out, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  assert(CUSPARSE_STATUS_SUCCESS == ret);

  // Create dense vector (input)
  ret = cusparseCreateDnVec(&v_in, n, vec_in, CUDA_R_64F);
  assert(CUSPARSE_STATUS_SUCCESS == ret);

  ret = cusparseGather(h_cusparse_, v_in, v_out);
  assert(CUSPARSE_STATUS_SUCCESS == ret);

  cusparseDestroySpVec(v_out);
  cusparseDestroyDnVec(v_in);

#else  // CUSPARSE_VERSION < 11400

  ret = cusparseDgthr(h_cusparse_, n, vec_in, vec_out, perm, CUSPARSE_INDEX_BASE_ZERO);
  assert(CUSPARSE_STATUS_SUCCESS == ret);
#endif
  return (CUSPARSE_STATUS_SUCCESS == ret);
}

}  // namespace hiop

#endif  // HIOP_USE_CUDA
