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

/* implements the linear solver class using the PARDISO solver
 * @file hiopLinSolverSparsePARDISO.cpp
 * @ingroup LinearSolvers
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 */

#include "hiopLinSolverSparsePARDISO.hpp"

#include "hiop_blasdefs.hpp"
#include <iostream>
#include <cstdlib>

namespace hiop
{
/*
 *  PARDISO for symmetric indefinite sparse matrix
 */
hiopLinSolverSymSparsePARDISO::hiopLinSolverSymSparsePARDISO(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      rhs_{nullptr},
      n_{n},
      nnz_{-1},
      is_initialized_{false}
{
  maxfct_ = 1;  // max number of fact having same sparsity pattern to keep at the same time
  mnum_ = 1;    // actual matrix (as in index from 1 to maxfct)
  msglvl_ = 0;  // messaging level
  mtype_ = -2;  // real and symmetric indefinite
  solver_ = 0;  // sparse direct solver
}

hiopLinSolverSymSparsePARDISO::~hiopLinSolverSymSparsePARDISO()
{
  /* Termination and release of memory */
  int phase = -1; /* Release internal memory . */
  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            NULL,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);

  if(kRowPtr_) delete[] kRowPtr_;
  if(jCol_) delete[] jCol_;
  if(kVal_) delete[] kVal_;
  if(index_covert_CSR2Triplet_) delete[] index_covert_CSR2Triplet_;
  if(index_covert_extra_Diag2CSR_) delete[] index_covert_extra_Diag2CSR_;

  if(rhs_) {
    delete rhs_;
  }
}

void hiopLinSolverSymSparsePARDISO::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  kRowPtr_ = new int[n_ + 1]{0};
  nnz_ = 0;

  // transfer triplet form to CSR upper triangular form
  // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the
  // additional diagonal elememts the 1st part is sorted by row
  {
    //
    // compute nnz in each row
    //
    // off-diagonal part
    kRowPtr_[0] = 0;
    for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
      if(M_->i_row()[k] != M_->j_col()[k]) {
        kRowPtr_[M_->j_col()[k] + 1]++;
        nnz_ += 1;
      }
    }
    // diagonal part
    for(int i = 0; i < n_; i++) {
      kRowPtr_[i + 1]++;
      nnz_ += 1;
    }
    // get correct row ptr index
    for(int i = 1; i < n_ + 1; i++) {
      kRowPtr_[i] += kRowPtr_[i - 1];
    }
    assert(nnz_ == kRowPtr_[n_]);

    kVal_ = new double[nnz_]{0.0};
    jCol_ = new int[nnz_]{0};
  }
  {
    //
    // set correct col index and value
    //
    index_covert_CSR2Triplet_ = new int[nnz_];
    index_covert_extra_Diag2CSR_ = new int[n_];

    int* nnz_each_row_tmp = new int[n_]{0};
    int total_nnz_tmp{0};
    int nnz_tmp{0};
    int rowID_tmp;
    int colID_tmp;
    for(int k = 0; k < n_; k++) {
      index_covert_extra_Diag2CSR_[k] = -1;
    }

    for(int k = 0; k < M_->numberOfNonzeros() - n_; k++) {
      rowID_tmp = M_->i_row()[k];
      colID_tmp = M_->j_col()[k];
      if(rowID_tmp == colID_tmp) {
        nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
        jCol_[nnz_tmp] = colID_tmp;
        kVal_[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        kVal_[nnz_tmp] += M_->M()[M_->numberOfNonzeros() - n_ + rowID_tmp];
        index_covert_extra_Diag2CSR_[rowID_tmp] = nnz_tmp;

        nnz_each_row_tmp[rowID_tmp]++;
        total_nnz_tmp++;
      } else {
        nnz_tmp = nnz_each_row_tmp[colID_tmp] + kRowPtr_[colID_tmp];
        jCol_[nnz_tmp] = rowID_tmp;
        kVal_[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        nnz_each_row_tmp[colID_tmp]++;
        total_nnz_tmp += 1;
      }
    }

    // correct the missing diagonal term
    for(int i = 0; i < n_; i++) {
      if(nnz_each_row_tmp[i] != kRowPtr_[i + 1] - kRowPtr_[i]) {
        assert(nnz_each_row_tmp[i] == kRowPtr_[i + 1] - kRowPtr_[i] - 1);
        nnz_tmp = nnz_each_row_tmp[i] + kRowPtr_[i];
        jCol_[nnz_tmp] = i;
        kVal_[nnz_tmp] = M_->M()[M_->numberOfNonzeros() - n_ + i];
        index_covert_CSR2Triplet_[nnz_tmp] = M_->numberOfNonzeros() - n_ + i;
        total_nnz_tmp += 1;

        std::vector<int> ind_temp(kRowPtr_[i + 1] - kRowPtr_[i]);
        std::iota(ind_temp.begin(), ind_temp.end(), 0);
        std::sort(ind_temp.begin(), ind_temp.end(), [&](int a, int b) {
          return jCol_[a + kRowPtr_[i]] < jCol_[b + kRowPtr_[i]];
        });

        reorder(kVal_ + kRowPtr_[i], ind_temp, kRowPtr_[i + 1] - kRowPtr_[i]);
        reorder(index_covert_CSR2Triplet_ + kRowPtr_[i], ind_temp, kRowPtr_[i + 1] - kRowPtr_[i]);
        std::sort(jCol_ + kRowPtr_[i], jCol_ + kRowPtr_[i + 1]);
      }
    }

    delete[] nnz_each_row_tmp;
  }

  // need Fortran indexes
  for(int i = 0; i < n_ + 1; i++) {
    kRowPtr_[i] += 1;
  }
  for(int i = 0; i < nnz_; i++) {
    jCol_[i] += 1;
  }

  /* initialize PARDISO */
  pardisoinit_d(pt_, &mtype_, &solver_, iparm_, dparm_, &error_);
  if(error_ != 0) {
    std::cout << "PardisoSolver ERROR during pardisoinit:" << error_ << "." << std::endl;
    assert(false);
  }

  /* Numbers of processors, value of OMP_NUM_THREADS */
  char* var = getenv("OMP_NUM_THREADS");
  if(var != NULL) {
    sscanf(var, "%d", &num_threads_);
  } else {
    num_threads_ = 1;
  }

  iparm_[2] = num_threads_;
  iparm_[1] = 2;   // 2 is for metis, 0 for min degree
  iparm_[7] = 3;   // # iterative refinements
  iparm_[10] = 1;  // scaling for IPM KKT; used with IPARM(13)=1 or 2
  iparm_[12] = 2;  // improved accuracy for IPM KKT; used with IPARM(11)=1;
                   // if needed, use 2 for advanced matchings and higer accuracy.
  iparm_[23] = 1;  // Parallel Numerical Factorization
                   // (0=used in the last years, 1=two-level scheduling)

  /* symbolic analysis from PARDISO */
  int phase = 11;  // analysis
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);
  if(error_ != 0) {
    printf("PardisoSolver - ERROR during symbolic factorization: %d\n", error_);
    assert(false);
  }
}

int hiopLinSolverSymSparsePARDISO::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(!is_initialized_) {
    this->firstCall();
    is_initialized_ = true;
  } else {
    // update matrix
    int rowID_tmp{0};
    for(int k = 0; k < nnz_; k++) {
      kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(int i = 0; i < n_; i++) {
      if(index_covert_extra_Diag2CSR_[i] != -1)
        kVal_[index_covert_extra_Diag2CSR_[i]] += M_->M()[M_->numberOfNonzeros() - n_ + i];
    }
  }

  /* do numerical factorization */
  int phase = 22;
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);

  if(error_ != 0) {
    printf("PardisoSolver - ERROR during numerical factorization: %d\n", error_);
    assert(false);
  }

  nlp_->runStats.linsolv.tmInertiaComp.start();

  int negEigVal = iparm_[22];
  if(iparm_[21] + iparm_[22] != n_) {
    // singular matrix
    negEigVal = -1;
  }

  nlp_->runStats.linsolv.tmInertiaComp.stop();

  return negEigVal;
}

bool hiopLinSolverSymSparsePARDISO::solve(hiopVector& b)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(b.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  /* do backsolve */
  hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&b);
  assert(x != nullptr);
  if(rhs_ == nullptr) {
    rhs_ = dynamic_cast<hiopVectorPar*>(x->new_copy());
  } else {
    rhs_->copyFrom(*x);
  }
  double* dx = x->local_data();
  double* drhs = rhs_->local_data();

  int phase = 33;
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            drhs,
            dx,
            &error_,
            dparm_);

  if(error_ != 0) {
    printf("PardisoSolver - ERROR during backsolve: %d\n", error_);
    assert(false);
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return 1;
}

/*
 *  PARDISO for unsymmetric sparse matrix
 */
hiopLinSolverNonSymSparsePARDISO::hiopLinSolverNonSymSparsePARDISO(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      rhs_{nullptr},
      n_{n},
      nnz_{-1},
      is_initialized_{false}
{
  maxfct_ = 1;  // max number of fact having same sparsity pattern to keep at the same time
  mnum_ = 1;    // actual matrix (as in index from 1 to maxfct)
  msglvl_ = 0;  // messaging level
  mtype_ = 11;  // real and unsymmetric
  solver_ = 0;  // sparse direct solver
}

hiopLinSolverNonSymSparsePARDISO::~hiopLinSolverNonSymSparsePARDISO()
{
  /* Termination and release of memory */
  int phase = -1; /* Release internal memory . */
  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            NULL,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);

  if(kRowPtr_) delete[] kRowPtr_;
  if(jCol_) delete[] jCol_;
  if(kVal_) delete[] kVal_;
  if(index_covert_CSR2Triplet_) delete[] index_covert_CSR2Triplet_;
  if(index_covert_extra_Diag2CSR_) delete[] index_covert_extra_Diag2CSR_;

  if(rhs_) {
    delete rhs_;
  }
}

void hiopLinSolverNonSymSparsePARDISO::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  // transfer triplet form to CSR form
  // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the
  // additional diagonal elememts the 1st part is sorted by row

  M_->convert_to_csr_arrays(nnz_,
                            &kRowPtr_,
                            &jCol_,
                            &kVal_,
                            &index_covert_CSR2Triplet_,
                            &index_covert_extra_Diag2CSR_,
                            extra_diag_nnz_map);

  // need Fortran indexes
  for(int i = 0; i < n_ + 1; i++) {
    kRowPtr_[i] += 1;
  }
  for(int i = 0; i < nnz_; i++) {
    jCol_[i] += 1;
  }

  /* initialize PARDISO */
  pardisoinit_d(pt_, &mtype_, &solver_, iparm_, dparm_, &error_);
  if(error_ != 0) {
    std::cout << "PardisoSolver ERROR during pardisoinit:" << error_ << "." << std::endl;
    assert(false);
  }

  /* Numbers of processors, value of OMP_NUM_THREADS */
  char* var = getenv("OMP_NUM_THREADS");
  if(var != NULL) {
    sscanf(var, "%d", &num_threads_);
  } else {
    num_threads_ = 1;
  }

  iparm_[2] = num_threads_;
  iparm_[1] = 2;   // 2 is for metis, 0 for min degree
  iparm_[7] = 3;   // # iterative refinements
  iparm_[10] = 1;  // scaling for IPM KKT; used with IPARM(13)=1 or 2
  iparm_[12] = 2;  // improved accuracy for IPM KKT; used with IPARM(11)=1;
                   // if needed, use 2 for advanced matchings and higer accuracy.
  iparm_[23] = 1;  // Parallel Numerical Factorization
                   // (0=used in the last years, 1=two-level scheduling)

  /* symbolic analysis from PARDISO */
  int phase = 11;  // analysis
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);
  if(error_ != 0) {
    printf("PardisoSolver - ERROR during symbolic factorization: %d\n", error_);
    assert(false);
  }
}

int hiopLinSolverNonSymSparsePARDISO::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(!is_initialized_) {
    this->firstCall();
    is_initialized_ = true;
  } else {
    // update matrix
    int rowID_tmp{0};
    for(int k = 0; k < nnz_; k++) {
      kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(auto p: extra_diag_nnz_map) {
      kVal_[p.first] += M_->M()[p.second];
    }
  }

  /* do numerical factorization */
  int phase = 22;
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            NULL,
            NULL,
            &error_,
            dparm_);

  if(error_ != 0) {
    printf("PardisoSolver - ERROR during numerical factorization: %d\n", error_);
    assert(false);
  }

  nlp_->runStats.linsolv.tmInertiaComp.start();
  nlp_->runStats.linsolv.tmInertiaComp.stop();

  return 0;
}

bool hiopLinSolverNonSymSparsePARDISO::solve(hiopVector& b)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(b.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  /* do backsolve */
  hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&b);
  assert(x != nullptr);
  if(rhs_ == nullptr) {
    rhs_ = dynamic_cast<hiopVectorPar*>(x->new_copy());
  } else {
    rhs_->copyFrom(*x);
  }
  double* dx = x->local_data();
  double* drhs = rhs_->local_data();

  int phase = 33;
  int nrhs = 1;

  pardiso_d(pt_,
            &maxfct_,
            &mnum_,
            &mtype_,
            &phase,
            &n_,
            kVal_,
            kRowPtr_,
            jCol_,
            NULL,
            &nrhs,
            iparm_,
            &msglvl_,
            drhs,
            dx,
            &error_,
            dparm_);

  if(error_ != 0) {
    printf("PardisoSolver - ERROR during backsolve: %d\n", error_);
    return false;
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  return true;
}

}  // end namespace hiop
