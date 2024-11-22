// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC.
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
 * @file hiopLinSolverSparseSTRUMPACK.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopLinSolverSparseSTRUMPACK.hpp"

#include "hiop_blasdefs.hpp"

using namespace strumpack;

namespace hiop
{
hiopLinSolverSymSparseSTRUMPACK::hiopLinSolverSymSparseSTRUMPACK(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
{}

hiopLinSolverSymSparseSTRUMPACK::~hiopLinSolverSymSparseSTRUMPACK()
{
  if(kRowPtr_) delete[] kRowPtr_;
  if(jCol_) delete[] jCol_;
  if(kVal_) delete[] kVal_;
  if(index_covert_CSR2Triplet_) delete[] index_covert_CSR2Triplet_;
  if(index_covert_extra_Diag2CSR_) delete[] index_covert_extra_Diag2CSR_;
}

void hiopLinSolverSymSparseSTRUMPACK::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  kRowPtr_ = new int[n_ + 1]{0};

  // transfer triplet form to CSR form
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
        kRowPtr_[M_->i_row()[k] + 1]++;
        kRowPtr_[M_->j_col()[k] + 1]++;
        nnz_ += 2;
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
    int total_nnz_tmp{0}, nnz_tmp{0}, rowID_tmp, colID_tmp;
    for(int k = 0; k < n_; k++) index_covert_extra_Diag2CSR_[k] = -1;

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
        nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
        jCol_[nnz_tmp] = colID_tmp;
        kVal_[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        nnz_tmp = nnz_each_row_tmp[colID_tmp] + kRowPtr_[colID_tmp];
        jCol_[nnz_tmp] = rowID_tmp;
        kVal_[nnz_tmp] = M_->M()[k];
        index_covert_CSR2Triplet_[nnz_tmp] = k;

        nnz_each_row_tmp[rowID_tmp]++;
        nnz_each_row_tmp[colID_tmp]++;
        total_nnz_tmp += 2;
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

  /*
   * initialize strumpack parameters
   */

  // possible values for MatchingJob (from STRUMPACK's source code)
  // NONE,                         /*!< Don't do anything, but it can provide inertia info
  //                                   MC64 provides non-symmetric permutation and hence inertia is inaccurate */
  // MAX_CARDINALITY,              /*!< Maximum cardinality                 */
  // MAX_SMALLEST_DIAGONAL,        /*!< Maximum smallest diagonal value     */
  // MAX_SMALLEST_DIAGONAL_2,      /*!< Same as MAX_SMALLEST_DIAGONAL,
  //                                but different algorithm                */
  // MAX_DIAGONAL_SUM,             /*!< Maximum sum of diagonal values      */
  // MAX_DIAGONAL_PRODUCT_SCALING, /*!< Maximum product of diagonal values
  //                               and row and column scaling             */
  // COMBBLAS                      /*!< Use AWPM from CombBLAS              */
  //
  // If you encounter numerical issues during the factorization
  //(such as small pivots, failure in LU factorization), you can
  // also try a different matching (MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING
  // is the most robust option)

  spss.options().set_matching(MatchingJob::NONE);
  spss.options().enable_METIS_NodeNDP();

  if(nlp_->options->GetString("compute_mode") == "cpu") {
    spss.options().disable_gpu();
  }
  spss.options().set_verbose(false);

  spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);
  //    spss.reorder(n_, n_);
}

int hiopLinSolverSymSparseSTRUMPACK::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  int num_neg_eig_val{0};
  int num_zero_eig_val{0};
  int num_pos_eig_val{0};

  nlp_->runStats.linsolv.tmFactTime.start();

  if(!kRowPtr_) {
    this->firstCall();
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

    spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);
  }

  strumpack::ReturnCode retval = spss.factor();  // not really necessary, called if needed by solve

  if(strumpack::ReturnCode::ZERO_PIVOT == retval) {
    return -1;
  } else if(strumpack::ReturnCode::SUCCESS == retval) {
    retval = spss.inertia(num_neg_eig_val, num_zero_eig_val, num_pos_eig_val);
    if(strumpack::ReturnCode::SUCCESS == retval) {
      num_neg_eig_val = num_neg_eig_val;
    } else {
      if(nlp_->options->GetString("fact_acceptor") == "inertia_correction") {
        nlp_->log->printf(hovError,
                          "strumpack: failed to provide accurate inertia infomation. Please use inertia-free approach.");
      }
    }
  } else {
    // unknown error
    assert(false && "unknown error from strumpack factor()");
  }

  nlp_->runStats.linsolv.tmInertiaComp.start();
  nlp_->runStats.linsolv.tmInertiaComp.stop();
  return num_neg_eig_val;
}

bool hiopLinSolverSymSparseSTRUMPACK::solve(hiopVector& x_)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(x_.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
  assert(x != NULL);
  hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
  double* dx = x->local_data();
  double* drhs = rhs->local_data();

  spss.solve(drhs, dx);

  nlp_->runStats.linsolv.tmTriuSolves.stop();

  delete rhs;
  rhs = nullptr;
  return 1;
}

hiopLinSolverNonSymSparseSTRUMPACK::hiopLinSolverNonSymSparseSTRUMPACK(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
      kRowPtr_{nullptr},
      jCol_{nullptr},
      kVal_{nullptr},
      index_covert_CSR2Triplet_{nullptr},
      index_covert_extra_Diag2CSR_{nullptr},
      n_{n},
      nnz_{0}
{}

hiopLinSolverNonSymSparseSTRUMPACK::~hiopLinSolverNonSymSparseSTRUMPACK()
{
  if(kRowPtr_) delete[] kRowPtr_;
  if(jCol_) delete[] jCol_;
  if(kVal_) delete[] kVal_;
  if(index_covert_CSR2Triplet_) delete[] index_covert_CSR2Triplet_;
  if(index_covert_extra_Diag2CSR_) delete[] index_covert_extra_Diag2CSR_;
}

void hiopLinSolverNonSymSparseSTRUMPACK::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  // transfer triplet form to CSR form
  // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the
  // additional diagonal elememts the 1st part is sorted by row
  hiop::hiopMatrixSparseTriplet* M_triplet = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(M_);
  if(M_triplet == nullptr) {
    nlp_->log->printf(hovError, "M_triplet is nullptr");
    return;
  }

  M_triplet->convert_to_csr_arrays(nnz_,
                                   &kRowPtr_,
                                   &jCol_,
                                   &kVal_,
                                   &index_covert_CSR2Triplet_,
                                   &index_covert_extra_Diag2CSR_,
                                   extra_diag_nnz_map);

  /*
   * initialize strumpack parameters
   */

  // possible values for MatchingJob (from STRUMPACK's source code)
  // NONE,                         /*!< Don't do anything                   */
  // MAX_CARDINALITY,              /*!< Maximum cardinality                 */
  // MAX_SMALLEST_DIAGONAL,        /*!< Maximum smallest diagonal value     */
  // MAX_SMALLEST_DIAGONAL_2,      /*!< Same as MAX_SMALLEST_DIAGONAL,
  //                                but different algorithm                */
  // MAX_DIAGONAL_SUM,             /*!< Maximum sum of diagonal values      */
  // MAX_DIAGONAL_PRODUCT_SCALING, /*!< Maximum product of diagonal values
  //                               and row and column scaling             */
  // COMBBLAS                      /*!< Use AWPM from CombBLAS              */
  //
  // If you encounter numerical issues during the factorization
  //(such as small pivots, failure in LU factorization), you can
  // also try a different matching (MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING
  // is the most robust option)

  // spss.options().set_matching(MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING);
  // spss.options().enable_METIS_NodeNDP();
  if(nlp_->options->GetString("compute_mode") == "cpu") {
    spss.options().disable_gpu();
  }
  spss.options().set_verbose(false);

  spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);
  //    spss.reorder(n_, n_);
}

int hiopLinSolverNonSymSparseSTRUMPACK::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  if(!kRowPtr_) {
    this->firstCall();
  } else {
    // update matrix
    int rowID_tmp{0};
    for(int k = 0; k < nnz_; k++) {
      kVal_[k] = M_->M()[index_covert_CSR2Triplet_[k]];
    }
    for(auto p: extra_diag_nnz_map) {
      kVal_[p.first] += M_->M()[p.second];
    }

    spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);
  }

  spss.factor();  // not really necessary, called if needed by solve

  nlp_->runStats.linsolv.tmInertiaComp.start();
  nlp_->runStats.linsolv.tmInertiaComp.stop();
  return 0;
}

bool hiopLinSolverNonSymSparseSTRUMPACK::solve(hiopVector& x_)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(x_.get_size() == M_->n());

  nlp_->runStats.linsolv.tmTriuSolves.start();

  hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
  assert(x != NULL);
  hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
  double* dx = x->local_data();
  double* drhs = rhs->local_data();

  spss.solve(drhs, dx);

  nlp_->runStats.linsolv.tmTriuSolves.stop();
  delete rhs;
  rhs = nullptr;
  return 1;
}

}  // end namespace hiop
