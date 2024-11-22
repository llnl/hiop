// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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

#ifndef HIOP_LINSOLVER_MA57
#define HIOP_LINSOLVER_MA57

#include "hiopLinSolver.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "hiopMatrixSparseCSRSeq.hpp"
#include "FortranCInterface.hpp"

#define MA57ID FC_GLOBAL(ma57id, MA57ID)
#define MA57AD FC_GLOBAL(ma57ad, MA57AD)
#define MA57BD FC_GLOBAL(ma57bd, MA57BD)
#define MA57CD FC_GLOBAL(ma57cd, MA57CD)
#define MA57DD FC_GLOBAL(ma57dd, MA57DD)
#define MA57ED FC_GLOBAL(ma57ed, MA57ED)

/** implements the linear solver class using the HSL MA57 solver
 *
 * @ingroup LinearSolvers
 */

namespace hiop
{

extern "C" {
void MA57ID(double cntl[], int icntl[]);

void MA57AD(int* n,
            int* ne,
            int irn[],
            int jcn[],
            int* lkeep,
            int keep[],
            int iwork[],
            int icntl[],
            int info[],
            double rinfo[]);

void MA57BD(int* n,
            int* ne,
            double a[],
            double fact[],
            int* lfact,
            int ifact[],
            int* lifact,
            int* lkeep,
            int keep[],
            int ppos[],
            int* icntl,
            double cntl[],
            int info[],
            double rinfo[]);
void MA57CD(int* job,
            int* n,
            double fact[],
            int* lfact,
            int ifact[],
            int* lifact,
            int* nrhs,
            double rhs[],
            int* lrhs,
            double w[],
            int* lw,
            int iw1[],
            int icntl[],
            int info[]);
void MA57DD(int* job,
            int* n,
            int* ne,
            double a[],
            int irn[],
            int jcn[],
            double fact[],
            int* lfact,
            int ifact[],
            int* lifact,
            double rhs[],
            double x[],
            double resid[],
            double w[],
            int iw[],
            int icntl[],
            double cntl[],
            int info[],
            double rinfo[]);
void MA57ED(int* n,
            int* ic,
            int keep[],
            double fact[],
            int* lfact,
            double* newfac,
            int* lnew,
            int ifact[],
            int* lifact,
            int newifc[],
            int* linew,
            int* info);
}

/**
 * Wrapper class for using MA57 solver to solve symmetric sparse indefinite KKT linearizations.
 *
 * This class uses a triplet sparse matrix (member `M_`) to store the KKT linear system. This matrix
 * is populated by the KKT linsys classes.
 */
class hiopLinSolverSymSparseMA57 : public hiopLinSolverSymSparse
{
public:
  /// Constructor that allocates and ownes the system matrix
  hiopLinSolverSymSparseMA57(const int& n, const int& nnz, hiopNlpFormulation* nlp);

  /**
   * Constructor that does not create nor owns the system matrix. Used by specializations of this
   * class that takes CSR matrix as input.
   */
  hiopLinSolverSymSparseMA57(hiopMatrixSparse* M, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymSparseMA57();

protected:
  hiopLinSolverSymSparseMA57() = delete;

  /// Method holding the code common to the constructors. Initializes MA57 global parameters
  void constructor_part();

public:
  /** Triggers a refactorization of the matrix, if necessary.
   * Overload from base class. */
  int matrixChanged();

  /**
   * Solves a linear system.
   * @param x is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  bool solve(hiopVector& x_);

protected:
  /**
   * Fill `irowM_` and `jcolM_` by copying row and col indexes from the member matrix `M_`. Overridden by
   * specialized classes, such as the one that takes as input a CSR matrix.
   *
   * Note: the indexes should be only for the lower or only for the upper triangular part, as per MA57
   * requirement. Also, the indexes should be 1-based.
   */
  virtual void fill_triplet_index_arrays()
  {
    assert(nnz_ == M_->numberOfNonzeros());
    for(int k = 0; k < nnz_; k++) {
      irowM_[k] = M_->i_row()[k] + 1;
      jcolM_[k] = M_->j_col()[k] + 1;
    }
  }

  /// Return the pointer to array of triplet values that should be passed to MA57
  virtual double* get_triplet_values_array() { return M_->M(); }

  /// Fill the array passed as argument with the triplet nonzeros referred to by `irowM_` and `jcolM_`
  virtual void fill_triplet_values_array(double* values_triplet)
  {
    // no fill is required for this base "triplet" implementation since the triplet array is already populated
    assert(dynamic_cast<hiopMatrixSparseTriplet*>(M_));
    assert(M_->M() == values_triplet);  // pointers should coincide
  }

protected:
  int icntl_[20];
  int info_[40];
  double cntl_[5];
  double rinfo_[20];

  int n_;    // dimension of the whole matrix
  int nnz_;  // number of nonzeros in the matrix

  /// row indexes used by the factorization
  int* irowM_;

  /// col indexes used by the factorization
  int* jcolM_;
  // note: the values array is reused (from the sys matrix)

  int lkeep_;          // temporary storage
  int* keep_;          // temporary storage
  int lifact_;         // temporary storage
  int* ifact_;         // temporary storage
  int lfact_;          // temporary storage for the factorization process
  double* fact_;       // storage for the factors
  double ipessimism_;  // amounts by which to increase allocated factorization space
  double rpessimism_;  // amounts by which to increase allocated factorization space

  int* iwork_;
  double* dwork_;

  /// Right-hand side working array
  hiopVector* rhs_;

  /// Working array used for residual computation
  hiopVector* resid_;

  /// parameters to control pivoting
  double pivot_tol_;
  double pivot_max_;
  bool pivot_changed_;

public:
  /**
   * Called the very first time a matrix is factorized, this method allocates space for the
   * factorization and performs ordering.
   */
  virtual void firstCall();

  // increase pivot tolarence
  virtual bool increase_pivot_tol();
};

/**
 * MA57 solver class that takes CSR sparse input and offers the boilerplate to copy this into the internal
 * triplet matrix used with MA57 API.
 *
 * The CSR matrix is understood to be symmetric. The underlying CSR storage can contain all the nonzero entries
 * or only the lower triangular part. In both cases, this class will copy ONLY the lower triangular entries to
 * the underlying triplet storage.
 */
class hiopLinSolverSparseCsrMa57 : public hiopLinSolverSymSparseMA57
{
public:
  /// Constructor that takes a CSR matrix as input.
  hiopLinSolverSparseCsrMa57(hiopMatrixSparseCSRSeq* csr_in, hiopNlpFormulation* nlp_in)
      : hiopLinSolverSymSparseMA57(csr_in, nlp_in),  // csr input pointer not owned
        values_(nullptr)
  {
    // count nnz for the lower triangle in the csr input
    index_type* i_rowptr = M_->i_row();
    index_type* j_colidx = M_->j_col();

    n_ = M_->m();

    nnz_ = 0;
    for(int r = 0; r < n_; ++r) {
      for(int itnz = i_rowptr[r]; itnz < i_rowptr[r + 1]; ++itnz) {
        if(r >= j_colidx[itnz]) {
          nnz_++;
        }
      }
    }
    values_ = new double[nnz_];
  }

  virtual ~hiopLinSolverSparseCsrMa57() { delete[] values_; }

protected:
  hiopLinSolverSparseCsrMa57() = delete;

  /**
   * Fill `irowM_` and `jcolM_` by copying row and col indexes from the CSR matrix `mat_csr_`.
   *
   * Note: the indexes should be only for the lower or only for the upper triangular part, as per MA57
   * requirement. Also, the indexes should be 1-based.
   */
  virtual void fill_triplet_index_arrays()
  {
    assert(n_ == M_->m());
    assert(nnz_ <= M_->numberOfNonzeros());
    index_type* i_rowptr = M_->i_row();
    index_type* j_colidx = M_->j_col();

#ifdef HIOP_DEEPCHECKS
    bool is_upper_tri = true;
    for(index_type r = 0; r < n_ && is_upper_tri; ++r) {
      for(index_type itnz = i_rowptr[r]; itnz < i_rowptr[r + 1]; ++itnz) {
        if(r > j_colidx[itnz]) {
          is_upper_tri = false;
          break;
        }
      }
    }
    if(is_upper_tri) {
      nlp_->log->printf(hovWarning,
                        "MA57 expects full or lower triangular CSR KKT matrix. Input CSR is detected "
                        "to be upper triangular. [hiopLinSolverSparseCsrMa57 HIOP_DEEPCHECKS]\n");
    }
#endif
    index_type nnz_triplet = 0;

    for(index_type r = 0; r < n_; ++r) {
      for(index_type itnz = i_rowptr[r]; itnz < i_rowptr[r + 1]; ++itnz) {
        if(r >= j_colidx[itnz]) {
          irowM_[nnz_triplet] = r + 1;
          jcolM_[nnz_triplet] = j_colidx[itnz] + 1;
          nnz_triplet++;
        }
      }
    }
    assert(nnz_ == nnz_triplet);
  }

  /// Fill the array passed as argument with nonzeros corresponding to triplet entries from `irowM_` and `jcolM_`
  virtual void fill_triplet_values_array(double* values_triplet)
  {
    assert(values_ == values_triplet);  // pointers should coincide
    // copy lower triangular elements from M_ (which is CSR) to values_
    assert(n_ == M_->m());
    assert(nnz_ <= M_->numberOfNonzeros());
    const index_type* i_rowptr = M_->i_row();
    const index_type* j_colidx = M_->j_col();
    const double* Mvals = M_->M();
    index_type nnz_triplet = 0;
    for(index_type r = 0; r < n_; ++r) {
      for(index_type itnz = i_rowptr[r]; itnz < i_rowptr[r + 1]; ++itnz) {
        if(r >= j_colidx[itnz]) {
          values_triplet[nnz_triplet++] = Mvals[itnz];
        }
      }
    }
    assert(nnz_ == nnz_triplet);
  }

  /// Return the pointer to array of triplet values that should be passed to MA57
  virtual double* get_triplet_values_array() { return values_; }

protected:
  double* values_;
};

}  // namespace hiop
#endif
