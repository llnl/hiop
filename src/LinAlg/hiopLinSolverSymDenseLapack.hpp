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

#ifndef HIOP_LINSOLVER_LAPACK
#define HIOP_LINSOLVER_LAPACK

#include "hiopLinSolver.hpp"

namespace hiop
{

/** Wrapper for LAPACK's DSYTRF */
class hiopLinSolverSymDenseLapack : public hiopLinSolverSymDense
{
public:
  hiopLinSolverSymDenseLapack(int n, hiopNlpFormulation* nlp)
      : hiopLinSolverSymDense(n, nlp)
  {
    ipiv = new int[n];
    // a "legacy" hiopVector within in the CPU memory space is sufficient
    dwork = LinearAlgebraFactory::create_vector("DEFAULT", 0);
  }
  virtual ~hiopLinSolverSymDenseLapack()
  {
    delete[] ipiv;
    delete dwork;
  }

  /** Triggers a refactorization of the matrix, if necessary.
   * Overload from base class. */
  int matrixChanged()
  {
    assert(M_->n() == M_->m());
    int N = M_->n(), lda = N, info;
    if(N == 0) return 0;

    nlp_->runStats.linsolv.tmFactTime.start();

    double dwork_tmp;
    char uplo = 'L';  // M is upper in C++ so it's lower in fortran

    //
    // query sizes
    //
    int lwork = -1;
    DSYTRF(&uplo, &N, M_->local_data(), &lda, ipiv, &dwork_tmp, &lwork, &info);
    assert(info == 0);

    lwork = (int)dwork_tmp;
    if(lwork != dwork->get_size()) {
      delete dwork;
      dwork = LinearAlgebraFactory::create_vector("DEFAULT", lwork);
    }

    //
    // factorization
    //
    DSYTRF(&uplo, &N, M_->local_data(), &lda, ipiv, dwork->local_data(), &lwork, &info);
    if(info < 0) {
      nlp_->log->printf(hovError, "hiopLinSolverSymDense error: %d argument to dsytrf has an illegal value.\n", -info);
      return -1;
    } else {
      if(info > 0) {
        nlp_->log->printf(hovWarning,
                          "hiopLinSolverSymDense error: %d entry in the factorization's diagonal\n"
                          "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
                          info);
        // matrix is singular
        return -1;
      }
    }
    assert(info == 0);
    nlp_->runStats.linsolv.tmFactTime.stop();

    nlp_->runStats.linsolv.tmInertiaComp.start();
    //
    // Compute the inertia. Only negative eigenvalues are returned.
    // Code originally written by M. Schanenfor PIPS based on
    // LINPACK's dsidi Fortran routine (http://www.netlib.org/linpack/dsidi.f)
    // 04/08/2020 - petra: fixed the test for non-positive pivots (was only for negative pivots)
    int negEigVal = 0;
    [[maybe_unused]] int posEigVal = 0;
    int nullEigVal = 0;
    double t = 0;
    double* MM = M_->local_data();

    for(int k = 0; k < N; k++) {
      // c       2 by 2 block
      // c       use det (d  s)  =  (d/t * c - t) * t  ,  t = dabs(s)
      // c               (s  c)
      // c       to avoid underflow/overflow troubles.
      // c       take two passes through scaling.  use  t  for flag.
      double d = MM[k * N + k];
      if(ipiv[k] <= 0) {
        if(t == 0) {
          assert(k + 1 < N);
          if(k + 1 < N) {
            t = fabs(MM[k * N + k + 1]);
            d = (d / t) * MM[(k + 1) * N + k + 1] - t;
          }
        } else {
          d = t;
          t = 0.;
        }
      }
      // printf("d = %22.14e \n", d);
      // if(d<0) negEigVal++;
      if(d < -1e-14) {
        negEigVal++;
      } else if(d < 1e-14) {
        nullEigVal++;
        // break;
      } else {
        posEigVal++;
      }
    }
    // printf("(pos,null,neg)=(%d,%d,%d)\n", posEigVal, nullEigVal, negEigVal);
    nlp_->runStats.linsolv.tmInertiaComp.stop();

    if(nullEigVal > 0) return -1;
    return negEigVal;
  }

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  bool solve(hiopVector& x)
  {
    assert(M_->n() == M_->m());
    assert(x.get_size() == M_->n());
    int N = M_->n(), LDA = N, info;
    if(N == 0) return true;

    nlp_->runStats.linsolv.tmTriuSolves.start();

    char uplo = 'L';  // M is upper in C++ so it's lower in fortran
    int NRHS = 1, LDB = N;
    DSYTRS(&uplo, &N, &NRHS, M_->local_data(), &LDA, ipiv, x.local_data(), &LDB, &info);
    if(info < 0) {
      nlp_->log->printf(hovError, "hiopLinSolverSymDenseLapack: DSYTRS returned error %d\n", info);
    } else if(info > 0) {
      nlp_->log->printf(hovError, "hiopLinSolverSymDenseLapack: DSYTRS returned warning %d\n", info);
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    return info == 0;
  }

protected:
  int* ipiv;
  hiopVector* dwork;

private:
  hiopLinSolverSymDenseLapack()
      : ipiv(NULL),
        dwork(NULL)
  {
    assert(false);
  }
};

}  // namespace hiop
#endif
