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
 * @file KktLinSysLowRank.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#include "KktLinSysLowRank.hpp"

namespace hiop
{

KktLinSysLowRank::KktLinSysLowRank(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp)
{
  auto* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(nlp_);

  kxn_mat_ = nlpd->alloc_multivector_primal(nlpd->m());
  assert("DEFAULT" == toupper(nlpd->options->GetString("mem_space")));
  N_ = LinearAlgebraFactory::create_matrix_dense(nlpd->options->GetString("mem_space"), nlpd->m(), nlpd->m());
#ifdef HIOP_DEEPCHECKS
  Nmat_ = N_->alloc_clone();
#endif
  k_vec1_ = nlpd->alloc_dual_vec();
}

KktLinSysLowRank::~KktLinSysLowRank()
{
  delete N_;
#ifdef HIOP_DEEPCHECKS
  delete Nmat_;
#endif
  delete kxn_mat_;
  delete k_vec1_;
}

bool KktLinSysLowRank::update(const hiopIterate* iter,
                              const hiopVector* grad_f,
                              const hiopMatrixDense* Jac_c,
                              const hiopMatrixDense* Jac_d,
                              HessianDiagPlusRowRank* hess_low_rank)
{
  nlp_->runStats.tmSolverInternal.start();

  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVector*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = hess_low_rank;

  // compute the diagonals
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  hess_low_rank->update_logbar_diag(*Dx_);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_inv_->setToZero();
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
  assert(true == Dd_inv_->allPositive());
#endif
  Dd_->copyFrom(*Dd_inv_);
  Dd_inv_->invert();

  nlp_->runStats.tmSolverInternal.stop();

  nlp_->log->write("Dd_inv in KKT", *Dd_inv_, hovMatrices);
  return true;
}

/* Solves the system corresponding to directions for x, yc, and yd, namely
 * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx  ]
 * [    Jc          0     0     ] [dyc] = [ ryc ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd ]
 *
 * This is done by forming and solving
 * [ Jc*(H+Dx)^{-1}*Jc^T   Jc*(H+Dx)^{-1}*Jd^T          ] [dyc] = [ Jc(H+Dx)^{-1} rx - ryc ]
 * [ Jd*(H+Dx)^{-1}*Jc^T   Jd*(H+Dx)^{-1}*Jd^T + Dd^{-1}] [dyd]   [ Jd(H+dx)^{-1} rx - ryd ]
 * and then solving for dx from
 *  dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
 *
 * Note that ops H+Dx are provided by HessianDiagPlusRowRank
 */
bool KktLinSysLowRank::solveCompressed(hiopVector& rx,
                                       hiopVector& ryc,
                                       hiopVector& ryd,
                                       hiopVector& dx,
                                       hiopVector& dyc,
                                       hiopVector& dyd)
{
#ifdef HIOP_DEEPCHECKS
  // some outputing
  nlp_->log->write("KKT Low rank: solve compressed RHS", hovIteration);
  nlp_->log->write("  rx: ", rx, hovIteration);
  nlp_->log->write(" ryc: ", ryc, hovIteration);
  nlp_->log->write(" ryd: ", ryd, hovIteration);
  nlp_->log->write("  Jc: ", *Jac_c_, hovMatrices);
  nlp_->log->write("  Jd: ", *Jac_d_, hovMatrices);
  nlp_->log->write("  Dd_inv: ", *Dd_inv_, hovMatrices);
  assert(Dd_inv_->isfinite_local() && "Something bad happened: nan or inf value");
#endif

  hiopMatrixDense& J = *kxn_mat_;
  const hiopMatrixDense* Jac_c_de = dynamic_cast<const hiopMatrixDense*>(Jac_c_);
  assert(Jac_c_de);
  const hiopMatrixDense* Jac_d_de = dynamic_cast<const hiopMatrixDense*>(Jac_d_);
  assert(Jac_d_de);
  J.copyRowsFrom(*Jac_c_de, nlp_->m_eq(), 0);               //! opt
  J.copyRowsFrom(*Jac_d_de, nlp_->m_ineq(), nlp_->m_eq());  //! opt

  auto* hess_low_rank = dynamic_cast<HessianDiagPlusRowRank*>(Hess_);

  // N =  J*(Hess\J')
  // Hess->symmetricTimesMat(0.0, *N, 1.0, J);
  hess_low_rank->sym_mat_times_inverse_times_mattrans(0.0, *N_, 1.0, J);

  // subdiag of N += 1., Dd_inv
  N_->addSubDiagonal(1., nlp_->m_eq(), *Dd_inv_);
#ifdef HIOP_DEEPCHECKS
  assert(J.isfinite());
  nlp_->log->write("solveCompressed: N is", *N_, hovMatrices);
  nlp_->log->write("solveCompressed: rx is", rx, hovMatrices);
  nlp_->log->printf(hovLinAlgScalars, "inf norm of Dd_inv is %g\n", Dd_inv_->infnorm());
  N_->assertSymmetry(1e-10);
#endif

  // compute the rhs of the lin sys involving N
  //   1. first compute (H+Dx)^{-1} rx_tilde and store it temporarily in dx
  hess_low_rank->solve(rx, dx);
#ifdef HIOP_DEEPCHECKS
  assert(rx.isfinite_local() && "Something bad happened: nan or inf value");
  assert(dx.isfinite_local() && "Something bad happened: nan or inf value");
#endif

  // 2 . then rhs =   [ Jc(H+Dx)^{-1}*rx - ryc ]
  //                  [ Jd(H+dx)^{-1}*rx - ryd ]
  hiopVector& rhs = *k_vec1_;
  rhs.copyFromStarting(0, ryc);
  rhs.copyFromStarting(nlp_->m_eq(), ryd);
  J.timesVec(-1.0, rhs, 1.0, dx);

#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("solveCompressed: dx sol is", dx, hovMatrices);
  nlp_->log->write("solveCompressed: rhs for N is", rhs, hovMatrices);
  Nmat_->copyFrom(*N_);
  hiopVector* r = rhs.new_copy();  // save the rhs to check the norm of the residual
#endif

  //
  // solve N * dyc_dyd = rhs
  //
  int ierr = solveWithRefin(*N_, rhs);
  // int ierr = solve(*N,rhs);

  hiopVector& dyc_dyd = rhs;
  dyc_dyd.copyToStarting(0, dyc);
  dyc_dyd.copyToStarting(nlp_->m_eq(), dyd);

  // now solve for dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
  // first rx = -(Jc^T*dyc+Jd^T*dyd - rx)
  J.transTimesVec(1.0, rx, -1.0, dyc_dyd);
  // then dx = (H+Dx)^{-1} rx
  hess_low_rank->solve(rx, dx);

#ifdef HIOP_DEEPCHECKS
  // some outputing
  nlp_->log->write("KKT Low rank: solve compressed SOL", hovIteration);
  nlp_->log->write("  dx: ", dx, hovIteration);
  nlp_->log->write(" dyc: ", dyc, hovIteration);
  nlp_->log->write(" dyd: ", dyd, hovIteration);
  delete r;
#endif

  return ierr == 0;
}

int KktLinSysLowRank::solveWithRefin(hiopMatrixDense& M, hiopVector& rhs)
{
  // 1. Solve dposvx (solve + equilibrating + iterative refinement + forward and backward error estimates)
  // 2. Check the residual norm
  // 3. If residual norm is not small enough, then perform iterative refinement. This is because dposvx
  // does not always provide a small enough residual since it stops (possibly without refinement) based on
  // the forward and backward estimates

  int N = M.n();
  if(N <= 0) return 0;

  hiopMatrixDense* Aref = M.new_copy();
  hiopVector* rhsref = rhs.new_copy();

  char FACT = 'E';
  char UPLO = 'L';

  int NRHS = 1;
  double* A = M.local_data();
  int LDA = N;
  double* AF = new double[N * N];
  int LDAF = N;
  char EQUED = 'N';  // it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB = N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, FERR, BERR;
  double* WORK = new double[3 * N];
  int* IWORK = new int[N];
  int INFO;

  //
  // 1. solve
  //
  DPOSVX(&FACT, &UPLO, &N, &NRHS, A, &LDA, AF, &LDAF, &EQUED, S, B, &LDB, X, &LDX, &RCOND, &FERR, &BERR, WORK, IWORK, &INFO);
  // printf("INFO ===== %d  RCOND=%g  FERR=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, FERR, BERR, EQUED);
  //
  //  2. check residual
  //
  hiopVector* x = rhs.alloc_clone();
  hiopVector* dx = rhs.alloc_clone();
  hiopVector* resid = rhs.alloc_clone();
  int nIterRefin = 0;
  double nrmResid;
  int info;
  const int MAX_ITER_REFIN = 3;
  while(true) {
    x->copyFrom(X);
    resid->copyFrom(*rhsref);
    Aref->timesVec(1.0, *resid, -1.0, *x);

    nlp_->log->write("resid", *resid, hovLinAlgScalars);

    nrmResid = resid->infnorm();
    nlp_->log->printf(hovScalars, "KktLinSysLowRank::solveWithRefin iterrefin=%d  residual norm=%g\n", nIterRefin, nrmResid);

    if(nrmResid < 1e-8) break;

    if(nIterRefin >= MAX_ITER_REFIN) {
      nlp_->log->write("N", *Aref, hovMatrices);
      nlp_->log->write("sol", *x, hovMatrices);
      nlp_->log->write("rhs", *rhsref, hovMatrices);

      nlp_->log->printf(
          hovWarning,
          "KktLinSysLowRank::solveWithRefin reduced residual to ONLY (inf-norm) %g after %d iterative refinements\n",
          nrmResid,
          nIterRefin);
      break;
      // assert(false && "too many refinements");
    }
    if(0) {  // iter refin based on symmetric indefinite factorization+solve

      int _V_ipiv_vec[1000];
      double _V_work_vec[1000];
      int lwork = 1000;
      M.copyFrom(*Aref);
      DSYTRF(&UPLO, &N, M.local_data(), &LDA, _V_ipiv_vec, _V_work_vec, &lwork, &info);
      assert(info == 0);
      DSYTRS(&UPLO, &N, &NRHS, M.local_data(), &LDA, _V_ipiv_vec, resid->local_data(), &LDB, &info);
      assert(info == 0);
    } else {
      // iter refin based on symmetric positive definite factorization+solve
      M.copyFrom(*Aref);
      DPOTRF(&UPLO, &N, M.local_data(), &LDA, &info);
      if(info > 0) {
        nlp_->log->printf(hovError,
                          "KktLinSysLowRank::factorizeMat: dpotrf (Chol fact) detected %d minor being indefinite.\n",
                          info);
      } else {
        if(info < 0) {
          nlp_->log->printf(hovError, "KktLinSysLowRank::factorizeMat: dpotrf returned error %d\n", info);
        }
      }

      DPOTRS(&UPLO, &N, &NRHS, M.local_data(), &LDA, resid->local_data(), &LDA, &info);
      if(info < 0) {
        nlp_->log->printf(hovError, "KktLinSysLowRank::solveWithFactors: dpotrs returned error %d\n", info);
      }
    }

    dx->copyFrom(*resid);
    x->axpy(1., *dx);

    nIterRefin++;
  }
  rhs.copyFrom(*x);
  delete[] AF;
  delete[] S;
  delete[] X;
  delete[] WORK;
  delete[] IWORK;
  delete Aref;
  delete rhsref;
  delete x;
  delete dx;
  delete resid;

  return 0;
}

int KktLinSysLowRank::solve(hiopMatrixDense& M, hiopVector& rhs)
{
  char FACT = 'E';
  char UPLO = 'L';
  int N = M.n();
  int NRHS = 1;
  double* A = M.local_data();
  int LDA = N;
  double* AF = new double[N * N];
  int LDAF = N;
  char EQUED = 'N';  // it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB = N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, FERR, BERR;
  double* WORK = new double[3 * N];
  int* IWORK = new int[N];
  int INFO;

  DPOSVX(&FACT, &UPLO, &N, &NRHS, A, &LDA, AF, &LDAF, &EQUED, S, B, &LDB, X, &LDX, &RCOND, &FERR, &BERR, WORK, IWORK, &INFO);

  rhs.copyFrom(S);
  nlp_->log->write("Scaling S", rhs, hovSummary);

  rhs.copyFrom(X);
  delete[] AF;
  delete[] S;
  delete[] X;
  delete[] WORK;
  delete[] IWORK;
  return 0;
}

/* this code works fine but requires xblas
int KktLinSysLowRank::solveWithRefin(hiopMatrixDense& M, hiopVectorPar& rhs)
{
  char FACT='E';
  char UPLO='L';
  int N=M.n();
  int NRHS=1;
  double* A=M.local_buffer();
  int LDA=N;
  double* AF=new double[N*N];
  int LDAF=N;
  char EQUED='N'; //it is an output if FACT='E'
  double* S = new double[N];
  double* B = rhs.local_data();
  int LDB=N;
  double* X = new double[N];
  int LDX = N;
  double RCOND, BERR;
  double RPVGRW; //Reciprocal pivot growth
  int N_ERR_BNDS=3;
  double* ERR_BNDS_NORM = new double[NRHS*N_ERR_BNDS];
  double* ERR_BNDS_COMP = new double[NRHS*N_ERR_BNDS];
  int NPARAMS=3;
  double PARAMS[NPARAMS];
  PARAMS[0]=1.0;  //Use the extra-precise refinement algorithm
  PARAMS[1]=3.0; //Maximum number of residual computations allowed for refinement
  PARAMS[2]=1.0; //attempt to find a solution with small componentwise
  double* WORK = new double[4*N];
  int* IWORK = new int[N];
  int INFO;

  dposvxx_(&FACT, &UPLO, &N, &NRHS,
           A, &LDA,
           AF, &LDAF,
           &EQUED,
           S,
           B, &LDB,
           X, &LDX,
           &RCOND, &RPVGRW, &BERR,
           &N_ERR_BNDS, ERR_BNDS_NORM, ERR_BNDS_COMP,
           &NPARAMS, PARAMS,
           WORK, IWORK,
           &INFO);

  //rhs.copyFrom(S);
  //nlp_->log->write("Scaling S", rhs, hovSummary);

  //M.copyFrom(AF);
  //nlp_->log->write("Factoriz ", M, hovSummary);

  printf("INFO ===== %d  RCOND=%g  RPVGRW=%g   BERR=%g  EQUED=%c\n", INFO, RCOND, RPVGRW, BERR, EQUED);
  printf("               ERR_BNDS_NORM=%g %g %g    ERR_BNDS_COMP=%g %g %g \n", ERR_BNDS_NORM[0], ERR_BNDS_NORM[1],
ERR_BNDS_NORM[2], ERR_BNDS_COMP[0], ERR_BNDS_COMP[1], ERR_BNDS_COMP[2]); printf("               PARAMS=%g %g %g \n",
PARAMS[0], PARAMS[1], PARAMS[2]);


  rhs.copyFrom(X);
  delete [] AF;
  delete [] S;
  delete [] X;
  delete [] ERR_BNDS_NORM;
  delete [] ERR_BNDS_COMP;
  delete [] WORK;
  delete [] IWORK;
  return 0;
}
*/

#ifdef HIOP_DEEPCHECKS

double KktLinSysLowRank::errorCompressedLinsys(const hiopVector& rx,
                                               const hiopVector& ryc,
                                               const hiopVector& ryd,
                                               const hiopVector& dx,
                                               const hiopVector& dyc,
                                               const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "KktLinSysLowRank::errorCompressedLinsys residuals norm:\n");
  auto* hess_low_rank = dynamic_cast<HessianDiagPlusRowRank*>(Hess_);

  double derr = -1.;
  double aux;
  hiopVector* RX = rx.new_copy();
  // RX=rx-H*dx-J'c*dyc-J'*dyd
  hess_low_rank->timesVec(1.0, *RX, -1.0, dx);
  // RX->axzpy(-1.0,*Dx,dx);
  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux = RX->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>>  rx=%g\n", aux);
  delete RX;

  hiopVector* RC = ryc.new_copy();
  Jac_c_->timesVec(1.0, *RC, -1.0, dx);
  aux = RC->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>> ryc=%g\n", aux);
  delete RC;

  hiopVector* RD = ryd.new_copy();
  Jac_d_->timesVec(1.0, *RD, -1.0, dx);
  RD->axzpy(1.0, *Dd_inv_, dyd);
  aux = RD->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, "  >>> ryd=%g\n", aux);
  delete RD;

  return derr;
}

double KktLinSysLowRank::solveError(const hiopMatrixDense& M, const hiopVector& x, hiopVector& rhs)
{
  double relError;
  M.timesVec(1.0, rhs, -1.0, x);
  double resnorm = rhs.infnorm();

  relError = resnorm;  // / (1+rhsnorm);
  return relError;
}
#endif

}  // namespace hiop
