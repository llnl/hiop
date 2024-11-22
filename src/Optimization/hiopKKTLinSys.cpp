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

#include "hiopKKTLinSys.hpp"
#include "LinAlgFactory.hpp"
#include "hiop_blasdefs.hpp"
#include "hiopPDPerturbation.hpp"

#include <cmath>

namespace hiop
{

hiopKKTLinSys::hiopKKTLinSys(hiopNlpFormulation* nlp)
    : nlp_(nlp),
      iter_(NULL),
      grad_f_(NULL),
      Jac_c_(NULL),
      Jac_d_(NULL),
      Hess_(NULL),
      perturb_calc_(NULL),
      safe_mode_(true),
      kkt_opr_(nullptr),
      prec_opr_(nullptr),
      bicgIR_(nullptr),
      delta_wx_(nullptr),
      delta_wd_(nullptr),
      delta_cc_(nullptr),
      delta_cd_(nullptr)

{
  perf_report_ = "on" == hiop::tolower(nlp_->options->GetString("time_kkt"));
  mu_ = nlp_->options->GetNumeric("mu0");
}

hiopKKTLinSys::~hiopKKTLinSys()
{
  delete kkt_opr_;
  delete prec_opr_;
  delete bicgIR_;
}

// computes the solve error for the KKT Linear system; used only for correctness checking
double hiopKKTLinSys::errorKKT(const hiopResidual* resid, const hiopIterate* sol)
{
  nlp_->log->printf(hovLinAlgScalars, "KKT LinSys::errorKKT KKT_large residuals norm:\n");
  assert(perturb_calc_ && "perturb_calc_ is not assigned");

  if(perturb_calc_) {
    delta_wx_ = perturb_calc_->get_curr_delta_wx();
    delta_wd_ = perturb_calc_->get_curr_delta_wd();
    delta_cc_ = perturb_calc_->get_curr_delta_cc();
    delta_cd_ = perturb_calc_->get_curr_delta_cd();
  } else {
  }

  double derr = 1e20, aux;
  hiopVector* RX = resid->rx->new_copy();

  // RX = rx-H*dx-J'c*dyc-J'*dyd +dzl-dzu
  HessianTimesVec_noLogBarrierTerm(1.0, *RX, -1.0, *sol->x);
  RX->axzpy(-1., *delta_wx_, *sol->x);

  Jac_c_->transTimesVec(1.0, *RX, -1.0, *sol->yc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, *sol->yd);
  RX->axpy(1.0, *sol->zl);
  RX->axpy(-1.0, *sol->zu);
  aux = RX->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rx=%g\n", aux);

  // RD = rd - (-dyd - dvl + dvu + delta_wd_*dd)
  hiopVector* RD = resid->rd->new_copy();
  RD->axpy(+1., *sol->yd);
  RD->axpy(+1., *sol->vl);
  RD->axpy(-1., *sol->vu);
  RD->axzpy(-1., *delta_wd_, *sol->d);
  aux = RD->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rd=%g\n", aux);

  // RYC = ryc - Jc*dx + delta_cc_*dyc
  hiopVector* RYC = resid->ryc->new_copy();
  Jac_c_->timesVec(1.0, *RYC, -1.0, *sol->x);
  RYC->axzpy(1., *delta_cc_, *sol->yc);
  aux = RYC->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- ryc=%g\n", aux);
  delete RYC;

  // RYD=ryd - Jd*dx + dd + delta_cd_*dyd
  hiopVector* RYD = resid->ryd->new_copy();
  Jac_d_->timesVec(1.0, *RYD, -1.0, *sol->x);
  RYD->axpy(1.0, *sol->d);
  RYD->axzpy(1., *delta_cd_, *sol->yd);
  aux = RYD->infnorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- ryd=%g\n", aux);
  delete RYD;

  // RXL=rxl+x-sxl
  RX->copyFrom(*resid->rxl);
  RX->axpy(1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxl);
  RX->selectPattern(nlp_->get_ixl());
  aux = RX->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rxl=%g\n", aux);
  // RXU=rxu-x-sxu
  RX->copyFrom(*resid->rxu);
  RX->axpy(-1.0, *sol->x);
  RX->axpy(-1.0, *sol->sxu);
  RX->selectPattern(nlp_->get_ixu());
  aux = RX->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rxu=%g\n", aux);

  // RDL=rdl+d-sdl
  RD->copyFrom(*resid->rdl);
  RD->axpy(1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdl);
  RD->selectPattern(nlp_->get_idl());
  aux = RD->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rdl=%g\n", aux);

  // RDU=rdu-d-sdu
  RD->copyFrom(*resid->rdu);
  RD->axpy(-1.0, *sol->d);
  RD->axpy(-1.0, *sol->sdu);
  RD->selectPattern(nlp_->get_idu());
  aux = RD->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rdu=%g\n", aux);

  // complementarity residuals checks: rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszl);
  RX->axzpy(-1.0, *iter_->sxl, *sol->zl);
  RX->axzpy(-1.0, *iter_->zl, *sol->sxl);
  aux = RX->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rszl=%g\n", aux);
  // rszl - Sxl dzxl - Zxl dsxl
  RX->copyFrom(*resid->rszu);
  RX->axzpy(-1.0, *iter_->sxu, *sol->zu);
  RX->axzpy(-1.0, *iter_->zu, *sol->sxu);
  aux = RX->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rszu=%g\n", aux);
  delete RX;
  RX = NULL;

  // complementarity residuals checks: rsvl - Sdl dvl - Vl dsdl
  RD->copyFrom(*resid->rsvl);
  RD->axzpy(-1.0, *iter_->sdl, *sol->vl);
  RD->axzpy(-1.0, *iter_->vl, *sol->sdl);
  aux = RD->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rsvl=%g\n", aux);
  // complementarity residuals checks: rsvu - Sdu dvu - Vu dsdu
  RD->copyFrom(*resid->rsvu);
  RD->axzpy(-1.0, *iter_->sdu, *sol->vu);
  RD->axzpy(-1.0, *iter_->vu, *sol->sdu);
  aux = RD->twonorm();
  derr = fmax(aux, derr);
  nlp_->log->printf(hovLinAlgScalars, "  --- rsvu=%g\n", aux);
  delete RD;

  return derr;
}

bool hiopKKTLinSys::compute_directions_for_full_space(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.kkt.tmSolveRhsManip.start();
  const hiopResidual& r = *resid;

  /***********************************************************************
   * compute the rest of the directions
   *
   */
  // dsxl = rxl + dx  and dzl= [Sxl]^{-1} ( - Zl*dsxl + rszl)
  if(nlp_->n_low_local()) {
    dir->sxl->copyFrom(*r.rxl);
    dir->sxl->axpy(1.0, *dir->x);
    dir->sxl->selectPattern(nlp_->get_ixl());

    dir->zl->copyFrom(*r.rszl);
    dir->zl->axzpy(-1.0, *iter_->zl, *dir->sxl);
    dir->zl->componentDiv_w_selectPattern(*iter_->sxl, nlp_->get_ixl());
  } else {
    dir->sxl->setToZero();
    dir->zl->setToZero();
  }

  // dir->sxl->print();
  // dir->zl->print();
  // dsxu = rxu - dx and dzu = [Sxu]^{-1} ( - Zu*dsxu + rszu)
  if(nlp_->n_upp_local()) {
    dir->sxu->copyFrom(*r.rxu);
    dir->sxu->axpy(-1.0, *dir->x);
    dir->sxu->selectPattern(nlp_->get_ixu());

    dir->zu->copyFrom(*r.rszu);
    dir->zu->axzpy(-1.0, *iter_->zu, *dir->sxu);
    dir->zu->selectPattern(nlp_->get_ixu());
    dir->zu->componentDiv_w_selectPattern(*iter_->sxu, nlp_->get_ixu());
  } else {
    dir->sxu->setToZero();
    dir->zu->setToZero();
  }

  // dir->sxu->print();
  // dir->zu->print();
  // dsdl = rdl + dd and dvl = [Sdl]^{-1} ( - Vl*dsdl + rsvl)
  if(nlp_->m_ineq_low()) {
    dir->sdl->copyFrom(*r.rdl);
    dir->sdl->axpy(1.0, *dir->d);
    dir->sdl->selectPattern(nlp_->get_idl());

    dir->vl->copyFrom(*r.rsvl);
    dir->vl->axzpy(-1.0, *iter_->vl, *dir->sdl);
    dir->vl->selectPattern(nlp_->get_idl());
    dir->vl->componentDiv_w_selectPattern(*iter_->sdl, nlp_->get_idl());
  } else {
    dir->sdl->setToZero();
    dir->vl->setToZero();
  }

  // dsdu = rdu - dd and dvu = [Sdu]^{-1} ( - Vu*dsdu + rsvu )
  if(nlp_->m_ineq_upp() > 0) {
    dir->sdu->copyFrom(*r.rdu);
    dir->sdu->axpy(-1.0, *dir->d);
    dir->sdu->selectPattern(nlp_->get_idu());

    dir->vu->copyFrom(*r.rsvu);
    dir->vu->axzpy(-1.0, *iter_->vu, *dir->sdu);
    dir->vu->selectPattern(nlp_->get_idu());
    dir->vu->componentDiv_w_selectPattern(*iter_->sdu, nlp_->get_idu());
  } else {
    dir->sdu->setToZero();
    dir->vu->setToZero();
  }
  nlp_->runStats.kkt.tmSolveRhsManip.stop();

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  assert(dir->sxl->matchesPattern(nlp_->get_ixl()));
  assert(dir->sxu->matchesPattern(nlp_->get_ixu()));
  assert(dir->sdl->matchesPattern(nlp_->get_idl()));
  assert(dir->sdu->matchesPattern(nlp_->get_idu()));
  assert(dir->zl->matchesPattern(nlp_->get_ixl()));
  assert(dir->zu->matchesPattern(nlp_->get_ixu()));
  assert(dir->vl->matchesPattern(nlp_->get_idl()));
  assert(dir->vu->matchesPattern(nlp_->get_idu()));

  // CHECK THE SOLUTION
  errorKKT(resid, dir);
  nlp_->runStats.kkt.tmResid.stop();
#endif

  return true;
}

int hiopKKTLinSysCurvCheck::factorizeWithCurvCheck() { return linSys_->matrixChanged(); }

bool hiopKKTLinSysCurvCheck::factorize()
{
  assert(nlp_);

  // factorization + inertia correction if needed
  const size_t max_refactorization = 10;
  size_t num_refactorization = 0;
  int continue_re_fact;

  if(!perturb_calc_->compute_initial_deltas()) {
    nlp_->log->printf(hovWarning, "linsys: Regularization perturbation on new linsys failed.\n");
    return false;
  }

  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  while(num_refactorization <= max_refactorization) {
#ifdef HIOP_DEEPCHECKS
    assert(perturb_calc_->check_consistency() && "something went wrong with IC");
#endif
    if(hovScalars <= nlp_->options->GetInteger("verbosity_level")) {
      nlp_->log->printf(hovScalars,
                        "linsys: norminf(delta_w)=%12.5e norminf(delta_c)=%12.5e (ic %d)\n",
                        delta_wx_->infnorm(),
                        delta_cc_->infnorm(),
                        num_refactorization);
    }

    // the update of the linear system, including IC perturbations
    this->build_kkt_matrix(*perturb_calc_);

    nlp_->runStats.kkt.tmUpdateInnerFact.start();

    // factorization
    int n_neg_eig = factorizeWithCurvCheck();

    nlp_->runStats.kkt.tmUpdateInnerFact.stop();

    continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, n_neg_eig);

    if(-1 == continue_re_fact) {
      return false;
    } else if(0 == continue_re_fact) {
      break;
    }

    // will do an inertia correction
    num_refactorization++;
    nlp_->runStats.kkt.nUpdateICCorr++;
  }  // end of IC loop

  if(num_refactorization > max_refactorization) {
    nlp_->log->printf(hovError,
                      "Reached max number (%d) of refactorization within an outer iteration.\n",
                      max_refactorization);
    return false;
  }
  return true;
}

bool hiopKKTLinSysCurvCheck::factorize_inertia_free()
{
  assert(nlp_);

  int non_singular_mat = 1;
  int continue_re_fact;

  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, non_singular_mat, true);

#ifdef HIOP_DEEPCHECKS
  assert(perturb_calc_->check_consistency() && "something went wrong with IC");
#endif
  if(hovScalars <= nlp_->options->GetInteger("verbosity_level")) {
    nlp_->log->printf(hovScalars,
                      "linsys: norminf(delta_w)=%12.5e norminf(delta_c)=%12.5e \n",
                      delta_wx_->infnorm(),
                      delta_cc_->infnorm());
  }

  // the update of the linear system, including IC perturbations
  this->build_kkt_matrix(*perturb_calc_);

  nlp_->runStats.kkt.tmUpdateInnerFact.start();

  // factorization
  int solver_flag = factorizeWithCurvCheck();

  // if solver_flag<0, matrix becomes singular, or not pd (in condensed system) after adding regularization
  // this should not happen, but some linear solver may have numerical difficulty.
  // adding more regularization till it succeeds
  const size_t max_refactorization = 10;
  size_t num_refactorization = 0;

  while(num_refactorization <= max_refactorization && solver_flag < 0) {
    nlp_->log->printf(hovWarning, "linsys: matrix becomes singular after adding primal regularization!\n");

    continue_re_fact = fact_acceptor_->requireReFactorization(*nlp_, solver_flag);

    if(-1 == continue_re_fact) {
      return false;
    } else {
      // this while loop is used to correct singularity
      assert(1 == continue_re_fact);
    }

    if(hovScalars <= nlp_->options->GetInteger("verbosity_level")) {
      nlp_->log->printf(hovScalars,
                        "linsys: norminf(delta_w)=%12.5e norminf(delta_c)=%12.5e \n",
                        delta_wx_->infnorm(),
                        delta_cc_->infnorm());
    }

    // the update of the linear system, including IC perturbations
    this->build_kkt_matrix(*perturb_calc_);

    nlp_->runStats.kkt.tmUpdateInnerFact.start();

    // factorization
    solver_flag = factorizeWithCurvCheck();

    nlp_->runStats.kkt.tmUpdateInnerFact.stop();

    // will do an inertia correction
    num_refactorization++;
    nlp_->runStats.kkt.nUpdateICCorr++;
  }  // end of IC loop

  nlp_->runStats.kkt.tmUpdateInnerFact.stop();

  return true;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressed
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
bool hiopKKTLinSysCompressed::test_direction(const hiopIterate* dir, hiopMatrix* Hess)
{
  bool retval;
  nlp_->runStats.tmSolverInternal.start();

  if(!x_wrk_) {
    x_wrk_ = nlp_->alloc_primal_vec();
    x_wrk_->setToZero();
  }
  if(!d_wrk_) {
    d_wrk_ = nlp_->alloc_dual_ineq_vec();
    d_wrk_->setToZero();
  }

  hiopVector* sol_x = dir->get_x();
  hiopVector* sol_d = dir->get_d();
  double dWd = 0;
  double xs_nrmsq = 0.0;
  double dbl_wrk;
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  /* compute xWx = x(H+Dx_)x (for primal var [x,d] */
  Hess_->timesVec(0.0, *x_wrk_, 1.0, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  x_wrk_->copyFrom(*sol_x);
  x_wrk_->componentMult(*Dx_);
  x_wrk_->axzpy(1., *delta_wx_, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  d_wrk_->copyFrom(*sol_d);
  d_wrk_->componentMult(*Dd_);
  d_wrk_->axzpy(1., *delta_wd_, *sol_d);
  dWd += d_wrk_->dotProductWith(*sol_d);

  /* compute rhs for the dWd test */
  dbl_wrk = sol_x->twonorm();
  xs_nrmsq += dbl_wrk * dbl_wrk;
  dbl_wrk = sol_d->twonorm();
  xs_nrmsq += dbl_wrk * dbl_wrk;

  if(dWd < xs_nrmsq * nlp_->options->GetNumeric("neg_curv_test_fact")) {
    // have negative curvature. Add regularization and re-factorize the matrix
    retval = false;
  } else {
    // have positive curvature. Accept this factoraizaiton and direction.
    retval = true;
  }

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressedXYcYd
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/**
 * Provides the functionality for reducing the KKT linear system to the
 * compressed linear below in dx, dd, dyc, and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions.
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * Relies on the pure virtual 'solveCompressed' to solve the compressed linear system
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 */
hiopKKTLinSysCompressedXYcYd::hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressed(nlp)
{
  Dd_inv_ = dynamic_cast<hiopVector*>(nlp_->alloc_dual_ineq_vec());
  assert(Dd_inv_ != NULL);

  ryd_tilde_ = Dd_inv_->alloc_clone();
}

hiopKKTLinSysCompressedXYcYd::~hiopKKTLinSysCompressedXYcYd()
{
  delete Dd_inv_;
  delete ryd_tilde_;
}

bool hiopKKTLinSysCompressedXYcYd::update(const hiopIterate* iter,
                                          const hiopVector* grad_f,
                                          const hiopMatrix* Jac_c,
                                          const hiopMatrix* Jac_d,
                                          hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.reset();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();

  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = Hess;

  int nx = Hess_->m();
  assert(nx == Hess_->n());
  assert(nx == Jac_c_->n());
  assert(nx == Jac_d_->n());

  // compute and put the barrier diagonals in
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  assert(true == Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();

  // factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysCompressedXYcYd::computeDirections(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual& r = *resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde  = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * ryd_tilde = ryd + [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}*
   *                     [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)]
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
   * Dd_inv = [(Sdl^{-1}Vl+Sdu^{-1}Vu)]^{-1}
   * yd_tilde = ryd + Dd_inv*rd_tilde
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local() > 0) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector& rl = *(dir->x);  // temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    // rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern(1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local() > 0) {
    // ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector& ru = *(dir->x);  // temporary working buffer
    ru.copyFrom(*r.rszu);
    ru.axzpy(-1.0, *iter_->zu, *r.rxu);
    // rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  // for ryd_tilde:
  ryd_tilde_->copyFrom(*r.ryd);
  // 1. the diag (Sdl^{-1}Vl+Sdu^{-1}Vu)^{-1} has already computed in Dd_inv in 'update'
  // 2. compute the left multiplicand in ryd2 (using buffer dir->sdl), that is
  //   ryd2 = [rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)] (this is \tilde{r}_d in the notes)
  //    Inner ops are performed by accumulating in rd2  (buffer dir->sdu)
  hiopVector& ryd2 = *dir->sdl;
  ryd2.copyFrom(*r.rd);

  if(nlp_->m_ineq_low() > 0) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    // ryd2 +=  Sdl^{-1}*(rsvl-Vl*rdl)
    ryd2.axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp() > 0) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    // ryd2 += -Sdu^{-1}(rsvu-Vu*rdu)
    ryd2.axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }

  nlp_->log->write("Dinv (in computeDirections)", *Dd_inv_, hovMatrices);

  // now the final ryd_tilde += Dd^{-1}*ryd2
  ryd_tilde_->axzpy(1.0, ryd2, *Dd_inv_);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  hiopVector* rx_tilde_save = rx_tilde_->new_copy();
  hiopVector* ryc_save = r.ryc->new_copy();
  hiopVector* ryd_tilde_save = ryd_tilde_->new_copy();
  nlp_->runStats.kkt.tmResid.stop();
#endif

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*rx_tilde_, *r.ryc, *ryd_tilde_, *dir->x, *dir->yc, *dir->yd);

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  // recover dir->d = (D)^{-1}*(dir->yd + ryd2)
  dir->d->copyFrom(ryd2);
  dir->d->axpy(1.0, *dir->yd);
  dir->d->componentMult(*Dd_inv_);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  // dir->d->print();

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  errorCompressedLinsys(*rx_tilde_save, *ryc_save, *ryd_tilde_save, *dir->x, *dir->yc, *dir->yd);
  delete rx_tilde_save;
  delete ryc_save;
  delete ryd_tilde_save;
  nlp_->runStats.kkt.tmResid.stop();
#endif

  if(false == sol_ok) {
    return false;
  }

  const bool bret = compute_directions_for_full_space(resid, dir);

  nlp_->runStats.tmSolverInternal.stop();
  return bret;
}

#ifdef HIOP_DEEPCHECKS
// this method needs a bit of revisiting if becomes critical (mainly avoid dynamic allocations)
double hiopKKTLinSysCompressedXYcYd::errorCompressedLinsys(const hiopVector& rx,
                                                           const hiopVector& ryc,
                                                           const hiopVector& ryd,
                                                           const hiopVector& dx,
                                                           const hiopVector& dyc,
                                                           const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXYcYd::errorCompressedLinsys residuals norm:\n");
  assert(perturb_calc_);
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  double derr = 1e20, aux;
  hiopVector* RX = rx.new_copy();
  // RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess_->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx_, dx);
  RX->axzpy(-1., *delta_wx_, dx);

  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux = RX->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX;
  RX = NULL;

  hiopVector* RC = ryc.new_copy();
  Jac_c_->timesVec(1.0, *RC, -1.0, dx);
  RC->axzpy(1., *delta_cc_, dyc);
  aux = RC->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC;
  RC = NULL;

  hiopVector* RD = ryd.new_copy();
  Jac_d_->timesVec(1.0, *RD, -1.0, dx);

  RD->axzpy(1.0, *Dd_inv_, dyd);
  RD->axzpy(1., *delta_cd_, dyd);
  aux = RD->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RD;

  return derr;
}
#endif

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysCompressedXDYcYd
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/* Provides the functionality for reducing the KKT linear system system to the
 * compressed linear below in dx, dd, dyc, and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions.
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
 * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
 * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]
 * and then to compute the rest of the search directions
 */
hiopKKTLinSysCompressedXDYcYd::hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressed(nlp)
{
  //  Dd_ = dynamic_cast<hiopVector*>(nlp_->alloc_dual_ineq_vec());
  //  assert(Dd_ != NULL);

  rd_tilde_ = Dd_->alloc_clone();
}

hiopKKTLinSysCompressedXDYcYd::~hiopKKTLinSysCompressedXDYcYd()
{
  //  delete Dd_;
  delete rd_tilde_;
}

bool hiopKKTLinSysCompressedXDYcYd::update(const hiopIterate* iter,
                                           const hiopVector* grad_f,
                                           const hiopMatrix* Jac_c,
                                           const hiopMatrix* Jac_d,
                                           hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.reset();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();

  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;

  Hess_ = Hess;

  int nx = Hess_->m();
  assert(nx == Hess_->n());
  assert(nx == Jac_c_->n());
  assert(nx == Jac_d_->n());

  // compute barrier diagonals (these change only between outer optimiz iterations)
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  assert(true == Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();

  // factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysCompressedXDYcYd::computeDirections(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual& r = *resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde = rx+Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector& rl = *(dir->x);  // temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    // rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern(1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local()) {
    // ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector& ru = *(dir->x);  // temporary working buffer
    ru.copyFrom(*r.rszu);
    ru.axzpy(-1.0, *iter_->zu, *r.rxu);
    // rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  // for rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
  rd_tilde_->copyFrom(*r.rd);
  if(nlp_->m_ineq_low()) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    // rd_tilde +=  Sdl^{-1}*(rsvl-Vl*rdl)
    rd_tilde_->axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp() > 0) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    // rd_tilde += -Sdu^{-1}(rsvu-Vu*rdu)
    rd_tilde_->axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }
  nlp_->log->write("Dd (in computeDirections)", *Dd_, hovMatrices);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  hiopVector* rx_tilde_save = rx_tilde_->new_copy();
  hiopVector* rd_tilde_save = rd_tilde_->new_copy();
  hiopVector* ryc_save = r.ryc->new_copy();
  hiopVector* ryd_save = r.ryd->new_copy();
  nlp_->runStats.kkt.tmResid.stop();
#endif

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*rx_tilde_, *rd_tilde_, *r.ryc, *r.ryd, *dir->x, *dir->d, *dir->yc, *dir->yd);

#ifdef HIOP_DEEPCHECKS
  nlp_->runStats.kkt.tmResid.start();
  double derr =
      errorCompressedLinsys(*rx_tilde_save, *rd_tilde_save, *ryc_save, *ryd_save, *dir->x, *dir->d, *dir->yc, *dir->yd);
  if(derr > 1e-8) nlp_->log->printf(hovWarning, "solve compressed high absolute resid norm (=%12.5e)\n", derr);
  delete rx_tilde_save;
  delete ryc_save;
  delete rd_tilde_save;
  delete ryd_save;
  nlp_->runStats.kkt.tmResid.stop();
#endif

  if(false == sol_ok) {
    nlp_->runStats.tmSolverInternal.stop();
    return false;
  }

  const bool bret = compute_directions_for_full_space(resid, dir);

  nlp_->runStats.tmSolverInternal.stop();
  return bret;
}

bool hiopKKTLinSys::compute_directions_w_IR(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();

  // skip IR if user set ir_outer_maxit to 0 or negative values
  if(0 >= nlp_->options->GetInteger("ir_outer_maxit")) {
    nlp_->runStats.tmSolverInternal.stop();
    return computeDirections(resid, dir);
  }
  const hiopResidual& r = *resid;

  // in the order of rx, rd, ryc, ryd, rxl, rxu, rdl, rdu, rszl, rszu, rsvl, rsvu
  const size_type nx = r.rx->get_local_size();
  const size_type nd = r.rd->get_local_size();
  const size_type nyc = r.ryc->get_local_size();
  const size_type nyd = r.ryd->get_local_size();
  size_type dim_rhs = 5 * nx + 5 * nd + nyc + nyd;
  /***********************************************************************
   * solve the compressed system as a preconditioner
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/

  if(nullptr == kkt_opr_) {
    kkt_opr_ = new hiopMatVecKKTFullOpr(this, iter_);
    prec_opr_ = new hiopPrecondKKTOpr(this, iter_);
    bicgIR_ = new hiopBiCGStabSolver(dim_rhs, kkt_opr_, prec_opr_);
  }

  // need to reset the pointer to the current iter, since the outer loop keeps swtiching between curr_iter and trial_iter
  kkt_opr_->reset_curr_iter(iter_);

  double tol =
      std::min(mu_ * nlp_->options->GetNumeric("ir_outer_tol_factor"), nlp_->options->GetNumeric("ir_outer_tol_min"));
  bicgIR_->set_max_num_iter(nlp_->options->GetInteger("ir_outer_maxit"));
  bicgIR_->set_tol(tol);
  bicgIR_->set_x0(0.0);

  bool bret = bicgIR_->solve(dir, resid);

  nlp_->runStats.kkt.nIterRefinInner += bicgIR_->get_sol_num_iter();
  if(!bret) {
    nlp_->log->printf(hovWarning, "%s", bicgIR_->get_convergence_info().c_str());

    // accept the stpe since this is IR
    bret = true;
  } else {
    nlp_->log->printf(hovScalars, "%s", bicgIR_->get_convergence_info().c_str());
  }

  nlp_->runStats.tmSolverInternal.stop();
  return bret;
}

#ifdef HIOP_DEEPCHECKS
double hiopKKTLinSysCompressedXDYcYd::errorCompressedLinsys(const hiopVector& rx,
                                                            const hiopVector& rd,
                                                            const hiopVector& ryc,
                                                            const hiopVector& ryd,
                                                            const hiopVector& dx,
                                                            const hiopVector& dd,
                                                            const hiopVector& dyc,
                                                            const hiopVector& dyd)
{
  nlp_->log->printf(hovLinAlgScalars, "hiopKKTLinSysDenseXDYcYd::errorCompressedLinsys residuals norm:\n");
  assert(perturb_calc_);
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  double derr = -1., aux;
  hiopVector* RX = rx.new_copy();
  // RX=rx-H*dx-J'c*dyc-J'*dyd
  Hess_->timesVec(1.0, *RX, -1.0, dx);
  RX->axzpy(-1.0, *Dx_, dx);
  RX->axzpy(-1, *delta_wx_, dx);
  Jac_c_->transTimesVec(1.0, *RX, -1.0, dyc);
  Jac_d_->transTimesVec(1.0, *RX, -1.0, dyd);
  aux = RX->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rx=%g\n", aux);
  delete RX;

  // RD = rd + dyd - Dd*dd
  hiopVector* RD = rd.new_copy();
  RD->axpy(1., dyd);
  RD->axzpy(-1., *Dd_, dd);
  RD->axzpy(-1., *delta_wd_, dd);
  aux = RD->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >>  rd=%g\n", aux);
  delete RD;

  hiopVector* RC = ryc.new_copy();
  Jac_c_->timesVec(1.0, *RC, -1.0, dx);
  RC->axzpy(1., *delta_cc_, dyc);
  aux = RC->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryc=%g\n", aux);
  delete RC;

  // RYD = ryd+dyd - Jd*dx
  hiopVector* RYD = ryd.new_copy();
  Jac_d_->timesVec(1.0, *RYD, -1.0, dx);
  RYD->axpy(1.0, dd);
  RYD->axzpy(1., *delta_cd_, dyd);
  aux = RYD->twonorm();
  derr = fmax(derr, aux);
  nlp_->log->printf(hovLinAlgScalars, " >> ryd=%g\n", aux);
  delete RYD;
  RYD = NULL;

  return derr;
}
#endif

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopKKTLinSysFull
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
bool hiopKKTLinSysFull::update(const hiopIterate* iter,
                               const hiopVector* grad_f,
                               const hiopMatrix* Jac_c,
                               const hiopMatrix* Jac_d,
                               hiopMatrix* Hess)
{
  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = Hess;
  nlp_->runStats.linsolv.reset();
  nlp_->runStats.tmSolverInternal.start();

  // factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysFull::computeDirections(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();

  const hiopResidual& r = *resid;

  /***********************************************************************
   * solve the full system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solve(*r.rx,
                      *r.ryc,
                      *r.ryd,
                      *r.rd,
                      *r.rdl,
                      *r.rdu,
                      *r.rxl,
                      *r.rxu,
                      *r.rsvl,
                      *r.rsvu,
                      *r.rszl,
                      *r.rszu,
                      *dir->x,
                      *dir->yc,
                      *dir->yd,
                      *dir->d,
                      *dir->vl,
                      *dir->vu,
                      *dir->zl,
                      *dir->zu,
                      *dir->sdl,
                      *dir->sdu,
                      *dir->sxl,
                      *dir->sxu);

  nlp_->runStats.tmSolverInternal.stop();
  return sol_ok;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopMatVecKKTFullOpr
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
hiopMatVecKKTFullOpr::hiopMatVecKKTFullOpr(hiopKKTLinSys* kkt, const hiopIterate* iter)
    : kkt_(kkt),
      iter_(iter),
      resid_(nullptr),
      dir_(nullptr)
{
  resid_ = new hiopResidual(kkt_->nlp_);
  dir_ = new hiopIterate(kkt_->nlp_);

  // set compound vector pointed to dir_/resid_
  dir_cv_ = new hiopVectorCompoundPD(dir_);
  res_cv_ = new hiopVectorCompoundPD(resid_);
}

bool hiopMatVecKKTFullOpr::split_vec_to_build_it(const hiopVector& x) { return true; }

bool hiopMatVecKKTFullOpr::combine_res_to_build_vec(hiopVector& y) { return true; }

/**
 * Full KKT matrix is
 * [   H    0   Jc^T  Jd^T |  -I  I   0   0   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  0     0     0    -I  |  0   0  -I   I   |  0   0   0   0  ] [  dd]   [    rd    ]
 * [  Jc    0     0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    -I    0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [ -I     0     0     0  |  0   0   0   0   |  I   0   0   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0     0  |  0   0   0   0   |  0   I   0   0  ] [ dzu]   [   rxu    ]
 * [  0     -I    0     0  |  0   0   0   0   |  0   0   I   0  ] [ dvl]   [   rdl    ]
 * [  0     I     0     0  |  0   0   0   0   |  0   0   0   I  ] [ dvu]   [   rdu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0     0  | Sl^x 0   0   0   | Zl   0   0   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0     0  |  0  Su^x 0   0   |  0  Zu   0   0  ] [dsxu]   [  rszu    ]
 * [  0     0     0     0  |  0   0  Sl^d 0   |  0   0  Vl   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0     0  |  0   0   0  Su^d |  0   0   0  Vu  ] [dsdu]   [  rsvu    ]
 *
 * this method computes y = KKT * x
 */
bool hiopMatVecKKTFullOpr::times_vec(hiopVector& yvec, const hiopVector& xvec)
{
  const hiopMatrix* Jac_c = kkt_->Jac_c_;
  const hiopMatrix* Jac_d = kkt_->Jac_d_;
  const hiopMatrix* Hess = kkt_->Hess_;

  assert(kkt_->get_perturb_calc());
  hiopVector* delta_wx = kkt_->delta_wx_;
  hiopVector* delta_wd = kkt_->delta_wd_;
  hiopVector* delta_cc = kkt_->delta_cc_;
  hiopVector* delta_cd = kkt_->delta_cd_;
  assert(kkt_->get_perturb_calc());

  delta_wx = kkt_->perturb_calc_->get_curr_delta_wx();
  delta_wd = kkt_->perturb_calc_->get_curr_delta_wd();
  delta_cc = kkt_->perturb_calc_->get_curr_delta_cc();
  delta_cd = kkt_->perturb_calc_->get_curr_delta_cd();

  hiopVectorCompoundPD& y = dynamic_cast<hiopVectorCompoundPD&>(yvec);
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(xvec);

  assert(x.get_num_parts() == y.get_num_parts() && x.get_num_parts() == 12);

  hiopVector* dx_ = &(x.getVector(0));
  hiopVector* dd_ = &(x.getVector(1));
  hiopVector* dyc_ = &(x.getVector(2));
  hiopVector* dyd_ = &(x.getVector(3));
  hiopVector* dsxl_ = &(x.getVector(4));
  hiopVector* dsxu_ = &(x.getVector(5));
  hiopVector* dsdl_ = &(x.getVector(6));
  hiopVector* dsdu_ = &(x.getVector(7));
  hiopVector* dzl_ = &(x.getVector(8));
  hiopVector* dzu_ = &(x.getVector(9));
  hiopVector* dvl_ = &(x.getVector(10));
  hiopVector* dvu_ = &(x.getVector(11));

  hiopVector* yrx_ = &(y.getVector(0));
  hiopVector* yrd_ = &(y.getVector(1));
  hiopVector* yryc_ = &(y.getVector(2));
  hiopVector* yryd_ = &(y.getVector(3));
  hiopVector* yrsxl_ = &(y.getVector(4));
  hiopVector* yrsxu_ = &(y.getVector(5));
  hiopVector* yrsdl_ = &(y.getVector(6));
  hiopVector* yrsdu_ = &(y.getVector(7));
  hiopVector* yrzl_ = &(y.getVector(8));
  hiopVector* yrzu_ = &(y.getVector(9));
  hiopVector* yrvl_ = &(y.getVector(10));
  hiopVector* yrvu_ = &(y.getVector(11));

  // rx = H*dx + delta_wx*I*dx + Jc'*dyc + Jd'*dyd - dzl + dzu
  Hess->timesVec(0.0, *yrx_, +1.0, *dx_);
  yrx_->axzpy(1., *delta_wx, *dx_);
  Jac_c->transTimesVec(1.0, *yrx_, 1.0, *dyc_);
  Jac_d->transTimesVec(1.0, *yrx_, 1.0, *dyd_);
  yrx_->axpy(-1.0, *dzl_);
  yrx_->axpy(1.0, *dzu_);

  // RD = delta_wd_*dd - dyd - dvl + dvu
  yrd_->setToZero();
  yrd_->axpy(-1., *dyd_);
  yrd_->axpy(-1., *dvl_);
  yrd_->axpy(+1., *dvu_);
  yrd_->axzpy(1., *delta_wd, *dd_);

  // RYC = Jc*dx - delta_cc_*dyc
  Jac_c->timesVec(0.0, *yryc_, 1.0, *dx_);
  yryc_->axzpy(-1., *delta_cc, *dyc_);

  // RYD = Jd*dx - dd - delta_cd_*dyd
  Jac_d->timesVec(0.0, *yryd_, 1.0, *dx_);
  yryd_->axpy(-1.0, *dd_);
  yryd_->axzpy(-1., *delta_cd, *dyd_);

  // RXL = -dx + dsxl
  yrsxl_->copyFrom(*dsxl_);
  yrsxl_->axpy(-1.0, *dx_);
  yrsxl_->selectPattern(kkt_->nlp_->get_ixl());

  // RXU = dx + dsxu
  yrsxu_->copyFrom(*dsxu_);
  yrsxu_->axpy(1.0, *dx_);
  yrsxu_->selectPattern(kkt_->nlp_->get_ixu());

  // RDL = -dd + dsdl
  yrsdl_->copyFrom(*dsdl_);
  yrsdl_->axpy(-1.0, *dd_);
  yrsdl_->selectPattern(kkt_->nlp_->get_idl());

  // RDU = dd + dsdu
  yrsdu_->copyFrom(*dsdu_);
  yrsdu_->axpy(1.0, *dd_);
  yrsdu_->selectPattern(kkt_->nlp_->get_idu());

  // rszl = Sxl dzxl + Zxl dsxl
  yrzl_->setToZero();
  yrzl_->axzpy(1.0, *iter_->get_sxl(), *dzl_);
  yrzl_->axzpy(1.0, *iter_->get_zl(), *dsxl_);

  // rszu = Sxu dzxu + Zxu dsxu
  yrzu_->setToZero();
  yrzu_->axzpy(1.0, *iter_->get_sxu(), *dzu_);
  yrzu_->axzpy(1.0, *iter_->get_zu(), *dsxu_);

  // rsvl = Sdl dzdl + Zdl dsdl
  yrvl_->setToZero();
  yrvl_->axzpy(1.0, *iter_->get_sdl(), *dvl_);
  yrvl_->axzpy(1.0, *iter_->get_vl(), *dsdl_);

  // rszu = Sdu dzdu + Zdu dsdu
  yrvu_->setToZero();
  yrvu_->axzpy(1.0, *iter_->get_sdu(), *dvu_);
  yrvu_->axzpy(1.0, *iter_->get_vu(), *dsdu_);

  return true;
}

/**
 * Full KKT matrix is
 * [   H    0   Jc^T  Jd^T |  -I  I   0   0   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  0     0     0    -I  |  0   0  -I   I   |  0   0   0   0  ] [  dd]   [    rd    ]
 * [  Jc    0     0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    -I    0     0  |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [ -I     0     0     0  |  0   0   0   0   |  I   0   0   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0     0  |  0   0   0   0   |  0   I   0   0  ] [ dzu]   [   rxu    ]
 * [  0     -I    0     0  |  0   0   0   0   |  0   0   I   0  ] [ dvl]   [   rdl    ]
 * [  0     I     0     0  |  0   0   0   0   |  0   0   0   I  ] [ dvu]   [   rdu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0     0  | Sl^x 0   0   0   | Zl   0   0   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0     0  |  0  Su^x 0   0   |  0  Zu   0   0  ] [dsxu]   [  rszu    ]
 * [  0     0     0     0  |  0   0  Sl^d 0   |  0   0  Vl   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0     0  |  0   0   0  Su^d |  0   0   0  Vu  ] [dsdu]   [  rsvu    ]
 *
 * this method computes y = KKT' * x
 */
bool hiopMatVecKKTFullOpr::trans_times_vec(hiopVector& yvec, const hiopVector& xvec)
{
  hiopVectorCompoundPD& y = dynamic_cast<hiopVectorCompoundPD&>(yvec);
  const hiopVectorCompoundPD& x = dynamic_cast<const hiopVectorCompoundPD&>(xvec);

  // full KKT is not symmetric!
  const hiopMatrix* Jac_c = kkt_->Jac_c_;
  const hiopMatrix* Jac_d = kkt_->Jac_d_;
  const hiopMatrix* Hess = kkt_->Hess_;

  assert(kkt_->get_perturb_calc());
  hiopVector* delta_wx = kkt_->delta_wx_;
  hiopVector* delta_wd = kkt_->delta_wd_;
  hiopVector* delta_cc = kkt_->delta_cc_;
  hiopVector* delta_cd = kkt_->delta_cd_;
  assert(kkt_->get_perturb_calc());

  delta_wx = kkt_->perturb_calc_->get_curr_delta_wx();
  delta_wd = kkt_->perturb_calc_->get_curr_delta_wd();
  delta_cc = kkt_->perturb_calc_->get_curr_delta_cc();
  delta_cd = kkt_->perturb_calc_->get_curr_delta_cd();

  assert(x.get_num_parts() == y.get_num_parts() && x.get_num_parts() == 12);

  hiopVector* dx_ = &(x.getVector(0));
  hiopVector* dd_ = &(x.getVector(1));
  hiopVector* dyc_ = &(x.getVector(2));
  hiopVector* dyd_ = &(x.getVector(3));
  hiopVector* dsxl_ = &(x.getVector(4));
  hiopVector* dsxu_ = &(x.getVector(5));
  hiopVector* dsdl_ = &(x.getVector(6));
  hiopVector* dsdu_ = &(x.getVector(7));
  hiopVector* dzl_ = &(x.getVector(8));
  hiopVector* dzu_ = &(x.getVector(9));
  hiopVector* dvl_ = &(x.getVector(10));
  hiopVector* dvu_ = &(x.getVector(11));

  hiopVector* yrx_ = &(y.getVector(0));
  hiopVector* yrd_ = &(y.getVector(1));
  hiopVector* yryc_ = &(y.getVector(2));
  hiopVector* yryd_ = &(y.getVector(3));
  hiopVector* yrsxl_ = &(y.getVector(4));
  hiopVector* yrsxu_ = &(y.getVector(5));
  hiopVector* yrsdl_ = &(y.getVector(6));
  hiopVector* yrsdu_ = &(y.getVector(7));
  hiopVector* yrzl_ = &(y.getVector(8));
  hiopVector* yrzu_ = &(y.getVector(9));
  hiopVector* yrvl_ = &(y.getVector(10));
  hiopVector* yrvu_ = &(y.getVector(11));

  // rx = H*dx + delta_wx_*I*dx + Jc'*dyc + Jd'*dyd - dzl + dzu
  Hess->timesVec(0.0, *yrx_, +1.0, *dx_);
  yrx_->axzpy(1., *delta_wx, *dx_);
  Jac_c->transTimesVec(1.0, *yrx_, 1.0, *dyc_);
  Jac_d->transTimesVec(1.0, *yrx_, 1.0, *dyd_);
  yrx_->axpy(-1.0, *dzl_);
  yrx_->axpy(1.0, *dzu_);

  // RD = delta_wd_*dd - dyd - dvl + dvu
  yrd_->setToZero();
  yrd_->axpy(-1., *dyd_);
  yrd_->axpy(-1., *dvl_);
  yrd_->axpy(+1., *dvu_);
  yrd_->axzpy(1., *delta_wd, *dd_);

  // RYC = Jc*dx - delta_cc_*dyc
  Jac_c->timesVec(0.0, *yryc_, 1.0, *dx_);
  yryc_->axzpy(-1., *delta_cc, *dyc_);

  // RYD = Jd*dx - dd - delta_cd_*dyd
  Jac_d->timesVec(0.0, *yryd_, 1.0, *dx_);
  yryd_->axpy(-1.0, *dd_);
  yryd_->axzpy(-1., *delta_cd, *dyd_);

  // RXL = -dx + Sxl*dsxl
  yrsxl_->setToZero();
  yrsxl_->axpy(-1.0, *dx_);
  yrsxl_->axzpy(1.0, *iter_->get_sxl(), *dsxl_);
  yrsxl_->selectPattern(kkt_->nlp_->get_ixl());

  // RXU = dx + Sxu*dsxu
  yrsxu_->copyFrom(*dx_);
  yrsxu_->axzpy(1.0, *iter_->get_sxu(), *dsxu_);
  yrsxu_->selectPattern(kkt_->nlp_->get_ixu());

  // RDL = -dd + Sdl*dsdl
  yrsdl_->setToZero();
  yrsdl_->axpy(-1.0, *dd_);
  yrsdl_->axzpy(1.0, *iter_->get_sdl(), *dsdl_);
  yrsdl_->selectPattern(kkt_->nlp_->get_idl());

  // RDU = dd + Sdu*dsdu
  yrsdu_->setToZero();
  yrsdu_->axpy(1.0, *dd_);
  yrsdu_->axzpy(1.0, *iter_->get_sdu(), *dsdu_);
  yrsdu_->selectPattern(kkt_->nlp_->get_idu());

  // rszl = dzxl + Zxl*dsxl
  yrzl_->copyFrom(*dzl_);
  yrzl_->axzpy(1.0, *iter_->get_zl(), *dsxl_);

  // rszu = dzxu + Zxu*dsxu
  yrzu_->copyFrom(*dzu_);
  yrzu_->axzpy(1.0, *iter_->get_zu(), *dsxu_);

  // rsvl = dzdl + Zdl dsdl
  yrvl_->copyFrom(*dvl_);
  yrvl_->axzpy(1.0, *iter_->get_vl(), *dsdl_);

  // rszu = dzdu + Zdu dsdu
  yrvu_->copyFrom(*dvu_);
  yrvu_->axzpy(1.0, *iter_->get_vu(), *dsdu_);

  return true;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// hiopPrecondKKTOpr
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
hiopPrecondKKTOpr::hiopPrecondKKTOpr(hiopKKTLinSys* kkt, const hiopIterate* iter)
    : kkt_(kkt),
      iter_(iter),
      resid_(nullptr),
      dir_(nullptr)
{
  resid_ = new hiopResidual(kkt_->nlp_);
  dir_ = new hiopIterate(kkt_->nlp_);

  // set compound vector pointed to dir_/resid_
  dir_cv_ = new hiopVectorCompoundPD(dir_);
  res_cv_ = new hiopVectorCompoundPD(resid_);
}

bool hiopPrecondKKTOpr::split_vec_to_build_res(const hiopVector& vec) { return true; }

bool hiopPrecondKKTOpr::combine_dir_to_build_vec(hiopVector& vec) { return true; }

bool hiopPrecondKKTOpr::times_vec(hiopVector& y, const hiopVector& x)
{
  res_cv_->copyFrom(x);

  const bool bret = kkt_->computeDirections(resid_, dir_);

  y.copyFrom(*dir_cv_);

  return bret;
}

bool hiopPrecondKKTOpr::trans_times_vec(hiopVector& y, const hiopVector& x)
{
  // compressed preconditioner is symmetric
  return times_vec(y, x);
}

hiopKKTLinSysNormalEquation::hiopKKTLinSysNormalEquation(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressed(nlp)
{
  rd_tilde_ = Dd_->alloc_clone();
  ryc_tilde_ = nlp->alloc_dual_eq_vec();
  ryd_tilde_ = Dd_->alloc_clone();
  Hx_ = Dx_->alloc_clone();
  Hd_ = Dd_->alloc_clone();
  x_wrk_ = Dx_->alloc_clone();
  d_wrk_ = Dd_->alloc_clone();
}

hiopKKTLinSysNormalEquation::~hiopKKTLinSysNormalEquation()
{
  delete rd_tilde_;
  delete ryc_tilde_;
  delete ryd_tilde_;
  delete Hx_;
  delete Hd_;
  delete x_wrk_;
  delete d_wrk_;
}

bool hiopKKTLinSysNormalEquation::update(const hiopIterate* iter,
                                         const hiopVector* grad_f,
                                         const hiopMatrix* Jac_c,
                                         const hiopMatrix* Jac_d,
                                         hiopMatrix* Hess)
{
  nlp_->runStats.linsolv.reset();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();

  iter_ = iter;
  grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = Hess;

  size_type nx = Hess_->m();
  assert(nx == Hess_->n());
  assert(nx == Jac_c_->n());
  assert(nx == Jac_d_->n());

  // compute barrier diagonals (these change only between outer optimiz iterations)
  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  Dd_->setToZero();
  Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  assert(true == Dd_->allPositive());
#endif
  nlp_->runStats.kkt.tmUpdateInit.stop();

  // factorization + inertia correction if needed
  bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysNormalEquation::computeDirections(const hiopResidual* resid, hiopIterate* dir)
{
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmSolveRhsManip.start();

  const hiopResidual& r = *resid;

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * rx_tilde = rx + Sxl^{-1}*[rszl-Zl*rxl] - Sxu^{-1}*(rszu-Zu*rxu)
   * rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl) - Sdu^{-1}*(rsvu-Vu*rdu)
   */
  rx_tilde_->copyFrom(*r.rx);
  if(nlp_->n_low_local()) {
    // rl:=rszl-Zl*rxl (using dir->x as working buffer)
    hiopVector& rl = *(dir->x);  // temporary working buffer
    rl.copyFrom(*r.rszl);
    rl.axzpy(-1.0, *iter_->zl, *r.rxl);
    // rx_tilde = rx+Sxl^{-1}*rl
    rx_tilde_->axdzpy_w_pattern(1.0, rl, *iter_->sxl, nlp_->get_ixl());
  }
  if(nlp_->n_upp_local()) {
    // ru:=rszu-Zu*rxu (using dir->x as working buffer)
    hiopVector& ru = *(dir->x);  // temporary working buffer
    ru.copyFrom(*r.rszu);
    ru.axzpy(-1.0, *iter_->zu, *r.rxu);
    // rx_tilde = rx_tilde - Sxu^{-1}*ru
    rx_tilde_->axdzpy_w_pattern(-1.0, ru, *iter_->sxu, nlp_->get_ixu());
  }

  // for rd_tilde = rd + Sdl^{-1}*(rsvl-Vl*rdl)-Sdu^{-1}(rsvu-Vu*rdu)
  rd_tilde_->copyFrom(*r.rd);
  if(nlp_->m_ineq_low()) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvl-Vl*rdl
    rd2.copyFrom(*r.rsvl);
    rd2.axzpy(-1.0, *iter_->vl, *r.rdl);
    // rd_tilde +=  Sdl^{-1}*(rsvl-Vl*rdl)
    rd_tilde_->axdzpy_w_pattern(1.0, rd2, *iter_->sdl, nlp_->get_idl());
  }
  if(nlp_->m_ineq_upp() > 0) {
    hiopVector& rd2 = *dir->sdu;
    // rd2=rsvu-Vu*rdu
    rd2.copyFrom(*r.rsvu);
    rd2.axzpy(-1.0, *iter_->vu, *r.rdu);
    // rd_tilde += -Sdu^{-1}(rsvu-Vu*rdu)
    rd_tilde_->axdzpy_w_pattern(-1.0, rd2, *iter_->sdu, nlp_->get_idu());
  }

  /***********************************************************************
   * perform the reduction to the compressed linear system
   * [ ryc_tilde ] = [ Jc  0 ] [ H+Dx+delta_wx_     0       ]^{-1}  [ rx_tilde ] - [ ryc ]
   * [ ryd_tilde ]   [ Jd -I ] [   0           Dd+delta_wd_ ]       [ rd_tilde ]   [ ryd ]
   */

  /***********************************************************************
   * TODO: now we assume H is empty or diagonal
   * hence we have
   * [ ryc_tilde ] = [ Jc ] [H+Dx+delta_wx_]^{-1} [ rx_tilde ] - [ ryc ]
   * [ ryd_tilde ]   [ Jd ] [H+Dx+delta_wx_]^{-1} [ rx_tilde ] - [ Dd+delta_wd_ ]^{-1} [ rd_tilde ] - [ ryd ]
   */
  {
    /* x_wrk_ = [H+Dx+delta_wx_]^{-1} [ rx_tilde ] */
    x_wrk_->copyFrom(*rx_tilde_);
    x_wrk_->componentDiv(*Hx_);

    ryc_tilde_->copyFrom(*r.ryc);
    Jac_c_->timesVec(-1.0, *ryc_tilde_, 1.0, *x_wrk_);

    /* d_wrk_ = [ Dd+delta_wd_ ]^{-1} [ rd_tilde ] */
    d_wrk_->copyFrom(*rd_tilde_);
    d_wrk_->componentDiv(*Hd_);

    ryd_tilde_->copyFrom(*r.ryd);
    Jac_d_->timesVec(-1.0, *ryd_tilde_, 1.0, *x_wrk_);
    ryd_tilde_->axpy(-1.0, *d_wrk_);
  }

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  /***********************************************************************
   * solve the compressed system
   * (be aware that rx_tilde is reused/modified inside this function)
   ***********************************************************************/
  bool sol_ok = solveCompressed(*ryc_tilde_, *ryd_tilde_, *dir->yc, *dir->yd);

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  /***********************************************************************
   * TODO: now we assume H is empty or diagonal
   * hence from
   *   [ H+Dx+delta_wx_     0       ] [dx] = [ rx_tilde ] - [ Jc^T  Jd^T] [dyc]
   *   [   0           Dd+delta_wd_ ] [dd]   [ rd_tilde ]   [  0     -I ] [dyd]
   * we can recover
   *   [dx] = [ H+Dx+delta_wx_ ]^{-1} ( [ rx_tilde ] - [ Jc^T ] [dyc] - [Jd^T] [dyd] )
   *   [dd] = [ Dd+delta_wd_ ]^{-1}   ( [ rd_tilde ] + [dyd] )
   */
  dir->x->copyFrom(*rx_tilde_);
  Jac_c_->transTimesVec(1.0, *dir->x, -1.0, *dir->yc);
  Jac_d_->transTimesVec(1.0, *dir->x, -1.0, *dir->yd);
  dir->x->componentDiv(*Hx_);

  dir->d->copyFrom(*rd_tilde_);
  dir->d->axpy(1.0, *dir->yd);
  dir->d->componentDiv(*Hd_);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  if(false == sol_ok) {
    nlp_->runStats.tmSolverInternal.stop();
    return false;
  }

  const bool bret = compute_directions_for_full_space(resid, dir);

  nlp_->runStats.tmSolverInternal.stop();
  return bret;
}

bool hiopKKTLinSysFull::test_direction(const hiopIterate* dir, hiopMatrix* Hess)
{
  bool retval;
  nlp_->runStats.tmSolverInternal.start();

  if(!x_wrk_) {
    x_wrk_ = nlp_->alloc_primal_vec();
    x_wrk_->setToZero();
  }
  if(!d_wrk_) {
    d_wrk_ = nlp_->alloc_dual_ineq_vec();
    d_wrk_->setToZero();
  }

  hiopVector* sol_x = dir->get_x();
  hiopVector* sol_d = dir->get_d();
  double dWd = 0;
  double xs_nrmsq = 0.0;
  double dbl_wrk;

  assert(perturb_calc_);
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  /* compute xWx = x(H+Dx_)x (for primal var [x,d] */
  Hess_->timesVec(0.0, *x_wrk_, 1.0, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
  x_wrk_->setToZero();
  x_wrk_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
  x_wrk_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
  x_wrk_->componentMult(*sol_x);
  x_wrk_->axzpy(1., *delta_wx_, *sol_x);
  dWd += x_wrk_->dotProductWith(*sol_x);

  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
  d_wrk_->setToZero();
  d_wrk_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  d_wrk_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
  d_wrk_->componentMult(*sol_d);
  d_wrk_->axzpy(1., *delta_wd_, *sol_d);
  dWd += d_wrk_->dotProductWith(*sol_d);

  /* compute rhs for the dWd test */
  dbl_wrk = sol_x->twonorm();
  xs_nrmsq += dbl_wrk * dbl_wrk;
  dbl_wrk = sol_d->twonorm();
  xs_nrmsq += dbl_wrk * dbl_wrk;

  if(dWd < xs_nrmsq * nlp_->options->GetNumeric("neg_curv_test_fact")) {
    // have negative curvature. Add regularization and re-factorize the matrix
    retval = false;
  } else {
    // have positive curvature. Accept this factoraizaiton and direction.
    retval = true;
  }

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

};  // namespace hiop
