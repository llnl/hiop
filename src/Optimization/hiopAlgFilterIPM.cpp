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
 * @file hiopAlgFilterIPM.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>,  LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>,  LLNL
 *
 */

#include "hiopAlgFilterIPM.hpp"

#include "hiopKKTLinSys.hpp"
#include "KktLinSysLowRank.hpp"
#include "hiopKKTLinSysDense.hpp"
#include "hiopKKTLinSysMDS.hpp"
#include "hiopKKTLinSysSparse.hpp"
#include "hiopKKTLinSysSparseCondensed.hpp"
#include "hiopKKTLinSysSparseNormalEqn.hpp"

#include "hiopFRProb.hpp"

#include "hiopCppStdUtils.hpp"

#ifdef HIOP_USE_AXOM
#include "SidreHelper.hpp"
using namespace axom;
#endif

#include <cmath>
#include <cstring>
#include <cassert>
#include <stdio.h>
#include <ctype.h>

namespace hiop
{

hiopAlgFilterIPMBase::hiopAlgFilterIPMBase(hiopNlpFormulation* nlp_in, const bool within_FR)
    : nlp(nlp_in),
      logbar(nullptr),
      it_curr(nullptr),
      it_trial(nullptr),
      dir(nullptr),
      soc_dir(nullptr),
      resid(nullptr),
      resid_trial(nullptr),
      iter_num_(0),
      iter_num_total_(0),
      onenorm_pr_curr_(0.0),
      c_soc(nullptr),
      d_soc(nullptr),
      within_FR_(within_FR),
      pd_perturb_(nullptr),
      fact_acceptor_(nullptr)
{
  // force completion of the nlp's initialization
  nlp->finalizeInitialization();
}

void hiopAlgFilterIPMBase::dealloc_alg_objects()
{
  delete it_curr;
  delete it_trial;
  delete dir;

  delete _c;
  delete _d;
  delete _grad_f;
  delete _Jac_c;
  delete _Jac_d;

  delete _Hess_Lagr;

  delete resid;

  delete _c_trial;
  delete _d_trial;
  delete _grad_f_trial;
  delete _Jac_c_trial;
  delete _Jac_d_trial;

  delete resid_trial;

  delete logbar;

  delete dualsUpdate_;

  delete c_soc;
  delete d_soc;
  delete soc_dir;
}
hiopAlgFilterIPMBase::~hiopAlgFilterIPMBase()
{
  dealloc_alg_objects();
  delete fact_acceptor_;
  delete pd_perturb_;
}

void hiopAlgFilterIPMBase::alloc_alg_objects()
{
  it_curr = new hiopIterate(nlp);
  it_trial = it_curr->alloc_clone();
  dir = it_curr->alloc_clone();

  if(nlp->options->GetString("KKTLinsys") == "full") {
    it_curr->selectPattern();
    it_trial->selectPattern();
    dir->selectPattern();
  }

  logbar = new hiopLogBarProblem(nlp);

  _f_nlp = 0.;
  _f_log = 0.;
  _c = nlp->alloc_dual_eq_vec();
  _d = nlp->alloc_dual_ineq_vec();

  _grad_f = nlp->alloc_primal_vec();
  _Jac_c = nlp->alloc_Jac_c();
  _Jac_d = nlp->alloc_Jac_d();

  _f_nlp_trial = 0.;
  _f_log_trial = 0.;
  _c_trial = nlp->alloc_dual_eq_vec();
  _d_trial = nlp->alloc_dual_ineq_vec();

  _grad_f_trial = nlp->alloc_primal_vec();
  _Jac_c_trial = nlp->alloc_Jac_c();
  _Jac_d_trial = nlp->alloc_Jac_d();

  _Hess_Lagr = nlp->alloc_Hess_Lagr();

  resid = new hiopResidual(nlp);
  resid_trial = new hiopResidual(nlp);

  c_soc = nlp->alloc_dual_eq_vec();
  d_soc = nlp->alloc_dual_ineq_vec();
  soc_dir = it_curr->alloc_clone();
}

void hiopAlgFilterIPMBase::reInitializeNlpObjects()
{
  dealloc_alg_objects();

  alloc_alg_objects();

  // 0 LSQ (default), 1 linear update (more stable)
  duals_update_type = nlp->options->GetString("duals_update_type") == "lsq" ? 0 : 1;
  // 0 LSQ (default), 1 set to zero
  dualsInitializ = nlp->options->GetString("duals_init") == "lsq" ? 0 : 1;

  if(duals_update_type == 0) {
    hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(nlp);
    if(NULL == nlpd) {
      duals_update_type = 1;
      dualsInitializ = 1;
      nlp->log->printf(hovWarning,
                       "Option duals_update_type=lsq not compatible with the requested NLP formulation and will "
                       "be set to duals_update_type=linear together with duals_init=zero\n");
    }
  }

  // parameter based initialization
  if(duals_update_type == 0) {
    // lsq update
    // dualsUpdate_ = new hiopDualsLsqUpdate(nlp);
    dualsUpdate_ = nlp->alloc_duals_lsq_updater();
  } else {
    if(duals_update_type == 1) {
      dualsUpdate_ = new hiopDualsNewtonLinearUpdate(nlp);
    } else {
      assert(false && "duals_update_type has an unrecognized value");
    }
  }
}

void hiopAlgFilterIPMBase::reload_options()
{
  // algorithm parameters parameters
  mu0 = _mu = nlp->options->GetNumeric("mu0");
  kappa_mu = nlp->options->GetNumeric("kappa_mu");       // linear decrease factor
  theta_mu = nlp->options->GetNumeric("theta_mu");       // exponent for higher than linear decrease of mu
  tau_min = nlp->options->GetNumeric("tau_min");         // min value for the fraction-to-the-boundary
  eps_tol = nlp->options->GetNumeric("tolerance");       // absolute error for the nlp
  cons_tol_ = nlp->options->GetNumeric("cons_tol");      // absolute error for the constraints
  dual_tol_ = nlp->options->GetNumeric("dual_tol");      // absolute error for the dual optimality
  comp_tol_ = nlp->options->GetNumeric("comp_tol");      // absolute error for the complementary
  eps_rtol = nlp->options->GetNumeric("rel_tolerance");  // relative error (to errors for the initial point)
  kappa_eps = nlp->options->GetNumeric("kappa_eps");     // relative (to mu) error for the log barrier

  kappa1 = nlp->options->GetNumeric("kappa1");  // projection params for starting point (default 1e-2)
  kappa2 = nlp->options->GetNumeric("kappa2");
  p_smax = nlp->options->GetNumeric("smax");  // threshold for the magnitude of the multipliers

  max_n_it = nlp->options->GetInteger("max_iter");

  accep_n_it = nlp->options->GetInteger("acceptable_iterations");
  eps_tol_accep = nlp->options->GetNumeric("acceptable_tolerance");

  // 0 LSQ (default), 1 linear update (more stable)
  duals_update_type = nlp->options->GetString("duals_update_type") == "lsq" ? 0 : 1;
  // 0 LSQ (default), 1 set to zero
  dualsInitializ = nlp->options->GetString("duals_init") == "lsq" ? 0 : 1;

  if(duals_update_type == 0) {
    hiopNlpDenseConstraints* nlpd = dynamic_cast<hiopNlpDenseConstraints*>(nlp);
    if(NULL == nlpd) {
      // this is sparse or mds linear algebra
      duals_update_type = 1;
      nlp->log->printf(hovWarning,
                       "Option duals_update_type=lsq not compatible with the requested NLP formulation. "
                       " Will use duals_update_type=linear.\n");
    }
  }

  gamma_theta = 1e-5;  // sufficient progress parameters for the feasibility violation
  gamma_phi = 1e-8;    // and log barrier objective
  s_theta = 1.1;       // parameters in the switch condition of
  s_phi = 2.3;         // the linearsearch (equation 19) in
  delta = 1.;          // the WachterBiegler paper
  // parameter in the Armijo rule
  eta_phi = nlp->options->GetNumeric("eta_phi");
  // parameter in resetting the duals to guarantee closedness of the primal-dual logbar Hessian to the primal
  // logbar Hessian
  kappa_Sigma = 1e10;
  _tau = fmax(tau_min, 1.0 - _mu);

  theta_max_fact_ = nlp->options->GetNumeric("theta_max_fact");
  theta_min_fact_ = nlp->options->GetNumeric("theta_min_fact");

  theta_max = 1e7;  // temporary - will be updated after ini pt is computed
  theta_min = 1e7;  // temporary - will be updated after ini pt is computed

  perf_report_kkt_ = "on" == hiop::tolower(nlp->options->GetString("time_kkt"));

  // Set memory space for computations
  // hiop::LinearAlgebraFactory::set_mem_space(nlp->options->GetString("mem_space"));
}

void hiopAlgFilterIPMBase::resetSolverStatus()
{
  n_accep_iters_ = 0;
  solver_status_ = NlpSolve_IncompleteInit;
  filter.clear();
}

int hiopAlgFilterIPMBase::startingProcedure(hiopIterate& it_ini,
                                            double& f,
                                            hiopVector& c,
                                            hiopVector& d,
                                            hiopVector& gradf,
                                            hiopMatrix& Jac_c,
                                            hiopMatrix& Jac_d)
{
  bool duals_avail = false;
  bool slacks_avail = false;
  bool warmstart_avail = false;
  bool ret_bool = false;

  if(nlp->options->GetString("warm_start") == "yes") {
    ret_bool = nlp->get_warmstart_point(*it_ini.get_x(),
                                        *it_ini.get_zl(),
                                        *it_ini.get_zu(),
                                        *it_ini.get_yc(),
                                        *it_ini.get_yd(),
                                        *it_ini.get_d(),
                                        *it_ini.get_vl(),
                                        *it_ini.get_vu());
    warmstart_avail = duals_avail = slacks_avail = true;
  } else {
    ret_bool = nlp->get_starting_point(*it_ini.get_x(),
                                       duals_avail,
                                       *it_ini.get_zl(),
                                       *it_ini.get_zu(),
                                       *it_ini.get_yc(),
                                       *it_ini.get_yd(),
                                       slacks_avail,
                                       *it_ini.get_d());
  }

  if(!ret_bool) {
    nlp->log->printf(hovWarning, "user did not provide a starting point; will be set to all zeros\n");
    it_ini.get_x()->setToZero();
    // in case user wrongly set this to true when he/she returned false
    warmstart_avail = duals_avail = slacks_avail = false;
  }

  // before evaluating the NLP, make sure that iterate, including dual variables are initialized
  // to zero; many of these will be updated later in this method, but we want to make sure
  // that the user's NLP evaluator functions, in particular the Hessian of the Lagrangian,
  // receives initialized arrays

  if(!duals_avail) {
    // initialization for yc and yd
    it_ini.setEqualityDualsToConstant(0.);
  } else {
    // yc and yd were provided by the user
  }

  if(!this->evalNlp_noHess(it_ini, f, c, d, gradf, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Failure in evaluating user provided NLP functions.");
    assert(false);
    return false;
  }

  [[maybe_unused]] const bool do_nlp_scaling = nlp->apply_scaling(c, d, gradf, Jac_c, Jac_d);

  nlp->runStats.tmSolverInternal.start();
  nlp->runStats.tmStartingPoint.start();

  if(!warmstart_avail) {
    it_ini.projectPrimalsXIntoBounds(kappa1, kappa2);
  }

  nlp->runStats.tmStartingPoint.stop();
  nlp->runStats.tmSolverInternal.stop();

  // do function evaluation again after we adjust the primals and/or add scaling
  if(!this->evalNlp_noHess(it_ini, f, c, d, gradf, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Failure in evaluating user provided NLP functions.");
    assert(false);
    return false;
  }

  nlp->runStats.tmSolverInternal.start();
  nlp->runStats.tmStartingPoint.start();

  if(!slacks_avail) {
    it_ini.get_d()->copyFrom(d);
  }

  if(!warmstart_avail) {
    it_ini.projectPrimalsDIntoBounds(kappa1, kappa2);
  }
  int num_adjusted_slacks = it_ini.compute_safe_slacks(it_ini, mu0);

  // adjust small/negative slacks
  if(num_adjusted_slacks > 0) {
    nlp->log->printf(hovWarning, "%d slacks are too small. Adjust corresponding variable slacks!\n", num_adjusted_slacks);
    nlp->adjust_bounds(it_ini);
  }

  if(!duals_avail) {
    // initialization for zl, zu, vl, vu
    it_ini.setBoundsDualsToConstant(1.);
  } else {
    // zl and zu were provided by the user

    // compute vl and vu from vl = mu e ./ sdl and vu = mu e ./ sdu
    // sdl and sdu were initialized above in 'determineSlacks'
    if(!warmstart_avail) {
      it_ini.determineDualsBounds_d(mu0);
    }
  }

  if(!duals_avail) {
    if(0 == dualsInitializ) {
      // LSQ-based initialization of yc and yd

      // is the dualsUpdate_ already the LSQ-based updater?
      hiopDualsLsqUpdate* updater = dynamic_cast<hiopDualsLsqUpdate*>(dualsUpdate_);
      bool deleteUpdater = false;
      if(updater == nullptr) {
        // updater = new hiopDualsLsqUpdate(nlp);
        updater = nlp->alloc_duals_lsq_updater();
        deleteUpdater = true;
      }

      // this will update yc and yd in it_ini
      updater->compute_initial_duals_eq(it_ini, gradf, Jac_c, Jac_d);

      if(deleteUpdater) delete updater;
    } else {
      it_ini.setEqualityDualsToConstant(0.);
    }
  }  // end of if(!duals_avail)
  else {
    // duals eq ('yc' and 'yd') were provided by the user
  }

  // we have the duals
  if(!this->evalNlp_HessOnly(it_ini, *_Hess_Lagr)) {
    assert(false);
    return false;
  }

  nlp->log->write("Using initial point:", it_ini, hovIteration);
  nlp->runStats.tmStartingPoint.stop();
  nlp->runStats.tmSolverInternal.stop();

  solver_status_ = NlpSolve_SolveNotCalled;

  return true;
}

bool hiopAlgFilterIPMBase::evalNlp(hiopIterate& iter,
                                   double& f,
                                   hiopVector& c,
                                   hiopVector& d,
                                   hiopVector& gradf,
                                   hiopMatrix& Jac_c,
                                   hiopMatrix& Jac_d,
                                   hiopMatrix& Hess_L)
{
  bool new_x = true;
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();//local_data_const();
  // //f(x)
  // if(!nlp->eval_f(x, new_x, f)) {
  hiopVector& x = *iter.get_x();
  // f(x)
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x = false;  // same x for the rest

  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }

  // bret = nlp->eval_c        (x, new_x, c.local_data());  assert(bret);
  // bret = nlp->eval_d        (x, new_x, d.local_data());  assert(bret);
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }

  // nlp->log->write("Eq   body c:", c, hovFcnEval);
  // nlp->log->write("Ineq body d:", d, hovFcnEval);

  // bret = nlp->eval_Jac_c    (x, new_x, Jac_c);           assert(bret);
  // bret = nlp->eval_Jac_d    (x, new_x, Jac_d);           assert(bret);
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }
  const hiopVector* yc = iter.get_yc();
  assert(yc);
  const hiopVector* yd = iter.get_yd();
  assert(yd);
  const int new_lambda = true;

  if(!nlp->eval_Hess_Lagr(x, new_x, 1., *yc, *yd, new_lambda, Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_noHess(hiopIterate& iter,
                                          double& f,
                                          hiopVector& c,
                                          hiopVector& d,
                                          hiopVector& gradf,
                                          hiopMatrix& Jac_c,
                                          hiopMatrix& Jac_d)
{
  bool new_x = true;
  // hiopVectorPar& it_x = dynamic_cast<hiopVectorPar&>(*iter.get_x());
  // hiopVectorPar& c=dynamic_cast<hiopVectorPar&>(c_);
  // hiopVectorPar& d=dynamic_cast<hiopVectorPar&>(d_);
  // hiopVectorPar& gradf=dynamic_cast<hiopVectorPar&>(gradf_);
  hiopVector& x = *iter.get_x();
  // f(x)
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x = false;  // same x for the rest

  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }

  // bret = nlp->eval_c        (x, new_x, c.local_data());  assert(bret);
  // bret = nlp->eval_d        (x, new_x, d.local_data());  assert(bret);
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }

  // nlp->log->write("Eq   body c:", c, hovFcnEval);
  // nlp->log->write("Ineq body d:", d, hovFcnEval);

  // bret = nlp->eval_Jac_c    (x, new_x, Jac_c);           assert(bret);
  // bret = nlp->eval_Jac_d    (x, new_x, Jac_d);           assert(bret);
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_HessOnly(hiopIterate& iter, hiopMatrix& Hess_L)
{
  const bool new_x = false;  // precondition is that 'evalNlp_noHess' was called just before

  const hiopVector* yc = iter.get_yc();
  assert(yc);
  const hiopVector* yd = iter.get_yd();
  assert(yd);
  const int new_lambda = true;

  hiopVector& x = *iter.get_x();
  if(!nlp->eval_Hess_Lagr(x, new_x, 1., *yc, *yd, new_lambda, Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::update_log_barrier_params(hiopIterate& it,
                                                     const double& mu_curr,
                                                     const double& tau_curr,
                                                     const bool& elastic_mode_on,
                                                     double& mu_new,
                                                     double& tau_new)
{
  const double target_comp_tol = comp_tol_ / nlp->get_obj_scale();
  double new_mu = std::fmax(0.0, std::fmin(kappa_mu * mu_curr, std::pow(mu_curr, theta_mu)));
  new_mu = std::fmax(new_mu, std::fmin(eps_tol, target_comp_tol) / (10. + 1.));
  if(fabs(new_mu - mu_curr) < 1e-16) {
    return false;
  }
  mu_new = new_mu;
  tau_new = fmax(tau_min, 1.0 - mu_new);

  if(elastic_mode_on) {
    const double target_mu = eps_tol;
    const double bound_relax_perturb_init = nlp->options->GetNumeric("elastic_mode_bound_relax_initial");
    const double bound_relax_perturb_min = nlp->options->GetNumeric("elastic_mode_bound_relax_final");
    double bound_relax_perturb = bound_relax_perturb_init;

    if(nlp->options->GetString("elastic_bound_strategy") == "mu_scaled") {
      bound_relax_perturb = 0.995 * mu_new;
    } else if(nlp->options->GetString("elastic_bound_strategy") == "mu_projected") {
      bound_relax_perturb = (mu_new - target_mu) / (mu0 - target_mu) * (bound_relax_perturb_init - bound_relax_perturb_min) +
                            bound_relax_perturb_min;
    }

    if(bound_relax_perturb > bound_relax_perturb_init) {
      bound_relax_perturb = bound_relax_perturb_init;
    }

    if(bound_relax_perturb < bound_relax_perturb_min) {
      bound_relax_perturb = bound_relax_perturb_min;
    }

    nlp->log->printf(hovLinAlgScalars, "Tighen variable/constraint bounds --- %10.6g\n", bound_relax_perturb);

    nlp->reset_bounds(bound_relax_perturb);

    if(nlp->options->GetString("elastic_mode") != "tighten_bound") {
      assert(nlp->options->GetString("elastic_mode") == "correct_it" ||
             nlp->options->GetString("elastic_mode") == "correct_it_adjust_bound");
      // recompute slacks according to the new bounds
      int num_adjusted_slacks = it.compute_safe_slacks(it, mu_new);

      // adjust small/negative slacks
      if(num_adjusted_slacks > 0) {
        nlp->log->printf(hovLinAlgScalars,
                         "update_log_barrier_params: %d slacks are too small after tightening the bounds. "
                         "Adjust corresponding slacks!\n",
                         num_adjusted_slacks);

        // adjust bounds according to `it`
        if(nlp->options->GetString("elastic_mode") == "correct_it_adjust_bound") {
          nlp->adjust_bounds(it);
        }

        // adjust duals
        bool bret = it.adjustDuals_primalLogHessian(mu_new, kappa_Sigma);
        assert(bret);
      }
    }
    // compute infeasibility theta at trial point, since slacks and/or bounds are modified
    [[maybe_unused]] const double theta_temp = resid->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);
  }  // end of if elastic_mode_on

  return true;
}

double hiopAlgFilterIPMBase::thetaLogBarrier(const hiopIterate& it, const hiopResidual& resid, const double& mu)
{
  // actual nlp errors
  double optim, feas, complem, cons_violation;
  resid.getNlpErrors(optim, feas, complem, cons_violation);
  return feas;
}

bool hiopAlgFilterIPMBase::evalNlpAndLogErrors(const hiopIterate& it,
                                               const hiopResidual& resid,
                                               const double& mu,
                                               double& nlpoptim,
                                               double& nlpfeas,
                                               double& nlpcomplem,
                                               double& nlpoverall,
                                               double& logoptim,
                                               double& logfeas,
                                               double& logcomplem,
                                               double& logoverall,
                                               double& cons_violation)
{
  nlp->runStats.tmSolverInternal.start();

  size_type n = nlp->n_complem(), m = nlp->m();
  // the one norms
  // double nrmDualBou=it.normOneOfBoundDuals();
  // double nrmDualEqu=it.normOneOfEqualityDuals();
  double nrmDualBou, nrmDualEqu;
  it.normOneOfDuals(nrmDualEqu, nrmDualBou);

  nlp->log->printf(hovScalars, "nrmOneDualEqu %g   nrmOneDualBo %g\n", nrmDualEqu, nrmDualBou);
  if(nrmDualBou > 1e+10) {
    nlp->log->printf(hovWarning,
                     "Unusually large bound dual variables (norm1=%g) occured, "
                     "which may cause numerical instabilities if it persists. Convergence "
                     " issues or inacurate optimal solutions may be experienced. Possible causes: "
                     " tight bounds or bad scaling of the optimization variables.\n",
                     nrmDualBou);
    if(nlp->options->GetString("fixed_var") == "remove") {
      nlp->log->printf(hovWarning,
                       "For example, increase 'fixed_var_tolerance' to remove "
                       "additional variables.\n");
    } else if(nlp->options->GetString("fixed_var") == "relax") {
      nlp->log->printf(hovWarning,
                       "For example, increase 'fixed_var_tolerance' to relax "
                       "aditional (tight) variables and/or increase 'fixed_var_perturb' "
                       "to decrease the tightness.\n");
    } else {
      nlp->log->printf(hovWarning,
                       "Potential fixes: fix or relax variables with tight bounds "
                       "(see 'fixed_var' option) or rescale variables.\n");
    }
  }

  // scaling factors
  double sd = fmax(p_smax, (nrmDualBou + nrmDualEqu) / (n + m)) / p_smax;
  double sc = n == 0 ? 0 : fmax(p_smax, nrmDualBou / n) / p_smax;

  sd = fmin(sd, 1e+8);
  sc = fmin(sc, 1e+8);

  // actual nlp errors
  resid.getNlpErrors(nlpoptim, nlpfeas, nlpcomplem, cons_violation);

  // finally, the scaled nlp error
  nlpoverall = fmax(nlpoptim / sd, fmax(cons_violation, nlpcomplem / sc));

  nlp->log->printf(hovScalars,
                   "nlpoverall %g  nloptim %g  sd %g  nlpfeas %g  nlpcomplem %g  sc %g cons_violation %g\n",
                   nlpoverall,
                   nlpoptim,
                   sd,
                   nlpfeas,
                   nlpcomplem,
                   cons_violation,
                   sc);

  // actual log errors
  resid.getBarrierErrors(logoptim, logfeas, logcomplem);

  // finally, the scaled barrier error
  logoverall = fmax(logoptim / sd, fmax(cons_violation, logcomplem / sc));
  nlp->runStats.tmSolverInternal.stop();
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_funcOnly(hiopIterate& iter, double& f, hiopVector& c, hiopVector& d)
{
  bool new_x = true;
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();
  // if(!nlp->eval_f(x, new_x, f)) {
  hiopVector& x = *iter.get_x();
  if(!nlp->eval_f(x, new_x, f)) {
    nlp->log->printf(hovError, "Error occured in user objective evaluation\n");
    return false;
  }
  new_x = false;  // same x for the rest
  if(!nlp->eval_c_d(x, new_x, c, d)) {
    nlp->log->printf(hovError, "Error occured in user constraint(s) function evaluation\n");
    return false;
  }
  return true;
}

bool hiopAlgFilterIPMBase::evalNlp_derivOnly(hiopIterate& iter,
                                             hiopVector& gradf,
                                             hiopMatrix& Jac_c,
                                             hiopMatrix& Jac_d,
                                             hiopMatrix& Hess_L)
{
  bool new_x = false;  // functions were previously evaluated in the line search
  // hiopVector& it_x = *iter.get_x();
  // double* x = it_x.local_data();
  hiopVector& x = *iter.get_x();
  if(!nlp->eval_grad_f(x, new_x, gradf)) {
    nlp->log->printf(hovError, "Error occured in user gradient evaluation\n");
    return false;
  }
  if(!nlp->eval_Jac_c_d(x, new_x, Jac_c, Jac_d)) {
    nlp->log->printf(hovError, "Error occured in user Jacobian function evaluation\n");
    return false;
  }

  const hiopVector* yc = iter.get_yc();
  assert(yc);
  const hiopVector* yd = iter.get_yd();
  assert(yd);
  const int new_lambda = true;
  if(!nlp->eval_Hess_Lagr(x, new_x, 1., *yc, *yd, new_lambda, Hess_L)) {
    nlp->log->printf(hovError, "Error occured in user Hessian function evaluation\n");
    return false;
  }
  return true;
}

/* returns the objective value; valid only after 'run' method has been called */
double hiopAlgFilterIPMBase::getObjective() const
{
  if(solver_status_ == NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->printf(hovError, "getObjective: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_ == NlpSolve_Pending) {
    nlp->log->printf(hovWarning, "getObjective: HiOp has not completed and objective value may not be optimal.");
  }
  return nlp->user_obj(_f_nlp);
}
/* returns the primal vector x; valid only after 'run' method has been called */
void hiopAlgFilterIPMBase::getSolution(double* x) const
{
  if(solver_status_ == NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->printf(hovError, "getSolution: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_ == NlpSolve_Pending) {
    nlp->log->printf(hovWarning, "getSolution: HiOp has not completed yet and solution returned may not be optimal.");
  }
  hiopVector& it_x = *it_curr->get_x();
  // it_curr->get_x()->copyTo(x);
  nlp->user_x(it_x, x);
}

void hiopAlgFilterIPMBase::getDualSolutions(double* zl_a, double* zu_a, double* lambda_a)
{
  if(solver_status_ == NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled) {
    nlp->log->printf(hovError, "getDualSolutions: HiOp did not initialize entirely or the 'run' function was not called.");
  }
  if(solver_status_ == NlpSolve_Pending) {
    nlp->log->printf(hovWarning, "getSolution: HiOp has not completed yet and solution returned may not be optimal.");
  }
  [[maybe_unused]] hiopVector& zl = *it_curr->get_zl();
  [[maybe_unused]] hiopVector& zu = *it_curr->get_zu();

  nlp->get_dual_solutions(*it_curr, zl_a, zu_a, lambda_a);
}

int hiopAlgFilterIPMBase::getNumIterations() const
{
  if(solver_status_ == NlpSolve_IncompleteInit || solver_status_ == NlpSolve_SolveNotCalled)
    nlp->log->printf(hovError, "getNumIterations: HiOp did not initialize or the 'run' function was not called.");
  if(solver_status_ == NlpSolve_Pending)
    nlp->log->printf(hovWarning, "getNumIterations: HiOp has not completed upon this call of 'getNumIterations'");
  return nlp->runStats.nIter;
}

bool hiopAlgFilterIPMBase::checkTermination(const double& err_nlp, const int& iter_num, hiopSolveStatus& status)
{
  if(err_nlp <= eps_tol && _err_nlp_optim <= dual_tol_ && _err_cons_violation <= cons_tol_ &&
     _err_nlp_complem <= comp_tol_) {
    solver_status_ = Solve_Success;
    return true;
  }
  if(iter_num >= max_n_it) {
    solver_status_ = Max_Iter_Exceeded;
    return true;
  }

  if(eps_rtol > 0) {
    if(_err_nlp_optim <= eps_rtol * _err_nlp_optim0 && _err_nlp_feas <= eps_rtol * _err_nlp_feas0 &&
       _err_nlp_complem <= std::max(eps_rtol, 1e-6) * std::min(1., _err_nlp_complem0)) {
      solver_status_ = Solve_Success_RelTol;
      return true;
    }
  }

  if(err_nlp <= eps_tol_accep) {
    n_accep_iters_++;
  } else {
    n_accep_iters_ = 0;
  }

  if(n_accep_iters_ >= accep_n_it) {
    solver_status_ = Solve_Acceptable_Level;
    return true;
  }

  return false;
}
/***** Termination message *****/
void hiopAlgFilterIPMBase::displayTerminationMsg()
{
  std::string strStatsReport = nlp->runStats.get_summary() + nlp->runStats.kkt.get_summary_total();
  switch(solver_status_) {
    case Solve_Success: {
      nlp->log->printf(hovSummary, "Successfull termination.\n%s\n", strStatsReport.c_str());
      break;
    }
    case Solve_Success_RelTol: {
      nlp->log->printf(hovSummary,
                       "Successfull termination (error within the relative tolerance).\n%s\n",
                       strStatsReport.c_str());
      break;
    }
    case Solve_Acceptable_Level: {
      nlp->log->printf(hovSummary, "Solve to only to the acceptable tolerance(s).\n%s\n", strStatsReport.c_str());
      break;
    }
    case Max_Iter_Exceeded: {
      nlp->log->printf(hovSummary,
                       "Maximum number of iterations reached.\n%s\n",
                       strStatsReport.c_str());  // nlp->runStats.getSummary().c_str());
      break;
    }
    case Steplength_Too_Small: {
      nlp->log->printf(hovSummary, "Couldn't solve the problem.\n");
      nlp->log->printf(hovSummary,
                       "Linesearch returned unsuccessfully (small step). Probable cause: "
                       "inaccurate gradients/Jacobians or locally infeasible problem.\n");
      nlp->log->printf(hovSummary, "%s\n", strStatsReport.c_str());
      break;
    }
    case User_Stopped: {
      nlp->log->printf(hovSummary,
                       "Stopped by the user through the user provided iterate callback.\n%s\n",
                       strStatsReport.c_str());
      break;
    }
    case Error_In_FR: {
      nlp->log->printf(hovSummary, "Feasibility restoration problem failed to converge.\n%s\n", strStatsReport.c_str());
      break;
    }
    case Infeasible_Problem: {
      nlp->log->printf(hovSummary,
                       "Inaccurate gradients/Jacobians or locally infeasible problem.\n%s\n",
                       strStatsReport.c_str());
      break;
    }
    case Err_Step_Computation: {
      nlp->log->printf(hovSummary, "Error in step computation/linear algebra (unrecoverable)\n%s\n", strStatsReport.c_str());
      break;
    }
    default: {
      nlp->log->printf(hovSummary, "Unclear why HiOp stopped. This shouldn't happen. \n%s\n", strStatsReport.c_str());
      assert(false && "Do not know why hiop stopped. This shouldn't happen.");
      break;
    }
  };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// hiopAlgFilterIPMQuasiNewton
///////////////////////////////////////////////////////////////////////////////////////////////////
hiopAlgFilterIPMQuasiNewton::hiopAlgFilterIPMQuasiNewton(hiopNlpDenseConstraints* nlp_in, const bool within_FR)
    : hiopAlgFilterIPMBase(nlp_in, within_FR),
      load_state_api_called_(false)
{
  nlpdc = nlp_in;
  reload_options();

  alloc_alg_objects();

  // parameter based initialization
  if(duals_update_type == 0) {
    dualsUpdate_ = nlp->alloc_duals_lsq_updater();
  } else if(duals_update_type == 1) {
    dualsUpdate_ = new hiopDualsNewtonLinearUpdate(nlp);
  } else {
    assert(false && "duals_update_type has an unrecognized value");
  }

  resetSolverStatus();
}

hiopAlgFilterIPMQuasiNewton::~hiopAlgFilterIPMQuasiNewton() {}

hiopSolveStatus hiopAlgFilterIPMQuasiNewton::run()
{
  // hiopNlpFormulation nlp may need an update since user may have changed options and
  // reruning with the same hiopAlgFilterIPMQuasiNewton instance
  nlp->finalizeInitialization();
  // also reload options
  reload_options();

  // if nlp changed internally, we need to reinitialize 'this'
  if(it_curr->get_x()->get_size() != nlp->n() ||
     // Jac_c->get_local_size_n()!=nlpdc->n_local()) { <- this is prone to racing conditions
     _Jac_c->n() != nlpdc->n()) {
    // size of the nlp changed internally ->  reInitializeNlpObjects();
    reInitializeNlpObjects();
  }
  resetSolverStatus();

  // types of linear algebra objects are known now
  auto* Hess = dynamic_cast<HessianDiagPlusRowRank*>(_Hess_Lagr);

  nlp->runStats.initialize();
  nlp->runStats.kkt.initialize();
  ////////////////////////////////////////////////////////////////////////////////////
  // run baby run
  ////////////////////////////////////////////////////////////////////////////////////

  nlp->log->printf(hovSummary, "===============\nHiop SOLVER\n===============\n");
  if(nlp->options->GetString("print_options") != "no") {
    nlp->log->write(nullptr, *nlp->options, hovSummary);
  }

#ifdef HIOP_USE_MPI
  nlp->log->printf(hovSummary, "Using %d MPI ranks.\n", nlp->get_num_ranks());
#endif
  nlp->log->write("---------------\nProblem Summary\n---------------", *nlp, hovSummary);

  nlp->runStats.tmOptimizTotal.start();

  iter_num_ = 0;

  //
  // starting point:
  // - user provided (with slack adjustments and lsq eq. duals initialization)
  // - load checkpoint API (method load_state_from_sidre_group) called before calling this method
  // - checkpoint from file (option "checkpoint_load_on_start")
  //
  if(nlp->options->GetString("checkpoint_load_on_start") != "yes" && !load_state_api_called_) {
    // this also evaluates the nlp
    startingProcedure(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    _mu = mu0;
    iter_num_total_ = 0;
  } else {
    if(!load_state_api_called_) {
      //
      // checkpoint load from file
      //
#ifdef HIOP_USE_AXOM
      // load from file: will populate it_curr, _Hess_lagr, and algorithmic parameters
      auto chkpnt_ok = load_state_from_file(nlp->options->GetString("checkpoint_file"));
      if(!chkpnt_ok) {
        nlp->log->printf(hovWarning, "Using default starting procedure (no checkpoint load!).\n");
        iter_num_total_ = 0;
        // fall back on the default starting procedure (it also evaluates the nlp)
        startingProcedure(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
        _mu = mu0;
        iter_num_total_ = 0;
      }
#else
      nlp->log->printf(hovWarning,
                       "Unexpected checkpoint misconfiguration. "
                       "Will use user-provided starting point.\n");
#endif
    }
    // additionally: need to evaluate the nlp
    if(!this->evalNlp_noHess(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d)) {
      nlp->log->printf(hovError, "Failure in evaluating user NLP functions at loaded checkpoint.");
      return Error_In_User_Function;
    }
    solver_status_ = NlpSolve_SolveNotCalled;
  }

  // update log bar
  logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  nlp->log->printf(hovScalars, "log bar obj: %g\n", logbar->f_logbar);
  // recompute the residuals
  resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

  nlp->log->write("First residual-------------", *resid, hovIteration);

  nlp->runStats.nIter = iter_num_;
  bool disableLS = nlp->options->GetString("accept_every_trial_step") == "yes";

  theta_max = theta_max_fact_ * fmax(1.0, resid->get_theta());
  theta_min = theta_min_fact_ * fmax(1.0, resid->get_theta());

  KktLinSysLowRank* kkt = new KktLinSysLowRank(nlp);
  assert(kkt != nullptr);

  // assign an Null pd_perturb, i.e., all the deltas = 0.0
  pd_perturb_ = new hiopPDPerturbationNull();
  if(!pd_perturb_->initialize(nlp)) {
    delete kkt;
    return SolveInitializationError;
  }
  kkt->set_PD_perturb_calc(pd_perturb_);
  kkt->set_logbar_mu(_mu);

  if(fact_acceptor_) {
    delete fact_acceptor_;
    fact_acceptor_ = nullptr;
  }
  fact_acceptor_ = decideAndCreateFactAcceptor(pd_perturb_, nlp, kkt);
  kkt->set_fact_acceptor(fact_acceptor_);

  _alpha_primal = _alpha_dual = 0;

  _err_nlp_optim0 = -1.;
  _err_nlp_feas0 = -1.;
  _err_nlp_complem0 = -1;

  // --- Algorithm status 'algStatus ----
  //-1 couldn't solve the problem (most likely because small search step. Restauration phase likely needed)
  // 0 stopped due to tolerances, including acceptable tolerance, or relative tolerance
  // 1 max iter reached
  // 2 user stop via the iteration callback

  // int algStatus=0;
  bool bret = true;
  int lsStatus = -1, lsNum = 0;
  int use_soc = 0;
  int use_fr = 0;
  int num_adjusted_slacks = 0;
  bool linsol_safe_mode_on = true;  // always use safe mode in the quasi-newton solver
  bool linsol_forcequick = false;   // always use safe mode in the quasi-newton solver
  bool elastic_mode_on = nlp->options->GetString("elastic_mode") != "none";
  solver_status_ = NlpSolve_Pending;

  while(true) {
    bret = evalNlpAndLogErrors(*it_curr,
                               *resid,
                               _mu,
                               _err_nlp_optim,
                               _err_nlp_feas,
                               _err_nlp_complem,
                               _err_nlp,
                               _err_log_optim,
                               _err_log_feas,
                               _err_log_complem,
                               _err_log,
                               _err_cons_violation);
    if(!bret) {
      solver_status_ = Error_In_User_Function;
      nlp->runStats.tmOptimizTotal.stop();
      delete kkt;
      return Error_In_User_Function;
    }

    nlp->log->printf(
        hovScalars,
        "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n cons_violation: %23.17e\n",
        _err_nlp_feas,
        _err_nlp_optim,
        _err_nlp_complem,
        _err_nlp,
        _err_cons_violation);
    nlp->log->printf(hovScalars,
                     "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
                     _err_log_feas,
                     _err_log_optim,
                     _err_log_complem,
                     _err_log);
    outputIteration(lsStatus, lsNum, use_soc, use_fr);

    if(_err_nlp_optim0 < 0) {  // && _err_nlp_feas0<0 && _err_nlp_complem0<0
      _err_nlp_optim0 = _err_nlp_optim;
      _err_nlp_feas0 = _err_nlp_feas;
      _err_nlp_complem0 = _err_nlp_complem;
    }

    // user callback
    if(!nlp->user_callback_iterate(iter_num_,
                                   _f_nlp,
                                   logbar->f_logbar,
                                   *it_curr->get_x(),
                                   *it_curr->get_zl(),
                                   *it_curr->get_zu(),
                                   *it_curr->get_d(),
                                   *_c,
                                   *_d,
                                   *it_curr->get_yc(),
                                   *it_curr->get_yd(),  // lambda,
                                   _err_nlp_feas,
                                   _err_nlp_optim,
                                   onenorm_pr_curr_,
                                   _mu,
                                   _alpha_dual,
                                   _alpha_primal,
                                   lsNum)) {
      solver_status_ = User_Stopped;
      break;
    }

#ifdef HIOP_USE_AXOM
    // checkpointing - based on options provided by the user
    checkpointing_stuff();
#endif

    /*************************************************
     * Termination check
     ************************************************/
    if(checkTermination(_err_nlp, iter_num_, solver_status_)) {
      break;
    }
    if(NlpSolve_Pending != solver_status_) break;  // failure of the line search or user stopped.

    /************************************************
     * update mu and other parameters
     ************************************************/
    while(_err_log <= kappa_eps * _mu) {
      // update mu and tau (fraction-to-boundary)
      auto mu_updated = update_log_barrier_params(*it_curr, _mu, _tau, elastic_mode_on, _mu, _tau);
      if(!mu_updated) {
        break;
      }
      nlp->log->printf(hovScalars, "Iter[%d] barrier params reduced: mu=%g tau=%g\n", iter_num_, _mu, _tau);

      // update only logbar problem  and residual (the NLP didn't change)
      logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);

      //! should perform only a partial update since NLP didn't change
      resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);
      bret = evalNlpAndLogErrors(*it_curr,
                                 *resid,
                                 _mu,
                                 _err_nlp_optim,
                                 _err_nlp_feas,
                                 _err_nlp_complem,
                                 _err_nlp,
                                 _err_log_optim,
                                 _err_log_feas,
                                 _err_log_complem,
                                 _err_log,
                                 _err_cons_violation);
      if(!bret) {
        solver_status_ = Error_In_User_Function;
        delete kkt;
        return Error_In_User_Function;
      }
      nlp->log->printf(
          hovScalars,
          "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n cons_violation:%23.17e\n",
          _err_nlp_feas,
          _err_nlp_optim,
          _err_nlp_complem,
          _err_nlp,
          _err_cons_violation);
      nlp->log->printf(hovScalars,
                       "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
                       _err_log_feas,
                       _err_log_optim,
                       _err_log_complem,
                       _err_log);

      filter.reinitialize(theta_max);

      if(elastic_mode_on) {
        // reduce mu only once under elastic mode so that bounds do not get tighten too agressively,
        // which may result in small steps and invocation of FR
        break;
      }
    }
    nlp->log->printf(hovScalars, "Iter[%d] logbarObj=%23.17e (mu=%12.5e)\n", iter_num_, logbar->f_logbar, _mu);
    /****************************************************
     * Search direction calculation
     ***************************************************/
    kkt->set_logbar_mu(_mu);
    pd_perturb_->set_mu(_mu);

    nlp->runStats.kkt.start_optimiz_iteration();

    // update the Hessian and kkt system
    Hess->update(*it_curr, *_grad_f, *_Jac_c, *_Jac_d);
    if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr)) {
      nlp->log->write("Unrecoverable error in step computation (factorization) [1]. Will exit here.", hovError);
      delete kkt;
      return solver_status_ = Err_Step_Computation;
    }  // end of if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr))

    auto* fact_acceptor_ic = dynamic_cast<hiopFactAcceptorIC*>(fact_acceptor_);
    if(fact_acceptor_ic) {
      // compute_search_direction call below updates linsol safe mode flag and linsol_safe_mode_lastiter
      if(!compute_search_direction(kkt, linsol_safe_mode_on, linsol_forcequick, iter_num_)) {
        delete kkt;
        return solver_status_ = Err_Step_Computation;
      }
    } else {
      auto* fact_acceptor_dwd = dynamic_cast<hiopFactAcceptorInertiaFreeDWD*>(fact_acceptor_);
      assert(fact_acceptor_dwd);
      // compute_search_direction call below updates linsol safe mode flag and linsol_safe_mode_lastiter
      if(!compute_search_direction_inertia_free(kkt, linsol_safe_mode_on, linsol_forcequick, iter_num_)) {
        // it failed under safe mode
        delete kkt;
        return solver_status_ = Err_Step_Computation;
      }
    }

    nlp->runStats.kkt.end_optimiz_iteration();
    if(perf_report_kkt_) {
      nlp->log->printf(hovSummary, "%s", nlp->runStats.kkt.get_summary_last_iter().c_str());
    }

    nlp->log->printf(hovIteration, "Iter[%d] full search direction -------------\n", iter_num_);
    nlp->log->write("", *dir, hovIteration);
    /***************************************************************
     * backtracking line search
     ****************************************************************/
    nlp->runStats.tmSolverInternal.start();

    // maximum  step
    bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual);
    assert(bret);
    // Step `_alpha_primal` may be reduced when option 'moving_lim_abs' or 'moving_lim_rel' is active.
    // Returned bool indicates if reduction was done or not.
    bret = ensure_moving_lims(*it_curr, *dir, _alpha_primal);

    double theta = onenorm_pr_curr_ = resid->get_theta();  // at it_curr
    double theta_trial;
    nlp->runStats.tmSolverInternal.stop();

    // lsStatus: line search status for the accepted trial point. Needed to update the filter
    //-1 uninitialized (first iteration)
    // 0 unsuccessful (small step size)
    // 1 "sufficient decrease" when far away from solution (theta_trial>theta_min)
    // 2 close to solution but switching condition does not hold, so trial accepted based on "sufficient decrease"
    // 3 close to solution and switching condition is true; trial accepted based on Armijo
    lsStatus = 0;
    lsNum = 0;
    use_soc = 0;
    use_fr = 0;

    bool grad_phi_dx_computed = false, iniStep = true;
    double grad_phi_dx;

    // this will cache the primal infeasibility norm for (reuse)use in the dual updating
    double infeas_nrm_trial = -1.;

    //
    // this is the linesearch loop
    //
    double min_ls_step_size = nlp->options->GetNumeric("min_step_size");
    while(true) {
      nlp->runStats.tmSolverInternal.start();  //---

      // check the step against the minimum step size, but accept small
      // fractionToTheBdry since these may occur for tight bounds at the first iteration(s)
      if(!iniStep && _alpha_primal < min_ls_step_size) {
        nlp->log->write(
            "Minimum step size reached. The problem may be locally infeasible or the "
            "gradient inaccurate. Will try to restore feasibility.",
            hovError);
        solver_status_ = Steplength_Too_Small;

        nlp->runStats.tmSolverInternal.stop();
        break;
      }
      bret = it_trial->takeStep_primals(*it_curr, *dir, _alpha_primal, _alpha_dual);
      assert(bret);
      num_adjusted_slacks = it_trial->compute_safe_slacks(*it_curr, _mu);
      nlp->runStats.tmSolverInternal.stop();  //---

      // evaluate the problem at the trial iterate (functions only)
      if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
        solver_status_ = Error_In_User_Function;
        nlp->runStats.tmOptimizTotal.stop();
        return Error_In_User_Function;
      }

      logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

      nlp->runStats.tmSolverInternal.start();  //---

      // compute infeasibility theta at trial point.
      infeas_nrm_trial = theta_trial = resid->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);

      lsNum++;

      nlp->log->printf(hovLinesearch,
                       "  trial point %d: alphaPrimal=%14.8e barier:(%22.16e)>%15.9e theta:(%22.16e)>%22.16e\n",
                       lsNum,
                       _alpha_primal,
                       logbar->f_logbar,
                       logbar->f_logbar_trial,
                       theta,
                       theta_trial);

      if(disableLS) {
        nlp->runStats.tmSolverInternal.stop();
        break;
      }

      nlp->log->write("Filter IPM: ", filter, hovLinesearch);

      lsStatus = accept_line_search_conditions(theta, theta_trial, _alpha_primal, grad_phi_dx_computed, grad_phi_dx);

      if(lsStatus > 0) {
        nlp->runStats.tmSolverInternal.stop();
        break;
      }

      nlp->runStats.tmSolverInternal.start();
      // second order correction
      if(iniStep && theta <= theta_trial) {
        bool grad_phi_dx_soc_computed = false;
        double grad_phi_dx_soc = 0.0;
        int num_adjusted_slacks_soc = 0;
        lsStatus = apply_second_order_correction(kkt,
                                                 theta,
                                                 theta_trial,
                                                 grad_phi_dx_soc_computed,
                                                 grad_phi_dx_soc,
                                                 num_adjusted_slacks_soc);
        if(lsStatus > 0) {
          num_adjusted_slacks = num_adjusted_slacks_soc;
          grad_phi_dx_computed = grad_phi_dx_soc_computed;
          grad_phi_dx = grad_phi_dx_soc;
          use_soc = 1;
          nlp->runStats.tmSolverInternal.stop();
          break;
        }
      }

      assert(lsStatus == 0);
      _alpha_primal *= 0.5;

      iniStep = false;
      nlp->runStats.tmSolverInternal.stop();
    }  // end of while for the linesearch loop

    // adjust slacks and bounds if necessary
    if(num_adjusted_slacks > 0) {
      nlp->log->printf(hovWarning, "%d slacks are too small. Adjust corresponding variable slacks!\n", num_adjusted_slacks);
      nlp->adjust_bounds(*it_trial);
      // compute infeasibility theta at trial point, since bounds changed --- note that the returned value won't change
      [[maybe_unused]] const double theta_temp = resid->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);
#ifndef NDEBUG
      if(0 == use_soc) {
        // TODO: check why this assertion fails
        // assert(theta_temp == theta_trial);
      }
#endif
    }

    // post line-search stuff
    // filter is augmented whenever the switching condition or Armijo rule do not hold for the trial point that was just
    // accepted
    if(nlp->options->GetString("force_resto") == "yes" && !within_FR_ && iter_num_ == 1) {
      use_fr = apply_feasibility_restoration(kkt);
      if(use_fr) {
        // continue iterations if FR is accepted
        solver_status_ = NlpSolve_Pending;
        nlp->runStats.tmSolverInternal.stop();
      }
    } else if(lsStatus == 1) {
      // need to check switching cond and Armijo to decide if filter is augmented
      if(!grad_phi_dx_computed) {
        grad_phi_dx = logbar->directionalDerivative(*dir);
        grad_phi_dx_computed = true;
      }

      // this is the actual switching condition
      if(grad_phi_dx < 0 && _alpha_primal * pow(-grad_phi_dx, s_phi) > delta * pow(theta, s_theta)) {
        // check armijo
        if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi * _alpha_primal * grad_phi_dx) {
          // filter does not change
        } else {
          // Armijo does not hold
          filter.add(theta_trial, logbar->f_logbar_trial);
        }
      } else {  // switching condition does not hold
        filter.add(theta_trial, logbar->f_logbar_trial);
      }
      nlp->runStats.tmSolverInternal.stop();
    } else if(lsStatus == 2) {
      // switching condition does not hold for the trial
      filter.add(theta_trial, logbar->f_logbar_trial);
      nlp->runStats.tmSolverInternal.stop();
    } else if(lsStatus == 3) {
      // Armijo (and switching condition) hold, nothing to do.
      nlp->runStats.tmSolverInternal.stop();
    } else if(lsStatus == 0) {
      // small step
      if(linsol_safe_mode_on) {
        // try to do FR
        use_fr = apply_feasibility_restoration(kkt);

        if(use_fr) {
          // continue iterations if FR is accepted
          solver_status_ = NlpSolve_Pending;
        }

        // exit the linear solve (compute_search_direction) loop
        nlp->runStats.tmSolverInternal.stop();
        break;
      }
      nlp->runStats.tmSolverInternal.stop();
    } else {
      nlp->runStats.tmSolverInternal.stop();
      assert(false && "unrecognized value for lsStatus");
    }

    if(NlpSolve_Pending != solver_status_) {
      break;  // failure of the line search or user stopped.
    }

    nlp->log->printf(hovScalars,
                     "Iter[%d] -> accepted step primal=[%17.11e] dual=[%17.11e]\n",
                     iter_num_,
                     _alpha_primal,
                     _alpha_dual);
    nlp->log->printf(hovScalars, "Finished iter[%d]  global iter[%d]\n.", iter_num_, iter_num_total_);
    iter_num_++;
    iter_num_total_++;
    nlp->runStats.nIter = iter_num_;

    // fr problem has already updated dual, slacks and NLP functions
    if(!use_fr) {
      nlp->runStats.tmSolverInternal.start();
      // update and adjust the duals
      // this needs to be done before evalNlp_derivOnly so that the user's NLP functions
      // get the updated duals
      assert(infeas_nrm_trial >= 0 && "this should not happen");
      bret = dualsUpdate_->go(*it_curr,
                              *it_trial,
                              _f_nlp,
                              *_c,
                              *_d,
                              *_grad_f,
                              *_Jac_c,
                              *_Jac_d,
                              *dir,
                              _alpha_primal,
                              _alpha_dual,
                              _mu,
                              kappa_Sigma,
                              infeas_nrm_trial);
      assert(bret);
      nlp->runStats.tmSolverInternal.stop();

      // evaluate derivatives at the trial (and to be accepted) trial point
      if(!this->evalNlp_derivOnly(*it_trial, *_grad_f, *_Jac_c, *_Jac_d, *_Hess_Lagr)) {
        solver_status_ = Error_In_User_Function;
        nlp->runStats.tmOptimizTotal.stop();
        delete kkt;
        return Error_In_User_Function;
      }
    }

    nlp->runStats.tmSolverInternal.start();  //-----
    // reuse function values
    _f_nlp = _f_nlp_trial;
    hiopVector* pvec = _c_trial;
    _c_trial = _c;
    _c = pvec;
    pvec = _d_trial;
    _d_trial = _d;
    _d = pvec;

    // update current iterate (do a fast swap of the pointers)
    hiopIterate* pit = it_curr;
    it_curr = it_trial;
    it_trial = pit;

    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", iter_num_);
    nlp->log->write("", *it_curr, hovIteration);
    nlp->runStats.tmSolverInternal.stop();  //-----

    // notify logbar about the changes
    _f_log = _f_nlp;

    logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    // update residual
    resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num_);
    nlp->log->write("", *resid, hovIteration);
  }

  nlp->runStats.tmOptimizTotal.stop();

  // solver_status_ contains the termination information
  displayTerminationMsg();

  // user callback
  nlp->user_callback_solution(solver_status_,
                              *it_curr->get_x(),
                              *it_curr->get_zl(),
                              *it_curr->get_zu(),
                              *_c,
                              *_d,
                              *it_curr->get_yc(),
                              *it_curr->get_yd(),
                              _f_nlp);
  delete kkt;

  return solver_status_;
}

void hiopAlgFilterIPMQuasiNewton::outputIteration(int lsStatus, int lsNum, int use_soc, int use_fr)
{
  if(iter_num_ / 10 * 10 == iter_num_)
    nlp->log->printf(hovSummary, "iter    objective     inf_pr     inf_du   lg(mu)  alpha_du   alpha_pr linesrch\n");

  if(lsStatus == -1)
    nlp->log->printf(hovSummary,
                     "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  -(-)\n",
                     iter_num_total_,
                     _f_nlp / nlp->get_obj_scale(),
                     _err_nlp_feas,
                     _err_nlp_optim,
                     log10(_mu),
                     _alpha_dual,
                     _alpha_primal);
  else {
    char stepType[2];
    if(lsStatus == 1)
      strcpy(stepType, "s");
    else if(lsStatus == 2)
      strcpy(stepType, "h");
    else if(lsStatus == 3)
      strcpy(stepType, "f");
    else
      strcpy(stepType, "?");

    if(use_soc && lsStatus >= 1 && lsStatus <= 3) {
      stepType[0] = (char)::toupper(stepType[0]);
    }

    if(use_fr) {
      strcpy(stepType, "R");
    }

    nlp->log->printf(hovSummary,
                     "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  %d(%s)\n",
                     iter_num_total_,
                     _f_nlp / nlp->get_obj_scale(),
                     _err_nlp_feas,
                     _err_nlp_optim,
                     log10(_mu),
                     _alpha_dual,
                     _alpha_primal,
                     lsNum,
                     stepType);
  }
}

#ifdef HIOP_USE_AXOM

bool hiopAlgFilterIPMQuasiNewton::save_state_to_file(const ::std::string& path) noexcept
{
  try {
    sidre::DataStore ds;
    sidre::Group* group = ds.getRoot()->createGroup("HiOp quasi-Newton alg state");
    this->save_state_to_sidre_group(*group);

    sidre::IOManager writer(this->get_nlp()->get_comm());
    int n_files;
    MPI_Comm_size(this->get_nlp()->get_comm(), &n_files);
    writer.write(ds.getRoot(), n_files, path, sidre::Group::getDefaultIOProtocol());
    return true;
  } catch(const std::exception& exp) {
    nlp->log->printf(hovError, "Error when saving checkpoint to file '%s'\n", path.c_str());
    nlp->log->printf(hovError, "  Addtl info: %s\n", exp.what());
    return false;
  }
}

bool hiopAlgFilterIPMQuasiNewton::load_state_from_file(const ::std::string& path) noexcept
{
  try {
    sidre::DataStore ds;

    sidre::IOManager reader(this->get_nlp()->get_comm());
    auto path2 = SidreHelper::check_path(path);
    reader.read(ds.getRoot(), path2, false);
    nlp->log->printf(hovScalars, "Loaded checkpoint file [%s].\n", path2.c_str());

    const sidre::Group* group = ds.getRoot()->getGroup("HiOp quasi-Newton alg state");
    this->load_state_from_sidre_group(*group);
    return true;
  } catch(const std::exception& exp) {
    nlp->log->printf(hovError, "Error in loading checkpoint from file '%s'\n", path.c_str());
    nlp->log->printf(hovError, "  Addtl info: %s\n", exp.what());
    return false;
  }
}

void hiopAlgFilterIPMQuasiNewton::save_state_to_sidre_group(::axom::sidre::Group& group)
{
  using IndType = sidre::IndexType;

  // iterate state
  // create views for each member that needs to be saved
  SidreHelper::copy_iterate_to_views(group, "alg_iterate_", *it_curr);

  // state of quasi-Newton Hessian approximation
  HessianDiagPlusRowRank& hqn = dynamic_cast<HessianDiagPlusRowRank&>(*_Hess_Lagr);
  const double hqn_params[] = {(double)hqn.l_max_,
                               (double)hqn.l_curr_,
                               hqn.sigma_,
                               hqn.sigma0_,
                               (double)hqn.matrix_changed_};
  const size_type nhqn_params = sizeof(hqn_params) / sizeof(double);
  SidreHelper::copy_array_to_view(group, "Hess_quasiNewton_params", hqn_params, nhqn_params);

  // quasi-Newton Hessian stores the previous iterate and corresponding derivatives
  SidreHelper::copy_iterate_to_views(group, "Hess_quasiNewton_prev_iter", *hqn.it_prev_);
  SidreHelper::copy_vec_to_view(group, "Hess_quasiNewton_prev_grad", *hqn.grad_f_prev_);

  auto* Jac_c = hqn.Jac_c_prev_;
  SidreHelper::copy_array_to_view(group,
                                  "Hess_quasiNewton_prev_Jacc",
                                  Jac_c->local_data_const(),
                                  Jac_c->get_local_size_n() * Jac_c->get_local_size_m());

  auto* Jac_d = hqn.Jac_d_prev_;
  SidreHelper::copy_array_to_view(group,
                                  "Hess_quasiNewton_prev_Jacd",
                                  Jac_d->local_data_const(),
                                  Jac_d->get_local_size_n() * Jac_d->get_local_size_m());

  // quasi-Newton Hessian internal states
  // memory matrices and internal representation
  SidreHelper::copy_array_to_view(group,
                                  "Hess_quasiNewton_St",
                                  hqn.St_->local_data_const(),
                                  hqn.St_->get_local_size_n() * hqn.St_->get_local_size_m());
  SidreHelper::copy_array_to_view(group,
                                  "Hess_quasiNewton_Yt",
                                  hqn.Yt_->local_data_const(),
                                  hqn.Yt_->get_local_size_n() * hqn.Yt_->get_local_size_m());
  SidreHelper::copy_vec_to_view(group, "Hess_quasiNewton_D", *hqn.D_);
  SidreHelper::copy_array_to_view(group,
                                  "Hess_quasiNewton_L",
                                  hqn.L_->local_data_const(),
                                  hqn.L_->get_local_size_n() * hqn.L_->get_local_size_m());

  // algorithmic parameters for this state
  // mu, iteration number, num MPI ranks
  int nranks = 1;
#ifdef HIOP_USE_MPI
  MPI_Comm_size(get_nlp()->get_comm(), &nranks);
#endif
  constexpr double version = HIOP_VERSION_MAJOR * 100 + HIOP_VERSION_MINOR * 10 + HIOP_VERSION_PATCH;
  const double alg_params[] = {_mu, (double)iter_num_total_, (double)nranks, version};
  const size_type nparams = sizeof(alg_params) / sizeof(double);
  SidreHelper::copy_array_to_view(group, "alg_params", alg_params, nparams);

  nlp->log->printf(hovScalars,
                   "Saved checkpoint to sidre::Group : ver %d.%d.%d on %d MPI ranks at "
                   "mu=%12.5e from iter=%d.\n",
                   HIOP_VERSION_MAJOR,
                   HIOP_VERSION_MINOR,
                   HIOP_VERSION_PATCH,
                   nranks,
                   _mu,
                   iter_num_total_);
}

void hiopAlgFilterIPMQuasiNewton::load_state_from_sidre_group(const sidre::Group& group)
{
  load_state_api_called_ = true;

  //
  // algorithmic parameters
  //
  //!!!note: nparams needs to match the nparams from save_state_to_data_store
  const int nparams = 4;
  double alg_params[nparams];
  SidreHelper::copy_array_from_view(group, "alg_params", alg_params, nparams);
  //!!! dev note: match order in save_state_to_data_store
  _mu = alg_params[0];
  iter_num_total_ = alg_params[1];

  int nranks = 1;
#ifdef HIOP_USE_MPI
  MPI_Comm_size(get_nlp()->get_comm(), &nranks);
#endif
  if((int)alg_params[2] != nranks) {
    ::std::stringstream ss;
    ss << "Mismatch in the number of MPI ranks used to checkpoint. Checkpointing was " << "done on " << (int)alg_params[2]
       << " ranks while HiOp currently runs on " << nranks << " ranks.\n";
    throw std::runtime_error(ss.str());
  }
  const int ver_major = ((int)alg_params[3] / 100);
  const int ver_minor = ((int)alg_params[3] - ver_major * 100) / 10;
  const int ver_patch = (int)alg_params[3] - ver_major * 100 - ver_minor * 10;
  nlp->log->printf(hovScalars,
                   "Loaded checkpoint from sidre::Group. Found ver %d.%d.%d on %d MPI ranks at "
                   "mu=%12.5e from iter=%d.\n",
                   ver_major,
                   ver_minor,
                   ver_patch,
                   nranks,
                   _mu,
                   iter_num_total_);

  //
  // iterate states
  //
  SidreHelper::copy_iterate_from_views(group, "alg_iterate_", *it_curr);

  //
  // state of quasi-Newton Hessian approximation
  //
  HessianDiagPlusRowRank& hqn = dynamic_cast<HessianDiagPlusRowRank&>(*_Hess_Lagr);
  //!!!note: nparams needs to match the # of params from save_state_to_sidre_group
  const int nhqn_params = 5;
  double hqn_params[nhqn_params];
  SidreHelper::copy_array_from_view(group, "Hess_quasiNewton_params", hqn_params, nhqn_params);

  const size_type lim_mem_length = (size_type)hqn_params[1];
  // ensure the internals are allocated for this mem length
  hqn.alloc_for_limited_mem(lim_mem_length);

  hqn.l_max_ = (size_type)hqn_params[0];
  hqn.l_curr_ = lim_mem_length;
  hqn.sigma_ = hqn_params[2];
  hqn.sigma0_ = hqn_params[3];
  hqn.matrix_changed_ = hqn_params[4];

  assert(hqn.it_prev_);
  // quasi-Newton Hessian stores the previous iterate and corresponding derivatives
  SidreHelper::copy_iterate_from_views(group, "Hess_quasiNewton_prev_iter", *hqn.it_prev_);
  SidreHelper::copy_vec_from_view(group, "Hess_quasiNewton_prev_grad", *hqn.grad_f_prev_);

  auto* Jac_c = hqn.Jac_c_prev_;
  SidreHelper::copy_array_from_view(group,
                                    "Hess_quasiNewton_prev_Jacc",
                                    Jac_c->local_data(),
                                    Jac_c->get_local_size_n() * Jac_c->get_local_size_m());

  auto* Jac_d = hqn.Jac_d_prev_;
  SidreHelper::copy_array_from_view(group,
                                    "Hess_quasiNewton_prev_Jacd",
                                    Jac_d->local_data(),
                                    Jac_d->get_local_size_n() * Jac_d->get_local_size_m());

  // quasi-Newton Hessian internal states
  // memory matrices
  SidreHelper::copy_array_from_view(group,
                                    "Hess_quasiNewton_St",
                                    hqn.St_->local_data(),
                                    hqn.St_->get_local_size_n() * hqn.St_->get_local_size_m());
  SidreHelper::copy_array_from_view(group,
                                    "Hess_quasiNewton_Yt",
                                    hqn.Yt_->local_data(),
                                    hqn.Yt_->get_local_size_n() * hqn.Yt_->get_local_size_m());
  SidreHelper::copy_vec_from_view(group, "Hess_quasiNewton_D", *hqn.D_);
  SidreHelper::copy_array_from_view(group,
                                    "Hess_quasiNewton_L",
                                    hqn.L_->local_data(),
                                    hqn.L_->get_local_size_n() * hqn.L_->get_local_size_m());
}

void hiopAlgFilterIPMQuasiNewton::checkpointing_stuff()
{
  if(nlp->options->GetString("checkpoint_save") == "no") {
    return;
  }
  int chk_every_N = nlp->options->GetInteger("checkpoint_save_every_N_iter");
  // check iteration
  if(iter_num_ > 0 && iter_num_ % chk_every_N == 0) {
    using ::std::string;
    // replace "#" in checkpointing file with iteration number
    string path = nlp->options->GetString("checkpoint_file");
    const auto s_it_num = ::std::to_string(iter_num_);
    auto pos = path.find("#");
    while(string::npos != pos) {
      path.replace(pos, 1, s_it_num);
      pos = path.find("#", pos);
    }

    nlp->log->printf(hovSummary, "Saving checkpoint at iter %d in '%s'.\n", iter_num_, path.c_str());
    // actual checkpointing via axom::sidre
    save_state_to_file(path);
  }
}
#endif  // HIOP_USE_AXOM

/******************************************************************************************************
 * FULL NEWTON IPM
 *****************************************************************************************************/
hiopAlgFilterIPMNewton::hiopAlgFilterIPMNewton(hiopNlpFormulation* nlp_in, const bool within_FR)
    : hiopAlgFilterIPMBase(nlp_in, within_FR)
{
  reload_options();

  alloc_alg_objects();

  // parameter based initialization
  if(duals_update_type == 0) {
    dualsUpdate_ = nlp->alloc_duals_lsq_updater();
  } else if(duals_update_type == 1) {
    dualsUpdate_ = new hiopDualsNewtonLinearUpdate(nlp);
  } else {
    assert(false && "duals_update_type has an unrecognized value");
  }

  resetSolverStatus();
}

hiopAlgFilterIPMNewton::~hiopAlgFilterIPMNewton() {}

void hiopAlgFilterIPMNewton::reload_options()
{
  auto hess_opt_val = nlp->options->GetString("Hessian");
  if(hess_opt_val != "analytical_exact") {
    // it can occur since "analytical_exact" is not the default value
    nlp->options->set_val("Hessian", "analytical_exact");
    if(nlp->options->is_user_defined("Hessian")) {
      nlp->log->printf(hovWarning,
                       "Option Hessian=%s not compatible with the requested NLP formulation and will "
                       "be set to 'analytical_exact'\n",
                       hess_opt_val.c_str());
    }
  }

  auto duals_update_type = nlp->options->GetString("duals_update_type");
  if("linear" != duals_update_type) {
    // 'duals_update_type' should be 'lsq' or 'linear' for  'Hessian=quasinewton_approx'
    // 'duals_update_type' can only be 'linear' for Newton methods 'Hessian=analytical_exact'

    // warn only if these are defined by the user (option file or via SetXXX methods)
    if(nlp->options->is_user_defined("duals_update_type")) {
      nlp->log->printf(hovWarning,
                       "The option 'duals_update_type=%s' is not valid with 'Hessian=analytical_exact'. "
                       "Will use 'duals_update_type=linear'.[2]\n",
                       duals_update_type.c_str());
    }
    nlp->options->set_val("duals_update_type", "linear");
  }

  hiopAlgFilterIPMBase::reload_options();
}

hiopKKTLinSys* hiopAlgFilterIPMNewton::decideAndCreateLinearSystem(hiopNlpFormulation* nlp)
{
  // hiopNlpMDS* nlpMDS = nullptr;
  hiopNlpMDS* nlpMDS = dynamic_cast<hiopNlpMDS*>(nlp);

  if(nullptr == nlpMDS) {
    hiopNlpSparse* nlpSp = dynamic_cast<hiopNlpSparse*>(nlp);
    if(nullptr == nlpSp) {
      // this is dense linear system. This is the default case.
      std::string strKKT = nlp->options->GetString("KKTLinsys");
      if(strKKT == "xdycyd") {
        return new hiopKKTLinSysDenseXDYcYd(nlp);
      } else {
        //'auto' or 'XYcYd'
        return new hiopKKTLinSysDenseXYcYd(nlp);
      }
    } else {
#ifdef HIOP_SPARSE
      // this is sparse linear system
      std::string strKKT = nlp->options->GetString("KKTLinsys");
      if(strKKT == "full") {
        return new hiopKKTLinSysSparseFull(nlp);
      } else if(strKKT == "xdycyd") {
        return new hiopKKTLinSysCompressedSparseXDYcYd(nlp);
      } else if(strKKT == "condensed") {
        return new hiopKKTLinSysCondensedSparse(nlp);
      } else if(strKKT == "normaleqn") {
        if(nlp->m() > 0) {
          return new hiopKKTLinSysSparseNormalEqn(nlp);
        } else {
          if(nlp->options->is_user_defined("KKTLinsys")) {
            nlp->log->printf(hovWarning,
                             "The option 'KKTLinsys = %s' is not valid for unconstrainted problem. "
                             "Will use 'KKTLinsys = XYcYd'.[2]\n",
                             strKKT.c_str());
          }
          return new hiopKKTLinSysCompressedSparseXYcYd(nlp);
        }
      } else {
        //'auto' or 'XYcYd'
        return new hiopKKTLinSysCompressedSparseXYcYd(nlp);
      }
#endif
    }
  } else {
    return new hiopKKTLinSysCompressedMDSXYcYd(nlp);
  }
  assert(false &&
         "Could not match linear algebra to NLP formulation. Likely, HiOp was not built with "
         "all linear algebra modules/options or with an incorrect combination of them");

  return nullptr;
}

hiopKKTLinSys* hiopAlgFilterIPMNewton::switch_to_safer_KKT(hiopKKTLinSys* kkt_curr,
                                                           const double& mu,
                                                           const int& iter_num,
                                                           bool& linsol_safe_mode_on,
                                                           const int& linsol_safe_mode_max_iters,
                                                           int& linsol_safe_mode_last_iter_switched_on,
                                                           double& theta_mu,
                                                           double& kappa_mu,
                                                           bool& switched)
{
#ifdef HIOP_SPARSE
  if(linsol_safe_mode_on) {
    // attempt switching only when running under "condensed" KKT formulation
    auto* kkt_condensed = dynamic_cast<hiopKKTLinSysCondensedSparse*>(kkt_curr);
    if(kkt_condensed) {
      assert(nlp->options->GetString("KKTLinsys") == "condensed");
      delete kkt_condensed;

      // allocate the "safer" KKT formulation
      auto* kkt = new hiopKKTLinSysCompressedSparseXDYcYd(nlp);

      switched = true;

      // more aggressive mu reduction (this is safe with the stable KKT above)
      theta_mu = 1.2;
      kappa_mu = 0.4;

      kkt->set_safe_mode(linsol_safe_mode_on);

      pd_perturb_->initialize(nlp);
      pd_perturb_->set_mu(_mu);
      kkt->set_PD_perturb_calc(pd_perturb_);

      delete fact_acceptor_;
      // use inertia correction just be safe
      fact_acceptor_ = new hiopFactAcceptorIC(pd_perturb_, nlp->m_eq() + nlp->m_ineq());
      // fact_acceptor_ = decideAndCreateFactAcceptor(pd_perturb_, nlp, kkt);
      kkt->set_fact_acceptor(fact_acceptor_);

      linsol_safe_mode_last_iter_switched_on = iter_num;

      return kkt;
    }  // end of if(kkt)
  }
#endif

// TODO: turn to 0 --- keep using fast mode till linear solver (kkt->update) fails
#if 0
  //
  // MDS starts with fast mode, when linsol_mode = speculative
  // Safe mode is on when IPM is arroching convergence, or a high accurate solution is required
  //
  if(nullptr!=dynamic_cast<hiopNlpMDS*>(nlp)) {
    if("speculative"==hiop::tolower(nlp->options->GetString("linsol_mode"))) {
      const double target_mu = nlp->options->GetNumeric("tolerance");
      if( (false==linsol_safe_mode_on) &&
          (iter_num - linsol_safe_mode_last_iter_switched_on < linsol_safe_mode_max_iters) &&
          (mu < target_mu*10) ){
        // keep using safe mode for couple of iterations when close to convergence
        linsol_safe_mode_on=true;
        switched = true;
      }
    } else {
      //
      // don't switch, keep using the same mode
      //
      switched = false;
    }
    return kkt_curr;
  }
#endif

  //
  // all other KKT formulations -> never switch back to safe mode
  //
  switched = false;
  return kkt_curr;
}

hiopKKTLinSys* hiopAlgFilterIPMNewton::switch_to_fast_KKT(hiopKKTLinSys* kkt_curr,
                                                          const double& mu,
                                                          const int& iter_num,
                                                          bool& linsol_safe_mode_on,
                                                          int& linsol_safe_mode_max_iters,
                                                          int& linsol_safe_mode_last_iter_switched_on,
                                                          double& theta_mu,
                                                          double& kappa_mu,
                                                          bool& switched)

{
  assert("speculative" == hiop::tolower(nlp->options->GetString("linsol_mode")));

#ifdef HIOP_SPARSE
  //
  // Switch to quick mode for condensed
  //
  auto* kkt = dynamic_cast<hiopKKTLinSysCondensedSparse*>(kkt_curr);
  // KKT should not be a condensed KKT (this is what we switch to) and we should be under
  // the condensed KKT user option

  if(nullptr == kkt && nlp->options->GetString("KKTLinsys") == "condensed") {
    if(linsol_safe_mode_on && (iter_num - linsol_safe_mode_last_iter_switched_on > linsol_safe_mode_max_iters) &&
       (mu > 1e-6)) {
      linsol_safe_mode_on = false;

      delete kkt;
      kkt = new hiopKKTLinSysCondensedSparse(nlp);
      switched = true;

      kkt->set_safe_mode(linsol_safe_mode_on);

      // let safe mode do more iterations next time we switch to safe mode
      linsol_safe_mode_max_iters *= 2;

      // reset last iter safe mode was switched on
      linsol_safe_mode_last_iter_switched_on = 100000;

      // decrease mu reduction strategies since they are numerically friendlier with the Cholesky solve
      theta_mu = 1.05;
      kappa_mu = 0.8;

      pd_perturb_->initialize(nlp);
      pd_perturb_->set_mu(mu);
      kkt->set_PD_perturb_calc(pd_perturb_);

      delete fact_acceptor_;
      // use options passed by the user for the IC acceptor
      fact_acceptor_ = decideAndCreateFactAcceptor(pd_perturb_, nlp, kkt);
      kkt->set_fact_acceptor(fact_acceptor_);

      return kkt;
    }
  }
#endif

  // MDS system
  // if linsol_mode = speculative, linsol_safe_mode_on = false by initialization, and hiop starts from fast mode.
  // if in safe mode and mu is large, switch back to fast model after couple of iters.
  //
  if(nullptr != dynamic_cast<hiopNlpMDS*>(nlp)) {
    if(linsol_safe_mode_on && (iter_num - linsol_safe_mode_last_iter_switched_on > linsol_safe_mode_max_iters) &&
       (mu > 1e-6)) {
      linsol_safe_mode_on = false;
      switched = true;

      // let safe mode do more iterations next time we switch to safe mode
      linsol_safe_mode_max_iters *= 2;

      // reset last iter safe mode was switched on
      linsol_safe_mode_last_iter_switched_on = 100000;

      return kkt_curr;
    }
  }

  // all other KKT formulations -> do nothing (and return switched=false)
  switched = false;
  return kkt_curr;
}

hiopFactAcceptor* hiopAlgFilterIPMBase::decideAndCreateFactAcceptor(hiopPDPerturbation* p,
                                                                    hiopNlpFormulation* nlp,
                                                                    hiopKKTLinSys* kkt)
{
  std::string strKKT = nlp->options->GetString("fact_acceptor");
  if(strKKT == "inertia_free") {
#ifdef HIOP_SPARSE
    if(nullptr != dynamic_cast<hiopKKTLinSysCondensedSparse*>(kkt)) {
      // for LinSysCondensedSparse correct inertia is different
      assert(nullptr != dynamic_cast<hiopNlpSparseIneq*>(nlp) && "wrong combination of optimization objects was created");
      return new hiopFactAcceptorInertiaFreeDWD(p, 0);
    }
#endif
    return new hiopFactAcceptorInertiaFreeDWD(p, nlp->m_eq() + nlp->m_ineq());
  } else {
#ifdef HIOP_SPARSE
#ifdef HIOP_USE_RAJA
    if(nullptr != dynamic_cast<hiopKKTLinSysCondensedSparse*>(kkt)) {
      // for LinSysCondensedSparse correct inertia is different
      assert(nullptr != dynamic_cast<hiopNlpSparseIneq*>(nlp) && "wrong combination of optimization objects was created");
      return new hiopFactAcceptorIC(p, 0);
    }
#endif
#endif
    return new hiopFactAcceptorIC(p, nlp->m_eq() + nlp->m_ineq());
  }
}

hiopSolveStatus hiopAlgFilterIPMNewton::run()
{
  // hiopNlpFormulation nlp may need an update since user may have changed options and
  // reruning with the same hiopAlgFilterIPMNewton instance
  nlp->finalizeInitialization();

  // also reload options
  reload_options();

  // if nlp changed internally, we need to reinitialize `this`
  if(it_curr->get_x()->get_size() != nlp->n() ||
     // Jac_c->get_local_size_n()!=nlpdc->n_local()) { <- this is prone to racing conditions
     _Jac_c->n() != nlp->n()) {
    // size of the nlp changed internally ->  reInitializeNlpObjects();
    reInitializeNlpObjects();
  }
  resetSolverStatus();

  nlp->runStats.initialize();
  nlp->runStats.kkt.initialize();

  // todo: have this as option maybe
  // number of safe mode iteration to run once linsol mode is switched to on
  // double every time linsol mode is switched on
  int linsol_safe_mode_max_iters = 10;
  ////////////////////////////////////////////////////////////////////////////////////
  // run baby run
  ////////////////////////////////////////////////////////////////////////////////////

  nlp->log->printf(hovSummary, "===============\nHiop SOLVER\n===============\n");
  if(nlp->options->GetString("print_options") != "no") {
    nlp->log->write(nullptr, *nlp->options, hovSummary);
  }

#ifdef HIOP_USE_MPI
  nlp->log->printf(hovSummary, "Using %d MPI ranks.\n", nlp->get_num_ranks());
#endif
  nlp->log->write("---------------\nProblem Summary\n---------------", *nlp, hovSummary);

  nlp->runStats.tmOptimizTotal.start();

  startingProcedure(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);  // this also evaluates the nlp
  _mu = mu0;

  // update log bar
  logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
  nlp->log->printf(hovScalars, "log bar obj: %g\n", logbar->f_logbar);
  // recompute the residuals
  resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

  nlp->log->write("First residual-------------", *resid, hovIteration);

  iter_num_ = 0;
  iter_num_total_ = 0;
  nlp->runStats.nIter = iter_num_;
  bool disableLS = nlp->options->GetString("accept_every_trial_step") == "yes";

  theta_max = theta_max_fact_ * fmax(1.0, resid->get_theta());
  theta_min = theta_min_fact_ * fmax(1.0, resid->get_theta());

  hiopKKTLinSys* kkt = decideAndCreateLinearSystem(nlp);
  assert(kkt != NULL);

  if(nlp->options->GetString("normaleqn_regularization_priority") == "dual_first" &&
     nlp->options->GetString("KKTLinsys") == "normaleqn") {
    if(nlp->options->GetString("regularization_method") == "randomized") {
      pd_perturb_ = new hiopPDPerturbationDualFirstRand();
    } else {
      pd_perturb_ = new hiopPDPerturbationDualFirstScalar();
    }
  } else {
    if(nlp->options->GetString("regularization_method") == "randomized") {
      pd_perturb_ = new hiopPDPerturbationPrimalFirstRand();
    } else {
      pd_perturb_ = new hiopPDPerturbationPrimalFirstScalar();
    }
  }

  if(!pd_perturb_->initialize(nlp)) {
    delete kkt;
    return SolveInitializationError;
  }

  kkt->set_PD_perturb_calc(pd_perturb_);
  kkt->set_logbar_mu(_mu);

  if(fact_acceptor_) {
    delete fact_acceptor_;
    fact_acceptor_ = nullptr;
  }
  fact_acceptor_ = decideAndCreateFactAcceptor(pd_perturb_, nlp, kkt);
  kkt->set_fact_acceptor(fact_acceptor_);

  _alpha_primal = _alpha_dual = 0;

  _err_nlp_optim0 = -1.;
  _err_nlp_feas0 = -1.;
  _err_nlp_complem0 = -1;

  // --- Algorithm status `algStatus` ----
  //-1 couldn't solve the problem (most likely because small search step. Restauration phase likely needed)
  // 0 stopped due to tolerances, including acceptable tolerance, or relative tolerance
  // 1 max iter reached
  // 2 user stop via the iteration callback

  bool bret = true;
  int lsStatus = -1, lsNum = 0;
  int use_soc = 0;
  int use_fr = 0;
  int num_adjusted_slacks = 0;

  int linsol_safe_mode_last_iter_switched_on = 100000;
  bool linsol_safe_mode_on = "stable" == hiop::tolower(nlp->options->GetString("linsol_mode"));
  bool linsol_forcequick = "forcequick" == hiop::tolower(nlp->options->GetString("linsol_mode"));
  bool elastic_mode_on = nlp->options->GetString("elastic_mode") != "none";
  solver_status_ = NlpSolve_Pending;
  while(true) {
    bret = evalNlpAndLogErrors(*it_curr,
                               *resid,
                               _mu,
                               _err_nlp_optim,
                               _err_nlp_feas,
                               _err_nlp_complem,
                               _err_nlp,
                               _err_log_optim,
                               _err_log_feas,
                               _err_log_complem,
                               _err_log,
                               _err_cons_violation);
    if(!bret) {
      solver_status_ = Error_In_User_Function;
      nlp->runStats.tmOptimizTotal.stop();
      delete kkt;
      return Error_In_User_Function;
    }

    nlp->log->printf(
        hovScalars,
        "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n cons_violation:%23.17e\n",
        _err_nlp_feas,
        _err_nlp_optim,
        _err_nlp_complem,
        _err_nlp,
        _err_cons_violation);
    nlp->log->printf(hovScalars,
                     "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
                     _err_log_feas,
                     _err_log_optim,
                     _err_log_complem,
                     _err_log);
    outputIteration(lsStatus, lsNum, use_soc, use_fr);

    if(_err_nlp_optim0 < 0) {  // && _err_nlp_feas0<0 && _err_nlp_complem0<0
      _err_nlp_optim0 = _err_nlp_optim;
      _err_nlp_feas0 = _err_nlp_feas;
      _err_nlp_complem0 = _err_nlp_complem;
    }

    // user callback
    if(!nlp->user_callback_iterate(iter_num_,
                                   _f_nlp,
                                   logbar->f_logbar,
                                   *it_curr->get_x(),
                                   *it_curr->get_zl(),
                                   *it_curr->get_zu(),
                                   *it_curr->get_d(),
                                   *_c,
                                   *_d,
                                   *it_curr->get_yc(),
                                   *it_curr->get_yd(),  // lambda,
                                   _err_nlp_feas,
                                   _err_nlp_optim,
                                   onenorm_pr_curr_,
                                   _mu,
                                   _alpha_dual,
                                   _alpha_primal,
                                   lsNum)) {
      solver_status_ = User_Stopped;
      break;
    }

    /*************************************************
     * Termination check
     ************************************************/
    if(checkTermination(_err_nlp, iter_num_, solver_status_)) {
      break;
    }
    if(NlpSolve_Pending != solver_status_) break;  // failure of the line search or user stopped.

    /************************************************
     * update mu and other parameters
     ************************************************/
    while(_err_log <= kappa_eps * _mu) {
      // update mu and tau (fraction-to-boundary)
      auto mu_updated = update_log_barrier_params(*it_curr, _mu, _tau, elastic_mode_on, _mu, _tau);
      if(!mu_updated) {
        break;
      }
      nlp->log->printf(hovScalars, "Iter[%d] barrier params reduced: mu=%g tau=%g\n", iter_num_, _mu, _tau);

      // update only logbar problem  and residual (the NLP didn't change)
      logbar->updateWithNlpInfo(*it_curr, _mu, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);

      //! should perform only a partial update since NLP didn't change
      resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

      bret = evalNlpAndLogErrors(*it_curr,
                                 *resid,
                                 _mu,
                                 _err_nlp_optim,
                                 _err_nlp_feas,
                                 _err_nlp_complem,
                                 _err_nlp,
                                 _err_log_optim,
                                 _err_log_feas,
                                 _err_log_complem,
                                 _err_log,
                                 _err_cons_violation);
      if(!bret) {
        solver_status_ = Error_In_User_Function;
        delete kkt;
        return Error_In_User_Function;
      }
      nlp->log->printf(
          hovScalars,
          "  Nlp    errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n cons_violation:%23.17e\n",
          _err_nlp_feas,
          _err_nlp_optim,
          _err_nlp_complem,
          _err_nlp,
          _err_cons_violation);
      nlp->log->printf(hovScalars,
                       "  LogBar errs: pr-infeas:%23.17e   dual-infeas:%23.17e  comp:%23.17e  overall:%23.17e\n",
                       _err_log_feas,
                       _err_log_optim,
                       _err_log_complem,
                       _err_log);

      filter.reinitialize(theta_max);

      if(elastic_mode_on) {
        // reduce mu only once under elastic mode so that bounds do not get tighten too agressively,
        // which may result in small steps and invocation of FR
        break;
      }
    }
    nlp->log->printf(hovScalars, "Iter[%d] logbarObj=%23.17e (mu=%12.5e)\n", iter_num_, logbar->f_logbar, _mu);
    /****************************************************
     * Search direction calculation
     ***************************************************/
    kkt->set_logbar_mu(_mu);
    pd_perturb_->set_mu(_mu);

    // this will cache the primal infeasibility norm for (re)use in the dual updating
    double infeas_nrm_trial;

    nlp->runStats.kkt.start_optimiz_iteration();
    //
    // this is the linear solve (compute_search_direction) loop that iterates at most two times
    //
    //  - two times when the step is small (search direction is assumed to be invalid, of ascent): first time
    // linear solve with safe mode=off failed; second time with safe mode on
    //  - one time when the linear solve with the safe mode off is successfull (descent search direction)
    //
    {
      if(linsol_forcequick) {
        linsol_safe_mode_on = false;
      } else {
        //
        // here linsol mode is speculative or stable
        //

        // see if safe mode needs to be switched off
        if("speculative" == hiop::tolower(nlp->options->GetString("linsol_mode"))) {
          bool switched;
          kkt = switch_to_fast_KKT(kkt,
                                   _mu,
                                   iter_num_,
                                   linsol_safe_mode_on,
                                   linsol_safe_mode_max_iters,
                                   linsol_safe_mode_last_iter_switched_on,
                                   theta_mu,
                                   kappa_mu,
                                   switched);
          if(switched) {
            nlp->log->printf(hovWarning, "Switched to the fast KKT linsys\n");
            assert(false == linsol_safe_mode_on);
          }

        } else {
          assert("stable" == hiop::tolower(nlp->options->GetString("linsol_mode")));
          linsol_safe_mode_on = true;
        }
      }
    }

    for(int linsolve = 1; linsolve <= 2; ++linsolve) {
      bool switched;
      kkt = switch_to_safer_KKT(kkt,
                                _mu,
                                iter_num_,
                                linsol_safe_mode_on,
                                linsol_safe_mode_max_iters,
                                linsol_safe_mode_last_iter_switched_on,
                                theta_mu,
                                kappa_mu,
                                switched);
      if(switched) {
        nlp->log->printf(hovWarning, "Switched to a stable/safe KKT formulation\n");
      }
      kkt->set_safe_mode(linsol_safe_mode_on);

      //
      // update the Hessian and kkt system; usually a matrix factorization occurs
      //
      if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr)) {
        if(linsol_safe_mode_on) {
          nlp->log->write("Unrecoverable error in step computation (factorization) [1]. Will exit here.", hovError);
          delete kkt;
          return solver_status_ = Err_Step_Computation;
        } else {
          // failed with 'linsol_mode'='forcequick' means unrecoverable
          if(linsol_forcequick) {
            nlp->log->write("Unrecoverable error in step computation (factorization) [2]. Will exit here.", hovError);
            delete kkt;
            return solver_status_ = Err_Step_Computation;
          }

          // turn on safe mode to repeat linear solve (kkt->update(...) and kkt->compute_directions_w_IR(...)
          //(meaning additional accuracy and stability is requested, possibly from a new kkt class)
          linsol_safe_mode_on = true;
          // linsol_safe_mode_lastiter = iter_num;

          nlp->log->printf(hovWarning,
                           "Requesting additional accuracy and stability from the KKT linear system "
                           "at iteration %d (safe mode ON) [1]\n",
                           iter_num_);
          continue;
        }
      }  // end of if(!kkt->update(it_curr, _grad_f, _Jac_c, _Jac_d, _Hess_Lagr))

      auto* fact_acceptor_ic = dynamic_cast<hiopFactAcceptorIC*>(fact_acceptor_);
      if(fact_acceptor_ic) {
        bool linsol_safe_mode_on_before = linsol_safe_mode_on;
        // compute_search_direction call below updates linsol safe mode flag and linsol_safe_mode_lastiter
        if(!compute_search_direction(kkt, linsol_safe_mode_on, linsol_forcequick, iter_num_)) {
          if(linsol_safe_mode_on_before || linsol_forcequick) {
            // it fails under safe mode, this is fatal
            delete kkt;
            return solver_status_ = Err_Step_Computation;
          }
          // safe mode was turned on in the above call because kkt->compute_directions_w_IR(...) failed
          continue;
        }
      } else {
        auto* fact_acceptor_dwd = dynamic_cast<hiopFactAcceptorInertiaFreeDWD*>(fact_acceptor_);
        assert(fact_acceptor_dwd);
        bool linsol_safe_mode_on_before = linsol_safe_mode_on;
        // compute_search_direction call below updates linsol safe mode flag and linsol_safe_mode_lastiter
        if(!compute_search_direction_inertia_free(kkt, linsol_safe_mode_on, linsol_forcequick, iter_num_)) {
          if(linsol_safe_mode_on_before || linsol_forcequick) {
            // it failed under safe mode
            delete kkt;
            return solver_status_ = Err_Step_Computation;
          }
          // safe mode was turned on in the above call because kkt->compute_directions_w_IR(...) failed or the number
          // of inertia corrections reached max number allowed
          continue;
        }
      }

      nlp->runStats.kkt.end_optimiz_iteration();
      if(perf_report_kkt_) {
        nlp->log->printf(hovSummary, "%s", nlp->runStats.kkt.get_summary_last_iter().c_str());
      }

      nlp->log->printf(hovIteration, "Iter[%d] full search direction -------------\n", iter_num_);
      nlp->log->write("", *dir, hovIteration);
      /***************************************************************
       * backtracking line search
       ****************************************************************/
      nlp->runStats.tmSolverInternal.start();

      // maximum  step
      bret = it_curr->fractionToTheBdry(*dir, _tau, _alpha_primal, _alpha_dual);
      assert(bret);
      // Step `_alpha_primal` may be reduced when option 'moving_lim_abs' or 'moving_lim_rel' is active.
      // Returned bool indicates if reduction was done or not.
      bret = ensure_moving_lims(*it_curr, *dir, _alpha_primal);

      double theta = onenorm_pr_curr_ = resid->get_theta();  // at it_curr
      double theta_trial;
      nlp->runStats.tmSolverInternal.stop();

      // lsStatus: line search status for the accepted trial point. Needed to update the filter
      //-1 uninitialized (first iteration)
      // 0 unsuccessful (small step size)
      // 1 "sufficient decrease" when far away from solution (theta_trial>theta_min)
      // 2 close to solution but switching condition does not hold; trial accepted based on "sufficient decrease"
      // 3 close to solution and switching condition is true; trial accepted based on Armijo
      lsStatus = 0;
      lsNum = 0;
      use_soc = 0;
      use_fr = 0;

      bool grad_phi_dx_computed = false, iniStep = true;
      double grad_phi_dx;

      // this will cache the primal infeasibility norm for (re)use in the dual updating
      infeas_nrm_trial = -1.;
      //
      // linesearch loop
      //
      double min_ls_step_size = nlp->options->GetNumeric("min_step_size");
      while(true) {
        nlp->runStats.tmSolverInternal.start();  //---

        // check the step against the minimum step size, but accept small
        // fractionToTheBdry since these may occur for tight bounds at the first iteration(s)
        if(!iniStep && _alpha_primal < min_ls_step_size) {
          if(linsol_safe_mode_on) {
            nlp->log->write(
                "Minimum step size reached. The problem may be locally infeasible or the "
                "gradient inaccurate. Will try to restore feasibility.",
                hovError);
            solver_status_ = Steplength_Too_Small;
          } else {
            // (silently) take the step if not under safe mode
            lsStatus = 0;
          }
          nlp->runStats.tmSolverInternal.stop();
          break;
        }
        bret = it_trial->takeStep_primals(*it_curr, *dir, _alpha_primal, _alpha_dual);
        assert(bret);
        num_adjusted_slacks = it_trial->compute_safe_slacks(*it_curr, _mu);
        nlp->runStats.tmSolverInternal.stop();  //---

        // evaluate the problem at the trial iterate (functions only)
        if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
          solver_status_ = Error_In_User_Function;
          nlp->runStats.tmOptimizTotal.stop();
          delete kkt;
          return Error_In_User_Function;
        }

        logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

        nlp->runStats.tmSolverInternal.start();  //---

        // compute infeasibility theta at trial point.
        infeas_nrm_trial = theta_trial = resid->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);

        lsNum++;

        nlp->log->printf(hovLinesearch,
                         "  trial point %d: alphaPrimal=%14.8e barier:(%22.16e)>%15.9e "
                         "theta:(%22.16e)>%22.16e\n",
                         lsNum,
                         _alpha_primal,
                         logbar->f_logbar,
                         logbar->f_logbar_trial,
                         theta,
                         theta_trial);

        if(disableLS) {
          nlp->runStats.tmSolverInternal.stop();
          break;
        }

        nlp->log->write("Filter IPM: ", filter, hovLinesearch);

        lsStatus = accept_line_search_conditions(theta, theta_trial, _alpha_primal, grad_phi_dx_computed, grad_phi_dx);

        if(lsStatus > 0) {
          nlp->runStats.tmSolverInternal.stop();
          break;
        }

        // second order correction
        if(iniStep && theta <= theta_trial) {
          bool grad_phi_dx_soc_computed = false;
          double grad_phi_dx_soc = 0.0;
          int num_adjusted_slacks_soc = 0;
          lsStatus = apply_second_order_correction(kkt,
                                                   theta,
                                                   theta_trial,
                                                   grad_phi_dx_soc_computed,
                                                   grad_phi_dx_soc,
                                                   num_adjusted_slacks_soc);
          if(lsStatus > 0) {
            num_adjusted_slacks = num_adjusted_slacks_soc;
            grad_phi_dx_computed = grad_phi_dx_soc_computed;
            grad_phi_dx = grad_phi_dx_soc;
            use_soc = 1;
            nlp->runStats.tmSolverInternal.stop();
            break;
          }
        }

        assert(lsStatus == 0);
        _alpha_primal *= 0.5;

        iniStep = false;
        nlp->runStats.tmSolverInternal.stop();
      }  // end of while for the linesearch loop

      nlp->runStats.tmSolverInternal.start();
      // adjust slacks and bounds if necessary
      if(num_adjusted_slacks > 0) {
        nlp->log->printf(hovWarning,
                         "%d slacks are too small. Adjust corresponding variable slacks!\n",
                         num_adjusted_slacks);
        nlp->adjust_bounds(*it_trial);
        // compute infeasibility theta at trial point, since bounds changed --- note that the returned value won't change
        [[maybe_unused]] const double theta_temp = resid->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);
#ifndef NDEBUG
        if(0 == use_soc) {
          // TODO: check why this assertion fails
          // assert(theta_temp == theta_trial);
        }
#endif
      }

      // post line-search: filter is augmented whenever the switching condition or Armijo rule do not
      // hold for the trial point that was just accepted
      if(nlp->options->GetString("force_resto") == "yes" && !within_FR_ && iter_num_ == 1) {
        use_fr = apply_feasibility_restoration(kkt);
        if(use_fr) {
          // continue iterations if FR is accepted
          solver_status_ = NlpSolve_Pending;
          nlp->runStats.tmSolverInternal.stop();
          break;
        }
      } else if(lsStatus == 1) {
        // need to check switching cond and Armijo to decide if filter is augmented
        if(!grad_phi_dx_computed) {
          grad_phi_dx = logbar->directionalDerivative(*dir);
          grad_phi_dx_computed = true;
        }

        // this is the actual switching condition
        if(grad_phi_dx < 0 && (_alpha_primal * pow(-grad_phi_dx, s_phi) > delta * pow(theta, s_theta))) {
          // check armijo
          if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi * _alpha_primal * grad_phi_dx) {
            // filter does not change
          } else {
            // Armijo does not hold
            filter.add(theta_trial, logbar->f_logbar_trial);
          }
        } else {  // switching condition does not hold
          filter.add(theta_trial, logbar->f_logbar_trial);
        }
        nlp->runStats.tmSolverInternal.stop();
        break;  // from the linear solve (compute_search_direction) loop

      } else if(lsStatus == 2) {
        // switching condition does not hold for the trial
        filter.add(theta_trial, logbar->f_logbar_trial);

        nlp->runStats.tmSolverInternal.stop();
        break;  // from the linear solve (compute_search_direction) loop

      } else if(lsStatus == 3) {
        // Armijo (and switching condition) hold, nothing to do.

        nlp->runStats.tmSolverInternal.stop();
        break;  // from the linear solve (compute_search_direction) loop

      } else if(lsStatus == 0) {
        //
        // small step
        //

        if(linsol_safe_mode_on) {
          // try to do FR
          use_fr = apply_feasibility_restoration(kkt);

          if(use_fr) {
            // continue iterations if FR is accepted
            solver_status_ = NlpSolve_Pending;
          }

          // exit the linear solve (compute_search_direction) loop
          nlp->runStats.tmSolverInternal.stop();
          break;
        } else {
          // here false == linsol_safe_mode_on
          if(linsol_forcequick) {
            // this is  likely catastrophic as under 'linsol_mode'='forcequick' we deliberately
            // won't switch to safe mode
            //
            // however take the update;
            // if the update doesn't pass the convergence test, the optimiz. loop will exit

            // first exit the linear solve (compute_search_direction) loop
            nlp->runStats.tmSolverInternal.stop();
            break;
          }

          linsol_safe_mode_on = true;
          // linsol_safe_mode_lastiter = iter_num;

          nlp->log->printf(hovWarning,
                           "Requesting additional accuracy and stability from the KKT linear system "
                           "at iteration %d (safe mode ON) [2]\n",
                           iter_num_);

          // repeat linear solve (compute_search_direction) in safe mode (meaning additional accuracy
          // and stability is requested)
          nlp->runStats.tmSolverInternal.stop();
          continue;
        }
      } else {
        nlp->runStats.tmSolverInternal.stop();
        assert(false && "unrecognized value for lsStatus");
      }
    }  // end of the linear solve (compute_search_direction) loop

    if(NlpSolve_Pending != solver_status_) {
      break;  // failure of the line search or user stopped.
    }

    nlp->log->printf(hovScalars,
                     "Iter[%d] -> accepted step primal=[%17.11e] dual=[%17.11e]\n",
                     iter_num_,
                     _alpha_primal,
                     _alpha_dual);
    iter_num_++;
    iter_num_total_++;
    nlp->runStats.nIter = iter_num_;

    // fr problem has already updated dual, slacks and NLP functions
    if(!use_fr) {
      nlp->runStats.tmSolverInternal.start();
      // update and adjust the duals
      // this needs to be done before evalNlp_derivOnly so that the user's NLP functions
      // get the updated duals
      assert(infeas_nrm_trial >= 0 && "this should not happen");
      bret = dualsUpdate_->go(*it_curr,
                              *it_trial,
                              _f_nlp,
                              *_c,
                              *_d,
                              *_grad_f,
                              *_Jac_c,
                              *_Jac_d,
                              *dir,
                              _alpha_primal,
                              _alpha_dual,
                              _mu,
                              kappa_Sigma,
                              infeas_nrm_trial);
      assert(bret);
      nlp->runStats.tmSolverInternal.stop();

      // evaluate derivatives at the trial (and to be accepted) trial point
      if(!this->evalNlp_derivOnly(*it_trial, *_grad_f, *_Jac_c, *_Jac_d, *_Hess_Lagr)) {
        solver_status_ = Error_In_User_Function;
        nlp->runStats.tmOptimizTotal.stop();
        delete kkt;
        return Error_In_User_Function;
      }
    }

    nlp->runStats.tmSolverInternal.start();  //-----
    // reuse function values
    _f_nlp = _f_nlp_trial;
    hiopVector* pvec = _c_trial;
    _c_trial = _c;
    _c = pvec;
    pvec = _d_trial;
    _d_trial = _d;
    _d = pvec;

    //
    // update current iterate (do a fast swap of the pointers)
    //
    hiopIterate* pit = it_curr;
    it_curr = it_trial;
    it_trial = pit;

    nlp->log->printf(hovIteration, "Iter[%d] -> full iterate:", iter_num_);
    nlp->log->write("", *it_curr, hovIteration);
    nlp->runStats.tmSolverInternal.stop();  //-----

    // notify logbar about the changes
    _f_log = _f_nlp;

    logbar->updateWithNlpInfo(*it_curr, _mu, _f_log, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d);
    // update residual
    resid->update(*it_curr, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

    nlp->log->printf(hovIteration, "Iter[%d] full residual:-------------\n", iter_num_);
    nlp->log->write("", *resid, hovIteration);
  }

  nlp->runStats.tmOptimizTotal.stop();

  // solver_status_ contains the termination information
  displayTerminationMsg();

  // user callback
  nlp->user_callback_solution(solver_status_,
                              *it_curr->get_x(),
                              *it_curr->get_zl(),
                              *it_curr->get_zu(),
                              *_c,
                              *_d,
                              *it_curr->get_yc(),
                              *it_curr->get_yd(),
                              _f_nlp);
  delete kkt;

  return solver_status_;
}

void hiopAlgFilterIPMNewton::outputIteration(int lsStatus, int lsNum, int use_soc, int use_fr)
{
  if(iter_num_ / 10 * 10 == iter_num_)
    nlp->log->printf(hovSummary, "iter    objective     inf_pr     inf_du   lg(mu)  alpha_du   alpha_pr linesrch\n");

  if(lsStatus == -1)
    nlp->log->printf(hovSummary,
                     "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  -(-)\n",
                     iter_num_total_,
                     _f_nlp / nlp->get_obj_scale(),
                     _err_nlp_feas,
                     _err_nlp_optim,
                     log10(_mu),
                     _alpha_dual,
                     _alpha_primal);
  else {
    char stepType[2];
    if(lsStatus == 1)
      strcpy(stepType, "s");
    else if(lsStatus == 2)
      strcpy(stepType, "h");
    else if(lsStatus == 3)
      strcpy(stepType, "f");
    else
      strcpy(stepType, "?");

    if(use_soc && lsStatus >= 1 && lsStatus <= 3) {
      stepType[0] = (char)::toupper(stepType[0]);
    }

    if(use_fr) {
      lsNum = 0;
      strcpy(stepType, "R");
    }

    nlp->log->printf(hovSummary,
                     "%4d %14.7e %7.3e  %7.3e %6.2f  %7.3e  %7.3e  %d(%s)\n",
                     iter_num_total_,
                     _f_nlp / nlp->get_obj_scale(),
                     _err_nlp_feas,
                     _err_nlp_optim,
                     log10(_mu),
                     _alpha_dual,
                     _alpha_primal,
                     lsNum,
                     stepType);
  }
}

bool hiopAlgFilterIPMBase::ensure_moving_lims(const hiopIterate& it, const hiopIterate& dir, double& alpha_pr)
{
  auto moving_lim_rel = nlp->options->GetNumeric("moving_lim_rel");
  if(moving_lim_rel > 0) {
    const auto alpha_pr_in = alpha_pr;
    alpha_pr = moving_lim_rel * alpha_pr;
    nlp->log->printf(hovLinesearch,
                     "Moving lim (relative, [%7.3e]): step reduced: %7.3e -> %7.3e.\n",
                     moving_lim_rel,
                     alpha_pr_in,
                     alpha_pr);
    return true;
  }

  auto moving_lim_abs = nlp->options->GetNumeric("moving_lim_abs");
  if(moving_lim_abs > 0) {
    const auto alpha_pr_in = alpha_pr;
    auto x_nrm = dir.get_x()->infnorm();
    auto step_nrm = alpha_pr * x_nrm;
    if(step_nrm > moving_lim_abs) {
      alpha_pr = moving_lim_abs / step_nrm;
      nlp->log->printf(hovLinesearch,
                       "Moving lim (absolute, [%7.3e]): step reduced: %7.3e -> %7.3e.\n",
                       moving_lim_abs,
                       alpha_pr_in,
                       alpha_pr);
      return true;
    } else {
      nlp->log->printf(hovLinesearch,
                       "Moving lim (absolute, [%7.3e]) satisfied, step (norm %7.3e) not reduced.\n",
                       moving_lim_abs,
                       step_nrm);
    }
  }  // end of moving lim abs
  return false;
}

int hiopAlgFilterIPMBase::accept_line_search_conditions(const double theta_curr,
                                                        const double theta_trial,
                                                        const double alpha_primal,
                                                        bool& grad_phi_dx_computed,
                                                        double& grad_phi_dx)
{
  int bret = 0;
  trial_is_rejected_by_filter = false;

  // Do the cheap, "sufficient progress" test first, before more involved/expensive tests.
  // This simple test is good enough when iterate is far away from solution
  if(theta_curr >= theta_min) {
    // check the sufficient decrease condition (18)
    if(theta_trial <= (1 - gamma_theta) * theta_curr ||
       logbar->f_logbar_trial <= logbar->f_logbar - gamma_phi * theta_curr) {
      // trial good to go
      nlp->log->printf(hovLinesearchVerb,
                       "Linesearch: accepting based on suff. decrease "
                       "(far from solution)\n");
      bret = 1;
    } else {
      // there is no sufficient progress
      trial_is_rejected_by_filter = false;
      bret = 0;
      return bret;
    }

    // check filter condition
    if(filter.contains(theta_trial, logbar->f_logbar_trial)) {
      // it is in the filter, reject this trial point
      trial_is_rejected_by_filter = true;
      bret = 0;
    }
    return bret;
  } else {
    // if(theta_curr<theta_min,  then check the switching condition and, if true, rely on Armijo rule.
    // first compute grad_phi^T d_x if it hasn't already been computed
    if(!grad_phi_dx_computed) {
      grad_phi_dx = logbar->directionalDerivative(*dir);
      grad_phi_dx_computed = true;
    }
    nlp->log->printf(hovLinesearch, "Linesearch: grad_phi_dx = %22.15e\n", grad_phi_dx);

    // this is the actual switching condition (19)
    if(grad_phi_dx < 0. && alpha_primal * pow(-grad_phi_dx, s_phi) > delta * pow(theta_curr, s_theta)) {
      // test Armijo
      if(logbar->f_logbar_trial <= logbar->f_logbar + eta_phi * alpha_primal * grad_phi_dx) {
        nlp->log->printf(hovLinesearchVerb, "Linesearch: accepting based on Armijo (switch cond also passed)\n");

        // iterate good to go since it satisfies Armijo
        bret = 3;
      } else {
        // Armijo is not satisfied
        trial_is_rejected_by_filter = false;
        bret = 0;
        return bret;
      }

      // check filter condition
      if(filter.contains(theta_trial, logbar->f_logbar_trial)) {
        // it is in the filter, reject this trial point
        trial_is_rejected_by_filter = true;
        bret = 0;
      }
      return bret;
    } else {  // switching condition does not hold

      // ok to go with  "sufficient progress" condition even when close to solution, provided the
      // switching condition is not satisfied

      // check the filter and the sufficient decrease condition (18)
      if(theta_trial <= (1 - gamma_theta) * theta_curr ||
         logbar->f_logbar_trial <= logbar->f_logbar - gamma_phi * theta_curr) {
        // trial good to go
        nlp->log->printf(hovLinesearchVerb, "Linesearch: accepting based on suff. decrease (switch cond also passed)\n");
        bret = 2;
      } else {
        // there is no sufficient progress
        trial_is_rejected_by_filter = false;
        return bret;
      }

      // check filter condition
      if(filter.contains(theta_trial, logbar->f_logbar_trial)) {
        // it is in the filter, reject this trial point
        trial_is_rejected_by_filter = true;
        bret = 0;
      }
      return bret;
    }  // end of else: switching condition does not hold
    assert(0 && "cannot reach here!");
  }  // end of else: theta_trial<theta_min
}

int hiopAlgFilterIPMBase::apply_second_order_correction(hiopKKTLinSys* kkt,
                                                        const double theta_curr,
                                                        const double theta_trial0,
                                                        bool& grad_phi_dx_computed,
                                                        double& grad_phi_dx,
                                                        int& num_adjusted_slacks)
{
  int max_soc_iter = nlp->options->GetInteger("max_soc_iter");
  double kappa_soc = nlp->options->GetNumeric("kappa_soc");

  if(max_soc_iter == 0) {
    return false;
  }

  if(!soc_dir) {
    soc_dir = dir->alloc_clone();
    if(nlp->options->GetString("KKTLinsys") == "full") {
      soc_dir->selectPattern();
    }
    c_soc = nlp->alloc_dual_eq_vec();
    d_soc = nlp->alloc_dual_ineq_vec();
  }

  double theta_trial_last = 0.;
  double theta_trial = theta_trial0;
  double alpha_primal_soc = _alpha_primal;
  double alpha_dual_soc = alpha_primal_soc;

  int num_soc = 0;
  bool bret = true;
  int ls_status = 0;

  // set initial c/d for soc
  c_soc->copyFrom(nlp->get_crhs());
  c_soc->axpy(-1.0, *_c);

  d_soc->copyFrom(*it_curr->get_d());
  d_soc->axpy(-1.0, *_d);

  while(num_soc < max_soc_iter && (num_soc == 0 || theta_trial <= kappa_soc * theta_trial_last)) {
    theta_trial_last = theta_trial;

    c_soc->scale(alpha_primal_soc);
    c_soc->axpy(1.0, nlp->get_crhs());
    c_soc->axpy(-1.0, *_c_trial);

    d_soc->scale(alpha_primal_soc);
    d_soc->axpy(1.0, *it_trial->get_d());
    d_soc->axpy(-1.0, *_d_trial);

    // compute rhs for soc. Use resid_trial since it hasn't been used
    resid_trial->update_soc(*it_curr, *c_soc, *d_soc, *_grad_f, *_Jac_c, *_Jac_d, *logbar);

    // solve for search directions
    bret = kkt->computeDirections(resid_trial, soc_dir);
    assert(bret);

    // Compute step size
    bret = it_curr->fractionToTheBdry(*soc_dir, _tau, alpha_primal_soc, alpha_dual_soc);
    assert(bret);

    // Compute trial point
    bret = it_trial->takeStep_primals(*it_curr, *soc_dir, alpha_primal_soc, alpha_dual_soc);
    assert(bret);
    num_adjusted_slacks = it_trial->compute_safe_slacks(*it_curr, _mu);

    // evaluate the problem at the trial iterate (functions only)
    if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
      solver_status_ = Error_In_User_Function;
      return Error_In_User_Function;
    }

    logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

    // compute infeasibility theta at trial point.
    theta_trial = resid_trial->compute_nlp_infeasib_onenorm(*it_trial, *_c_trial, *_d_trial);

    ls_status = accept_line_search_conditions(theta_curr, theta_trial, _alpha_primal, grad_phi_dx_computed, grad_phi_dx);

    if(ls_status > 0) {
      _alpha_primal = alpha_primal_soc;
      dir->copyFrom(*soc_dir);
      resid->copyFrom(*resid_trial);
      break;
    } else {
      num_soc++;
    }
  }
  return ls_status;
}

bool hiopAlgFilterIPMBase::apply_feasibility_restoration(hiopKKTLinSys* kkt)
{
  bool fr_solved = true;
  bool reset_dual = true;
  if(!within_FR_) {
    // try soft FR first
    bool is_soft_fr = solve_soft_feasibility_restoration(kkt);
    if(is_soft_fr) {
      // variables have already been updated inside the above function
      return true;
    }

    // continue robust FR
    hiopNlpFormulation* nlpFR{nullptr};
    hiopNlpMDS* nlpMDS = dynamic_cast<hiopNlpMDS*>(nlp);
    if(nlpMDS == nullptr) {
      hiopNlpSparse* nlpSp = dynamic_cast<hiopNlpSparse*>(nlp);
      if(nullptr == nlpSp) {
        hiopNlpDenseConstraints* nlpD = dynamic_cast<hiopNlpDenseConstraints*>(nlp);
        assert(nlpD && "Unkown system is provided. Please pick one from Dense, Sparse or MDS. ");
        // this is Dense system
        hiopFRProbDense nlp_fr_interface(*this);
        nlpFR = new hiopNlpDenseConstraints(nlp_fr_interface, nlp->options->GetString("options_file_fr_prob").c_str());
        fr_solved = solve_feasibility_restoration(kkt, *nlpFR);
        if(fr_solved) {
          nlp->log->printf(hovScalars, "FR problem provides sufficient reduction in primal feasibility!\n");
          // FR succeeds, update it_trial->x and it_trial->d to the next search point
          it_trial->get_x()->copyFrom(nlp_fr_interface.get_fr_sol_x());
          it_trial->get_d()->copyFrom(nlp_fr_interface.get_fr_sol_d());
          reset_var_from_fr_sol(kkt, reset_dual = true);
        }
      } else {
        // this is Sparse linear system
        hiopFRProbSparse nlp_fr_interface(*this);
        nlpFR = new hiopNlpSparse(nlp_fr_interface, nlp->options->GetString("options_file_fr_prob").c_str());
        fr_solved = solve_feasibility_restoration(kkt, *nlpFR);
        if(fr_solved) {
          nlp->log->printf(hovScalars, "FR problem provides sufficient reduction in primal feasibility!\n");
          // FR succeeds, update it_trial->x and it_trial->d to the next search point
          it_trial->get_x()->copyFrom(nlp_fr_interface.get_fr_sol_x());
          it_trial->get_d()->copyFrom(nlp_fr_interface.get_fr_sol_d());
          reset_var_from_fr_sol(kkt, reset_dual = true);
        }
      }
    } else {
      // this is MDS system
      hiopFRProbMDS nlp_fr_interface(*this);
      nlpFR = new hiopNlpMDS(nlp_fr_interface, nlp->options->GetString("options_file_fr_prob").c_str());
      fr_solved = solve_feasibility_restoration(kkt, *nlpFR);
      if(fr_solved) {
        nlp->log->printf(hovScalars, "FR problem provides sufficient reduction in primal feasibility!\n");
        // FR succeeds, update it_trial->x and it_trial->d to the next search point
        it_trial->get_x()->copyFrom(nlp_fr_interface.get_fr_sol_x());
        it_trial->get_d()->copyFrom(nlp_fr_interface.get_fr_sol_d());
        reset_var_from_fr_sol(kkt, reset_dual = true);
      }
    }
    delete nlpFR;
  } else {
    // FR problem inside a FR problem, see equation (33)
    // use wildcard function to update primal variables x
    it_trial->copyFrom(*it_curr);
    if(!nlp->user_force_update(iter_num_,
                               _f_nlp,
                               *it_trial->get_x(),
                               *it_trial->get_zl(),
                               *it_trial->get_zu(),
                               *_c,
                               *_d,
                               *it_trial->get_yc(),
                               *it_trial->get_yd(),
                               _mu,
                               _alpha_dual,
                               _alpha_primal)) {
      solver_status_ = Error_In_FR;
      fr_solved = false;
    } else {
      nlp->log->printf(hovSummary, "FR problem converged! Now apply LSQ to reset duals.\n");
      reset_var_from_fr_sol(kkt, reset_dual = false);
    }
  }

  return fr_solved;
}

bool hiopAlgFilterIPMBase::solve_feasibility_restoration(hiopKKTLinSys* kkt, hiopNlpFormulation& nlpFR)
{
  nlpFR.options->SetStringValue("Hessian", nlp->options->GetString("Hessian").c_str());
  nlpFR.options->SetStringValue("duals_update_type", "linear");
  nlpFR.options->SetStringValue("duals_init", "zero");
  nlpFR.options->SetStringValue("compute_mode", nlp->options->GetString("compute_mode").c_str());
  nlpFR.options->SetStringValue("mem_space", nlp->options->GetString("mem_space").c_str());
  nlpFR.options->SetStringValue("KKTLinsys", "xdycyd");
  nlpFR.options->SetIntegerValue("verbosity_level", 0);
  nlpFR.options->SetStringValue("warm_start", "yes");
  nlpFR.options->SetNumericValue("bound_relax_perturb", 0.0);
  nlpFR.options->SetStringValue("scaling_type", "none");

  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  double theta_ref = resid->getInfeasInfNorm();  // at current point, i.e., reference point
  double mu_FR = std::max(_mu, theta_ref);

  nlpFR.options->SetNumericValue("mu0", mu_FR);

  hiopSolveStatus FR_status;
  if(nlpFR.options->GetString("Hessian") == "analytical_exact") {
    hiopAlgFilterIPMNewton solver(&nlpFR, true);  // solver fr problem
    FR_status = solver.run();
  } else {
    hiopNlpDenseConstraints* nlpFR_dense = dynamic_cast<hiopNlpDenseConstraints*>(&nlpFR);
    assert(nlpFR_dense);
    hiopAlgFilterIPM solver(nlpFR_dense, true);  // solver fr problem
    FR_status = solver.run();
  }

  if(FR_status == User_Stopped) {
    // FR succeeds
    return true;
  } else if(FR_status == Solve_Success || FR_status == Solve_Acceptable_Level) {
    solver_status_ = Infeasible_Problem;
    return false;
  } else {
    solver_status_ = Error_In_FR;
    return false;
  }
}

bool hiopAlgFilterIPMBase::reset_var_from_fr_sol(hiopKKTLinSys* kkt, bool reset_dual)
{
  // FR succeeds, it_trial->x and it_trial->d have been updated. Now we update other values for the next iter
  if(!this->evalNlp_noHess(*it_trial, _f_nlp, *_c, *_d, *_grad_f, *_Jac_c, *_Jac_d)) {
    nlp->log->printf(hovError, "Failure in evaluating user provided NLP functions.");
    assert(false);
    return false;
  } else {
    nlp->log->printf(hovScalars, "FR: Update slacks and duals from the modified primals.\n");
  }
  // determine other slacks
  //  it_trial->determineSlacks();  // TODO: adjust small slacks after hard FR?

  int num_adjusted_slacks = it_trial->compute_safe_slacks(*it_curr, mu0);

  // adjust small/negative slacks
  if(num_adjusted_slacks > 0) {
    nlp->log->printf(hovWarning, "%d slacks are too small. Adjust corresponding variable slacks!\n", num_adjusted_slacks);
    nlp->adjust_bounds(*it_trial);
  }

  // compute dx = x_{k+1} - x_k, assuming all the fr iterations contrubute one step
  dir->get_x()->copyFrom(*it_trial->get_x());
  dir->get_x()->axpy(-1.0, *it_curr->get_x());
  dir->get_d()->copyFrom(*it_trial->get_d());
  dir->get_d()->axpy(-1.0, *it_curr->get_d());

  if(reset_dual) {
    // compute directions for bound duals (zl, zu, vl, vu)
    kkt->compute_directions_for_full_space(resid, dir);

    // compute bound duals (zl, zu, vl, vu) for trial point
    double alpha_primal_temp = 1.0;
    double alpha_dual_temp = 1.0;
    bool bret = it_curr->fractionToTheBdry(*dir, _tau, alpha_primal_temp, alpha_dual_temp);
    assert(bret);

    it_trial->takeStep_duals(*it_curr, *dir, alpha_primal_temp, alpha_dual_temp);

    // TODO: set this as a user option. Now we set duals to 0.0 as the default option
    bool reset_dual_from_lsq_after_FR = false;

    if(reset_dual_from_lsq_after_FR) {
      // solve a LSQ to update yc and yd
      hiopDualsLsqUpdate* updater = dynamic_cast<hiopDualsLsqUpdate*>(dualsUpdate_);
      bool deleteUpdater = false;
      if(!updater) {
        updater = nlp->alloc_duals_lsq_updater();
        deleteUpdater = true;
      }
      // this will update yc and yd in it_trial
      updater->go(*it_trial, *_grad_f, *_Jac_c, *_Jac_d);
      if(deleteUpdater) {
        delete updater;
      }
    } else {
      it_trial->setEqualityDualsToConstant(0.);
    }
  }

  // set step size to 1
  _alpha_primal = 1.0;
  _alpha_dual = 1.0;

  return true;
}

bool hiopAlgFilterIPMBase::solve_soft_feasibility_restoration(hiopKKTLinSys* kkt)
{
  int max_soft_fr_iter = 10;  // nlp->options->GetInteger("max_soft_fr_iter");
  double kappa_f = 0.999;     // nlp->options->GetNumeric("kappa_f");
  int num_soft_fr = 0;

  if(max_soft_fr_iter == 0 || kappa_f == 0.0) {
    return false;
  }

  // use vectors from second order correction
  if(!soc_dir) {
    soc_dir = dir->alloc_clone();
    if(nlp->options->GetString("KKTLinsys") == "full") {
      soc_dir->selectPattern();
    }
    c_soc = nlp->alloc_dual_eq_vec();
    d_soc = nlp->alloc_dual_ineq_vec();
  }

  // shortcut --- use soc_dir as a temporary solution
  hiopIterate* soft_dir = soc_dir;

  double kkt_err_curr = resid->get_nrmOne_bar_optim() + resid->get_nrmOne_bar_feasib();
  ;
  double kkt_err_trial;
  double alpha_primal_soft;
  double alpha_dual_soft;
  double infeas_nrm_soft = 0;  // to avoid uninitialized use below

  bool bret = false;

  while(num_soft_fr < max_soft_fr_iter) {
    // solve for search directions
    if(num_soft_fr == 0) {
      soft_dir->copyFrom(*dir);
      _c_trial->copyFrom(*_c);
      _d_trial->copyFrom(*_d);

      bret = true;
    } else {
      // evaluate the problem at the trial iterate (functions only)
      if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
        solver_status_ = Error_In_User_Function;
        return true;
      }
      // compute rhs for soft feasibility restoration. Use resid_trial since it hasn't been used
      resid_trial->update(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial, *_grad_f, *_Jac_c, *_Jac_d, *logbar);
      bret = kkt->compute_directions_w_IR(resid_trial, soft_dir);
    }
    assert(bret);

    // Compute step size
    bret = it_curr->fractionToTheBdry(*soft_dir, _tau, alpha_primal_soft, alpha_dual_soft);
    alpha_primal_soft = std::min(alpha_primal_soft, alpha_dual_soft);
    alpha_dual_soft = alpha_primal_soft;
    assert(bret);

    // Compute trial point
    bret = it_trial->takeStep_primals(*it_curr, *soft_dir, alpha_primal_soft, alpha_dual_soft);
    assert(bret);
    it_trial->determineSlacks();  // TODO: adjust small slacks in soft FR?

    // evaluate the problem at the trial iterate (functions only)
    if(!this->evalNlp_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial)) {
      solver_status_ = Error_In_User_Function;
      return true;
    }

    // update and adjust the duals
    bret = dualsUpdate_->go(*it_curr,
                            *it_trial,
                            _f_nlp_trial,
                            *_c_trial,
                            *_d_trial,
                            *_grad_f,
                            *_Jac_c,
                            *_Jac_d,
                            *soft_dir,
                            alpha_primal_soft,
                            alpha_dual_soft,
                            _mu,
                            kappa_Sigma,
                            infeas_nrm_soft);
    assert(bret);

    logbar->updateWithNlpInfo_trial_funcOnly(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial);

    // compute primal-dual error at trial point.
    resid_trial->update(*it_trial, _f_nlp_trial, *_c_trial, *_d_trial, *_grad_f, *_Jac_c, *_Jac_d, *logbar);
    kkt_err_trial = resid_trial->get_nrmOne_bar_optim() + resid_trial->get_nrmOne_bar_feasib();

    // sufficient reduction in the KKT error is not achieved, return
    if(kkt_err_trial > kappa_f * kkt_err_curr) {
      bret = false;
      break;
    }

    // check filter condition
    double theta_trial = resid_trial->get_nrmOne_bar_feasib();
    if(filter.contains(theta_trial, logbar->f_logbar_trial)) {
      // it is in the filter, reject this trial point and continue the iterates
      num_soft_fr++;
    } else {
      // continue the regular iterate from the trial point
      bret = true;
      break;
    }
  }
  return bret;
}

bool hiopAlgFilterIPMBase::compute_search_direction(hiopKKTLinSys* kkt,
                                                    bool& linsol_safe_mode_on,
                                                    const bool linsol_forcequick,
                                                    const int iter_num)
{
  //
  // solve for search directions
  //
  if(!kkt->compute_directions_w_IR(resid, dir)) {
    if(linsol_safe_mode_on) {
      nlp->log->write("Unrecoverable error in step computation (solve)[1]. Will exit here.", hovError);
      return false;  //  will trigger a solver_status_ = Err_Step_Computation;
    } else {
      if(linsol_forcequick) {
        nlp->log->write("Unrecoverable error in step computation (solve)[2]. Will exit here.", hovError);
        return false;  // will trigger a solver_status_ = Err_Step_Computation;
      }
      linsol_safe_mode_on = true;
      // linsol_safe_mode_lastiter = iter_num;

      nlp->log->printf(hovWarning,
                       "Requesting additional accuracy and stability from the KKT linear system "
                       "at iteration %d (safe mode ON) [3]\n",
                       iter_num);

      // return false and use safe mode to repeat linear solve (kkt->update(...) and kkt->compute_directions_w_IR(...)
      // (meaning additional accuracy and stability is requested, possibly from a new kkt class)
      return false;
    }
  }  // end of if(!kkt->compute_directions_w_IR(resid, dir))

  // at this point all is good in terms of searchDirections computations as far as the linear solve
  // is concerned; the search direction can be of ascent because some fast factorizations do not
  // support inertia calculation; this case will be handled later on in the optimization loop

  return true;
}

bool hiopAlgFilterIPMBase::compute_search_direction_inertia_free(hiopKKTLinSys* kkt,
                                                                 bool& linsol_safe_mode_on,
                                                                 const bool linsol_forcequick,
                                                                 const int iter_num)
{
  size_type num_refact = 0;
  const size_t max_refactorization = 10u;

  while(true) {
    //
    // solve for search directions
    //
    if(!kkt->compute_directions_w_IR(resid, dir)) {
      if(linsol_safe_mode_on) {
        nlp->log->write("Unrecoverable error in step computation (solve)[1]. Will exit here.", hovError);
        return false;  // solver_status_ = Err_Step_Computation;
      } else {
        if(linsol_forcequick) {
          nlp->log->write("Unrecoverable error in step computation (solve)[2]. Will exit here.", hovError);
          return false;  // solver_status_ = Err_Step_Computation;
        }
        linsol_safe_mode_on = true;

        nlp->log->printf(hovWarning,
                         "Requesting additional accuracy and stability from the KKT linear system "
                         "at iteration %d (safe mode ON)[4]\n",
                         iter_num);

        // return false and use safe mode to repeat linear solve (kkt->update(...) and kkt->compute_directions_w_IR(...)
        // (meaning additional accuracy and stability is requested, possibly from a new kkt class)
        return false;
      }
    }  // end of if(!kkt->compute_directions_w_IR(resid, dir))

    // at this point all is good in terms of searchDirections computations as far as the linear solve
    // is concerned; the search direction can be of ascent because some fast factorizations do not
    // support inertia calculation; this case will be handled later on in this loop
    //( //! todo nopiv inertia calculation ))

    if(kkt->test_direction(dir, _Hess_Lagr)) {
      break;
    } else {
      if(static_cast<size_t>(num_refact) >= max_refactorization) {
        nlp->log->printf(hovError,
                         "Reached max number (%lu) of refactorization within an outer iteration.\n",
                         max_refactorization);
        return false;
      }
      kkt->factorize_inertia_free();
      num_refact++;
      nlp->runStats.kkt.nUpdateICCorr++;
    }
  }

  return true;
}

}  // namespace hiop
