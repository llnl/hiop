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

#ifndef HIOP_LOGBARRPROB
#define HIOP_LOGBARRPROB

namespace hiop
{

class hiopLogBarProblem
{
public:
  hiopLogBarProblem(hiopNlpFormulation* nlp_)
      : kappa_d(1e-5),
        nlp(nlp_)
  {
    _grad_x_logbar = nlp->alloc_primal_vec();
    _grad_d_logbar = nlp->alloc_dual_ineq_vec();
  };
  virtual ~hiopLogBarProblem()
  {
    delete _grad_x_logbar;
    delete _grad_d_logbar;
  };

public:  // members
  double mu;
  double f_logbar, f_logbar_trial;
  hiopVector *_grad_x_logbar, *_grad_d_logbar;  // of the log barrier
  // just proxies: keeps pointers to the problem's data and updates LogBar func, grad and all that on the fly
  const hiopIterate *iter, *iter_trial;
  const hiopVector *c_nlp, *d_nlp, *c_nlp_trial, *d_nlp_trial;
  const hiopMatrix *Jac_c_nlp, *Jac_d_nlp;

  // algorithm's parameters
  // factor in computing the linear damping terms used to control unboundness in the log-barrier problem (Section 3.7) */
  double kappa_d;

public:
  // update with the NLP problem data given by the parameters
  inline void updateWithNlpInfo(const hiopIterate& iter_,
                                const double& mu_,
                                const double& f,
                                const hiopVector& c_,
                                const hiopVector& d_,
                                const hiopVector& gradf_,
                                const hiopMatrix& Jac_c_,
                                const hiopMatrix& Jac_d_)
  {
    nlp->runStats.tmSolverInternal.start();

    mu = mu_;
    c_nlp = &c_;
    d_nlp = &d_;
    Jac_c_nlp = &Jac_c_;
    Jac_d_nlp = &Jac_d_;
    iter = &iter_;
    _grad_x_logbar->copyFrom(gradf_);
    _grad_d_logbar->setToZero();
    // add log terms to function
    double aux = -mu * iter->evalLogBarrier();
    f_logbar = f + aux;

#ifdef HIOP_DEEPCHECKS
    nlp->log->write("gradx_log_bar grad_f:", *_grad_x_logbar, hovLinesearchVerb);
#endif
    // add log terms to gradient
    iter->addLogBarGrad_x(mu, *_grad_x_logbar);
    iter->addLogBarGrad_d(mu, *_grad_d_logbar);

#ifdef HIOP_DEEPCHECKS
    nlp->log->write("gradx_log_bar grad_log:", *_grad_x_logbar, hovLinesearchVerb);
#endif

    // add damping terms
    if(kappa_d > 0.) {
      iter->addLinearDampingTermToGrad_x(mu, kappa_d, 1.0, *_grad_x_logbar);
      iter->addLinearDampingTermToGrad_d(mu, kappa_d, 1.0, *_grad_d_logbar);

      f_logbar += iter->linearDampingTerm(mu, kappa_d);
#ifdef HIOP_DEEPCHECKS
      nlp->log->write("gradx_log_bar final, with damping:", *_grad_x_logbar, hovLinesearchVerb);
      nlp->log->write("gradd_log_bar final, with damping:", *_grad_d_logbar, hovLinesearchVerb);
#endif
      nlp->runStats.tmSolverInternal.stop();
    }
  }
  inline void updateWithNlpInfo_trial_funcOnly(const hiopIterate& iter_,
                                               const double& f,
                                               const hiopVector& c_,
                                               const hiopVector& d_)
  {
    nlp->runStats.tmSolverInternal.start();

    c_nlp_trial = &c_;
    d_nlp_trial = &d_;
    iter_trial = &iter_;
    f_logbar_trial = f - mu * iter_trial->evalLogBarrier();
    if(kappa_d > 0.) f_logbar_trial += iter_trial->linearDampingTerm(mu, kappa_d);

    nlp->runStats.tmSolverInternal.stop();
  }

  /* @brief Adds beta*(damping terms) to the gradient `gradx` w.r.t. x */
  inline void addNonLogBarTermsToGrad_x(const double& beta, hiopVector& gradx) const
  {
    if(kappa_d > 0.) iter->addLinearDampingTermToGrad_x(mu, kappa_d, beta, gradx);
  }

  /* @brief Adds beta*(damping terms) to the gradient `gradx` w.r.t. d */
  inline void addNonLogBarTermsToGrad_d(const double& beta, hiopVector& gradd) const
  {
    // if(kappa_d>0.) iter->addLinearDampingTermToGrad_d(mu,kappa_d,beta,gradd);
    if(kappa_d > 0.) iter->addLinearDampingTermToGrad_d(mu, kappa_d, beta, gradd);
  }
  /* grad_log^T * [ dx ] =  grad_f^T * dx + grad_x_dampingTerm^T * dx + grad_d_dampingTerm^T *ds
                  [ dd ]
  */
  inline double directionalDerivative(const hiopIterate& dir)
  {
    nlp->runStats.tmSolverInternal.start();
    double tr = dir.get_x()->dotProductWith(*_grad_x_logbar);
    tr += dir.get_d()->dotProductWith(*_grad_d_logbar);
    nlp->runStats.tmSolverInternal.stop();
    return tr;
  }

protected:
  hiopNlpFormulation* nlp;

private:
  hiopLogBarProblem() {};
  hiopLogBarProblem(const hiopLogBarProblem&) {};
  hiopLogBarProblem& operator=(const hiopLogBarProblem&) { return *this; };
};
}  // namespace hiop
#endif
