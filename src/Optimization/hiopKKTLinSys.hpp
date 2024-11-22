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

#ifndef HIOP_KKTLINSYSY
#define HIOP_KKTLINSYSY

#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "HessianDiagPlusRowRank.hpp"
#include "hiopPDPerturbation.hpp"
#include "hiopLinSolver.hpp"
#include "hiopFactAcceptor.hpp"
#include "hiopKrylovSolver.hpp"
#include "hiopVectorCompoundPD.hpp"

#include "hiopCppStdUtils.hpp"

namespace hiop
{

class hiopMatVecKKTFullOpr;
class hiopPrecondKKTOpr;

class hiopKKTLinSys
{
public:
  hiopKKTLinSys(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSys();

  /**
   * Updates the parts in KKT system that are dependent on the iterate.
   * It may trigger a refactorization for direct linear systems, or it may not do
   * anything, for example, LowRank KKT linear system
   */
  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess) = 0;

  /**
   * Forms the residual of the underlying linear system. It uses the factorization
   * computed by `update` to compute the "reduced-space" (i.e., compressed, condensed, etc.)
   * search directions by solving with the factors, then computes the "full-space" directions */
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;
  virtual bool compute_directions_w_IR(const hiopResidual* resid, hiopIterate* direction);

  virtual bool compute_directions_for_full_space(const hiopResidual* resid, hiopIterate* direction);

  virtual bool factorize_inertia_free() = 0;

  /* curvature test for inertia-free approach */
  virtual bool test_direction(const hiopIterate* dir, hiopMatrix* Hess) = 0;

  virtual void set_PD_perturb_calc(hiopPDPerturbation* p) { perturb_calc_ = p; }

  virtual void set_fact_acceptor(hiopFactAcceptor* p_fact_acceptor) { fact_acceptor_ = p_fact_acceptor; }

  inline void set_safe_mode(bool val) { safe_mode_ = val; }

  /// @brief Sets the log barrier parameter `mu`
  inline void set_logbar_mu(double mu) { mu_ = mu; }

  /**
   * Returns the absolute residual norm at the last KKT solve.
   *
   * The returned norm can be an only hint/approximation of the true residual norm in cases the last
   * solve is successful. If the KKT solve fails (i.e., one of the `compute_directions` methods fails)
   * the KKT class should return a good approximation of the norm of residual; if this is not feasible,
   * it is better to return an optimistic underestimate (lower than the true residual norm) so that the
   * IPM does not activate agressive regularization strategies unnecessarily.
   */
  virtual double get_resid_norm_abs() const { return 0.0; }

  /**
   * Returns the relative residual norm at the last KKT solve.
   *
   * The returned norm can be an only hint/approximation of the true residual norm in cases the last
   * solve is successful. If the KKT solve fails (i.e., one of the `compute_directions` methods fails)
   * the KKT class should return a good approximation of the norm of residual; if this is not feasible,
   * it is better to return an optimistic underestimate (lower than the true residual norm) so that the
   * IPM does not activate agressive regularization strategies unnecessarily.
   */
  virtual double get_resid_norm_rel() const { return 0.0; }

  /**
   * Compute the inf norm of residual for the KKT linear system.
   *
   * This is not currently used by the IPM algorithm since small-enough residual error
   * for the inner linear system, as reported by the linear solver, is indicative of
   * small KKT error. The method is called under HIOP_DEEPCHECKS to report residuals of
   * large inf-norm.
   */
  virtual double errorKKT(const hiopResidual* resid, const hiopIterate* sol);

  inline hiopPDPerturbation* get_perturb_calc() const { return perturb_calc_; }

protected:
  /**
   * @brief y=beta*y+alpha*H*x
   *
   * @pre Should not include log barrier diagonal terms
   * @pre Should not include IC perturbations
   *
   * A default implementation is below
   */
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector& x)
  {
    Hess_->timesVec(beta, y, alpha, x);
  }

protected:
  hiopNlpFormulation* nlp_;
  const hiopIterate* iter_;
  const hiopVector* grad_f_;
  const hiopMatrix *Jac_c_, *Jac_d_;
  hiopMatrix* Hess_;
  hiopPDPerturbation* perturb_calc_;
  hiopFactAcceptor* fact_acceptor_;
  bool perf_report_;
  bool safe_mode_;
  double mu_;

  /// Matrix operator performing mat-vec with given kkt linear system
  hiopMatVecKKTFullOpr* kkt_opr_;

  /// Preconditioner operator that solves with the given (usually compressed) KKT system
  hiopPrecondKKTOpr* prec_opr_;

  /// iterative refinement from BiCGStab solver
  hiopBiCGStabSolver* bicgIR_;

  friend class hiopMatVecKKTFullOpr;
  friend class hiopPrecondKKTOpr;

  // vectors for pd pertubations
  hiopVector* delta_wx_;
  hiopVector* delta_wd_;
  hiopVector* delta_cc_;
  hiopVector* delta_cd_;
};

class hiopKKTLinSysCurvCheck : public hiopKKTLinSys
{
public:
  hiopKKTLinSysCurvCheck(hiopNlpFormulation* nlp)
      : hiopKKTLinSys(nlp),
        linSys_{nullptr}
  {}

  virtual ~hiopKKTLinSysCurvCheck() { delete linSys_; }

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess) = 0;

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

  virtual bool factorize();

  virtual bool factorize_inertia_free();

  /* curvature test for inertia-free approach */
  virtual bool test_direction(const hiopIterate* dir, hiopMatrix* Hess) = 0;

  /**
   * @brief factorize the matrix and check curvature
   */
  virtual int factorizeWithCurvCheck();

  /**
   * @brief updates the iterate matrix, given regularizations 'delta_wx', 'delta_wd', 'delta_cc' and 'delta_cd'.
   */
  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

  hiopLinSolver* linSys_;
};

class hiopKKTLinSysCompressed : public hiopKKTLinSysCurvCheck
{
public:
  hiopKKTLinSysCompressed(hiopNlpFormulation* nlp)
      : hiopKKTLinSysCurvCheck(nlp),
        Dx_(nullptr),
        Dd_(nullptr),
        rx_tilde_(nullptr),
        x_wrk_(nullptr),
        d_wrk_(nullptr)
  {
    Dx_ = nlp->alloc_primal_vec();
    assert(Dx_ != nullptr);
    rx_tilde_ = Dx_->alloc_clone();
    Dd_ = nlp->alloc_dual_ineq_vec();
  }
  virtual ~hiopKKTLinSysCompressed()
  {
    delete Dx_;
    delete rx_tilde_;
    delete Dd_;
    if(x_wrk_) {
      delete x_wrk_;
    }
    if(d_wrk_) {
      delete d_wrk_;
    }
  }
  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess) = 0;

  virtual bool test_direction(const hiopIterate* dir, hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

protected:
  hiopVector* Dx_;
  hiopVector* Dd_;
  hiopVector* rx_tilde_;
  hiopVector* x_wrk_;
  hiopVector* d_wrk_;
};

/* Provides the functionality for reducing the KKT linear system to the
 * compressed linear below in dx, dyc, and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions.
 *
 * Relies on the pure virtual 'solveCompressed' to solve the compressed linear system
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 */
class hiopKKTLinSysCompressedXYcYd : public hiopKKTLinSysCompressed
{
public:
  hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedXYcYd();

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dyc,
                               hiopVector& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVector& rx,
                                       const hiopVector& ryc,
                                       const hiopVector& ryd,
                                       const hiopVector& dx,
                                       const hiopVector& dyc,
                                       const hiopVector& dyd);
#endif

protected:
  hiopVector* Dd_inv_;
  hiopVector* ryd_tilde_;
};

/* Provides the functionality for reducing the KKT linear system to the
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
class hiopKKTLinSysCompressedXDYcYd : public hiopKKTLinSysCompressed
{
public:
  hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedXDYcYd();

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& rd,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dd,
                               hiopVector& dyc,
                               hiopVector& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVector& rx,
                                       const hiopVector& rd,
                                       const hiopVector& ryc,
                                       const hiopVector& ryd,
                                       const hiopVector& dx,
                                       const hiopVector& dd,
                                       const hiopVector& dyc,
                                       const hiopVector& dyd);
#endif

protected:
  hiopVector* rd_tilde_;

#ifdef HIOP_DEEPCHECKS
  // y=beta*y+alpha*H*x
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector& x)
  {
    Hess_->timesVec(beta, y, alpha, x);
  }
#endif
};

/*
 * Solves hiopKKTLinSysFull by exploiting the sparse structure
 *
 * In general, the so-called XYcYd system has the form
 * [   H   Jc^T  Jd^T | 0 |  0   0  -I   I   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  Jc    0     0   | 0 |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    0     0   |-I |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0    -I   | 0 |  -I  I   0   0   |  0   0   0   0  ] [  dd]   [    rd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   |-I |  0   0   0   0   |  I   0   0   0  ] [ dvl]   [   rdl    ]
 * [  0     0     0   | I |  0   0   0   0   |  0   I   0   0  ] [ dvu]   [   rdu    ]
 * [ -I     0     0   | 0 |  0   0   0   0   |  0   0   I   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0   | 0 |  0   0   0   0   |  0   0   0   I  ] [ dzu]   [   rxu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   | 0 | Sl^d 0   0   0   | Vl   0   0   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0   | 0 |  0  Su^d 0   0   |  0  Vu   0   0  ] [dsdu]   [  rsvu    ]
 * [  0     0     0   | 0 |  0   0  Sl^x 0   |  0   0  Zl   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0   | 0 |  0   0   0  Su^x |  0   0   0  Zu  ] [dsxu]   [  rszu    ]
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *
 * TODO: use the following sys:
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
 */
class hiopKKTLinSysFull : public hiopKKTLinSysCurvCheck
{
public:
  hiopKKTLinSysFull(hiopNlpFormulation* nlp)
      : hiopKKTLinSysCurvCheck(nlp),
        x_wrk_{nullptr},
        d_wrk_{nullptr}
  {}

  virtual ~hiopKKTLinSysFull()
  {
    delete x_wrk_;
    delete d_wrk_;
  }

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess);

  virtual bool test_direction(const hiopIterate* dir, hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

  virtual bool solve(hiopVector& rx,
                     hiopVector& ryc,
                     hiopVector& ryd,
                     hiopVector& rd,
                     hiopVector& rdl,
                     hiopVector& rdu,
                     hiopVector& rxl,
                     hiopVector& rxu,
                     hiopVector& rsvl,
                     hiopVector& rsvu,
                     hiopVector& rszl,
                     hiopVector& rszu,
                     hiopVector& dx,
                     hiopVector& dyc,
                     hiopVector& dyd,
                     hiopVector& dd,
                     hiopVector& dvl,
                     hiopVector& dvu,
                     hiopVector& dzl,
                     hiopVector& dzu,
                     hiopVector& dsdl,
                     hiopVector& dsdu,
                     hiopVector& dsxl,
                     hiopVector& dsxu) = 0;

protected:
  hiopVector* x_wrk_;
  hiopVector* d_wrk_;
};

/**
 * @brief Provides the functionality for reducing the KKT linear system to the
 * normal equation system below in dyc and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * [ Jc  0 ] [ H + Dx   0 ]^{-1} [ Jc^T  Jd^T]  [dyc] = [   ryc_tilde    ]
 * [ Jd -I ] [   0     Dd ]      [  0     -I ]  [dyd]   [   ryd_tilde    ]
 *
 * [ ryc_tilde ] = [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1}  [ rx_tilde ] - [ ryc ]
 * [ ryd_tilde ]   [ Jd -I ] [   0           Dd+delta_wd ]       [ rd_tilde ]   [ ryd ]
 *
 * and then to compute the rest of the search directions from
 * [ H+Dx+delta_wx     0       ] [dx] = [ rx_tilde ] - [ Jc^T  Jd^T] [dyc]
 * [   0           Dd+delta_wd ] [dd]   [ rd_tilde ]   [  0     -I ] [dyd]
 *
 */
class hiopKKTLinSysNormalEquation : public hiopKKTLinSysCompressed
{
public:
  hiopKKTLinSysNormalEquation(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysNormalEquation();

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg) = 0;

  virtual bool solveCompressed(hiopVector& ryc_tilde, hiopVector& ryd_tilde, hiopVector& dyc, hiopVector& dyd) = 0;

  /**
   * @brief factorize the matrix and check curvature
   */
  virtual int factorizeWithCurvCheck() = 0;

protected:
  hiopVector* rd_tilde_;
  hiopVector* ryc_tilde_;
  hiopVector* ryd_tilde_;

  hiopVector* Hx_;  // [diag(H)+Dx+delta_wx]
  hiopVector* Hd_;  // [Dd+delta_wd ]

  hiopVector* x_wrk_;
  hiopVector* d_wrk_;
};

/**
 * operators for KKT mat-vec operations
 *
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
 */
class hiopMatVecKKTFullOpr : public hiopLinearOperator
{
public:
  hiopMatVecKKTFullOpr(hiopKKTLinSys* kkt, const hiopIterate* iter);

  virtual ~hiopMatVecKKTFullOpr()
  {
    delete resid_;
    delete dir_;
    delete dir_cv_;
    delete res_cv_;
  };

  /** y = KKT * x */
  virtual bool times_vec(hiopVector& y, const hiopVector& x);

  /** y = KKT' * x */
  virtual bool trans_times_vec(hiopVector& y, const hiopVector& x);

  /* need to reset the pointer to the current iter, since the outer loop keeps swtiching between curr_iter and trial_iter */
  inline void reset_curr_iter(const hiopIterate* iter) { iter_ = iter; }

private:
  hiopKKTLinSys* kkt_;
  const hiopIterate* iter_;
  hiopResidual* resid_;
  hiopIterate* dir_;

  hiopMatVecKKTFullOpr()
      : kkt_(nullptr),
        resid_(nullptr),
        dir_(nullptr)
  {
    assert(false && "this constructor should not be used");
  }

  /** @brief split a large vector to build a hiopIterate object.
   *  Note that the size of vector is equal to the size of full KKT.
   *  TODO: revisit this function after we implement compound vector
   */
  bool split_vec_to_build_it(const hiopVector& vec);

  /** @brief combine vectors from a hiopResidual object into a large vector.
   *  Note that the size of vector is equal to the size of full KKT.
   *  TODO: revisit this function after we implement compound vector
   */
  bool combine_res_to_build_vec(hiopVector& vec);

  hiopVectorCompoundPD* dir_cv_;
  hiopVectorCompoundPD* res_cv_;
};

/**
 * operators for KKT preconditioner
 */
class hiopPrecondKKTOpr : public hiopLinearOperator
{
public:
  hiopPrecondKKTOpr(hiopKKTLinSys* kkt, const hiopIterate* iter);

  virtual ~hiopPrecondKKTOpr()
  {
    delete resid_;
    delete dir_;
    delete dir_cv_;
    delete res_cv_;
  };

  /** y = inv(Preconditioner) * x = Preconditioner/x */
  virtual bool times_vec(hiopVector& y, const hiopVector& x);

  /** y = inv(Preconditioner)' * x = Preconditioner'/x */
  virtual bool trans_times_vec(hiopVector& y, const hiopVector& x);

protected:
  hiopKKTLinSys* kkt_;
  const hiopIterate* iter_;
  hiopResidual* resid_;
  hiopIterate* dir_;

  hiopPrecondKKTOpr()
      : kkt_(nullptr),
        resid_(nullptr),
        dir_(nullptr)
  {
    assert(false && "this constructor should not be used");
  }

  /** @brief split a large vector to build a hiopResidual object.
   *  Note that the size of vector is equal to the size of full KKT.
   *  TODO: revisit this function after we implement compound vector
   */
  virtual bool split_vec_to_build_res(const hiopVector& vec);

  /** @brief combine vectors from a hiopIterate object into a large vector.
   *  Note that the size of vector is equal to the size of full KKT.
   *  TODO: revisit this function after we implement compound vector
   */
  virtual bool combine_dir_to_build_vec(hiopVector& vec);

  hiopVectorCompoundPD* dir_cv_;
  hiopVectorCompoundPD* res_cv_;
};

};  // namespace hiop

#endif
