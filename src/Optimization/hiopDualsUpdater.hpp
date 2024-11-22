// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
 * @file hiopDualsUpdater.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_DUALSUPDATER
#define HIOP_DUALSUPDATER

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopMatrix.hpp"
#include "hiopLinSolver.hpp"

namespace hiop
{

class hiopDualsUpdater
{
public:
  hiopDualsUpdater(hiopNlpFormulation* nlp)
      : nlp_(nlp) {};
  virtual ~hiopDualsUpdater() {};

  /* The method is called after each iteration to update the duals. Implementations for different
   * multiplier updating strategies are provided by child classes
   * - linear (Newton) update in hiopDualsNewtonLinearUpdate
   * - lsq in hiopDualsLsqUpdate
   * The parameters are:
   * - iter: incumbent iterate that is going to be updated with iter_plus by the caller of this method.
   * - iter_plus: [in/out] on return the duals should be updated; primals are already updated, but
   * the function should not rely on this. If a particular implementation of this method requires
   * accessing primals, it should do so by working with 'iter'. In the algorithm class, iter_plus
   * corresponds to 'iter_trial'.
   * - f,c,d: fcn evals at  iter_plus
   * - grad_f, jac_c, jac_d: derivatives at iter_plus
   * - search_dir: search direction (already used to update primals, potentially to be used to
   * update duals (in linear update))
   * - alpha_primal: step taken for primals (also taken for eq. duals for the linear Newton duals update)
   * - alpha_dual: max step for the duals based on the fraction-to-the-boundary rule (not used
   * by lsq update)
   */
  virtual bool go(const hiopIterate& iter,
                  hiopIterate& iter_plus,
                  const double& f,
                  const hiopVector& c,
                  const hiopVector& d,
                  const hiopVector& grad_f,
                  const hiopMatrix& jac_c,
                  const hiopMatrix& jac_d,
                  const hiopIterate& search_dir,
                  const double& alpha_primal,
                  const double& alpha_dual,
                  const double& mu,
                  const double& kappa_sigma,
                  const double& infeas_nrm_trial) = 0;

protected:
  hiopNlpFormulation* nlp_;

protected:
  hiopDualsUpdater() {};

private:
  hiopDualsUpdater(const hiopDualsUpdater&) {};
  void operator=(const hiopDualsUpdater&) {};
};

class hiopDualsLsqUpdate : public hiopDualsUpdater
{
public:
  hiopDualsLsqUpdate(hiopNlpFormulation* nlp);
  virtual ~hiopDualsLsqUpdate();

  /** LSQ update of the constraints duals (yc and yd). Source file describe the math. */
  virtual bool go(const hiopIterate& iter,
                  hiopIterate& iter_plus,
                  const double& f,
                  const hiopVector& c,
                  const hiopVector& d,
                  const hiopVector& grad_f,
                  const hiopMatrix& jac_c,
                  const hiopMatrix& jac_d,
                  const hiopIterate& search_dir,
                  const double& alpha_primal,
                  const double& alpha_dual,
                  const double& mu,
                  const double& kappa_sigma,
                  const double& infeas_nrm_trial);

  /** LSQ update of the constraints duals (yc and yd). Source file describe the math. */
  virtual bool go(hiopIterate& it_ini, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d)
  {
    bool bret = init_for_duals_update(it_ini, grad_f, jac_c, jac_d);
    if(!bret) {
      // non-fatal
      nlp_->log->printf(hovScalars, "LSQ Duals update error in initialization.\n");
    }
    bret = do_lsq_update(it_ini, grad_f, jac_c, jac_d);
    return bret;
  }

  /** LSQ-based initialization of the  constraints duals (yc and yd). Source file describes the math. */
  virtual inline bool compute_initial_duals_eq(hiopIterate& it_ini,
                                               const hiopVector& grad_f,
                                               const hiopMatrix& jac_c,
                                               const hiopMatrix& jac_d)
  {
    bool bret = init_for_ini_duals_comp(it_ini, grad_f, jac_c, jac_d);
    if(!bret) {
      // non-fatal
      nlp_->log->printf(hovScalars, "Initial Duals error in initialization.\n");
      return false;
    }

    bret = do_lsq_update(it_ini, grad_f, jac_c, jac_d);

    double ycnrm = it_ini.get_yc()->infnorm();
    double ydnrm = it_ini.get_yd()->infnorm();
    double ynrm = (ycnrm > ydnrm) ? ycnrm : ydnrm;

    // do not use the LSQ duals if their norm is greater than 'duals_lsq_ini_max'; instead,
    double lsq_dual_init_max = nlp_->options->GetNumeric("duals_lsq_ini_max");
    if(ynrm > lsq_dual_init_max || !bret) {
      it_ini.get_yc()->setToZero();
      it_ini.get_yd()->setToZero();
      if(bret) {
        nlp_->log->printf(hovScalars,
                          "Will not use lsq dual initial point since its norm (%g) is larger than "
                          "the tolerance duals_lsq_ini_max=%g.\n",
                          ynrm,
                          lsq_dual_init_max);
      }
    }
    return bret;
  }

protected:
  /// Helper method doing the work for both `go` and `compute_initial_duals_eq`
  virtual bool do_lsq_update(hiopIterate& it,
                             const hiopVector& grad_f,
                             const hiopMatrix& jac_c,
                             const hiopMatrix& jac_d) = 0;

  /// Performs internal (re)initializations related to computations of initial duals that are needed by `do_lsq_update`
  virtual bool init_for_ini_duals_comp(hiopIterate& it,
                                       const hiopVector& grad_f,
                                       const hiopMatrix& jac_c,
                                       const hiopMatrix& jac_d)
  {
    return true;
  }

  /// Performs internal (re)initializations related to computations LSQ-based duals that are needed by `do_lsq_update`
  virtual bool init_for_duals_update(hiopIterate& it,
                                     const hiopVector& grad_f,
                                     const hiopMatrix& jac_c,
                                     const hiopMatrix& jac_d)
  {
    return true;
  }

protected:
  hiopVector *rhs_, *rhsc_, *rhsd_;
  hiopVector *vec_n_, *vec_mi_;

private:
  hiopDualsLsqUpdate() {};
  hiopDualsLsqUpdate(const hiopDualsLsqUpdate&) {};
  void operator=(const hiopDualsLsqUpdate&) {};
};

/** Given xk, zk_l, zk_u, vk_l, and vk_u (contained in 'iter'), this method solves an LSQ problem
 * corresponding to dual infeasibility equation
 *    min_{y_c,y_d} ||  \nabla f(xk) + J^T_c(xk) y_c + J_d^T(xk) y_d - zk_l+zk_u  ||^2
 *                  || - y_d - vk_l + vk_u                                        ||_2,
 *  which is
 *   min_{y_c, y_d} || [ J_c^T  J_d^T ] [ y_c ]  -  [ -\nabla f(xk) + zk_l-zk_u ]  ||^2
 *                  || [  0       I   ] [ y_d ]     [ - vk_l + vk_u             ]  ||_2
 * ******************************
 * NLPs with dense constraints
 * ******************************
 * For NLPs with dense constraints, the above LSQ problem is solved by solving the linear
 *  system in y_c and y_d:
 *   [ J_c J_c^T    J_c J_d^T     ] [ y_c ]  =  [ J_c   0 ] [ -\nabla f(xk) + zk_l-zk_u ]
 *   [ J_d J_c^T    J_d J_d^T + I ] [ y_d ]     [ J_d   I ] [ - vk_l + vk_u             ]
 * This linear system is small (of size m=m_E+m_I) (so it is replicated for all MPI ranks).
 *
 * The matrix of the above system is stored in the member variable M_ of this class and the
 *  right-hand side in 'rhs_'.
 *
 * **************
 * MDS NLPs
 * **************
 * For MDS NLPs, the linear system exploits the block structure of the Jacobians Jc and Jd.
 * Namely, since Jc = [Jxdc  Jxsc] and Jd = [Jxdd  Jxsd], the following
 * dense linear system is to be solved for y_c and y_d
 *
 *    [ Jxdc Jxdc^T + Jxsc Jxsc^T   Jxdc Jxdd^T + Jxsc Jxsd^T     ] [ y_c ] = same rhs as
 *    [ Jxdd Jxdc^T + Jxsd Jxsc^T   Jxdd Jxdd^T + Jxsd Jxsd^T + I ] [ y_d ]     above
 *
 * The above linear systems are solved as dense linear systems using Cholesky factorization
 * of LAPACK or MAGMA.
 *
 */
class hiopDualsLsqUpdateLinsysRedDense : public hiopDualsLsqUpdate
{
public:
  hiopDualsLsqUpdateLinsysRedDense(hiopNlpFormulation* nlp);
  virtual ~hiopDualsLsqUpdateLinsysRedDense();

private:
  virtual bool do_lsq_update(hiopIterate& it, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d);

protected:
  // not part of hiopDualsLsqUpdate but overridden in child classes

  /* Returns reference to the underlying system matrix, which is maintained / allocated differently
   * by child classes
   */
  virtual hiopMatrixDense* get_lsq_sysmatrix() = 0;

  /// Factorizes the LSQ matrix and returns true if successfull, otherwise returns false
  virtual bool factorize_mat() = 0;

  /* Performs triangular solves based on the factorize matrix and returns true if successfull,
   * otherwise returns false
   */
  virtual bool solve_with_factors(hiopVector& r) = 0;

private:
  hiopMatrix *mexme_, *mexmi_, *mixmi_, *mxm_;
#ifdef HIOP_DEEPCHECKS
  hiopMatrix* M_copy_;
  hiopVector* rhs_copy_;
  hiopMatrix* mixme_;
#endif
};

/** Provides functionality to solve the LSQ system as a symmetric system and is used to offload
 * computations to the device via MAGMA linear solver when this is possible (or required by the user)
 */
class hiopDualsLsqUpdateLinsysRedDenseSym : public hiopDualsLsqUpdateLinsysRedDense
{
public:
  hiopDualsLsqUpdateLinsysRedDenseSym(hiopNlpFormulation* nlp);

  virtual ~hiopDualsLsqUpdateLinsysRedDenseSym() { delete linsys_; }

protected:
  /// Returns reference to the underlying system matrix, which is maintained by the linear solver
  hiopMatrixDense* get_lsq_sysmatrix() { return &linsys_->sysMatrix(); }

  bool factorize_mat();
  bool solve_with_factors(hiopVector& r);

protected:
  hiopLinSolverSymDense* linsys_;
};

/** Provides functionality to solve the LSQ system as a symmetric positive definite system on the host
 */
class hiopDualsLsqUpdateLinsysRedDenseSymPD : public hiopDualsLsqUpdateLinsysRedDense
{
public:
  hiopDualsLsqUpdateLinsysRedDenseSymPD(hiopNlpFormulation* nlp)
      : hiopDualsLsqUpdateLinsysRedDense(nlp)
  {
    M_ = LinearAlgebraFactory::create_matrix_dense(nlp->options->GetString("mem_space"), nlp_->m(), nlp_->m());
  }

  virtual ~hiopDualsLsqUpdateLinsysRedDenseSymPD() { delete M_; }

protected:
  /// Returns reference to the underlying system matrix, which is maintained by the linear solver
  hiopMatrixDense* get_lsq_sysmatrix() { return M_; }

  bool factorize_mat();
  bool solve_with_factors(hiopVector& r);

protected:
  hiopMatrixDense* M_;
};

/**
 * @brief LSQ-based initialization for sparse linear algebra (NLPs with sparse Jac/Hes)
 *
 * With the same notation used above for hiopDualsLsqUpdateLinsysRedDense class,
 * for sparse NLPs, the corresponding LSQ problem is the following augmented
 * linear system:
 * [    I    0     Jc^T  Jd^T  ] [ dx]      [ \nabla f(xk) - zk_l + zk_u  ]
 * [    0    I     0     -I    ] [ dd]      [        -vk_l + vk_u         ]
 * [    Jc   0     0     0     ] [dyc] =  - [              0              ]
 * [    Jd   -I    0     0     ] [dyd]      [              0              ]
 *
 * The matrix of the above system is stored in the member variable M_ of this class and the
 * right-hand side in 'rhs'.
 */
class hiopDualsLsqUpdateLinsysAugSparse : public hiopDualsLsqUpdate
{
public:
  hiopDualsLsqUpdateLinsysAugSparse(hiopNlpFormulation* nlp);
  virtual ~hiopDualsLsqUpdateLinsysAugSparse();

private:
  virtual bool do_lsq_update(hiopIterate& iter, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d);

  /// Performs internal (re)initializations related to computations of initial duals that are needed by `do_lsq_update`
  virtual bool init_for_ini_duals_comp(hiopIterate& iter,
                                       const hiopVector& grad_f,
                                       const hiopMatrix& jac_c,
                                       const hiopMatrix& jac_d)
  {
    return instantiate_linear_solver("duals_init_linear_solver_sparse", iter, grad_f, jac_c, jac_d);
  }

  /// Performs internal (re)initializations related to computations LSQ-based duals that are needed by `do_lsq_update`
  virtual bool init_for_duals_update(hiopIterate& iter,
                                     const hiopVector& grad_f,
                                     const hiopMatrix& jac_c,
                                     const hiopMatrix& jac_d)
  {
    return instantiate_linear_solver("linear_solver_sparse", iter, grad_f, jac_c, jac_d);
  }

  /// Internal helper that instantiates the linear solver based on user options
  bool instantiate_linear_solver(const char* lin_solver_option_name,
                                 hiopIterate& iter,
                                 const hiopVector& grad_f,
                                 const hiopMatrix& jac_c,
                                 const hiopMatrix& jac_d);

private:
  hiopLinSolver* lin_sys_;
};

/**
 * Performs Newton update for the duals, which is a simple linear update along the dual Newton direction
 * with a given dual step.
 */
class hiopDualsNewtonLinearUpdate : public hiopDualsUpdater
{
public:
  hiopDualsNewtonLinearUpdate(hiopNlpFormulation* nlp)
      : hiopDualsUpdater(nlp) {};
  virtual ~hiopDualsNewtonLinearUpdate() {};

  /* Linear update of step length alpha_primal in eq. duals yc and yd and step length
   * alpha_dual in the (signed or bounds) duals zl, zu, vl, and vu.
   * This is standard in (full) Newton IPMs that is very low cost.
   */
  virtual bool go(const hiopIterate& iter,
                  hiopIterate& iter_plus,
                  const double& f,
                  const hiopVector& c,
                  const hiopVector& d,
                  const hiopVector& grad_f,
                  const hiopMatrix& jac_c,
                  const hiopMatrix& jac_d,
                  const hiopIterate& search_dir,
                  const double& alpha_primal,
                  const double& alpha_dual,
                  const double& mu,
                  const double& kappa_sigma,
                  const double& infeas_nrm_trial)
  {
    if(!iter_plus.takeStep_duals(iter, search_dir, alpha_primal, alpha_dual)) {
      nlp_->log->printf(hovError, "dual Newton updater: error in standard update of the duals");
      return false;
    }
    return iter_plus.adjustDuals_primalLogHessian(mu, kappa_sigma);
  }

private:
  hiopDualsNewtonLinearUpdate() {};
  hiopDualsNewtonLinearUpdate(const hiopDualsNewtonLinearUpdate&) {};
  void operator=(const hiopDualsNewtonLinearUpdate&) {};
};

}  // namespace hiop
#endif
