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
 * @file hiopAlgFilterIPM.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>,  LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>,  LLNL
 *
 */

#ifndef HIOP_ALGFilterIPM
#define HIOP_ALGFilterIPM

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopFilter.hpp"
#include "HessianDiagPlusRowRank.hpp"
#include "hiopKKTLinSys.hpp"
#include "hiopLogBarProblem.hpp"
#include "hiopDualsUpdater.hpp"
#include "hiopPDPerturbation.hpp"
#include "hiopFactAcceptor.hpp"

#ifdef HIOP_USE_AXOM
namespace axom
{
namespace sidre
{
class Group;  // forward declaration
}
}  // namespace axom
#endif

#include "hiopTimer.hpp"

namespace hiop
{

class hiopAlgFilterIPMBase
{
public:
  hiopAlgFilterIPMBase(hiopNlpFormulation* nlp_, const bool within_FR = false);
  virtual ~hiopAlgFilterIPMBase();

  /** numerical optimization */
  virtual hiopSolveStatus run() = 0;

  /** computes primal-dual point and returns the evaluation of the problem at this point */
  virtual int startingProcedure(hiopIterate& it_ini,
                                double& f,
                                hiopVector& c_,
                                hiopVector& d_,
                                hiopVector& grad_,
                                hiopMatrix& Jac_c,
                                hiopMatrix& Jac_d);
  /* returns the objective value; valid only after 'run' method has been called */
  double getObjective() const;
  /* returns the primal vector x; valid only after 'run' method has been called */
  void getSolution(double* x) const;
  /* returns dual solutions; valid only after 'run' method has been called */
  void getDualSolutions(double* zl, double* zu, double* lambda);
  /* returns the status of the solver */
  inline hiopSolveStatus getSolveStatus() const { return solver_status_; }
  /* returns the number of iterations */
  int getNumIterations() const;
  /* returns the logbar object */
  hiopLogBarProblem* get_logbar() { return logbar; }

  inline hiopNlpFormulation* get_nlp() const { return nlp; }
  inline hiopIterate* get_it_curr() const { return it_curr; }
  inline hiopIterate* get_it_trial() const { return it_trial; }
  inline hiopIterate* get_it_trial_nonconst() { return it_trial; }
  inline hiopIterate* get_dir() const { return dir; }
  inline double get_mu() const { return _mu; }
  inline hiopMatrix* get_Jac_c() const { return _Jac_c; }
  inline hiopMatrix* get_Jac_d() const { return _Jac_d; }
  inline hiopMatrix* get_Hess_Lagr() const { return _Hess_Lagr; }
  inline hiopVector* get_c() const { return _c; }
  inline hiopVector* get_d() const { return _d; }
  inline hiopResidual* get_resid() const { return resid; }
  inline bool filter_contains(const double theta, const double logbar_obj) const
  {
    return filter.contains(theta, logbar_obj);
  }

  /// Setter for the primal steplength.
  inline void set_alpha_primal(const double alpha_primal) { _alpha_primal = alpha_primal; }

protected:
  bool evalNlp(hiopIterate& iter,
               double& f,
               hiopVector& c_,
               hiopVector& d_,
               hiopVector& grad_,
               hiopMatrix& Jac_c,
               hiopMatrix& Jac_d,
               hiopMatrix& Hess_L);
  bool evalNlp_funcOnly(hiopIterate& iter, double& f, hiopVector& c_, hiopVector& d_);
  bool evalNlp_derivOnly(hiopIterate& iter, hiopVector& gradf_, hiopMatrix& Jac_c, hiopMatrix& Jac_d, hiopMatrix& Hess_L);

  /* Evaluates all the functions and derivatives, excepting the Hessian, which is supposed
   * to be evaluated at a later time.
   */
  bool evalNlp_noHess(hiopIterate& iter,
                      double& f,
                      hiopVector& c_,
                      hiopVector& d_,
                      hiopVector& grad_,
                      hiopMatrix& Jac_c,
                      hiopMatrix& Jac_d);
  /* Evaluates the Hessian
   *
   * Assumes that @evalNlp_noHess has just been called, so the user provided Hessian callback
   * is called with 'new_x' set to false.
   */
  bool evalNlp_HessOnly(hiopIterate& iter, hiopMatrix& Hess_L);

  /** Internal helper for NLP error/residuals computation.
   * TODO: add support for the 'true' infeasibility measure and propagate this downstream in
   * i.  the iteration output
   * ii. the 'theta' part of the filter's pairs 'theta' entries.
   * Instead, we currently use the inf-norm of d-d(x)=0, d-sdl=dl, d-sdu+du=0, c(x)-c=0
   *
   * The 'true' infeasibility (also used by Ipopt) would be the max of the inf norm of the
   * violation of d_l <= d(x) <= d_u and the inf norm of the residual of c(x)-c=0.
   */
  virtual bool evalNlpAndLogErrors(const hiopIterate& it,
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
                                   double& cons_violation);

  virtual double thetaLogBarrier(const hiopIterate& it, const hiopResidual& resid, const double& mu);

  /**
   * Reduces log barrier parameters `mu` and `tau`  and returns true if it was possible to reduce them. The
   * parameter `mu` may reach its min value and may not be reduced (same for `tau`), in which case the
   * method returns false.
   */
  bool update_log_barrier_params(hiopIterate& it,
                                 const double& mu_curr,
                                 const double& tau_curr,
                                 const bool& elastic_mode_on,
                                 double& mu_new,
                                 double& tau_new);

protected:
  // second order correction
  virtual int apply_second_order_correction(hiopKKTLinSys* kkt,
                                            const double theta_curr,
                                            const double theta_trial0,
                                            bool& grad_phi_dx_computed,
                                            double& grad_phi_dx,
                                            int& num_adjusted_slacks);

  // check if all the line search conditions are accepted or not
  virtual int accept_line_search_conditions(const double theta_curr,
                                            const double theta_trial,
                                            const double alpha_primal,
                                            bool& grad_phi_dx_computed,
                                            double& grad_phi_dx);
  /// @brief Step-length `alpha_pr` may be reduced when option 'moving_lim_abs' or 'moving_lim_rel' is active.
  bool ensure_moving_lims(const hiopIterate& it, const hiopIterate& dir, double& alpha_pr);

public:
  /// @brief do feasibility restoration
  virtual bool apply_feasibility_restoration(hiopKKTLinSys* kkt);
  virtual bool solve_soft_feasibility_restoration(hiopKKTLinSys* kkt);
  virtual bool solve_feasibility_restoration(hiopKKTLinSys* kkt, hiopNlpFormulation& nlpFR);
  virtual bool reset_var_from_fr_sol(hiopKKTLinSys* kkt, bool reset_dual = false);

  virtual void outputIteration(int lsStatus, int lsNum, int use_soc = 0, int use_fr = 0) = 0;

  // returns whether the algorithm should stop and set an appropriate solve status
  bool checkTermination(const double& _err_nlp, const int& iter_num, hiopSolveStatus& status);
  void displayTerminationMsg();

  void resetSolverStatus();
  virtual void reInitializeNlpObjects();
  virtual void reload_options();

  /// @brief Decides and creates regularization objects based on user options and NLP formulation.
  virtual hiopFactAcceptor* decideAndCreateFactAcceptor(hiopPDPerturbation* p, hiopNlpFormulation* nlp, hiopKKTLinSys* kkt);

  virtual bool compute_search_direction(hiopKKTLinSys* kkt,
                                        bool& linsol_safe_mode_on,
                                        const bool linsol_forcequick,
                                        const int iter_num);

  virtual bool compute_search_direction_inertia_free(hiopKKTLinSys* kkt,
                                                     bool& linsol_safe_mode_on,
                                                     const bool linsol_forcequick,
                                                     const int iter_num);

protected:
  /* Helper method containing all the allocations done by the base algorithm class.
   *
   * @note: Should not be virtual nor be overridden since it is called in the constructor.
   */
  void alloc_alg_objects();

  /// Helper method containing all the deallocations done by the base algorithm class. Avoid overidding it.
  void dealloc_alg_objects();

protected:
  hiopNlpFormulation* nlp;
  hiopFilter filter;

  hiopLogBarProblem* logbar;

  /* Iterate, search directions (managed by this (algorithm) class) */
  hiopIterate* it_curr;
  hiopIterate* it_trial;
  hiopIterate* dir;
  hiopIterate* soc_dir;

  hiopResidual *resid, *resid_trial;

  /// Iteration number maintained internally by the algorithm and reset at each solve/run
  int iter_num_;
  /// Total iteration number over multiple solves/restarts using checkpoints.
  int iter_num_total_;

  double _err_nlp_optim, _err_nlp_feas, _err_nlp_complem;     // not scaled by sd, sc, and sc
  double _err_nlp_optim0, _err_nlp_feas0, _err_nlp_complem0;  // initial errors, not scaled by sd, sc, and sc
  double _err_log_optim, _err_log_feas, _err_log_complem;     // not scaled by sd, sc, and sc
  double _err_nlp, _err_log;                                  // max of the above (scaled)
  double _err_cons_violation;  // constraint violation (Note: this is slightly different from _err_nlp_feas)
  double onenorm_pr_curr_;     // one norm of the constraint infeasibility

  // class for updating the duals multipliers
  hiopDualsUpdater* dualsUpdate_;

  /* Log-barrier problem data
   *  The algorithm manages these and updates them by calling the
   *  problem formulation and then adding the contribution from the
   *  log-barrier term(s). The data that is not iterate dependent,
   *  such as lower or upper bounds, is in the NlpFormulation
   */
  double _f_nlp, _f_log, _f_nlp_trial, _f_log_trial;
  hiopVector *_c, *_d, *_c_trial, *_d_trial;
  hiopVector *c_soc, *d_soc;
  hiopVector *_grad_f, *_grad_f_trial;  // gradient of the log-barrier objective function
  hiopMatrix *_Jac_c, *_Jac_c_trial;    // Jacobian of c(x), the equality part
  hiopMatrix *_Jac_d, *_Jac_d_trial;    // Jacobian of d(x), the inequality part
  hiopMatrix* _Hess_Lagr;

  /** Algorithms's working quantities */
  double _mu, _tau, _alpha_primal, _alpha_dual;
  // initialized to 1e4*max{1,\theta(x_0)} and used in the filter as an upper acceptability
  // limit for infeasibility
  double theta_max;
  // 1e-4*max{1,\theta(x_0)} used in the switching condition during the line search
  double theta_min;
  double theta_max_fact_;
  double theta_min_fact_;

  /*** Algorithm's parameters ***/
  double mu0;             // intial mu
  double kappa_mu;        // linear decrease factor in mu
  double theta_mu;        // exponent for a Mehtrotra-style decrease of mu
  double eps_tol;         // abs tolerance for the NLP error
  double eps_rtol;        // rel tolerance for the NLP error
  double dual_tol_;       // abs tolerance for the dual infeasibility
  double cons_tol_;       // abs tolerance for the constraint violation
  double comp_tol_;       // abs tolerance for the complementary conditions
  double tau_min;         // min value for the fraction-to-the-boundary parameter: tau_k=max{tau_min,1-\mu_k}
  double kappa_eps;       // tolerance for the barrier problem, relative to mu: error<=kappa_eps*mu
  double kappa1, kappa2;  // params for default starting point
  double p_smax;          // threshold for the magnitude of the multipliers used in the error estimation
  double gamma_theta,     // sufficient progress parameters for the feasibility violation
      gamma_phi;          // and log barrier objective
  double s_theta,         // parameters in the switch condition of the linearsearch (eq 19)
      s_phi, delta;
  double eta_phi;      // parameter in the Armijo rule
  double kappa_Sigma;  // parameter in resetting the duals to guarantee closedness of the
                       // primal-dual logbar Hessian to the primal logbar Hessian
  int duals_update_type;  // type of the update for dual multipliers: 0 LSQ (default, recommended
                          // for quasi-Newton); 1 Newton
  int max_n_it;
  int dualsInitializ;    // type of initialization for the duals of constraints: 0 LSQ (default), 1 set to zero
  int accep_n_it;        // after how many iterations with acceptable tolerance should the alg. stop
  double eps_tol_accep;  // acceptable tolerance

  // internal flags related to the state of the solver
  hiopSolveStatus solver_status_;
  int n_accep_iters_;
  bool trial_is_rejected_by_filter;

  /* Flag for timing and timing breakdown report for the KKT solve */
  bool perf_report_kkt_;

  /* Flag to tell if this is a FR problem */
  bool within_FR_;

  hiopPDPerturbation* pd_perturb_;
  hiopFactAcceptor* fact_acceptor_;
};

class hiopAlgFilterIPMQuasiNewton : public hiopAlgFilterIPMBase
{
public:
  hiopAlgFilterIPMQuasiNewton(hiopNlpDenseConstraints* nlp, const bool within_FR = false);
  virtual ~hiopAlgFilterIPMQuasiNewton();

  virtual hiopSolveStatus run();

  // note that checkpointing is only available with a axom-enabled build
#ifdef HIOP_USE_AXOM
  /**
   * @brief Save state of HiOp algorithm to a sidre::Group as a checkpoint.
   *
   * @param group a reference to the group where state will be saved to
   *
   * @exception std::runtime indicates the group contains a view whose size does not match
   * the size of the corresponding HiOp algorithm state variable of parameter.
   *
   * @details
   * Each state variable of each parameter of HiOp algorithm will be saved in a named
   * view within the group. A new view will be created within the group if it does not
   * already exist. If it exists, the view must have same number of elements as the
   * as the size of the corresponding state variable. This means that this method will
   * throw an exception if an existing group is reused to save a problem that changed
   * sizes since the group was created.
   */
  virtual void save_state_to_sidre_group(::axom::sidre::Group& group);

  /**
   * @brief Load state of HiOp algorithm from a sidre::Group checkpoint.
   *
   * @param group a pointer to group containing the a prevously saved HiOp algorithm state.
   *
   * @exception std::runtime indicates the group does not contain a view expected by this
   * method or the view's number of elements mismatches the size of the corresponding HiOp
   * state. The latter can occur if the file was saved with a different number of MPI ranks.
   *
   * @details
   * Copies views from the sidre::Group passed as argument to HiOp algorithm's state variables
   * and parameters. The group should be created by first calling save_state_to_sidre_group
   * for a problem/NLP of the same sizes as the problem for which this method is called.
   * The method expects views within the group with certain names. If one such view is not
   * found or has a number of elements different than the size of the corresponding HiOp state,
   * then a std::runtime_error exception is thrown. The latter can occur when the loading
   * occurs for a instance of HiOp that is not ran on the same number of MPI ranks used to
   * save the file.
   */
  virtual void load_state_from_sidre_group(const ::axom::sidre::Group& group);

  /**
   * @brief Save the state of the algorithm to the file for checkpointing.
   *
   * @param path the name of the file
   * @return true if successful, false otherwise
   *
   * @details
   * Internally, HiOp uses axom::sidre::DataStore and sidre's scalable IO. A detailed
   * error description is sent to the log if an error or exception is caught.
   */
  bool save_state_to_file(const ::std::string& path) noexcept;

  /**
   * @brief Load the state of the algorithm from checkpoint file.
   *
   * @param path the name of the file to load from
   * @return true if successful, false otherwise
   *
   * @details
   * The file should contains a axom::sidre::DataStore that was previously saved using
   * save_state_to_file(). A detailed error description is sent to the log if an error
   * or exception is caught.
   */
  bool load_state_from_file(const ::std::string& path) noexcept;
#endif  // HIOP_USE_AXOM
private:
  virtual void outputIteration(int lsStatus, int lsNum, int use_soc = 0, int use_fr = 0);

#ifdef HIOP_USE_AXOM
  ///@brief The options-based logic for saving checkpoint and the call to save_state().
  void checkpointing_stuff();
#endif  // HIOP_USE_AXOM

private:
  hiopNlpDenseConstraints* nlpdc;
  ///@brief Indicates whether load checkpoint API was called previous to run method.
  bool load_state_api_called_;

private:
  hiopAlgFilterIPMQuasiNewton()
      : hiopAlgFilterIPMBase(NULL) {};
  hiopAlgFilterIPMQuasiNewton(const hiopAlgFilterIPMQuasiNewton&)
      : hiopAlgFilterIPMBase(NULL) {};
  hiopAlgFilterIPMQuasiNewton& operator=(const hiopAlgFilterIPMQuasiNewton&) { return *this; };
};
// for backward compatibility we make 'hiopAlgFilterIPM' name available
typedef hiopAlgFilterIPMQuasiNewton hiopAlgFilterIPM;

class hiopAlgFilterIPMNewton : public hiopAlgFilterIPMBase
{
public:
  hiopAlgFilterIPMNewton(hiopNlpFormulation* nlp, const bool within_FR = false);
  virtual ~hiopAlgFilterIPMNewton();

  virtual hiopSolveStatus run();

protected:
  virtual void outputIteration(int lsStatus, int lsNum, int use_soc = 0, int use_fr = 0);

  /// @brief Decides and creates the KKT linear system based on user options and NLP formulation.
  virtual hiopKKTLinSys* decideAndCreateLinearSystem(hiopNlpFormulation* nlp);

  /**
   * Switch to the safer (more stable) KKT formulation and linear solver.
   *
   * This is currently done only for `hiopNlpSparseIneq` NLP formulation. In this case
   * `hiopKKTLinSysCondensedSparse` is the quick KKT formulation and `hiopKKTLinSysCompressedSparseXDYcYd`
   * is the safe KKT formulation. For other combinations of NLP and KKT formulations the method
   * returns the KKT passed as argument.
   */
  virtual hiopKKTLinSys* switch_to_safer_KKT(hiopKKTLinSys* kkt_curr,
                                             const double& mu,
                                             const int& iter_num,
                                             bool& linsol_safe_mode_on,
                                             const int& linsol_safe_mode_max_iters,
                                             int& linsol_safe_mode_last_iter_switched_on,
                                             double& theta_mu,
                                             double& kappa_mu,
                                             bool& switched);

  /**
   * Switch to the quick KKT formulation and linear solver is switching conditions are met.
   */
  virtual hiopKKTLinSys* switch_to_fast_KKT(hiopKKTLinSys* kkt_curr,
                                            const double& mu,
                                            const int& iter_num,
                                            bool& linsol_safe_mode_on,
                                            int& linsol_safe_mode_max_iters,
                                            int& linsol_safe_mode_last_iter_switched_on,
                                            double& theta_mu,
                                            double& kappa_mu,
                                            bool& switched);

  /// Overridden method from base class that does some preprocessing specific to Newton solver
  void reload_options();

private:
  hiopAlgFilterIPMNewton()
      : hiopAlgFilterIPMBase(NULL) {};
  hiopAlgFilterIPMNewton(const hiopAlgFilterIPMNewton&)
      : hiopAlgFilterIPMBase(NULL) {};
  hiopAlgFilterIPMNewton& operator=(const hiopAlgFilterIPMNewton&) { return *this; };
};

}  // namespace hiop
#endif
