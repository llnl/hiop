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
 * @file hiopInterface.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#ifndef HIOP_INTERFACE_BASE
#define HIOP_INTERFACE_BASE

#include "hiop_defs.hpp"
#include "hiopMPI.hpp"

namespace hiop
{
/** Solver status codes. */
enum hiopSolveStatus
{
  //(partial) success
  Solve_Success = 0,
  Solve_Success_RelTol = 1,
  Solve_Acceptable_Level = 2,
  Infeasible_Problem = 5,
  Iterates_Diverging = 6,
  Feasible_Not_Optimal = 7,
  // solver stopped based on user-defined criteria that are not related to optimality
  Max_Iter_Exceeded = 10,
  Max_CpuTime_Exceeded = 11,
  User_Stopped = 12,

  // NLP algorithm/solver reports issues in solving the problem and stops without being certain
  // that is solved the problem to optimality or that the problem is infeasible.
  // Feasible_Point_Found,
  NlpAlgorithm_failure = -1,
  Diverging_Iterates = -2,
  Search_Dir_Too_Small = -3,
  Steplength_Too_Small = -4,
  Err_Step_Computation = -5,
  // errors related to user-provided data (e.g., inconsistent problem specification, 'nans' in the
  // function/sensitivity evaluations, invalid options)
  Invalid_Problem_Definition = -11,
  Invalid_Parallelization = -12,
  Invalid_UserOption = -13,
  Invalid_Number = -14,
  Error_In_User_Function = -15,
  Error_In_FR = -16,

  // ungraceful errors and returns
  Exception_Unrecoverable = -100,
  Memory_Alloc_Problem = -101,
  SolverInternal_Error = -199,

  // unknown NLP solver errors or return codes
  UnknownNLPSolveStatus = -1000,
  SolveInitializationError = -1001,

  // intermediary statuses for the solver
  NlpSolve_IncompleteInit = -10001,
  NlpSolve_SolveNotCalled = -10002,
  NlpSolve_Pending = -10003
};

/** Base class for the solver's interface that has no assumptions how the
 *  matrices are stored. The vectors are dense and distributed row-wise.
 *  The data distribution is decided by the calling code (that implements
 *  this interface) and specified to the optimization via 'get_vecdistrib_info'
 *
 *  Three possible implementations are for sparse NLPs (hiopInterfaceSparse),
 *  mixed dense-sparse NLPs (hiopInterfaceMDS), and NLPs with small
 *  number of global constraints (hiopInterfaceDenseConstraints).
 *
 * @note Please take notice of the following notes regarding the implementation of
 * hiop::hiopInterfaceMDS on the device. All pointers marked as "managed by Umpire"
 * are allocated by HiOp using the Umpire's API. They all are addressed in the
 * same memory space; however, the memory space can be host (typically CPU),
 * device (typically GPU), or unified memory (um) spaces as per Umpire
 * specification. The selection of the memory space is done via the option
 * "mem_space" of HiOp. It is the responsibility of the implementers of the
 * HiOp's interfaces to work with the "managed by Umpire" pointers in the same
 * memory space as the one specified by the "mem_space" option.
 *
 * @note The above note does not currently apply to the NLP interfaces
 * hiop::hiopInterfaceDenseConstraints and hiop::hiopInterfaceSparse) and the pointers
 * marked as "managed by Umpire" are in the host/CPU memory space (subject to change
 * in future versions of HiOp).
 */
class hiopInterfaceBase
{
public:
  // Types indicating linearity or nonlinearity.
  enum NonlinearityType
  {
    hiopLinear = 0,
    hiopQuadratic,
    hiopNonlinear
  };

public:
  hiopInterfaceBase() {};
  virtual ~hiopInterfaceBase() {};

  /** Specifies the problem dimensions.
   *
   * @param n global number of variables
   * @param m number of constraints
   */
  virtual bool get_prob_sizes(size_type& n, size_type& m) = 0;

  /** Specifies the type of optimization problem
   * @param[out] type indicating whether the optimization problem is
   *                  linearily, quadratically, or general nonlinearily.
   * TODO: need to `deepcheck` is this return value matches the returned type array from
   * `get_vars_info` and `get_cons_info`
   */
  virtual bool get_prob_info(NonlinearityType& type)
  {
    type = hiopInterfaceBase::hiopNonlinear;
    return true;
  }

  /** Specifies bounds on the variables.
   *
   * @param[in] n global number of constraints
   * @param[out] xlow array of lower bounds. A value of -1e20 or less means no lower
   *                  bound is present (managed by Umpire)
   * @param[out] xupp array of upper bounds. A value of 1e20 or more means no upper
   *                  bound is present  (managed by Umpire)
   * @param[out] type array of indicating whether the variables enters the objective
   *                  linearily, quadratically, or general nonlinearily. Momentarily
   *                  all bounds should be marked as nonlinear (allocated on host).
   */
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type) = 0;

  /** Specififes the bounds on the constraints.
   *
   * @param[in] m number of constraints
   * @param[out] clow array of lower bounds for constraints. A value of -1e20 or less means no lower
   *                  bound is present (managed by Umpire)
   * @param[out] cupp array of upper bounds for constraints. A value of 1e20 or more means no upper
   *                  bound is present (managed by Umpire)
   * @param[out] type array of indicating whether the constraint is linear, quadratic, or general
   *                  nonlinear. Momentarily all bounds should be marked as nonlinear (allocated on host).
   */
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type) = 0;

  /** Method the evaluation of the objective function.
   *
   * @param[in] n global size of the problem
   * @param[in] x array with the local entries of the primal variable (managed by Umpire)
   * @param[in] new_x whether x has been changed from the previous calls to other evaluation methods
   *                  (gradient, constraints, Jacobian, and Hessian).
   * @param[out] obj_value the value of the objective function at @p x
   *
   * @note When MPI is enabled, each rank returns the objective value in @p obj_value. @p x points to
   * the local entries and the function is responsible for knowing the local buffer size.
   */
  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value) = 0;

  /** Method for the evaluation of the gradient of objective.
   *
   * @param[in] n global size of the problem
   * @param[in] x array with the local entries of the primal variable  (managed by Umpire)
   * @param[in] new_x whether x has been changed from the previous calls to other evaluation methods
   * (                function, constraints, Jacobian, and Hessian)
   * @param[out] gradf the entries of the gradient of the objective function at @p x, local to the
   * MPI rank (managed by Umpire)
   *
   *  @note When MPI is enabled, each rank should access only the local buffers @p x and @p gradf.
   */
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf) = 0;

  /** Evaluates a subset of the constraints @p cons(@p x). The subset is of size
   *  @p num_cons and is described by indexes in the @p idx_cons array. The method will be called at each
   *  iteration separately for the equality constraints subset and for the inequality constraints subset.
   *  This is done for performance considerations, to avoid auxiliary/temporary storage and copying.
   *
   * @param[in] n the global number of variables
   * @param[in] m the number of constraints
   * @param[in] num_cons the number constraints/size of subset to be evaluated
   * @param[in] idx_cons: indexes in {1,2,...,m} of the constraints to be evaluated  (managed by Umpire)
   * @param[in] x the point where the constraints need to be evaluated  (managed by Umpire)
   * @param[in] new_x whether x has been changed from the previous call to f, grad_f, or Jac
   * @param[out] cons array of size num_cons containing the value of the  constraints indicated by
   *                    @p idx_cons  (managed by Umpire)
   *
   *  @note When MPI is enabled, every rank populates @p cons since the constraints are not distributed.
   */
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons) = 0;

  /** Evaluates the constraints body @p cons(@p x), both equalities and inequalities, in one call.
   *
   * @param[in] n the global number of variables
   * @param[in] m the number of constraints
   * @param[in] x the point where the constraints need to be evaluated  (managed by Umpire)
   * @param[in] new_x whether x has been changed from the previous call to f, grad_f, or Jac
   * @param[out] cons array of size num_cons containing the value of the  constraints indicated by
   *                     @p idx_cons  (managed by Umpire)
   *
   * HiOp will first call the other hiopInterfaceBase::eval_cons() twice. If the implementer/user wants the
   * functionality  of this "one-call" overload, he should return false from the other
   * hiopInterfaceBase::eval_cons() (during both calls).
   *
   *  @note When MPI is enabled, every rank populates @p cons since the constraints are not distributed.
   */
  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons) { return false; }

  /** Passes the communicator, defaults to MPI_COMM_WORLD (dummy for non-MPI builds)  */
  virtual bool get_MPI_comm(MPI_Comm& comm_out)
  {
    comm_out = MPI_COMM_WORLD;
    return true;
  }

  /**
   * Method for column partitioning specification for distributed memory vectors. Process P owns
   * cols[P], cols[P]+1, ..., cols[P+1]-1, P={0,1,...,NumRanks}.
   *
   * Example: for a vector x of @p global_n=6 elements on 3 ranks, the column partitioning is
   * @p cols=[0,2,4,6].
   *
   * The caller manages memory associated with @p cols, which is an array of size NumRanks+1
   */
  virtual bool get_vecdistrib_info(size_type global_n, index_type* cols)
  {
    return false;  // defaults to serial
  }

  /**
   * Method provides a primal or starting point. This point is subject to internal adjustments.
   *
   * @note Avoid using this method since it will be removed in a future release and replaced with
   * the same-name method below.
   *
   * The method returns true (and populates @p x0) or returns false, in which case HiOp will
   * internally set @p x0 to all zero (still subject to internal adjustements).
   *
   * By default, HiOp first calls the overloaded primal-dual starting point specification
   * (overloaded) method get_starting_point() (see below). If the above returns false, HiOp will then call
   * this method.
   *
   * @param[in] n the global number of variables
   * @param[out] x0 the user-defined initial values for the primal variablers (managed by Umpire)
   *
   */
  virtual bool get_starting_point(const size_type& n, double* x0) { return false; }

  /**
   * Method provides a primal or a primal-dual starting point. This point is subject
   * to internal adjustments in HiOp.
   *
   * If the user (implementer of this method) has good estimates only of the primal variables,
   * the method should populate @p x0 with these values and return true. The @p duals_avail
   * should be set to false; internally, HiOp will not access @p z_bndL0, @p z_bndU0, and
   * @p lambda0 in this case.
   *
   * If the user (implementer of this method) has good estimates of the duals of bound constraints
   * and of inequality and equality constraints, @p duals_avail boolean argument should
   * be set to true and the respective duals should be provided (in @p z_bndL0 and @p z_bndU0 and
   * @p lambda0, respectively). In this case, the user should also set @p x0 to his/her estimate
   * of primal variables and return true.
   *
   * If user does not have high-quality (primal or primal-dual) starting points, the method should
   * return false (see note below).
   *
   * @note When this method returns false, HiOp will call the overload
   * get_starting_point() for only primal variables (see the above function). This behaviour is for backward compatibility
   * and will be removed in a future release.
   *
   * @param[in] n the global number of variables
   * @param[in] m the number of constraints
   * @param[out] x0 the user-defined initial values for the primal variablers (managed by Umpire)
   * @param[out] duals_avail a boolean argument which indicates whether the initial values of duals are given by the user
   * @param[out] z_bndL0 the user-defined initial values for the duals of the variable lower bounds (managed by Umpire)
   * @param[out] z_bndU0 the user-defined initial values for the duals of the variable upper bounds (managed by Umpire)
   * @param[out] lambda0 the user-defined initial values for the duals of the constraints (managed by Umpire)
   * @param[out] slacks_avail a boolean argument which indicates whether the initial values for the inequality slacks
   *                            (added by HiOp internally) are given by the user
   * @param[out] ineq_slack the user-defined initial values for the slacks added to transfer inequalities to equalities
   *                          (managed by Umpire)
   *
   */
  virtual bool get_starting_point(const size_type& n,
                                  const size_type& m,
                                  double* x0,
                                  bool& duals_avail,
                                  double* z_bndL0,
                                  double* z_bndU0,
                                  double* lambda0,
                                  bool& slacks_avail,
                                  double* ineq_slack)
  {
    duals_avail = false;
    slacks_avail = false;
    return false;
  }

  /**
   * Method provides a primal-dual starting point for warm start. This point is subject
   * to internal adjustments in HiOp.
   *
   * User provides starting point for all the iterate variable used in HiOp.
   * This method is for advanced users, as it will skip all the other safeguard in HiOp, e.g., project x into bounds.
   *
   * @param[in] n the global number of variables
   * @param[in] m the number of constraints
   * @param[out] x0 the user-defined initial values for the primal variablers (managed by Umpire)
   * @param[out] z_bndL0 the user-defined initial values for the duals of the variable lower bounds (managed by Umpire)
   * @param[out] z_bndU0 the user-defined initial values for the duals of the variable upper bounds (managed by Umpire)
   * @param[out] lambda0 the user-defined initial values for the duals of the constraints (managed by Umpire)
   * @param[out] ineq_slack the user-defined initial values for the slacks added to transfer inequalities to equalities
   *                          (managed by Umpire)
   * @param[out] vl0 the user-defined initial values for the duals of the constraint lower bounds (managed by Umpire)
   * @param[out] vu0 the user-defined initial values for the duals of the constraint upper bounds (managed by Umpire)
   *
   */
  virtual bool get_warmstart_point(const size_type& n,
                                   const size_type& m,
                                   double* x0,
                                   double* z_bndL0,
                                   double* z_bndU0,
                                   double* lambda0,
                                   double* ineq_slack,
                                   double* vl0,
                                   double* vu0)
  {
    return false;
  }

  /**
   * Callback method called by HiOp when the optimal solution is reached. User should use it
   * to retrieve primal-dual optimal solution.
   *
   * @param[in] status status of the solution process
   * @param[in] n global number of variables
   * @param[in] x array of (local) entries of the primal variables at solution (managed by Umpire, see note below)
   * @param[in] z_L array of (local) entries of the dual variables for lower bounds at solution (managed by Umpire,
   * see note below)
   * @param[in] z_U array of (local) entries of the dual variables for upper bounds at solution (managed by Umpire,
   * see note below)
   * @param[in] g array of the values of the constraints body at solution (managed by Umpire, see note below)
   * @param[in] lambda array of (local) entries of the dual variables for constraints at solution (managed by Umpire,
   * see note below)
   * @param[in] obj_value objective value at solution
   *
   * @note HiOp's option `callback_mem_space` can be used to change the memory location of array parameters managaged by
   * Umpire. More specifically, when `callback_mem_space` is set to `host` (and `mem_space` is `device`), HiOp transfers the
   * arrays from device to host first, and then passes/returns pointers on host for the arrays managed by Umpire. These
   * pointers can be then used in host memory space (without the need to rely on or use Umpire).
   *
   */
  virtual void solution_callback(hiopSolveStatus status,
                                 size_type n,
                                 const double* x,
                                 const double* z_L,
                                 const double* z_U,
                                 size_type m,
                                 const double* g,
                                 const double* lambda,
                                 double obj_value)
  {}

  /**
   * Callback for the (end of) iteration. This method is not called during the line-searche
   * procedure. @see solution_callback() for an explanation of the parameters.
   *
   * @note If the user (implementer) of this methods returns false, HiOp will stop the
   * the optimization with hiop::hiopSolveStatus ::User_Stopped return code.
   *
   * @param[in] iter the current iteration number
   * @param[in] obj_value objective value
   * @param[in] logbar_obj_value log barrier objective value
   * @param[in] n global number of variables
   * @param[in] x array of (local) entries of the primal variables (managed by Umpire, see note below)
   * @param[in] z_L array of (local) entries of the dual variables for lower bounds (managed by Umpire, see note below)
   * @param[in] z_U array of (local) entries of the dual variables for upper bounds (managed by Umpire, see note below)
   * @param[in] m_ineq the number of inequality constraints
   * @param[in] s array of the slacks added to transfer inequalities to equalities (managed by Umpire, see note below)
   * @param[in] m the number of constraints
   * @param[in] g array of the values of the constraints body (managed by Umpire, see note below)
   * @param[in] lambda array of (local) entries of the dual variables for constraints (managed by Umpire, see note below)
   * @param[in] inf_pr inf norm of the primal infeasibilities
   * @param[in] inf_du inf norm of the dual infeasibilities
   * @param[in] onenorm_pr one norm of the primal infeasibilities
   * @param[in] mu the log barrier parameter
   * @param[in] alpha_du dual step size
   * @param[in] alpha_pr primal step size
   * @param[in] ls_trials the number of line search iterations
   *
   * @note HiOp's option `callback_mem_space` can be used to change the memory location of array parameters managaged by
   * Umpire. More specifically, when `callback_mem_space` is set to `host` (and `mem_space` is `device`), HiOp transfers the
   * arrays from device to host first, and then passes/returns pointers on host for the arrays managed by Umpire. These
   * pointers can be then used in host memory space (without the need to rely on or use Umpire).
   *
   */
  virtual bool iterate_callback(int iter,
                                double obj_value,
                                double logbar_obj_value,
                                int n,
                                const double* x,
                                const double* z_L,
                                const double* z_U,
                                int m_ineq,
                                const double* s,
                                int m,
                                const double* g,
                                const double* lambda,
                                double inf_pr,
                                double inf_du,
                                double onenorm_pr,
                                double mu,
                                double alpha_du,
                                double alpha_pr,
                                int ls_trials)
  {
    return true;
  }

  /**
   * This method is used to provide user all the internal hiop iterates. @see solution_callback()
   * for an explanation of the parameters.
   *
   * @param[in] x array of (local) entries of the primal variables (managed by Umpire, see note below)
   * @param[in] z_L array of (local) entries of the dual variables for lower bounds (managed by Umpire, see note below)
   * @param[in] z_U array of (local) entries of the dual variables for upper bounds (managed by Umpire, see note below)
   * @param[in] yc array of (local) entries of the dual variables for equality constraints (managed by Umpire, see note
   * below)
   * @param[in] yd array of (local) entries of the dual variables for inequality constraints (managed by Umpire, see note
   * below)
   * @param[in] s array of the slacks added to transfer inequalities to equalities (managed by Umpire, see note below)
   * @param[in] v_L array of (local) entries of the dual variables for constraint lower bounds (managed by Umpire, see note
   * below)
   * @param[in] v_U array of (local) entries of the dual variables for constraint upper bounds (managed by Umpire, see note
   * below)
   *
   * @note HiOp's option `callback_mem_space` can be used to change the memory location of array parameters managaged by
   * Umpire. More specifically, when `callback_mem_space` is set to `host` (and `mem_space` is `device`), HiOp transfers the
   * arrays from device to host first, and then passes/returns pointers on host for the arrays managed by Umpire. These
   * pointers can be then used in host memory space (without the need to rely on or use Umpire).
   *
   */
  virtual bool iterate_full_callback(const double* x,
                                     const double* z_L,
                                     const double* z_U,
                                     const double* yc,
                                     const double* yd,
                                     const double* s,
                                     const double* v_L,
                                     const double* v_U)
  {
    return true;
  }

  /**
   * A wildcard function used to change the primal variables.
   *
   * @note If the user (implementer) of this methods returns false, HiOp will stop the
   * the optimization with hiop::hiopSolveStatus::User_Stopped return code.
   */
  virtual bool force_update_x(const int n, double* x) { return true; }

private:
  hiopInterfaceBase(const hiopInterfaceBase&) {};
  void operator=(const hiopInterfaceBase&) {};
};

/** Specialized interface for NLPs with 'global' but few constraints.
 */
class hiopInterfaceDenseConstraints : public hiopInterfaceBase
{
public:
  hiopInterfaceDenseConstraints() {};
  virtual ~hiopInterfaceDenseConstraints() {};
  /**
   * Evaluates the Jacobian of the subset of constraints indicated by idx_cons and of size num_cons.
   * Example: Assuming idx_cons[k]=i, which means that the gradient of the (i+1)th constraint is
   * to be evaluated, one needs to do Jac[k][0]=d/dx_0 con_i(x), Jac[k][1]=d/dx_1 con_i(x), ...
   * When MPI enabled, each rank computes only the local columns of the Jacobian, that is the partials
   * with respect to local variables.
   *
   * The parameter 'Jac' is passed as as a contiguous array storing the dense Jacobian matrix by rows.
   *
   * Parameters: see eval_cons
   */
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             double* Jac) = 0;

  /**
   * Evaluates the Jacobian of equality and inequality constraints in one call.
   *
   * The main difference from the above 'eval_Jac_cons' is that the implementer/user of this
   * method does not have to split the constraints into equalities and inequalities; instead,
   * HiOp does this internally.
   *
   * The parameter 'Jac' is passed as as a contiguous array storing the dense Jacobian matrix by rows.
   *
   * TODO: build an example (new one-call Nlp formulation derived from ex2) to illustrate this
   * feature and to test HiOp's internal implementation of eq.-ineq. spliting.
   */
  virtual bool eval_Jac_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* Jac)
  {
    return false;
  }
};

/**
 * Specialized interface for NLPs having mixed DENSE and sparse (MDS) blocks in the
 * Jacobian and Hessian.
 *
 * More specifically, this interface is for specifying optimization problem in x
 * split as (xs,xd), the rule of thumb being that xs have sparse derivatives and
 * xd have dense derivatives
 *
 * min f(x) s.t. g(x) <= or = 0, lb<=x<=ub
 * such that
 *  - Jacobian w.r.t. xs and LagrHessian w.r.t. (xs,xs) are sparse
 *  - Jacobian w.r.t. xd and LagrHessian w.r.t. (xd,xd) are dense
 *  - LagrHessian w.r.t (xs,xd) is zero (later this assumption will be relaxed)
 *
 * @note HiOp expects the sparse variables first and then the dense variables. In many cases,
 * the implementer has to (inconviniently) keep a map between his internal variables
 * indexes and the indexes HiOp.
 *
 * @note This interface is 'local' in the sense that data is not assumed to be
 * distributed across MPI ranks ('get_vecdistrib_info' should return 'false')
 *
 */
class hiopInterfaceMDS : public hiopInterfaceBase
{
public:
  hiopInterfaceMDS() {};
  virtual ~hiopInterfaceMDS() {};

  /**
   *  Returns the sizes and number of nonzeros of the sparse and dense blocks within MDS
   *
   * @param[out] nx_sparse number of sparse variables
   * @param[out] nx_ense number of dense variables
   * @param[out] nnz_sparse_Jace number of nonzeros in the Jacobian of the equalities w.r.t.
   *             sparse variables
   * @param[out] nnz_sparse_Jaci number of nonzeros in the Jacobian of the inequalities w.r.t.
   *             sparse variables
   * @param[out] nnz_sparse_Hess_Lagr_SS number of nonzeros in the (sparse) Hessian w.r.t.
   *             sparse variables
   * @param[out] nnz_sparse_Hess_Lagr_SD reserved, should be set to 0
   */
  virtual bool get_sparse_dense_blocks_info(int& nx_sparse,
                                            int& nx_dense,
                                            int& nnz_sparse_Jaceq,
                                            int& nnz_sparse_Jacineq,
                                            int& nnz_sparse_Hess_Lagr_SS,
                                            int& nnz_sparse_Hess_Lagr_SD) = 0;

  /**
   * Evaluates the Jacobian of constraints split in the sparse (triplet format) and
   * dense matrices (rows storage)
   *
   * This method is called twice per Jacobian evaluation, once for equalities and once for
   * inequalities (see 'eval_cons' for more information). It is advantageous to provide
   * this method when the underlying NLP's constraints come naturally split in equalities
   * and inequalities. When it is not convenient to do so, use 'eval_Jac_cons' below.
   *
   * @param[in] n number of variables
   * @param[in] m Number of constraints
   * @param[in] num_cons number of constraints to evaluate (size of idx_cons array)
   * @param[in] idx_cons indexes of the constraints to evaluate (managed by Umpire)
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *                  previously (false) or not (true) at x
   * @param[in] nsparse number of sparse variables
   * @param[in] ndense number of dense variables
   * @param[in] nnzJacS number of nonzeros in the sparse Jacobian
   * @param[out] iJacS array of row indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] jJacS array of column indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] MJacS array of nonzero values in the sparse Jacobian (managed by Umpire)
   * @param[out] JacD array with the values of the dense Jacobian (managed by Umpire)
   *
   * The implementer of this method should be aware of the following observations.
   * 1) 'JacD' parameter will be always non-null
   * 2) When 'iJacS' and 'jJacS' are non-null, the implementer should provide the (i,j)
   * indexes.
   * 3) When 'MJacS' is non-null, the implementer should provide the values corresponding to
   * entries specified by 'iJacS' and 'jJacS'
   * 4) 'iJacS' and 'jJacS' are both either non-null or null during a call.
   * 5) Both 'iJacS'/'jJacS' and 'MJacS' can be non-null during the same call or only one of
   * them non-null; but they will not be both null.
   */
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nsparse,
                             const size_type& ndense,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS,
                             double* JacD) = 0;

  /**
   * Evaluates the Jacobian of equality and inequality constraints in one call. This Jacobian is
   * mixed dense-sparse (MDS), which means is structurally split in the sparse (triplet format) and
   * dense matrices (rows storage)
   *
   * The main difference from the above 'eval_Jac_cons' is that the implementer/user of this
   * method does not have to split the constraints into equalities and inequalities; instead,
   * HiOp does this internally.  HiOp will call this method whenever the implementer/user returns
   * false from the 'eval_Jac_cons' above (which is called for equalities and inequalities separately).
   *
   * @param[in] n number of variables
   * @param[in] m Number of constraints
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated previously
   *                  (false) or not (true) at x
   * @param[in] nsparse number of sparse variables
   * @param[in] ndense number of dense variables
   * @param[in] nnzJacS number of nonzeros in the sparse Jacobian
   * @param[out] iJacS array of row indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] jJacS array of column indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] MJacS array of nonzero values in the sparse Jacobian (managed by Umpire)
   * @param[out] JacD array with the values of the dense Jacobian (managed by Umpire)
   *
   * Notes for implementer of this method:
   * 1) 'JacD' parameter will be always non-null.
   * 2) When 'iJacS' and 'jJacS' are non-null, the implementer should provide the (i,j) indexes.
   * 3) When 'MJacS' is non-null, the implementer should provide the values corresponding to
   * entries specified by 'iJacS' and 'jJacS' (managed by Umpire).
   * 4) 'iJacS' and 'jJacS' are both either non-null or null during a call.
   * 5) Both 'iJacS'/'jJacS' and 'MJacS' can be non-null during the same call or only one of them
   * non-null; but they will not be both null.
   */
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const size_type& nsparse,
                             const size_type& ndense,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS,
                             double* JacD)
  {
    return false;
  }

  /**
   * Evaluates the Hessian of the Lagrangian function in 3 structural blocks: HSS is the Hessian
   * w.r.t. (xs,xs),  HDD is the Hessian w.r.t. (xd,xd), and HSD is the Hessian w.r.t (xs,xd).
   * Please consult the user manual for a details on the form the Lagrangian function takes.
   *
   * @note HSD is for now assumed to be zero. The implementer should return nnzHSD=0
   * during the first call to 'eval_Hess_Lagr'. On subsequent calls, HiOp will pass the
   * triplet arrays for HSD set to NULL and the implementer (obviously) should not use them.
   *
   * @param[in] n number of variables
   * @param[in] m Number of constraints
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *                  previously (false) or not (true) at x
   * @param[in] obj_factor scalar that multiplies the objective term in the Lagrangian function
   * @param[in] lambda array with values of the multipliers used by the Lagrangian function
   * @param[in] new_lambda indicates whether lambda  values changed since last call
   * @param[in] nsparse number of sparse variables
   * @param[in] ndense number of dense variables
   * @param[in] nnzHSS number of nonzeros in the (sparse) Hessian w.r.t. sparse variables
   * @param[out] iHSS array of row indexes in the Hessian w.r.t. sparse variables (managed by
   *                  Umpire)
   * @param[out] jHSS array of column indexes in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] MHSS array of nonzero values in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] HDDD array with the values of the Hessian w.r.t. to dense variables
   *                  (managed by Umpire)
   * @param[out] iHSD is reserved and should not be accessed
   * @param[out] jHSD is reserved and should not be accessed
   * @param[out] MHSD is reserved and should not be accessed
   * @param[out] HHSD is reserved and should not be accessed
   *
   * Notes
   * 1)-5) from 'eval_Jac_cons' apply to xxxHSS and HDD arrays
   * 6) The order is multipliers is: lambda=[lambda_eq, lambda_ineq]
   */
  virtual bool eval_Hess_Lagr(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const size_type& nsparse,
                              const size_type& ndense,
                              const size_type& nnzHSS,
                              index_type* iHSS,
                              index_type* jHSS,
                              double* MHSS,
                              double* HDD,
                              size_type& nnzHSD,
                              index_type* iHSD,
                              index_type* jHSD,
                              double* MHSD) = 0;
};

/** Specialized interface for NLPs with sparse Jacobian and Hessian matrices.
 *
 * More specifically, this interface is for specifying optimization problem:
 *
 * min f(x) s.t. g(x) <=, =, or >= 0, lb<=x<=ub
 *
 * such that Jacobian w.r.t. x and Hessian of the Lagrangian w.r.t. x are sparse
 *
 * @note this interface is 'local' in the sense that data is not assumed to be
 * distributed across MPI ranks ('get_vecdistrib_info' should return 'false').
 * Acceleration can be however obtained using OpenMP and CUDA via Raja
 * abstraction layer that HiOp uses and via linear solver.
 *
 */
class hiopInterfaceSparse : public hiopInterfaceBase
{
public:
  hiopInterfaceSparse() {};
  virtual ~hiopInterfaceSparse() {};

  /** Get the number of variables and constraints, nonzeros
   * and get the number of nonzeros in Jacobian and Heesian
   */
  virtual bool get_sparse_blocks_info(size_type& nx,
                                      size_type& nnz_sparse_Jaceq,
                                      size_type& nnz_sparse_Jacineq,
                                      size_type& nnz_sparse_Hess_Lagr) = 0;

  /** Evaluates the sparse Jacobian of constraints.
   *
   * This method is called twice per Jacobian evaluation, once for equalities and once for
   * inequalities (see 'eval_cons' for more information). It is advantageous to provide
   * this method when the underlying NLP's constraints come naturally split in equalities
   * and inequalities. When it is not convenient to do so, see the overloaded method.
   *
   * Parameters:
   *  - first six: see eval_cons (in parent class)
   *  - nnzJacS, iJacS, jJacS, MJacS: number of nonzeros, (i,j) indexes, and values of
   * the sparse Jacobian.
   *
   */
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS) = 0;

  /** Evaluates the sparse Jacobian of equality and inequality constraints in one call.
   *
   * The main difference from the overloaded counterpart is that the implementer/user of this
   * method does not have to split the constraints into equalities and inequalities; instead,
   * HiOp does this internally.
   *
   * Parameters:
   *  - first four: number of variables, number of constraints, (primal) variables at which the
   * Jacobian should be evaluated, and boolean flag indicating whether the variables 'x' have
   * changed since a previous call to ny of the function and derivative evaluations.
   *  - nnzJacS, iJacS, jJacS, MJacS: number of nonzeros, (i,j) indexes, and values of
   * the sparse Jacobian block; indexes are within the sparse Jacobian block
   *
   * Notes for implementer of this method:
   * 1) When 'iJacS' and 'jJacS' are non-null, the implementer should provide the (i,j)
   * indexes.
   * 2) When 'MJacS' is non-null, the implementer should provide the values corresponding to
   * entries specified by 'iJacS' and 'jJacS'
   * 3) 'iJacS' and 'jJacS' are both either non-null or null during a call.
   * 4) Both 'iJacS'/'jJacS' and 'MJacS' can be non-null during the same call or only one of them
   * non-null; but they will not be both null.
   *
   * HiOp will call this method whenever the implementer/user returns false from the 'eval_Jac_cons'
   * (which is called for equalities and inequalities separately) above.
   */
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS)
  {
    return false;
  }

  /** Evaluates the sparse Hessian of the Lagrangian function.
   *
   * @note 1)-4) from 'eval_Jac_cons' applies to xxxHSS
   * @note 5) The order of multipliers is: lambda=[lambda_eq, lambda_ineq]
   */
  virtual bool eval_Hess_Lagr(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const size_type& nnzHSS,
                              index_type* iHSS,
                              index_type* jHSS,
                              double* MHSS) = 0;

  /** Specifying the get_MPI_comm code defined in the base class
   */
  virtual bool get_MPI_comm(MPI_Comm& comm_out)
  {
    comm_out = MPI_COMM_SELF;
    return true;
  }
};

}  // namespace hiop
#endif
