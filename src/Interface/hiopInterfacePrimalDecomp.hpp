#ifndef HIOP_INTERFACE_PRIDEC
#define HIOP_INTERFACE_PRIDEC

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "LinAlgFactory.hpp"
#include <cassert>
#include <cstring>  //for memcpy
#include <vector>

namespace hiop
{

/**
 * Base class (interface) for specifying optimization NLPs that have separable terms in the
 * objective, which we coin as "primal decomposable" problems. More specifically, these problems
 * have the following structure (please also take a note of the terminology):
 *
 *   min_x  basecase(x) + 1/S sum { r_i(x) : i=1,...,S}      (primal decomposable NLP)
 *
 * The subproblem <b>'basecase'</b> refers to a general nonlinear nonconvex NLP in `x`. We point out
 * that the basecase can have general <i>twice continously differentiable</i> objective and
 * constraints; the latter can be equalities, inequalities, and bounds on `x`.
 * Furthermore, borrowing from stochastic programming terminology, the terms `r_i` are
 * called <em> recourse terms </em>, or, in short, <b> r-terms </b>.
 *
 * In order to solve the above problem, HiOp solver will perform a series of approximations and will
 * require the user to solve a so-called 'master' problem
 *
 *   min  basecase(x) +  q(x)                            (master NLP)
 *    x
 * where the function q(x) is a convex differentiable approximation of sum { r_i(x) : i=1,...,S}
 * that we refer to as <i> quadratic regularization </i>.
 *
 * The user is required to maintain and solve the master problem, more specifically:
 *  - to add the quadratic regularization to the basecase NLP; the quadratic regularization is
 * provided by HiOp hiopInterfacePriDecProblem::RecourseApproxEvaluator classs. the user is
 * expected to implement hiopInterfacePriDecProblem::set_recourse_approx_evaluator in the master
 * problem class.
 *  - to (re)solve master NLP  and return the primal optimal solution `x` to HiOp; for doing this,
 * the user is required to implement hiopInterfacePriDecProblem::solve_master method.
 *
 * In addition, the user is required to implement
 *     - hiopInterfacePriDecProblem::eval_f_rterm
 *     - hiopInterfacePriDecProblem::eval_grad_rterm
 * which solves the individual recourse subproblems.
 *
 * These methods will be used by the HiOp's primal decomposition solver to  compute function value
 * and  gradient vector individually for each recourse term  r_i, which are needed to build the
 * convex regularizations q(x). The above methods will be called at arbitrary vectors `x` that
 * are decided internally  by HiOp.
 *
 */

class hiopInterfacePriDecProblem
{
public:
  /**
   * Constructor
   */
  hiopInterfacePriDecProblem() {}

  virtual ~hiopInterfacePriDecProblem() {}

  /**
   * Solves the master problem consisting of the basecase problem plus the recourse terms.
   * The recourse terms have been added by the outer optimization loop (hiopAlgPrimalDecomposition)
   * via the 'add_' methods below. (this does not appear to be case anymore Frank TODO)
   *
   * @param x : output, will contain the primal optimal solution of the master
   * @param include_r: boolean that determines whether the (approximated) recourse objective should be added to the basecase
   * @param rval/grad/hess: optional input that updates the function value, gradient and hessian of the recourse,
   * respectively. NOT implemented currently.
   *
   * @param master_options_file : input string specifying the name of the options file the NLP solver
   * should use when solving the master problem. A null value indicates that the NLP solver should use
   * its default options file.
   *
   */
  virtual hiopSolveStatus solve_master(hiopVector& x,
                                       const bool& include_r,
                                       const double& rval = 0,
                                       const double* grad = 0,
                                       const double* hess = 0,
                                       const char* master_options_file = nullptr) = 0;

  virtual bool eval_f_rterm(size_type idx, const int& n, const double* x, double& rval) = 0;
  virtual bool eval_grad_rterm(size_type idx, const int& n, double* x, hiopVector& grad) = 0;

  /**
   * Returns the number S of recourse terms
   */
  virtual size_type get_num_rterms() const = 0;
  /**
   * Return the number of primal optimization variables
   */
  virtual size_type get_num_vars() const = 0;

  virtual void get_solution(double* x) const = 0;
  virtual double get_objective() = 0;

  /**
   * Define the evaluator class called by the base case problem class to add the quadratic
   * recourse approximation.
   * This class is intened for internal use of hiopInterfacePriDecProblem class only
   * In the cases where only RecourseApproxEvaluator is needed, a shell hiopInterfacePriDecProblem
   * is still required to be created.
   */

  // Notes:
  // since this class allocates vectors, it (as well as hiopInterfacePriDecProblem) needs to be aware of the memory space
  //
  class RecourseApproxEvaluator
  {
  public:
    RecourseApproxEvaluator(int nc, const std::string& mem_space);

    RecourseApproxEvaluator(int nc, int S, const std::string& mem_space);

    RecourseApproxEvaluator(const int nc, const int S, const int* list, const std::string& mem_space);

    RecourseApproxEvaluator(const int nc,
                            const int S,
                            const double& rval,
                            const hiopVector& rgrad,
                            const hiopVector& rhess,
                            const hiopVector& x0,
                            const std::string& mem_space);

    RecourseApproxEvaluator(int nc,
                            int S,
                            const int* list,
                            const double& rval,
                            const hiopVector& rgrad,
                            const hiopVector& rhess,
                            const hiopVector& x0,
                            const std::string& mem_space);

    virtual ~RecourseApproxEvaluator();

    bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);

    bool eval_grad(const size_type& n, const double* x, bool new_x, double* grad);

    bool eval_hess(const size_type& n, const hiopVector& x, bool new_x, hiopVector& hess);

    virtual bool get_MPI_comm(MPI_Comm& comm_out);

    void set_rval(const double rval);
    void set_rgrad(const int n, const hiopVector& rgrad);
    void set_rhess(const int n, const hiopVector& rhess);
    void set_x0(const int n, const hiopVector& x0);
    // void set_xc_idx(const int* idx);

    int get_S() const;
    double get_rval() const;
    hiopVector* get_rgrad() const;
    hiopVector* get_rhess() const;
    hiopVector* get_x0() const;

  protected:
    int nc_, S_;
    hiopVectorInt* xc_idx_;
    double rval_;
    hiopVector* rgrad_;
    hiopVector* rhess_;  // diagonal Hessian vector
    hiopVector* x0_;     // current solution

    /// working buffer in the size of coupling variables (same size as x0_, rgrad_, and rhess_)
    hiopVector* vec_work_coupling_;

    /// working buffer in the size of the basecase (primal variables)
    hiopVector* vec_work_basecase_;

    /// memory space of the PriDec solver (must match the memory space of the NLP solver)
    std::string mem_space_;
  };

  virtual bool set_recourse_approx_evaluator(const int n, RecourseApproxEvaluator* evaluator) = 0;
};

}  // namespace hiop
#endif
