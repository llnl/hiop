#include "NlpPriDecEx1.hpp"

#include <cassert>
#include <cstring>  //for memcpy
#include <cstdio>
#include <cmath>

PriDecEx1::PriDecEx1(int ns_)
    : PriDecEx1(ns_, ns_)
{}  // ns = nx, nd=S

PriDecEx1::PriDecEx1(int ns_, int S_)
    : ns(ns_),
      evaluator_(nullptr)
{
  if(ns < 0) {
    ns = 0;
  } else {
    if(4 * (ns / 4) != ns) {
      ns = 4 * ((4 + ns) / 4);
      printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", ns_, ns);
    }
  }
  if(S_ < 0) {
    S = 0;
  } else {
    S = S_;
  }
  if(S < ns) {
    S = ns;
    printf(
        "[warning] number (%d) of recourse problems should be larger than sparse vars  %d,"
        " changed to be the same\n",
        S,
        ns);
  }
  nc = ns;
}

PriDecEx1::PriDecEx1(int ns_, int S_, int nc_)
    : PriDecEx1(ns_, S_)
{
  nc = nc_;
}

PriDecEx1::PriDecEx1(int ns_, int S_, int nc_, bool include_)
    : PriDecEx1(ns_, S_, nc_)
{
  include_r = include_;
  if(include_r) {
    evaluator_ = new hiopInterfacePriDecProblem::RecourseApproxEvaluator(nc_, "default");
  }
}

PriDecEx1::PriDecEx1(int ns_, int S_, bool include, hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
    : PriDecEx1(ns_, S_)
{
  include_r = include;
  evaluator_ = evaluator;
}

PriDecEx1::~PriDecEx1() {}

bool PriDecEx1::get_prob_sizes(size_type& n, size_type& m)
{
  n = ns;
  m = 0;
  return true;
}

bool PriDecEx1::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  // assert(n>=4 && "number of variables should be greater than 4 for this example");
  assert(n == ns);
  // define x bounds
  for(int i = 0; i < ns; ++i) xlow[i] = 0.;
  for(int i = 0; i < ns; ++i) xupp[i] = +1e+20;
  for(int i = 0; i < ns; ++i) type[i] = hiopNonlinear;
  // uncoupled x fixed
  // for testing
  if(nc < ns) {
    for(int i = nc + 1; i < ns; ++i) xlow[i] = 1.;
    for(int i = nc + 1; i < ns; ++i) xupp[i] = 1.;
    xupp[0] = 1.;
    xupp[0] = 1.;
  }
  return true;
};

bool PriDecEx1::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == 0);
  return true;
};

bool PriDecEx1::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  obj_value = 0.;  // x[0]*(x[0]-1.);
  // sum 0.5 {(x_i-1)*(x_{i}-1) : i=1,...,ns}
  for(int i = 0; i < n; i++) obj_value += (x[i] - 1.) * (x[i] - 1.);
  obj_value *= 0.5;

  if(include_r) {
    assert(evaluator_->get_rgrad() != NULL);
    evaluator_->eval_f(n, x, new_x, obj_value);
  }
  return true;
};

bool PriDecEx1::eval_cons(const size_type& n,
                          const size_type& m,
                          const size_type& num_cons,
                          const index_type* idx_cons,
                          const double* x,
                          bool new_x,
                          double* cons)
{
  assert(num_cons == 0);
  return true;
};

bool PriDecEx1::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
  for(int i = 0; i < n; i++) {
    gradf[i] = x[i] - 1.;
  }
  if(include_r) {
    assert(evaluator_->get_rgrad() != NULL);
    evaluator_->eval_grad(n, x, new_x, gradf);
  }
  return true;
};

bool PriDecEx1::get_starting_point(const size_type& global_n, double* x0_)
{
  assert(global_n == ns);
  for(int i = 0; i < global_n; i++) x0_[i] = 2.;
  return true;
};

bool PriDecEx1::get_starting_point(const size_type& n,
                                   const size_type& m,
                                   double* x0_,
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
};

bool PriDecEx1::get_MPI_comm(MPI_Comm& comm_out)
{
  comm_out = MPI_COMM_SELF;
  return true;
};

bool PriDecEx1::eval_Jac_cons(const size_type& n,
                              const size_type& m,
                              const size_type& num_cons,
                              const index_type* idx_cons,
                              const double* x,
                              bool new_x,
                              double* Jac)
{
  assert(m == 0);
  return true;
};

bool PriDecEx1::quad_is_defined()
{
  if(evaluator_ != NULL)
    return true;
  else
    return false;
};

bool PriDecEx1::set_quadratic_terms(const int& n, hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
{
  assert(nc == n);
  evaluator_ = evaluator;
  return true;
};

bool PriDecEx1::set_include(bool include)
{
  include_r = include;
  return true;
};

hiopSolveStatus PriDecMasterProblemEx1::solve_master(hiopVector& x,
                                                     const bool& include_r,
                                                     const double& rval /* = 0*/,
                                                     const double* grad /*=0*/,
                                                     const double* hess /*=0*/,
                                                     const char* master_options_file /*=nullptr*/)
{
  obj_ = -1e+20;
  hiopSolveStatus status;
  if(my_nlp == NULL) {
    if(n_ == nc_) {
      my_nlp = new PriDecEx1(n_, S_);
    } else {
      my_nlp = new PriDecEx1(n_, S_, nc_);
    }
  }

  [[maybe_unused]] bool ierr = my_nlp->set_include(include_r);
  if(include_r) {
    assert(my_nlp->quad_is_defined());
  }
  // check to see if the resource value and gradient are correct
  // printf("recourse value: is %18.12e)\n", rval_);
  hiopNlpDenseConstraints nlp(*my_nlp, master_options_file);

  // any of the options below can be overwritten by specifying them in the 'hiop_pridec_master.options' file

  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("duals_init", "zero");  // "lsq" or "zero"
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("fixed_var", "relax");
  /*
  nlp.options->SetStringValue("dualsInitialization", "zero");
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("compute_mode", "hybrid");
  nlp.options->SetNumericValue("mu0", 1e-1);
  nlp.options->SetNumericValue("tolerance", 1e-5);
  */

  nlp.options->SetIntegerValue("verbosity_level", 1);
  hiopAlgFilterIPM solver(&nlp);
  status = solver.run();
  obj_ = solver.getObjective();

  double* x_vec = x.local_data();
  solver.getSolution(x_vec);

  if(status < 0) {
    printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_);
    return status;
  }
  if(sol_ == nullptr) {
    sol_ = new double[n_];
  }

  memcpy(sol_, x_vec, n_ * sizeof(double));
  // assert("for debugging" && false); //for debugging purpose
  return Solve_Success;
};

bool PriDecMasterProblemEx1::eval_f_rterm(size_type idx, const int& n, const double* x, double& rval)
{
  rval = 0.;
  for(int i = 0; i < n; i++) {
    if(i == static_cast<int>(idx)) {
      rval += (x[i] + S_) * (x[i] + S_);
    } else {
      rval += x[i] * x[i];
    }
  }
  rval *= 0.5;
  // rval /= S_;
  return true;
};

// x is handled by primalDecomp to be the correct coupled x
bool PriDecMasterProblemEx1::eval_grad_rterm(size_type idx, const int& n, double* x, hiopVector& grad)
{
  assert(static_cast<int>(nc_) == n);
  double* grad_vec = grad.local_data();
  for(int i = 0; i < n; i++) {
    if(i == static_cast<int>(idx)) {
      grad_vec[i] = (x[i] + S_);
    } else {
      grad_vec[i] = x[i];
    }
  }
  return true;
};

bool PriDecMasterProblemEx1::set_recourse_approx_evaluator(const int n,
                                                           hiopInterfacePriDecProblem::RecourseApproxEvaluator* evaluator)
{
  my_nlp->set_quadratic_terms(n, evaluator);
  return true;
}
