#include "IpIpoptApplication.hpp"

#include "IpoptAdapter.hpp"
// use HiOp's SparseEx2 - sparse NLP
#include "NlpSparseEx2.hpp"

#include "NlpSparseEx1.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <iostream>

using namespace Ipopt;
using namespace hiop;
// Example of how to use IpoptAdapter to solve HiOP-specified problems with Ipopt

int main(int argv, char** argc)
{
  // instantiate a HiOp problem
  int nx = 1000;
  int S = 1920;
  int nS = 5;
  double x0[nx];
  for(int i = 0; i < nx; i++) x0[i] = 1.0;

  SparseEx1 nlp_interface(nx, 1.0);
  hiopNlpSparse nlp(nlp_interface);
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "linear");
  //  nlp.options->SetStringValue("duals_init", "zero"); // "lsq" or "zero"
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  // nlp.options->SetStringValue("KKTLinsys", "full");
  // nlp.options->SetStringValue("write_kkt", "yes");
  // nlp.options->SetIntegerValue("max_iter", 100);
  nlp.options->SetNumericValue("mu0", 0.1);
  hiopAlgFilterIPMNewton solver(&nlp);
  hiopSolveStatus status0 = solver.run();
  solver.getSolution(x0);

  double x[nx + S * nx];
  PriDecEx2 hiopNlp(nx, S, nS);
  hiopNlp.set_starting_point(x0);

  // Create a new instance of the Ipopt nlp
  //  (use a SmartPtr, not raw)
  SmartPtr<TNLP> mynlp = new hiopSparse2IpoptTNLP(&hiopNlp);

  // Create a new instance of IpoptApplication
  // (use a SmartPtr, not raw)
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

  //
  // HiOp-compatible Ipopt Options (Ipopt behaves or should behave like HiOp)
  //
  // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  // app->Options()->SetStringValue("derivative_test", "first-order");
  // app->Options()->SetNumericValue("derivative_test_perturbation", 1e-7);
  // app->Options()->SetNumericValue("bound_push", 1e-2);
  // app->Options()->SetNumericValue("bound_relax_factor", 0.);
  // app->Options()->SetNumericValue("constr_mult_init_max", 0.001);

  // app->Options()->SetNumericValue("tol", 1e-7);
  // app->Options()->SetStringValue("recalc_y", "no");
  // app->Options()->SetIntegerValue("print_level", 11);
  // app->Options()->SetStringValue("mu_strategy", "monotone");
  // app->Options()->SetNumericValue("bound_frac", 1e-8);
  // app->Options()->SetNumericValue("bound_push", 1e-8);
  // app->Options()->SetNumericValue("slack_bound_push", 1e-24);
  // app->Options()->SetNumericValue("bound_relax_factor", 0.);
  // app->Options()->SetNumericValue("constr_mult_init_max", 0.001);
  // app->Options()->SetNumericValue("kappa1", 1e-8);
  // app->Options()->SetNumericValue("kappa2", 1e-8);

  // app->Options()->SetStringValue("output_file", "ipopt.out");
  // app->Options()->SetStringValue("derivative_test", "second-order"); //"only-second-order"
  // Initialize the IpoptApplication and process the options
  ApplicationReturnStatus status;
  status = app->Initialize();
  if(status != Solve_Succeeded) {
    std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    return (int)status;
  }

  // Ask Ipopt to solve the problem
  status = app->OptimizeTNLP(mynlp);

  if(status == Solve_Succeeded) {
    std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
  } else {
    std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
  }

  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.

  return (int)status;
}
