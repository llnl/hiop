#include "NlpMdsEx1.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include <cstdlib>
#include <string>

using namespace hiop;

// static bool self_check(size_type n, double obj_value);

static bool parse_arguments(int argc,
                            char** argv,
                            bool& self_check,
                            size_type& n_sp,
                            size_type& n_de,
                            bool& one_call_cons,
                            bool& empty_sp_row)
{
  self_check = false;
  empty_sp_row = false;
  n_sp = 1000;
  n_de = 1000;
  one_call_cons = false;
  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 6:  // 5 arguments
    {
      if(std::string(argv[5]) == "-selfcheck") self_check = true;
    }
    case 5:  // 4 arguments
    {
      if(std::string(argv[4]) == "-selfcheck") {
        self_check = true;
      }
      if(std::string(argv[4]) == "-empty_sp_row") {
        empty_sp_row = true;
      }
    }
    case 4:  // 3 arguments
    {
      one_call_cons = (bool)atoi(argv[3]);
    }
    case 3:  // 2 arguments
    {
      n_de = atoi(argv[2]);
      if(n_de < 0) n_de = 0;
    }
    case 2:  // 1 argument
    {
      n_sp = atoi(argv[1]);
      if(n_sp < 0) n_sp = 0;
    } break;
    default:
      return false;  // 5 or more arguments
  }

  if(self_check && (n_sp != 400 || n_de != 100)) return false;

  return true;
};

static void usage(const char* exeName)
{
  printf(
      "HiOp driver %s that solves a synthetic problem of variable size in the "
      "mixed dense-sparse formulation.\n",
      exeName);
  printf("Usage: \n");
  printf("  '$ %s sp_vars_size de_vars_size eq_ineq_combined_nlp -empty_sp_row -selfcheck'\n", exeName);
  printf("Arguments, all integers, excepting string '-selfcheck'\n");
  printf("  'sp_vars_size': # of sparse variables [default 400, optional]\n");
  printf("  'de_vars_size': # of dense variables [default 100, optional]\n");
  printf("  '-empty_sp_row': set an empty row in sparser inequality Jacobian. [optional]\n");
  printf(
      "  '-selfcheck': compares the optimal objective with sp_vars_size being 400 and "
      "de_vars_size being 100 (these two exact values must be passed as arguments). [optional]\n");
  printf(
      "  'eq_ineq_combined_nlp': 0 or 1, specifying whether the NLP formulation with split "
      "constraints should be used (0) or not (1) [default 0, optional]\n");
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  assert(MPI_SUCCESS == ierr);
  // int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(MPI_SUCCESS==ierr);
  if(comm_size != 1) {
    printf(
        "[error] driver detected more than one rank but the driver should be run "
        "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

#ifdef HIOP_USE_MAGMA
  magma_init();
#endif

  bool selfCheck, one_call_cons;
  bool has_empty_sp_row;
  size_type n_sp, n_de;
  if(!parse_arguments(argc, argv, selfCheck, n_sp, n_de, one_call_cons, has_empty_sp_row)) {
    usage(argv[0]);
    return 1;
  }

  double obj_value = -1e+20;
  hiopSolveStatus status;

  // user's NLP -> implementation of hiop::hiopInterfaceMDS
  MdsEx1* my_nlp;
  if(one_call_cons) {
    my_nlp = new MdsEx1OneCallCons(n_sp, n_de, has_empty_sp_row);
  } else {
    my_nlp = new MdsEx1(n_sp, n_de, has_empty_sp_row);
  }

  hiopNlpMDS nlp(*my_nlp);

  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("duals_init", "zero");

  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetStringValue("compute_mode", "hybrid");

  nlp.options->SetIntegerValue("verbosity_level", 3);
  nlp.options->SetNumericValue("mu0", 1e-1);
  nlp.options->SetNumericValue("tolerance", 1e-5);

  hiopAlgFilterIPMNewton solver(&nlp);
  status = solver.run();
  obj_value = solver.getObjective();

  int ret_code = 0;  // 0 success, -1 failure

  if(selfCheck) {  // && has_empty_sp_row) {
    if(fabs(obj_value - (-4.9994906229741609e+01)) > 1e-6) {
      printf(
          "selfcheck: objective mismatch for MDS Ex1 problem with 400 sparse variables and 100 "
          "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n",
          obj_value);
      ret_code = -1;
    } else {
      printf("selfcheck passed\n");
      ret_code = 0;
    }

  } else {
    if(status < 0) {
      if(rank == 0) {
        printf("solver returned negative solve status: %d (objective is %18.12e)\n", status, obj_value);
      }
      ret_code = -1;
    } else {
      if(rank == 0) {
        printf("solver returned successfully: objective is %18.12e)\n", obj_value);
      }
      ret_code = 0;
    }
  }

#if 0
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Reoptimize
  // -----------
  // 1. get solution from previous solve
  // 2. give it to the (user's) nlp, which will provide HiOp a full primal-dual restart via
  // 'get_starting_point' callback
  // Normally, the user would also change her nlp between steps 1 and 2 above, for example, different
  // bounds on variables or on inequalities
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  size_type n_vars, n_cons;
  my_nlp->get_prob_sizes(n_vars, n_cons);

  double x[n_vars];
  double zl[n_vars];
  double zu[n_vars];
  double lambdas[n_cons];

  solver.getSolution(x);
  solver.getDualSolutions(zl, zu, lambdas);

  my_nlp->set_solution_primal(x);
  my_nlp->set_solution_duals(zl, zu, lambdas);

  //
  // set options for solver re-optimization
  //
  
  //less agressive log-barrier parameter is always a safe bet
  nlp.options->SetNumericValue("mu0", 1e-6);
  nlp.options->SetNumericValue("tolerance", 1e-8);

  //nlp.options->SetIntegerValue("verbosity_level", 7);

  //nlp.options->SetNumericValue("kappa1", 1e-15);
  //nlp.options->SetNumericValue("kappa2", 1e-15);
  
  //solve
  status = solver.run();
  obj_value = solver.getObjective();
  
  if(status<0) {
    if(rank==0)
      printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  if(selfCheck) {
    if(fabs(obj_value-(-4.999509728895e+01))>1e-6) {
      printf("selfcheck: objective mismatch for MDS Ex1 problem with 400 sparse variables and 100 "
             "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", obj_value);
      return -1;
    }
  } else {
    if(rank==0) {
      printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
    }
  }

#endif

  delete my_nlp;

#ifdef HIOP_USE_MAGMA
  magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return ret_code;
}
