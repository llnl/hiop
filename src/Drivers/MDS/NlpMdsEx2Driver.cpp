#include "NlpMdsEx2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include <cstdlib>
#include <string>

using namespace hiop;

// static bool self_check(size_type n, double obj_value);

static bool parse_arguments(int argc, char** argv, bool& self_check, size_type& n_sp, size_type& n_de, bool& rdJac)
{
  self_check = rdJac = false;
  n_sp = 400;
  n_de = 100;

  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 5:  // 4 arguments
    {
      if(std::string(argv[4]) == "-withrdJ") rdJac = true;
    }
    case 4:  // 3 arguments
    {
      if(std::string(argv[3]) == "-selfcheck")
        self_check = true;
      else if(std::string(argv[3]) == "-withrdJ")
        rdJac = true;
    }
    case 3:  // 2 arguments
    {
      n_de = atoi(argv[2]);
      if(n_de < 0) n_de = 2;
    }
    case 2:  // 1 argument
    {
      n_sp = atoi(argv[1]);
      if(n_sp < 0) n_sp = 4;
    } break;
    default:
      return false;  // 4 or more arguments
  }

  if(self_check && n_sp != 400 && n_de != 100) {
    if(!rdJac) {
      printf(
          "Error: incorrect input parameters: '-selfcheck' must be used with predefined "
          "values for input  parameters, sp_vars_size=400 de_vars_size=100.\n");
      return false;
    }
  }

  return true;
};

static void usage(const char* exeName)
{
  printf(
      "HiOp driver %s that solves a nonconvex synthetic problem of variable size in the "
      "mixed dense-sparse formulation. In addition, the driver can be instructed to "
      "solve additional problems that have rank-deficient Jacobian (use '-withrdJ' option)\n",
      exeName);
  printf("Usage: \n");
  printf("  '$ %s sp_vars_size de_vars_size -selfcheck -withrdJ'\n", exeName);
  printf(
      "Arguments, all integers, excepting strings '-selfcheck' and '-withrdJ', should be "
      "specified in the order below.\n");
  printf("  'sp_vars_size': # of sparse variables [default 400, optional, nonnegative integer].\n");
  printf("  'de_vars_size': # of dense variables [default 100, optional, nonnegative integer].\n");
  printf(
      "  '-selfcheck': compares the optimal objective with sp_vars_size being 400 and "
      "de_vars_size being 100 (these two exact values must be passed as arguments). [optional]\n");
  printf(
      "  '-withrdJ': solves additional problems with rank-deficient Jacobians; discards "
      "'-selfcheck' option. [optional]\n");
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

  bool selfCheck, rdJac;
  size_type n_sp, n_de;
  if(!parse_arguments(argc, argv, selfCheck, n_sp, n_de, rdJac)) {
    usage(argv[0]);
    return 1;
  }

  hiopSolveStatus status1, status2, status3, status4;

  // we can do four tests
  // 1. convex obj, rank deficient Jacobian of eq, full rank Jacobian of ineq
  // 2. convex obj, full rank Jacobian of eq, rank deficient Jacobian of ineq
  // 3. nonconvex obj, full rank Jacobian of eq, full rank Jacobian of ineq
  // 4. nonconvex obj, rank deficient Jacobian of eq,  rank deficient Jacobian of ineq
  //
  // all four tests are done when '-withrdJ' is on
  // only test 3 is done when '-withrdJ' is off

  double obj_value1, obj_value2, obj_value3, obj_value4;

  // test 1
  if(rdJac) {
    bool convex_obj = true;
    bool rankdefic_Jac_eq = true;
    bool rankdefic_Jac_ineq = false;

    hiopInterfaceMDS* nlp_interface = new MdsEx2(n_sp, n_de, convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq);

    hiopNlpMDS nlp(*nlp_interface);

    nlp.options->SetStringValue("duals_update_type", "linear");
    //    nlp.options->SetStringValue("duals_init", "zero");

    nlp.options->SetStringValue("Hessian", "analytical_exact");
    // nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    nlp.options->SetStringValue("compute_mode", "hybrid");

    nlp.options->SetIntegerValue("verbosity_level", 3);
    nlp.options->SetNumericValue("mu0", 1e-1);
    hiopAlgFilterIPMNewton solver(&nlp);
    status1 = solver.run();
    obj_value1 = solver.getObjective();

    delete nlp_interface;

    if(status1 < 0) {
      if(rank == 0) printf("solve1 trouble: returned %d (with objective is %18.12e)\n", status1, obj_value1);
      return -1;
    }
  }  // end of test 1

  // test 2
  if(rdJac) {
    bool convex_obj = true;
    bool rankdefic_Jac_eq = false;
    bool rankdefic_Jac_ineq = true;

    hiopInterfaceMDS* nlp_interface = new MdsEx2(n_sp, n_de, convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq);

    hiopNlpMDS nlp(*nlp_interface);

    nlp.options->SetStringValue("duals_update_type", "linear");
    //    nlp.options->SetStringValue("duals_init", "zero");

    nlp.options->SetStringValue("Hessian", "analytical_exact");
    // nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    nlp.options->SetStringValue("compute_mode", "hybrid");

    nlp.options->SetIntegerValue("verbosity_level", 3);
    nlp.options->SetNumericValue("mu0", 1e-1);
    hiopAlgFilterIPMNewton solver(&nlp);
    status2 = solver.run();
    obj_value2 = solver.getObjective();

    delete nlp_interface;

    if(status2 < 0) {
      if(rank == 0) printf("solve2 trouble: returned %d (with objective is %18.12e)\n", status2, obj_value2);
      return -1;
    }
  }  // end of test 2

  // test 3
  {
    bool convex_obj = false;
    bool rankdefic_Jac_eq = false;
    bool rankdefic_Jac_ineq = false;

    hiopInterfaceMDS* nlp_interface = new MdsEx2(n_sp, n_de, convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq);

    hiopNlpMDS nlp(*nlp_interface);

    nlp.options->SetStringValue("duals_update_type", "linear");
    //    nlp.options->SetStringValue("duals_init", "zero");

    nlp.options->SetStringValue("Hessian", "analytical_exact");
    // nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    nlp.options->SetStringValue("compute_mode", "hybrid");

    nlp.options->SetIntegerValue("verbosity_level", 3);
    nlp.options->SetNumericValue("mu0", 1e-1);
    hiopAlgFilterIPMNewton solver(&nlp);
    status3 = solver.run();
    obj_value3 = solver.getObjective();

    delete nlp_interface;

    if(status3 < 0) {
      if(rank == 0) printf("solve3 trouble: returned %d (with objective is %18.12e)\n", status3, obj_value3);
      return -1;
    }
  }  // end of test 3

  // test 4
  if(rdJac) {
    bool convex_obj = false;
    bool rankdefic_Jac_eq = true;
    bool rankdefic_Jac_ineq = true;

    hiopInterfaceMDS* nlp_interface = new MdsEx2(n_sp, n_de, convex_obj, rankdefic_Jac_eq, rankdefic_Jac_ineq);

    hiopNlpMDS nlp(*nlp_interface);

    nlp.options->SetStringValue("duals_update_type", "linear");
    //    nlp.options->SetStringValue("duals_init", "zero");

    nlp.options->SetStringValue("Hessian", "analytical_exact");
    // nlp.options->SetStringValue("KKTLinsys", "xdycyd");
    nlp.options->SetStringValue("compute_mode", "hybrid");

    nlp.options->SetIntegerValue("verbosity_level", 3);
    nlp.options->SetNumericValue("mu0", 1e-1);
    hiopAlgFilterIPMNewton solver(&nlp);
    status4 = solver.run();
    obj_value4 = solver.getObjective();

    delete nlp_interface;

    if(status4 < 0) {
      if(rank == 0) printf("solve4 trouble: returned %d (with objective is %18.12e)\n", status4, obj_value4);
      return -1;
    }
  }  // end of test 4

  bool selfcheck_ok = true;
  // this is used for testing when the driver is in '-selfcheck' mode
  if(selfCheck) {
    // if(rdJac && fabs(obj_value1-(-3.160999998751e+03))>1e-6) {
    //   printf("selfcheck1: objective mismatch for MDS Ex2 problem with 400 sparse variables and 100 "
    //          "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", obj_value1);
    //   selfcheck_ok = false;
    // }
    // if(rdJac && fabs(obj_value2-(-1.24881064633628e+01))>1e-6) {
    //   printf("selfcheck2: objective mismatch for MDS Ex2 problem with 400 sparse variables and 100 "
    //          "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", obj_value2);
    //   selfcheck_ok = false;
    // }
    if((fabs(obj_value3 - (-3.160999998751e+03)) / 3.160999998751e+03) > 1e-6) {
      printf(
          "selfcheck3: objective mismatch for MDS Ex2 problem with 400 sparse variables and 100 "
          "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n",
          obj_value3);
      selfcheck_ok = false;
    }
    // if(rdJac && fabs(obj_value4-(-1.35649999989221e+03))>1e-6) {
    //   printf("selfcheck4: objective mismatch for MDS Ex2 problem with 400 sparse variables and 100 "
    //              "dense variables did. BTW, obj=%18.12e was returned by HiOp.\n", obj_value4);
    //   selfcheck_ok = false;
    // }

    if(false == selfcheck_ok) {
      std::cout << "Selfcheck failed!\n";
      return -1;
    }
  } else {
    if(rank == 0) {
      if(rdJac) printf("Optimal objective 1: %22.14e. Solver status: %d\n", obj_value1, status1);
      if(rdJac) printf("Optimal objective 2: %22.14e. Solver status: %d\n", obj_value2, status2);
      printf("Optimal objective 3: %22.14e. Solver status: %d\n", obj_value3, status3);
      if(rdJac) printf("Optimal objective 4: %22.14e. Solver status: %d\n", obj_value4, status4);
    }
  }
#ifdef HIOP_USE_MAGMA
  magma_finalize();
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  std::cout << "Return successful!\n";
  return 0;
}
