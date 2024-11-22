#include "NlpSparseEx3.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value, hiopSolveStatus status);

static bool parse_arguments(int argc,
                            char** argv,
                            size_type& n,
                            double& scala_a,
                            bool& eq_feas,
                            bool& eq_infeas,
                            bool& ineq_feas,
                            bool& ineq_infeas,
                            bool& self_check)
{
  self_check = false;
  eq_feas = false;
  eq_infeas = false;
  ineq_feas = false;
  ineq_infeas = false;
  n = 50;
  scala_a = 1e-6;
  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 7:  // 6 arguments
    {
      if(std::string(argv[6]) == "-eq_feas") {
        eq_feas = true;
      } else if(std::string(argv[6]) == "-eq_infeas") {
        eq_infeas = true;
      } else if(std::string(argv[6]) == "-ineq_feas") {
        ineq_feas = true;
      } else if(std::string(argv[6]) == "-ineq_infeas") {
        ineq_infeas = true;
      } else if(std::string(argv[6]) == "-selfcheck") {
        self_check = true;
      }
    }
    case 6:  // 5 arguments
    {
      if(std::string(argv[5]) == "-eq_feas") {
        eq_feas = true;
      } else if(std::string(argv[5]) == "-eq_infeas") {
        eq_infeas = true;
      } else if(std::string(argv[5]) == "-ineq_feas") {
        ineq_feas = true;
      } else if(std::string(argv[5]) == "-ineq_infeas") {
        ineq_infeas = true;
      } else if(std::string(argv[5]) == "-selfcheck") {
        self_check = true;
      }
    }
    case 5:  // 4 arguments
    {
      if(std::string(argv[4]) == "-eq_feas") {
        eq_feas = true;
      } else if(std::string(argv[4]) == "-eq_infeas") {
        eq_infeas = true;
      } else if(std::string(argv[4]) == "-ineq_feas") {
        ineq_feas = true;
      } else if(std::string(argv[4]) == "-ineq_infeas") {
        ineq_infeas = true;
      } else if(std::string(argv[4]) == "-selfcheck") {
        self_check = true;
      }
    }
    case 4:  // 3 arguments
    {
      if(std::string(argv[3]) == "-eq_feas") {
        eq_feas = true;
      } else if(std::string(argv[3]) == "-eq_infeas") {
        eq_infeas = true;
      } else if(std::string(argv[3]) == "-ineq_feas") {
        ineq_feas = true;
      } else if(std::string(argv[3]) == "-ineq_infeas") {
        ineq_infeas = true;
      } else if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;
      }
    }
    case 3:  // 2 arguments
    {
      if(std::string(argv[2]) == "-eq_feas") {
        eq_feas = true;
      } else if(std::string(argv[2]) == "-eq_infeas") {
        eq_infeas = true;
      } else if(std::string(argv[2]) == "-ineq_feas") {
        ineq_feas = true;
      } else if(std::string(argv[2]) == "-ineq_infeas") {
        ineq_infeas = true;
      } else if(std::string(argv[2]) == "-selfcheck") {
        self_check = true;
      } else {
        scala_a = std::atof(argv[2]);
        assert(scala_a > 0.0);
      }
    }
    case 2:  // 1 argument
    {
      n = std::atoi(argv[1]);
      if(n <= 0) {
        return false;
      }
    } break;
    default:
      return false;  // 6 or more arguments
  }

  if(self_check) {
    scala_a = 1e-6;
    eq_feas = false;
    eq_infeas = false;
    ineq_feas = false;
    ineq_infeas = true;
  }
  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size scalar -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50]\n");
  printf("  'scala_a': small pertubation added to the inequality bounds [optional, default is 1e-6]\n");
  printf("  '-eq_feas': include feasible equality constraints, with rank deficient Jacobian [optional, default is no]\n");
  printf(
      "  '-eq_infeas': include infeasible equality constraints, with rank deficient Jacobian [optional, default is no]\n");
  printf("  '-eq_feas': include feasible inequality constraints, with rank deficient Jacobian [optional, default is no]\n");
  printf(
      "  '-eq_infeas': include infeasible inequality constraints, with rank deficient Jacobian [optional, default is no]\n");
  printf(
      "  '-selfcheck': compares the optimal objective with a previously saved value for the "
      "problem specified by 'problem_size'. `-ineq_feas` is set to `yes` internally. [optional]\n");
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  assert(MPI_SUCCESS == ierr);
  if(comm_size != 1) {
    printf(
        "[error] driver detected more than one rank but the driver should be run "
        "in serial only; will exit\n");
    MPI_Finalize();
    return 1;
  }
#endif

  size_type n;
  double scala_a;
  bool eq_feas;
  bool eq_infeas;
  bool ineq_feas;
  bool ineq_infeas;
  bool selfCheck;

  if(!parse_arguments(argc, argv, n, scala_a, eq_feas, eq_infeas, ineq_feas, ineq_infeas, selfCheck)) {
    usage(argv[0]);
    return 1;
  }

  SparseEx3 nlp_interface(n, scala_a, eq_feas, eq_infeas, ineq_feas, ineq_infeas);
  hiopNlpSparse nlp(nlp_interface);

  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "lsq");  // "lsq" or "linear"
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetNumericValue("mu0", 0.1);

  hiopAlgFilterIPMNewton solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();
  if(status < 0) {
    if(rank == 0) {
      printf("solver returned negative solve status: %d (obj. is %18.12e)\n", status, obj_value);
    }
    return -1;
  }

  // this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(n, obj_value, status)) {
      return -1;
    }
  } else {
    if(rank == 0) {
      printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
    }
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

static bool self_check(size_type n, double objval, hiopSolveStatus status)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {50, 500, 5000};
  const double objval_saved[] = {7.565566821330e+00, 8.284201575839e+01, 8.066106777964e+02};

#define relerr 1e-6
  if(status != hiopSolveStatus::Infeasible_Problem) {
    printf("selfcheck: driver expects an infeasible problem, but HiOp return code is %d \n", status);
    return false;
  }

  bool found = false;
  for(int it = 0; it < num_n_saved; it++) {
    if(n_saved[it] == n) {
      found = true;
      if(fabs((objval_saved[it] - objval) / (1 + objval_saved[it])) > relerr) {
        printf(
            "selfcheck failure. Objective (%18.12e) does not agree (%d digits) with the saved value (%18.12e) for n=%d.\n",
            objval,
            -(int)log10(relerr),
            objval_saved[it],
            n);
        return false;
      } else {
        printf("selfcheck success (%d digits)\n", -(int)log10(relerr));
      }
      break;
    }
  }

  if(!found) {
    printf("selfcheck: driver does not have the objective for n=%d saved. BTW, obj=%18.12e was obtained for this n.\n",
           n,
           objval);
    return false;
  }

  return true;
}
