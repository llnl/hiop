#include "NlpDenseConsEx3.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);

static bool parse_arguments(int argc, char** argv, size_type& n, bool& self_check)
{
  self_check = false;
  n = 50000;
  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 3:  // 2 arguments
    {
      if(std::string(argv[2]) == "-selfcheck") {
        self_check = true;
      } else {
        n = std::atoi(argv[2]);
        if(n <= 0) {
          return false;
        }
      }
    }
    case 2:  // 1 argument
    {
      if(std::string(argv[1]) == "-selfcheck") {
        self_check = true;
      } else {
        n = std::atoi(argv[1]);
        if(n <= 0) {
          return false;
        }
      }
    } break;
    default:
      return false;  // 3 or more arguments
  }

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size with fixed variables.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50k]\n");
  printf(
      "  '-selfcheck': compares the optimal objective with a previously saved value for the problem specified by "
      "'problem_size'. [optional]\n");
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(MPI_SUCCESS == ierr);
  // if(0==rank) printf("Support for MPI is enabled\n");
#endif
  bool selfCheck;
  size_type n;
  if(!parse_arguments(argc, argv, n, selfCheck)) {
    usage(argv[0]);
    return 1;
  }

  double obj_value;
  bool do_second_round = true;

  hiopSolveStatus status;

  DenseConsEx3 nlp_interface(n);

  hiopNlpDenseConstraints nlp(nlp_interface);

  // relax var/con bounds before solving the problem
  nlp.options->SetNumericValue("bound_relax_perturb", 1e-10);

  // keep multipliers small
  nlp.options->SetStringValue("elastic_mode", "correct_it_adjust_bound");
  nlp.options->SetNumericValue("elastic_mode_bound_relax_final", 1e-12);
  nlp.options->SetNumericValue("elastic_mode_bound_relax_initial", 1e-2);

  // quasi-Newton tolerance is smaller than the default
  nlp.options->SetNumericValue("tolerance", 1e-6);

  {
    hiopAlgFilterIPM solver(&nlp);
    nlp.options->SetStringValue("fixed_var", "remove");
    status = solver.run();
    obj_value = solver.getObjective();

    // change options and resolve
    nlp.options->SetStringValue("fixed_var", "relax");
    status = solver.run();
    obj_value = solver.getObjective();
  }
  // do the same as above but force deallocation of the solver
  if(do_second_round) {
    {
      hiopAlgFilterIPM solver(&nlp);
      nlp.options->SetStringValue("fixed_var", "remove");
      status = solver.run();
      obj_value = solver.getObjective();
    }
    {
      hiopAlgFilterIPM solver(&nlp);
      nlp.options->SetStringValue("fixed_var", "relax");
      status = solver.run();
      obj_value = solver.getObjective();
    }
  }

  if(status < 0) {
    if(rank == 0) printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  // this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(n, obj_value)) return -1;
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

static bool self_check(size_type n, double objval)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000};
  const double objval_saved[] = {2.057860427672e+00, 2.02870382737020e+01, 2.02578703828247e+02};

#define relerr 1e-6
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
