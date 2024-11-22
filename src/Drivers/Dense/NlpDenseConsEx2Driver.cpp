#include "NlpDenseConsEx2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);
static bool self_check_uncon(size_type n, double obj_value);

static bool parse_arguments(int argc, char** argv, size_type& n, bool& self_check, bool& no_con)
{
  self_check = false;
  no_con = false;
  n = 50000;
  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 4:  // 3 arguments
    {
      if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;
      }
    }
    case 3:  // 2 arguments
    {
      if(std::string(argv[2]) == "-unconstrained") {
        no_con = true;
      } else if(std::string(argv[2]) == "-selfcheck") {
        self_check = true;
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
      return false;  // 4 or more arguments
  }

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size -unconstrained -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  'problem_size': number of decision variables [optional, default is 50k]\n");
  printf("  '-unconstrained': unconstrainted optimization problem [optional]\n");
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
  bool unconstrained;
  size_type n;
  if(!parse_arguments(argc, argv, n, selfCheck, unconstrained)) {
    usage(argv[0]);
    return 1;
  }

  DenseConsEx2 nlp_interface(n, unconstrained);
  // if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(nlp_interface);
  // if(rank==0) printf("nlp formulation created\n");

  hiopAlgFilterIPM solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();

  if(status < 0) {
    if(rank == 0) printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  // this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!unconstrained) {
      if(!self_check(n, obj_value)) {
        return -1;
      }
    } else {
      if(!self_check_uncon(n, obj_value)) {
        return -1;
      }
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

static bool self_check(size_type n, double objval)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000};
  const double objval_saved[] = {1.56251020819349e-02, 1.56251019995139e-02, 1.56251028980352e-02};

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

static bool self_check_uncon(size_type n, double objval)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000};
  const double objval_saved[] = {1.56250004019985e-02, 1.56250035348275e-02, 1.56250304912460e-02};

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