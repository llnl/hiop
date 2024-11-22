#include "NlpDenseConsEx4.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(double obj_value);

static bool parse_arguments(int argc, char** argv, bool& self_check)
{
  self_check = false;
  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 2:  // 1 arguments
    {
      if(std::string(argv[1]) == "-selfcheck") {
        self_check = true;
      }
    } break;
    default:
      return false;  // 2 or more arguments
  }

  return true;
};

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a tiny concave problem.\n", exeName);
  printf("Usage: \n");
  printf("  '$ %s -selfcheck'\n", exeName);
  printf("Arguments:\n");
  printf("  '-selfcheck': compares the optimal objective with a previously saved value. [optional]\n");
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

  if(!parse_arguments(argc, argv, selfCheck)) {
    usage(argv[0]);
    return 1;
  }

  DenseConsEx4 nlp_interface;
  // if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(nlp_interface);
  // if(rank==0) printf("nlp formulation created\n");

  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetNumericValue("mu0", 0.1);

  hiopAlgFilterIPM solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  if(status < 0) {
    if(rank == 0) printf("solver returned negative solve status: %d (with objective is %18.12e)\n", status, obj_value);
    return -1;
  }

  // this is used for "regression" testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(obj_value)) {
      return -1;
    }
  } else {
    if(rank == 0) {
      printf("Optimal objective: %22.14e. Solver status: %d\n", obj_value, status);
    }
  }

  return 0;
}

static bool self_check(double objval)
{
  const double objval_saved = -3.32231409044575e+02;

#define relerr 1e-6
  if(fabs((objval_saved - objval) / (1 + objval_saved)) > relerr) {
    printf("selfcheck failure. Objective (%18.12e) does not agree (%d digits) with the saved value (%18.12e).\n",
           objval,
           -(int)log10(relerr),
           objval_saved);
    return false;
  } else {
    printf("selfcheck success (%d digits)\n", -(int)log10(relerr));
  }
  return true;
}
