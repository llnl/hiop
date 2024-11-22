#include "NlpPriDecEx2Sparse.hpp"
#include "hiopAlgPrimalDecomp.hpp"

#include <cstdlib>
#include <string>

#ifdef HIOP_USE_MPI
#include "mpi.h"

#else
#ifndef MPI_Comm
#define MPI_Comm int
#endif

#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#endif

/**
 * Driver for example 9 that illustrates the use of hiop::hiopAlgPrimalDecomposition
 *
 * @note This example is built only when HIOP_USE_MPI and HIOP_SPARSE is enabled during cmake build
 * and require at least two MPI ranks in MPI_COMM_WORLD.
 *
 * The recourse problems are implemented with HIOP Sparse class.
 */

using namespace hiop;

// checking the solution of a given nx and S
static bool self_check(int nx, int S, double obj_value);
static bool parse_arguments(int argc, char** argv, bool& self_check, int& nx, int& S)
{
  self_check = false;
  nx = 20;
  S = 5;

  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 4:  // 3 arguments
    {
      if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;
        nx = std::atoi(argv[1]);
        S = std::atof(argv[2]);
        if(S < 3) {
          S = 4;
        }
        if(nx <= 0) {
          return false;
        }
      } else {
        return false;
      }
    }
    case 3:  // 2 arguments
    {
      nx = atoi(argv[1]);
      if(nx <= 0) return false;
      S = atoi(argv[2]);
      if(S < 3) S = 4;
    }
    case 2:  // 1 argument
    {
      if(std::string(argv[1]) == "-selfcheck") {
        self_check = true;
      } else {
        nx = atoi(argv[1]);
        if(nx <= 0) {
          return false;
        }
      }
    } break;
    default:
      return false;  // 4 or more arguments not supported
  }

  if(self_check && nx != 20 && S != 5) {
    printf(
        "Error: incorrect input parameters: '-selfcheck' must be used with predefined "
        "values for input  parameters, nx=20 S=5\n");
    return false;
  }

  return true;
};

static void usage(const char* exeName)
{
  printf(
      "HiOp driver %s that solves a nonconvex synthetic problem of variable size in the "
      "primal decomposition formulation. )\n",
      exeName);
  printf("Usage: \n");
  printf("  '$ %s nx S -selfcheck '\n", exeName);
  printf("Arguments, all integers, except strings '-selfcheck' \n");
  printf("  'nx': # of base case variables [default 20, optional, nonnegative integer].\n");
  printf("  'S': # of recourse/contingency problems [default 5, optional, nonnegative integer].\n");
  printf(
      "  '-selfcheck': compares the optimal objective with nx being 20 and "
      "S being 5 (these two exact values must be passed as arguments). [optional]\n");
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int comm_size;
  int ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  assert(MPI_SUCCESS == ierr);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(MPI_SUCCESS == ierr);
#endif

  int nx = 20;  // nx == ny for this problem
  int nS = 5;   // number of \xi
  int S = 5;

  bool selfCheck;

  if(!parse_arguments(argc, argv, selfCheck, nx, S)) {
    usage(argv[0]);
    return 1;
  }

  PriDecMasterProbleEx2Sparse pridec_problem(nx, nx, nS, S);
  hiop::hiopAlgPrimalDecomposition pridec_solver(&pridec_problem, MPI_COMM_WORLD);
  pridec_solver.set_initial_alpha_ratio(0.5);
  pridec_solver.set_alpha_min(0.3);
  // pridec_solver.set_local_accum("true");
  // pridec_solver.set_tolerance(1e-6);
  // pridec_solver.set_max_iteration(5);
  auto status = pridec_solver.run();

  int it = pridec_solver.getNumIterations();
  printf("number of iterations %d\n", it);
  if(status != Solve_Success) {
    if(rank == 0) {
      printf("Solve was NOT successfull.");
    }
  } else {
    if(rank == 0) {
      printf("Solve was successfull. Optimal value: %12.5e\n", pridec_solver.getObjective());
    }
  }

  if(selfCheck) {
    if(rank == 0) {
      if(!self_check(nx, S, pridec_solver.getObjective())) {
        return -1;
      }
    }
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  // printf("Returned successfully from driver! Rank=%d\n", rank);
  return 0;
}

static bool self_check(int nx, int S, double obj_value)
{
  double obj_true = 0.2633379371706;
  double err = 1e-5;
  if(fabs((obj_value)-obj_true) < 1e-5) {
    printf("selfcheck success (error less than %18.12e), objective value is %18.12e \n", err, obj_value);
    return true;
  } else {
    printf("selfcheck failure. Objective (%18.12e) does not agree  with the saved value (%18.12e) for nx=%d,S=%d.\n",
           obj_value,
           obj_true,
           nx,
           S);
    return false;
  }
  return true;
}
