#include "NlpSparseEx1.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

static bool self_check(size_type n, double obj_value);

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

  // Problem size - small example for testing SuperLU
  size_type n = 50;
  double scal = 1.0;
  bool selfCheck = false;

  // Parse command line arguments
  if(argc >= 2) {
    n = std::atoi(argv[1]);
    if(n <= 0 || n < 3) {
      printf("Problem size must be >= 3. Using default n=50.\n");
      n = 50;
    }
  }
  if(argc >= 3) {
    if(std::string(argv[2]) == "-selfcheck") {
      selfCheck = true;
      scal = 1.0;
    } else {
      scal = std::atof(argv[2]);
    }
  }
  if(argc >= 4 && std::string(argv[3]) == "-selfcheck") {
    selfCheck = true;
  }

  printf("=========================================================\n");
  printf("HiOp Sparse Example 1 with SuperLU_DIST\n");
  printf("=========================================================\n");
  printf("Problem size: n=%d\n", n);
  printf("Scaling factor: scal=%g\n", scal);
  printf("\n");

#ifndef HIOP_USE_SUPERLU
  printf("ERROR: HiOp not built with SuperLU_DIST support!\n");
  printf("Please rebuild HiOp with -DHIOP_USE_SUPERLU=ON\n");
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 1;
#endif

  // Create the NLP problem
  SparseEx1 nlp_interface(n, scal);
  hiopNlpSparse nlp(nlp_interface);

  // Configure HiOp to use SuperLU_DIST
  printf("Configuring HiOp to use SuperLU_DIST sparse linear solver...\n");

  // Set basic options
  nlp.options->SetStringValue("Hessian", "analytical_exact");
  nlp.options->SetStringValue("duals_update_type", "linear");
  nlp.options->SetStringValue("compute_mode", "cpu");
  nlp.options->SetStringValue("KKTLinsys", "xdycyd");
  nlp.options->SetNumericValue("mu0", 0.1);

  // Configure SuperLU as the linear solver
  nlp.options->SetStringValue("linear_solver_sparse", "superlu");

  // CRITICAL: SuperLU does NOT provide inertia information
  // Must use inertia-free factorization acceptor
  nlp.options->SetStringValue("fact_acceptor", "inertia_free");

  // Increase verbosity to see SuperLU messages
  nlp.options->SetIntegerValue("verbosity_level", 3);

  printf("\n");
  printf("Key Settings:\n");
  printf("  - Linear solver: superlu\n");
  printf("  - Factorization acceptor: inertia_free (REQUIRED for SuperLU)\n");
  printf("  - Matching method: Auto-selected (SUMAC for GPU, SUITOR for CPU)\n");
  printf("\n");
  printf("Note: SuperLU_DIST does NOT provide inertia information.\n");
  printf("      The KKT system correctness is verified using the\n");
  printf("      'inertia_free' approach with iterative refinement.\n");
  printf("\n");

  // Create solver and run
  printf("Starting optimization...\n");
  printf("---------------------------------------------------------\n");
  hiopAlgFilterIPMNewton solver(&nlp);
  hiopSolveStatus status = solver.run();

  double obj_value = solver.getObjective();
  printf("---------------------------------------------------------\n");
  printf("\n");

  if(status < 0) {
    if(rank == 0) {
      printf("FAILED: Solver returned error status: %d\n", status);
      printf("        Objective value: %22.14e\n", obj_value);
    }
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  // Check results
  if(selfCheck) {
    printf("Running self-check...\n");
    if(!self_check(n, obj_value)) {
      printf("FAILED: Self-check failed\n");
#ifdef HIOP_USE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    printf("SUCCESS: Self-check passed\n");
  } else {
    printf("SUCCESS: Optimization converged\n");
    printf("         Optimal objective: %22.14e\n", obj_value);
    printf("         Solver status: %d\n", status);
  }

  printf("\n");
  printf("=========================================================\n");
  printf("SuperLU_DIST Test Completed Successfully\n");
  printf("=========================================================\n");

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

static bool self_check(size_type n, double objval)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {50, 500, 5000};
  const double objval_saved[] = {1.10351564683176e-01, 1.10351566513480e-01, 1.10351578644469e-01};

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
