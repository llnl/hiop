#include "NlpDenseConsRajaEx2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>
#include <cassert>

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
      return true;
    case 4: {
      if(std::string(argv[3]) == "-selfcheck") self_check = true;
    }
    case 3: {
      if(std::string(argv[2]) == "-unconstrained") no_con = true;
      else if(std::string(argv[2]) == "-selfcheck") self_check = true;
    }
    case 2: {
      n = std::atoi(argv[1]);
      if(n <= 0) return false;
    } break;
    default:
      return false;
  }
  return true;
}

static void usage(const char* exeName)
{
  printf("hiOp driver %s that solves a synthetic convex problem of variable size.\n", exeName);
  printf("Usage: %s problem_size -unconstrained -selfcheck\n", exeName);
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  bool do_selfcheck, unconstrained;
  size_type n;
  if(!parse_arguments(argc, argv, n, do_selfcheck, unconstrained)) {
    usage(argv[0]); return 1;
  }

  DenseConsRajaEx2 nlp_interface(n, unconstrained);
  hiopNlpDenseConstraints nlp(nlp_interface);
  hiopAlgFilterIPM solver(&nlp);
  hiopSolveStatus status = solver.run();
  double obj_value = solver.getObjective();

  if(status < 0) {
    if(rank == 0) printf("solver returned negative status %d (obj=%18.12e)\n", status, obj_value);
    return -1;
  }

  if(do_selfcheck) {
    if(!unconstrained) self_check(n, obj_value);
    else self_check_uncon(n, obj_value);
  } else if(rank == 0) {
    printf("Optimal objective: %22.14e. Status: %d\n", obj_value, status);
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}

static bool self_check(size_type n, double objval)
{
  const size_type n_saved[] = {500, 5000, 50000};
  const double obj_saved[] = {1.56251020819349e-02, 1.56251019995139e-02, 1.56251028980352e-02};
  const double relerr = 1e-6;
  for(int i=0; i<3; ++i) {
    if(n_saved[i]==n) {
      if(fabs((obj_saved[i]-objval)/(1+obj_saved[i]))>relerr) return false;
      else { printf("selfcheck success\n"); return true; }
    }
  }
  printf("no saved result for n=%d, got obj=%18.12e\n", n, objval);
  return false;
}

static bool self_check_uncon(size_type n, double objval)
{
  const size_type n_saved[] = {500, 5000, 50000};
  const double obj_saved[] = {1.56250004019985e-02, 1.56250035348275e-02, 1.56250304912460e-02};
  const double relerr = 1e-6;
  for(int i=0; i<3; ++i) {
    if(n_saved[i]==n) {
      if(fabs((obj_saved[i]-objval)/(1+obj_saved[i]))>relerr) return false;
      else { printf("selfcheck success\n"); return true; }
    }
  }
  printf("no saved result for n=%d, got obj=%18.12e\n", n, objval);
  return false;
}
