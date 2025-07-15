#include "NlpDenseConsRajaEx2.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cstdlib>
#include <string>
#include <cmath>
#include <cstdio>    // for printf
#include <iostream>  // for std::cout

#ifdef HIOP_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace hiop;

static bool self_check(int rank, size_type n, double obj_value);
static bool self_check_uncon(int rank, size_type n, double obj_value);

static bool parse_arguments(int argc, char** argv,
                            size_type& n,
                            bool& do_selfcheck,
                            bool& unconstrained)
{
  do_selfcheck  = false;
  unconstrained = false;
  n             = 50000;
  switch(argc) {
    case 1: return true;
    case 4:
      if(std::string(argv[3]) == "-selfcheck") do_selfcheck = true;
      // fallthrough
    case 3:
      if(std::string(argv[2]) == "-unconstrained") unconstrained = true;
      else if(std::string(argv[2]) == "-selfcheck")      do_selfcheck = true;
      // fallthrough
    case 2:
      n = std::atoi(argv[1]);
      return (n > 0);
    default: return false;
  }
}

static void usage(const char* exeName)
{
  std::printf("Usage: %s [problem_size] [-unconstrained] [-selfcheck]\n", exeName);
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

#ifdef HIOP_USE_CUDA
  int dev{-1};
  cudaError_t err = cudaGetDevice(&dev);
  if(rank==0) {
    if(err==cudaSuccess) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, dev);
      std::cout << "[CUDA CHECK] Using device " << dev
                << ": " << prop.name << "\n";
    } else {
      std::cout << "[CUDA CHECK] cudaGetDevice failed: "
                << cudaGetErrorString(err) << "\n";
    }
  }
#endif

  bool do_selfcheck, unconstrained;
  size_type n;
  if(!parse_arguments(argc, argv, n, do_selfcheck, unconstrained)) {
    usage(argv[0]);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  DenseConsRajaEx2     nlp_interface(n, unconstrained);
  hiopNlpDenseConstraints nlp(nlp_interface);
  nlp.options->SetString("mem_space", "device");
  hiopAlgFilterIPM       solver(&nlp);
  hiopSolveStatus        status = solver.run();
  double                 objv   = solver.getObjective();

  if(status < 0) {
    if(rank==0) std::printf(
      "Solver returned negative status %d (obj=%18.12e)\n",
      status, objv);
#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if(do_selfcheck) {
    bool ok = unconstrained
            ? self_check_uncon(rank, n, objv)
            : self_check(rank,     n, objv);
    if(!ok) {
#ifdef HIOP_USE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
  } else if(rank==0) {
    std::printf("Optimal objective: %18.12e. Status: %d\n",
                objv, status);
  }

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}

static bool self_check(int rank, size_type n, double objv)
{
  const size_type n_saved[] = {  500,   5000,   50000 };
  const double obj_saved[] = { 
    1.56251020819349e-02, 
    1.56251019995139e-02, 
    1.56251028980352e-02 
  };
  const double relerr = 1e-6;
  for(int i=0;i<3;i++) {
    if(n_saved[i]==n) {
      double err = std::fabs(obj_saved[i] - objv)/(1 + obj_saved[i]);
      if(err > relerr) {
        if(rank==0) std::printf(
          "selfcheck failure for n=%lld: got %18.12e vs %18.12e\n",
          (long long)n, objv, obj_saved[i]);
        return false;
      }
      if(rank==0) std::printf(
        "selfcheck success (%d digits)\n",
        (int)(-std::log10(relerr)));
      return true;
    }
  }
  if(rank==0) std::printf(
    "no saved result for n=%lld, got %18.12e\n",
    (long long)n, objv);
  return false;
}

static bool self_check_uncon(int rank, size_type n, double objv)
{
  const size_type n_saved[] = {  500,   5000,   50000 };
  const double    obj_saved[] = { 1.5625000e-2, 1.5625004e-2, 1.5625030e-2 };
  const double relerr = 1e-6;
  for(int i=0;i<3;i++) {
    if(n_saved[i]==n) {
      double err = std::fabs(obj_saved[i] - objv)/(1 + obj_saved[i]);
      if(err > relerr) {
        if(rank==0) std::printf(
          "selfcheck failure (uncon) for n=%lld: got %18.12e vs %18.12e\n",
          (long long)n, objv, obj_saved[i]);
        return false;
      }
      if(rank==0) std::printf(
        "selfcheck success (uncon, %d digits)\n",
        (int)(-std::log10(relerr)));
      return true;
    }
  }
  if(rank==0) std::printf(
    "no saved result (uncon) for n=%lld, got %18.12e\n",
    (long long)n, objv);
  return false;
}