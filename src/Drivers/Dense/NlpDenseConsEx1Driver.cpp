#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#include "NlpDenseConsEx1.hpp"

#include <cstdlib>
#include <string>

#ifdef HIOP_USE_AXOM
#include <axom/sidre/core/DataStore.hpp>
#include <axom/sidre/core/Group.hpp>
#include <axom/sidre/core/View.hpp>
#include <axom/sidre/spio/IOManager.hpp>
using namespace axom;
#endif

using namespace hiop;

static bool self_check(size_type n, double obj_value);
#ifdef HIOP_USE_AXOM
static bool do_load_checkpoint_test(const size_type& mesh_size, const double& ratio, const double& obj_val_expected);
#endif
static bool parse_arguments(int argc, char** argv, size_type& n, double& distortion_ratio, bool& self_check)
{
  n = 20000;
  distortion_ratio = 1.;
  self_check = false;  // default options

  switch(argc) {
    case 1:
      // no arguments
      return true;
      break;
    case 4:  // 3 arguments - -selfcheck expected
    {
      if(std::string(argv[3]) == "-selfcheck") {
        self_check = true;
      } else {
        return false;
      }
    }
    case 3:  // 2 arguments: pick up distortion ratio here
    {
      distortion_ratio = atof(argv[2]);
    }
    case 2:  // 1 argument
    {
      n = std::atoi(argv[1]);
    } break;
    default:
      return false;  // 3 or more arguments
  }

  if(n <= 0) return false;
  if(distortion_ratio <= 1e-8 || distortion_ratio > 1.) return false;
  return true;
};

static void usage(const char* exeName)
{
  printf(
      "hiOp driver '%s' that solves a synthetic infinite dimensional problem of variable size. A 1D mesh is created by the "
      "example, and the size and the distortion of the mesh can be specified as options to this executable. The distortion "
      "of the mesh is the ratio of the smallest element and the largest element in the mesh.\n",
      exeName);
  printf("Usage: \n");
  printf("  '$ %s problem_size mesh_distortion_ratio -selfcheck'\n", exeName);
  printf("Arguments (specify in the order above): \n");
  printf("  'problem_size': number of decision variables [optional, default is 20k]\n");
  printf("  'dist_ratio': mesh distortion ratio, see above; a number in (0,1)  [optional, default 1.0]\n");
  printf(
      "  '-selfcheck': compares the optimal objective with a previously saved value for the problem specified by "
      "'problem_size'. [optional]\n");
}

int main(int argc, char** argv)
{
  int rank = 0;
#ifdef HIOP_USE_MPI
  int numRanks = 1;
  int err;
  err = MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS == err);
  err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(MPI_SUCCESS == err);
  err = MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  assert(MPI_SUCCESS == err);
  if(0 == rank) {
    printf("Support for MPI is enabled\n");
  }
#endif
  bool selfCheck;
  size_type mesh_size;
  double ratio;
  double objective = 0.;
  if(!parse_arguments(argc, argv, mesh_size, ratio, selfCheck)) {
    usage(argv[0]);
    return 1;
  }

  DenseConsEx1 problem(mesh_size, ratio);
  hiop::hiopNlpDenseConstraints nlp(problem);

  hiop::hiopAlgFilterIPM solver(&nlp);
  problem.set_solver(&solver);

  hiop::hiopSolveStatus status = solver.run();
  objective = solver.getObjective();

  // this is used for testing when the driver is called with -selfcheck
  if(selfCheck) {
    if(!self_check(mesh_size, objective)) return -1;
  } else {
    if(rank == 0) {
      printf("Optimal objective: %22.14e. Solver status: %d. Number of iterations: %d\n",
             objective,
             status,
             solver.getNumIterations());
    }
  }

  if(0 == rank) {
    printf("Objective: %18.12e\n", objective);
  }

#ifdef HIOP_USE_AXOM
  // example/test for HiOp's load checkpoint API.
  if(!do_load_checkpoint_test(mesh_size, ratio, objective)) {
    if(rank == 0) {
      printf("Load checkpoint and restart test failed.");
    }
    return -1;
  }
#endif
#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

static bool self_check(size_type n, double objval)
{
#define num_n_saved 3  // keep this is sync with n_saved and objval_saved
  const size_type n_saved[] = {500, 5000, 50000};
  const double objval_saved[] = {8.6156700e-2, 8.6156106e-02, 8.6161001e-02};

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

#ifdef HIOP_USE_AXOM
/**
 * An illustration on how to use load_state_from_sidre_group API method of HiOp's algorithm class.
 *
 *
 */
static bool do_load_checkpoint_test(const size_type& mesh_size, const double& ratio, const double& obj_val_expected)
{
  // Pretend this is new job and recreate the HiOp objects.
  DenseConsEx1 problem(mesh_size, ratio);
  hiop::hiopNlpDenseConstraints nlp(problem);

  hiop::hiopAlgFilterIPM solver(&nlp);

  //
  // example of how to use load_state_sidre_group to warm-start
  //

  // Supposedly, the user code should have the group in hand before asking HiOp to load from it.
  // We will manufacture it by loading a sidre checkpoint file. Here the checkpoint file
  //"hiop_state_ex1.root" was created from the interface class' iterate_callback method
  //(saved every 5 iterations)
  sidre::DataStore ds;

  try {
    sidre::IOManager reader(MPI_COMM_WORLD);
    reader.read(ds.getRoot(), "hiop_state_ex1.root", false);
  } catch(std::exception& e) {
    printf("Failed to read checkpoint file. Error: [%s]", e.what());
    return false;
  }

  // the actual API call
  try {
    const sidre::Group* group = ds.getRoot()->getGroup("HiOp quasi-Newton alg state");
    solver.load_state_from_sidre_group(*group);
  } catch(std::runtime_error& e) {
    printf("Failed to load from sidre::group. Error: [%s]", e.what());
    return false;
  }

  hiop::hiopSolveStatus status = solver.run();
  double obj_val = solver.getObjective();
  if(obj_val != obj_val_expected) {
    return false;
  }
  return true;
}
#endif  // HIOP_USE_AXOM
