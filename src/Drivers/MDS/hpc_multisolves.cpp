#include "NlpMdsEx1.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"
#endif

#include "mpi.h"

#include <thread>  // std::this_thread::sleep_for
#include <chrono>  // std::chrono::seconds

#include "hiopTimer.hpp"

using namespace hiop;

/** The driver performs multiple solves per MPI process using MDS Ex1
 *
 * Intended to be used to test intra-node CPU cores affinity or GPU streams multiprocessing
 *
 *
 * Usage with bsub,  for example, on Summit:  see end of file for a submission script
 */
int main(int argc, char* argv[])
{
  int ret;
  ret = MPI_Init(&argc, &argv);
  assert(ret == MPI_SUCCESS);
  if(MPI_SUCCESS != ret) {
    printf("MPI_Init failed\n");
    return -1;
  }

  hiopTimer glob_timer, t;
  glob_timer.start();

  int my_rank = 0, comm_size;
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  assert(ret == MPI_SUCCESS);

  ret = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  assert(ret == MPI_SUCCESS);

  const int num_probs_per_rank = 5;
  const int n_de = 2000;
  const int n_sp = 2 * n_de;

  for(int i = 0; i < num_probs_per_rank; i++) {
    t.start();
    printf("[driver] Rank %d solves problem %d\n", my_rank, (i + 1));
    fflush(stdout);

    double obj_value = -1e+20;
    [[maybe_unused]] hiopSolveStatus status;

    // user's NLP -> implementation of hiop::hiopInterfaceMDS
    MdsEx1* my_nlp = new MdsEx1(n_sp, n_de);

    hiopNlpMDS nlp(*my_nlp);
    hiopAlgFilterIPMNewton solver(&nlp);
    status = solver.run();
    obj_value = solver.getObjective();

    delete my_nlp;

    t.stop();
    printf("[driver] Rank %d solved problem %d (obj=%12.5e) in %g sec\n", my_rank, (i + 1), obj_value, t.getElapsedTime());
    fflush(stdout);
  }

  glob_timer.stop();
  double tmElapsed = glob_timer.getElapsedTime();

  MPI_Barrier(MPI_COMM_WORLD);
  std::this_thread::sleep_for(std::chrono::milliseconds((1 + my_rank) * 100));

  printf("[driver] Rank %d finished solves in %g seconds\n", my_rank, tmElapsed);
  fflush(stdout);

  double tmAvg, stdDevTm, aux;
  ret = MPI_Allreduce(&tmElapsed, &tmAvg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  assert(ret == MPI_SUCCESS);
  tmAvg /= comm_size;

  if(comm_size > 1) {
    aux = (tmElapsed - tmAvg) * (tmElapsed - tmAvg) / (comm_size - 1);

    ret = MPI_Allreduce(&aux, &stdDevTm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    assert(ret == MPI_SUCCESS);
    stdDevTm = sqrt(stdDevTm);
  } else {
    aux = 0.;
  }
  if(0 == my_rank) {
    printf("\n\nSummary: average time %g sec, std dev %.2f percent \n\n", tmAvg, 100 * stdDevTm / tmAvg);
  }

  MPI_Finalize();
  return 0;
}

/* -- BSUB submission file --

# Begin LSF Directives
#BSUB -P csc359
#BSUB -W 00:20
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J RunSim16mpiprs
#BSUB -o RunSim16mpiprs.%J
#BSUB -e RunSim16mpiprs.%J


export NVBLAS_CONFIG_FILE=/ccs/home/cpetra/work/projects/gocompet/specCpp/runs_wecc10k/nvblas.conf

jsrun -n 1 -c 16 -a 16 -g 1 -d packed -l GPU-CPU ./src/Drivers/hpc_multisolves.exe


*/
