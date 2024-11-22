#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#ifdef HIOP_USE_MPI
#include "mpi.h"
#endif

#include <vector>

using namespace std;

void net_benchmark(const size_type baseDim);

static const size_type default_num_doubles_per_rank = 32768;
int main(int argc, char **argv)
{
  int nranks = 1;
#ifdef HIOP_USE_MPI
  MPI_Init(&argc, &argv);
  int err = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  assert(MPI_SUCCESS == err);
#endif
  size_type base_dim = nranks * default_num_doubles_per_rank;
  if(argc > 1) base_dim = nranks * atol(argv[1]);

  net_benchmark(base_dim);

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}

const static int NUM_REPETES = 100;
const static int NUM_REDUCES = 8;
const static int NUM_TESTS = 5;
const static int TEST_X_SIZE = 2;
void net_benchmark(const size_type baseDim)
{
#ifndef HIOP_USE_MPI
  printf("non-MPI build, skipping network benchmark\n");
#else

  int nranks, my_rank;
  int err = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  assert(MPI_SUCCESS == err);
  err = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  assert(MPI_SUCCESS == err);

  if(0 == my_rank) printf("Network benchmark: base dimension is %lu\n", baseDim);

  vector<vector<double> > results(NUM_TESTS, vector<double>(NUM_REPETES, 0.));

  for(int r = 0; r < NUM_REPETES; r++) {
    size_type loc_size = baseDim / nranks;
    for(int t = 1; t <= NUM_TESTS; t++) {
      double* bufSend = new double[loc_size];
      double* bufRecv = new double[loc_size];

      for(int i = 0; i < loc_size; i++) bufSend[i] = (1e-6 + i) / (2. * loc_size);

      double tm_start = MPI_Wtime();
      for(int i = 0; i < NUM_REDUCES; i++) {
        err = MPI_Allreduce(bufSend, bufRecv, loc_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        assert(MPI_SUCCESS == err);
      }
      double tm_end = MPI_Wtime();
      if(0 == my_rank) printf("  buffer size %10lu reduced in %.10f seconds\n", loc_size, tm_end - tm_start);
      results[t - 1][r] = tm_end - tm_start;

      // increase loc_size for the next benchmark
      loc_size = loc_size * TEST_X_SIZE;

      delete[] bufSend;
      delete[] bufRecv;
    }
  }

  // outputing
  if(0 == my_rank) {
    printf("\nSummary: MPI ranks=%d baseDim=%lu X_size=%d TESTS=%d REDUCES=%d REPETITIONS=%d\n",
           nranks,
           baseDim,
           TEST_X_SIZE,
           NUM_TESTS,
           NUM_REDUCES,
           NUM_REPETES);
    size_type loc_size = baseDim / nranks;
    for(int t = 0; t < NUM_TESTS; t++) {
      double mean = 0.;
      for(int r = 0; r < NUM_REPETES; r++) mean += results[t][r];
      mean /= NUM_REPETES;
      double stdd = 0.;
      for(int r = 0; r < NUM_REPETES; r++) stdd += (results[t][r] - mean) * (results[t][r] - mean);
      stdd = sqrt(stdd / NUM_REPETES);

      printf("  buffer size %10lu reduced in: mean %.10f seconds stddev %.3f percent\n", loc_size, mean, stdd / mean * 100);
      loc_size = loc_size * TEST_X_SIZE;
    }
  }
#endif
}
