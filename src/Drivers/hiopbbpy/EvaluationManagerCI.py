"""
This is a class to manage function evaluations using multiple parallel executors.
It supports both intra-node and inter-node parallelism.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Weslley S Pereira <wdasilv@nrel.gov>
"""
import logging
import argparse
import time
import sys
from hiopbbpy.utils import EvaluationManager, is_running_with_mpi
from concurrent.futures import ThreadPoolExecutor

def _fn_for_test(x, sleep_time=0.1):
  if is_running_with_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Rank {rank}: Processing {x}")
  time.sleep(sleep_time)  # Simulate some work
  return x * x



if __name__ == "__main__":
  # Arguments for command line
  parser = argparse.ArgumentParser(
    description="Execute n function calls with t duration.",
    epilog="To properly run the example with mpi4py, use: env MPI4PY_FUTURES_MAX_WORKERS=<N> mpiexec -n 1 python evaluation_manager.py",
  )
  parser.add_argument("-n", type=int, default=100, help="Number of tasks to execute")
  parser.add_argument(
      "-t", "--sleep_time", type=float, default=1, help="Sleep time for each task"
  )
  args = parser.parse_args()

  if is_running_with_mpi():
    executor_type = "mpi"
  else:
    executor_type = "cpu"

  # Set up logging
  logging.basicConfig(level=logging.INFO)

  # Create manager
  cpu_executor = ThreadPoolExecutor()
  manager = EvaluationManager(cpu_executor=cpu_executor)

  # Submit tasks to the manager
  t0 = time.perf_counter()
  manager.submit_tasks(
      _fn_for_test,
      [i for i in range(args.n)],
      execute_at=executor_type,
      sleep_time=args.sleep_time,
  )

  # Do some other work while tasks are running
  print("Doing other work", end="", flush=True)
  for i in range(5):
    print(".", end="", flush=True)
    time.sleep(args.sleep_time)
  print(" Done.")

  # Wait for all tasks to complete
  print("Waiting for tasks to complete...")
  manager.sync()
  t1 = time.perf_counter()

  # Retrieve and show results
  X, F = manager.retrieve_results()
  print("X:", X)
  print("F:", F)
  print(f"Total time: {t1 - t0:.2f} seconds")

  # Clean up
  del manager
  retval = 0
  sys.exit(retval)
