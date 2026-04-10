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
import os
import socket
import threading
from hiopbbpy.utils import EvaluationManager, is_running_with_mpi
from concurrent.futures import ThreadPoolExecutor

def _fn_for_test(x, sleep_time=0.1, slow_first=False, driver_rank=0):
    hostname = socket.gethostname()
    pid = os.getpid()

    if slow_first and x == 0:
        actual_sleep = 3 * sleep_time
    else:
        actual_sleep = sleep_time

    print(
        f"rank={driver_rank} pid={pid} host={hostname}: processing x={x}",
        flush=True,
    )

    time.sleep(actual_sleep)
    return x * x

if __name__ == "__main__":
  # Arguments for command line
  parser = argparse.ArgumentParser(
    description="Execute n function calls with t duration.",
    epilog="To properly run the example with mpi4py, use: env MPI4PY_FUTURES_MAX_WORKERS=<N> mpiexec -n 1 python evaluation_manager.py",
  )
  parser.add_argument("-n", type=int, default=20, help="Number of tasks to execute")
  parser.add_argument(
      "-t", "--sleep_time", type=float, default=1, help="Sleep time for each task"
  )
  parser.add_argument(
      "-p",
      "--profile",
      action="store_true",
      help="Enable profiling",
  )
  parser.add_argument(
    "-s",
    "--slow_first",
    action="store_true",
    help="Make the first task slower (3x sleep time)",
  )
  args = parser.parse_args()

  # Set up logging
  logging.basicConfig(level=logging.INFO)

  # Create manager
  cpu_executor = ThreadPoolExecutor()
  manager = EvaluationManager(
      executor=cpu_executor,
      profiling=args.profile,
      task_name="CI_TASK"
  )

  # Submit tasks
  t0 = time.perf_counter()
  manager.submit_tasks(
      _fn_for_test,
      [i for i in range(args.n)],
      sleep_time=args.sleep_time,
      slow_first=args.slow_first,
  )

  # Do some other work while tasks are running
  for i in range(5):
    print("Doing other work (Master)", flush=True)
    time.sleep(args.sleep_time)
  print("Doing other work (Master) --- Done.")

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
  sys.exit(0)
