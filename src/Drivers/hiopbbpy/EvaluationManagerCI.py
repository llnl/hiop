"""
Test script demonstrating EvaluationManager with multiple executor types.
Supports both intra-node and inter-node parallelism.

Usage examples:
  Single-node with threads:    python EvaluationManagerCI.py -e thread -w 4
  Single-node with processes:  python EvaluationManagerCI.py -e process -w 4
  Multi-node with MPI:         mpiexec -n 8 python EvaluationManagerCI.py -e mpi

For MPI, use N = number of total processes across all nodes.
The script will use rank 0 as master and ranks 1-(N-1) as workers.

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
from hiopbbpy.utils import EvaluationManager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  Single-node with threads:     python EvaluationManagerCI.py -e thread -w 4
  Single-node with processes:   python EvaluationManagerCI.py -e process -w 4
  Multi-node with MPI:          mpiexec -n <N> python EvaluationManagerCI.py -e mpi

For MPI, use N = number of nodes * processes_per_node.
The script uses rank 0 as master and ranks 1-(N-1) as workers.
    """,
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
  parser.add_argument(
    "-e",
    "--executor",
    type=str,
    default="thread",
    choices=["thread", "process", "mpi"],
    help="Executor type: thread (ThreadPoolExecutor), process (ProcessPoolExecutor), or mpi (MPIPoolExecutor)",
  )
  parser.add_argument(
    "-w",
    "--max_workers",
    type=int,
    default=None,
    help="Maximum number of workers (for thread/process executors)",
  )
  args = parser.parse_args()

  # Set up logging
  logging.basicConfig(level=logging.INFO)

  # Create executor based on user choice
  if args.executor == "thread":
    cpu_executor = ThreadPoolExecutor(max_workers=args.max_workers)
    executor_name = "ThreadPool"
  elif args.executor == "process":
    cpu_executor = ProcessPoolExecutor(max_workers=args.max_workers)
    executor_name = "ProcessPool"
  elif args.executor == "mpi":
    try:
      from mpi4py import MPI
      from mpi4py.futures import MPIPoolExecutor

      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()

      if size < 2:
        print("ERROR: MPI executor requires at least 2 processes (1 master + 1 worker)")
        print("Run with: mpiexec -n <N> python EvaluationManagerCI.py -e mpi")
        sys.exit(1)

      # Only rank 0 will run the main logic
      if rank != 0:
        # Worker ranks just need to participate in the MPIPoolExecutor
        # They will block in MPIPoolExecutor() and process tasks
        MPIPoolExecutor()
        sys.exit(0)

      cpu_executor = MPIPoolExecutor()
      executor_name = f"MPIPool (rank={rank}/{size}, {size-1} workers)"
      print(f"MPI executor initialized: {size} total processes, {size-1} workers across nodes")

    except ImportError:
      print("ERROR: mpi4py not installed. Install with: pip install mpi4py")
      sys.exit(1)

  # Create manager (only rank 0 reaches here for MPI)
  manager = EvaluationManager(
      executor=cpu_executor,
      profiling=args.profile,
      task_name=f"CI_TASK_{executor_name}"
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
