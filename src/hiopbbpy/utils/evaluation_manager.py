"""
This is a class to manage function evaluations using multiple parallel executors.
It supports both intra-node and inter-node parallelism.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Weslley S Pereira <wdasilv@nrel.gov>
"""

import threading
import logging
import copy
import os
import time
import math
from concurrent.futures import ProcessPoolExecutor, CancelledError
from collections import deque


def is_running_with_mpi():
  """Returns True if the code is running in an MPI environment."""
  _MPI_RANK_ENV_VARS = [
      "OMPI_COMM_WORLD_RANK",  # Open MPI
      "PMI_RANK",              # MPICH, Intel MPI, Cray MPI
      "MPI_RANK",              # Intel MPI (sometimes)
      "MV2_COMM_WORLD_RANK",   # MVAPICH
  ]
  return any(var in os.environ for var in _MPI_RANK_ENV_VARS)


# Loads MPIPoolExecutor if MPI is available
if is_running_with_mpi():
  from mpi4py.futures import MPIPoolExecutor, wait
  _EVALUATION_MANAGER_USES_MPI4PY = True
else:
  _EVALUATION_MANAGER_USES_MPI4PY = False
  from concurrent.futures import wait


def _timed_call(fn, x, kwargs):
  """Run fn(x, **kwargs) and record worker-side timing."""
  start_time = time.perf_counter()
  fx = fn(x, **kwargs)
  done_time = time.perf_counter()
  return {
      "result": fx,
      "start_time": start_time,
      "done_time": done_time,
      "execution_time": done_time - start_time,
  }


def _summary_stats(values):
  """Return mean, std_dev, min, max for a sequence."""
  if not values:
    return None

  n = len(values)
  mean = sum(values) / n
  if n > 1:
    var = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(var)
  else:
    std_dev = 0.0

  return {
      "mean": mean,
      "std_dev": std_dev,
      "min": min(values),
      "max": max(values),
  }


class EvaluationManager:
  """Class that manages the evaluation of functions using multiple executors."""

  def __init__(
    self,
    cpu_executor=None,
    mpi_executor=None,
<<<<<<< HEAD
    max_workers=None) -> None:
=======
    profiling=False,
    task_name="TASK") -> None:
>>>>>>> origin/develop
    self._queue = deque([])
    self._queue_lock = threading.Lock()
    self.logger = logging.getLogger(self.__class__.__name__)
    self.profiling = profiling
    self._first_submit_time = None
    self.task_name = task_name

    self.executors = {
        "cpu": ProcessPoolExecutor() if cpu_executor is None else cpu_executor
    }
    if _EVALUATION_MANAGER_USES_MPI4PY:
      self.executors["mpi"] = (
          MPIPoolExecutor(max_workers=max_workers) if mpi_executor is None else mpi_executor
      )
    elif mpi_executor is not None:
      self.executors["mpi"] = mpi_executor

    self.logger.info("EvaluationManager initialized with executors:")
    for key, executor in self.executors.items():
      self.logger.info(f"  - {key}: {executor}")

  def __del__(self) -> None:
    for executor in self.executors.values():
      executor.shutdown(wait=False)
    self.logger.info(f"{self.task_name} EvaluationManager destroyed and executors shut down.")

  def _get_num_workers(self):
    """Return number of workers."""
    if "mpi" in self.executors and is_running_with_mpi():
      return int(os.environ.get("MPI4PY_FUTURES_MAX_WORKERS", 1))
    try:
      return self.executors["cpu"]._max_workers
    except AttributeError:
      return 1

  def set_task_name(self, task_name):
    self.task_name = task_name

  def _print_timing_stats(self, label, values):
    stats = _summary_stats(values)
    if stats is None:
      print(f"{label}: no completed tasks")
      return

    print(
        f"{label:<18} "
        f"Mean={stats['mean']:.6e}  "
        f"StdDev={stats['std_dev']:.6e}  "
        f"Min={stats['min']:.6e}  "
        f"Max={stats['max']:.6e}"
    )

  def sync(self) -> None:
    """Wait for all submitted tasks to complete."""
    future_objs = [queue_obj["future"] for queue_obj in self._queue]
    wait(future_objs)

  def submit_tasks(self, fn, X, execute_at="cpu", **kwargs) -> None:
    """Submits tasks to the specified executor."""
    key = execute_at.lower()
    with self._queue_lock:
      for x in X:
        submit_time = time.perf_counter()
        if self._first_submit_time is None:
          self._first_submit_time = submit_time

        if self.profiling:
          future_obj = self.executors[key].submit(_timed_call, fn, x, kwargs)
        else:
          future_obj = self.executors[key].submit(fn, x, **kwargs)

        self._queue.append({
            "x": copy.deepcopy(x),
            "future": future_obj,
            "submit_time": submit_time,
        })
        self.logger.info(f"{self.task_name} Submitted f({x})")

  def retrieve_results(self) -> tuple[list, list]:
    """Retrieves the results of completed tasks."""
    X = deque([])
    F = deque([])

    execution_times = []
    wait_times = []
    turnaround_times = []

    # Master-side wall clock for the whole batch
    batch_done_time = time.perf_counter()

    with self._queue_lock:
      new_queue = deque([])

      for item in self._queue:
        x = item["x"]
        future = item["future"]
        submit_time = item["submit_time"]

        if future.done():
          try:
            fx = future.result()
          except CancelledError:
            self.logger.warning(f"{self.task_name} The execution of x={x} was cancelled.")
            continue

          if self.profiling:
            # These are fine for local inspection, but note:
            # worker_start_time / worker_done_time are on worker clocks.
            worker_start_time = fx["start_time"]
            worker_done_time = fx["done_time"]

            execution_time = fx["execution_time"]

            # These are not robust across different node clocks, but kept here
            # because you already had them.
            wait_time = worker_start_time - submit_time
            turnaround_time = worker_done_time - submit_time

            execution_times.append(execution_time)
            wait_times.append(wait_time)
            turnaround_times.append(turnaround_time)

            fx = fx["result"]

          X.append(x)
          F.append(fx)
          self.logger.info(f"{self.task_name} Completed: f({x}) = {fx}")
        else:
          new_queue.append(item)

      self._queue = new_queue

      num_workers = self._get_num_workers()
      total_work = sum(execution_times)

      # Ideal walltime = total work / number of workers
      ideal_walltime = total_work / num_workers if num_workers > 0 else 0.0

      # Actual walltime = master-side elapsed time for the whole batch
      actual_walltime = (
          batch_done_time - self._first_submit_time
          if self._first_submit_time is not None else 0.0
      )

      print("\n=== Parallel Performance ===")
      print(f"{self.task_name} Workers:                                     {num_workers}")
      print(f"{self.task_name} Total work in seconds:                       {total_work:.6e}")
      print(f"{self.task_name} Ideal walltime in seconds (perfect balance): {ideal_walltime:.6e}")
      print(f"{self.task_name} Actual walltime in seconds (observed):       {actual_walltime:.6e}")

    return list(X), list(F)
  def completed_tasks(self) -> bool:
    return (len(self._queue) == 0)
  def num_submitted_tasks(self) -> int:
    return len(self._queue)  
  
