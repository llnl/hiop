"""
This is a class to manage function evaluations using multiple parallel executors.
It supports both intra-node and inter-node parallelism.

Supported executors:
  - concurrent.futures.ThreadPoolExecutor (single-node, multi-threaded)
  - concurrent.futures.ProcessPoolExecutor (single-node, multi-process)
  - mpi4py.futures.MPIPoolExecutor (multi-node, MPI-based)

For multi-node execution with MPI:
  1. Install mpi4py: pip install mpi4py
  2. Run with: mpiexec -n <N> python your_script.py
     where N >= 2 (1 master + N-1 workers distributed across nodes)
  3. Only rank 0 should create the EvaluationManager
  4. Worker ranks should call MPIPoolExecutor() to enter worker loop

Example multi-node usage:
  from mpi4py import MPI
  from mpi4py.futures import MPIPoolExecutor

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    # Master: create manager and submit tasks
    manager = EvaluationManager({"mpi": MPIPoolExecutor()}, profiling=True)
    manager.submit_tasks(my_func, data_list, execute_at="mpi")
    manager.sync()
    X, F = manager.retrieve_results()
  else:
    # Workers: enter worker loop
    MPIPoolExecutor()

See EvaluationManagerCI.py for a complete working example with multi-node MPI.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Weslley S Pereira <wdasilv@nrel.gov>
"""

import threading
import logging
from concurrent.futures import CancelledError, wait
from collections import deque
import os
import time
import math
import copy

def _timed_call(fn, args, kwargs):
  """Run fn(*args, **kwargs) and record worker-side timing."""
  start_time = time.perf_counter()
  fx = fn(*args, **kwargs)
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
  """Manage asynchronous function evaluations over one or more executors.

    The manager is executor-agnostic: each configured executor only needs a
    ``submit(fn, *args, **kwargs)`` method that returns a Future-like object
    exposing ``done()`` and ``result()``.

    Tasks are submitted asynchronously via :meth:`submit_tasks`. Completed
    results are collected lazily by :meth:`retrieve_results` and eagerly by
    :meth:`sync` (which blocks until the running queue is empty).

    Parameters
    ----------
    executor:
        Either a single executor instance or a ``dict[str, executor]`` mapping.
        When a single executor is provided, it is stored under key ``"0"``.
    profiling:
        If True, wrap calls with worker-side timing.
    task_name:
        Label used in logging and profiling output.
  """

  def __init__(self, executor, profiling=False, task_name="TASK") -> None:
    self._queue = deque([])
    self._completed_X = deque([])
    self._completed_F = deque([])
    self._queue_lock = threading.Lock()

    if isinstance(executor, dict):
      self.executors = executor
    else:
      self.executors = {"0": executor}

    self.logger = logging.getLogger(self.__class__.__name__)
    self.task_name = task_name
    self.profiling = profiling
    self._first_submit_time = None

    # Store timing data if profiling is enabled
    self._execution_times = [] if profiling else None
    self._wait_times = [] if profiling else None
    self._turnaround_times = [] if profiling else None

    self.logger.info(f"{self.task_name} EvaluationManager initialized with executors:")
    for key, executor in self.executors.items():
      self.logger.info(f"  - {key}: {executor}")

  def __del__(self) -> None:
    """Shutdown managed executors during object destruction."""
    for executor in self.executors.values():
      try:
        executor.shutdown(wait=False)
        self.logger.info(f"{self.task_name} EvaluationManager destroyed and executors shut down.")
      except Exception as e:
        self.logger.warning(f"{self.task_name} Error shutting down executor: {e}")    

  def _get_num_workers(self):
    """Return number of workers."""
    # For standard executors like ThreadPoolExecutor / ProcessPoolExecutor
    for ex in self.executors.values():
      try:
        return ex._max_workers
      except AttributeError:
        pass

    # For MPI executors, try to get MPI communicator size
    if "mpi" in self.executors:
      try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        # Return size - 1 because rank 0 is the master
        return comm.Get_size() - 1
      except Exception:
        # Fallback to environment variable (legacy single-node mode)
        try:
          return int(os.environ.get("MPI4PY_FUTURES_MAX_WORKERS", 1))
        except Exception:
          pass

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
    """Block until all queued tasks finish.

    This method repeatedly waits on the currently queued futures and then
    harvests completed items into the internal completion buffers. Harvested
    results can be consumed using :meth:`retrieve_results`.
    """
    while True:
      with self._queue_lock:
        futures = [queue_obj[1] for queue_obj in self._queue]
        if len(futures) == 0:
          break

      wait(futures)

      with self._queue_lock:
        self._harvest_completed_locked(
          execution_times=self._execution_times,
          wait_times=self._wait_times,
          turnaround_times=self._turnaround_times
        )

  def submit_tasks(self, fn, X, execute_at=None, **kwargs) -> None:
    """Submit tasks to the specified executor.

    Parameters
    ----------
    fn:
      The function to be executed.
    X:
      Sequence of input data for the function. If an element is a tuple,
      it is expanded as positional arguments (``fn(*x, **kwargs)``);
      otherwise it is passed as a single argument (``fn(x, **kwargs)``).
    execute_at:
      Executor key to use for task submission. If ``None``, the first key
      in ``executors`` is used. The key lookup is case-insensitive.
    kwargs:
      Additional keyword arguments passed to the function.
    """
    
    if execute_at is None:
      execute_at = next(iter(self.executors))

    key = execute_at.lower()
    if key not in self.executors:
        raise KeyError(f"Executor '{execute_at}' not found. Available: {list(self.executors.keys())}")

    with self._queue_lock:
      for x in X:
        submit_time = time.perf_counter()
        if self._first_submit_time is None:
          self._first_submit_time = submit_time

        args = x if isinstance(x, tuple) else (x,)

        if self.profiling:
          future_obj = self.executors[key].submit(_timed_call, fn, args, kwargs)
        else:
          future_obj = self.executors[key].submit(fn, *args, **kwargs)

        self._queue.append([x, future_obj, key, submit_time])
        self.logger.info(f"{self.task_name} Submitted f({x})")

  def retrieve_results(self) -> tuple[list, list]:
    """Retrieves the results of completed tasks.
    Returns
    -------
    tuple[list, list]
      Inputs and corresponding results for completed tasks. If a task
      failed or was cancelled, its result entry is ``None``.
    """
    # Master-side wall clock for the whole batch
    batch_done_time = time.perf_counter()

    with self._queue_lock:
      # Harvest any remaining completed tasks
      self._harvest_completed_locked(
            execution_times=self._execution_times,
            wait_times=self._wait_times,
            turnaround_times=self._turnaround_times,
      )

      X = list(self._completed_X)
      F = list(self._completed_F)
      self._completed_X.clear()
      self._completed_F.clear()

    # Use the stored timing data collected during all harvests
    execution_times = self._execution_times or []
    wait_times = self._wait_times or []
    turnaround_times = self._turnaround_times or []

    if self.profiling:
      print(f"\nDEBUG: Profiling enabled, collected {len(execution_times)} execution times", flush=True)
      if execution_times:
        self._print_timing_stats(f"{self.task_name} Execution times", execution_times)
      else:
        print(f"WARNING: Profiling enabled but no execution times collected!", flush=True)

    if self.profiling and execution_times:
      pass  # Timing stats already printed above

      # Optional: only print these if you are comfortable with cross-clock values
      # self._print_timing_stats("Wait times", wait_times)
      # self._print_timing_stats("Turnaround times", turnaround_times)

      num_workers = self._get_num_workers()
      total_work = sum(execution_times)

      # Ideal walltime = total work / number of workers
      ideal_walltime = total_work / num_workers if num_workers > 0 else 0.0

      # Actual walltime = master-side elapsed time for the whole batch
      actual_walltime = (
          batch_done_time - self._first_submit_time
          if self._first_submit_time is not None else 0.0
      )

      print("\n=== Parallel Performance ===", flush=True)
      print(f"{self.task_name} Workers:                                     {num_workers}", flush=True)
      print(f"{self.task_name} Total work in seconds:                       {total_work:.6e}", flush=True)
      print(f"{self.task_name} Ideal walltime in seconds (perfect balance): {ideal_walltime:.6e}", flush=True)
      print(f"{self.task_name} Actual walltime in seconds (observed):       {actual_walltime:.6e}", flush=True)

    # Clear timing data for next batch
    if self.profiling:
      self._execution_times.clear()
      self._wait_times.clear()
      self._turnaround_times.clear()

    self._first_submit_time = None
    return X, F
  
  def _harvest_completed_locked(
        self,
        execution_times=None,
        wait_times=None,
        turnaround_times=None,
    ) -> None:
    """Move completed task results from running queue into completion buffers.

       This method assumes the caller holds ``_queue_lock``.
    """
    new_queue = deque([])

    for item in self._queue:
      x = item[0]
      future = item[1]
      submit_time = item[3]
      if future.done():
        self._completed_X.append(x)
        self._completed_F.append(None)

        try:
          fx = future.result()

          if self.profiling:
            # Check if fx is a timing dict (from _timed_call)
            if isinstance(fx, dict) and "start_time" in fx:
              worker_start_time = fx["start_time"]
              worker_done_time = fx["done_time"]
              execution_time = fx["execution_time"]

              # These are OK for local runs, but can be unreliable across nodes
              wait_time = worker_start_time - submit_time
              turnaround_time = worker_done_time - submit_time

              if execution_times is not None:
                execution_times.append(execution_time)
              if wait_times is not None:
                wait_times.append(wait_time)
              if turnaround_times is not None:
                turnaround_times.append(turnaround_time)

              fx = fx["result"]
            else:
              # Profiling enabled but result is not a timing dict
              # This happens when function is wrapped (e.g., by MPIEvaluator)
              print(f"DEBUG: Profiling enabled but result type is {type(fx)}, not a timing dict", flush=True)

          self._completed_F[-1] = fx
          self.logger.info(f"{self.task_name} Completed: f({x}) = {fx}")

        except CancelledError:
          self.logger.warning(f"{self.task_name} The execution of x={x} was cancelled.")
        except Exception as e:
          self.logger.warning(f"{self.task_name} Task f({x}) raised an exception: {e}")

      else:
        new_queue.append(item)

    self._queue = new_queue

  def completed_tasks(self) -> bool:
    return (len(self._queue) == 0)
  def num_submitted_tasks(self) -> int:
    return len(self._queue)


  def print_status(self) -> None:
    """Print the current status of the task queue and completion buffers."""
    with self._queue_lock:
      futures = [queue_obj[1] for queue_obj in self._queue]
      n_running_futures = sum(1 for f in futures if not f.done())
      n_done_futures = len(futures) - n_running_futures
      self.logger.info(
          f"Status: {len(self._completed_X)} harvested results, "
          f"{n_running_futures} running tasks, {n_done_futures} completed tasks still in queue."
      )
