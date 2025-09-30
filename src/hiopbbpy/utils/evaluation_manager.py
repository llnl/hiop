"""
This is a class to manage function evaluations using multiple parallel executors.
It supports both intra-node and inter-node parallelism.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Weslley S Pereira <wdasilv@nrel.gov>
"""

import logging
import copy
import os
from concurrent.futures import ProcessPoolExecutor, CancelledError
from collections import deque


def is_running_with_mpi():
  """Returns True if the code is running in an MPI environment."""
  _MPI_RANK_ENV_VARS = [
      "OMPI_COMM_WORLD_RANK",  # Open MPI
      "PMI_RANK",  # MPICH, Intel MPI, Cray MPI
      "MPI_RANK",  # Intel MPI (sometimes)
      "MV2_COMM_WORLD_RANK",  # MVAPICH
  ]
  return any(var in os.environ for var in _MPI_RANK_ENV_VARS)


# Loads MPIPoolExecutor if MPI is available
if is_running_with_mpi():
  from mpi4py.futures import MPIPoolExecutor, wait
  _EVALUATION_MANAGER_USES_MPI4PY = True
else:
  _EVALUATION_MANAGER_USES_MPI4PY = False
  from concurrent.futures import wait


class EvaluationManager:
  """Class that manages the evaluation of functions using multiple executors.

  It allows for the submission of tasks to different executors:

  - Intra-node parallelism using ProcessPoolExecutor (default)
  - Inter-node parallelism using MPIPoolExecutor (if available)

  Tasks are submitted to the executors, and their results can be retrieved
  once they are completed. The submission of tasks is asynchronous, and the
  retrieval of results is done in a blocking manner. The syncronization of all
  tasks is done using the `sync` method, which waits for all tasks to complete
  before proceeding.

  :param cpu_executor: The executor to use for intra-node parallelism, i.e.,
      that controls tasks on the executor's local node. If None, a new
      ProcessPoolExecutor is created.
  :param mpi_executor: The executor to use for inter-node parallelism, i.e.,
      that controls tasks on the executor's remote nodes. If None, a new
      MPIPoolExecutor is created. If MPI is not available, this parameter is
      ignored.

  .. attribute:: logger:

      The logger for the EvaluationManager. It is used to log messages.

  .. attribute:: executors:
      A dictionary of executors used for task submission. The keys are
      "cpu" and "mpi", and the values are the corresponding executors.
      The default is a ProcessPoolExecutor for "cpu" and an MPIPoolExecutor
      for "mpi" if MPI is available.
  """

  def __init__(
    self,
    cpu_executor=None,
    mpi_executor=None) -> None:
    self._queue = deque([])
    self.logger = logging.getLogger(self.__class__.__name__)

    self.executors = {
        "cpu": ProcessPoolExecutor() if cpu_executor is None else cpu_executor
    }
    if _EVALUATION_MANAGER_USES_MPI4PY:
      self.executors["mpi"] = (
          MPIPoolExecutor() if mpi_executor is None else mpi_executor
      )
    elif mpi_executor is not None:
      self.executors["mpi"] = mpi_executor

    self.logger.info("EvaluationManager initialized with executors:")
    for key, executor in self.executors.items():
      self.logger.info(f"  - {key}: {executor}")

  def __del__(self) -> None:
    for executor in self.executors.values():
      executor.shutdown(wait=False)
    self.logger.info("EvaluationManager destroyed and executors shut down.")

  def sync(self) -> None:
    """Synchronizes all tasks by waiting for their completion.

    Results can be retrieved using :meth:`retrieve_results()`.
    """
    future_objs = [queue_obj[1] for queue_obj in self._queue]
    wait(future_objs)

  def submit_tasks(self, fn, X, execute_at="cpu", *args, **kwargs) -> None:
    """Submits tasks to the specified executor.

    :param fn: The function to be executed.
    :param X: The sequence of input data for the function.
    :param execute_at: The executor to use for task submission. It can be
        "cpu" for intra-node parallelism or "mpi" for inter-node
        parallelism.
    :param args: Additional positional arguments to be passed to the
        function.
    :param kwargs: Additional keyword arguments to be passed to the
        function.
    """
    key = execute_at.lower()
    for x in X:
      future_obj = self.executors[key].submit(fn, x, *args, **kwargs)
      self._queue.append([copy.deepcopy(x), future_obj])
      self.logger.info(f"Submitted f({x})")

  def retrieve_results(self) -> tuple[list, list]:
    """Retrieves the results of completed tasks.

    :return: A tuple containing two lists: the inputs and the results of
        the completed tasks.
    """
    X = deque([])
    F = deque([])
    Idxs = deque([])
    new_queue = deque([])
    for i, item in enumerate(self._queue):
      x = item[0]
      future = item[1]
      if future.done():
        # Try to get result
        try:
          fx = future.result()
          Idxs.append(i)
        except CancelledError:
          self.logger.warning(f"The execution of x={x} was cancelled.")
          continue

        # Add result to the output
        X.append(x)
        F.append(fx)
        self.logger.info(f"Completed: f({x}) = {fx}")
      else:
        # Keep the future in the queue
        new_queue.append(item)

    # Remove completed tasks from the queue
    self._queue = new_queue

    X = [X[Idxs[i]] for i in range(len(Idxs))]
    F = [F[Idxs[i]] for i in range(len(Idxs))]
    return list(X), list(F)


