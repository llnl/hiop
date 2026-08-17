"""
This file provides some helper functions for hiopbb.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>

Parts of this file are derivatives of 
SMT: Surrogate Modeling Toolkit
P. Saves and R. Lafage and N. Bartoli and Y. Diouane and J. H. Bussemaker and T. Lefebvre and J. T. Hwang and J. Morlier and J. R. R. A. Martins.

SMT 2.0: A Surrogate Modeling Toolbox with a focus on Hierarchical and Mixed Variables Gaussian Processes, Advances in Engineering Software, 2024.

SMT is released under Copyright (c) 2017, SMT developers
under a BSD 3-Clause License and the following disclaimer:

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from .evaluation_manager import EvaluationManager
import logging

import os
import time
import uuid
from pathlib import Path

def check_required_keys(user_dict, required_keys):
  for key in required_keys:
    if key not in user_dict:
      raise KeyError(f"Missing required key: '{key}'")


class Evaluator(object):
  """
  An interface for evaluation of a function at x points (nsamples of dimension nx).
  User can derive this interface and override the run() method to implement custom multiprocessing.
  """

  def run(self, fun, x):
    """
    Evaluates fun at x.

    Parameters
    ---------
    fun : function to evaluate: (nsamples, nx) -> (nsample, 1)

    x : np.ndarray[nsamples, nx]
        nsamples points of nx dimensions.

    Returns
    -------
    np.ndarray[nsample, 1]
        fun evaluations at the nsamples points.

    """
    return fun(x)


class MPIEvaluator(Evaluator):
  """
  A wrapper of the evaluation_manager code that supports multiple execution modes.

  Execution modes:

  1. Single-node with ProcessPoolExecutor or ThreadPoolExecutor:
     python application.py
    
     It should be noted that performance issues due to (lack of) thready affinity
     was observed on LC clusters with ThreadPoolExecutor. In all these cases, 
     ProcessPoolExecutor worked flawelessly. We used srun -n 1 -c cores_or_threads 

  2. Single-node with MPI (legacy mode):
     env MPI4PY_FUTURES_MAX_WORKERS=8 mpiexec -n 1 python application.py

  3. Multi-node with MPI (recommended for HPC clusters):
     mpiexec -n <N> python application.py
     where N >= 2 (1 master rank 0 + N-1 workers distributed across nodes)

     For multi-node, your application.py must have:
       if __name__ == "__main__":
         from mpi4py import MPI
         from mpi4py.futures import MPIPoolExecutor
         import sys

         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()

         if rank != 0:
           MPIPoolExecutor()  # Workers enter executor loop
           sys.exit(0)

         # Master (rank 0) continues here
         executor = MPIPoolExecutor()
         evaluator = MPIEvaluator(executor=executor, ...)
         # ... rest of your code

  Output format:
  Function evaluations return arrays with structure [[eval0], [eval1], [eval2], ...]
  which are reformatted to [eval0, eval1, eval2, ...]
  """
  def __init__(self, function_mode=True, executor=None, profiling=False,
                 task_name="MPITASK", run_root="./hiop_temp", use_run_dir=False):
    # If no executor provided, create a default ProcessPoolExecutor
    if executor is None:
      from concurrent.futures import ProcessPoolExecutor
      #from concurrent.futures import ThreadPoolExecutor
      import multiprocessing
      max_workers = 4 #multiprocessing.cpu_count()
      #executor = ThreadPoolExecutor(max_workers=max_workers)
      executor = ProcessPoolExecutor(max_workers=max_workers)
      print(f"No executor provided for {task_name}, using ProcessPoolExecutor with {max_workers} workers")

    self.manager = EvaluationManager(executor, profiling=profiling, task_name=task_name)
    self.function_mode = function_mode
    self.run_root = Path(run_root)
    self.use_run_dir = use_run_dir
    if self.use_run_dir:
      self.run_root.mkdir(parents=True, exist_ok=True)
    print(f"Create Evaluator for task: {task_name}")
  
  def __del__(self):
    del self.manager
  
  def set_task_name(self, task_name) -> None:
    self.manager.set_task_name(task_name)
    return
 
  def submit_tasks(self, fun, Xin) -> None:
    nevals = Xin.shape[0]

    if self.use_run_dir:
      # unique batch directory so repeated calls do not reuse temp_dir_0, temp_dir_1, ...
      batch_id = f"{self.manager.task_name}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
      batch_dir = self.run_root / batch_id
      batch_dir.mkdir(parents=True, exist_ok=False)

    for i in range(nevals):
      xi = np.atleast_2d(Xin[i])

      kwargs = {}
      if self.use_run_dir:
        run_dir = batch_dir / f"eval_{i:04d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        kwargs["run_dir"] = str(run_dir)

      # submit (index, x) so we can restore original order later
      self.manager.submit_tasks(_run_indexed_fun, [(fun, i, xi)], **kwargs)
    return None


  def run(self, fun, Xin):  
    nevals = Xin.shape[0]
    self.submit_tasks(fun, Xin)
    self.manager.sync()
    print(f"\n{'='*50}")
    print(f"Retrieving results for {self.manager.task_name}...")
    print(f"Profiling enabled: {self.manager.profiling}")
    print(f"{'='*50}\n", flush=True)
    Xout, Fout = self.manager.retrieve_results()

    # restore original order using returned indices
    ordered = [None] * nevals
    for out in Fout:
      if out is None:
        continue
      # Handle profiling case where _run_indexed_fun returns (idx, result)
      # but result might be a timing dict if profiling is enabled
      idx, val = out
      ordered[idx] = val

    missing = [i for i, v in enumerate(ordered) if v is None]
    if missing:
      raise RuntimeError(f"Missing evaluation results for indices {missing}")

    if self.function_mode:
      Y = np.empty((nevals, 1), dtype=float)
      for i, val in enumerate(ordered):
        arr = np.asarray(val, dtype=float)
        Y[i, 0] = float(arr.reshape(-1)[0])
    else:
      Y = [val[0] for val in ordered]
    return Y
  def num_submitted_tasks(self):
    return self.manager.num_submitted_tasks()
  def num_workers(self):
    return self.manager._get_num_workers()
  def sync(self):
    self.manager.sync()
    return
  def retrieve_results(self):
    inputs, indexed_results = self.manager.retrieve_results()
    values = []
    for task_input, out in zip(inputs, indexed_results):
      if out is None:
        # EvaluationManager records failed/cancelled futures as None.  Silently
        # skipping one would leave its BnB parent permanently marked in-flight.
        raise RuntimeError(f"Asynchronous evaluation failed for input: {task_input!r}")
      if not isinstance(out, tuple) or len(out) != 2:
        raise RuntimeError(f"Unexpected indexed evaluator result: {out!r}")
      _, value = out
      values.append(value)

    if self.function_mode:
      y = np.empty((len(values), 1), dtype=float)
      for i, value in enumerate(values):
        arr = np.asarray(value, dtype=float)
        y[i, 0] = float(arr.reshape(-1)[0])
      return y

    return [value[0] for value in values]

def _run_indexed_fun(fun, idx, x, **kwargs):
    return idx, fun(x, **kwargs)

class Logger:
  """
  A simple wrapper for Python's logging module that sets up a reusable logger.
  Logs to the console using a consistent format.

  Set the log level as a string from 'DEBUG', 'INFO', 'SCALARS', 'ITERATION', 'WARNING', 'ERROR', 'CRITICAL' and 'NONE'
  """

  def __init__(self, name='hiopbbpy'):
    # ---- Custom levels ----
    SCALARS = logging.INFO + 1    # between INFO(20) and WARNING(30)
    ITERATION = logging.INFO + 5  # between INFO(20) and WARNING(30)
    NONE = logging.CRITICAL + 1

    logging.addLevelName(SCALARS, "SCALARS")
    logging.addLevelName(ITERATION,   "ITERATION")
    logging.addLevelName(NONE,   "NONE")

    # Register names on the logging module so getattr works
    setattr(logging, "ITERATION", ITERATION)
    setattr(logging, "SCALARS", SCALARS)
    setattr(logging, "NONE", NONE)

    # Create a logger instance with a given name        
    self._logger = logging.getLogger(name)
    self._logger.propagate = False  # prevent double logging

    # Create a console output handler
    if not self._logger.handlers:     
      ch = logging.StreamHandler()

      # Define the output format: logger name, and message
      formatter = logging.Formatter('%(name)s %(message)s')

      # Add the handle
      ch.setFormatter(formatter)
      self._logger.addHandler(ch)

  def setlevel(self, level_str):
    level = getattr(logging, str(level_str).upper(), logging.INFO)
  
    self._logger.setLevel(level)
    for handler in self._logger.handlers:
      handler.setLevel(level)

  # ---- Convenience methods for custom levels ----
  def scalars(self, msg, *args, **kwargs):
    if self._logger.isEnabledFor(logging.SCALARS):
      self._logger._log(logging.SCALARS, msg, args, **kwargs)

  def iterations(self, msg, *args, **kwargs):
    if self._logger.isEnabledFor(logging.ITERATION):
      self._logger._log(logging.ITERATION, msg, args, **kwargs)

  def __getattr__(self, attr):
    # Forward all unknown attributes to the underlying logger
    return getattr(self._logger, attr)
