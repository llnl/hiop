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
import sys
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
  A wrapper of the evaluation_manager code.
  Note that application codes application.py that use this Evaluator should be run as
  env MPI4PY_FUTURES_MAX_WORKERS=8 mpiexec -n 1 python application.py
  Also, the application code should have a "main" section wrapped in
  if __name__ == "__main__":
  Expecting the function evaluations to return an array.
  Fout has then the structure of
  [[eval0], [[eval1]], [eval2],...]]
  We reformat to 
  [eval0, eval1, eval2,...]
  """
  def __init__(self, function_mode=True, executor=None, profiling=False,
                 task_name="MPITASK", run_root="./hiop_temp", use_run_dir=False):
    self.manager = EvaluationManager(executor, profiling=profiling, task_name=task_name)
    self.function_mode = function_mode
    self.run_root = Path(run_root)
    self.run_root.mkdir(parents=True, exist_ok=True)
    self.use_run_dir = use_run_dir
    print(f"Create Evaluator for task: {task_name}")
  
  def __del__(self):
    del self.manager
  
  def set_task_name(self, task_name):
    self.manager.set_task_name(task_name)
  
  def run(self, fun, Xin):  
    nevals = Xin.shape[0]
    print("in Evaluator::run")

    # unique batch directory so repeated calls do not reuse temp_dir_0, temp_dir_1, ...
    batch_id = f"{self.manager.task_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
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
      self.manager.submit_tasks(
        _run_indexed_fun,
        [(fun, i, xi)],
        **kwargs,
      )
      print(f"Submitted task {i + 1}", flush=True)

    self.manager.sync()
    Xout, Fout = self.manager.retrieve_results()

    # restore original order using returned indices
    ordered = [None] * nevals
    for out in Fout:
      if out is None:
        continue
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
      ch = logging.StreamHandler(sys.stdout)

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