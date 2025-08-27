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
from .evaluation_manager import EvaluationManager, is_running_with_mpi
import logging


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
  def __init__(self, function_mode=True,cpu_executor=None, mpi_executor=None):
    self.manager = EvaluationManager(cpu_executor,mpi_executor)
    self.function_mode = function_mode
    if is_running_with_mpi():
      self.executor_type = "mpi"
    else:
      self.executor_type = "cpu"
  def __del__(self):
    del self.manager
  def run(self, fun, Xin):
    nevals = Xin.shape[0]
    self.manager.submit_tasks(fun, [np.atleast_2d(Xin[i]) for i in range(nevals)], execute_at=self.executor_type)
    self.manager.sync()
    Xout, Fout = self.manager.retrieve_results()
    if self.function_mode:
      Y = np.ndarray((nevals, 1))
      Y[:,0] = np.array(Fout)[:,0,0]
    else:
      Y = [Fi[0] for Fi in Fout]
    return Y

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

    # Create a console output handler
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