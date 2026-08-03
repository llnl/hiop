"""
  Code description:
     for an example LpNormProblem
        1) randomly sample training points and evaluate them in parallel
        2) define a Kriging-based Gaussian Process (smt) as the surrogate
        3) determine the minimizer via TurboAlgorithm (TuRBO / TuRBO-m)

  Run in parallel like BODriverEX.py:
     env MPI4PY_FUTURES_MAX_WORKERS=8 mpiexec -n 1 python TurboDriverEX.py
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import TurboAlgorithm
from hiopbbpy.problems import LpNormProblem
from hiopbbpy.utils import MPIEvaluator

### parameters
n_samples = 20   # number of the initial samples to train GP
theta = 1.e-2    # hyperparameter for GP kernel
nx = 10          # dimension of the problem
xlimits = np.array([[-5, 5]] * nx)  # bounds on optimization variable


if __name__ == "__main__":
  # ----- evaluator (parallel objective batch: the expensive-evaluation seam)
  obj_evaluator = MPIEvaluator()

  problem = LpNormProblem(nx, xlimits)

  ### initial training set (evaluated in parallel)
  x_train = problem.sample(n_samples)
  y_train = obj_evaluator.run(problem.evaluate, x_train)

  ### surrogate FACTORY: TuRBO builds new models per region and on restart.
  ### Swap this single line for `lambda: muyGP(nx, xlimits)` to use the scalable
  ### GP backend; nothing else in the driver changes.
  surrogate_factory = lambda: smtKRG(theta, xlimits, nx)

  options = {
    'log_level': 'ITERATION',
    'bo_maxiter': 50,
    'batch_size': 4,          # q: candidates evaluated in parallel per iteration
    'n_trust_regions': 3,     # m: TuRBO-m
    'n_candidates': 2000,     # Thompson-sampling candidates per region
    'local_gp': False,        # False = one global GP; True = one GP per region
    'seed': 0,
    'obj_evaluator': obj_evaluator
  }

  # Instantiate and run Trust-Region Bayesian Optimization
  turbo = TurboAlgorithm(problem, surrogate_factory, x_train, y_train, options = options)
  x_opt, y_opt = turbo.optimize()
  print("best x:", x_opt)
  print("best y:", float(np.array(y_opt).reshape(-1)[0]))
