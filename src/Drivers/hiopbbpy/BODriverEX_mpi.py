"""
  Code description:
     for a 2D example LpNormProblem (MPI version for scaling tests)
        1) randomly sample training points
        2) define a Kriging-based Gaussian-process (smt backend)
           trained on said data
        3) determine the minimizer via BOAlgorithm

  This version uses MPIPoolExecutor for multi-node parallel execution.

  Usage:
    mpiexec -n 16 python -m mpi4py.futures BODriverEX_mpi.py

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.problems import BraninProblem, LpNormProblem
from hiopbbpy.utils import MPIEvaluator
from mpi4py.futures import MPIPoolExecutor

### parameters
n_samples = 64 # number of the initial samples to train GP
theta = 1.e-2  # hyperparameter for GP kernel
nx = 2         # dimension of the problem
xlimits = np.array([[-5, 5], [-5, 5]]) # bounds on optimization variable

prob_type_l = ["LpNorm"]      # ["LpNorm", "Branin"]
acq_type_l = ["LCB"]          # ["LCB", "EI"]

def con_eq(x):
  return  x[0] + x[1] - 4

def con_jac_eq(x):
  return  np.array([1.0, 1.0])

def con_ineq(x):
  return  x[0] - x[1]

def con_jac_ineq(x):
  return  np.array([1.0, -1.0])

# 'SLSQP' requires constraints defined in a list of dict.
# IPOPT can support this format, too
user_constraint_list = [{'type': 'eq',   'fun': con_eq,   'jac': con_jac_eq},
                   {'type': 'ineq', 'fun': con_ineq, 'jac': con_jac_ineq}]

def cons_vec(x):
    x1, x2 = x
    return np.array([
        (x1 - 2)**2 + (x2 - 2.5)**2 - 2,
        x1 + x2 - 5,
        -x1
    ])

# Jacobian of constraints
def cons_jac_vec(x):
    x1, x2 = x
    return np.array([
        [2 * (x1 - 2), 2 * (x2 - 2.5)],
        [1, 1],
        [-1, 0]
    ])

cl = -np.inf * np.ones(3)
cu = np.zeros(3)

# 'trust-constr' and IPOPT support vector-valued constraints
user_constraint_dict = {'cons': cons_vec, 'jac': cons_jac_vec, 'cl': cl, 'cu': cu}


if __name__ == "__main__":
  do_profiling = True

  for prob_type in prob_type_l:
    print()
    print(f"========================================")
    print(f"Testing BO with {prob_type} problem")
    print(f"========================================")

    # ----- Create MPI evaluators
    # MPIPoolExecutor() will use all available MPI processes when
    # launched with: mpiexec -n N python -m mpi4py.futures script.py
    obj_evaluator = MPIEvaluator(
        executor=MPIPoolExecutor(),
        profiling=do_profiling,
        task_name="MPI_OBJ_EVAL"
    )
    opt_evaluator = MPIEvaluator(
        function_mode=False,
        executor=MPIPoolExecutor(),
        profiling=do_profiling,
        task_name="MPI_OPT_EVAL"
    )

    if prob_type == "LpNorm":
      problem = LpNormProblem(nx, xlimits)
    else:
      problem = BraninProblem()
    problem.set_constraints(user_constraint_list) # for solver 'trust-constr' and IPOPT, use user_constraint_dict; for solver 'SLSQP' and IPOPT, user_constraint_list

    for acq_type in acq_type_l:
      print(f"\nAcquisition type: {acq_type}")

      ### initial training set
      x_train = problem.sample(n_samples)
      y_train = obj_evaluator.run(problem.evaluate, x_train)

      ### Define the GP surrogate model
      gp_model = smtKRG(theta, xlimits, nx)
      gp_model.train(x_train, y_train)

      opt_solver = 'SLSQP'   #"SLSQP" "IPOPT" "trust-constr"
      if opt_solver == "SLSQP" or opt_solver == "trust-constr":
        solver_options = {"maxiter": 100}  #for scipy solvers
      elif opt_solver == "IPOPT":
        solver_options = {"max_iter": 100, "print_level": 1}

      options = {
        'acquisition_type': acq_type,
        'log_level': 'info',
        'bo_maxiter': 20,
        'opt_solver': opt_solver,
        'batch_size': 2,
        'n_start': 64,
        'solver_options': solver_options,
        'obj_evaluator': obj_evaluator,
        'opt_evaluator': opt_evaluator
      }

      # Instantiate and run Bayesian Optimization
      print(f"Starting Bayesian Optimization...")
      bo = BOAlgorithm(problem, gp_model, x_train, y_train, options = options) #EI or LCB
      bo.optimize()

      print(f"\nOptimization complete!")
      print(f"Optimal solution: {bo.x_opt}")
      print(f"Optimal value: {bo.y_opt}")
