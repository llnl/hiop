"""
  Code description:
     for a 2D example LpNormProblem
        1) randomly sample training points
        2) define a Kriging-based Gaussian-process (smt backend)
           trained on said data
        3) determine the minimizer via BOAlgorithm

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.problems import BraninProblem, LpNormProblem
from hiopbbpy.utils import MPIEvaluator

# Get user input for the number of repetitions from command-line arguments
if len(sys.argv) != 2 or int(sys.argv[1]) < 0:
    num_repeat = 1
else:
    num_repeat = int(sys.argv[1])

### parameters
n_samples = 5  # number of the initial samples to train GP
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

# 'SLSQP' requires constraints defined in a list of dict
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

# 'trust-constr' method supports vector-valued constraints
user_constraint_dict = {'cons': cons_vec, 'jac': cons_jac_vec, 'cl': cl, 'cu': cu}


if __name__ == "__main__":
  for prob_type in prob_type_l:
    print()
    # ----- evaluator
    evaluator = MPIEvaluator() 
    opt_evaluator = MPIEvaluator(function_mode=False)
    if prob_type == "LpNorm":
      problem = LpNormProblem(nx, xlimits)
    else:
      problem = BraninProblem()
    problem.set_constraints(user_constraint_dict) #use user_constraint_dict or user_constraint_list
   
    for acq_type in acq_type_l:
      print("Problem name: ", problem.name)
      print("Acquisition type: ", acq_type)
   
      ### initial training set
      x_train = problem.sample(n_samples)
      y_train = evaluator.run(problem.evaluate, x_train)
   
      ### Define the GP surrogate model
      gp_model = smtKRG(theta, xlimits, nx)
      gp_model.train(x_train, y_train)
    
      options = {
        'acquisition_type': acq_type,
        'bo_maxiter': 10,
        'opt_solver': 'IPOPT',  #"SLSQP" "IPOPT" "trust-constr"
        'batch_size': 3,
        'solver_options': {
           'max_iter': 200,
           'print_level': 1
           },
        'evaluator': evaluator,
        'opt_evaluator': opt_evaluator
      }
    
      # Instantiate and run Bayesian Optimization
      bo = BOAlgorithm(problem, gp_model, x_train, y_train, options = options) #EI or LCB
      bo.optimize()
