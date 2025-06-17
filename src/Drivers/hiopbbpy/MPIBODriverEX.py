import numpy as np
import sys

from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.problems import LpNormProblem
from hiopbbpy.utils import MPIEvaluator

if __name__ == "__main__":
    # ------ objective
    nx = 2         # dimension of the problem
    xlimits = np.array([[-5, 5], [-5, 5]]) # bounds on optimization variable
    problem = LpNormProblem(nx, xlimits)

    # ----- evaluator
    evaluator = MPIEvaluator()    

    # ----- GP-surrogate
    theta =  1.e-2
    n_samples = 5
    x_train = problem.sample(n_samples)

    y_train = evaluator.run(problem.evaluate, x_train)
    
    ### Define the GP surrogate model
    gp_model = smtKRG(theta, xlimits, nx)
    gp_model.train(x_train, y_train)

    # --- setup the BO loop
    acq_type = 'EI'
    batch_size = 4
    options = {
        'acquisition_type': acq_type,
        'batch_size': batch_size,
        'bo_maxiter': 10,
        'opt_solver': 'SLSQP',
        'evaluator': evaluator
      }
    # Instantiate and run Bayesian Optimization
    bo = BOAlgorithm(problem, gp_model, x_train, y_train, options = options) #EI or LCB
    bo.optimize()
