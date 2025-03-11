"""
  Code description:
     for a 1-D example problem
     randomly sample training points
     define a Kriging-based Gaussian-process (smt backend)
     trained on said data
     define an LCB acquisition function (not smt backend)
     plot the acquisition function and determine
     the minimizer so as to test some of the infastructure
     from BOAlgorithm

"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from lp_problem import LpProblem
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import LCBacquisition
from hiopbbpy.opt import BOAlgorithm


### parameters
n_samples = 5 # number of the samples
theta = 1.e-2 # hyperparameter for GP kernel

nx = 1 # dimension of the problem
xlimits = np.array([[-1. ,1.]])
nx = 2
xlimits = np.array([[-5, 5], [-5, 5]])

problem = LpProblem(nx, xlimits)
print(problem.name, " problem")

### initial training set
x_train = problem.sample(n_samples)
y_train = problem.evaluate(x_train)

# Define the GP surrogate model
gp_model = smtKRG(theta, xlimits, nx)
gp_model.train(x_train, y_train)

# Instantiate and run Bayesian Optimization
bo = BOAlgorithm(gp_model, x_train, y_train)
bo.optimize(problem)

# Retrieve optimal point
x_opt, y_opt = bo.getOptimalPoint()
print(f"Optimal x: {x_opt}, Optimal y: {y_opt}")