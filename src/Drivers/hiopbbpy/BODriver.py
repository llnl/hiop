"""
  Code description:
     for a 2D example LpNormProblem
        1) randomly sample training points
        2) define a Kriging-based Gaussian-process (smt backend)
           trained on said data
        3) determine the minimizer via BOAlgorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from LpNormProblem import LpNormProblem
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.problems import BraninProblem


### parameters
n_samples = 5 # number of the initial samples to train GP
theta = 1.e-2 # hyperparameter for GP kernel

nx = 2 # dimension of the problem
xlimits = np.array([[-5, 5], [-5, 5]]) # bounds on optimization variable

problem = LpNormProblem(nx, xlimits)
problem = BraninProblem()

print(problem.name, " problem")

### initial training set
x_train = problem.sample(n_samples)
y_train = problem.evaluate(x_train)

# Define the GP surrogate model
gp_model = smtKRG(theta, xlimits, nx)
gp_model.train(x_train, y_train)

acquisition_type = "LCB"
print("acquisition type: ", acquisition_type)

# Instantiate and run Bayesian Optimization
bo = BOAlgorithm(gp_model, x_train, y_train, acquisition_type = "LCB") #EI or LCB
bo.optimize(problem)

# Retrieve optimal point
x_opt, y_opt = bo.getOptimalPoint()
