"""
  Code description:
     for a 2D example LpNormProblem
        1) randomly sample training points
        2) define a Kriging-based Gaussian-process (smt backend)
           trained on said data
        3) determine the minimizer via BOAlgorithm
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from LpNormProblem import LpNormProblem
from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.problems import BraninProblem


# Get user input for the number of repetitions from command-line arguments
if len(sys.argv) != 2:
    num_repeat = 1
else:
    num_repeat = int(sys.argv[1])

### parameters
n_samples = 5  # number of the initial samples to train GP
theta = 1.e-2  # hyperparameter for GP kernel
nx = 2         # dimension of the problem
xlimits = np.array([[-5, 5], [-5, 5]]) # bounds on optimization variable

### saved solutions
saved_sol = {"LpNorm": {"LCB": 0.04618462, "EI": 0.44954611}, "Branin": {"LCB": 0.62655919, "EI": 1.9838798}}

prob_type_l = ["LpNorm", "Branin"]
acq_type_l = ["LCB", "EI"]

mean_obj = {}

retval = 0
for prob_type in prob_type_l:
   print()
   if prob_type == "LpNorm":
      problem = LpNormProblem(nx, xlimits)
   else:
      problem = BraninProblem()

   if prob_type not in mean_obj:
      mean_obj[prob_type] = {}

   for acq_type in acq_type_l:
      if acq_type not in mean_obj[prob_type]:
         mean_obj[prob_type][acq_type] = 0

      print("Problem name: ", problem.name)
      print("Acquisition type: ", acq_type)
   
      for n_repeat in range(num_repeat):
         print("Run: ", n_repeat, "/", num_repeat)
         ### initial training set
         x_train = problem.sample(n_samples)
         y_train = problem.evaluate(x_train)

         ### Define the GP surrogate model
         gp_model = smtKRG(theta, xlimits, nx)
         gp_model.train(x_train, y_train)
   
         # Instantiate and run Bayesian Optimization
         bo = BOAlgorithm(gp_model, x_train, y_train, acquisition_type = acq_type) #EI or LCB
         bo.optimize(problem)
         
         # Retrieve optimal objec
         y_opt = bo.getOptimalObjective()
         
         mean_obj[prob_type][acq_type] += y_opt

for prob_type in prob_type_l:
   for acq_type in acq_type_l:
      mean_obj[prob_type][acq_type] /= num_repeat
      print("Mean Opt.Obj for ", prob_type, "-", acq_type, mean_obj[prob_type][acq_type])
      
      r_error = np.abs((mean_obj[prob_type][acq_type] - saved_sol[prob_type][acq_type])/saved_sol[prob_type][acq_type])
      if r_error > 0.5:
         print("Relative Error > 0.5: ", r_error)
         print("Recorded Solution:", saved_sol[prob_type][acq_type])
         retval = 1

sys.exit(retval)



