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

### saved solutions --- from 1000 repetitions
saved_mean_obj = {"LpNorm": {"LCB": 0.01913481, "EI": 0.19634178}, "Branin": {"LCB": 0.51033727, "EI": 1.3722849}}
saved_min_obj = {"LpNorm": {"LCB": 0.00014948, "EI": 0.01305073}, "Branin": {"LCB": 0.39810948, "EI": 0.40188902}}
saved_max_obj = {"LpNorm": {"LCB": 0.10684082, "EI": 0.93578515}, "Branin": {"LCB": 1.4407172, "EI": 4.6802864}}

prob_type_l = ["LpNorm", "Branin"]
acq_type_l = ["LCB", "EI"]

mean_obj = {}
max_obj = {}
min_obj = {}

retval = 1
sys.exit(retval)
for prob_type in prob_type_l:
   print()
   if prob_type == "LpNorm":
      problem = LpNormProblem(nx, xlimits)
   else:
      problem = BraninProblem()

   if prob_type not in mean_obj:
      mean_obj[prob_type] = {}
      max_obj[prob_type] = {}
      min_obj[prob_type] = {}

   for acq_type in acq_type_l:
      if acq_type not in mean_obj[prob_type]:
         mean_obj[prob_type][acq_type] = 0
         max_obj[prob_type][acq_type] = -np.inf
         min_obj[prob_type][acq_type] = np.inf

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
         max_obj[prob_type][acq_type] = max(max_obj[prob_type][acq_type], y_opt)
         min_obj[prob_type][acq_type] = min(min_obj[prob_type][acq_type], y_opt)

print("Summary:")
for prob_type in prob_type_l:
   for acq_type in acq_type_l:
      mean_obj[prob_type][acq_type] /= num_repeat
      print("(Min,Mean,Max) Opt.Obj for", prob_type, "-", acq_type, ":\t(", min_obj[prob_type][acq_type], ",",mean_obj[prob_type][acq_type], ",", max_obj[prob_type][acq_type], ")")
      
      #r_error = np.abs((mean_obj[prob_type][acq_type] - saved_mean_obj[prob_type][acq_type])/(1+saved_mean_obj[prob_type][acq_type]))
      #print("Relative Error from Mean: ", r_error)
      
      if max_obj[prob_type][acq_type] > saved_max_obj[prob_type][acq_type] or min_obj[prob_type][acq_type] < saved_min_obj[prob_type][acq_type]:
         print("Recorded (min, mean, max): (", saved_min_obj[prob_type][acq_type], ",", saved_mean_obj[prob_type][acq_type], ",", saved_max_obj[prob_type][acq_type], ")")
         retval = 1

sys.exit(retval)



