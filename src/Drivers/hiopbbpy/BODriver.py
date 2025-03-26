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
if len(sys.argv) != 2 or int(sys.argv[1]) < 0:
    num_repeat = 1
else:
    num_repeat = int(sys.argv[1])

### parameters
n_samples = 5  # number of the initial samples to train GP
theta = 1.e-2  # hyperparameter for GP kernel
nx = 2         # dimension of the problem
xlimits = np.array([[-5, 5], [-5, 5]]) # bounds on optimization variable

### saved solutions --- from 1000 repetitions
saved_min_obj = {"LpNorm": {"LCB": 0.0007586314501994839, "EI": 0.002094016049616341}, "Branin": {"LCB": 0.3979820338569908, "EI": 0.39789916461969455}}
saved_mean_obj = {"LpNorm": {"LCB": 0.018774638321851504, "EI": 0.11583915178648867}, "Branin": {"LCB": 0.5079001079219421, "EI": 0.4377466109837465}}
saved_max_obj = {"LpNorm": {"LCB": 0.0755173754382861, "EI": 0.4175676394969743}, "Branin": {"LCB": 1.107240543567082, "EI": 0.7522382699410031}}
saved_yopt = np.load("yopt_20iter_1000run.npy",allow_pickle=True).item()

prob_type_l = ["LpNorm", "Branin"]
acq_type_l = ["LCB", "EI"]

mean_obj = {}
max_obj = {}
min_obj = {}
y_opt = {}

retval = 0
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
      y_opt[prob_type] = {}

   for acq_type in acq_type_l:
      if acq_type not in mean_obj[prob_type]:
         mean_obj[prob_type][acq_type] = 0
         max_obj[prob_type][acq_type] = -np.inf
         min_obj[prob_type][acq_type] = np.inf
         y_opt[prob_type][acq_type] = np.zeros(num_repeat)

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
         y_opt[prob_type][acq_type][n_repeat] = bo.getOptimalObjective()
         
         mean_obj[prob_type][acq_type] += y_opt[prob_type][acq_type][n_repeat]
         max_obj[prob_type][acq_type] = max(max_obj[prob_type][acq_type], y_opt[prob_type][acq_type][n_repeat])
         min_obj[prob_type][acq_type] = min(min_obj[prob_type][acq_type], y_opt[prob_type][acq_type][n_repeat])

#if(num_repeat >= 1000 ):
#   np.save("yopt_20iter_1000run.npy", y_opt)

# Define percentiles
left_percentile = 5  # or 1
right_percentile = 100 - left_percentile  # 95 or 99

print("Summary:")
for prob_type in prob_type_l:
   for acq_type in acq_type_l:
      allowed_error = max(1e-6, 0.01*(saved_max_obj[prob_type][acq_type]-saved_min_obj[prob_type][acq_type]))

      mean_obj[prob_type][acq_type] /= num_repeat
      print("(Min,Mean,Max) Opt.Obj for", prob_type, "-", acq_type, ":\t(", min_obj[prob_type][acq_type], ",",mean_obj[prob_type][acq_type], ",", max_obj[prob_type][acq_type], ")")
   
      ### verify the results with the saved results
      left_value = np.percentile(saved_yopt[prob_type][acq_type], left_percentile)
      right_value = np.percentile(saved_yopt[prob_type][acq_type], right_percentile)

      lb = left_value - allowed_error
      ub = right_value + allowed_error

      is_failed = (y_opt[prob_type][acq_type] < lb) | (y_opt[prob_type][acq_type] > ub)
      num_fail = np.sum(is_failed)

      if num_fail > 1:
         print(num_fail, "test(s) fail(s):", y_opt[prob_type][acq_type][is_failed])
         print("Recorded (min, mean, max): (", saved_min_obj[prob_type][acq_type], ",", saved_mean_obj[prob_type][acq_type], ",", saved_max_obj[prob_type][acq_type], ")")
         retval = 1

sys.exit(retval)




