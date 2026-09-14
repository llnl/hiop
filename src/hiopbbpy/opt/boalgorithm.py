"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from numpy.random import uniform
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import qmc
from ..surrogate_modeling.gp import GaussianProcess
from .acquisition import LCBacquisition, EIacquisition
from ..problems.problem import Problem
from .optproblem import IpoptProb
from ..utils.util import Evaluator, Logger
import os

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
  def __init__(self):
    self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
    self.batch_type = "KB"        # Batched BO strategy
    self.xtrain = None            # Training data
    self.ytrain = None            # Training data
    self.init_ntrain = 0          # Initial (prior to BO optimization) number of GP training pts 
    self.prob   = None            # Problem structure
    self.obj_evaluator = Evaluator()  # (batch) objective function evaluations
    self.opt_evaluator = Evaluator()  # (multi-start) local optimizer evaluations
    self.bo_maxiter = 20          # Maximum number of Bayesian optimization steps
    self.n_start = 10             # estimating acquisition global optima by determining local optima n_start times and then determining the discrete max of that set
    self.batch_size = 1           # batch size
    # save some internal member train
    self.y_hist = None            # History of evaluations
    self.x_hist = None            # History of evaluations
    self.x_BO_opt = None          # Best point generated via BO
    self.y_BO_opt = None          # Best objective value generated via BO
    self.x_opt = None             # Best feasible point  (BO + feasible initial training points)
    self.y_opt = None             # Best objective value (BO + feasible initial training points) 
    self.logger = Logger()        # logger

  # Sets the acquisition function type and batch size
  def setAcquisitionType(self, acquisition_type, batch_size=1):
    self.acquisition_type = acquisition_type
    assert isinstance(batch_size, int), f"batch_size {batch_size} not an integer"
    assert batch_size > 0, f"batch_size {batch_size} is not strictly positive"
    self.batch_size = batch_size

  # Sets the training data
  def setTrainingData(self, xtrain, ytrain):
    self.xtrain = xtrain
    self.ytrain = ytrain

  # Method to perform Bayesian optimization
  def optimize(self, fun):
    raise NotImplementedError("Child class of hiopEGO should implement method optimize")

  # Method to return the recorded optimization iterations and objectives
  def getOptimizationHistory(self):
    x_hist = np.array(self.x_hist, copy=True)
    y_hist = np.array(self.y_hist, copy=True)
    return x_hist, y_hist

  # Method to return the optimal solution 
  def getOptimalPoint(self):
    x_opt = np.array(self.x_opt, copy=True)
    return x_opt

  # Method to return the optimal objective
  def getOptimalObjective(self):
    y_opt = np.array(self.y_opt, copy=True)
    return y_opt[0]

# A subclass of BOAlgorithmBase implementing a full Bayesian Optimization workflow
class BOAlgorithm(BOAlgorithmBase):
  def __init__(self, prob:Problem, gpsurrogate:GaussianProcess, xtrain, ytrain,
               user_grad = None,
               options = {}):
    super().__init__()
    assert isinstance(gpsurrogate, GaussianProcess)
    assert len(xtrain) == len(ytrain), "xtrain, ytrain must be the same length"
    assert ytrain.ndim == 2 and ytrain.shape[1] == 1, "ytrain must be a (n, 1) array"
    assert xtrain.ndim == 2, "xtrain must be a (n, d) array"
    self.setTrainingData(xtrain, ytrain)
    self.init_ntrain = len(ytrain)
    self.prob = prob
    self.gpsurrogate = gpsurrogate
    self.bounds = self.gpsurrogate.get_bounds()
    self.fun_grad = None

    logger_level = options.get('log_level', "INFO")
    self.logger.setlevel(logger_level)

    self.bo_maxiter = options.get('bo_maxiter', self.bo_maxiter)
    assert self.bo_maxiter > 0, f"Invalid bo_maxiter: {self.bo_maxiter}"

    self.n_start = options.get('n_start', self.n_start)
    assert self.n_start > 0, f"Invalid n_start: {self.n_start}"

    acquisition_type = options.get('acquisition_type', "LCB")
    assert acquisition_type in ["LCB", "EI"], f"Invalid acquisition_type: {acquisition_type}"

    batch_size = options.get('batch_size', 1)

    self.setAcquisitionType(acquisition_type, batch_size)

    self.obj_evaluator = options.get('obj_evaluator', self.obj_evaluator)
    assert isinstance(self.obj_evaluator, Evaluator)
    
    self.opt_evaluator = options.get('opt_evaluator', self.opt_evaluator)
    assert isinstance(self.opt_evaluator, Evaluator)

    if options and 'opt_solver' in options:
      opt_solver = options['opt_solver']
      assert opt_solver in ["SLSQP", "trust-constr", "IPOPT"], f"Invalid opt_solver: {opt_solver}"
    else:
      opt_solver = "SLSQP"

    if isinstance(prob.constraints, dict):
      assert opt_solver in ["trust-constr", "IPOPT"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a dict"
    elif isinstance(prob.constraints, list):
      assert opt_solver in ["SLSQP", "IPOPT"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a list of dict"

    if opt_solver == "SLSQP" or opt_solver == "trust-constr":
      self.solver_options = {"maxiter": 200}  #for scipy solvers
      self.solver_options = options.get('solver_options', self.solver_options)
    elif opt_solver == "IPOPT":
      self.solver_options = {"max_iter": 200, "print_level": 1}
      self.solver_options = options.get('solver_options', self.solver_options)
      self.solver_options['sb'] = 'yes'

    self.set_method(opt_solver)

    if user_grad:
      self.fun_grad = user_grad

    self.logger.info(f"Problem name: {prob.name}")
    self.logger.info(f"Max BO iter: {self.bo_maxiter}")
    self.logger.info(f"Optimizing acquisition ({self.acquisition_type}) "
                     f"with {self.n_start} random initial points")
    self.logger.info(f"Batch type: {self.batch_type}")
    self.logger.info(f"Batch size: {batch_size}")
    self.logger.info(f"Internal optimization solver: {opt_solver}")
    self.logger.info(f"Internal optimization solver options: {self.solver_options}")
    self.logger.info(f"Initial training set: {xtrain.shape[0]} samples, {xtrain.shape[1]} dimensions")
    self.logger.debug(f"Bounds on optimization variable: {self.bounds}")
    self.logger.info(f"Logger level: {logger_level}")

  # Method to train the GP model
  def _train_surrogate(self, x_train, y_train):
    self.logger.debug("Training surrogate model with "
                      f"{x_train.shape[0]} samples...")
    self.gpsurrogate.train(x_train, y_train)
    self.logger.debug("Surrogate training complete.")

  # Method to find the best next sampling point via optimizing the acquisition function
  def _find_best_point(self, x_train, y_train, x0 = None):
    self.logger.info(f"Start finding the best sampling point:")
    self._train_surrogate(x_train, y_train)
    if self.acquisition_type == "LCB":
      acqf = LCBacquisition(self.gpsurrogate)
    elif self.acquisition_type == "EI":
      acqf = EIacquisition(self.gpsurrogate)
    else:
      raise NotImplementedError("No implemented acquisition_type associated to"+self.acquisition_type)

    acqf_callback = {'obj' : acqf.scalar_evaluate}
    if acqf.has_gradient:
      self.logger.debug(f"  Using gradient information of the acquisition function.")
      acqf_callback['grad'] = acqf.scalar_eval_g

    acqf_minimizer = minimizer_wrapper(acqf_callback, self.opt_solver, self.bounds, self.prob.constraints, self.solver_options)

    if self.prob is not None:
      x0_pts = np.array([self.prob.sample(1)[0] for _ in range(self.n_start)])
    else:
      x0_pts = np.array([[uniform(b[0], b[1]) for b in self.bounds] for _ in range(self.n_start)])

    opt_output = self.opt_evaluator.run(acqf_minimizer.minimizer_callback, x0_pts)

    x_all = []
    y_all = []
    n_failures = 0
    for ii in range(self.n_start):
      success = False
      xopt, yopt, success, msg = opt_output[ii]
      if success:
        x_all.append(xopt)
        y_all.append(yopt)
      else:
        n_failures += 1
        self.logger.debug(f"Acquisition optimizer failed at start {ii}: {msg}")

    if not x_all:
      self.logger.error("All acquisition minimizations failed.")
      raise RuntimeError("Optimization failed for all initial points — no solution found.")

    # Compute some stats
    y_all = np.array(y_all)
    best_xopt = x_all[np.argmin(y_all)]
    y_min, y_max, y_mean = np.min(y_all), np.max(y_all), np.mean(y_all)

    self.logger.scalars(
        f"  Acquisition optimization finished with {len(y_all)} successes, {n_failures} failures"
    )
    self.logger.scalars(
        f"  Acquisition values: min = {y_min:.4e}, mean = {y_mean:.4e}, max = {y_max:.4e}"
    )
    self.logger.debug(f"Estimated optimal point x: {best_xopt}")

    return best_xopt
  
  def _get_virtual_point(self, x):
    if self.batch_type not in ["CLmin", "KB", "KBUB", "KBLB", "KBRand"]:
      raise NotImplementedError("No implemented batch_type associated to"+self.batch_type)
    # constant-liar, Kriging-believer and Kriging-believer variants
    if self.batch_type == "CLmin":
      return min(self.gpsurrogate.training_y)
    elif self.batch_type == "KB":
      beta = 0.
    elif self.batch_type == "KBUB":
      beta = 3.0
    elif self.batch_type == "KBLB":
      beta = -3.0
    elif self.batch_type == "KBRand":
      beta = np.random.randn()
    return self.gpsurrogate.mean(x) + beta * np.sqrt(self.gpsurrogate.variance(x))

  # Set the optimization method
  def set_method(self, method):
    self.opt_solver = method

  # Set the options for the internal optimization solver
  def set_options(self, solver_options):
    self.solver_options = solver_options

  # Method to perform Bayesian optimization
  def optimize(self):
    x_train = self.xtrain
    y_train = self.ytrain
    self.logger.iterations(f"Best UNCONSTRAINED objective from {np.size(x_train, 0)} initial samples: {np.min(y_train):.4e} ")

    # determine which initial training points are feasible
    fea_idxs = self.prob.if_feasible(x_train) & np.isfinite(y_train).ravel() # feasible points with finite objectives
    y_train_fea = y_train[fea_idxs] 
    x_train_fea = x_train[fea_idxs]

    # determine the most optimal objective value from the set of feasible initial training points
    if y_train_fea.size > 0:
      best_fea_idx = np.argmin(y_train_fea)
      best_constrained_train_y = y_train_fea[best_fea_idx][0]
      best_constrained_train_x = x_train_fea[best_fea_idx]
      
      feasible_train_indices = np.flatnonzero(fea_idxs)
      train_idx_opt = int(feasible_train_indices[best_fea_idx])

      self.logger.info(
          f"Best objective: {best_constrained_train_y:.4e} from {y_train_fea.size} feasible initial training points"
        )

    else:
      best_constrained_train_y = np.inf
      self.logger.info("No feasible initial training points.")

    self.x_hist = []
    self.y_hist = []
    
    prev_best_y = best_constrained_train_y
    for i in range(self.bo_maxiter):
      self.logger.critical(f"*****************************")
      self.logger.critical(f"Iteration {i+1}/{self.bo_maxiter}")

      y_train_virtual = y_train.copy() # old training + batch_size num of virtual points
      for j in range(self.batch_size):
        # Get a new sample point
        self.logger.scalars(f"In batch {j+1}/{self.batch_size}")
        x_new = self._find_best_point(x_train, y_train_virtual)
        
        # Update training sample points
        x_train = np.vstack([x_train, x_new])

        # if this is not the last point in the current batch
        # then obtain a virtual point
        if j < max(range(self.batch_size)):
          # Get a virtual point
          y_virtual = self._get_virtual_point(np.atleast_2d(x_new))

          # Update training set with the virtual point
          y_train_virtual = np.vstack([y_train_virtual, y_virtual])

        mean_val = self.gpsurrogate.mean(np.array([x_new])).item()
        sd_val = np.sqrt(self.gpsurrogate.variance(np.array([x_new])).item())
        self.logger.scalars(f"  (mu, sigma) at new sample x: {mean_val}, {sd_val} ")
      
      y_new = self.obj_evaluator.run(self.prob.evaluate, x_train[-self.batch_size:])
      y_new = np.array(y_new)
      y_train = np.vstack([y_train, y_new])

      feas_new = self.prob.if_feasible(x_train[-self.batch_size:])
      self.logger.debug(f"Feasible samples: {np.sum(feas_new)}/{self.batch_size}")

      min_y_new = np.min(y_new)
      curr_best_y = np.min([prev_best_y, min_y_new])

      self.logger.iterations(f"Best objective found in this iteration: {min_y_new:.4e} ")
      self.logger.scalars(f"Training set size is now {x_train.shape[0]}")
      self.logger.iterations(f"Current best objective: {curr_best_y:.4e} "
                             f"(previous best: {prev_best_y:.4e})")
      self.logger.scalars(f"Objective function improvement: {prev_best_y - curr_best_y:.4e}")

      # Save the new sample points and objective evaluations
      for j in range(1, self.batch_size+1):
        self.x_hist.append(x_train[-j].flatten())
        self.y_hist.append(y_train[-j].flatten())

      if self.batch_size == 1:
        self.logger.debug(f"Sample point X:")
      else:
        self.logger.debug(f"Sample points X:")
      for j in range(self.batch_size):
        self.logger.debug(f"  {x_train[-j-1]}")

      if self.batch_size == 1:
        self.logger.debug(f"Observation Y:")
      else:
        self.logger.debug(f"Observations Y:")
      for j in range(self.batch_size):
        self.logger.debug(f"  {y_new[-j-1]}")

      prev_best_y = curr_best_y

    self.setTrainingData(x_train, y_train)
    
    # Save the BO optimal (excluding initial training pts) results
    idx_BO_opt = np.argmin(self.y_hist)
    self.x_BO_opt = self.x_hist[idx_BO_opt]
    self.y_BO_opt = self.y_hist[idx_BO_opt][0]
    
    self.logger.critical("===================================")
    self.logger.critical("Bayesian Optimization completed")
    self.logger.critical(f"Total objective evaluations for initial samples: {self.init_ntrain}")
    self.logger.critical(f"Total objective evaluations for BO iterations: {len(self.y_hist)}")
    
    if np.isfinite(self.y_BO_opt) or y_train_fea.size > 0:
      if self.y_BO_opt < best_constrained_train_y:
        self.logger.critical(f"Optimal at BO iteration: {idx_BO_opt//self.batch_size+1} ")
      else:
        self.logger.critical(f"BO did not generate points more optimal than initial training points")
      self.logger.critical(f"Best (BO) point: {self.x_BO_opt.flatten()}")
      self.logger.critical(f"Best (BO) objective value: {self.y_BO_opt}")
      
      if self.y_BO_opt < best_constrained_train_y:
        self.x_opt = self.x_BO_opt
        self.y_opt = self.y_BO_opt
      else: # y_BO_opt finite and best_constrained_train_y <= y_BO_opt means best_constrained_train_y is not inf and there is at least one feasible training point
        self.logger.critical(f"Optimal at training point: {train_idx_opt}")
        self.logger.critical(f"Best (training) point: {best_constrained_train_x.flatten()}")
        self.logger.critical(f"Best (training) point objective value: {best_constrained_train_y}")
        self.x_opt = best_constrained_train_x
        self.y_opt = best_constrained_train_y
    else:
      self.logger.critical(f"BO or consistent BB objective failure")
      self.logger.critical(f"None of the BO points resulted in finite BB objective functions evaluations")
      self.logger.critical(f"None of the initial training points were both feasible and had finite BB objective function values")
    self.logger.critical("===================================")
    self.y_opt = np.array([self.y_opt])


class minimizer_wrapper:
  def __init__(self, fun, method, bounds, constraints, solver_options):
    self.fun = fun
    self.method = method
    self.bounds = bounds
    self.constraints = constraints
    self.solver_options = solver_options
  # Find the minimum of the input objective `fun`, using the minimize function from SciPy. 
  def minimizer_callback(self, x0s):
    output = []
    print(f"Worker pid={os.getpid()}: doing minimizer_callback ...", flush=True)
    msg = ""
    for x0 in x0s:
      if self.method == "SLSQP":
        if 'grad' in self.fun:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, jac=self.fun['grad'], constraints=self.constraints, options=self.solver_options)
        else:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=self.constraints, options=self.solver_options)
        success = y.success
        if not success:
          msg = y.message
        xopt = y.x
        yopt = y.fun
      elif self.method == "trust-constr":
        constraints = []
        if self.constraints:  # non-empty dict → constrained problem
          nonlinear_constraint = NonlinearConstraint(
              self.constraints['cons'],
              self.constraints['cl'],
              self.constraints['cu'],
              jac=self.constraints.get('jac', None)
          )
          constraints.append(nonlinear_constraint)

        y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=constraints, options=self.solver_options)
        success = y.success
        if not success:
          msg = y.message
        xopt = y.x
        yopt = y.fun
      else:
        ipopt_prob = IpoptProb(self.fun['obj'], self.fun['grad'], self.constraints, self.bounds, self.solver_options)
        sol, info = ipopt_prob.solve(x0)
    
        status = info.get('status', -999)
        msg = info.get('status_msg', b'unknown error')
        if status == 0:
          # ipopt returns 0 as success
          success = True
        else:
          msg = f"Ipopt failed to solve the problem. Status msg: {msg}"
          success = False
    
        yopt = info['obj_val']
        xopt = sol
      output.append([xopt, yopt, success, msg])
    return output
