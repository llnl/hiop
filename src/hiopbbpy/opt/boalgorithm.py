"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from numpy.random import uniform
from scipy.optimize import minimize
from scipy.stats import qmc
from ..surrogate_modeling.gp import GaussianProcess
from .acquisition import LCBacquisition, EIacquisition
from ..problems.problem import Problem
from .optproblem import IpoptProb
from ..utils.util import Evaluator, Logger

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
  def __init__(self):
    self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
    self.batch_type = "KB"        # strategy for qEI
    self.xtrain = None            # Training data
    self.ytrain = None            # Training data
    self.prob   = None            # Problem structure
    self.obj_evaluator = Evaluator()  # (batch) objective function evaluations
    self.opt_evaluator = Evaluator()  # (multi-start) local optimizer evaluations
    self.bo_maxiter = 20          # Maximum number of Bayesian optimization steps
    self.n_start = 10             # estimating acquisition global optima by determining local optima n_start times and then determining the discrete max of that set
    self.batch_size = 1           # batch size
    # save some internal member train
    self.y_hist = None            # History of evaluations
    self.x_hist = None            # History of evaluations
    self.x_opt = None             # Best observed point
    self.y_opt = None             # Best observed value
    self.idx_opt = None           # Index of the best observed value in the history
    self.logger = Logger()        # logger

  # Sets the acquisition function type and batch size
  def setAcquisitionType(self, acquisition_type, batch_size=1):
    self.acquisition_type = acquisition_type
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
    return y_opt

# A subclass of BOAlgorithmBase implementing a full Bayesian Optimization workflow
class BOAlgorithm(BOAlgorithmBase):
  def __init__(self, prob:Problem, gpsurrogate:GaussianProcess, xtrain, ytrain,
               user_grad = None,
               options = {}):
    super().__init__()
    
    assert isinstance(gpsurrogate, GaussianProcess)
    
    self.setTrainingData(xtrain, ytrain)
    self.prob = prob
    self.gpsurrogate = gpsurrogate
    self.bounds = self.gpsurrogate.get_bounds()
    self.fun_grad = None

    self.bo_maxiter = options.get('bo_maxiter', self.bo_maxiter)
    assert self.bo_maxiter > 0, f"Invalid bo_maxiter: {self.bo_maxiter }"
    
    self.solver_options = {"maxiter": 200}
    self.solver_options = options.get('solver_options', self.solver_options)

    acquisition_type = options.get('acquisition_type', "LCB")
    assert acquisition_type in ["LCB", "EI"], f"Invalid acquisition_type: {acquisition_type}"
    batch_size = options.get('batch_size', 1)
    assert isinstance(batch_size, int), f"batch_size {batch_size} not an integer"
    assert batch_size > 0, f"batch_size {batch_size} is not strictly positive"
    self.setAcquisitionType(acquisition_type, batch_size)

    self.obj_evaluator = options.get('obj_evaluator', self.obj_evaluator)
    assert isinstance(self.obj_evaluator, Evaluator)
    
    self.opt_evaluator = options.get('opt_evaluator', self.opt_evaluator)
    assert isinstance(self.opt_evaluator, Evaluator)

    self.logger.setlevel(options.get('log_level', "INFO"))

    if options and 'opt_solver' in options:
      opt_solver = options['opt_solver']
      assert opt_solver in ["SLSQP", "trust-constr", "IPOPT"], f"Invalid opt_solver: {opt_solver}"
    else:
      opt_solver = "SLSQP"

    if isinstance(prob.constraints, dict):
      assert opt_solver in ["trust-constr", "IPOPT"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a dict"
    elif isinstance(prob.constraints, list):
      assert opt_solver in ["SLSQP", "IPOPT"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a list of dict"

    self.set_method(opt_solver)

    if user_grad:
      self.fun_grad = user_grad

  # Method to train the GP model
  def _train_surrogate(self, x_train, y_train):
    self.gpsurrogate.train(x_train, y_train)

  # Method to find the best next sampling point via optimizing the acquisition function
  def _find_best_point(self, x_train, y_train, x0 = None):
    self._train_surrogate(x_train, y_train)
    if self.acquisition_type == "LCB":
      acqf = LCBacquisition(self.gpsurrogate)
    elif self.acquisition_type == "EI":
      acqf = EIacquisition(self.gpsurrogate)
    else:
      raise NotImplementedError("No implemented acquisition_type associated to"+self.acquisition_type)

    acqf_callback = {'obj' : acqf.scalar_evaluate}
    if acqf.has_gradient:
      acqf_callback['grad'] = acqf.scalar_eval_g

    x_all = []
    y_all = []
    acqf_minimizer = minimizer_wrapper(acqf_callback, self.opt_solver, self.bounds, self.prob.constraints, self.solver_options)

    if self.prob is not None:
      x0_pts = np.array([self.prob.sample(1)[0] for _ in range(self.n_start)])
    else:
      x0_pts = np.array([[uniform(b[0], b[1]) for b in self.bounds] for _ in range(self.n_start)])
    opt_output = self.opt_evaluator.run(acqf_minimizer.minimizer_callback, x0_pts)
    for ii in range(self.n_start):
      success = False
      xopt, yopt, success = opt_output[ii]
      if success:
        x_all.append(xopt)
        y_all.append(yopt) 
    if not x_all:
      raise RuntimeError("Optimization failed for all initial points — no solution found.")

    best_xopt = x_all[np.argmin(np.array(y_all))]
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
    
    n_init_sample = np.size(x_train, 0)

    self.x_hist = []
    self.y_hist = []

    for i in range(self.bo_maxiter):
      print(f"*****************************")
      print(f"Iteration {i+1}/{self.bo_maxiter}")

      y_train_virtual = y_train.copy() # old training + batch_size num of virtual points
      for j in range(self.batch_size):
        # Get a new sample point
        x_new = self._find_best_point(x_train, y_train_virtual)
        
        # Update training sample points
        x_train         = np.vstack([x_train,         x_new    ])

        # if this is not the last point in the current batch
        # then obtain a virtual point
        if j < max(range(self.batch_size)):
          # Get a virtual point
          y_virtual = self._get_virtual_point(np.atleast_2d(x_new))

          # Update training set with the virtual point
          y_train_virtual = np.vstack([y_train_virtual, y_virtual])
      
      y_new = self.obj_evaluator.run(self.prob.evaluate, x_train[-self.batch_size:])
      y_new = np.array(y_new)
      y_train = np.vstack([y_train, y_new])
      
      # Save the new sample points and objective evaluations
      for j in range(1, self.batch_size+1):
        self.x_hist.append(x_train[-j].flatten())
        self.y_hist.append(y_train[-j].flatten())
      if self.batch_size == 1:
        self.logger.info(f"Sample point X:")
      else:
        self.logger.info(f"Sample points X:")
      for j in range(self.batch_size):
        self.logger.info(f"{x_train[-j-1]}")
      if self.batch_size == 1:
        self.logger.info(f"Observation Y:")
      else:
        self.logger.info(f"Observations Y:")
      for j in range(self.batch_size):
        self.logger.info(f"{y_new[-j-1]}")

    # Save the optimal results and all the training data
    self.idx_opt = np.argmin(self.y_hist)
    self.x_opt = self.x_hist[self.idx_opt]
    self.y_opt = self.y_hist[self.idx_opt]
    self.setTrainingData(x_train, y_train)

    print(f"\n\nOptimal at BO iteration: {self.idx_opt+1} ")
    print(f"Optimal point: {self.x_opt.flatten()}, Optimal value: {self.y_opt}\n\n")

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
    for x0 in x0s:
      if self.method == "SLSQP":
        if 'grad' in self.fun:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, jac=self.fun['grad'], constraints=self.constraints, options=self.solver_options)
        else:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=self.constraints, options=self.solver_options)
        success = y.success
        if not success:
          self.logger.warning(y.message)
        xopt = y.x
        yopt = y.fun
      elif self.method == "trust-constr":
        nonlinear_constraint = NonlinearConstraint(self.constraints['cons'], self.constraints['cl'], self.constraints['cu'], jac=self.constraints['jac'])
        y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=[nonlinear_constraint], options=self.solver_options)
        success = y.success
        if not success:
          self.logger.warning(y.message)
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
          self.logger.warning(f"Ipopt failed to solve the problem. Status msg: {msg}")
          success = False
    
        yopt = info['obj_val']
        xopt = sol
      output.append([xopt, yopt, success])
    return output
