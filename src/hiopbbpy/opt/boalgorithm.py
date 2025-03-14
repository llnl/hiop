"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from numpy.random import uniform
from scipy.optimize import minimize
from ..surrogate_modeling.gp import GaussianProcess
from .acquisition import LCBacquisition
from ..problems.problem import Problem

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
    def __init__(self):
        self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
        self.xtrain = None            # Training data
        self.ytrain = None            # Training data
        self.n_iter = 20              # Maximum number of optimization steps
        self.n_start = 10             # estimating acquisition global optima by determining local optima n_start times and then determining the discrete max of that set
        self.q = 1                    # batch size
        # save some internal member train
        self.y_hist = None            # History of evaluations
        self.x_hist = None            # History of evaluations
        self.x_opt = None             # Best observed point
        self.y_opt = None             # Best observed value
        self.idx_opt = None           # Index of the best observed value in the history

    # Sets the acquisition function type and batch size
    def setAcquisitionType(self, acquisition_type, q=1):
        self.acquisition_type = acquisition_type
        self.q = q

    # Sets the training data
    def setTrainingData(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    # Method to perform Bayesian optimization
    def optimize(self, fun):
        assert NotImplementedError("Child class of hiopEGO should implement method optimize")

    # Method to return the recorded optimization iterations and objectives
    def getOptimizationHistory(self):
        x_hist = np.array(self.x_hist, copy=True)
        y_hist = np.array(self.y_hist, copy=True)
        return x_hist, y_hist

    # Method to return the optimal solution and objective
    def getOptimalPoint(self):
        x_opt = np.array(self.x_opt, copy=True)
        y_opt = np.array(self.y_opt, copy=True)
        return x_opt, y_opt

# A subclass of BOAlgorithmBase implementing a full Bayesian Optimization workflow
class BOAlgorithm(BOAlgorithmBase):
    def __init__(self, gpsurrogate, xtrain, ytrain, acquisition_type = "LCB"):
        super().__init__()
        assert isinstance(gpsurrogate, GaussianProcess)
        assert acquisition_type in ["LCB"]
        self.setTrainingData(xtrain, ytrain)
        self.setAcquisitionType(acquisition_type)
        self.gpsurrogate = gpsurrogate
        self.n_iter = 20
        self.method = "SLSQP"
        self.bounds = self.gpsurrogate.get_bounds()
        self.constraints = ()
        self.options = {"maxiter": 200}
        self.acqf_minimizer_callback = None

    # Method to set up a callback function to minimize the acquisition function
    def _setup_acqf_minimizer_callback(self):
        self.acqf_minimizer_callback = lambda fun, x0: pyminimize(fun, x0, self.method, self.bounds, self.constraints, self.options)

    # Method to train the GP model
    def _train_surrogate(self, x_train, y_train):
        self.gpsurrogate.train(x_train, y_train)

    # Method to find the best next sampling point via optimizing the acquisition function
    def _find_best_point(self, x_train, y_train, x0 = None):
        self._train_surrogate(x_train, y_train)

        if self.acquisition_type == "LCB":
            acqf = LCBacquisition(self.gpsurrogate)
        else:
            raise NotImplementedError("No implemented acquisition_type associated to"+self.acquisition_type)

        acqf_callback = lambda x: float(np.array(acqf.evaluate(np.atleast_2d(x))).flat[0])
        
        x_all = []
        y_all = []
        for ii in range(self.n_start):
            success = False
            # Generate random starting point if x0 is not provided
            if x0 is None:
                x0 = np.array([uniform(b[0], b[1]) for b in self.bounds])
            xopt, yout, success = self.acqf_minimizer_callback(acqf_callback, x0)
            
            if success:
                x_all.append(xopt)
                y_all.append(yout)
        
        best_xopt = x_all[np.argmin(np.array(y_all))]

        return best_xopt

    # Set the optimization method
    def set_method(self, method):
        self.method = method

    # Set the user options for the Bayesian optimization
    def set_options(self, options):
        self.options = options

    # Method to perform Bayesian optimization
    def optimize(self, prob:Problem):
      x_train = self.xtrain
      y_train = self.ytrain
      
      n_init_sample = np.size(x_train,0)
      print(f"n_init_sample: {n_init_sample}")
      self._setup_acqf_minimizer_callback()

      self.x_hist = []
      self.y_hist = []

      for i in range(self.n_iter):
          print(f"*****************************")
          print(f"Iteration {i+1}/{self.n_iter}")

          # Get a new sample point
          x_new = self._find_best_point(x_train, y_train)

          # Evaluate the new sample point
          y_new = prob.evaluate(np.atleast_2d(x_new))

          # Update training set
          x_train = np.vstack([x_train, x_new])
          y_train = np.vstack([y_train, y_new])

          # Save the new sample point and its observation
          self.x_hist.append(x_new)
          self.y_hist.append(y_new)

          print(f"Sampled point X: {x_new.flatten()}, Observation Y: {y_new.flatten()}")

      # Save the optimal results and all the training data
      self.idx_opt = np.argmin(y_train)
      self.x_opt = x_train[self.idx_opt]
      self.y_opt = y_train[self.idx_opt]
      self.setTrainingData(x_train, y_train)

      print()
      print()
      if self.idx_opt < n_init_sample:
          print(f"Optimal at initial sample: {self.idx_opt+1}")
      else:
          print(f"Optimal at BO iteration: {self.idx_opt-n_init_sample+1} ")
          
      print(f"Optimal point: {self.x_opt.flatten()}, Optimal value: {self.y_opt}")


# Find the minimum of the input objective `fun`, using the minimize function from SciPy. 
def pyminimize(fun, x0, method, bounds, constraints, options):
    y = minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)
    success = y.success
    if not success:
        print(y.message)
    xopt = y.x
    yopt = y.fun
    return xopt, yopt, success
