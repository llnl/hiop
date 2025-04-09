"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from numpy.random import uniform
from scipy.optimize import minimize
from scipy.stats import qmc
import warnings
from ..surrogate_modeling.gp import GaussianProcess
from .acquisition import LCBacquisition, EIacquisition
from ..problems.problem import Problem
from .optproblem import IpoptProb

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
    def __init__(self):
        self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
        self.xtrain = None            # Training data
        self.ytrain = None            # Training data
        self.prob   = None            # Problem structure
        self.bo_maxiter = 20          # Maximum number of Bayesian optimization steps
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
    def __init__(self, gpsurrogate, xtrain, ytrain,
                 user_grad = None,
                 user_constraints = None,
                 options = None):
        super().__init__()
        
        assert isinstance(gpsurrogate, GaussianProcess)
        
        self.setTrainingData(xtrain, ytrain)
        self.gpsurrogate = gpsurrogate
        self.bounds = self.gpsurrogate.get_bounds()
        self.fun_grad = None
        self.constraints = None

        if options and 'bo_maxiter' in options:
            self.bo_maxiter = options['bo_maxiter']
            assert self.bo_maxiter > 0, f"Invalid bo_maxiter: {self.bo_maxiter }"

        if options and 'solver_options' in options:
            self.solver_options = options['solver_options']
        else:
            self.solver_options = {"maxiter": 200}

        if options and 'acquisition_type' in options:
            acquisition_type = options['acquisition_type']
            assert acquisition_type in ["LCB", "EI"], f"Invalid acquisition_type: {acquisition_type}"
        else:
            acquisition_type = "LCB"
        self.setAcquisitionType(acquisition_type)

        if options and 'opt_solver' in options:
            opt_solver = options['opt_solver']
            assert opt_solver in ["SLSQP", "IPOPT"], f"Invalid opt_solver: {opt_solver}"
        else:
            opt_solver = "SLSQP"
        self.set_method(opt_solver)

        if user_constraints:
            self.constraints = user_constraints

        if user_grad:
            self.fun_grad = user_grad


    # Method to set up a callback function to minimize the acquisition function
    def _setup_acqf_minimizer_callback(self):
        self.acqf_minimizer_callback = lambda fun, x0: minimizer(fun, x0, self.opt_solver, self.bounds, self.constraints, self.solver_options)

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

        acqf_obj_callback = lambda x: float(np.array(acqf.evaluate(np.atleast_2d(x))).flat[0])
        acqf_callback = {'obj': acqf_obj_callback}
        if acqf.has_gradient == True:
            acqf_grad_callback = lambda x: np.array(acqf.eval_g(np.atleast_2d(x)))
            acqf_callback['grad'] = acqf_grad_callback

        x_all = []
        y_all = []
        for ii in range(self.n_start):
            success = False
            # Generate random starting point if x0 is not provided
            if x0 is None and self.prob is not None:
                x0 = self.prob.sample(1)[0]
            else:
                x0 = np.array([uniform(b[0], b[1]) for b in self.bounds])
            xopt, yout, success = self.acqf_minimizer_callback(acqf_callback, x0)

            if success:
                x_all.append(xopt)
                y_all.append(yout)

        if not x_all:
            raise RuntimeError("Optimization failed for all initial points — no solution found.")

        best_xopt = x_all[np.argmin(np.array(y_all))]

        return best_xopt

    # Set the optimization method
    def set_method(self, method):
        self.opt_solver = method

    # Set the options for the internal optimization solver
    def set_options(self, solver_options):
        self.solver_options = solver_options

    # Method to perform Bayesian optimization
    def optimize(self, prob:Problem):
      self.prob = prob
      x_train = self.xtrain
      y_train = self.ytrain
      
      n_init_sample = np.size(x_train,0)
      print(f"n_init_sample: {n_init_sample}")
      self._setup_acqf_minimizer_callback()

      self.x_hist = []
      self.y_hist = []

      for i in range(self.bo_maxiter):
          print(f"*****************************")
          print(f"Iteration {i+1}/{self.bo_maxiter}")

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
      self.idx_opt = np.argmin(self.y_hist)
      self.x_opt = self.y_hist[self.idx_opt]
      self.y_opt = self.y_hist[self.idx_opt]
      self.setTrainingData(x_train, y_train)

      print()
      print()
      print(f"Optimal at BO iteration: {self.idx_opt+1} ")
      #if self.idx_opt < n_init_sample:
      #    print(f"Optimal at initial sample: {self.idx_opt+1}")
      #else:
      #    print(f"Optimal at BO iteration: {self.idx_opt-n_init_sample+1} ")
          
      print(f"Optimal point: {self.x_opt.flatten()}, Optimal value: {self.y_opt}")
      print()

# Find the minimum of the input objective `fun`, using the minimize function from SciPy. 
def minimizer(fun, x0, method, bounds, constraints, solver_options):
    if method != "IPOPT":
        if 'grad' in fun:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, jac=fun['grad'], constraints=constraints, options=solver_options)
        else:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, constraints=constraints, options=solver_options)
        success = y.success
        if not success:
            print(y.message)
        xopt = y.x
        yopt = y.fun
    else:
        ipopt_prob = IpoptProb(fun['obj'], fun['grad'], constraints, bounds, solver_options)
        sol, info = ipopt_prob.solve(x0)

        status = info.get('status', -1)
        msg = info.get('status_msg', -1)
        if status == 0:
            # ipopt returns 0 as success
            success = True
        else:
            warnings.warn(f"Ipopt failed to solve the problem. Status msg: {msg}")
            success = False

        yopt = info['obj_val']
        xopt = sol

    return xopt, yopt, success
