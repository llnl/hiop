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
from smt.applications.ego import Evaluator
from .bbproblem import BnBAlgorithm
from .minimizer import minimizer

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
    def __init__(self):
        self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
        self.batch_type = "KB"        # strategy for qEI
        self.xtrain = None            # Training data
        self.ytrain = None            # Training data
        self.prob   = None            # Problem structure
        self.evaluator = Evaluator()  # compute control objective evaluations
        self.bo_maxiter = 20          # Maximum number of Bayesian optimization steps
        self.n_start = 10             # estimating acquisition global optima by determining local optima n_start times and then determining the discrete max of that set
        self.batch_size = 1           # batch size
        # save some internal member train
        self.y_hist = None            # History of evaluations
        self.x_hist = None            # History of evaluations
        self.x_opt = None             # Best observed point
        self.y_opt = None             # Best observed value
        self.idx_opt = None           # Index of the best observed value in the history

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

        self.evaluator = options.get('evaluator', self.evaluator)
        assert isinstance(self.evaluator, Evaluator)

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


    # Method to set up a callback function to minimize the acquisition function
    def _setup_acqf_minimizer_callback(self):
        self.acqf_minimizer_callback = lambda fun, x0: minimizer(fun, x0, self.opt_solver, self.bounds, self.prob.constraints, self.solver_options)

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
      self._setup_acqf_minimizer_callback()

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
          
          y_new = self.evaluator.run(self.prob.evaluate, x_train[-self.batch_size:])
          y_train = np.vstack([y_train, y_new])
          
          # Save the new sample points and objective evaluations
          for j in range(1, self.batch_size+1):
              self.x_hist.append(x_train[-j].flatten())
              self.y_hist.append(y_train[-j].flatten())
          if self.batch_size == 1:
              print(f"Sample point X: {x_train[-self.batch_size:]}, Observation Y: {y_new}")
          else:
              print(f"Sample points X: {x_train[-self.batch_size:]}, Observations Y: {y_new}")


      # Save the optimal results and all the training data
      self.idx_opt = np.argmin(self.y_hist)
      self.x_opt = self.x_hist[self.idx_opt]
      self.y_opt = self.y_hist[self.idx_opt]
      self.setTrainingData(x_train, y_train)

      print(f"\n\nOptimal at BO iteration: {self.idx_opt+1} ")
      #if self.idx_opt < n_init_sample:
      #    print(f"Optimal at initial sample: {self.idx_opt+1}")
      #else:
      #    print(f"Optimal at BO iteration: {self.idx_opt-n_init_sample+1} ")
          
      print(f"Optimal point: {self.x_opt.flatten()}, Optimal value: {self.y_opt}")
      print()

def minimizer(fun, x0, method, bounds, constraints, solver_options):
    if method == "SLSQP":
        if 'grad' in fun:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, jac=fun['grad'], constraints=constraints, options=solver_options)
        else:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, constraints=constraints, options=solver_options)
        success = y.success
        if not success:
            print(y.message)
        xopt = y.x
        yopt = y.fun
    elif method == "trust-constr":
        nonlinear_constraint = NonlinearConstraint(constraints['cons'], constraints['cl'], constraints['cu'], jac=constraints['jac'])
        y = minimize(fun['obj'], x0, method=method, bounds=bounds, constraints=[nonlinear_constraint], options=solver_options)
        success = y.success
        if not success:
            print(y.message)
        xopt = y.x
        yopt = y.fun
    else:
        ipopt_prob = IpoptProb(fun['obj'], fun['grad'], constraints, bounds, solver_options)
        print(x0)
        sol, info = ipopt_prob.solve(x0)

        status = info.get('status', -999)
        msg = info.get('status_msg', b'unknown error')
        if status == 0:
            # ipopt returns 0 as success
            success = True
        else:
            warnings.warn(f"Ipopt failed to solve the problem. Status msg: {msg}")
            success = False

        yopt = info['obj_val']
        xopt = sol

    return xopt, yopt, success