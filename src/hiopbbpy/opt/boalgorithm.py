"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..surrogate_modeling.gp import GaussianProcess
from ..surrogate_modeling.krg import smt_theta_bounds
from .acquisition import LCBacquisition, EIacquisition
from ..problems.problem import Problem
from ..utils.util import Evaluator, Logger
from .bnbalgorithm import BnBAlgorithm, AffordableLCBTransfer
from .opt_utils import minimizer_wrapper
from .optproblem import IpoptProb
import os

def _smt_option(options, name, default=None):
  try:
    return options[name] if name in options else default
  except Exception:
    try:
      return options[name]
    except Exception:
      return default


def _smt_se_geometry(gpsurrogate):
  """Geometry of the fitted SE/pow-exp(p=2) SMT model."""
  if not hasattr(gpsurrogate, "surrogatesmt"):
    raise TypeError("These diagnostics require the smtKRG surrogate")

  sm = gpsurrogate.surrogatesmt
  corr = str(_smt_option(sm.options, "corr", "pow_exp")).lower()
  power = float(_smt_option(sm.options, "pow_exp_power", 2.0))

  if corr not in ("pow_exp", "squar_exp"):
    raise NotImplementedError(
        f"Clustering diagnostics currently assume an SE kernel; corr={corr}"
    )

  if corr == "pow_exp" and not np.isclose(power, 2.0):
    raise NotImplementedError(
        f"Clustering diagnostics currently assume pow_exp_power=2; "
        f"got {power}"
    )

  theta = getattr(sm, "optimal_theta", None)
  if theta is None:
    theta = sm.corr.theta

  theta = np.asarray(theta, dtype=float).reshape(-1)
  if theta.size == 1:
    theta = np.repeat(theta, gpsurrogate.ndim)

  if theta.size != gpsurrogate.ndim:
    raise RuntimeError(
        f"Expected {gpsurrogate.ndim} theta values, got {theta.size}"
    )

  x_offset = np.asarray(sm.X_offset, dtype=float).reshape(-1)
  x_scale = np.asarray(sm.X_scale, dtype=float).reshape(-1)

  if np.any(np.abs(x_scale) <= np.finfo(float).tiny):
    raise RuntimeError("SMT returned a zero input scale")

  return theta, x_offset, x_scale


def _domain_normalize(gpsurrogate, x):
  """Normalize the points to the original BO domain [0,1]^n."""
  x = np.atleast_2d(np.asarray(x, dtype=float))
  xlimits = np.asarray(gpsurrogate.xlimits, dtype=float)
  widths = xlimits[:, 1] - xlimits[:, 0]

  if np.any(widths <= 0.0):
    raise RuntimeError("All BO-domain widths must be positive")

  return (x - xlimits[:, 0]) / widths


def _pairwise_euclidean(x):
  """Full Euclidean pairwise-distance matrix."""
  delta = x[:, None, :] - x[None, :, :]
  return np.sqrt(
      np.maximum(0.0, np.sum(delta * delta, axis=2))
  )


def _se_kernel_distance_and_correlation(
    x_left_smt,
    x_right_smt,
    theta,
):
  """Pairwise SE kernel distance and correlation.

  The inputs must already be in SMT-standardized coordinates.

      distance^2 = sum_j theta_j * delta_j^2
      correlation = exp(-distance^2)
  """
  x_left_smt = np.atleast_2d(
      np.asarray(x_left_smt, dtype=float)
  )
  x_right_smt = np.atleast_2d(
      np.asarray(x_right_smt, dtype=float)
  )

  delta = (
      x_left_smt[:, None, :]
      - x_right_smt[None, :, :]
  )

  distance_squared = np.sum(
      theta[None, None, :] * delta * delta,
      axis=2,
  )

  distance = np.sqrt(
      np.maximum(0.0, distance_squared)
  )
  correlation = np.exp(-distance_squared)

  return distance, correlation


def _sample_set_clustering_metrics(gpsurrogate, x_train):
  """
  Clustering metrics for the sample set used by the current BO step.

  domain_nn_*: Euclidean distances after normalizing each coordinate by the original domain width.
  domain_nn_p01, p05, and p50: percentiles over the per-sample nearest-neighbor distances, not over all pair distances.
  smt_nn: ordinary Euclidean distance in SMT-standardized coordinates.
  kernel_nn: theta-weighted distance 
  pairs_corr_ge_*: number of unordered sample pairs; each pair is counted once.
  nearest_old_index: zero-based index of the old point nearest in the kernel metric.
  domain_nn and smt_nn are independently minimized, so their nearest points can differ from nearest_old_index for an anisotropic GP.
  corr_max_offdiag off-diagonal maximal correlation in the covariance matrix
  """
  
  x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
  n_train = x_train.shape[0]

  if n_train < 2:
    return {"domain_nn_min": np.nan, "domain_nn_p01": np.nan, "domain_nn_p05": np.nan, "domain_nn_p50": np.nan,
            "kernel_nn_min": np.nan, "corr_max_offdiag": np.nan, "pairs_corr_ge_0p95": 0, "pairs_corr_ge_0p99": 0}

  theta, x_offset, x_scale = _smt_se_geometry(gpsurrogate)

  # Domain-normalized Euclidean distances.
  x_domain = _domain_normalize(gpsurrogate, x_train,)
  domain_dist = _pairwise_euclidean(x_domain)

  # Exclude self-distance when finding the nearest neighbor.
  np.fill_diagonal(domain_dist, np.inf)
  domain_nn = np.min(domain_dist, axis=1)

  # SMT-standardized and theta-weighted distances.
  x_smt = (x_train - x_offset) / x_scale
  kernel_dist, correlation = _se_kernel_distance_and_correlation(x_smt, x_smt, theta)

  np.fill_diagonal(kernel_dist, np.inf)
  kernel_nn = np.min(kernel_dist, axis=1)

  # Extract each unordered pair exactly once.
  ii, jj = np.triu_indices(n_train, k=1)
  corr_offdiag = correlation[ii, jj]

  return {
      "domain_nn_min": float(np.min(domain_nn)),
      "domain_nn_p01": float(np.percentile(domain_nn, 1.0)),
      "domain_nn_p05": float(np.percentile(domain_nn, 5.0)),
      "domain_nn_p50": float(np.percentile(domain_nn, 50.0)),
      "kernel_nn_min": float(np.min(kernel_nn)),
      "corr_max_offdiag": float(np.max(corr_offdiag)),
      "pairs_corr_ge_0p95": int(np.count_nonzero(corr_offdiag >= 0.95)),
      "pairs_corr_ge_0p99": int(np.count_nonzero(corr_offdiag >= 0.99)),
  }


def _new_point_clustering_metrics(gpsurrogate, old_x, x_new):
  """Distances from x_new to the samples present when it was selected."""
  old_x = np.atleast_2d(np.asarray(old_x, dtype=float))
  x_new = np.asarray(x_new, dtype=float,).reshape(1, -1)

  theta, x_offset, x_scale = _smt_se_geometry(gpsurrogate)

  # Distance in the original domain-normalized coordinates.
  old_domain = _domain_normalize(gpsurrogate, old_x)
  new_domain = _domain_normalize(gpsurrogate, x_new)
  domain_dist = np.linalg.norm(old_domain - new_domain, axis=1)

  # Distance in SMT-standardized coordinates.
  old_smt = (old_x - x_offset) / x_scale
  new_smt = (x_new - x_offset) / x_scale
  smt_dist = np.linalg.norm(old_smt - new_smt, axis=1)

  # Theta-weighted kernel distance and exact SE correlation.
  kernel_dist, correlation = _se_kernel_distance_and_correlation(new_smt, old_smt, theta)

  kernel_dist = kernel_dist.reshape(-1)
  correlation = correlation.reshape(-1)

  # Define nearest_old_index using the GP/kernel metric.
  nearest_old_index = int(np.argmin(kernel_dist))

  return {
      "domain_nn": float(np.min(domain_dist)),
      "smt_nn": float(np.min(smt_dist)),
      "kernel_nn": float(kernel_dist[nearest_old_index]),
      "kernel_corr_to_nearest": float(correlation[nearest_old_index]),
      "nearest_old_index": nearest_old_index,
  }

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
  def __init__(self):
    self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")

    self.LCB_beta = 3.0
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
    self.nretraingp = 1           # number of BO iterations after which the GP is fully retrained
    # save some internal member training data and true function evaluations (that do not end up in the GP train)
    self.y_hist = None            # History of evaluations
    self.x_hist = None            # History of evaluations
    self.x_BO_opt = None          # Best point generated via BO
    self.y_BO_opt = None          # Best objective value generated via BO
    self.x_opt = None             # Best feasible point  (BO + feasible initial training points)
    self.y_opt = None             # Best objective value (BO + feasible initial training points)
    self.x_evaluated = None       # points at which true function was evaluated
    self.y_evaluated = None       # evaluated values for the above
    self.logger = Logger()        # logger
    self.bnb_num_branch_hist = [] # number of BnB branches visited per BO iter

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

  def getEvaluationArchive(self):
    return (np.array(self.x_evaluated, copy=True), np.array(self.y_evaluated, copy=True))
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

    self.nretraingp = options.get('nretraingp', self.nretraingp)
    if(type(self.nretraingp) is not int or self.nretraingp < 1):
      raise ValueError("nretraingp must be a positive integer")
    
    acquisition_type = options.get('acquisition_type', "LCB")
    assert acquisition_type in ["LCB", "EI"], f"Invalid acquisition_type: {acquisition_type}"

    self.LCB_beta = options.get('LCB_beta', self.LCB_beta)
    assert self.LCB_beta > 0., f"Invalid LCB beta (variance penalty): {self.LCB_beta}"

    batch_size = options.get('batch_size', 1)
    self.setAcquisitionType(acquisition_type, batch_size)

    self.obj_evaluator = options.get('obj_evaluator', self.obj_evaluator)
    assert isinstance(self.obj_evaluator, Evaluator)
    
    self.opt_evaluator = options.get('opt_evaluator', self.opt_evaluator)
    assert isinstance(self.opt_evaluator, Evaluator)

    if options and 'opt_solver' in options:
      opt_solver = options['opt_solver']
      assert opt_solver in ["SLSQP", "trust-constr", "IPOPT", "BnB"], f"Invalid opt_solver: {opt_solver}"
    else:
      opt_solver = "SLSQP"

    if isinstance(prob.constraints, dict):
      assert opt_solver in ["trust-constr", "IPOPT", "BnB"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a dict"
    elif isinstance(prob.constraints, list):
      assert opt_solver in ["SLSQP", "IPOPT", "BnB"], f"Invalid opt_solver: {opt_solver} while constraints are defined as a list of dict"

    if opt_solver == "SLSQP" or opt_solver == "trust-constr":
      self.solver_options = {"maxiter": 200}  #for scipy solvers
      self.solver_options = options.get('solver_options', self.solver_options)
    elif opt_solver == "IPOPT":
      self.solver_options = {"max_iter": 200, "print_level": 1}
      self.solver_options = options.get('solver_options', self.solver_options)
      self.solver_options['sb'] = 'yes'
    elif opt_solver == "BnB":
      self.solver_options = {}
      self.solver_options = options.get('solver_options', self.solver_options)

    self.opt_solver = opt_solver

    # BnB batching 
    self.bnb_batch_method = options.get("bnb_batch_method", "kmeans")
    self.bnb_batch_options = dict(options.get("bnb_batch_options", {}))

    self.bnb_batch_max_add = options.get("bnb_batch_max_add", 1)
    if (isinstance(self.bnb_batch_max_add, bool) or not isinstance(self.bnb_batch_max_add, (int, np.integer))
        or self.bnb_batch_max_add < 1 or self.bnb_batch_max_add > self.batch_size):
      raise ValueError("bnb_batch_max_add must be an integer and satisfy 1 <= bnb_batch_max_add <= batch_size")
    
    if self.bnb_batch_method not in ("kmeans", "conditional_variance"):
      raise ValueError("Unknown bnb_batch_method")

    if self.bnb_batch_method == "conditional_variance":
      if self.opt_solver != "BnB" or self.acquisition_type != "LCB":
        raise ValueError("Conditional-variance batching requires BnB and LCB")
      self.batch_type = "BnB-CV"
      self.logger.info(f"BnB-CV GP additions per iteration: at most {self.bnb_batch_max_add}")

    # BnB default initializations 
    self.bnb_queue = None  # legacy; not a complete spatial partition
    self.bnb_partition = None
    self.bnb_lower_bound_transfer = options.get('BnBLowerBoundTransfer', None)
    if user_grad:
      self.fun_grad = user_grad

    self.bnb_warm_start = True
    self.bnb_warm_start = options.get('bnb_warmstart', self.bnb_warm_start)
    assert isinstance(self.bnb_warm_start, bool), "provided bnb_warmstart is not a boolean type"

    # transfer of BnB bounds
    self.bnb_affordable_lcb_transfer = options.get("bnb_affordable_lcb_transfer", False)
    if not isinstance(self.bnb_affordable_lcb_transfer, bool):
      raise TypeError("bnb_affordable_lcb_transfer must be bool")
    # Transfer prepared at the end of the preceding BO iteration.
    self._bnb_affordable_transfer = None

    ##################################################################################################
    # Options consistency checks
    ##################################################################################################
    if self.bnb_affordable_lcb_transfer:
      if (self.opt_solver != "BnB" or self.acquisition_type != "LCB"):
        raise ValueError("Affordable transfer requires BnB and LCB with BO solver")
      #fixme
      if self.batch_size != 1:
        raise ValueError("Affordable tranfer only supports batch_size=1")
      if not self.bnb_warm_start:
        raise ValueError("Affordable transfer requires bnb_warmstart=True")
      if self.bnb_lower_bound_transfer is not None:
        raise ValueError("Do not combine automatic and user-supplied transfers for affordable bound tranfer")

      
    self.logger.info(f"Problem name: {prob.name}")
    self.logger.info(f"Max BO iter: {self.bo_maxiter}")
    self.logger.info(f"Optimizing acquisition ({self.acquisition_type}) "
                     f"with {self.n_start} random initial points")
    self.logger.info(f"Batch type: {self.batch_type}")
    self.logger.info(f"Batch size: {batch_size}")
    self.logger.info(f"Internal optimization solver: {opt_solver}")
    self.logger.info(f"Internal optimization solver options")
    for key, value in self.solver_options.items():
      self.logger.info(f"  {key} : {value}")
    self.logger.info(f"Initial training set: {xtrain.shape[0]} samples, {xtrain.shape[1]} dimensions")
    self.logger.debug(f"Bounds on optimization variable: {self.bounds}")
    self.logger.info(f"Logger level: {logger_level}")

  # Method to train the GP model
  def _train_surrogate(self, x_train, y_train, *, full_retrain, preserve_prior=False):
    self.logger.debug(f"Training surrogate model with {x_train.shape[0]} samples...")
    theta_bounds = None

    if full_retrain:
      sm = self.gpsurrogate.surrogatesmt
      corr = sm.options["corr"]
      power = float(sm.options["pow_exp_power"])

      theta_bounds = smt_theta_bounds(S=x_train.shape[0], N=x_train.shape[1], corr=corr, pow_exp_power=power)
      self.logger.info(f"Full GP retrain: S={x_train.shape[0]}, theta_bounds={theta_bounds}")
    else:
      msg = "Fixed-prior GP refit" if preserve_prior else "Fixed-theta GP refit"
      self.logger.debug(msg)      


    self.gpsurrogate.train(x_train, y_train, optimize_theta=full_retrain,
                           theta_bounds=theta_bounds, preserve_prior=preserve_prior)
    self.logger.debug("Surrogate training complete.")

    self.logger.info(self.gpsurrogate.cov_cond_info())

  # Method to find the best next sampling point via optimizing the acquisition function
  def _find_best_point(self, x_train, y_train, x0 = None, BOit=0):
    self.logger.info(f"Start finding the best sampling point:")

    if self.acquisition_type == "LCB":
      acqf = LCBacquisition(self.gpsurrogate, beta=self.LCB_beta)
    elif self.acquisition_type == "EI":
      acqf = EIacquisition(self.gpsurrogate)
    else:
      raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)

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
    #else:
    #  # Instantiate BnB with GP surrogate and BO callback
    #  bnb = BnBAlgorithm(acqf, options=self.solver_options, BOit=BOit)
    # 
    #  # Initialize BnB (perhaps use old set of boxes if self.bnb_queue is not None)
    #  bnb.initialize(queue=self.bnb_queue)
    #  
    #  # Run BnB optimization
    #  best_xopt = bnb.optimize()
    #  if self.bnb_warm_start:
    #    # Update queue in order to warm-start BnB at next BO step
    #    self.bnb_queue = bnb.queue
    #  self.bnb_num_branch_hist.append(bnb.num_branches)
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

  # Set the options for the internal optimization solver
  def set_options(self, solver_options):
    self.solver_options = solver_options

  # Method to perform Bayesian optimization
  def optimize(self):
    # x_train/y_train below denote only the active GP training set.
    x_train = np.array(self.xtrain, copy=True)
    y_train = np.array(self.ytrain, copy=True)

    # Complete true-evaluation archive.
    
    x_evaluated = np.array(x_train, copy=True)
    y_evaluated = np.array(y_train, copy=True)
    self.logger.iterations(f"Best objective from {np.size(x_train, 0)} initial samples: {np.min(y_train):.4e} ")

    self._train_surrogate(x_train, y_train, full_retrain=True)

    #
    # filter feasible points
    #
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

      self.logger.info(f"Best objective: {best_constrained_train_y:.4e} from {y_train_fea.size} feasible initial training points")

    else:
      best_constrained_train_y = np.inf
      self.logger.info("No feasible initial training points.")

    self.x_hist = []
    self.y_hist = []
    prev_best_y = best_constrained_train_y
    self.bo_iteration_hist = []    
    for i in range(self.bo_maxiter):
      self.logger.critical(f"*****************************")
      self.logger.critical(f"Iteration {i+1}/{self.bo_maxiter}")

      #
      # Diagnostics code
      #
      bo_iteration_number = i + 1

      sample_metrics = _sample_set_clustering_metrics(self.gpsurrogate, x_train)

      self.logger.scalars(f"Sample-set clustering (Euclidean distance) at start of BO iteration {bo_iteration_number}: ")
      self.logger.scalars(f"  domain_nn_min={sample_metrics['domain_nn_min']:.6e}, "
                          f"domain_nn_p01="
                          f"{sample_metrics['domain_nn_p01']:.6e}, "
                          f"domain_nn_p05="
                          f"{sample_metrics['domain_nn_p05']:.6e}, "
                          f"domain_nn_p50="
                          f"{sample_metrics['domain_nn_p50']:.6e}")

      self.logger.scalars(f"Sample-set kernel (theta-weighted) distance and GP correlation at start of BO iteration {bo_iteration_number}: ")
      self.logger.scalars(f"  kernel_nn_min="
                          f"{sample_metrics['kernel_nn_min']:.6e}, "
                          f"corr_max_offdiag="
                          f"{sample_metrics['corr_max_offdiag']:.6e}, "
                          f"pairs_corr_ge_0p95="
                          f"{sample_metrics['pairs_corr_ge_0p95']}, "
                          f"pairs_corr_ge_0p99="
                          f"{sample_metrics['pairs_corr_ge_0p99']}")

      selected_point_metrics = []
      q_batch = self.batch_size      
      y_train_virtual = y_train.copy() # old training + batch_size num of virtual points
      if self.opt_solver != "BnB":
        for j in range(self.batch_size):
          # Get a new sample point
          self.logger.scalars(f"In batch {j+1}/{self.batch_size}")
          x_new = self._find_best_point(x_train, y_train_virtual, BOit=i)

          selected_point_metrics.append(_new_point_clustering_metrics(self.gpsurrogate, x_train, x_new))
          
          # Update training sample points
          x_train = np.vstack([x_train, x_new])

          # if this is not the last point in the current batch
          # then obtain a virtual point
          if j < max(range(self.batch_size)):
            # Get a virtual point
            y_virtual = self._get_virtual_point(np.atleast_2d(x_new))

            # Update training set with the virtual point
            y_train_virtual = np.vstack([y_train_virtual, y_virtual])
            self._train_surrogate(x_train, y_train_virtual, full_retrain=False)

          mean_val = self.gpsurrogate.mean(np.array([x_new])).item()
          sd_val = np.sqrt(self.gpsurrogate.variance(np.array([x_new])).item())
          self.logger.scalars(f"  (mu, sigma) at new sample x: {mean_val}, {sd_val} ")
      else:
        # BNB execution path here
        if self.acquisition_type == "LCB":
          acqf = LCBacquisition(self.gpsurrogate, beta=self.LCB_beta)
        elif self.acquisition_type == "EI":
          acqf = EIacquisition(self.gpsurrogate)
        else:
          raise NotImplementedError("No implemented acquisition_type associated to"+self.acquisition_type)
        # Instantiate BnB with GP surrogate and BO callback
        bnb_options = dict(self.solver_options)
        bnb_options["collect_batch_candidates"] = (self.bnb_batch_method == "conditional_variance")
        bnb = BnBAlgorithm(acqf, options=bnb_options, BOit=i)        
     
        # Initialize BnB (perhaps use old set of boxes if self.bnb_queue is not None)
        #bnb.initialize(partition=self.bnb_partition, transfer_lower_bound=self.bnb_lower_bound_transfer)
        restart_transfer = self.bnb_lower_bound_transfer

        if self.bnb_affordable_lcb_transfer:
          restart_transfer = self._bnb_affordable_transfer

          # AffordableTransfer from the paper: callback describes exactly one GP update.
          self._bnb_affordable_transfer = None

        bnb.initialize(partition=self.bnb_partition, transfer_lower_bound=restart_transfer)
        # Run BnB optimization
        best_xopt = bnb.optimize()
        self.logger.info(f"BnB nodes explored: {bnb.num_branches}")
        self.logger.info(f"size of BnB queue = {len(bnb.queue)}")
        self.logger.info(f"optimal point = {best_xopt}")

        self.bo_stop_tol = 0.01
        # BO stopping criterion based on remaining LCB improvement potential and small exploration term

        if self.acquisition_type == "LCB" and self.bo_stop_tol > 0.0:

          # Best feasible true incumbent over all evaluated points.
          archive_feas = (self.prob.if_feasible(x_evaluated) & np.isfinite(y_evaluated).ravel())

          if np.any(archive_feas):
            feasible_idx = np.flatnonzero(archive_feas)
            k = int(np.argmin(y_evaluated[archive_feas].reshape(-1)))
            best_idx = int(feasible_idx[k])

            x_incumbent = np.asarray(x_evaluated[best_idx], dtype=float).reshape(-1)
            f_incumbent = float(np.asarray(y_evaluated[best_idx]).reshape(-1)[0])

            # Evaluate the CURRENT LCB at the best true incumbent.
            lcb_at_incumbent = float(acqf.scalar_evaluate(x_incumbent))

            # Globally minimized LCB returned by BnB.
            lcb_min = float(acqf.scalar_evaluate(np.asarray(best_xopt, dtype=float)))
            
            scale = max(1.0, abs(f_incumbent))
            incumbent_gap = abs(lcb_at_incumbent - f_incumbent) / scale

            global_lcb_gap = abs(lcb_min - f_incumbent) / scale

            self.logger.info(f"BO stopping diagnostics: incumbent f_best={f_incumbent:.6e}  "
                             f"LCB_at_best={lcb_at_incumbent:.6e} gap={incumbent_gap:.6e} | "
                             f"LCB best LCB_at_LCB_best={lcb_min:.6e} gap={global_lcb_gap:.6e}")



            if (incumbent_gap <= self.bo_stop_tol and global_lcb_gap <= self.bo_stop_tol):
              self.logger.critical(f"BO stopping: LCB converged: incumbent LCB gap={incumbent_gap:.3e} | "
                  f"gap at LCB best {global_lcb_gap:.3e}. tol={self.bo_stop_tol:.3e}")
              break            

        
        if self.bnb_batch_method == "conditional_variance":
          selection_options = dict(self.bnb_batch_options)
          selection_options.setdefault("is_feasible", self.prob.if_feasible)

          # Avoid reevaluating points that were previously evaluated but not assimilated in the active GP
          selection_options.setdefault("exclude_points", x_evaluated)
          
          x_new, batch_info = bnb.select_batch_condvar(self.batch_size, **selection_options)

          self.logger.info(f"BnB-CV batch: candidates={batch_info['candidate_count']} "
                           f"pool={batch_info['pool_size']} selected={len(x_new)}/{self.batch_size} "
                           f"delta={batch_info['delta']:.6e}")
          self.logger.info(f"  LCB values: {batch_info['lcb']}")
          self.logger.info(f"  Conditional variances at selection: {batch_info['conditional_variance']}")
          self.logger.scalars(f"Acquisition-suboptimality bounds: {batch_info['lcb_suboptimality_bound']}")
          if len(x_new) < self.batch_size:
            self.logger.info("Small BnB-CV batch: insufficient distinct candidates or remaining conditional variance.")

        elif self.batch_size == 1:
          x_new = np.atleast_2d(best_xopt)

        else:
          # K-means batching
          bnb_nodes = bnb.get_candidate_nodes()
          node_pts = np.asarray([node.aq_U_x for node in bnb_nodes])
          if len(bnb_nodes) < self.batch_size:
            raise RuntimeError("Not enough BnB nodes for K-means batching")

          labels = KMeans(n_clusters=self.batch_size, init="k-means++", n_init="auto",
                          random_state=self.solver_options.get("random_seed", 42)).fit_predict(node_pts)
          chosen = []
          for label in range(self.batch_size):
            members = np.flatnonzero(labels == label)
            if members.size == 0:
              raise RuntimeError("Empty K-means cluster in BnB batching")
            values = [float(bnb_nodes[k].aq_U) for k in members]
            chosen.append(members[int(np.argmin(values))])
          x_new = node_pts[chosen]

        q_batch = len(x_new)
        x_eval = np.asarray(x_new, dtype=float)
        
        diagnostic_old_x = np.array(x_train, copy=True)
        for point in x_new:
          selected_point_metrics.append(_new_point_clustering_metrics(self.gpsurrogate, diagnostic_old_x, point))

          # For batch_size > 1, later points are also compared
          # with points selected earlier in this batch.
          diagnostic_old_x = np.vstack([diagnostic_old_x, point])
        
        if self.bnb_warm_start:
          # Update queue in order to warm-start BnB at next BO step
          self.bnb_partition = bnb.export_partition()
          self.bnb_queue = bnb.queue  # compatibility/diagnostics only
        self.bnb_num_branch_hist.append(bnb.num_branches)

      ####################################################################
      # True / black-box evaluations: evaluate the complete BO batch.
      ###################################################################
      if self.opt_solver == "BnB":
        x_eval = np.asarray(x_new, dtype=float)
      else:
        # Existing non-BnB code has already appended the selected points
        # to x_train while constructing the virtual batch.
        x_eval = np.asarray(x_train[-q_batch:], dtype=float)

      # evaluate true function for all batch candidates
      y_new = np.asarray(self.obj_evaluator.run(self.prob.evaluate, x_eval))

      # Every expensive evaluation is retained
      x_evaluated = np.vstack([x_evaluated, x_eval])
      y_evaluated = np.vstack([y_evaluated, y_new])


      ###############################################################################
      # Select the subset assimilated into the active GP.
      #
      # For BnB-CV, x_eval is already ordered as
      #   0: LCB global minimizer point
      #   1: max conditional variance given point 0
      #   2: max conditional variance given points 0,1
      #   ...
      #
      # Hence the first 'bo_bnb_batch_max_add' points give a default exploitation
      # and the heuristical pure-exploration subset.
      #
      # If a batch point improves the best known true function, it will be added to
      # the GP (even when bnb_batch_max_add == 1)
      ###############################################################################
      if (self.opt_solver == "BnB" and self.bnb_batch_method == "conditional_variance"):
        n_gp_add = min(self.bnb_batch_max_add, q_batch)

        # assimilate first n_gp_add, and maybe one batch point that improves true function 
        gp_idx = np.arange(n_gp_add)

        # see if true func is improved
        feas_new = self.prob.if_feasible(x_eval) & np.isfinite(y_new).ravel()
        j_inc = -1
        if np.any(feas_new):
          feas_idx = np.flatnonzero(feas_new)
          j_inc = int(feas_idx[np.argmin(y_new[feas_new].reshape(-1))])
          f_inc = float(np.asarray(y_new[j_inc]).reshape(-1)[0])
          
          if f_inc < prev_best_y:
            if j_inc >= n_gp_add:
              gp_idx = np.append(gp_idx, j_inc)
              self.logger.info(f"BnB-CV: new incumbent found in batch at idx {j_inc}: adding it to GP")
          else:
            j_inc = -1 #invalidate idx since no global improvement
        # some output info
        if 'batch_info' in locals():
          cv = np.asarray(batch_info["conditional_variance"], dtype=float)
          lcb_vals = np.asarray(batch_info["lcb"], dtype=float,)
          self.logger.info("BnB-CV conditional variance:")
          for j in range(q_batch):
            suffix = "GP" if j < n_gp_add else "eval-only"
            if j==j_inc: suffix = "GP (new incumb)"
            f_true = float(np.asarray(y_new[j]).reshape(-1)[0])
            self.logger.info(f"  batch[{j}] LCB={lcb_vals[j]:.6e} var_cond={cv[j]:.6e} f_true={f_true:.12e} [{suffix}]")
        
        x_gp_add = x_eval[gp_idx]
        y_gp_add = y_new[gp_idx]

        x_train = np.vstack([x_train, x_gp_add,])
        y_train = np.vstack([y_train, y_gp_add,])

        self.logger.info(f"BnB-CV GP assimilation: {gp_idx.size}/{q_batch} evaluated points")

        #if 'batch_info' in locals():
        #  self.logger.info(f"  assimilated conditional variances: {batch_info['conditional_variance'][:n_gp_add]}")
        #  self.logger.info(f"  assimilated LCB values: {batch_info['lcb'][:n_gp_add]}")

      else:
        # Preserve existing behavior for the other, non BnB-CV batch methods.

        # for non-BnB batching, x_new / x_eval were already added to x_train
        if self.opt_solver == "BnB":
          x_train = np.vstack([x_train, x_eval,])
        y_train = np.vstack([y_train, y_new,])

      
      # Full theta optimization after each nretrainGP completed iterations.
      full_retrain = (i + 1) % self.nretraingp == 0
      next_transfer = None
      #self._train_surrogate(x_train, y_train, full_retrain=full_retrain)

      if(self.bnb_affordable_lcb_transfer and self.bnb_warm_start and not full_retrain):
        try:
          # Snapshot the old posterior before smtKRG is refitted in place.
          next_transfer = AffordableLCBTransfer(bnb, x_plus=x_new[0], y_plus=np.asarray(y_new).reshape(-1)[0])
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
          self.logger.info(f"Affordable LCB transfer unavailable; bounds will be recomputed: {error}")

      self._train_surrogate(x_train, y_train, full_retrain=full_retrain,
                            preserve_prior=(next_transfer is not None))

      if next_transfer is not None:
        try:
          next_transfer.validate_new_model(self.gpsurrogate, self.LCB_beta)
        except RuntimeError as error:
          # Safe fallback: the retained partition is still useful,
          # but restart_callback will recompute every lower bound.
          self.logger.info(f"Affordable LCB transfer rejected: bounds will be recomputed: {error}")
          next_transfer = None

      self._bnb_affordable_transfer = next_transfer
      
      feas_new = self.prob.if_feasible(x_eval)
      self.logger.debug(f"Feasible samples: {np.sum(feas_new)}/{q_batch}")

      min_y_new = np.min(y_new)
      curr_best_y = np.min([prev_best_y, min_y_new])

      self.logger.iterations(f"Best objective found in this iteration: {min_y_new:.4e} ")
      self.logger.scalars(f"Training set size is now {x_train.shape[0]}")
      self.logger.iterations(f"Current best objective: {curr_best_y:.4e} "
                             f"(previous best: {prev_best_y:.4e})")
      self.logger.scalars(f"Objective function improvement: {prev_best_y - curr_best_y:.4e}")

      # Save the new sample points and objective evaluations
      for j in range(q_batch):
        self.x_hist.append(x_eval[j].flatten())
        self.y_hist.append(y_new[j].flatten())
        self.bo_iteration_hist.append(i + 1)
        
      self.logger.debug(f"Sample point(s) X:")

      for j in range(q_batch):
        self.logger.debug(f"  {x_eval[j]}")

      self.logger.debug(f"Observation(s) Y:")
      for j in range(q_batch):
        self.logger.debug(f"  {y_new[j]}")

      #for j, point_metrics in enumerate(selected_point_metrics):
      #  self.logger.scalars(f"Selected-point clustering at end of BO iteration {bo_iteration_number}, batch point {j+1}: ")
      #  self.logger.scalars(f"  domain_nn(Euclidean dist)="
      #                      f"{point_metrics['domain_nn']:.6e}, "
      #                      f"smt_nn(distance in SMT coordinates)="
      #                      f"{point_metrics['smt_nn']:.6e}, "
      #                      f"kernel_nn(theta-weighted distances)="
      #                      f"{point_metrics['kernel_nn']:.6e}, "
      #                      f"kernel_corr_to_nearest="
      #                      f"{point_metrics['kernel_corr_to_nearest']:.6e}, "
      #                      f"nearest_old_index="
      #                      f"{point_metrics['nearest_old_index']}")

      prev_best_y = curr_best_y

    # active GP points
    self.setTrainingData(x_train, y_train)
    # All true evaluations, including points omitted from GP.
    self.x_evaluated = np.array(x_evaluated, copy=True)
    self.y_evaluated = np.array(y_evaluated, copy=True)

    # Save the BO optimal (excluding initial training pts) results
    # filter non-finite BB objective function values --> inf
    y_hist_filt = np.where(np.isfinite(self.y_hist), self.y_hist, np.inf)
    # if there is at least one finite value then argmin is well-defined
    if not np.isinf(y_hist_filt).all():
      idx_BO_opt = np.argmin(y_hist_filt)
    else:
      idx_BO_opt = 0 # choose an index from set of non-finite BB objective function values
    self.x_BO_opt = self.x_hist[idx_BO_opt]
    self.y_BO_opt = y_hist_filt[idx_BO_opt][0]
    
    self.logger.critical("===================================")
    self.logger.critical("Bayesian Optimization completed")
    self.logger.critical(f"Total objective evaluations for initial samples: {self.init_ntrain}")
    self.logger.critical(f"Total objective evaluations for BO iterations: {len(self.y_hist)}")
    
    if np.isfinite(self.y_BO_opt) or y_train_fea.size > 0:
      if self.y_BO_opt < best_constrained_train_y:
        self.logger.critical(f"Optimal at BO iteration: {self.bo_iteration_hist[idx_BO_opt]} ")
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
      self.logger.critical(f"BB objective non-finite at all BO sample points")
      self.logger.critical(f"Each initial GP training point either not feasible or BB objective is non-finite at it")
      self.x_opt = self.x_BO_opt
      self.y_opt = self.y_BO_opt
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
