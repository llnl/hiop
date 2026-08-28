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
from .acquisition import LCBacquisition, EIacquisition
from ..problems.problem import Problem
from ..utils.util import Evaluator, Logger
from .bnbalgorithm import BnBAlgorithm
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
  
  x_train = np.atleast_2d(
      np.asarray(x_train, dtype=float)
  )
  n_train = x_train.shape[0]

  if n_train < 2:
    return {
        "domain_nn_min": np.nan,
        "domain_nn_p01": np.nan,
        "domain_nn_p05": np.nan,
        "domain_nn_p50": np.nan,
        "kernel_nn_min": np.nan,
        "corr_max_offdiag": np.nan,
        "pairs_corr_ge_0p99": 0,
        "pairs_corr_ge_0p999": 0,
    }

  theta, x_offset, x_scale = _smt_se_geometry(
      gpsurrogate
  )

  # Domain-normalized Euclidean distances.
  x_domain = _domain_normalize(
      gpsurrogate,
      x_train,
  )
  domain_dist = _pairwise_euclidean(x_domain)

  # Exclude self-distance when finding the nearest neighbor.
  np.fill_diagonal(domain_dist, np.inf)
  domain_nn = np.min(domain_dist, axis=1)

  # SMT-standardized and theta-weighted distances.
  x_smt = (x_train - x_offset) / x_scale
  kernel_dist, correlation = (
      _se_kernel_distance_and_correlation(
          x_smt,
          x_smt,
          theta,
      )
  )

  np.fill_diagonal(kernel_dist, np.inf)
  kernel_nn = np.min(kernel_dist, axis=1)

  # Extract each unordered pair exactly once.
  ii, jj = np.triu_indices(n_train, k=1)
  corr_offdiag = correlation[ii, jj]

  return {
      "domain_nn_min": float(np.min(domain_nn)),
      "domain_nn_p01": float(
          np.percentile(domain_nn, 1.0)
      ),
      "domain_nn_p05": float(
          np.percentile(domain_nn, 5.0)
      ),
      "domain_nn_p50": float(
          np.percentile(domain_nn, 50.0)
      ),
      "kernel_nn_min": float(np.min(kernel_nn)),
      "corr_max_offdiag": float(
          np.max(corr_offdiag)
      ),
      "pairs_corr_ge_0p99": int(
          np.count_nonzero(corr_offdiag >= 0.99)
      ),
      "pairs_corr_ge_0p999": int(
          np.count_nonzero(corr_offdiag >= 0.999)
      ),
  }


def _new_point_clustering_metrics(
    gpsurrogate,
    old_x,
    x_new,
):
  """Distances from x_new to the samples present when it was selected."""
  old_x = np.atleast_2d(
      np.asarray(old_x, dtype=float)
  )
  x_new = np.asarray(
      x_new,
      dtype=float,
  ).reshape(1, -1)

  theta, x_offset, x_scale = _smt_se_geometry(
      gpsurrogate
  )

  # Distance in the original domain-normalized coordinates.
  old_domain = _domain_normalize(
      gpsurrogate,
      old_x,
  )
  new_domain = _domain_normalize(
      gpsurrogate,
      x_new,
  )
  domain_dist = np.linalg.norm(
      old_domain - new_domain,
      axis=1,
  )

  # Distance in SMT-standardized coordinates.
  old_smt = (old_x - x_offset) / x_scale
  new_smt = (x_new - x_offset) / x_scale
  smt_dist = np.linalg.norm(
      old_smt - new_smt,
      axis=1,
  )

  # Theta-weighted kernel distance and exact SE correlation.
  kernel_dist, correlation = (
      _se_kernel_distance_and_correlation(
          new_smt,
          old_smt,
          theta,
      )
  )

  kernel_dist = kernel_dist.reshape(-1)
  correlation = correlation.reshape(-1)

  # Define nearest_old_index using the GP/kernel metric.
  nearest_old_index = int(
      np.argmin(kernel_dist)
  )

  return {
      "domain_nn": float(np.min(domain_dist)),
      "smt_nn": float(np.min(smt_dist)),
      "kernel_nn": float(
          kernel_dist[nearest_old_index]
      ),
      "kernel_corr_to_nearest": float(
          correlation[nearest_old_index]
      ),
      "nearest_old_index": nearest_old_index,
  }

# A base class defining a general framework for Bayesian Optimization
class BOAlgorithmBase:
  def __init__(self):
    self.acquisition_type = "LCB" # Type of acquisition function (default = "LCB")
    self.LCB_beta = 3.0
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
    self.bnb_num_branch_hist = [] # number of BnB branches visited per BO iter

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
    return y_opt[0]

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

    logger_level = options.get('log_level', "INFO")
    self.logger.setlevel(logger_level)

    self.bo_maxiter = options.get('bo_maxiter', self.bo_maxiter)
    assert self.bo_maxiter > 0, f"Invalid bo_maxiter: {self.bo_maxiter}"

    self.n_start = options.get('n_start', self.n_start)
    assert self.n_start > 0, f"Invalid n_start: {self.n_start}"

    acquisition_type = options.get('acquisition_type', "LCB")
    assert acquisition_type in ["LCB", "EI"], f"Invalid acquisition_type: {acquisition_type}"

    self.LCB_beta = options.get('LCB_beta', self.LCB_beta)
    assert self.LCB_beta > 0., f"Invalid LCB beta (variance penalty): {self.LCB_beta}"

    batch_size = options.get('batch_size', 1)
    assert isinstance(batch_size, int), f"batch_size {batch_size} not an integer"
    assert batch_size > 0, f"batch_size {batch_size} is not strictly positive"

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
    self.bnb_queue = None  # legacy; not a complete spatial partition
    self.bnb_partition = None
    self.bnb_lower_bound_transfer = options.get('BnBLowerBoundTransfer', None)
    if user_grad:
      self.fun_grad = user_grad

    self.bnb_warm_start = True
    self.bnb_warm_start = options.get('BnBWarmStart', self.bnb_warm_start)
    assert isinstance(self.bnb_warm_start, bool), "provided BnBWarmStart is not a boolean type"
    
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
  def _train_surrogate(self, x_train, y_train):
    self.logger.debug("Training surrogate model with "
                      f"{x_train.shape[0]} samples...")
    self.gpsurrogate.train(x_train, y_train)
    self.logger.debug("Surrogate training complete.")

  # Method to find the best next sampling point via optimizing the acquisition function
  def _find_best_point(self, x_train, y_train, x0 = None, BOit=0):
    self.logger.info(f"Start finding the best sampling point:")
    self._train_surrogate(x_train, y_train)
    if self.acquisition_type == "LCB":
      acqf = LCBacquisition(self.gpsurrogate, beta=self.LCB_beta)
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
    x_train = self.xtrain
    y_train = self.ytrain
    self.logger.iterations(f"Best UNCONSTRAINED objective from {np.size(x_train, 0)} initial samples: {np.min(y_train):.4e} ")

    # filter feasible points
    fea_idx = self.prob.if_feasible(x_train, y_train)
    y_fea = y_train[fea_idx]
    if y_fea.size > 0:
      best_constrained = np.min(y_fea)
      self.logger.info(
            f"Best CONSTRAINED objective from {y_fea.size} feasible initial samples: {np.min(y_fea):.4e}"
        )
    else:
      self.logger.info("No feasible samples found.")

    self.x_hist = []
    self.y_hist = []
    
    prev_best_y = np.inf
    for i in range(self.bo_maxiter):
      self.logger.critical(f"*****************************")
      self.logger.critical(f"Iteration {i+1}/{self.bo_maxiter}")

      #
      # Diagnostics code
      #
      bo_iteration_number = i + 1

      sample_metrics = _sample_set_clustering_metrics(
          self.gpsurrogate,
          x_train,
      )

      self.logger.scalars(
          f"Sample-set clustering at start of BO iteration "
          f"{bo_iteration_number}: "
          f"domain_nn_min="
          f"{sample_metrics['domain_nn_min']:.6e}, "
          f"domain_nn_p01="
          f"{sample_metrics['domain_nn_p01']:.6e}, "
          f"domain_nn_p05="
          f"{sample_metrics['domain_nn_p05']:.6e}, "
          f"domain_nn_p50="
          f"{sample_metrics['domain_nn_p50']:.6e}"
      )

      self.logger.scalars(
          f"Sample-set kernel correlation at start of BO iteration "
          f"{bo_iteration_number}: "
          f"kernel_nn_min="
          f"{sample_metrics['kernel_nn_min']:.6e}, "
          f"corr_max_offdiag="
          f"{sample_metrics['corr_max_offdiag']:.6e}, "
          f"pairs_corr_ge_0p99="
          f"{sample_metrics['pairs_corr_ge_0p99']}, "
          f"pairs_corr_ge_0p999="
          f"{sample_metrics['pairs_corr_ge_0p999']}"
      )

      selected_point_metrics = []
      
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
            self.gpsurrogate.train(x_train, y_train_virtual)

          mean_val = self.gpsurrogate.mean(np.array([x_new])).item()
          sd_val = np.sqrt(self.gpsurrogate.variance(np.array([x_new])).item())
          self.logger.scalars(f"  (mu, sigma) at new sample x: {mean_val}, {sd_val} ")
      else:
        if self.acquisition_type == "LCB":
          acqf = LCBacquisition(self.gpsurrogate, beta=self.LCB_beta)
        elif self.acquisition_type == "EI":
          acqf = EIacquisition(self.gpsurrogate)
        else:
          raise NotImplementedError("No implemented acquisition_type associated to"+self.acquisition_type)
        # Instantiate BnB with GP surrogate and BO callback
        bnb = BnBAlgorithm(acqf, options=self.solver_options, BOit=i)
     
        # Initialize BnB (perhaps use old set of boxes if self.bnb_queue is not None)
        bnb.initialize(partition=self.bnb_partition, transfer_lower_bound=self.bnb_lower_bound_transfer)
        
        # Run BnB optimization
        best_xopt = bnb.optimize()
        self.logger.info(f"BnB nodes explored: {bnb.num_branches}")
        print("size of BnB queue = ", len(bnb.queue))
        print("optimal point = ", best_xopt)
        # experimental, testing clustering of BnB queue----
        bnb_nodes = bnb.get_candidate_nodes()
        node_pts = np.array([node.aq_U_x for node in bnb_nodes])
        """
          approach -- 1) split queue into batch_size 
                         number of clusters
                      2) from each cluster grab point
                         with best upper-bound
        """
        n_clusters = int(self.batch_size)
        x_new = []
        if n_clusters == 1:
          x_new.append(best_xopt)
        elif n_clusters > 1:
          assert len(bnb_nodes) >= n_clusters, "not enough BnB nodes to acquire requested number of batch points"
          kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=self.solver_options.get("random_seed", 42))
          cluster_labels = kmeans.fit_predict(node_pts)
          clusters = [[node_pts[i] for i, val in enumerate(cluster_labels) if val == lbl] for lbl in range(n_clusters)]
          UBs_by_cluster = [[bnb_nodes[i].aq_U for i, val in enumerate(cluster_labels) if val == lbl] for lbl in range(n_clusters)]
          s_score = silhouette_score(node_pts, cluster_labels)
          print("Silhouette score = ", s_score)
          for i in range(n_clusters):
            print("cluster # ", i, " contains ", len(clusters[i]), " pts")
            arg = np.argmin(UBs_by_cluster[i])
            print("smallest acqf UB in cluster = ", UBs_by_cluster[i][arg])
            print(" at x = ", clusters[i][arg])
            print("-"*40)
            x_new.append(clusters[i][arg])
            distances = np.zeros(int((len(clusters[i]) * len(clusters[i]) -1 ) / 2))
        x_new = np.atleast_2d(x_new)
        
        diagnostic_old_x = np.array(x_train, copy=True)
        for point in x_new:
          selected_point_metrics.append(_new_point_clustering_metrics(self.gpsurrogate, diagnostic_old_x, point))

          # For batch_size > 1, later points are also compared
          # with points selected earlier in this batch.
          diagnostic_old_x = np.vstack([diagnostic_old_x, point])
        
        x_train = np.vstack([x_train, x_new])
        if self.bnb_warm_start:
          # Update queue in order to warm-start BnB at next BO step
          self.bnb_partition = bnb.export_partition()
          self.bnb_queue = bnb.queue  # compatibility/diagnostics only
        self.bnb_num_branch_hist.append(bnb.num_branches)

      y_new = self.obj_evaluator.run(self.prob.evaluate, x_train[-self.batch_size:])
      y_new = np.array(y_new)
      y_train = np.vstack([y_train, y_new])
      self.gpsurrogate.train(x_train, y_train)

      feas_new = self.prob.if_feasible(x_train[-self.batch_size:])
      self.logger.debug(f"Feasible samples: {np.sum(feas_new)}/{self.batch_size}")

      min_y_new = np.min(y_new)
      curr_best_y = np.minimum(prev_best_y, min_y_new)

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

      for j, point_metrics in enumerate(selected_point_metrics):
        self.logger.scalars(
            f"Selected-point clustering at end of BO iteration "
            f"{bo_iteration_number}, batch point {j+1}: "
            f"domain_nn="
            f"{point_metrics['domain_nn']:.6e}, "
            f"smt_nn="
            f"{point_metrics['smt_nn']:.6e}, "
            f"kernel_nn="
            f"{point_metrics['kernel_nn']:.6e}, "
            f"kernel_corr_to_nearest="
            f"{point_metrics['kernel_corr_to_nearest']:.6e}, "
            f"nearest_old_index="
            f"{point_metrics['nearest_old_index']}"
        )

      prev_best_y = curr_best_y

    # Save the optimal results and all the training data
    self.idx_opt = np.argmin(self.y_hist)
    self.x_opt = self.x_hist[self.idx_opt]
    self.y_opt = self.y_hist[self.idx_opt]
    self.setTrainingData(x_train, y_train)

    self.logger.critical("===================================")
    self.logger.critical("Bayesian Optimization completed")
    self.logger.critical(f"Total evaluations for initial samples: {len(self.ytrain)-len(self.y_hist)}")
    self.logger.critical(f"Total evaluations for BO iterations: {len(self.y_hist)}")
    self.logger.critical(f"Optimal at BO iteration: {self.idx_opt//self.batch_size+1} ")
    self.logger.debug(f"Best point: {self.x_opt.flatten()}")
    self.logger.critical(f"Best value: {self.y_opt[0]}")
    self.logger.critical("===================================")


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
