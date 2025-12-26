import numpy as np
from numpy.random import uniform
import cvxpy as cp
import heapq
from scipy import linalg
from .acquisition import EIacquisition, LCBacquisition
from ..utils.util import Evaluator, MPIEvaluator
from ..utils.new_eval_manager import is_running_with_mpi
from .opt_utils import minimizer_wrapper
from itertools import count
try:
  from mpi4py import MPI
except ImportError:
  print("unable to import mpi4py")

import time


# BnBNode
# corners of interval [l, u]
# upper and lower bounds of acquisition function
class BnBNode:
  def __init__(self, l, u, aq_L, aq_U):
    self.l = l
    self.u = u
    self.aq_L = aq_L
    self.aq_U = aq_U
    self.diam = np.max(u - l)
    self.midpoint = 0.5 * (l + u)
  def __lt__(self, other):
    return self.aq_U > other.aq_U

def branch(l, u):
  # Force to float to avoid truncation issues
  l = l.astype(float)
  u = u.astype(float)

  # Pick the dimension with largest length
  d = np.argmax(u - l)
  mid = 0.5 * (l[d] + u[d])

  # If the midpoint is the same as one bound (degenerate split), return nothing
  # shouldn't this issue have been caught?
  if np.isclose(mid, l[d]) or np.isclose(mid, u[d]):
    return []

  # Generate child boxes
  l1, u1 = l.copy(), u.copy()
  l2, u2 = l.copy(), u.copy()
  
  # Split the largest axis
  # along along midpoint of said axis
  u1[d] = mid
  l2[d] = mid
  return [(l1, u1), (l2, u2)]


class BnBAlgorithmBase:
  def __init__(self, x = None, y = None):
    # Node class for priority queue
    # Kernel info for bounds
    self.kernel_spec = None
    self.kernel_func = None
    self.y_min = None

    # Evaluation parameters
    self.theta = None

    # Test
    self.enable_debug_checks = False  

    # BnB search state
    self.best_l = None
    self.best_u = None
    self.upper_bound = np.inf
    self.final_gap = None
    self.final_diameter = None
    self.verbose = False

    # Training data
    self.x = x
    self.y = y

  def sync_from_smt(self):
    sm = self.gpsurrogate.surrogatesmt
    par = sm.optimal_par

    # --- kernel / corr selection ---
    corr = sm.options["corr"]  # e.g., 'squar_exp', 'pow_exp', 'abs_exp', 'matern32', 'matern52'
    if corr == "pow_exp":
      # OptionsDictionary -> use membership + indexing (no .get)
      p = float(sm.options["pow_exp_power"]) if "pow_exp_power" in sm.options else 2.0
      if p not in (1.0, 2.0):
        # tighten if your 1D bound code only supports p in {1,2}
        raise ValueError("Single-d bounds support pow_exp only for p=1 or p=2")
      self.kernel_spec = "pow_exp"
      self.p = p
    elif corr == "squar_exp":
      # Gaussian is pow_exp with p=2
      self.kernel_spec = "pow_exp"
      self.p = 2.0
    elif corr == "abs_exp":
      # Exponential is pow_exp with p=1
      self.kernel_spec = "pow_exp"
      self.p = 1.0
    elif corr == "matern32":
      self.kernel_spec = "matern32"
    elif corr == "matern52":
      self.kernel_spec = "matern52"
    else:
      raise ValueError(f"Unsupported SMT corr '{corr}'")

    # --- hyperparameters / scaling ---
    theta = getattr(sm, "optimal_theta", None)
    if theta is None:
      theta = sm.corr.theta
    self.theta = np.asarray(theta, float).ravel()
    
    self.X_offset, self.X_scale = sm.X_offset, sm.X_scale
    self.Xc = (self.x - self.X_offset) / self.X_scale
    self._normalize = lambda x: (np.asarray(x, float) - self.X_offset) / self.X_scale

    y_mean = getattr(sm, "y_mean", None)
    y_std  = getattr(sm, "y_std",  None)
    if y_mean is None or y_std is None:
      # fall back to training y if available
      if self.y is not None:
        y_mean = float(np.mean(self.y))
        y_std  = float(np.std(self.y))
      else:
        y_mean, y_std = 0.0, 1.0
    if not np.isfinite(y_std) or abs(y_std) < 1e-15:
      y_std = 1.0  # avoid degenerate scaling
    assert len(y_mean) == 1, "expecting one entry"
    assert len(y_std) == 1, "expecting one entry"
    self.y_mean = float(y_mean[0])
    self.y_std  = float(y_std[0])
    
    # SMT uses a regression trend option ('poly'); enforce constant mean for your μ-bounds
    poly = sm.options["poly"] if "poly" in sm.options else "constant"
    if poly != "constant":
      raise NotImplementedError("μ-bounds assume poly='constant'")

    # params from training
    self.beta0 = float(np.asarray(par["beta"]).ravel()[0])
    self.gamma = np.asarray(par["gamma"], float).ravel()

    self.C = par["C"]
    self.sigma2 = np.asarray(par.get("sigma2", 1.0), float).reshape(()).item()
    self.sigma2_ri = float(par["sigma2_ri"] if "sigma2_ri" in par else self.sigma2)

    x_regression = np.array([np.mean(self.gpsurrogate.xlimits[i]) for i in range(self.gpsurrogate.ndim)])
    regression = (sm._regression_types['constant'](x_regression).T)
    regression = np.atleast_2d([regression[0][0]])
    w = linalg.solve_triangular(par["G"].T, regression)
    
    ntrain = sm.nt
    self.A_obj = 2.0 * (-1.0 * np.identity(ntrain) + par["Q"].dot(par["Q"].T))
    self.b_obj = -2.0 * par["Q"].dot(w)
    self.c_obj = 1. + np.inner(w[0], w[0])
    self.z = cp.Variable(ntrain)
    self.obj = 0.5 * cp.quad_form(self.z, self.A_obj) + self.b_obj.T @ self.z + self.c_obj

    

  def distance_bounds(self, l, u):
    # normalize the box
    l_c = self._normalize(l).ravel()
    u_c = self._normalize(u).ravel()

    Xc  = self.Xc                # (nt, d)
    th  = self.theta.ravel()     # (d,)
    spec = self.kernel_spec

    # per-point, per-dimension distance extremes (normalized space)
    dmin = np.maximum(0.0, np.maximum(l_c - Xc, Xc - u_c))        # (nt,d)
    dmax = np.maximum(np.abs(l_c - Xc), np.abs(u_c - Xc))         # (nt,d)
    return dmin, dmax   
  def ker_bounds(self, l, u):
   
    """
    Tight monotone bounds for k(x, X_i) over box [l,u] (original units).
    Returns kL, kU of shape (nt,), consistent with SMT’s kernels.
    """
    
    # normalize the box
    l_c = self._normalize(l).ravel()
    u_c = self._normalize(u).ravel()

    Xc  = self.Xc                # (nt, d)
    th  = self.theta.ravel()     # (d,)
    spec = self.kernel_spec

    # per-point, per-dimension distance extremes (normalized space)
    dmin = np.maximum(0.0, np.maximum(l_c - Xc, Xc - u_c))        # (nt,d)
    dmax = np.maximum(np.abs(l_c - Xc), np.abs(u_c - Xc))         # (nt,d)

    if spec == "pow_exp":
      # power-exponential: k = exp(-sum_j θ_j |dx_j|^p)
      p = getattr(self, "p", 2.0)
      s_min = (th * (dmin ** p)).sum(axis=1)
      s_max = (th * (dmax ** p)).sum(axis=1)
      kU = np.exp(-s_min)                                       # max on box
      kL = np.exp(-s_max)                                       # min on box
    elif spec == "matern12":
      # Matérn ν=1/2 (a.k.a. abs-exp): k = exp(-sum_j θ_j |dx_j|)
      s_min = (th * dmin).sum(axis=1)
      s_max = (th * dmax).sum(axis=1)
      kU = np.exp(-s_min)
      kL = np.exp(-s_max)
    elif spec == "matern32":
      # SMT separable form: ∏_j (1 + √3 θ_j |dx_j|) exp(-√3 θ_j |dx_j|)
      a = np.sqrt(3.0) * th
      gmin = (1 + a * dmin) * np.exp(-a * dmin)
      gmax = (1 + a * dmax) * np.exp(-a * dmax)
      kU = np.prod(gmin, axis=1)
      kL = np.prod(gmax, axis=1)
    elif spec == "matern52":
      # SMT separable form: ∏_j (1 + √5 θ_j |dx_j| + (5/3) θ_j^2 dx_j^2) exp(-√5 θ_j |dx_j|)
      b = np.sqrt(5.0) * th
      btmin, btmax = b * dmin, b * dmax
      gmin = (1 + btmin + (btmin**2)/3.0) * np.exp(-btmin)
      gmax = (1 + btmax + (btmax**2)/3.0) * np.exp(-btmax)
      kU = np.prod(gmin, axis=1)
      kL = np.prod(gmax, axis=1)
    else:
      raise ValueError(f"Unsupported kernel_spec: {spec}")
    return kL, kU

  def mu_bounds(self, kL, kU):
    # compute in normalized y-space
    lo = np.where(self.gamma >= 0.0, kL, kU)
    hi = np.where(self.gamma >= 0.0, kU, kL)
    mu_L_n = self.beta0 + float(np.dot(self.gamma, lo))  # normalized
    mu_U_n = self.beta0 + float(np.dot(self.gamma, hi))  # normalized

    # de-normalize like SMT
    mu_L = self.y_mean + self.y_std * mu_L_n
    mu_U = self.y_mean + self.y_std * mu_U_n
    return mu_L, mu_U

  def sigma2_bounds(self, kL, kU, l = None, u = None):
    """
    Variance bounds over r ∈ [kL,kU] for SMT KRG (poly='constant'),
    returned in ORIGINAL y-units (σ^2 * y_std^2).
    """
    
    kL = np.asarray(kL, float).ravel()
    kU = np.asarray(kU, float).ravel()
    n  = kL.size
    assert hasattr(self, "C") and hasattr(self, "sigma2"), "Call sync_from_smt() first"
    assert kL.shape == kU.shape == (n,) and np.all(kL <= kU), "kL/kU corrupted"

    # lower-bound
    if l is None or u is None:
      s2_L = 0.0
    else:
      x = np.atleast_2d( (l + u) / 2.)
      S2 = self.gpsurrogate.variance(x)
      s2_L = min(S2.flatten())

    # variance upper-bound (in terms of kernel k) defined by convex QP
    cons = [self.C @ self.z >= kL, self.C @ self.z <= kU]
    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP",verbose=False, eps_abs=1.e-14, eps_rel=1.e-10)
    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U



class BnBAlgorithm(BnBAlgorithmBase):
  def __init__(self, acqf, options = {}):
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    super().__init__(x = self.gpsurrogate.training_x, y = self.gpsurrogate.training_y)
    if not (isinstance(self.acqf, LCBacquisition) or isinstance(self.acqf, EIacquisition)):
      raise NotImplementedError("Unrecognized acquisition function type")
    self.sync_from_smt()
    
    # Stopping criteria parameters (default)    
    self.epsilon_gap = 1e-3
    self.epsilon_diam = 1e-2
    self.epsilon_prune = 1.e-14
    self.max_bnbiter = 2000
    self.nodes_per_batch = 1
    self.max_bnbtime = 12 * 60 # 12 minutes

    # Set options form command 
    self.epsilon_gap = options.get('epsilon_gap', self.epsilon_gap)
    self.epsilon_diam = options.get('epsilon_diam', self.epsilon_diam)
    self.epsilon_prune = options.get('epsilon_prune', self.epsilon_prune)
    self.max_bnbiter = options.get('max_iter', self.max_bnbiter)
    self.max_bnbtime = options.get('max_bnbtime', self.max_bnbtime)
    self.nodes_per_batch = options.get('nodes_per_batch', self.nodes_per_batch)

    if is_running_with_mpi():
      num_available_workers = MPI.COMM_WORLD.Get_size() - 1
      if num_available_workers > 1:
        # roughly evenly split workers for use in bbs and bfs evaluators
        #num_bbs_workers = num_available_workers
        #num_bfs_workers = 1
        num_bbs_workers = np.ceil(num_available_workers / 2).astype(int)
        num_bfs_workers = max(1, num_available_workers - num_bbs_workers)
      else:
        # num_available_workers == 1 or num_available_workers == 0
        # can occur when running on one process in which root is both master and the
        # only worker process. If there are 2 mpi processes then root is master and
        # there is one worker process. This worker process will be used for both
        # bbs and bfs evaluators
        num_bbs_workers = 1
        num_bfs_workers = 1
    else:
      num_bbs_workers = 1
      num_bfs_workers = 1
    self.bbsevaluator = MPIEvaluator(function_mode=False, max_workers = num_bbs_workers)
    self.bfsevaluator = MPIEvaluator(function_mode=False, max_workers = num_bfs_workers)  
    self.num_bbs_workers = num_bbs_workers
    self.num_bfs_workers = num_bfs_workers
    self.max_queue_size = 10 * self.num_bbs_workers

  # For minimization, we find a feasible function value as the upper bound on the minimum value of the acquisition function.
  def compute_acqf_upper_bound(self, l, u):
    # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
    # Compute the kernel bounds with given x
    kL, kU = self.ker_bounds(l, u)
    # Compute the mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    return self.acqf.evaluate_meansig2(np.atleast_1d(mu_U), np.atleast_1d(var_L))[0]
  # For minimization, we compute the lower bound explicitly using the acquisition function over mu, sigma.
  def compute_acqf_lower_bound(self, l, u):
    # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
    # Compute the kernel bounds with given x
    kL, kU = self.ker_bounds(l, u)
    # Compute the mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L,var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    return self.acqf.evaluate_meansig2(np.atleast_1d(mu_L), np.atleast_1d(var_U))[0]
  def compute_acqf_bounds(self, l, u):
    # kernel bounds
    kL, kU = self.ker_bounds(l, u)
    # mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
    mu  = np.array([mu_L, mu_U])
    var = np.array([var_U, var_L])
    acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
    
    #x_midpoint = np.atleast_2d(( l + u) / 2.)
    #acqf_U = self.acqf.evaluate(x_midpoint).flatten()[0]
    #n_points = 3 ** self.gpsurrogate.ndim
    #x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #for i in range(n_points):
    #  for j in range(self.gpsurrogate.ndim):
    #    x_points[i, j] = l[j] + (u[j] - l[j]) * (np.floor(i / (3**j)).astype(int) % 3).astype(float) / 2.
    #n_points = self.gpsurrogate.ndim
    #x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #for i in range(n_points):
    #  for j in range(self.gpsurrogate.ndim):
    #    x_points[i, j] = l[j] + (u[j] - l[j]) * float((i + j) % self.gpsurrogate.ndim) / float(self.gpsurrogate.ndim)
    n_points = 1
    x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    for i in range(n_points):
      for j in range(self.gpsurrogate.ndim):
        x_points[i, j] = (l[j] + u[j]) / 2.
    acqf_eval = self.acqf.evaluate(x_points)
    acqf_U = min(acqf_eval.flatten())
    if acqf_U < acqf_bounds[0]:
      print("ERROR in bound computations U < L")
      print(f"Acquisition function evaluations for node defined by bounds: {l} {u}")
      for i in range(n_points):
        print(f"acqf({x_points[i,:]}) = {acqf_eval[i]}")
      if np.any(acqf_eval >= acqf_bounds[0]):
        print("one point evaluation >= L")
        #feasible_idxs = np.argwhere(acqf_eval >= acqf_bounds[0])
        #acqf_eval = acqf_eval[feasible_idxs]
        #acqf_eval.sort()
        #acqf_U = min(acqf_eval)
      else:
        print("all point evaluations < L")
    
    return acqf_bounds[0], acqf_U
  def _prune_queue(self, queue, lub, eps):
    """Keep only nodes whose lower-bound is not greater or equal least upper-bound + eps; then re-heapify."""
    # queue items are (L, counter, node)
    pruned = [(L, c, n) for (L, c, n) in queue if L < lub + eps]
    heapq.heapify(pruned)
    return pruned
  def _prune_node_list(self, node_list, lub, eps):
    """Keep only nodes whose lower-bound is not greater or equal least upper-bound + eps."""
    pruned_node_list = [node for node in node_list if node.aq_L < lub + eps]
    return pruned_node_list 
  def optimize(self):
    opt = self.bnboptimize(self.gpsurrogate.xlimits[:,0], self.gpsurrogate.xlimits[:,1])
    lopt = opt[0]
    uopt = opt[1]
    midpoint_opt = np.mean(np.array([lopt, uopt]), axis=0)
    return midpoint_opt
  def initialize(self, l0 = None, u0 = None, queue = None):
    """
    Initialization, perhaps use an old tree structure given by optional queue
    argument
    """ 
    if l0 is None or u0 is None:
      l_init = self.gpsurrogate.xlimits[:,0]
      u_init = self.gpsurrogate.xlimits[:,1]
    else:
      # to do check that l0 and u0 is right shape
      l_init = l0
      u_init = u0
    # Root bounds
    aq_L_val, aq_U_val = self.compute_acqf_bounds(l_init, u_init) 
    print(f"\nInitial acquisition lower bound: {aq_L_val}")
    print(f"Initial acquisition upper bound: {aq_U_val}")

    # Init root + heap ordered by aq_L
    root = BnBNode(l_init, u_init, aq_L_val, aq_U_val)
    
    # --- HEAP STORES TUPLES: (L, counter, node) ---
    self._ctr = getattr(self, "_ctr", count())
    self.queue = [(root.aq_L, next(self._ctr), root)]
    """
    either use old queue to determine LUB or the previous queue provided
    as an argument
    """
    # Least upper bound (LUB)
    self.best_node = root
    self.LUB = self.best_node.aq_U
    if queue is not None:
      for _, _, node in queue:
        acqf_L, acqf_U = self.compute_acqf_bounds(node.l, node.u)
        if acqf_U < self.LUB:
          self.LUB = acqf_U
        
  def bnboptimize(self, l_init, u_init):
    """
    Branch & Bound minimization with tolerance stopping.
    Core logic only: correct heap order, pruning on LUB tightening,
    single global stop, diameter continue, consistent per-node prune.
    """
    print("=== Starting Branch & Bound Acquisition Function Optimization ===")
    print(f"=== Over lower/upper bounds: l = {l_init}, u = {u_init} ===")

    heapq.heapify(self.queue)    

    all_bfsnodes = []

    
    # stopping criterion should be on the total maximum number of branched nodes
    self.num_branches = 0

    initial_gap = self.best_node.aq_U - self.best_node.aq_L

    max_bbs_node_size = 0
    max_bfs_node_size = 0
    start_time = time.time()
    while self.num_branches < self.max_bnbiter: # iteration limit
      if time.time() - start_time > self.max_bnbtime: # time limit
        print("maximum time has elapsed")
        break
      
      # -- retrieve submitted tasks -- 
      # asynchronously retrieve results from Evaluator that have been processed
      #self.bbsevaluator.sync()
      bbschildren = self.bbsevaluator.retrieve_results()

      # not all children are return, hence children is a ragged array
      # need to flatten this ragged list
      bbschildren = [item for sublist in bbschildren for item in sublist]

      #self.bfsevaluator.sync()
      bfschildren = self.bfsevaluator.retrieve_results()
      bfschildren = [item for sublist in bfschildren for item in sublist]

      children = bbschildren + bfschildren # join child lists
      if len(children) == 0:
        if len(self.queue) == 0 and len(all_bfsnodes) == 0:
          if self.bbsevaluator.num_submitted_tasks() == 0 and self.bfsevaluator.num_submitted_tasks() == 0:
            print("no child nodes recovered from evaluators")
            print("no nodes in bfs/bbs queue lists")
            print("evaluators have no tasks to be evaluated")
            print("exiting")
            exit()
      else:
        self.num_branches += len(children)
        print(f"elapsed time: {time.time() - start_time}")
        print(f"evaluators returned {len(children)} children")
        # update best_node via children
        updated_best_node = False
        for child in children:
          if child.aq_U < child.aq_L:
            print("ERROR: child upper bound < child lower bound")
            print(f"upper - lower = {child.aq_U - child.aq_L}")
            exit()
          if child.aq_U <= self.LUB:
            self.best_node = child
            self.LUB = self.best_node.aq_U
            updated_best_node = True
        if not updated_best_node:
          print("best node not updated")
        else:
          print("best node updated")
        
        # pre-prune
        children_lower_bounds = [child.aq_L for child in children]
        args = np.argwhere(np.array(children_lower_bounds) < self.LUB + self.epsilon_prune).flatten()
        print(f"{len(args)} children to be appended to bbs/bfs lists")
        children = [children[arg] for arg in args]

        # now move pruned children to data structs for (potential) future evaluation
        children_lower_bounds = [child.aq_L for child in children]
        # sort the children in order of increasing acqf lower-bounds
        args = np.argsort(children_lower_bounds)
        children = [children[arg] for arg in args]
        for child in children:
          if len(self.queue) < self.max_queue_size:
            heapq.heappush(self.queue, (child.aq_L, next(self._ctr), child))
          else:
            all_bfsnodes.append(child)
        max_bbs_node_size = max(max_bbs_node_size, len(self.queue))
        max_bfs_node_size = max(max_bfs_node_size, len(all_bfsnodes))
        
        # reprune
        #print(f"|bbs nodes| = {len(self.queue)}, |bfs nodes| = {len(all_bfsnodes)} (prior to pruning)")
        self.queue = self._prune_queue(self.queue, self.LUB, self.epsilon_prune)
        all_bfsnodes = self._prune_node_list(all_bfsnodes, self.LUB, self.epsilon_prune)
        #print(f"|bbs nodes| = {len(self.queue)}, |bfs nodes| = {len(all_bfsnodes)} (after pruning)")
           

        # BnB opt progress report 
        gap = self.best_node.aq_U - self.best_node.aq_L
        print(f"\n--- Total number branches  {self.num_branches} ---")
        print(f"Best node bounds: l={self.best_node.l}, u={self.best_node.u}")
        print(f"Node acquisition bounds: L={self.best_node.aq_L}, U={self.best_node.aq_U}")
        print(f"Current best feasible value (LUB): {self.LUB}")
        print(f"gap = {gap}")
        print(f"size of bbs queue = {len(self.queue)}")
        print(f"size of bfs node list = {len(all_bfsnodes)}")
        print(f"number of submitted jobs (bbs): {self.bbsevaluator.num_submitted_tasks()}")
        print(f"number of submitted jobs (bfs): {self.bfsevaluator.num_submitted_tasks()}")
        print(f"total elapsed time: {time.time() - start_time}")
        print(f"--- ---\n")


        if updated_best_node:
          if gap  < self.epsilon_gap:
            print(f"STOP: optimality gap = {gap} < {self.epsilon_gap}")
            break
  
    
      # -- submit new tasks --


      # if the number of submitted jobs is too large then wait for some jobs to be processed
      if self.bbsevaluator.num_submitted_tasks() + self.bfsevaluator.num_submitted_tasks() > 10 * (self.num_bbs_workers + self.num_bfs_workers):

      
      
      # collect nodes to be branched on in list structure
      bbsnodes = []
      # only submit additional tasks if there aren't too many in the Evaluators queue
      if self.bbsevaluator.num_submitted_tasks() < 10 * self.num_bbs_workers:
        for i in range(self.nodes_per_batch):
          if (not self.queue):
            break # no more nodes available to send to evaluator for branching/bound computations
          _, _, node = heapq.heappop(self.queue)
          bbsnodes.append(node)

        # parallel branching and upper/lower bound node compuatations
        brancher = branching_wrapper(self.acqf, LUB = self.LUB, epsilon_prune=self.epsilon_prune)
        bbsnodes = np.array(bbsnodes)
        if len(bbsnodes) > 0:
          self.bbsevaluator.submit_tasks(brancher.callback, bbsnodes)
      
      bfsnodes  = []
      # only submit additional tasks if there aren't too many in the Evaluators queue
      if self.bfsevaluator.num_submitted_tasks() < 10 * self.num_bfs_workers:
        #n = len(all_bfsnodes)
        for i in range(self.nodes_per_batch):
        #for i in range(n):
          if len(all_bfsnodes) == 0: 
            break # no more nodes available to send to evaluator for branching/bound computations
          node = all_bfsnodes.pop(0)
          bfsnodes.append(node)
        bfsnodes = np.array(bfsnodes)
        if len(bfsnodes) > 0:
          self.bfsevaluator.submit_tasks(brancher.callback, bfsnodes)

    print("\n=== Optimization Finished ===")
    print(f"Total number of branches: {self.num_branches}")
    print(f"Max BBS node list size: {max_bbs_node_size}")
    print(f"Max BFS node list size: {max_bfs_node_size}")
    print(f"Best bounds: l={self.best_node.l}, u={self.best_node.u}")
    print(f"Best feasible acquisition value (LUB): {self.LUB}")
    print(f"Initial gap: {initial_gap}")
    print(f"Final gap: {gap}")
    print(f"Total elapsed time: {time.time() - start_time}")

    return self.best_node.l, self.best_node.u, self.LUB


class branching_wrapper:
  def __init__(self, acqf, LUB=np.inf, epsilon_prune=1.e-14):
    self.LUB = LUB # least upper bound
    self.epsilon_prune = epsilon_prune
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    self.x = self.gpsurrogate.training_x
    self.y = self.gpsurrogate.training_y
    if not (isinstance(self.acqf, LCBacquisition) or isinstance(self.acqf, EIacquisition)):
      raise NotImplementedError("Unrecognized acquisition function type")
    self.sync_from_smt()
  
  def sync_from_smt(self):
    sm = self.gpsurrogate.surrogatesmt
    par = sm.optimal_par

    # --- kernel / corr selection ---
    corr = sm.options["corr"]  # e.g., 'squar_exp', 'pow_exp', 'abs_exp', 'matern32', 'matern52'
    if corr == "pow_exp":
      # OptionsDictionary -> use membership + indexing (no .get)
      p = float(sm.options["pow_exp_power"]) if "pow_exp_power" in sm.options else 2.0
      if p not in (1.0, 2.0):
        # tighten if your 1D bound code only supports p in {1,2}
        raise ValueError("Single-d bounds support pow_exp only for p=1 or p=2")
      self.kernel_spec = "pow_exp"
      self.p = p
    elif corr == "squar_exp":
      # Gaussian is pow_exp with p=2
      self.kernel_spec = "pow_exp"
      self.p = 2.0
    elif corr == "abs_exp":
      # Exponential is pow_exp with p=1
      self.kernel_spec = "pow_exp"
      self.p = 1.0
    elif corr == "matern32":
      self.kernel_spec = "matern32"
    elif corr == "matern52":
      self.kernel_spec = "matern52"
    else:
      raise ValueError(f"Unsupported SMT corr '{corr}'")

    # --- hyperparameters / scaling ---
    theta = getattr(sm, "optimal_theta", None)
    if theta is None:
      theta = sm.corr.theta
    self.theta = np.asarray(theta, float).ravel()
    
    self.X_offset, self.X_scale = sm.X_offset, sm.X_scale
    self.Xc = (self.x - self.X_offset) / self.X_scale

    y_mean = getattr(sm, "y_mean", None)
    y_std  = getattr(sm, "y_std",  None)
    if y_mean is None or y_std is None:
      # fall back to training y if available
      if self.y is not None:
        y_mean = float(np.mean(self.y))
        y_std  = float(np.std(self.y))
      else:
        y_mean, y_std = 0.0, 1.0
    if not np.isfinite(y_std) or abs(y_std) < 1e-15:
      y_std = 1.0  # avoid degenerate scaling
    assert len(y_mean) == 1, "expecting one entry"
    assert len(y_std) == 1, "expecting one entry"
    self.y_mean = float(y_mean[0])
    self.y_std  = float(y_std[0])
    
    # SMT uses a regression trend option ('poly'); enforce constant mean for your μ-bounds
    poly = sm.options["poly"] if "poly" in sm.options else "constant"
    if poly != "constant":
      raise NotImplementedError("μ-bounds assume poly='constant'")

    # params from training
    self.beta0 = float(np.asarray(par["beta"]).ravel()[0])
    self.gamma = np.asarray(par["gamma"], float).ravel()

    self.C = par["C"]
    self.sigma2 = np.asarray(par.get("sigma2", 1.0), float).reshape(()).item()
    self.sigma2_ri = float(par["sigma2_ri"] if "sigma2_ri" in par else self.sigma2)

    x_regression = np.array([np.mean(self.gpsurrogate.xlimits[i]) for i in range(self.gpsurrogate.ndim)])
    regression = (sm._regression_types['constant'](x_regression).T)
    regression = np.atleast_2d([regression[0][0]])
    w = linalg.solve_triangular(par["G"].T, regression)

    ntrain = sm.nt
    self.A_obj = 2.0 * (-1.0 * np.identity(ntrain) + par["Q"].dot(par["Q"].T))
    self.b_obj = -2.0 * par["Q"].dot(w)
    self.c_obj = 1. + np.inner(w[0], w[0])
    self.z = cp.Variable(ntrain)
    self.obj = 0.5 * cp.quad_form(self.z, self.A_obj) + self.b_obj.T @ self.z + self.c_obj
  
  def _normalize(self, x):
    return (np.asarray(x, float) - self.X_offset) / self.X_scale
  
  def ker_bounds(self, l, u):
   
    """
    Tight monotone bounds for k(x, X_i) over box [l,u] (original units).
    Returns kL, kU of shape (nt,), consistent with SMT’s kernels.
    """
    
    # normalize the box
    l_c = self._normalize(l).ravel()
    u_c = self._normalize(u).ravel()

    Xc  = self.Xc                # (nt, d)
    th  = self.theta.ravel()     # (d,)
    spec = self.kernel_spec

    # per-point, per-dimension distance extremes (normalized space)
    dmin = np.maximum(0.0, np.maximum(l_c - Xc, Xc - u_c))        # (nt,d)
    dmax = np.maximum(np.abs(l_c - Xc), np.abs(u_c - Xc))         # (nt,d)
    """
    Question: can we access the kernel directly from the gp surrogate and not
              have this additional code that we have to ensure is consistent with smt?
    """    
    if spec == "pow_exp":
      # power-exponential: k = exp(-sum_j θ_j |dx_j|^p)
      p = getattr(self, "p", 2.0)
      s_min = (th * (dmin ** p)).sum(axis=1)
      s_max = (th * (dmax ** p)).sum(axis=1)
      kU = np.exp(-s_min)                                       # max on box
      kL = np.exp(-s_max)                                       # min on box
    elif spec == "matern12":
      # Matérn ν=1/2 (a.k.a. abs-exp): k = exp(-sum_j θ_j |dx_j|)
      s_min = (th * dmin).sum(axis=1)
      s_max = (th * dmax).sum(axis=1)
      kU = np.exp(-s_min)
      kL = np.exp(-s_max)
    elif spec == "matern32":
      # SMT separable form: ∏_j (1 + √3 θ_j |dx_j|) exp(-√3 θ_j |dx_j|)
      a = np.sqrt(3.0) * th
      gmin = (1 + a * dmin) * np.exp(-a * dmin)
      gmax = (1 + a * dmax) * np.exp(-a * dmax)
      kU = np.prod(gmin, axis=1)
      kL = np.prod(gmax, axis=1)
    elif spec == "matern52":
      # SMT separable form: ∏_j (1 + √5 θ_j |dx_j| + (5/3) θ_j^2 dx_j^2) exp(-√5 θ_j |dx_j|)
      b = np.sqrt(5.0) * th
      btmin, btmax = b * dmin, b * dmax
      gmin = (1 + btmin + (btmin**2)/3.0) * np.exp(-btmin)
      gmax = (1 + btmax + (btmax**2)/3.0) * np.exp(-btmax)
      kU = np.prod(gmin, axis=1)
      kL = np.prod(gmax, axis=1)
    else:
      raise ValueError(f"Unsupported kernel_spec: {spec}")
    return kL, kU
  
  def mu_bounds(self, kL, kU):
    # compute in normalized y-space
    lo = np.where(self.gamma >= 0.0, kL, kU)
    hi = np.where(self.gamma >= 0.0, kU, kL)
    mu_L_n = self.beta0 + float(np.dot(self.gamma, lo))  # normalized
    mu_U_n = self.beta0 + float(np.dot(self.gamma, hi))  # normalized

    # de-normalize like SMT
    mu_L = self.y_mean + self.y_std * mu_L_n
    mu_U = self.y_mean + self.y_std * mu_U_n
    return mu_L, mu_U
  
  def sigma2_bounds(self, kL, kU, l = None, u = None):
    """
    Variance bounds over r ∈ [kL,kU] for SMT KRG (poly='constant'),
    returned in ORIGINAL y-units (σ^2 * y_std^2).
    """
    
    kL = np.asarray(kL, float).ravel()
    kU = np.asarray(kU, float).ravel()
    n  = kL.size

    # lower-bound
    if l is None or u is None:
      s2_L = 0.0
    else:
      x = np.atleast_2d( (l + u) / 2. )
      S2 = self.gpsurrogate.variance(x)
      s2_L = min(S2.flatten())

    # variance upper-bound (in terms of kernel k) defined by convex QP
    cons = [self.C @ self.z >= kL, self.C @ self.z <= kU]
    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP",verbose=False,eps_abs=1.e-14, eps_rel=1.e-10)

    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U
  
  def compute_acqf_bounds(self, l, u):
    # kernel bounds
    kL, kU = self.ker_bounds(l, u)
    # mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
    mu  = np.array([mu_L, mu_U])
    var = np.array([var_U, var_L])
    acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
    #n_points = 3 ** self.gpsurrogate.ndim
    #x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #for i in range(n_points):
    #  for j in range(self.gpsurrogate.ndim):
    #    x_points[i, j] = l[j] + ((u[j] - l[j])/2.) * np.floor(i / (3**j)).astype(int) % 3
    #n_points = 1
    #x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #for j in range(self.gpsurrogate.ndim):
    #  x_points[0, j] = (l[j] + u[j]) / 2.
    #n_points = self.gpsurrogate.ndim
    #x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #for i in range(n_points):
    #  for j in range(self.gpsurrogate.ndim):
    #    x_points[i, j] = l[j] + (u[j] - l[j]) * float((i + j) % self.gpsurrogate.ndim) / float(self.gpsurrogate.ndim)
    n_points = 1
    x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    for i in range(n_points):
      for j in range(self.gpsurrogate.ndim):
        x_points[i, j] = (l[j] + u[j]) / 2.
    acqf_eval = self.acqf.evaluate(x_points)
    acqf_U = min(acqf_eval.flatten())
    if acqf_bounds[0] > acqf_U:
      print("ERROR in bound computations U < L")
      print(f"Acquisition function evaluations for node defined by bounds: {l} {u}")
      for i in range(n_points):
        print(f"acqf({x_points[i,:]}) = {acqf_eval[i]}")
      if np.any(acqf_eval >= acqf_bounds[0]):
        print("one point evaluation >= L")
      else:
        print("all point evaluations < L")

    #x_midpoint = np.atleast_2d(( l + u) / 2.)
    #acqf_U = self.acqf.evaluate(x_midpoint).flatten()[0]

    #acqf_callback = {'obj' : self.acqf.scalar_evaluate}
    #if self.acqf.has_gradient:
    #  acqf_callback['grad'] = self.acqf.scalar_eval_g
    #minimizer_method = "SLSQP"
    #minimizer_options = {"maxiter" : 100}
    #minimizer_constraints = ()
    #acqf_minimizer = minimizer_wrapper(acqf_callback, minimizer_method, self.gpsurrogate.xlimits, minimizer_constraints, minimizer_options)
    #x0_pts = np.array([[uniform(b[0], b[1]) for b in self.gpsurrogate.xlimits] for _ in range(1)])

    #opt_evaluator = Evaluator()    
    #opt_output = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0_pts)[0]
    #assert opt_output[2], f"local optimizer failed"


    return acqf_bounds[0], acqf_U
  def callback(self, nodes):
    output = []
    for node in nodes.flatten():
      for child_l, child_u in branch(node.l, node.u):
        acqf_L, acqf_U = self.compute_acqf_bounds(child_l, child_u)
        child = BnBNode(child_l, child_u, acqf_L, acqf_U)
        output.append(child)
    return [output]
