import numpy as np
import cvxpy as cp
import heapq
from scipy import linalg
from scipy.stats import norm
from scipy.optimize import minimize
from .acquisition import EIacquisition, LCBacquisition
from .opt_utils import minimizer_wrapper
from ..utils.util import Evaluator
from numpy.random import uniform
from itertools import count

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


class BnBAlgorithmBase:
  def __init__(self, x = None, y = None):
    # Node class for priority queue
    self.BnBNode = BnBNode

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
    self.total_nodes = 0
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
    
    # For LCB, need to eventually pull this from the BO Acquisition Function class
    self.beta = 3.0

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

    x_regression = [np.mean(self.gpsurrogate.xlimits[i]) for i in range(self.gpsurrogate.ndim)]
    w = linalg.solve_triangular(par["G"].T, sm._regression_types['constant'](x_regression).T)
    
    ntrain = sm.nt
    self.A_obj = 2.0 * (-1.0 * np.identity(ntrain) + par["Q"].dot(par["Q"].T))
    self.b_obj = -2.0 * par["Q"].dot(w)
    self.c_obj = 1. + np.inner(w[0], w[0])
    self.z = cp.Variable(ntrain)
    self.obj = 0.5 * cp.quad_form(self.z, self.A_obj) + self.b_obj.T @ self.z + self.c_obj

    

  def set_kernel(self, kernel_spec):
    assert kernel_spec in ["abs_exp", "pow_exp", "matern32", "matern52"]
    self.kernel_spec = kernel_spec

    if kernel_spec == "abs_exp":  # ν = 1/2
      self.kernel_func = lambda d: np.exp(-np.sqrt(d))

    elif kernel_spec == "pow_exp":  # SE, ν = ∞
      self.kernel_func = lambda d: np.exp(-d)

    elif kernel_spec == "matern32":  # ν = 3/2
      self.kernel_func = lambda d: (
          (1 + np.sqrt(3) * np.sqrt(d)) *
          np.exp(-np.sqrt(3) * np.sqrt(d))
      )

    elif kernel_spec == "matern52":  # ν = 5/2
      self.kernel_func = lambda d: (
          (1 + np.sqrt(5) * np.sqrt(d) + (5/3) * d) *
          np.exp(-np.sqrt(5) * np.sqrt(d))
      )
 
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

  def sigma2_bounds(self, kL, kU, l = None, u = None, clip_nonneg=True):
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
    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP")
    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U



class BnBAlgorithm(BnBAlgorithmBase):
  def __init__(self, acqf, options = {}):
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    super().__init__(x = self.gpsurrogate.training_x, y = self.gpsurrogate.training_y)
    if isinstance(self.acqf, LCBacquisition):
      self.acquisition_type = "LCB"
    elif isinstance(self.acqf, EIacquisition):
      self.acquisition_type = "EI"
    else:
      raise NotImplementedError("Unrecognized acquisition function type")
    self.sync_from_smt()
    
    # Stopping criteria parameters (default)    
    self.epsilon_gap = 1e-3
    self.epsilon_diam = 1e-2
    self.epsilon_prune = 1.e-14
    self.max_bnbiter = 2000

    # Set options form command 
    self.epsilon_gap = options.get('epsilon_gap', self.epsilon_gap)
    self.epsilon_diam = options.get('epsilon_diam', self.epsilon_diam)
    self.epsilon_prune = options.get('epsilon_prune', self.epsilon_prune)
    self.max_bnbiter = options.get('max_iter', self.max_bnbiter)

  def _branch(self, l, u):
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
    
    x_midpoint = np.atleast_2d(( l + u) / 2.)
    acqf_U = self.acqf.evaluate(x_midpoint).flatten()[0]
    return acqf_bounds[0], acqf_U
  def _prune_queue(self, queue, gub, eps):
    """Keep only nodes that can beat current GUB within tolerance; then re-heapify."""
    # queue items are (L, counter, node)
    pruned = [(L, c, n) for (L, c, n) in queue if L <= gub + eps]
    heapq.heapify(pruned)
    return pruned
  def optimize(self):
    opt = self.bnboptimize(self.gpsurrogate.xlimits[:,0], self.gpsurrogate.xlimits[:,1])
    lopt = opt[0]
    uopt = opt[1]
    midpoint_opt = np.mean(np.array([lopt, uopt]), axis=0)
    return midpoint_opt   
  def bnboptimize(self, l_init, u_init):
    """
    Branch & Bound minimization with tolerance stopping.
    Core logic only: correct heap order, pruning on GUB tightening,
    single global stop, diameter continue, consistent per-node prune.
    """
    print("=== Starting Branch & Bound Optimization (Minimization) ===")
    print(f"Initial bounds: l = {l_init}, u = {u_init}")
    print(f"Number of points: {self.x.shape[0]}, Dim = {self.x.shape[1]}")

    # Root bounds
    aq_L_val, aq_U_val = self.compute_acqf_bounds(l_init, u_init) 
    print(f"\nInitial acquisition lower bound: {aq_L_val}")
    print(f"Initial acquisition upper bound: {aq_U_val}")

    # Init root + heap ordered by aq_L
    root = BnBNode(l_init.astype(float), u_init.astype(float), aq_L_val, aq_U_val)
    
    # --- HEAP STORES TUPLES: (L, counter, node) ---
    self._ctr = getattr(self, "_ctr", count())
    queue = [(root.aq_L, next(self._ctr), root)]
    heapq.heapify(queue)

    # Least upper bound (LUB)
    self.LUB = aq_U_val
    self.best_l, self.best_u = l_init.copy(), u_init.copy()

    diameters = [float(np.max(u_init - l_init))]
    self.total_nodes = 1
    bnb_iter = 0

    while queue:
      bnb_iter += 1
      if bnb_iter > self.max_bnbiter:
        print(f"max bnb iterations ({self.max_bnbiter}) reached")
        return queue
        #break
      # pop the node with smallest L
      L_top, _, node = heapq.heappop(queue)

      # sanity: popped L must equal node.aq_L and be <= any remaining L
      assert abs(L_top - node.aq_L) <= 1e-12
      if queue:
        min_rest = min(L for (L,_,_) in queue)
        assert L_top <= min_rest + 1e-12, f"Heap not ordered by L (popped {L_top}, min_rest {min_rest})"

      # Update LUB and prune
      if node.aq_U < self.LUB:
        self.LUB = node.aq_U
        self.best_l, self.best_u = node.l, node.u
        queue = self._prune_queue(queue, self.LUB, self.epsilon_gap)

      print(f"\n--- BnB Iteration {bnb_iter} ---")
      print(f"Node bounds: l={node.l}, u={node.u}")
      print(f"Node acquisition bounds: L={node.aq_L}, U={node.aq_U}")
      print(f"Current best feasible value (LUB): {self.LUB}")

      # Stopping criterion: LUB - node_LB <= eps_gap (node_LB is least lower-bound)
      if self.LUB - node.aq_L <= self.epsilon_gap:
        print(f"STOP: LUB - Node.L = {self.LUB - node.aq_L} <= {self.epsilon_gap}")
        break

      # Diameter stop (node-local)
      node_diam = float(np.max(node.u - node.l))
      if node_diam <= self.epsilon_diam:
        print(f"Skip: Node diameter {node_diam} <= {self.epsilon_diam}")
        continue

      # Per-node prune (consistent with stop rule)
      if node.aq_L >= self.LUB + self.epsilon_prune:
        print("Pruned: Node cannot improve best within tolerance.")
        continue

      # --- Branch ---
      # --- this is where parallelism could show up
      # --- branch over all nodes (parallel over said nodes)?
      # --- do we do a more refined branching
      # --- rather than branching one node into 2 nodes we can
      # --- branch one node into 2^k nodes
      for l_child, u_child in self._branch(node.l, node.u):
        print(f"  Branching to child: l={l_child}, u={u_child}")

        aq_L_r, aq_U_r = self.compute_acqf_bounds(l_child, u_child)
        print(f"  Child acquisition bounds: L={aq_L_r}, U={aq_U_r}")

        self.total_nodes += 1
        diameters.append(float(np.max(u_child - l_child)))

        # Update LUB from child and prune if improved
        if aq_U_r < self.LUB:
          self.LUB = aq_U_r
          self.best_l, self.best_u = l_child, u_child
          queue = self._prune_queue(queue, self.LUB, self.epsilon_prune)

        # Child-level prune (same tolerance)
        if aq_L_r >= self.LUB + self.epsilon_prune:
          print("  Child pruned: L ≥ LUB + eps.")
          continue

        # PUSH AS TUPLE
        child = BnBNode(l_child, u_child, aq_L_r, aq_U_r)
        heapq.heappush(queue, (child.aq_L, next(self._ctr), child))

      if not queue:
        print("\nSTOP: Queue empty, no better nodes remain.")
        break

      # Optional visibility
      phi_LB = min(L for (L,_,_) in queue)
      print(f"Queue size: {len(queue)} | LB={phi_LB} | LUB={self.LUB} | Gap<={self.LUB - phi_LB}")

    self.final_gap = self.LUB - min([L for (L,_,_) in queue], default=self.LUB)
    self.final_diameter = min(diameters) if diameters else float('inf')

    print("\n=== Optimization Finished ===")
    print(f"Total nodes explored: {self.total_nodes}")
    print(f"Best bounds: l={self.best_l}, u={self.best_u}")
    print(f"Best feasible acquisition value (GUB): {self.LUB}")
    print(f"Final gap: {self.final_gap}, final diameter: {self.final_diameter}")

    return self.best_l, self.best_u, self.LUB
