import numpy as np
import cvxpy as cp
import heapq
from scipy import linalg
from scipy.stats import norm
from scipy.optimize import minimize
from .acquisition import EIacquisition, LCBacquisition
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
    self.BnB_LBmethod = "IPOPT"
    #self.BnB_LBmethod = None  # Use CVXPY for lower bounds

    # Stopping criteria
    self.epsilon_gap = 1e-3
    self.epsilon_diam = 1e-2

    # Kernel info for bounds
    self.kernel_spec = None
    self.kernel_func = None
    self.y_min = None

    # Evaluation parameters
    self.theta = None
    self.ell = None  # ARD scaling for distance

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
    self.c_obj = 1. + np.dot(w, w)
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

  def sigma2_bounds(self, kL, kU, clip_nonneg=True):
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
    s2_L_n = 0.0

    # variance upper-bound (in terms of kernel k) defined by convex QP
    cons = [self.C @ self.z >= kL, self.C @ self.z <= kU]
    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP")
    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_L = s2_L_n * self.sigma2
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U

  def rs_ei(self, mu, sigma):
    y_min = np.min(self.y)
    if sigma > 1e-12:
      z = (y_min - mu) / sigma
      ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
      return -ei
    else:
      # Deterministic case: EI = max(y_min - mu, 0)
      return -max(y_min - mu, 0.0)
  def rs_lcb(self, mu, sigma):
    return mu - self.beta * sigma






class BnBAlgorithm(BnBAlgorithmBase):
  def __init__(self, x, y, gpsurrogate, acqf_minimizer_callback, acquisition_type):
    super().__init__(x = x, y =y)
    self.gpsurrogate = gpsurrogate
    self.acqf_minimizer_callback = acqf_minimizer_callback
    self.acquisition_type = acquisition_type
    self.evaluator = Evaluator() 
    self.sync_from_smt()

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
  def compute_acq_upper_bound(self, l, u):
    if self.BnB_LBmethod == "IPOPT":
      if self.acquisition_type == "LCB":
        acqf = LCBacquisition(self.gpsurrogate)
      elif self.acquisition_type == "EI":
        acqf = EIacquisition(self.gpsurrogate)
      else:
        raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)
      acqf_callback = {'obj': acqf.scalar_evaluate}
      if acqf.has_gradient:
        acqf_callback['grad'] = acqf.scalar_eval_g
      x0 = np.array([[uniform(l[i], u[i]) for i in range(len(l))] for _ in range(1)])
      opt_output = self.evaluator.run(self.acqf_minimizer_callback, x0)
      xopt, yout, success, _ = opt_output[0] 
      if not success:
        raise RuntimeError("EI maximization failed")
      return float(yout)
    else:
      # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
      # Compute the kernel bounds with given x
      kL, kU = self.ker_bounds(l, u)
      # Compute the mean bounds
      mu_L, mu_U = self.mu_bounds(kL, kU)
      var_L, var_U = self.sigma2_bounds(kL, kU)
      if self.acquisition_type == "LCB":
        lcb_U = self.rs_lcb(mu_U, np.sqrt(var_L))
        return lcb_U
      elif self.acquisition_type == "EI":
        ei_U = self.rs_ei(mu_U, np.sqrt(var_L))
        return ei_U
      else:
        raise NotImplementedError(self.acquisition_type + " acquisition_type not supported")
  # For minimization, we compute the lower bound explicitly using the acquisition function over mu, sigma.
  def compute_acq_lower_bound(self, l, u):
    # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
    # Compute the kernel bounds with given x
    kL, kU = self.ker_bounds(l, u)
    # Compute the mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L,var_U = self.sigma2_bounds(kL, kU)
    
    if self.enable_debug_checks:
      l_check = np.asarray(l).reshape(-1)
      u_check = np.asarray(u).reshape(-1)
      c_check = 0.5*(l+u)

      # center + for each axis i: set x_i to l_i and u_i, others at center
      Xchk = [c_check]
      for i in range(l.size):
        x_lo = c_check.copy(); x_lo[i] = l_check[i]; Xchk.append(x_lo)
        x_hi = c_check.copy(); x_hi[i] = u_check[i]; Xchk.append(x_hi)
      Xchk = np.vstack(Xchk)  # shape (1+2d, d)

      # GP mean at those points (vector length 1+2d)
      mu_vec = np.asarray(self.gpsurrogate.mean(Xchk)).reshape(-1)
      var_vec = np.asarray(self.gpsurrogate.variance(Xchk)).reshape(-1)

      # check
      tol = 1e-8
      ok_mu = (mu_vec >= mu_L - tol) & (mu_vec <= mu_U + tol)
      if not np.all(ok_mu):
        bad = np.where(~ok_mu)[0].tolist()
        print(f"[μ] bounds violated at indices {bad}: "
          f"min={mu_vec.min():.6g}, max={mu_vec.max():.6g}, "
          f"bounds=({mu_L:.6g},{mu_U:.6g})")
      ok_var = (var_vec <= var_U + tol)
      if not np.all(ok_var):
        bad = np.where(~ok_var)[0].tolist()
        print(f"[σ] bounds violated at indices {bad}: "
          f"min={var_vec.min():.6g}, max={var_vec.max():.6g}, "
          f"bounds=(0,{var_U:.6g})")
    if self.acquisition_type == "LCB":
      lcb_U = self.rs_lcb(mu_L, np.sqrt(var_U))
      return lcb_U
    elif self.acquisition_type == "EI":
      ei_U = self.rs_ei(mu_L, np.sqrt(var_U))
      return ei_U
    else:
      raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)
  def _prune_queue(self, queue, gub, eps):
    """Keep only nodes that can beat current GUB within tolerance; then re-heapify."""
    # queue items are (L, counter, node)
    pruned = [(L, c, n) for (L, c, n) in queue if L < gub - eps]
    heapq.heapify(pruned)
    return pruned
  def optimize(self):
    opt = self.optimize(self.gpsurrogate.xlimits[:,0], self.gpsurrogate.xlimits[:,1])
    lopt = opt[0]
    uopt = opt[1]
    midpoint_opt = np.mean(np.array([lopt, uopt]), axis=0)
    return midpoint_opt   
  def optimize(self, l_init, u_init):
    """
    Branch & Bound minimization with tolerance stopping.
    Core logic only: correct heap order, pruning on GUB tightening,
    single global stop, diameter continue, consistent per-node prune.
    """
    print("=== Starting Branch & Bound Optimization (Minimization) ===")
    print(f"Initial bounds: l = {l_init}, u = {u_init}")
    print(f"Number of points: {self.x.shape[0]}, Dim = {self.x.shape[1]}")

    # Root bounds
    aq_L_val = self.compute_acq_lower_bound(l_init, u_init)
    aq_U_val = self.compute_acq_upper_bound(l_init, u_init)
    print(f"\nInitial acquisition lower bound: {aq_L_val}")
    print(f"Initial acquisition upper bound: {aq_U_val}")

    # Init root + heap ordered by aq_L
    root = BnBNode(l_init.astype(float), u_init.astype(float), aq_L_val, aq_U_val)
    
    # --- HEAP STORES TUPLES: (L, counter, node) ---
    self._ctr = getattr(self, "_ctr", count())
    queue = [(root.aq_L, next(self._ctr), root)]
    heapq.heapify(queue)

    # Global best (minimization GUB)
    self.best_val = aq_U_val
    self.best_l, self.best_u = l_init.copy(), u_init.copy()

    diameters = [float(np.max(u_init - l_init))]
    self.total_nodes = 1
    iteration = 0

    while queue:
      iteration += 1
      # pop the node with smallest L
      L_top, _, node = heapq.heappop(queue)

      # sanity: popped L must equal node.aq_L and be <= any remaining L
      assert abs(L_top - node.aq_L) <= 1e-12
      if queue:
        min_rest = min(L for (L,_,_) in queue)
        assert L_top <= min_rest + 1e-12, f"Heap not ordered by L (popped {L_top}, min_rest {min_rest})"

      # Update GUB
      # smallest upper bound
      if node.aq_U < self.best_val:
        self.best_val = node.aq_U
        self.best_l, self.best_u = node.l, node.u
        queue = self._prune_queue(queue, self.best_val, self.epsilon_gap)

      print(f"\n--- Iteration {iteration} ---")
      print(f"Node bounds: l={node.l}, u={node.u}")
      print(f"Node acquisition bounds: L={node.aq_L}, U={node.aq_U}")
      print(f"Current best feasible value (GUB): {self.best_val}")

      # Global stop: GUB - GLB <= eps  (GLB == node.aq_L == L_top)
      if self.best_val - node.aq_L <= self.epsilon_gap:
        print(f"STOP: GUB - Node.L = {self.best_val - node.aq_L} <= {self.epsilon_gap}")
        break

      # Diameter stop (node-local)
      node_diam = float(np.max(node.u - node.l))
      if node_diam <= self.epsilon_diam:
        print(f"Skip: Node diameter {node_diam} <= {self.epsilon_diam}")
        continue

      # Per-node prune (consistent with stop rule)
      if node.aq_L >= self.best_val - self.epsilon_gap:
        print("Pruned: Node cannot improve best within tolerance.")
        continue

      # --- Branch ---
      for l_child, u_child in self._branch(node.l, node.u):
        print(f"  Branching to child: l={l_child}, u={u_child}")

        aq_L_r = self.compute_acq_lower_bound(l_child, u_child)
        aq_U_r = self.compute_acq_upper_bound(l_child, u_child)
        print(f"  Child acquisition bounds: L={aq_L_r}, U={aq_U_r}")

        self.total_nodes += 1
        diameters.append(float(np.max(u_child - l_child)))

        # Update GUB from child and prune if improved
        if aq_U_r < self.best_val:
          self.best_val = aq_U_r
          self.best_l, self.best_u = l_child, u_child
          queue = self._prune_queue(queue, self.best_val, self.epsilon_gap)

        # Child-level prune (same tolerance)
        if aq_L_r >= self.best_val - self.epsilon_gap:
          print("  Child pruned: L ≥ GUB - eps.")
          continue

        # PUSH AS TUPLE
        child = BnBNode(l_child, u_child, aq_L_r, aq_U_r)
        heapq.heappush(queue, (child.aq_L, next(self._ctr), child))

      if not queue:
        print("\nSTOP: Queue empty, no better nodes remain.")
        break

      # Optional visibility
      phi_GLB = min(L for (L,_,_) in queue)
      print(f"Queue size: {len(queue)} | GLB={phi_GLB} | GUB={self.best_val} | Gap={self.best_val - phi_GLB}")

    self.final_gap = self.best_val - min([L for (L,_,_) in queue], default=self.best_val)
    self.final_diameter = min(diameters) if diameters else float('inf')

    print("\n=== Optimization Finished ===")
    print(f"Total nodes explored: {self.total_nodes}")
    print(f"Best bounds: l={self.best_l}, u={self.best_u}")
    print(f"Best feasible acquisition value (GUB): {self.best_val}")
    print(f"Final gap: {self.final_gap}, final diameter: {self.final_diameter}")

    return self.best_l, self.best_u, self.best_val
