import numpy as np
from numpy.random import uniform
import cvxpy as cp
import heapq
from scipy import linalg
from scipy.stats import qmc
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
import math


# define the variance upper bound problem
# these problems are in the form
# \max_{z} 1/2 z^T A z + b^T z + c
# s.t. kl <= C z <= ku
# where A is negative semi-definite
# here the evaluate function is flipped negative in order that we can use ipopt minimization
class variance_U_problem:
  def __init__(self, A, b, c, C):
    self.A = A
    self.b = b
    self.c = c
    self.C = C 
  # objective 1/2 x^T A x + b^T x + c
  def evaluate(self, z):
    return -1. * (np.inner(z, 0.5 * self.A.dot(z) + self.b) + self.c)
  def evaluate_grad(self, z):
    return -1. * (self.A.dot(z) + self.b)
  def constraint(self, z):
    return self.C.dot(z)
  def constraintJacobian(self, z):
    return self.C[:,:]

	    

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
    self.ntrain = self.x.shape[0]

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
    assert np.allclose(self.theta, self.theta[0]), "for now, we assume thetas are equal"   
 
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



    





    #variance_bound_problem = variance_U_problem(self.A_obj, self.b_obj, self.c_obj, self.C) 

    

  def distance_bounds(self, l, u):
    # normalize the box
    l_c = self._normalize(l).ravel()
    u_c = self._normalize(u).ravel()

    Xc  = self.Xc                # (nt, d)

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
    self.dmin = dmin
    self.dmax = dmax

    # construct the matrix of constraints that is specific to B k >= 0
    n_train = Xc.shape[0]
    B = []
    if spec == "pow_exp":
      # power-exponential: k = exp(-sum_j θ_j |dx_j|^p)
      p = getattr(self, "p", 2.0)
      s_min = (th * (dmin ** p)).sum(axis=1)
      s_max = (th * (dmax ** p)).sum(axis=1)
      kU = np.exp(-s_min)                                       # max on box
      kL = np.exp(-s_max)                                  # min on box
      idx = 0
      B = []
      for i in range(n_train):
        if p != 2.0 and p != 1.0:
          break
        for j in range(i):
          if p == 1.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            max_arg = th[0] * np.linalg.norm(dxij, 2)
            Kij_u = np.exp(max_arg)
            Kij_l = np.exp(-1.0 * max_arg)

            if Kij_u < 1.e4 and Kij_u > 1.e-4:
              constraint_row_u = np.zeros(n_train)
              # -ki + Kij_u * kj >= 0
              constraint_row_u[i] = -1. # -ki
              constraint_row_u[j] = Kij_u  
              B.append(constraint_row_u)
            # lower bound Kij_l * kj <= ki ==> ki - Kij_l >= 0
            if Kij_l < 1.e4 and Kij_l > 1.e-4:
              constraint_row_l = np.zeros(n_train)
              constraint_row_l[i] = 1.
              constraint_row_l[j] = -1. * Kij_l
              B.append(constraint_row_l)
            
          if p == 2.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            boxdxij = np.array([l_c * dxij, u_c * dxij])
            chimax = 2. * sum(np.max(boxdxij, axis = 0))
            chimin = 2. * sum(np.min(boxdxij, axis = 0))
            Kij_u = np.exp(th[0] * (chimax + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
            Kij_l = np.exp(th[0] * (chimin + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
            # upper bound ki <= Kij_u * kj ==> Kij_u * kj - ki >= 0
            if Kij_u < 1.e4 and Kij_u > 1.e-4:
              constraint_row_u = np.zeros(n_train)
              constraint_row_u[i] = -1.
              constraint_row_u[j] = 1. * Kij_u
              B.append(constraint_row_u)
            # lower bound Kij_l * kj <= ki ==> ki - Kij_l >= 0
            if Kij_l < 1.e4 and Kij_l > 1.e-4:
              constraint_row_l = np.zeros(n_train)
              constraint_row_l[i] = 1.
              constraint_row_l[j] = -1. * Kij_l
              B.append(constraint_row_l)
            idx += 1
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
    B = np.array(B)
    return kL, kU, B
  # determine if the 
  def ker_bound_violation(self, l, u, kOpt):
    # normalize the box
    l_c = self._normalize(l).ravel()
    u_c = self._normalize(u).ravel()

    Xc  = self.Xc                # (nt, d)
    th  = self.theta.ravel()     # (d,)
    spec = self.kernel_spec

    # per-point, per-dimension distance extremes (normalized space)
    dmin = np.maximum(0.0, np.maximum(l_c - Xc, Xc - u_c))        # (nt,d)
    dmax = np.maximum(np.abs(l_c - Xc), np.abs(u_c - Xc))         # (nt,d)
    n_train = Xc.shape[0]
    assert spec == "pow_exp", "expecting that you are using power exponential kernel"
    if spec == "pow_exp":
      # power-exponential: k = exp(-sum_j θ_j |dx_j|^p)
      p = getattr(self, "p", 2.0)
      assert p == 1.0 or p == 2.0, "expecting that you are using p = 1.0 or p = 2.0"      
      for i in range(n_train):
        if p != 2.0 and p != 1.0:
          break
        for j in range(i):
          if p == 1.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            max_arg = th[0] * np.linalg.norm(dxij, 2)
            Kij_u = np.exp(max_arg)
            Kij_l = np.exp(-1.0 * max_arg)
          if p == 2.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            boxdxij = np.array([l_c * dxij, u_c * dxij])
            chimax = 2. * sum(np.max(boxdxij, axis = 0))
            chimin = 2. * sum(np.min(boxdxij, axis = 0))
            Kij_u = np.exp(th[0] * (chimax + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
            Kij_l = np.exp(th[0] * (chimin + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
          #if p == 1.0 or p == 2.0:
          #  #if not (Kij_l <= kOpt[i] / kOpt[j] <= Kij_u):
          #  #  print("yes constraint violation")
          #  #else:
          #  #  print("no constraint violation")
          #  #print("Kl_ij = {0:1.8e}, ki / kj = {1:1.8e}, Ku_ij = {2:1.8e}, i = {3:d}, j = {4:d}".format(
          #  #         Kij_l, kOpt[i] / kOpt[j], Kij_u, i, j))
    return
  

  def mu_bounds(self, kL, kU):
    # compute in normalized y-space
    B = getattr(self, "B", np.array([]))
    if len(B) == 0:
      lo = np.where(self.gamma >= 0.0, kL, kU)
      hi = np.where(self.gamma >= 0.0, kU, kL)
      mu_L_n = self.beta0 + float(np.dot(self.gamma, lo))  # normalized
      mu_U_n = self.beta0 + float(np.dot(self.gamma, hi))  # normalized
    else:
      # min gamma^T k, where kL <= k <= kU and Bcon k >= 0
      kvar = cp.Variable(len(self.gamma))
      mu_obj = self.gamma.T @ kvar + self.beta0
      cons = [kvar >= kL, kvar <= kU, B @ kvar >= 0]
      mu_U_n = cp.Problem(cp.Maximize(mu_obj), cons).solve(solver="OSQP",verbose=self.verbose_cvx_solver, eps_abs=1.e-14, eps_rel=1.e-10)
      mu_L_n = cp.Problem(cp.Minimize(mu_obj), cons).solve(solver="OSQP",verbose=self.verbose_cvx_solver, eps_abs=1.e-14, eps_rel=1.e-10)
      assert np.isfinite(mu_U_n), "convex optimizer (max mean) did not converge"
      assert np.isfinite(mu_L_n), "convex optimizer (min mean) did not converge"

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
    B = getattr(self, "B", np.array([]))
    if len(B) == 0:
      cons = [self.C @ self.z >= kL, self.C @ self.z <= kU]
    else:
      Bcon = B.dot(self.C)
      cons = [self.C @ self.z >= kL, self.C @ self.z <= kU, Bcon @ self.z >= 0]

    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP",verbose=self.verbose_cvx_solver, eps_abs=1.e-14, eps_rel=1.e-10)
    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U



class BnBAlgorithm(BnBAlgorithmBase):
  def __init__(self, acqf, options = {}, BOit=0, saveData=False):
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    super().__init__(x = self.gpsurrogate.training_x, y = self.gpsurrogate.training_y)
    if not (isinstance(self.acqf, LCBacquisition) or isinstance(self.acqf, EIacquisition)):
      raise NotImplementedError("Unrecognized acquisition function type")
    if not (isinstance(self.acqf, LCBacquisition)):
      raise NotImplementedError("Only supprting LCB right now")
    self.sync_from_smt()
    
    # Stopping criteria parameters (default)    
    self.epsilon_gap = 1e-3
    self.epsilon_diam = 1e-2
    self.epsilon_prune = 1.e-14
    self.max_bnbiter = 2000
    self.nodes_per_batch = 1
    self.max_bnbtime = 12 * 60 # 12 minutes
    self.BOit = BOit
    self.saveData = saveData
    self.pure_BBS = False  # pure BBS search or hybrid BBS/BFS search
    self.sync_mode = False # synchronous or asynchronous evaluations
    self.verbose_cvx_solver = False # verbose convex optimizer solves
    self.opt_mode = 2

    self.acqf_UB_solver = "SLSQP"


    # Set options form command 
    self.epsilon_gap = options.get('epsilon_gap', self.epsilon_gap)
    self.epsilon_diam = options.get('epsilon_diam', self.epsilon_diam)
    self.epsilon_prune = options.get('epsilon_prune', self.epsilon_prune)
    self.max_bnbiter = options.get('max_iter', self.max_bnbiter)
    self.max_bnbtime = options.get('max_bnbtime', self.max_bnbtime)
    self.nodes_per_batch = options.get('nodes_per_batch', self.nodes_per_batch)
    self.acqf_UB_solver = options.get('acqf_ub_solver', self.acqf_UB_solver)
    self.pure_BBS = options.get('pure_BBS', self.pure_BBS)
    self.sync_mode =  options.get('sync_mode', self.sync_mode)
    self.verbose_cvx_solver = options.get('verbose_cvx_solver', self.verbose_cvx_solver)
    self.opt_mode = options.get('opt_mode', self.opt_mode)
    assert self.opt_mode in [0, 1, 2, 3, 4], "unknown opt_mode"
    assert self.acqf_UB_solver in ["SLSQP", "IPOPT", "MINEVAL"], "invalid acqf ub solver"

    if is_running_with_mpi():
      num_available_workers = MPI.COMM_WORLD.Get_size() - 1
      if num_available_workers > 1:
        # roughly evenly split workers for use in bbs and bfs evaluators
        if self.pure_BBS:
          num_bbs_workers = num_available_workers
          num_bfs_workers = 1
        else:
          num_bbs_workers = np.ceil(num_available_workers * 3 / 4).astype(int)
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
    
    # improved optimization problem to determine LCB lower bound
    ntrain = len(self.gamma)
    self.b_obj2 = np.zeros(ntrain + 1)
    if isinstance(self.acqf, LCBacquisition):
      self.b_obj2[-1] = -1.0 * self.acqf.beta
    else:
      self.b_obj2[-1] = -3.0 # -\beta
    self.b_obj2[:ntrain] = self.y_std * np.dot(self.gamma, self.C)
    self.c_obj2 = self.y_mean + self.y_std * self.beta0
    self.C2 = np.zeros((ntrain, ntrain+1))
    self.C2[:,:ntrain] = self.C[:,:]
    self.A_constraint2 = np.zeros((ntrain + 1, ntrain+1))
    ## regularize A_obj
    #U, s, Vh = np.linalg.svd(self.A_obj)
    #sreg = [min(si, -1.e-15) for si in s]
    #Areg = U @ np.diag(sreg) @ Vh


    #self.A_constraint2[:ntrain, :ntrain] = self.sigma2 * Areg #self.A_obj
    self.A_constraint2[:ntrain, :ntrain] = self.sigma2 * self.A_obj
    self.A_constraint2[ntrain, ntrain] = -2.
    self.b_constraint2 = np.zeros(ntrain + 1)
    self.b_constraint2[:ntrain] = self.sigma2 * self.b_obj[:, 0]
    self.c_constraint2 = self.sigma2 * self.c_obj
    self.en1 = np.zeros(ntrain+1)
    self.en1[ntrain] = 1.
    self.X = cp.Variable(ntrain+1) # (z, s), z = C * k
    self.obj2 = self.b_obj2.T @ self.X + self.c_obj2
    self.cons2 = 0.5 * cp.quad_form(self.X, self.A_constraint2) + self.b_constraint2 @ self.X + self.c_constraint2

  # For minimization, we find a feasible function value as the upper bound on the minimum value of the acquisition function.
  def compute_acqf_upper_bound(self, l, u):
    # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
    # Compute the kernel bounds with given x
    kL, kU, _ = self.ker_bounds(l, u)
    # Compute the mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    return self.acqf.evaluate_meansig2(np.atleast_1d(mu_U), np.atleast_1d(var_L))[0]
  # For minimization, we compute the lower bound explicitly using the acquisition function over mu, sigma.
  def compute_acqf_lower_bound(self, l, u):
    # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
    # Compute the kernel bounds with given x
    kL, kU, _= self.ker_bounds(l, u)
    # Compute the mean bounds
    mu_L, mu_U = self.mu_bounds(kL, kU)
    var_L,var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
    return self.acqf.evaluate_meansig2(np.atleast_1d(mu_L), np.atleast_1d(var_U))[0]
  def compute_acqf_bounds(self, l, u):
    # kernel bounds
    kL, kU, B = self.ker_bounds(l, u)
    self.B = B

    assert self.kernel_spec == "pow_exp", "not supporting other GP kernel types"
    assert self.p == 1.0 or self.p == 2.0, "not supporting p not equal to 1 or 2"
    if isinstance(self.acqf, LCBacquisition):
      if self.p == 1.0 or self.p == 2.0:
        # opt_mode = 0 (previous baseline w ratio constraints)
        # opt_mode = 1 (Convex relaxation w no ratio constraints on k and no affine constraints)
        # opt_mode = 2 (Most recent relaxation w ratio constraints and affine constraints
        opt_mode = self.opt_mode
     
        assert opt_mode in [0, 1, 2, 3, 4], "opt mode can only be 0, 1, or 2"
        # obtain acqf lower-bound 
        # B k >= 0, k = C2 z
        cons = [self.C2 @ self.X >= kL, self.C2 @ self.X <= kU, self.cons2 >= 0, self.en1 @ self.X >= 0]
        # TODO: enable additional constraints when problem dim > 1
        if len(B) > 0 and self.x.shape[1] == 1: # one or more kernel coupling constraints
          B2 = B.dot(self.C2)
          if opt_mode == 0 or opt_mode == 2 or opt_mode == 3:
            cons.append(B2 @ self.X >= 0)
        if opt_mode != 0:
          xvar = cp.Variable(self.x.shape[1])
          cons.append(l <= xvar)
          cons.append(xvar <= u)
          ntrain = self.x.shape[0]
          rhovar = cp.Variable(ntrain) 
          dmin = np.maximum(0.0, np.maximum((l - self.x) / self.X_scale, (self.x - u)/ self.X_scale))        # (nt,d)
          dmax = np.maximum(np.abs((l - self.x) / self.X_scale), np.abs((u - self.x) / self.X_scale))         # (nt,d)
          rhomin = (dmin**self.p).sum(axis=1)
          rhomax = (dmax**self.p).sum(axis=1) 
          #np.linalg.norm(dmin, axis=1)**self.p
          #rhomax = #np.linalg.norm(dmax, axis=1)**self.p
          #print("l = ", l)
          #print("u = ", u)
          #print("training pts = ", self.x)
          #print("dmin = ", dmin)
          #print("dmax = ", dmax)
          #print("rhomin = ", rhomin)
          #print("rhomax = ", rhomax)
          assert len(rhomin) == ntrain
          th  = self.theta.ravel()     # (d,)
          #print("theta = ", th)
          #print("opt_mode = ", opt_mode)
          #print("X_scale = ", self.X_scale)
          # --- constraints ---
          # || x - x_i || <= rho_i
          # k_i => exp(-th * \rho_i / xscale)
          for i in range(ntrain):
            cons.append(cp.atoms.norm((xvar-self.x[i]) / self.X_scale, p = self.p) ** self.p <= rhovar[i]) # || x - x_i|| <= \rho_i
            cons.append((self.C2 @ self.X)[i] >= cp.atoms.exp( -th[0] * rhovar[i]))# / (np.linalg.norm(self.X_scale, ord=self.p)**self.p))) # TODO: what to do with theta for dim(x) > 1? 
          kMax = np.exp(-th[0] * rhomin) #/ (np.linalg.norm(self.X_scale,ord=self.p)**self.p))
          kMin = np.exp(-th[0] * rhomax) #/ (np.linalg.norm(self.X_scale,ord=self.p)**self.p))
          # --- constraints --- 
          # k_i <= secant of k_i(x) over kMin, kMax
          cons.append(self.C2 @ self.X <= kMax + cp.atoms.multiply((kMin - kMax) / (rhomax - rhomin),  (rhovar  - rhomin)))
          # \rho_i = max{ \rho_i }
          cons.append(rhovar <= rhomax)
        
          # --- constraints ---
          # affine constraints on \rho_i via reverse triangle inequality
          if self.p == 1.0:
            if opt_mode == 2 or opt_mode == 4:
              for i in range(ntrain):
                for j in range(i+1, ntrain):
                  dxij_norm = np.linalg.norm((self.x[i] - self.x[j]) / self.X_scale, ord=self.p)**self.p
                  cons.append(rhovar[i] - rhovar[j] <= dxij_norm)
                  cons.append(rhovar[i] - rhovar[j] >= -1.0 * dxij_norm)
          # --- constraints ---
          # convex quadratic constraint on \rho_{i}
          #ncoupling = 2
          #for i in range(ntrain):
          #  for j in range(i):
          #    idxs = [i, j]
          #    q = np.sum(self.x[idxs], axis=0)
          #    Jl = 0.5 * ntrain * l **2 - q * l
          #    Ju = 0.5 * ntrain * u **2 - q * u
          #    Js = np.array([Jl, Ju])
          #    max_idx_set = np.argmax(Js, axis=0).flatten()
          #    xstar = np.array([l, u])[max_idx_set]
          #    R = 0.5 * ntrain * np.inner(xstar, xstar) - np.inner(q, xstar) + 0.5 * sum([np.inner(self.x[j], self.x[j]) for j in idxs])
          #    Acoupling = np.zeros((ntrain, ntrain))
          #    for idx in idxs:
          #      Acoupling[idx, idx] = 1.0
          #    #cons.append(0.5 * cp.quad_form(rhovar, Acoupling) <= R)
      opt_tol = 1.e-12
      try: 
        acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(verbose=False, tol_gap_abs=opt_tol)
        pass
      except Exception as e:
        print(f"An unexpected error occured: {e}")
        # solve again but verbosely
        acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(verbose=True, tol_gap_abs=opt_tol * 1.e8)
        #assert False, "convex optimizer did not converge"
      if not np.isfinite(acqf_L):
        acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(verbose=True, tol_gap_abs=opt_tol * 1.e8)
        if not np.isfinite(acqf_L):
          print("checking eigenvalues...")
          eigs = np.linalg.eigvalsh(self.A_obj)
          print("max eig = ", max(eigs))
          print("min eig ", min(eigs))

      assert np.isfinite(acqf_L), "convex optimizer did not converge"
      # check if rhos violate (42)
      if opt_mode != 0 and self.p == 1.0:
        for i in range(ntrain):
          for j in range(ntrain):
            drhoij = abs(rhovar.value[i] - rhovar.value[j])
            dxij   = np.linalg.norm((self.x[i] - self.x[j]) / self.X_scale, ord=self.p) 
            if drhoij > dxij + opt_tol * 10.:
              print("violation of constraint (42)")
              print("|rho_{0:d} - rho_{1:d}| = {2:1.8e}".format(i, j, drhoij))
              print("||x_{0:d} - x_{1:d}|| = {2:1.8e}".format(i, j, dxij))
      # check if k violates the ratio constraints
      kopt = self.C2 @ self.X.value
      # deprecated for now!!!
      #ker_bound_violation = self.ker_bound_violation(l, u, kopt)
      
      #if ker_bound_violation:
      #  print("kernel bound violation")
      #else:
      #  print("no kernel bound violation")
      
      #print("-------l = ", l, "u = ", u, " --------")
      #print("xstar = ", xvar.value)
      #for i in range(ntrain):
      #  print("||x - x_i|| = {0:1.2e}, rho_i = {1:1.2e}, rhomin_i = {2:1.2e}, rhomax_i = {3:1.2e}".format(np.linalg.norm(xvar.value - self.x[i]), rhovar.value[i], rhomin[i], rhomax[i]))
      #  print("k(rho_i) (convex relaxation underestimate) = {1:1.2e}, k(rho_i) = {0:1.2e}, k_i (convex relaxation overestimate) = {2:1.2e}".format((self.C2 @ self.X.value)[i], np.exp(-th[0] * rhovar.value[i] / self.X_scale[0]), kMax[i] + (kMin[i] - kMax[i]) / (rhomax[i] - rhomin[i]) * (rhovar.value[i] - rhomin[i])))
    else:
      # mean bounds
      mu_L, mu_U = self.mu_bounds(kL, kU)
      var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
      # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
      mu  = np.array([mu_L, mu_U])
      var = np.array([var_U, var_L])
      acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
      acqf_L = acqf_bounds[0]

    acqf_solve_success = False 
    if not self.acqf_UB_solver == "MINEVAL": # local gradient-based optimization method
      constraints = []
      #box_bounds = [[l[i], u[i]] for i in range(len(l))]
      box_bounds = np.array([l, u]).T
      acqf_callback = {'obj' : self.acqf.scalar_evaluate}
      if self.acqf.has_gradient:
        acqf_callback['grad'] = self.acqf.scalar_eval_g
      opt_evaluator = Evaluator()
      if self.acqf_UB_solver == "IPOPT": 
        opt_solver_options = {'max_iter' : 100, 'tol' : 1.e-5, 'honor_original_bounds' : 'yes', 'print_level' : 2, 'sb' : 'yes'}
      else: #SLSQP
        opt_solver_options = {'maxiter' : 100, 'tol' : 1.e-5}
      acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
      alpha = 0.05 + 0.9 * np.random.rand(len(u)) # rand numbers in [0.05, 0.95)
      x0 = [alpha * l + (1. - alpha) * u]
      opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
      assert (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)), f"acqf minimizer not within bounds"
      msg = opt_sol[3]
      acqf_solve_success = opt_sol[2]
      if not acqf_solve_success:
        print(self.acqf_UB_solver + " did not converge on BOX: ", l, u, "... trying again with more verbosity and at another initial point")
        print(self.acqf_UB_solver + " message: ", msg)
        if self.acqf_UB_solver == "IPOPT":
          opt_solver_options = {'max_iter' : 200, 'tol' : 1.e-5, 'honor_original_bounds' : 'yes', 'print_level' : 3, 'sb' : 'yes'}
        else: # SLSQP
          opt_solver_options = {'maxiter' : 200, 'tol' : 1.e-5, 'disp' : True}
        acqf_minimizer = minimizer_wrapper(acqf_callback, opt_solver, box_bounds, constraints, opt_solver_options)
        alpha = 0.05 + 0.9 * np.random.rand(len(u)) # rand numbers in [0.05, 0.95)
        x0 = [alpha * l + (1. - alpha) * u]
        opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
        acqf_solve_success = opt_sol[2]
        if not acqf_solve_success:
          print(self.acqf_UB_solver + "failed a second time. Will y take a the minimum of a small number of acqf function evaluations")
      if acqf_solve_success:
        acqf_U = self.acqf.evaluate(np.atleast_2d(opt_sol[0])).flatten()[0]
    # evaluate the acquisition over a skeleton of the box
    # and choose the smallest value as the upper bound
    # of the minimum over the box
    if (not acqf_solve_success) or (self.acqf_UB_solver == "MINEVAL"):
      s_per_dim = 3
      n_points = s_per_dim ** self.gpsurrogate.ndim
      x_points = np.zeros((n_points, self.gpsurrogate.ndim))
      for i in range(n_points):
        for j in range(self.gpsurrogate.ndim):
          x_points[i, j] = l[j] + (u[j] - l[j]) / (s_per_dim - 1.) * float(int(i / s_per_dim**j) % s_per_dim)
      acqf_eval = self.acqf.evaluate(x_points)
      acqf_U = min(acqf_eval.flatten())
    #assert acqf_bounds[0] <= acqf_U, "acqf_L > acqf_U"
    #return acqf_bounds[0], acqf_U
    return acqf_L, acqf_U
  def _prune_queue(self, queue, lub, eps):
    """Keep only nodes whose lower-bound is not greater or equal least upper-bound + eps; then re-heapify."""
    # queue items are (L, counter, node)
    pruned_queue = [(L, c, n) for (L, c, n) in queue if L < lub + eps]
    pruned_nodes = [n for (L, c, n) in queue if L >= lub + eps] # for now return the nodes that are pruned for analysis
    heapq.heapify(pruned_queue)
    return pruned_queue, pruned_nodes
  def _prune_node_list(self, node_list, lub, eps):
    """Keep only nodes whose lower-bound is not greater or equal least upper-bound + eps."""
    pruned_node_list = [node for node in node_list if node.aq_L < lub + eps]
    pruned_nodes = [node for node in node_list if node.aq_L >= lub + eps]
    return pruned_node_list, pruned_nodes
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
    
    all_prunednodes = []
    total_prunedvol = 0.

    
    # stopping criterion should be on the total maximum number of branched nodes
    self.num_branches = 0

    initial_gap = self.best_node.aq_U - self.best_node.aq_L
    initial_vol = math.prod(self.best_node.u - self.best_node.l)

    gap_history = [initial_gap]
    prunedvol_history = [0.]
    pruningratio_history = [1.]
    branch_history = [1]

    initial_diam = self.best_node.diam
    smallest_diam = initial_diam

    max_bbs_node_size = 0
    max_bfs_node_size = 0
    start_time = time.time()
    while self.num_branches < self.max_bnbiter: # iteration limit
      
      # -- retrieve submitted tasks -- 
      # asynchronously retrieve results from Evaluator that have been processed
      if self.sync_mode:
        self.bbsevaluator.sync()
      bbschildren = self.bbsevaluator.retrieve_results()

      # not all children are return, hence children is a ragged array
      # need to flatten this ragged list
      bbschildren = [item for sublist in bbschildren for item in sublist]

      if self.sync_mode:
        self.bfsevaluator.sync()
      bfschildren = self.bfsevaluator.retrieve_results()
      bfschildren = [item for sublist in bfschildren for item in sublist]

      children = bbschildren + bfschildren # join child lists
      if len(children) == 0:
        assert True, "trivial check"
        #if len(self.queue) == 0 and len(all_bfsnodes) == 0:
        #  assert self.bbsevaluator.num_submitted_tasks() + self.bfsevaluator.num_submitted_tasks() > 0, "node lists empty and Evaluators have no tasks submitted"
      else:
        self.num_branches += len(children)
        branch_history.append(self.num_branches)
        print(f"elapsed time: {time.time() - start_time}")
        print(f"evaluators returned {len(children)} children")
        # update best_node via children
        updated_best_node = False
        for child in children:
          assert child.aq_U >= child.aq_L, "ERROR: child upper bound < child lower bound"
          if child.aq_U <= self.LUB:
            self.best_node = child
            self.LUB = self.best_node.aq_U
            updated_best_node = True
        gap_history.append(self.best_node.aq_U - self.best_node.aq_L)
        if not updated_best_node:
          print("best node not updated")
        else:
          print("best node updated")
        
        # pre-prune
        children_lower_bounds = [child.aq_L for child in children]

        # now move pruned children to data structs for (potential) future evaluation
        children_lower_bounds = [child.aq_L for child in children]
        # sort the children in order of increasing acqf lower-bounds
        args = np.argsort(children_lower_bounds)
        children = [children[arg] for arg in args]
        
        # update smallest diameter
        child_diams = np.array([child.diam for child in children])
        smallest_diam = min(smallest_diam, min(child_diams))

        # sort children into bbs and bfs lists
        for child in children:
          if self.pure_BBS or len(self.queue) < 10 * self.num_bbs_workers:
            heapq.heappush(self.queue, (child.aq_L, next(self._ctr), child))
          else:
            all_bfsnodes.append(child) #TODO: prepend... according to number of workers
        max_bbs_node_size = max(max_bbs_node_size, len(self.queue))
        max_bfs_node_size = max(max_bfs_node_size, len(all_bfsnodes))
        
        # reprune
        self.queue, pruned_bbsnodes = self._prune_queue(self.queue, self.LUB, self.epsilon_prune)
        all_bfsnodes, pruned_bfsnodes = self._prune_node_list(all_bfsnodes, self.LUB, self.epsilon_prune)
        pruned_nodes = pruned_bbsnodes + pruned_bfsnodes
        for node in pruned_nodes:
          all_prunednodes.append(node)
          total_prunedvol += math.prod(node.u - node.l)
        prunedvol_history.append(total_prunedvol)


        if len(pruningratio_history) == 1:
          pruningratio = 1.
        else:
          pruningratio = len(all_prunednodes) / (len(all_prunednodes) + len(self.queue) + len(all_bfsnodes) + self.bbsevaluator.num_submitted_tasks() + self.bfsevaluator.num_submitted_tasks())
        pruningratio_history.append(pruningratio)

        # BnB opt progress report 
        gap = self.best_node.aq_U - self.best_node.aq_L
        print(f"\n--- Total number branches  {self.num_branches} ---")
        print(f"Corners of best node region: l={self.best_node.l}, u={self.best_node.u}")
        print(f"Node acquisition bounds: L={self.best_node.aq_L}, U={self.best_node.aq_U}")
        print(f"gap = {gap}")
        #print(f"Current best feasible value (LUB): {self.LUB}")
        print(f"total pruned vol: {total_prunedvol}")
        print(f"domain vol: {initial_vol}")
        print(f"pruning ratio: {pruningratio}")
        print(f"smallest node diam: {smallest_diam}")
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
        if np.linalg.norm(self.best_node.l - self.best_node.u, np.inf) < self.epsilon_diam:
          print("diameter of best node sufficiently small")
          break
      
      
      if time.time() - start_time > self.max_bnbtime: # time limit
        print("maximum time has elapsed")
        break
      # -- submit new tasks --


      # if the number of submitted jobs is too large then wait for some jobs to be processed
      #if self.bbsevaluator.num_submitted_tasks() + self.bfsevaluator.num_submitted_tasks() > 10 * (self.num_bbs_workers + self.num_bfs_workers):
      # collect nodes to be branched on in list structure
      # only submit additional tasks if there aren't too many in the Evaluators queue
      if self.sync_mode:
        num_bbs_tasks_to_submit = len(self.queue)
      else:
        num_bbs_tasks_to_submit = 10 * self.num_bbs_workers - self.bbsevaluator.num_submitted_tasks()
      if num_bbs_tasks_to_submit > 0:
        bbsnodes = []
        for i in range(num_bbs_tasks_to_submit):
          if (not self.queue):
            break # no more nodes available to send to evaluator for branching/bound computations
          _, _, node = heapq.heappop(self.queue)
          bbsnodes.append(node)

        # parallel branching and upper/lower bound node compuatations
        brancher = branching_wrapper(self.acqf, LUB = self.LUB, epsilon_prune=self.epsilon_prune, acqf_UB_solver = self.acqf_UB_solver)
        bbsnodes = np.array(bbsnodes)
        if len(bbsnodes) > 0:
          self.bbsevaluator.submit_tasks(brancher.callback, bbsnodes)
      
      # only submit additional tasks if there aren't too many in the Evaluators queue
      if self.sync_mode:
        num_bfs_tasks_to_submit = len(all_bfsnodes)
      else:
        num_bfs_tasks_to_submit = 10 * self.num_bfs_workers - self.bfsevaluator.num_submitted_tasks()
      if num_bfs_tasks_to_submit > 0:
        bfsnodes  = []
        for i in range(num_bfs_tasks_to_submit):
          if len(all_bfsnodes) == 0: 
            break # no more nodes available to send to evaluator for branching/bound computations
          node = all_bfsnodes.pop(0)
          bfsnodes.append(node)
        bfsnodes = np.array(bfsnodes)
        if len(bfsnodes) > 0:
          self.bfsevaluator.submit_tasks(brancher.callback, bfsnodes)


    self.all_nonpruned_nodes = all_bfsnodes + [n for (L, c, n) in self.queue]
    self.all_prunednodes = all_prunednodes
    self.prunedvol_history = prunedvol_history
    self.pruningratio_history = pruningratio_history
    self.gap_history = gap_history
    self.branch_history = branch_history
    if self.saveData:
      np.savetxt("branch_history_BOit"+str(self.BOit)+".dat", self.branch_history)
      np.savetxt("gap_history_BOit"+str(self.BOit)+".dat", self.gap_history)
      np.savetxt("prunedvol_history_BOit"+str(self.BOit)+".dat", self.prunedvol_history)
      np.savetxt("pruningratio_history_BOit"+str(self.BOit)+".dat", self.pruningratio_history)
      np.savetxt("pruned_nodes_ls_BOit"+str(self.BOit)+".dat", np.array([node.l for node in all_prunednodes]))
      np.savetxt("pruned_nodes_us_BOit"+str(self.BOit)+".dat", np.array([node.u for node in all_prunednodes]))
      np.savetxt("pruned_nodes_aqU_BOit"+str(self.BOit)+".dat", np.array([node.aq_U for node in all_prunednodes]))
      np.savetxt("pruned_nodes_aqL_BOit"+str(self.BOit)+".dat", np.array([node.aq_L for node in all_prunednodes]))
      np.savetxt("nonpruned_nodes_ls_BOit"+str(self.BOit)+".dat", np.array([node.l for node in self.all_nonpruned_nodes]))
      np.savetxt("nonpruned_nodes_us_BOit"+str(self.BOit)+".dat", np.array([node.u for node in self.all_nonpruned_nodes]))
      np.savetxt("nonpruned_nodes_aqU_BOit"+str(self.BOit)+".dat", np.array([node.aq_U for node in self.all_nonpruned_nodes]))
      np.savetxt("nonpruned_nodes_aqL_BOit"+str(self.BOit)+".dat", np.array([node.aq_L for node in self.all_nonpruned_nodes]))
      np.savetxt("kriging_weights_BOit"+str(self.BOit)+".dat", self.gamma) 

    ## TODO: sync step and prune
    ##       get final data
    ## grab any remaining tasks hanging in the evaluators
    #self.bbsevaluator.sync()
    #bbschildren = self.bbsevaluator.retrieve_results()

    ## not all children are return, hence children is a ragged array
    ## need to flatten this ragged list
    #bbschildren = [item for sublist in bbschildren for item in sublist]

    #self.bfsevaluator.sync()
    #bfschildren = self.bfsevaluator.retrieve_results()
    #bfschildren = [item for sublist in bfschildren for item in sublist]

    #children = bbschildren + bfschildren # join child lists
    # TODO: determine if any child nodes have a better LUB
    #       prune
    #       update pruned_vol
    #       if ndim == 2 plot pruned node region
    #                    color pruned node region via plot filling
    #                    non-pruned points plotted on top
    #       include another dimension independent measure of spread 
    # then embed in a BO loop 
   
    max_nodes_for_min_diam = int(1 + 2 * ((2. / smallest_diam) ** len(self.best_node.l) - 1.))

    print("\n=== Optimization Finished ===")
    print(f"Total number of branches: {self.num_branches}")
    print(f"Max BBS node list size: {max_bbs_node_size}")
    print(f"Max BFS node list size: {max_bfs_node_size}")
    print(f"Best bounds: l={self.best_node.l}, u={self.best_node.u}")
    print(f"total pruned vol: {total_prunedvol}")
    print(f"domain vol: {initial_vol}")
    print(f"smallest node diam: {smallest_diam}")
    print(f"max nodes to reach smallest node diam: {max_nodes_for_min_diam}")
    print(f"Best feasible acquisition value (LUB): {self.LUB}")
    print(f"Initial gap: {initial_gap}")
    print(f"Final gap: {gap}")
    print(f"Total elapsed time: {time.time() - start_time}")

    return self.best_node.l, self.best_node.u, self.LUB


class branching_wrapper:
  def __init__(self, acqf, LUB=np.inf, epsilon_prune=1.e-14, acqf_UB_solver="SLSQP"):
    self.LUB = LUB # least upper bound
    self.epsilon_prune = epsilon_prune
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    self.x = self.gpsurrogate.training_x
    self.y = self.gpsurrogate.training_y
    self.acqf_UB_solver = acqf_UB_solver
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
    self.b_obj2 = np.zeros(ntrain + 1)
    if isinstance(self.acqf, LCBacquisition):
      self.b_obj2[-1] = -1.0 * self.acqf.beta
    else:
      self.b_obj2[-1] = -3.0 # -\beta
    self.b_obj2[:ntrain] = self.y_std * np.dot(self.gamma, self.C)
    self.c_obj2 = self.y_mean + self.y_std * self.beta0
    self.C2 = np.zeros((ntrain, ntrain+1))
    self.C2[:,:ntrain] = self.C[:,:]
    self.A_constraint2 = np.zeros((ntrain + 1, ntrain+1))
    # regularize A_obj
    U, s, Vh = np.linalg.svd(self.A_obj)
    sreg = [min(si, -1.e-15) for si in s]
    Areg = U @ np.diag(sreg) @ Vh


    #self.A_constraint2[:ntrain, :ntrain] = self.sigma2 * Areg #self.A_obj
    self.A_constraint2[:ntrain, :ntrain] = self.sigma2 * self.A_obj
    self.A_constraint2[ntrain, ntrain] = -2.
    self.b_constraint2 = np.zeros(ntrain + 1)
    self.b_constraint2[:ntrain] = self.sigma2 * self.b_obj[:, 0]
    self.c_constraint2 = self.sigma2 * self.c_obj
    self.en1 = np.zeros(ntrain+1)
    self.en1[ntrain] = 1.
    self.X = cp.Variable(ntrain+1) # (z, s), z = C * k
    self.obj2 = self.b_obj2.T @ self.X + self.c_obj2
    self.cons2 = 0.5 * cp.quad_form(self.X, self.A_constraint2) + self.b_constraint2 @ self.X + self.c_constraint2
  
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
      # construct the matrix of constraints that is specific to B k >= 0
      n_train = Xc.shape[0]
      # construct a matrix of constraints that is specific to squared exponential.
      idx = 0
      B = []
      for i in range(n_train):
        if p != 2.0 and p != 1.0:
          break
        for j in range(i):
          if p == 1.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            max_arg = th[0] * np.linalg.norm(dxij, 2)
            Kij_u = np.exp(max_arg)
            Kij_l = np.exp(-1.0 * max_arg)
            if Kij_u < 1.e4 and Kij_u > 1.e-4:
              constraint_row_u = np.zeros(n_train)
              constraint_row_u[i] = -1.
              constraint_row_u[j] = 1. * Kij_u
              B.append(constraint_row_u)
              #B[2 * idx , i] = -1.
              #B[2 * idx , j] = 1. * Kij_u
            # lower bound Kij_l * kj <= ki ==> ki - Kij_l >= 0
            if Kij_l < 1.e4 and Kij_l > 1.e-4:
              constraint_row_l = np.zeros(n_train)
              constraint_row_l[i] = 1.
              constraint_row_l[j] = -1. * Kij_l
              B.append(constraint_row_l)
            
          if p == 2.0:
            xi = Xc[i]
            xj = Xc[j]
            dxij = xi - xj
            boxdxij = np.array([l_c * dxij, u_c * dxij])
            chimax = 2. * sum(np.max(boxdxij, axis = 0))
            chimin = 2. * sum(np.min(boxdxij, axis = 0))
            Kij_u = np.exp(th[0] * (chimax + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
            Kij_l = np.exp(th[0] * (chimin + np.linalg.norm(xj, 2) ** 2. - np.linalg.norm(xi, 2) ** 2.))
            # upper bound ki <= Kij_u * kj ==> Kij_u * kj - ki >= 0
            if Kij_u < 1.e4 and Kij_u > 1.e-4:
              constraint_row_u = np.zeros(n_train)
              constraint_row_u[i] = -1.
              constraint_row_u[j] = 1. * Kij_u
              B.append(constraint_row_u)
            # lower bound Kij_l * kj <= ki ==> ki - Kij_l >= 0
            if Kij_l < 1.e4 and Kij_l > 1.e-4:
              constraint_row_l = np.zeros(n_train)
              constraint_row_l[i] = 1.
              constraint_row_l[j] = -1. * Kij_l
              B.append(constraint_row_l)
            idx += 1

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
    return kL, kU, np.array(B)
  
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
    assert False, "not testing this now"
    kL, kU, B = self.ker_bounds(l, u)
    self.B = B 
    # obtain acqf lower-bound 
    # B k >= 0, k = C2 z
    if isinstance(self.acqf, LCBacquisition):
      if len(B) > 0: # one or more kernel coupling constraints
        B2 = B.dot(self.C2)
        cons = [self.C2 @ self.X >= kL, self.C2 @ self.X <= kU, self.cons2 >= 0, self.en1 @ self.X >= 0, B2 @ self.X >= 0]
      else: # no kernel coupling constraints
        cons = [self.C2 @ self.X >= kL, self.C2 @ self.X <= kU, self.cons2 >= 0, self.en1 @ self.X >= 0]
      acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(verbose=self.verbose_cvx_solver)
      assert np.isfinite(acqf_L), "convex optimizer did not converge"
    else:
      # mean bounds
      mu_L, mu_U = self.mu_bounds(kL, kU)
      var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
      # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
      mu  = np.array([mu_L, mu_U])
      var = np.array([var_U, var_L])
      acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
      acqf_L = acqf_bounds[0]
    
    acqf_solve_success = False 
    if not self.acqf_UB_solver == "MINEVAL": # local gradient-based optimization method
      constraints = []
      #box_bounds = [[l[i], u[i]] for i in range(len(l))]
      box_bounds = np.array([l, u]).T
      acqf_callback = {'obj' : self.acqf.scalar_evaluate}
      if self.acqf.has_gradient:
        acqf_callback['grad'] = self.acqf.scalar_eval_g
      opt_evaluator = Evaluator()
      if self.acqf_UB_solver == "IPOPT": 
        opt_solver_options = {'max_iter' : 100, 'tol' : 1.e-5, 'honor_original_bounds' : 'yes', 'print_level' : 2, 'sb' : 'yes'}
      else: #SLSQP
        opt_solver_options = {'maxiter' : 100, 'tol' : 1.e-5}
      acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
      alpha = 0.05 + 0.9 * np.random.rand(len(u)) # rand numbers in [0.05, 0.95)
      x0 = [alpha * l + (1. - alpha) * u]
      opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
      assert (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)), f"acqf minimizer not within bounds"
      msg = opt_sol[3]
      acqf_solve_success = opt_sol[2]
      if not acqf_solve_success:
        print(self.acqf_UB_solver + " did not converge on BOX: ", l, u, "... trying again with more verbosity and at another initial point")
        print(self.acqf_UB_solver + " message: ", msg)
        if self.acqf_UB_solver == "IPOPT":
          opt_solver_options = {'max_iter' : 200, 'tol' : 1.e-5, 'honor_original_bounds' : 'yes', 'print_level' : 3, 'sb' : 'yes'}
        else: # SLSQP
          opt_solver_options = {'maxiter' : 200, 'tol' : 1.e-5, 'disp' : True}
        acqf_minimizer = minimizer_wrapper(acqf_callback, opt_solver, box_bounds, constraints, opt_solver_options)
        alpha = 0.05 + 0.9 * np.random.rand(len(u)) # rand numbers in [0.05, 0.95)
        x0 = [alpha * l + (1. - alpha) * u]
        opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
        acqf_solve_success = opt_sol[2]
        if not acqf_solve_success:
          print(self.acqf_UB_solver + "failed a second time. Will y take a the minimum of a small number of acqf function evaluations")
      if acqf_solve_success:
        acqf_U = self.acqf.evaluate(np.atleast_2d(opt_sol[0])).flatten()[0]
    # evaluate the acquisition over a skeleton of the box
    # and choose the smallest value as the upper bound
    # of the minimum over the box
    if (not acqf_solve_success) or (self.acqf_UB_solver == "MINEVAL"):
      s_per_dim = 3
      n_points = s_per_dim ** self.gpsurrogate.ndim
      x_points = np.zeros((n_points, self.gpsurrogate.ndim))
      for i in range(n_points):
        for j in range(self.gpsurrogate.ndim):
          x_points[i, j] = l[j] + (u[j] - l[j]) / (s_per_dim - 1.) * float(int(i / s_per_dim**j) % s_per_dim)
      acqf_eval = self.acqf.evaluate(x_points)
      acqf_U = min(acqf_eval.flatten())
    #if self.acqf_UB_ipopt:
    #  opt_solver = "IPOPT"
    #  opt_solver_options = {'max_iter' : 1000, 'tol' : 1.e-5, 'honor_original_bounds' : 'yes', 'print_level' : 2, 'sb' : 'yes'}
    #  constraints = []
    #  box_bounds = [[l[i], u[i]] for i in range(len(l))]
    #  acqf_callback = {'obj' : self.acqf.scalar_evaluate}
    #  if self.acqf.has_gradient:
    #    acqf_callback['grad'] = self.acqf.scalar_eval_g
    #  
    #  opt_evaluator = Evaluator()
    #  acqf_minimizer = minimizer_wrapper(acqf_callback, opt_solver, box_bounds, constraints, opt_solver_options)

    #  x0 = [[(l[i] + u[i]) / 2. for i in range(len(u))]]
    #  opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
    #  assert (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)), f"acqf minimizer not within bounds"
    #  assert opt_sol[2], "optimizer did not converge"
    #  acqf_U = self.acqf.evaluate(np.atleast_2d(opt_sol[0])).flatten()[0]
    #else:
    #  s_per_dim = 3
    #  n_points = s_per_dim ** self.gpsurrogate.ndim
    #  x_points = np.zeros((n_points, self.gpsurrogate.ndim))
    #  for i in range(n_points):
    #    for j in range(self.gpsurrogate.ndim):
    #      x_points[i, j] = l[j] + (u[j] - l[j]) / (s_per_dim - 1.) * float(int(i / s_per_dim**j) % s_per_dim)
    #  acqf_eval = self.acqf.evaluate(x_points)
    #  acqf_U = min(acqf_eval.flatten())

    assert acqf_L <= acqf_U, "error: computed acquisition function bounds: acqf_U < acqf_L"
    return acqf_L, acqf_U
  def callback(self, nodes):
    output = []
    for node in nodes.flatten():
      for child_l, child_u in branch(node.l, node.u):
        acqf_L, acqf_U = self.compute_acqf_bounds(child_l, child_u)
        child = BnBNode(child_l, child_u, acqf_L, acqf_U)
        output.append(child)
    return [output]
