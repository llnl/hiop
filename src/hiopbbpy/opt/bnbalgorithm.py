import numpy as np
from numpy.random import uniform
import cvxpy as cp
import heapq
from scipy import linalg
from scipy.stats import qmc
from .acquisition import EIacquisition, LCBacquisition
from ..utils.util import Evaluator, MPIEvaluator, Logger
from .bnb_utils import * 
from .opt_utils import minimizer_wrapper, fit_common_se_point_from_ratios
from .async_bnb import BnBNode, BranchResult, RestartResult, CloseReason, LeafState, initialize_async_search, run_async_search
from itertools import count
try:
  from mpi4py import MPI
except ImportError:
  print("unable to import mpi4py")

import time
import math
import warnings
import sys

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

def dist_to_corner(l, u, x):
  box = np.array([l, u])
  return np.linalg.norm(np.min(np.abs(box - x), axis=0))

def inside_box(l, u, xk):
  return np.all(xk >= l) and np.all(xk <= u) # and not a corner node

def branch(l, u, X = None, corner_tol = 1.e-6):
  # Force to float to avoid truncation issues
  l = l.astype(float)
  u = u.astype(float)

  # Pick the dimension with largest length
  d = np.argmax(u - l)
  midpt = 0.5 * (l + u)

  # If the midpoint is the same as one bound (degenerate split), return nothing
  # shouldn't this issue have been caught?
  if np.isclose(midpt[d], l[d]) or np.isclose(midpt[d], u[d]):
    return []

  # Generate child boxes
  l1, u1 = l.copy(), u.copy()
  l2, u2 = l.copy(), u.copy()
  
  # Split the largest axis
  # along along midpoint of said axis
  if X is None:
    u1[d] = midpt[d]
    l2[d] = midpt[d]
  else:
    # training pts (X) that are internal to box defined by l, u
    internal_training_pts = []
    for x in X:
      if inside_box(l, u, x) and dist_to_corner(l, u, x) > corner_tol:
        internal_training_pts.append(x)
    if len(internal_training_pts) > 0:
      minpt_idx = np.argmin(np.linalg.norm(internal_training_pts - midpt, axis=1))
      xsplit = internal_training_pts[minpt_idx]
      # closer to l or u?
      if np.linalg.norm(xsplit - l, np.inf) >= np.linalg.norm(xsplit - u, np.inf):
        d = np.argmax(np.abs(xsplit -l))
      else:
        d = np.argmax(np.abs(xsplit - u))
    else:
      xsplit = midpt
    u1[d] = xsplit[d]
    l2[d] = xsplit[d]
  return [(l1, u1), (l2, u2)]

def gradient_branch(l, u, acqf, gradient_floor=0.05):
    l, u = np.asarray(l,dtype=float),np.asarray(u,dtype=float);
    w = u-l;
    #w = np.ones(l.shape, dtype=float)
    m = .5*(l+u)
    g = np.abs(np.asarray(acqf.eval_g(np.atleast_2d(m)),dtype=float).reshape(-1));
    ok = g.size==w.size and np.all(np.isfinite(g)) and np.max(g)>0.
    s = w*(gradient_floor+(1.-gradient_floor)*g/np.max(g)) if ok else w
    d = int(np.argmax(s))
    l0, u0, l1, u1 = l.copy(), u.copy(), l.copy(), u.copy()

    u0[d] = m[d]
    l1[d] = m[d]
    #g = np.asarray(acqf.eval_g(np.atleast_2d(m)),dtype=float).reshape(-1);
    #print("--- g ", g, " || ", d, " | ", l0, " ", u0, " | ", l1, " ", u1) 
    return [(l0, u0), (l1, u1)]

def minmax_expsec_branch(l, u, bnb, weights=None):
  l,u = np.asarray(l,float),np.asarray(u,float);
  n = l.size
  off,scale = np.asarray(bnb.X_offset).ravel(),np.asarray(bnb.X_scale).ravel()
  lc,uc = (l-off)/scale,(u-off)/scale
  X = np.asarray(bnb.Xc,float).reshape(-1,n)
  th = np.broadcast_to(np.asarray(bnb.theta,float).ravel(),(n,))
  spec = bnb.kernel_spec
  if spec not in ("pow_exp","matern32","matern52"):
    raise ValueError(f"Unsupported kernel {spec}")
  def q(d,j):
    if spec=="pow_exp": return -th[j]*d**bnb.p
    t = np.sqrt(3. if spec=="matern32" else 5.)*th[j]*d
    return np.log1p(t if spec=="matern32" else t+t*t/3.)-t
  def gap(a,b):
    d = np.maximum(b-a,0.);
    h = np.divide(-np.expm1(-d),d,out=np.ones_like(d),where=d>1.e-4)
    return np.where(d>1.e-4,np.exp(b)*(1.-h+h*np.log(h)),np.exp((a+b)/2.)*d*d/8.)
  dn = np.maximum(0.,np.maximum(lc-X,X-uc))
  df = np.maximum(abs(lc-X),abs(uc-X))
  L = np.column_stack([q(df[:,j],j) for j in range(n)])
  U = np.column_stack([q(dn[:,j],j) for j in range(n)])
  la,lb = L.sum(1),U.sum(1)
  w = np.ones(X.shape[0]) if weights is None else np.asarray(weights,float).ravel()
  if w.size != X.shape[0] or np.any(w<0) or not np.all(np.isfinite(w)):
    print("wwwweights:", weights, flush=True)
    raise ValueError("Invalid exp-sec weights")
  candidates=[]
  for j in range(n):
    m = .5*(l[j]+u[j])
    if not l[j]<m<u[j]: continue
    z = (m-off[j])/scale[j]
    scores = []
    for a,b in ((lc[j],z),(z,uc[j])):
      dn = np.maximum(0.,np.maximum(a-X[:,j],X[:,j]-b))
      df = np.maximum(abs(a-X[:,j]),abs(b-X[:,j]))
      scores.append(w@gap(la-L[:,j]+q(df,j), lb-U[:,j]+q(dn,j)))
    l1,u1,l2,u2 = l.copy(),u.copy(),l.copy(),u.copy()
    u1[j] = m
    l2[j] = m
    candidates.append((max(scores),-(uc[j]-lc[j]),j,[(l1,u1),(l2,u2)]))
  return min(candidates,key=lambda x:x[:3])[3] if candidates else []


def lcb_gradient_at_single_reference(owner, x_ref, sigma_floor=None):
  """Return the full LCB gradient with respect to k at one feasible kernel vector k_ref=k(x_ref)."""
  if not isinstance(owner.acqf, LCBacquisition):
    raise TypeError("lcb_gradient_at_single_reference requires an LCB acquisition")
  x_ref = np.asarray(x_ref, dtype=float).ravel()
  training_x = np.asarray(owner.x, dtype=float)
  x_scale = np.asarray(owner.X_scale, dtype=float).ravel()
  theta = np.asarray(owner.theta, dtype=float).ravel()
  dx = np.abs((x_ref[None, :] - training_x) / x_scale[None, :])
  if owner.kernel_spec == "pow_exp":
    k_ref = np.exp(-np.sum(theta[None, :] * dx**float(owner.p), axis=1))
  elif owner.kernel_spec == "matern32":
    t = np.sqrt(3.0) * theta[None, :] * dx;
    k_ref = np.prod((1.0 + t) * np.exp(-t), axis=1)
  elif owner.kernel_spec == "matern52":
    t = np.sqrt(5.0) * theta[None, :] * dx;
    k_ref = np.prod((1.0 + t + t**2 / 3.0) * np.exp(-t), axis=1)
  else:
    raise ValueError(f"Unsupported kernel specification '{owner.kernel_spec}'")
  gamma = np.asarray(owner.gamma, dtype=float).ravel()
  C = np.asarray(owner.C, dtype=float)
  A = np.asarray(owner.A_obj, dtype=float)
  A = 0.5 * (A + A.T)
  b = np.asarray(owner.b_obj, dtype=float).ravel()
  c = float(np.asarray(owner.c_obj, dtype=float).reshape(()))
  sigma2 = float(owner.sigma2)
  z_ref = np.linalg.solve(C, k_ref)
  variance_ref_raw = sigma2 * (0.5 * np.dot(z_ref, A @ z_ref) + np.dot(b, z_ref) + c)
  variance_scale = max(1.0, abs(sigma2 * c), abs(variance_ref_raw))
  if variance_ref_raw < -1.0e-10 * variance_scale:
    raise RuntimeError(f"Reference kernel produces negative variance {variance_ref_raw:.16e}")
  sigma_ref = float(np.sqrt(max(0.0, variance_ref_raw)))
  sigma_floor = 1.0e-8 * max(1.0, np.sqrt(abs(sigma2))) if sigma_floor is None else float(sigma_floor)
  grad_variance_z = sigma2 * (A @ z_ref + b)
  grad_variance_k = np.linalg.solve(C.T, grad_variance_z)
  grad_mean_k = float(owner.y_std) * gamma
  grad_variance_lcb_k = -float(owner.acqf.beta) * grad_variance_k / (2.0 * max(sigma_ref, sigma_floor))
  grad_lcb_k = grad_mean_k + grad_variance_lcb_k
  if not np.all(np.isfinite(grad_lcb_k)):
    raise RuntimeError("Nonfinite LCB kernel gradient")
  return grad_lcb_k, k_ref, sigma_ref, grad_mean_k, grad_variance_lcb_k

def compute_sigma_ir_bounds(l, u, theta, x_scale, x_i, x_r, kernel_spec, p=2.0):
  """
    Compute outward-rounded bounds
  
        sigma_ir_min <= lambda_i(x) + lambda_r(x) <= sigma_ir_max

    over x in [l, u].

    For every supported kernel, the coordinate function

        phi_ij(t) + phi_rj(t)

    is concave and symmetric about (x_i[j] + x_r[j]) / 2. Hence:

      * its minimum over [l_j, u_j] occurs at an endpoint;
      * its maximum occurs at the projected midpoint.

    Parameters are in the same convention as bnbalgorithm.py:
    l, u, x_i, and x_r are in original coordinates, while x_scale
    converts distances to SMT's normalized coordinates.

    Supported kernel_spec values
    ----------------------------
    "pow_exp"   with p in {1, 2}
    "squar_exp"
    "abs_exp"
    "matern12"
    "matern32"
    "matern52"

    Returns
    -------
    sigma_ir_min, sigma_ir_max : float
        Bounds denoted by underline{Sigma}_{ir} and
        overline{Sigma}_{ir} in the manuscript.
  """
  l = np.asarray(l, dtype=float).ravel()
  u = np.asarray(u, dtype=float).ravel()
  theta = np.asarray(theta, dtype=float).ravel()
  x_scale = np.asarray(x_scale, dtype=float).ravel()
  x_i = np.asarray(x_i, dtype=float).ravel()
  x_r = np.asarray(x_r, dtype=float).ravel()
  
  arrays = (u, theta, x_scale, x_i, x_r)
  if any(a.shape != l.shape for a in arrays):
    raise ValueError("l, u, theta, x_scale, x_i, and x_r must have the same shape")
  
  if not all(np.all(np.isfinite(a)) for a in (l,) + arrays):
    raise ValueError("Nonfinite data in log-kernel-sum bound computation")
  
  if np.any(l > u):
    raise ValueError("Invalid box: some l_j > u_j")
  
  if np.any(theta < 0.0):
    raise ValueError("Kernel parameters theta must be nonnegative")
  
  if np.any(x_scale <= 0.0):
    raise ValueError("Every normalization scale must be positive")
  
  # Accept either self.kernel_spec or the original SMT corr name.
  spec = str(kernel_spec).lower()
  
  if spec == "squar_exp":
    spec = "pow_exp"
    p = 2.0
  elif spec in ("abs_exp", "matern12"):
    spec = "pow_exp"
    p = 1.0
  if spec == "pow_exp":
    p = float(p)
    if p not in (1.0, 2.0):
      raise ValueError("pow_exp is supported here only for p in {1, 2}")
  elif spec not in ("matern32", "matern52"):
    raise ValueError(f"Unsupported kernel specification '{kernel_spec}'")

  def phi_component(t, center):
    """Return the vector (phi_{nu,ij}(t_j))_j."""
    distance = np.abs((t - center) / x_scale)
    
    if spec == "pow_exp":
      return -theta * distance**p

    if spec == "matern32":
      z = np.sqrt(3.0) * theta * distance
      return np.log1p(z) - z
    
    # Matérn-5/2:
    # log(1 + z + z^2/3) - z,
    # where z = sqrt(5) theta_j |t - center| / x_scale_j.
    z = np.sqrt(5.0) * theta * distance
    return np.log1p(z + z*z / 3.0) - z
  
  # A concave one-dimensional function attains its minimum over an
  # interval at one of the two endpoints.
  sum_at_l = phi_component(l, x_i) + phi_component(l, x_r)
  sum_at_u = phi_component(u, x_i) + phi_component(u, x_r)
  
  sigma_ir_min = float(np.sum(np.minimum(sum_at_l, sum_at_u)))
  
  # The maximum is attained at the midpoint of the two centers,
  # projected onto the coordinate interval.
  projected_midpoint = np.clip(0.5 * (x_i + x_r), l, u)
  
  sigma_ir_max = float(np.sum(phi_component(projected_midpoint, x_i) +
                              phi_component(projected_midpoint, x_r)))
  
  # Slight outward rounding protects validity against accumulated
  # floating-point error. Log correlations are always nonpositive.
  magnitude = max(1.0, abs(sigma_ir_min), abs(sigma_ir_max))
  roundoff = 32.0 * np.finfo(float).eps * max(1, l.size) * magnitude
  
  sigma_ir_min -= roundoff
  sigma_ir_max = min(0.0, sigma_ir_max + roundoff)
  
  return sigma_ir_min, sigma_ir_max

def add_ratio_informed_product_constraints(cons, ki, kr, dirL, dirU, sirL, sirU):
  """
    Add linear ratio and ratio-informed product constraints.
  
    Bounds:
        dirL <= lambda_i - lambda_r <= dirU,
        sirL <= lambda_i + lambda_r <= sirU.
  
    Assumes normalized kernels:
        0 <= ki <= 1,  0 <= kr <= 1.
  
    This function calls the existing add_ratio_constraints() and
    add_product_constraints(), then adds
        exp(-dm/2) ki + exp(dm/2) kr <= 2 exp(sirU/2) cosh(dw/2),
    where
        dm = (dirL + dirU)/2, dw = (dirU - dirL)/2,

    and the lower-product tangent rows
        exp(-d0/2) ki + exp(d0/2) kr >= 2 exp(sirL/2),
    for d0 in {dirL, dm, dirU}.
  """
  dirL, dirU = float(dirL), float(dirU)
  sirL, sirU = float(sirL), float(sirU)

  values = np.asarray([dirL, dirU, sirL, sirU])
  if not np.all(np.isfinite(values)):
    raise ValueError("Finite log-ratio and log-product bounds are required")
  
  if dirL > dirU:
    raise ValueError(f"Invalid log-ratio interval [{dirL}, {dirU}]")
  if sirL > sirU:
    raise ValueError(f"Invalid log-product interval [{sirL}, {sirU}]")
  if sirL > 0.0:
    raise ValueError("Normalized kernels require lambda_i + lambda_r <= 0")

  # The universal bound ki*kr <= 1 gives sirU <= 0.
  sirU = min(sirU, 0.0)
  dm = 0.5 * dirL + 0.5 * dirU
  dw = 0.5 * (dirU - dirL)

  def normalized_coefficients(d0):
    """
    Return exp(-d0/2) and exp(d0/2), divided by
    exp(abs(d0)/2). Thus the largest coefficient is one.
    """
    small = float(np.exp(-abs(d0)))
    if d0 >= 0.0:
      return small, 1.0
    return 1.0, small

  # --------------------------------------------------------------
  # Centered ratio-informed upper-product row.
  #
  # Before scaling:
  #   exp(-dm/2) ki + exp(dm/2) kr
  #       <= 2 exp(sirU/2) cosh(dw/2).
  #
  # Divide through by exp(abs(dm)/2).
  # --------------------------------------------------------------
  coef_i, coef_r = normalized_coefficients(dm)
  
  # Stable evaluation of
  #
  # log(2 exp(sirU/2) cosh(dw/2) / exp(abs(dm)/2)).
  log_upper_rhs = 0.5 * sirU + np.logaddexp(0.5 * dw, -0.5 * dw) - 0.5 * abs(dm)
  
  # With ki,kr <= 1, the maximum scaled left-hand side is
  # coef_i + coef_r. Omit a provably redundant row.
  if log_upper_rhs < np.log(coef_i + coef_r):
    upper_rhs = float(np.exp(log_upper_rhs))

    # If the positive RHS underflows, omitting the row preserves
    # the outer-relaxation property.
    if upper_rhs > 0.0:
      # For an upper inequality, round coefficients downward
      # and the right-hand side upward.
      coef_i_upper = float(np.nextafter(coef_i, 0.0)) if coef_i > 0.0 else 0.0
      coef_r_upper = float(np.nextafter(coef_r, 0.0)) if coef_r > 0.0 else 0.0
      upper_rhs = float(np.nextafter(upper_rhs, np.inf))

      cons.append(coef_i_upper * ki + coef_r_upper * kr <= upper_rhs)

  # --------------------------------------------------------------
  # Lower-product tangent rows.
  #
  # Before scaling:
  #   exp(-d0/2) ki + exp(d0/2) kr
  #       >= 2 exp(sirL/2).
  #
  # Divide through by exp(abs(d0)/2).
  # --------------------------------------------------------------
  tangent_points = np.unique(np.asarray([dirL, dm, dirU], dtype=float))

  for d0 in tangent_points:
    coef_i, coef_r = normalized_coefficients(d0)

    log_lower_rhs = (np.log(2.0) + 0.5 * sirL - 0.5 * abs(d0))
    lower_rhs = float(np.exp(log_lower_rhs))

    # If the RHS underflows, the row has numerically reduced to
    # a nonnegativity constraint.
    if lower_rhs == 0.0:
      continue

    # For a lower inequality, round coefficients upward and the
    # right-hand side downward.
    coef_i_lower = float(np.nextafter(coef_i, np.inf))
    coef_r_lower = float(np.nextafter(coef_r, np.inf))
    lower_rhs = float(np.nextafter(lower_rhs, 0.0))

    cons.append(coef_i_lower * ki + coef_r_lower * kr >= lower_rhs)

def add_product_constraints(cons, ki, kr, sir_min, sir_max):
    """
    Add product cuts
        2 exp(sir_min / 2) <= ki + kr <= 1 + exp(sir_max).

    Here sir_min and sir_max are valid bounds on
        lambda_i + lambda_r = log(ki * kr).

    This function assumes normalized correlations:
        0 <= ki <= 1,  0 <= kr <= 1.

    """
    sir_min = float(sir_min)
    sir_max = float(sir_max)

    if not np.all(np.isfinite([sir_min, sir_max])):
      raise ValueError("Finite bounds on lambda_i + lambda_r are required")

    if sir_min > sir_max:
      raise ValueError(f"Invalid log-product interval [{sir_min}, {sir_max}]")

    if sir_min > 0.0:
      raise ValueError("A normalized kernel must have lambda_i + lambda_r <= 0")

    # The universal normalized-kernel bound provides Sigma_ir <= 0.
    sir_max = min(sir_max, 0.0)

    # If either exponential underflows, its corresponding constraint is omitted rather
    # than replacing a positive exact quantity by zero.

    
    # Lower row:
    #     ki + kr >= 2 sqrt(ki*kr)
    #             >= 2 exp(sir_min/2).
    exp_half_min = float(np.exp(0.5 * sir_min))
    if exp_half_min > 0.0:
      lower_rhs = float(np.nextafter(2.0 * exp_half_min, 0.0))
      cons.append(ki + kr >= lower_rhs)

    # Upper row:
    #     ki + kr <= 1 + ki*kr
    #             <= 1 + exp(sir_max).
    product_upper = float(np.exp(sir_max))
    if product_upper > 0.0:
      upper_rhs = float(np.nextafter(1.0 + product_upper, np.inf))
      cons.append(ki + kr <= upper_rhs)

    
def add_ratio_constraints(cons, ki, kr, lir_min, lir_max):
    """
    Add:
        exp(lir_min) * kr <= ki <= exp(lir_max) * kr

    Every generated linear row has coefficients with magnitude <= 1.
    """

    # Lower ratio inequality:
    # exp(lir_min) * kr <= ki
    if lir_min <= 0.0:
        coef = np.exp(lir_min)

        # If coef underflows to zero, omitting this inequality is safe:
        # it has numerically reduced to the already-present ki >= 0.
        if coef > 0.0:
            cons.append(ki >= coef * kr)
    else:
        # Divide by exp(lir_min):
        # kr <= exp(-lir_min) * ki
        coef = np.exp(-lir_min)

        # Do not impose kr <= 0 if this underflows.
        # Omitting the inequality preserves the outer-relaxation property.
        if coef > 0.0:
            cons.append(kr <= coef * ki)

    # Upper ratio inequality:
    # ki <= exp(lir_max) * kr
    if lir_max <= 0.0:
        coef = np.exp(lir_max)

        # Do not impose ki <= 0 if this underflows.
        if coef > 0.0:
            cons.append(ki <= coef * kr)
    else:
        # Divide by exp(lir_max):
        # exp(-lir_max) * ki <= kr
        coef = np.exp(-lir_max)

        # If this underflows, the inequality reduces numerically to kr >= 0.
        if coef > 0.0:
            cons.append(kr >= coef * ki)

def add_mccormick_ratio_constraints(
    cons, ki, kr, lam_i, lam_r, lir_min, lir_max,
    ki_min, ki_max, kr_min, kr_max, name=None,
    add_linear_ratio_rows=True,
):
    """
    Add the lifted exponential/McCormick relaxation
        d_ir = lam_i - lam_r
        q_ir = exp(d_ir)
        ki   = q_ir * kr
    over
        lir_min <= d_ir <= lir_max,
        ki_min <= ki <= ki_max,
        kr_min <= kr <= kr_max.

    The exact equality q_ir = exp(d_ir) is relaxed by its convex hull:
        exp(d_ir) <= q_ir <= sec_exp(d_ir).

    The exact product ki = q_ir * kr is relaxed by the four  McCormick inequalities.

    To improve numerical scaling, the pair is automatically reversed when doing so 
    produces a smaller upper bound on q. Thus, internally, the function may represent either
        ki = q * kr
    or the equivalent relation
        kr = q * ki.
    """
    lir_min, lir_max = float(lir_min), float(lir_max)
    ki_min, ki_max = max(0.0, float(ki_min)), float(ki_max)
    kr_min, kr_max = max(0.0, float(kr_min)), float(kr_max)

    values = np.asarray([lir_min, lir_max, ki_min, ki_max, kr_min, kr_max], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite log-ratio or kernel bounds in McCormick ratio relaxation")
    if lir_min > lir_max:
        raise ValueError(f"Invalid log-ratio interval [{lir_min}, {lir_max}]")
    if ki_min > ki_max or kr_min > kr_max:
        raise ValueError("Invalid individual kernel bounds in McCormick ratio relaxation")
    if ki_max < 0.0 or kr_max < 0.0:
        raise ValueError("Kernel upper bounds must be nonnegative")

    swapped = lir_min + lir_max > 0.0

    if not swapped:
        k_num, k_den = ki, kr
        lam_num, lam_den = lam_i, lam_r
        d_min, d_max = lir_min, lir_max
        k_num_min, k_num_max = ki_min, ki_max
        k_den_min, k_den_max = kr_min, kr_max
    else:
        k_num, k_den = kr, ki
        lam_num, lam_den = lam_r, lam_i
        d_min, d_max = -lir_max, -lir_min
        k_num_min, k_num_max = kr_min, kr_max
        k_den_min, k_den_max = ki_min, ki_max

    d_expr = lam_num - lam_den
    cons += [d_expr >= d_min, d_expr <= d_max]

    def append_normalized_linear_ratio_rows():
        if d_min <= 0.0:
            coef = float(np.exp(d_min))
            if coef > 0.0: cons.append(k_num >= coef * k_den)
        else:
            coef = float(np.exp(-d_min))
            if coef > 0.0: cons.append(k_den <= coef * k_num)

        if d_max <= 0.0:
            coef = float(np.exp(d_max))
            if coef > 0.0: cons.append(k_num <= coef * k_den)
        else:
            coef = float(np.exp(-d_max))
            if coef > 0.0: cons.append(k_den >= coef * k_num)

    width = d_max - d_min
    log_max_float = float(np.log(np.finfo(float).max))

    if d_max >= log_max_float - 4.0:
        append_normalized_linear_ratio_rows()
        return {
            "q": None, "swapped": swapped, "fallback": True, "exact": False,
            "log_bounds": (d_min, d_max), "reason": "q upper bound would overflow",
        }

    q_max_raw = float(np.exp(d_max))

    if not np.isfinite(q_max_raw) or q_max_raw < np.finfo(float).tiny:
        append_normalized_linear_ratio_rows()
        return {
            "q": None, "swapped": swapped, "fallback": True, "exact": False,
            "log_bounds": (d_min, d_max),
            "reason": "q upper bound is numerically unrepresentable",
        }

    q_min_raw = float(np.exp(d_min))
    q_min = float(np.nextafter(q_min_raw, 0.0)) if q_min_raw > 0.0 else 0.0
    q_max = float(np.nextafter(q_max_raw, np.inf))

    if width == 0.0:
        cons.append(k_num == q_max_raw * k_den)
        if add_linear_ratio_rows: append_normalized_linear_ratio_rows()

        return {
            "q": None, "swapped": swapped, "fallback": False, "exact": True,
            "log_bounds": (d_min, d_max), "q_bounds": (q_max_raw, q_max_raw),
        }

    qvar = cp.Variable(nonneg=True) if name is None else cp.Variable(nonneg=True, name=f"q_{name}")

    cons += [
        qvar >= q_min,
        qvar <= q_max,
        qvar >= cp.exp(d_expr),
    ]

    secant_slope_raw = float(q_max_raw * (-np.expm1(-width)) / width)

    if not np.isfinite(secant_slope_raw) or secant_slope_raw <= 0.0:
        append_normalized_linear_ratio_rows()
        return {
            "q": None, "swapped": swapped, "fallback": True, "exact": False,
            "log_bounds": (d_min, d_max),
            "reason": "invalid exponential secant slope",
        }

    q_min_secant = (
        float(np.nextafter(q_min_raw, np.inf))
        if q_min_raw > 0.0
        else float(np.nextafter(0.0, np.inf))
    )
    secant_slope = float(np.nextafter(secant_slope_raw, np.inf))

    cons.append(qvar <= q_min_secant + secant_slope * (d_expr - d_min))

    cons += [
        k_num >= q_min * k_den + k_den_min * qvar - q_min * k_den_min,
        k_num >= q_max * k_den + k_den_max * qvar - q_max * k_den_max,
        k_num <= q_max * k_den + k_den_min * qvar - q_max * k_den_min,
        k_num <= q_min * k_den + k_den_max * qvar - q_min * k_den_max,
    ]

    if add_linear_ratio_rows:
      append_normalized_linear_ratio_rows()

    return {
        "q": qvar, "swapped": swapped, "fallback": False, "exact": False,
        "log_bounds": (d_min, d_max), "q_bounds": (q_min, q_max),
        "numerator_bounds": (k_num_min, k_num_max),
        "denominator_bounds": (k_den_min, k_den_max),
    }

def add_mccormick_sum_product_constraints(
    cons, ki, kr, lami, lamr,
    ki_min, ki_max, kr_min, kr_max,
    sir_min, sir_max, secant_width_tol=1.0e-12,
):
    """
    Add the complementary lifted sum/product relaxation
        sir = lami + lamr, pir = exp(sir), pir = ki * kr.

    The graph pir = exp(sir) is relaxed using its convex exponential
    lower envelope and affine secant upper envelope. The bilinear
    equality pir = ki * kr is relaxed using the four McCormick
    inequalities.
    """
    ki_min, ki_max = float(ki_min), float(ki_max)
    kr_min, kr_max = float(kr_min), float(kr_max)
    sir_min, sir_max = float(sir_min), float(sir_max)

    if ki_min < 0.0 or kr_min < 0.0:
        raise ValueError("Kernel lower bounds must be nonnegative")
    if ki_min > ki_max or kr_min > kr_max:
        raise ValueError("Invalid kernel bounds")
    if not np.isfinite(sir_min) or not np.isfinite(sir_max):
        raise ValueError("Finite bounds on lambda_i + lambda_r are required")
    if sir_min > sir_max:
        raise ValueError("Invalid bounds on lambda_i + lambda_r")

    pir_min, pir_max = float(np.exp(sir_min)), float(np.exp(sir_max))

    if pir_max == 0.0: return None, None
    if not np.isfinite(pir_max):
        raise FloatingPointError("exp(sir_max) overflowed in sum/product relaxation")

    sir, pir = cp.Variable(), cp.Variable(nonneg=True)

    cons += [
        sir == lami + lamr,
        sir >= sir_min,
        sir <= sir_max,
        pir >= cp.exp(sir),
        pir >= pir_min,
        pir <= pir_max,
    ]

    width = sir_max - sir_min

    if width > secant_width_tol:
        slope = -pir_max * np.expm1(-width) / width
        cons.append(pir <= pir_min + slope * (sir - sir_min))
    else:
        cons.append(pir <= pir_max)

    cons += [
        pir >= ki_min * kr + kr_min * ki - ki_min * kr_min,
        pir >= ki_max * kr + kr_max * ki - ki_max * kr_max,
        pir <= ki_max * kr + kr_min * ki - ki_max * kr_min,
        pir <= ki_min * kr + kr_max * ki - ki_min * kr_max,
    ]

    return sir, pir
  
class BnBAlgorithmBase:
  def __init__(self, x = None, y = None):
    # Kernel info for bounds
    self.kernel_spec = None
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

    self.log = Logger()
    self.diagnostics = False
    
  def sync_from_smt(self):
    sm = self.gpsurrogate.surrogatesmt
    par = sm.optimal_par

    # --- kernel / corr selection ---
    corr = sm.options["corr"]  # e.g., 'squar_exp', 'pow_exp', 'abs_exp', 'matern12', 'matern32', 'matern52'
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
    elif corr == "matern12":
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
    
    
    ell = 1.0 / np.sqrt(2.0 * self.theta)

    print("SMT theta =", theta, flush=True)
    print("equivalent conventional ell =", ell, flush=True)
    
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

    if spec == "pow_exp":
      # power-exponential: k = exp(-sum_j θ_j |dx_j|^p)
      p = getattr(self, "p", 2.0)
      s_min = (th * (dmin ** p)).sum(axis=1)
      s_max = (th * (dmax ** p)).sum(axis=1)
      kU = np.exp(-s_min)                                  # max on box
      kL = np.exp(-s_max)                                  # min on box
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
    B = getattr(self, "B", np.array([]))
    if len(B) == 0:
      cons = [self.C @ self.z >= kL, self.C @ self.z <= kU]
    else:
      Bcon = B.dot(self.C)
      cons = [self.C @ self.z >= kL, self.C @ self.z <= kU, Bcon @ self.z >= 0]

    #TODO: update me, embed in try loop... some other strategy for when the maximization does not work!!!

    s2_U_n = cp.Problem(cp.Maximize(self.obj), cons).solve(solver="OSQP",verbose=self.verbose_cvx_solver, eps_abs=1.e-14, eps_rel=1.e-10)
    assert np.isfinite(s2_U_n), "convex optimizer did not converge"
    
    # re-scale
    s2_U = s2_U_n * self.sigma2 
    return s2_L, s2_U


# diagnostics statistics related to the "common" x to all k kernels
def stats_common_se_point(owner, l, u, xvar, wvar, lamvar) -> str:
  output = ""
  k_sol = np.asarray((owner.C2 @ owner.X).value, dtype=float).ravel()
  common = fit_common_se_point_from_ratios(owner=owner, k_values=k_sol, l=l, u=u)
  output += f" l:      {np.array2string(l, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  output += f" x_lsq:  {np.array2string(common['x_common'], max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  output += f" x_conv: {np.array2string(np.asarray(xvar.value, dtype=float).ravel(), max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  output += f" u:      {np.array2string(u, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  output += (
    f'Ratio resid: RMS {common["pair_rms"]:10.3e}  '
    f'max={common["pair_max"]:10.3e}     '
  )
  output += (
    f'Abs log-kernel resid:  RMS {common["absolute_rms"]:10.3e}  '
    f'max={common["absolute_max"]:10.3e}  '
    f'mean={common["absolute_mean"]:10.3e}  '
    f'std={common["absolute_std"]:10.3e}\n'
  )

  x_opt = np.asarray(xvar.value).ravel()
  w_opt = np.asarray(wvar.value).ravel()
  lam_opt = np.asarray(lamvar.value).ravel()
  k_opt = np.asarray((owner.C2 @ owner.X).value).ravel()

  vals = w_opt - x_opt**2
  output += (
    "w-x^2= "
    f"{np.array2string(vals, max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  )

  vals = np.log(np.maximum(k_opt, np.finfo(float).tiny)) - lam_opt
  output += (
    "log(k) - lambda: "
    f"{np.array2string(vals,  max_line_width=100000, formatter={'float_kind': lambda x: f'{x:.3e}'})}\n"
  )
  return output

def stats_lcb_relaxation_gap(owner, xvar, relaxation_value) -> str:
  """
  Decompose the feasible-at-x_conv gap

      LCB(x_conv) - relaxation_value,

  where x_conv = xvar.value and relaxation_value is the objective
  returned by the conic solver.

  The top-level decomposition is

      LCB(x_conv) - L_rel
        = [mu(x_conv) - mu(k_rel)]
        + beta * [s_rel - sigma(x_conv)]
        + [(mu(k_rel) - beta*s_rel) - L_rel].

  The variance contribution is further decomposed as

      beta * [s_rel - sigma(x_conv)]
        = beta * [s_rel - sigma_model(k_rel)]
        + beta * [sigma_model(k_rel) - sigma_exact(k_rel)]
        + beta * [sigma_exact(k_rel) - sigma(x_conv)].

  Here:
    sigma_model(k_rel)
      uses the variance quadratic actually present in the conic
      constraint after the NSD eigenvalue modification;

    sigma_exact(k_rel)
      uses the original, unmodified SMT variance quadratic.

  This diagnostic is intended for the squared-exponential relaxation:
      owner.kernel_spec == "pow_exp" and owner.p == 2.
  """
  unavailable = "LCB relaxation gap decomposition unavailable: "

  try:
    if not isinstance(owner.acqf, LCBacquisition):
      return unavailable + "acquisition is not LCB\n"

    if owner.kernel_spec != "pow_exp" or float(owner.p) != 2.0:
      return (
        unavailable
        + "diagnostic currently supports only the SE kernel\n"
      )

    if owner.X.value is None or xvar.value is None:
      return unavailable + "missing conic primal values\n"

    X_opt = np.asarray(
      owner.X.value,
      dtype=float,
    ).ravel()

    gamma = np.asarray(
      owner.gamma,
      dtype=float,
    ).ravel()

    ntrain = gamma.size

    if X_opt.size != ntrain + 1:
      return (
        unavailable
        + f"expected X to have length {ntrain + 1}, "
        + f"got {X_opt.size}\n"
      )

    # X = (z,s), with k = C z.
    z_rel = X_opt[:ntrain]
    s_rel = float(X_opt[-1])

    x_rel = np.asarray(
      xvar.value,
      dtype=float,
    ).ravel()

    k_rel = np.asarray(
      owner.C2 @ X_opt,
      dtype=float,
    ).ravel()

    # Exact SE kernel vector generated by the shared spatial point xvar.
    theta = np.asarray(
      owner.theta,
      dtype=float,
    ).ravel()

    x_scale = np.asarray(
      owner.X_scale,
      dtype=float,
    ).ravel()

    training_x = np.asarray(
      owner.x,
      dtype=float,
    )

    dx = (
      x_rel[None, :]
      - training_x
    ) / x_scale[None, :]

    k_at_x = np.exp(
      -np.sum(
        theta[None, :] * dx**2,
        axis=1,
      )
    )

    beta = float(owner.acqf.beta)

    # Relaxed mean represented by k_rel.
    mu_rel = float(
      owner.y_mean
      + owner.y_std
      * (
        owner.beta0
        + np.dot(gamma, k_rel)
      )
    )

    # Actual GP mean and variance at the feasible point x_rel.
    x_row = np.atleast_2d(x_rel)

    mu_at_x = float(
      np.asarray(
        owner.gpsurrogate.mean(x_row),
        dtype=float,
      ).reshape(-1)[0]
    )

    var_at_x_raw = float(
      np.asarray(
        owner.gpsurrogate.variance(x_row),
        dtype=float,
      ).reshape(-1)[0]
    )

    sigma_at_x = float(
      np.sqrt(max(0.0, var_at_x_raw))
    )

    # Exact, unmodified SMT variance quadratic at the relaxed k.
    #
    # owner.A_obj, owner.b_obj, and owner.c_obj define
    #
    #   variance(k_rel)
    #     = owner.sigma2 *
    #       (0.5*z^T*A_obj*z + b_obj^T*z + c_obj).
    A_obj = np.asarray(
      owner.A_obj,
      dtype=float,
    )

    b_obj = np.asarray(
      owner.b_obj,
      dtype=float,
    ).ravel()

    c_obj = float(
      np.asarray(
        owner.c_obj,
        dtype=float,
      ).reshape(())
    )

    var_exact_k_raw = float(
      owner.sigma2
      * (
        0.5 * np.dot(
          z_rel,
          A_obj @ z_rel,
        )
        + np.dot(
          b_obj,
          z_rel,
        )
        + c_obj
      )
    )

    sigma_exact_k = float(
      np.sqrt(max(0.0, var_exact_k_raw))
    )

    # Variance quadratic actually used in the conic constraint after
    # modifying A_constraint2 to be numerically negative definite.
    #
    # owner.cons2 is:
    #
    #   variance_model(k_rel) - s_rel^2.
    #
    # Hence:
    #
    #   variance_model(k_rel) = owner.cons2.value + s_rel^2.
    cons2_value = getattr(
      owner.cons2,
      "value",
      None,
    )

    if cons2_value is not None:
      var_model_k_raw = float(
        np.asarray(
          cons2_value,
          dtype=float,
        ).reshape(())
        + s_rel**2
      )
    else:
      # Fallback that evaluates the same conic quadratic directly.
      A_con = np.asarray(
        owner.A_constraint2,
        dtype=float,
      )

      b_con = np.asarray(
        owner.b_constraint2,
        dtype=float,
      ).ravel()

      c_con = float(
        np.asarray(
          owner.c_constraint2,
          dtype=float,
        ).reshape(())
      )

      var_model_k_raw = float(
        0.5 * np.dot(
          X_opt,
          A_con @ X_opt,
        )
        + np.dot(
          b_con,
          X_opt,
        )
        + c_con
        + s_rel**2
      )

    sigma_model_k = float(
      np.sqrt(max(0.0, var_model_k_raw))
    )

    relaxation_value = float(
      np.asarray(
        relaxation_value,
        dtype=float,
      ).reshape(())
    )

    # Reconstruct the conic objective directly from the primal variables.
    relax_obj_from_X = float(
      mu_rel - beta * s_rel
    )

    # Feasible LCB at xvar.value.
    lcb_at_x = float(
      mu_at_x - beta * sigma_at_x
    )

    gap_at_x = float(
      lcb_at_x - relaxation_value
    )

    # ---------------------------------------------------------------
    # Top-level decomposition
    # ---------------------------------------------------------------

    mean_part = float(
      mu_at_x - mu_rel
    )

    variance_part = float(
      beta * (s_rel - sigma_at_x)
    )

    # This is normally close to zero. It measures the discrepancy
    # between the value returned by the solver and the objective
    # reconstructed from its primal solution.
    objective_value_part = float(
      relax_obj_from_X - relaxation_value
    )

    total_closure = float(
      gap_at_x
      - (
        mean_part
        + variance_part
        + objective_value_part
      )
    )

    # ---------------------------------------------------------------
    # Variance-part decomposition
    # ---------------------------------------------------------------

    # Activity/slack of the conic variance inequality.
    # This should normally be close to zero and nonpositive:
    #
    #     s_rel <= sigma_model(k_rel).
    variance_cone_part = float(
      beta
      * (
        s_rel
        - sigma_model_k
      )
    )

    # Effect of the numerical NSD eigenvalue modification made in
    # owner.A_constraint2.
    variance_projection_part = float(
      beta
      * (
        sigma_model_k
        - sigma_exact_k
      )
    )

    # Effect of replacing the kernel vector generated by x_rel with
    # the relaxed kernel vector.
    variance_kernel_part = float(
      beta
      * (
        sigma_exact_k
        - sigma_at_x
      )
    )

    variance_closure = float(
      variance_part
      - (
        variance_cone_part
        + variance_projection_part
        + variance_kernel_part
      )
    )

    # ---------------------------------------------------------------
    # Componentwise mean sensitivity
    # ---------------------------------------------------------------

    delta_k = k_at_x - k_rel

    mean_terms = (
      owner.y_std
      * gamma
      * delta_k
    )

    mean_terms_sum = float(
      np.sum(mean_terms)
    )

    mean_terms_l1 = float(
      np.sum(np.abs(mean_terms))
    )

    # This should be close to zero if the manually formed SE kernel
    # and the SMT posterior-mean formula agree.
    mean_formula_residual = float(
      mean_part - mean_terms_sum
    )

    if mean_terms.size:
      top_mean_idx = int(
        np.argmax(np.abs(mean_terms))
      )

      top_mean_term = float(
        mean_terms[top_mean_idx]
      )

      max_mean_term = float(
        np.max(np.abs(mean_terms))
      )
    else:
      top_mean_idx = -1
      top_mean_term = 0.0
      max_mean_term = 0.0

    if mean_terms_l1 == 0.0:
      cancellation_ratio = 1.0
    elif abs(mean_terms_sum) <= np.finfo(float).tiny:
      cancellation_ratio = np.inf
    else:
      cancellation_ratio = float(
        mean_terms_l1
        / abs(mean_terms_sum)
      )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    output = (
      "LCB relaxation-gap decomposition at x_conv:\n"
    )

    output += (
      f"  L_rel={relaxation_value: .6e}  "
      f"obj(X)={relax_obj_from_X: .6e}  "
      f"LCB(x_conv)={lcb_at_x: .6e}  "
      f"gap={gap_at_x: .6e}\n"
    )

    output += (
      f"  gap parts: "
      f"mean={mean_part: .6e}  "
      f"variance={variance_part: .6e}  "
      f"obj-value={objective_value_part: .3e}  "
      f"closure={total_closure: .3e}\n"
    )

    output += (
      f"  variance parts: "
      f"cone={variance_cone_part: .6e}  "
      f"NSD-proj={variance_projection_part: .6e}  "
      f"k-decoupling={variance_kernel_part: .6e}  "
      f"closure={variance_closure: .3e}\n"
    )

    output += (
      f"  states: "
      f"mu_rel={mu_rel: .6e}  "
      f"mu(x_conv)={mu_at_x: .6e}  "
      f"s={s_rel: .6e}  "
      f"sigma_model(k)={sigma_model_k: .6e}  "
      f"sigma_exact(k)={sigma_exact_k: .6e}  "
      f"sigma(x_conv)={sigma_at_x: .6e}\n"
    )

    output += (
      f"  mean k-terms: "
      f"sum={mean_terms_sum: .6e}  "
      f"L1={mean_terms_l1: .6e}  "
      f"maxabs={max_mean_term: .6e}  "
      f"top=({top_mean_idx},{top_mean_term: .6e})  "
      f"cancel={cancellation_ratio: .3e}  "
      f"formula-resid={mean_formula_residual: .3e}\n"
    )

    output += (
      f"  kernel mismatch: "
      f"||k(x_conv)-k_rel||_inf="
      f"{np.linalg.norm(delta_k, ord=np.inf): .6e}  "
      f"||.||_2={np.linalg.norm(delta_k): .6e};  "
      f"raw variances: "
      f"model={var_model_k_raw: .6e}  "
      f"exact-k={var_exact_k_raw: .6e}  "
      f"x_conv={var_at_x_raw: .6e}\n"
    )

    return output

  except Exception as exc:
    # This function is currently called inside the conic-solver retry
    # try-block. A diagnostics failure should not cause a successful
    # conic solve to be retried with looser tolerances.
    return (
      unavailable
      + f"{type(exc).__name__}: {exc}\n"
    )

class BnBAlgorithm(BnBAlgorithmBase):
  def __init__(self, acqf, options = {}, BOit=0):
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    super().__init__(x = self.gpsurrogate.training_x, y = self.gpsurrogate.training_y)
    if not (isinstance(self.acqf, LCBacquisition) or isinstance(self.acqf, EIacquisition)):
      raise NotImplementedError("Unrecognized acquisition function type")
    self.sync_from_smt()
    
    # Stopping criteria parameters (default)    
    self.epsilon_gap = 1e-4
    self.epsilon_diam = 1e-2
    self.epsilon_rel_gap = 1.e-4
    self.min_diam = 0.125
    self.epsilon_prune = 1.e-14
    self.epsilon_node = self.epsilon_gap/100
    self.inflight_factor = 1.
    self.poll_interval = 0.01
    self.max_task_retries = 1
    self.bound_consistency_tol = 1.e-4
    self.max_bnbiter = 2000
    self.nodes_per_batch = 1
    self.max_bnbtime = 12 * 60 # 12 minutes
    self.BOit = BOit
    self.saveData = False #saveData
    self.saveDataDir = ""
    self.pure_BBS = False  # pure BBS search or hybrid BBS/BFS search
    self.synchronous = False # synchronous or asynchronous evaluations
    self.verbose_cvx_solver = False # verbose convex optimizer solves
    self.opt_mode = 3

    self.acqf_UB_solver = "SLSQP"

    # Set options form command 
    self.epsilon_gap = options.get('abs_tol', self.epsilon_gap)
    self.epsilon_node = self.epsilon_gap/100
    self.epsilon_node = options.get('node_tol', self.epsilon_node)
    self.epsilon_rel_gap = options.get('rel_tol', self.epsilon_rel_gap)
    self.epsilon_diam = options.get('epsilon_diam', self.epsilon_diam)
    self.epsilon_prune = options.get('epsilon_prune', self.epsilon_prune)
    self.max_bnbiter = options.get('max_iter', self.max_bnbiter)
    self.max_bnbtime = options.get('max_bnbtime', self.max_bnbtime)
    self.nodes_per_batch = options.get('nodes_per_batch', self.nodes_per_batch)
    self.acqf_UB_solver = options.get('acqf_ub_solver', self.acqf_UB_solver)
    self.pure_BBS = options.get('pure_BBS', self.pure_BBS)
    self.synchronous =  options.get('synchronous', self.synchronous)
    self.verbose_cvx_solver = options.get('verbose_cvx_solver', self.verbose_cvx_solver)
    self.opt_mode = options.get('opt_mode', self.opt_mode)
    self.saveDataDir = options.get('save_data_dir', self.saveDataDir)
    self.saveData = options.get('save_data', self.saveData)
    self.min_diam = options.get('min_diameter', self.min_diam)
    self.inflight_factor = options.get('inflight_factor', self.inflight_factor)
    self.poll_interval = options.get('poll_interval', self.poll_interval)
    self.max_task_retries = options.get('max_task_retries', self.max_task_retries)
    self.bound_consistency_tol = options.get('bound_consistency_tol', self.bound_consistency_tol)
    self.random_seed = options.get("random_seed", None)
    if self.random_seed is None:
      self.rng = np.random.default_rng()
    else:
      # Different deterministic stream for each BO iteration
      self.rng = np.random.default_rng(int(self.random_seed) + 1000003 * int(self.BOit))
    
    assert self.opt_mode in [0, 1, 2, 3, 4, 5, 6], "unknown opt_mode"
    assert self.acqf_UB_solver in ["SLSQP", "trust-constr", "IPOPT", "MINEVAL"], "invalid acqf ub solver"
    assert isinstance(self.saveData, bool), "save_data is not of type bool"
    assert isinstance(self.saveDataDir, str), "save_data_dir is not of type string"

    supplied_evaluator = options.get("node_evaluator", None)
    if supplied_evaluator is None:
      self.node_evaluator = MPIEvaluator(
        function_mode=False,
        executor=options.get("executor", None),
        task_name="BNB",
        profiling=False,
      )
    else:
      self.node_evaluator = supplied_evaluator

    self.diagnostics = options.get('diagnostics',  self.diagnostics)
      
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

    self.A_constraint2[:ntrain, :ntrain] = self.sigma2 * self.A_obj
    self.A_constraint2[ntrain, ntrain] = -2.
    self.b_constraint2 = np.zeros(ntrain + 1)
    self.b_constraint2[:ntrain] = self.sigma2 * self.b_obj[:, 0]
    self.c_constraint2 = self.sigma2 * self.c_obj
    self.en1 = np.zeros(ntrain+1)
    self.en1[ntrain] = 1.
    self.X = cp.Variable(ntrain+1) # (z, s), z = C * k
    self.obj2 = self.b_obj2.T @ self.X + self.c_obj2
    # ensure that the matrix is negative semi-definite
    # //!
    lam, U = np.linalg.eigh(self.A_constraint2)
    lam_neg = np.minimum(lam, -1.e-16 * np.ones(len(lam)))
    self.A_constraint2[:,:] = U.dot(np.diag(lam_neg)).dot(U.T)
    
    self.cons2 = 0.5 * cp.quad_form(self.X, self.A_constraint2) + self.b_constraint2 @ self.X + self.c_constraint2

    # set up a third objective function whose value is s wherein we will include a constraint s^2 <= \sig^2(k) 
    # also in which we will maximize s making s = max sig
    self.b_obj3 = np.zeros(ntrain + 1)
    self.c_obj3 = 0.0 # no shift necessary
    self.b_obj3[-1] = 1.0
    self.obj3 = self.b_obj3.T @ self.X + self.c_obj3


    # constants for choosing number of "k ratio" constraints
    # first constraints are determined via nearest neighbor search
    # the kernel distances will be based on the forms of the kernels
    # for SE th * ((x - x^(i)) / X_scale))^2 is used
    # for all other kenels th * |(x - x^(i)) / X_scale| is used
    # for this reason the distance metric will be
    # || \sqrt(th) / X_scale * (x^(i) - x^(r))||_2 for SE
    # and || th / X_scale * (x^(i) - x^(r)||_1 for all other kernels
    distance_mat = np.zeros((ntrain,ntrain))
    theta = self.theta.ravel()
    # only go over the upper triangle
    self.pairs_dist = []
    for i in range(ntrain):
      for j in range(i+1, ntrain):
        if self.kernel_spec == "pow_exp" and self.p == 2.0:
          self.pairs_dist.append((np.linalg.norm(np.sqrt(theta) * (self.x[i] - self.x[j]) / self.X_scale), i, j))
        else:
          self.pairs_dist.append((np.linalg.norm(theta * (self.x[i] - self.x[j]) / self.X_scale, ord=1), i, j))
    # now extract smallest-distance pairs
    ##print("pairs     :" + " ".join(f"({i:2d},{j:2d},{val:12.5e})" for val, i, j in self.pairs_dist))
    self.pairs_dist.sort(key=lambda entry: entry[0])
    ##print("pairs sort:" + " ".join(f"({i:2d},{j:2d},{val:12.5e})" for val, i, j in self.pairs_dist))
    self.c0 = 10
    npairs = min(self.c0 * ntrain, len(self.pairs_dist))
    ##print("pairs slct:" + " ".join(f"({i:2d},{j:2d},{val:12.5e})" for val, i, j in self.pairs_dist[:npairs]))
    self.nearest_neighbor_pairs = np.asarray([(i, r) for _, i, r in self.pairs_dist[:npairs]], dtype=np.int64).reshape(-1, 2)

    # repurpose pairs_dist
    pairs = self.pairs_dist
    self.pairs_dist = {(i, j): dist for dist, i, j in pairs}


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

  def LCB_LB(self, l, u, kL, kU, opt_mode=2):
    return self.convex_relaxation(l, u, kL, kU, opt_mode=opt_mode)
  def sig_UB(self, l, u, kL, kU, opt_mode=2):
    return self.convex_relaxation(l, u, kL, kU, opt_mode=opt_mode, mode=1)
  def sig_LB(self, kL, kU, l = None, u =None):
    # lower-bound
    if l is None or u is None:
      s2_L = 0.0
    else:
      # TODO: replace with local minimizer routine
      x = np.atleast_2d( (l + u) / 2.)
      S2 = self.gpsurrogate.variance(x)
      s2_L = min(S2.flatten())
    return np.sqrt(s2_L)
  
  def convex_relaxation(self, l, u, kL, kU, opt_mode=2, mode=0):
    """
       mode: 0 --> convex relaxation for minimum of LCB acquisition function
       mode: 1 --> convex relaxation for maximum of variance        
    """
    assert mode in [0, 1], "mode can only be in 0, 1"
    assert opt_mode in [0, 1, 2, 3, 4, 5, 6], "opt mode can only be 0, 1, 2, 3, 4, or 5"
    assert not (opt_mode in [0, 1, 2, 3, 4] and self.kernel_spec != "pow_exp"), "opt mode 0,1,2,3, and 4 limited to pow_exp kernel"
    # opt_mode = 0 (previous baseline w ratio constraints)
    # opt_mode = 1 (Convex relaxation w no ratio constraints on k and no affine constraints)
    # opt_mode = 2 (Most recent relaxation w ratio constraints and affine constraints)
    # opt_mode = 3 (Relaxation in w)
    # opt_mode = 4 (opt_mode 3 but with alternative to ratio constraints on k)
    cons = [self.cons2 >= 0, self.en1 @ self.X >= 0]
    diagnostics_output = ""

    # Let us keep these for now even though they are redundant
    #if opt_mode != 5 and opt_mode != 6:
    cons.append(self.C2 @ self.X >= kL)
    cons.append(self.C2 @ self.X <= kU)
    if opt_mode != 0 and opt_mode != 5 and opt_mode != 6:
      # add x optimization variable constrained to box: l <= x <= u
      xvar = cp.Variable(self.x.shape[1])
      cons.append(l <= xvar)
      cons.append(xvar <= u)
      # add rho_i = \sum_j \theta_j |(x_j - x_j^(i)) / X_scale|^p variable
      ntrain = self.x.shape[0]
      rhovar = cp.Variable(ntrain) 
      # determine bounds for rho
      dmin = np.maximum(0.0, np.maximum((l - self.x) / self.X_scale, (self.x - u)/ self.X_scale))        # (nt,d)
      dmax = np.maximum(np.abs((l - self.x) / self.X_scale), np.abs((u - self.x) / self.X_scale))         # (nt,d)
      th  = self.theta.ravel()     # (d,)
      rhomin = (th * (dmin**self.p)).sum(axis=1)
      rhomax = (th * (dmax**self.p)).sum(axis=1) 
      assert len(rhomin) == ntrain
      # \rho_i = max_{x in box} { \rho_i(x) }
      # no need to include these constraints with equality constraint rho_i = ...
      cons.append(rhovar <= rhomax) # pre w-relaxation bounds
      #cons.append(rhomin <= rhovar)    
      # --- constraints coupling k and rho ---
      # k_i => exp(-\rho_i)
      for i in range(ntrain):
        cons.append((self.C2 @ self.X)[i] >= cp.atoms.exp(-rhovar[i]))
      kMax = np.exp(-rhomin)
      kMin = np.exp(-rhomax)
      #cons.append(kMin <= self.C2 @ self.X) implied by rhovar <= rhomax
      cons.append(self.C2 @ self.X <= kMax) # pre-secant relaxation
      # k_i <= secant of k_i(x) over kMin, kMax
      # //!: I suggest we check and handle small |rhomax-rhomin| 
      cons.append(self.C2 @ self.X <= kMax + cp.atoms.multiply((kMin - kMax) / (rhomax - rhomin),  (rhovar  - rhomin)))
      # ---
      

      # \sum_j theta_j * |(x_j - x_j^(i)) / X_scale|^p <= rho_i
      # for p = 1 or 2 is convex
      if opt_mode < 3:
        for i in range(ntrain):
          # Note that this is redundant when opt_mode is 3, being implied by the eq. constraint on rhovar
          cons.append(cp.atoms.norm((xvar-self.x[i]) / ((th**(-1./self.p)) * self.X_scale), p = self.p)**self.p <= rhovar[i])
      if opt_mode >= 3:
        assert self.p == 2.0, "opt_mode 3 only supports squared exponential kernel"
        wvar = cp.Variable(self.x.shape[1])
        for i in range(self.x.shape[1]):
          cons.append(wvar[i] >= xvar[i]**2.0)
          cons.append(wvar[i] <= (l[i] + u[i]) * xvar[i] - l[i] * u[i])
        for i in range(ntrain):
          cons.append(rhovar[i] == cp.atoms.sum(cp.atoms.multiply(th / self.X_scale**self.p, wvar - 2.0 * cp.atoms.multiply(self.x[i], xvar) + self.x[i] * self.x[i]))) 


      # --- constraints ---
      # affine constraints on \rho_i via reverse triangle inequality
      # no need for additional bound constraints when we have equality constraint
      if opt_mode >= 2:
        if self.p == 1.0:
          for i in range(ntrain):
            for j in range(i+1, ntrain):
              rhoij_bound = np.linalg.norm(th**(1./self.p) * (self.x[i] - self.x[j]) / self.X_scale, ord=self.p)**self.p
              cons.append(rhovar[i] - rhovar[j] <= rhoij_bound)
              cons.append(rhovar[i] - rhovar[j] >= -1.0 * rhoij_bound)
        elif self.p == 2.0:
          if opt_mode<3:
            for i in range(ntrain):
              for j in range(i+1, ntrain):
                dxji = self.x[j] - self.x[i]
                mult = 2. * th * dxji / (self.X_scale ** 2.0)
                shift = np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                assert opt_mode!=3, " constraint is redundant when opt_mode==3, being implied by the eq. constraint on rhovar"
                cons.append(rhovar[i] - rhovar[j] == mult.T @ xvar + shift)
          elif opt_mode == 3:
            for i in range(ntrain):
              for j in range(i+1, ntrain):
                dxji = self.x[j] - self.x[i]
                lo = np.where(dxji >= 0., l, u)
                hi = np.where(dxji >= 0., u, l)
                # rhoij_lbound <= rho_i - rho_j <= rhoij_ubound
                # k_i <= k_j * exp(-1 * rhoij_lbound)
                # k_i >= k_j * exp(-1 * rhoij_ubound)
                rhoij_ubound = 2. * np.inner((hi / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                rhoij_lbound = 2. * np.inner((lo / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                kij_ubound = np.exp(-1. * rhoij_lbound)
                kij_lbound = np.exp(-1. * rhoij_ubound)
                if not ((kij_ubound <= 1.e3 and kij_ubound >= 1.e-3) and (kij_lbound <= 1.e3 and kij_lbound >= 1.e-3)):
                  continue
                # it is more important that we place bounds on k than rho
                cons.append((self.C2 @ self.X)[i] <= (self.C2 @ self.X)[j] * kij_ubound)
                cons.append((self.C2 @ self.X)[i] >= (self.C2 @ self.X)[j] * kij_lbound)
        else: # opt_mode 4
          qvar = cp.Variable(int((ntrain * (ntrain -1) ) /2))
          dvar = cp.Variable(int((ntrain * (ntrain -1) ) /2))
          k = 0
          for i in range(ntrain):
            for j in range(i+1, ntrain):
              # d = 2 x^\top \Theta (x^{(j)} - x^{(i)}) + ||x^{(i)}||_{\Theta}^2 - ||x^{(j)}||_{\Theta}^2 
              cons.append(dvar[k] == cp.atoms.sum(cp.atoms.multiply(th / self.X_scale**self.p, 2.0 * cp.atoms.multiply(xvar, self.x[j] - self.x[i]) + self.x[i] * self.x[i] - self.x[j] * self.x[j]))) 
              # q >= exp(d)
              cons.append(qvar[k] >= cp.atoms.exp(-1.0 * dvar[k]))
              # --- begin d secant constraint ---
              
              dxji = self.x[j] - self.x[i]
              lo = np.where(dxji >= 0., l, u)
              hi = np.where(dxji >= 0., u, l)
              # rhoij_lbound <= rho_i - rho_j <= rhoij_ubound
              # k_i <= k_j * exp(-1 * rhoij_lbound)
              # k_i >= k_j * exp(-1 * rhoij_ubound)
              d_ubound = 2. * np.inner((hi / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
              d_lbound = 2. * np.inner((lo / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
              q_ubound = np.exp(-1.0 * d_lbound)
              q_lbound = np.exp(-1.0 * d_ubound)       
              cons.append(qvar[k] <= q_ubound + ((q_lbound - q_ubound) / (d_ubound - d_lbound)) * (dvar[k] - d_lbound))
              # --- end d secant constraint ---

              # McCormick relaxation on product k_i = q_k * k_j
              # z = x * y
              # z >= x_l y + x * y_l - x_l * y_l
              # z >= x_u y + x * y_u - x_u * y_u
              # z <= x_u y + x * y_l - x_u * y_l
              # z <= x_l * y + x * y_u - x_l * y_u
              cons.append((self.C2 @ self.X)[i] >= q_lbound * (self.C2 @ self.X)[j] + qvar[k] * kMin[j] - q_lbound * kMin[j])
              cons.append((self.C2 @ self.X)[i] >= q_ubound * (self.C2 @ self.X)[j] + qvar[k] * kMax[j] - q_ubound * kMax[j])
              cons.append((self.C2 @ self.X)[i] <= q_ubound * (self.C2 @ self.X)[j] + qvar[k] * kMin[j] - q_ubound * kMin[j])
              cons.append((self.C2 @ self.X)[i] <= q_lbound * (self.C2 @ self.X)[j] + qvar[k] * kMax[j] - q_lbound * kMax[j])
              # ---- end McCormick relaxation on product k_i = q_k * k_j
              
              # add additional bound constraints on on d
              cons.append(dvar[k] <= d_ubound)
              cons.append(d_lbound <= dvar[k])
              k = k + 1
    elif opt_mode == 5 or opt_mode == 6:
      ntrain = self.x.shape[0]
      dimx = self.x.shape[1]
      # add x optimization variable constrained to box: l <= x <= u
      xvar = cp.Variable(dimx)
      cons.append(l <= xvar)
      cons.append(xvar <= u)
      cons.append(cp.atoms.power(cp.atoms.norm(self.X[:-1]), 2) <= 1.0) # k^T R^-1 k = z^T z <= 1
      # determine bounds for k and lambda
      dmin = np.maximum(0.0, np.maximum((l - self.x) / self.X_scale, (self.x - u)/ self.X_scale))        # (nt,d)
      dmax = np.maximum(np.abs((l - self.x) / self.X_scale), np.abs((u - self.x) / self.X_scale))         # (nt,d)
      th  = self.theta.ravel()     # (d,)
      lamvar = cp.Variable(ntrain)
      lamU = np.log(kU)
      lamL = np.log(kL)
      cons.append(lamvar >= lamL)
      cons.append(lamvar <= lamU)
      for i in range(ntrain):
        cons.append((self.C2 @ self.X)[i] >= cp.atoms.exp(lamvar[i]))
      cons.append(self.C2 @ self.X <= kL + cp.atoms.multiply((kU - kL) / (lamU - lamL),  (lamvar  - lamL)))
      etavar = cp.Variable((ntrain, dimx))
      cons.append(lamvar == cp.atoms.sum(etavar, axis=1)) # sum along column of matrix-valued \eta
      if self.kernel_spec == "pow_exp":
        assert self.p in [1.0, 2.0], "opt_mode 5 only support matern 1/2 (a.k.a. pow exp) and SE kernels"
        if self.p == 2.0:
          wvar = cp.Variable(dimx)
          for i in range(ntrain):
            for j in range(dimx):
              cons.append(etavar[i,j] == (-1.0 * th[j] / (self.X_scale[j]**self.p)) * (wvar[j] - 2. * self.x[i][j] * xvar[j] + self.x[i][j]**2))
          for j in range(dimx):
            cons.append(xvar[j]**2 <= wvar[j])
            cons.append(wvar[j] <= (l[j] + u[j]) * xvar[j] - l[j] * u[j])
        elif self.p == 1.0:
          # --- tau and alpha are ragged arrays
          taus = [[] for j in range(dimx)]
          for j in range(dimx):
            taus[j].append(l[j])
            taus[j].append(u[j])
            for i in range(ntrain):
              if self.x[i][j] < u[j] and l[j] < self.x[i][j]:
                taus[j].append(self.x[i][j])
          alphavars = [cp.Variable(len(taus[j])) for j in range(dimx)]
          for i in range(ntrain):
            for j in range(dimx):
              cons.append(etavar[i][j] == cp.atoms.sum(cp.atoms.multiply(-1.0 * th[j] / (self.X_scale[j]) * np.abs(taus[j] - self.x[i][j]), alphavars[j])))
          for j in range(dimx):
            cons.append(xvar[j] == cp.atoms.sum(cp.atoms.multiply(taus[j], alphavars[j])))
            cons.append(cp.atoms.sum(alphavars[j]) == 1.0)
            for i in range(len(taus[j])):
              cons.append(alphavars[j][i] >= 0.0)
          



      else: #matern32 or matern52
        nu = 1.5
        if self.kernel_spec != "matern32":
          nu = 2.5
        for j in range(dimx):
          component_phi = matern_phi(self.x[:,j].tolist(), th[j] / self.X_scale[j], nu)
          D_rs = component_phi.generate_alpha_beta_r(l[j], u[j])
          for k in range(len(D_rs)):
            # alpha_m xj + beta_m^T eta_(:, j) <= r_m
            cons.append(D_rs[k][0] * xvar[j] + cp.atoms.scalar_product(D_rs[k][1], etavar[:,j]) <= D_rs[k][2])

      if opt_mode == 6:
        # add constraints based on downselected nearest neighbor pairs
        # downselect on available pairs
        Ei_exp = np.zeros(ntrain)
        Ai     = np.zeros(ntrain)
        kvec = self.C2 @ self.X

        sensitivity_floor = 0.05
        x_ref = 0.5 * (np.asarray(l, dtype=float) + np.asarray(u, dtype=float))
        lcb_grad_k, k_ref, sigma_ref, mean_grad_k, variance_grad_k = lcb_gradient_at_single_reference(self, x_ref)
        abs_lcb_grad_k = np.abs(lcb_grad_k)
        normalized_lcb_sensitivity = abs_lcb_grad_k / max(float(np.max(abs_lcb_grad_k)), np.finfo(float).eps)
        for i in range(ntrain):
          #compute Ei_exp
          if lamU[i] > lamL[i]:
            # point where gap between exp and its secant is largest

            # original code misses  "- exp(lamstar)" for Ei_exp[i]
            # lamstar = np.log((np.exp(lamU[i]) - np.exp(lamL[i])) / (lamU[i] - lamL[i]))
            # Ei_exp[i] = np.exp(lamL[i]) + (np.exp(lamU[i]) - np.exp(lamL[i])) / (lamU[i] - lamL[i]) * (lamstar - lamL[i])
            exp_lamL = np.exp(lamL[i])
            lam_exp_diff = np.exp(lamU[i]) - exp_lamL
            slope = lam_exp_diff / (lamU[i] - lamL[i])
            lamstar = np.log(slope)
            sec_at_star = exp_lamL + slope*(lamstar - lamL[i])                                                          
            Ei_exp[i] = sec_at_star - slope
            if Ei_exp[i] < 0:
              raise RuntimeError("roundoff error: diff between exp and sec should be zero")            
          else:
            Ei_exp[i] = 0.
          #Ai[i] = Ei_exp[i] * (gamma_floor + (1. - gamma_floor) * np.abs(self.gamma[i]) / (np.max(np.abs(self.gamma)) + eps_gamma))
          Ai[i] = Ei_exp[i] * (sensitivity_floor + (1.0 - sensitivity_floor) * normalized_lcb_sensitivity[i])
        #print("gamma::" + " ".join(f"({i:2d}, {self.gamma[i]:12.5e})" for i in range(len(self.gamma)))) 
        #print("Ai exp-sec gap:\n" + " ".join(f"({i:2d}, {Ai[i]:12.5e})" for i in range(len(Ai))))
        pair_selection_triplets = np.array([[pair[0], pair[1], (Ai[pair[0]] + Ai[pair[1]])/self.pairs_dist[pair[0],pair[1]]] for pair in self.nearest_neighbor_pairs])
        #print(pair_selection_triplets)
        args = np.argsort(pair_selection_triplets[:,-1])[::-1]
        pair_selection_triplets[:,:] = pair_selection_triplets[args,:]
        #print("triplets Ai+Aj:\n" + " ".join(f"({int(i):2d},{int(j):2d},{val:12.5e})" for i, j, val in pair_selection_triplets))

        # now find c1 * p pairs
        c1 = 5
        ndownselect_pairs = min(len(self.nearest_neighbor_pairs), c1 * ntrain)

        #print("triplets Ai+Aj: downselect\n" + " ".join(f"({int(i):2d},{int(j):2d},{val:12.5e})" for i, j, val in pair_selection_triplets[:ndownselect_pairs]))
        
        for pair in pair_selection_triplets[:ndownselect_pairs]:
          i_idx = int(pair[0])
          r_idx = int(pair[1])
          lir_min = 0.
          lir_max = 0.
          if self.kernel_spec == "pow_exp":
            dphi_ijr = lambda t,j: -th[j] / (self.X_scale[j]**self.p) * (np.abs(t - self.x[i_idx][j])**self.p - np.abs(t - self.x[r_idx][j])**self.p)
            lijr_mins = [min([dphi_ijr(l[j], j), dphi_ijr(u[j],j)]) for j in range(dimx)]
            lijr_maxs = [max([dphi_ijr(l[j], j), dphi_ijr(u[j],j)]) for j in range(dimx)]
            lir_min = sum(lijr_mins)
            lir_max = sum(lijr_maxs)
          else:
            lir_min = 0.
            lir_max = 0.
            for j in range(dimx): 
              if self.kernel_spec == "matern32":
                _, _, lijr_min, lijr_max = dphir_minmax_threehalves(l[j], u[j], th[j] / self.X_scale[j], [self.x[i_idx][j], self.x[r_idx][j]])
              else: #matern 5/2
                _, _, lijr_min, lijr_max = dphir_minmax_fivehalves(l[j], u[j], th[j] / self.X_scale[j], [self.x[i_idx][j], self.x[r_idx][j]])
              lir_min += lijr_min
              lir_max += lijr_max

          #add_mccormick_ratio_constraints(cons=cons, ki=kvec[i_idx], kr=kvec[r_idx], lam_i=lamvar[i_idx], lam_r=lamvar[r_idx],
          #                                lir_min=lir_min, lir_max=lir_max, ki_min=kL[i_idx], ki_max=kU[i_idx],
          #                                kr_min=kL[r_idx], kr_max=kU[r_idx], name=f"{i_idx}_{r_idx}")
          add_ratio_constraints(cons, kvec[i_idx], kvec[r_idx], lir_min, lir_max)
          
          #sir_min, sir_max = compute_sigma_ir_bounds(l=l, u=u, theta=th, x_scale=self.X_scale, x_i=self.x[i_idx], x_r=self.x[r_idx],
          #                                           kernel_spec=self.kernel_spec, p=getattr(self, "p", 2.0))
          #add_ratio_informed_product_constraints(cons=cons, ki=kvec[i_idx], kr=kvec[r_idx], dirL=lir_min, dirU=lir_max, sirL=sir_min, sirU=sir_max)

          #add_mccormick_sum_product_constraints(cons, kvec[i_idx], kvec[r_idx], lamvar[i_idx], lamvar[r_idx], kL[i_idx], kU[i_idx],
          #                                      kL[r_idx], kU[r_idx], sir_min, sir_max)
          #add_product_constraints(cons, kvec[i_idx], kvec[r_idx], sir_min, sir_max)
    
    opt_tol = 1.e-8
    opt_rel_tol = 1.e-8
    for i in range(3):
      verbose = False
      if i > 0:
        max_iters = 1000
      else:
        max_iters = 300
      if i == 2:
        opt_rel_tol = 1.e-4
        verbose = True
      try:
        if mode == 0:
          prob = cp.Problem(cp.Minimize(self.obj2), cons)
          if not prob.is_dcp():
            raise RuntimeError("LCB relaxation is not DCP")
          
          acqf_L = prob.solve(solver=cp.CLARABEL, verbose=verbose, tol_gap_abs=opt_tol, tol_gap_rel=opt_rel_tol, max_iter=max_iters)
          #acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(solver=cp.SCS, verbose=verbose, eps_abs=opt_tol, max_iters=max_iters)

          if prob.status != cp.OPTIMAL:
            if prob.status == cp.OPTIMAL_INACCURATE:
              # be conservative
              acqf_L -= 10*opt_tol
            else:
              raise RuntimeError("LCB relaxation solver did not return an optimal solution")
          if self.diagnostics and opt_mode==6:
            diagnostics_output = stats_common_se_point(owner=self, l=l, u=u, xvar=xvar, wvar=wvar, lamvar=lamvar)
            diagnostics_output = stats_lcb_relaxation_gap(owner=self, xvar=xvar, relaxation_value=acqf_L) + diagnostics_output
            
          #if opt_mode in (5,6):
          #  assert mode == 0
          #  self.save_expsec_weights(cons, lamvar, lamL, lamU)

        else:
          sig_U = cp.Problem(cp.Maximize(self.obj3), cons).solve(solver=cp.CLARABEL, verbose=verbose, tol_gap_abs=opt_tol, tol_gap_rel=opt_rel_tol, max_iter=max_iters)
          if not (np.all(rhovar.value >= rhomin) and np.all(rhovar.value <= rhomax)):
            print("optimal rho not within rho bounds")
          #sig_U = cp.Problem(cp.Maximize(self.obj3), cons).solve(solver=cp.SCS, verbose=verbose, eps_abs=opt_tol, max_iters=max_iters)
        pass
      except Exception as e:
        pass
        print(f"WARNING: convex solver at attempt {i+1} returned error: {e}", flush=True)
        if i == 0:
          opt_tol *= 1.e4
        if i == 1:
          opt_tol *= 1.e2
        print("WARNING: Loosening convex opt tolerance to ", opt_tol, flush=True)

        if mode == 0:
          acqf_L = -np.inf
        else:
          sig_U = np.inf
        continue
      else:
        break # exit loop acqf_L successfully computed :)
    if mode == 0:
      return acqf_L, diagnostics_output
    else:
      return sig_U, diagnostics_output
  def compute_acqf_bounds(self, l, u, skip_LB=False):
    diagnostics_str = ""
    self.expsec_weights=np.ones(np.asarray(self.gamma).size)
    # kernel bounds
    kL, kU = self.ker_bounds(l, u)
    if self.kernel_spec == "pow_exp":
      assert self.p == 1.0 or self.p == 2.0, "not supporting p not equal to 1 or 2"

    failed_LB_opt = False
    if isinstance(self.acqf, LCBacquisition):
      # opt_mode = 0 (previous baseline w ratio constraints)
      # opt_mode = 1 (Convex relaxation w no ratio constraints on k and no affine constraints)
      # opt_mode = 2 (Most recent relaxation w ratio constraints and affine constraints
      opt_mode = self.opt_mode
      
      if not skip_LB:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=UserWarning)
          acqf_L, d_str = self.LCB_LB(l, u, kL, kU, opt_mode=opt_mode)
          diagnostics_str += f"{d_str}"
        for i in range(self.opt_mode):
          if not np.isfinite(acqf_L):
            failed_LB_opt = True
            print("Warning: was not able to determine lower-bound in previous mode ", opt_mode, "... switching", flush=True)
            opt_mode -= 1
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              acqf_L, d_str = self.LCB_LB(l, u, kL, kU, opt_mode=opt_mode)
              diagnostics_str += f"{d_str}"
              print(f"finished in mode {opt_mode}!!!!!!!!!!!!!!!!!!")
          else:
            failed_LB_opt = False
      else:
        acqf_L = -np.inf 
        failed_LB_opt = False
    if not isinstance(self.acqf, LCBacquisition) or failed_LB_opt:
      # mean bounds
      mu_L, mu_U = self.mu_bounds(kL, kU)
      sig_L = self.sig_LB(kL, kU, l=l, u=u)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sig_U = self.sig_UB(l, u, kL, kU)
      if np.isfinite(sig_U):
        var_L = sig_L ** 2.
        var_U = sig_U ** 2.
      else:
        var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
      # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
      mu  = np.array([mu_L, mu_U])
      var = np.array([var_U, var_L])
      acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
      acqf_L = acqf_bounds[0]
    
    acqf_solve_success = False 
    if not self.acqf_UB_solver == "MINEVAL": # local gradient-based optimization method
      constraints = []
      box_bounds = np.array([l, u]).T
      acqf_callback = {'obj' : self.acqf.scalar_evaluate}
      if self.acqf.has_gradient:
        acqf_callback['grad'] = self.acqf.scalar_eval_g
      opt_evaluator = Evaluator()

      # We need to be carefull here since the errors in the gradient (compared to FD) are in the 1e-4 range
      # Relax tolerance for dual infeasibility/norm of gradient of the Lagrangian
      if self.acqf_UB_solver == "IPOPT": 
        opt_solver_options = {
          'max_iter': 100,
          'tol': 1.e-5,
          'honor_original_bounds': 'yes',
          'print_level': 0,
          'sb': 'yes',
          'acceptable_iter': 5,
          'acceptable_tol': 5e-4,
        }
      else: #SLSQP
        opt_solver_options = {'maxiter' : 100, 'tol' : 1.e-5}
      acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
      alpha = 0.5 #0.05 + 0.9 * self.rng.random(len(u)) # rand numbers in [0.05, 0.95)
      x0 = [alpha * l + (1. - alpha) * u]
      opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
      if not (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)):
        print(f"optimizer {opt_sol[0]} not within prescribed bounds: {l}, {u}")
      assert (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)), f"acqf minimizer not within bounds"
      msg = opt_sol[3]
      acqf_solve_success = opt_sol[2]
      if not acqf_solve_success:
        #print(self.acqf_UB_solver + " did not converge on BOX: ", l, u, "... trying again with more verbosity and at another initial point", flush=True)
        print(self.acqf_UB_solver + " did not converge on BOX ... trying again with more verbosity and at another initial point", flush=True)
        print(self.acqf_UB_solver + " message: ", msg, flush=True)
        if self.acqf_UB_solver == "IPOPT":
          opt_solver_options = {
            'max_iter': 200,
            'tol': 1.e-3,
            'honor_original_bounds': 'yes',
            'print_level': 0,
            'sb': 'yes',
            'acceptable_iter': 5,
            'acceptable_tol': 1e-2
          }
        else: # SLSQP
          opt_solver_options = {'maxiter' : 200, 'tol' : 1.e-3, 'disp' : True}
        acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
        alpha = 0.5# 0.05 + 0.9 * self.rng.random(len(u)) # rand numbers in [0.05, 0.95)
        x0 = [alpha * l + (1. - alpha) * u]
        opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
        acqf_solve_success = opt_sol[2]
        if not acqf_solve_success:
          print(self.acqf_UB_solver + " failed a second time. Will take the minimum of a small number of acqf function evaluations", flush=True)
      if acqf_solve_success:
        acqf_U = self.acqf.evaluate(np.atleast_2d(opt_sol[0])).flatten()[0]
        acqf_U_x = opt_sol[0]
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
      min_arg = np.argmin(acqf_eval.flatten())
      acqf_U_x = x_points[min_arg]
      acqf_U = acqf_eval[min_arg]
    if acqf_L > acqf_U:
      if abs(acqf_U - acqf_L) / abs(acqf_U) < 1.e-4:
        acqf_L = acqf_U - 1.e-8
      else:
        print("issue with upper and lower-bound computations...", flush=True)
        print("acqf_L = {0:1.12e}, acqf_U = {1:1.12e}".format(acqf_L, acqf_U), flush=True)
    #make sure output is flush out to get all the info in case code asserts
    sys.stdout.flush()
    sys.stderr.flush()
    assert acqf_L <= acqf_U, "error: computed acquisition function bounds: acqf_U < acqf_L"
    if isinstance(acqf_L, (list, np.ndarray)):
      acqf_L = acqf_L[0]
    if isinstance(acqf_U, (list, np.ndarray)):
      acqf_U = acqf_U[0]
    return acqf_L, acqf_U, acqf_U_x, diagnostics_str

  def _refresh_legacy_views(self):
    """Expose compatibility views without duplicating authoritative records."""
    store = self.leaf_partition
    ready_nodes = sorted(
      store.ready_nodes(),
      key=lambda node: (node.aq_L, -node.depth, node.node_id),
    )
    self.queue = [(node.aq_L, int(node.node_id), node) for node in ready_nodes]
    heapq.heapify(self.queue)
    self.all_nonpruned_nodes = store.candidate_nodes()
    self.all_prunednodes = [
      node for node in store.leaves.values()
      if node.close_reason == CloseReason.PRUNED.value
    ]

  def export_partition(self):
    """Return every ready, in-flight, and closed leaf for BO warm start."""
    return self.leaf_partition.export_partition()

  def get_candidate_nodes(self):
    """Return leaves useful for batching, excluding only pruned leaves."""
    return self.leaf_partition.candidate_nodes()

  def optimize(self):
    opt = self.bnboptimize(self.gpsurrogate.xlimits[:,0], self.gpsurrogate.xlimits[:,1])
    lopt = opt[0]
    uopt = opt[1]
    aq_U_x = opt[-1]
    if aq_U_x is None:
      midpoint_opt = np.mean(np.array([lopt, uopt]), axis=0)
      return midpoint_opt
    else:
      return aq_U_x
  def initialize(self, l0=None, u0=None, queue=None, partition=None, transfer_lower_bound=None):
    """Initialize a root or reclassify a full leaf partition from previous iteration."""
    restart_worker = None
    if partition is not None:
      restart_worker = branching_wrapper(self.acqf, LUB=np.inf, epsilon_prune=self.epsilon_prune,
                                         acqf_UB_solver=self.acqf_UB_solver, random_seed=self.random_seed,
                                         opt_mode=self.opt_mode, nearest_neighbor_pairs=self.nearest_neighbor_pairs,
                                         diagnostics=self.diagnostics, restart_lower_bound=transfer_lower_bound)
      
    return initialize_async_search(self, l0=l0, u0=u0, queue=queue, partition=partition,
                                   transfer_lower_bound=transfer_lower_bound, restart_worker=restart_worker)

  def bnboptimize(self, l_init, u_init):
    """Run the certified asynchronous leaf-partition event loop."""
    return run_async_search(self, branching_wrapper, l_init, u_init)

class branching_wrapper:
  def __init__(self, acqf, LUB=np.inf, epsilon_prune=1.e-14, acqf_UB_solver="SLSQP", random_seed=None, opt_mode=3, nearest_neighbor_pairs=None, diagnostics=False, restart_lower_bound=None):
    self.LUB = LUB # least upper bound
    self.epsilon_prune = epsilon_prune
    self.acqf = acqf
    self.gpsurrogate = acqf.gpsurrogate
    self.x = self.gpsurrogate.training_x
    self.y = self.gpsurrogate.training_y
    self.acqf_UB_solver = acqf_UB_solver

    if nearest_neighbor_pairs is None:
      self.nearest_neighbor_pairs = np.empty((0, 2), dtype=np.int64)
    else:
      self.nearest_neighbor_pairs = nearest_neighbor_pairs

    self.random_seed = random_seed
    if random_seed is None:
      self.rng = np.random.default_rng()
    else:
      self.rng = np.random.default_rng(int(random_seed))
    
    if not (isinstance(self.acqf, LCBacquisition) or isinstance(self.acqf, EIacquisition)):
      raise NotImplementedError("Unrecognized acquisition function type")

    self.sync_from_smt()
    self.cvxpy_problem = None

    self.opt_mode = opt_mode
    self.diagnostics = diagnostics
    self.restart_lower_bound = restart_lower_bound
  
  def sync_from_smt(self):
    sm = self.gpsurrogate.surrogatesmt
    par = sm.optimal_par

    # --- kernel / corr selection ---
    corr = sm.options["corr"]  # e.g., 'squar_exp', 'pow_exp', 'abs_exp', 'matern12', 'matern32', 'matern52'
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
    elif corr == "matern12":
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
    # ensure that the matrix is negative semi-definite
    # //! 
    lam, U = np.linalg.eigh(self.A_constraint2)
    lam_neg = np.minimum(lam, -1.e-16 * np.ones(len(lam)))
    self.A_constraint2[:,:] = U.dot(np.diag(lam_neg)).dot(U.T)

    self.cons2 = 0.5 * cp.quad_form(self.X, self.A_constraint2) + self.b_constraint2 @ self.X + self.c_constraint2

    # set up a third objective function whose value is s wherein we will include a constraint s^2 <= \sig^2(k) 
    # also in which we will maximize s making s = max sig
    self.b_obj3 = np.zeros(ntrain + 1)
    self.c_obj3 = 0.0 # no shift necessary
    self.b_obj3[-1] = 1.0
    self.obj3 = self.b_obj3.T @ self.X + self.c_obj3
  
  def _normalize(self, x):
    return (np.asarray(x, float) - self.X_offset) / self.X_scale

  def save_expsec_weights(self, cons, lamvar, lamL, lamU, of=.05, rf=.05, eps=1e-12):
    X = np.asarray(self.X.value,float).ravel()
    gam = np.asarray(self.gamma,float).ravel()
    n = gam.size
    C = np.asarray(self.C,float)
    A = np.asarray(self.A_constraint2,float)
    b = np.asarray(self.b_constraint2,float).ravel()
    z,s = X[:-1],X[-1]
    k = C@z
    lam = np.asarray(lamvar.value,float).ravel()
    dv = cons[0].dual_value
    nu = float(np.asarray(dv).squeeze()) if dv is not None else np.nan
    if not np.isfinite(nu) or nu<0:
      nu = float(self.acqf.beta)/(2*max(abs(s),np.sqrt(eps)))
    qz = A[:n]@X+b[:n]
    g = float(np.asarray(self.y_std).ravel()[0])*gam-nu*np.linalg.solve(C.T,qz)
    #a = np.abs(g)
    a = np.maximum(-g,0.)
    a = np.abs(g) if a.max()<=eps else a
    m = a.max()
    omega = np.ones(n) if m<=eps else of+(1-of)*a/m
    lo,up = np.asarray(lamL,float).ravel(),np.asarray(lamU,float).ravel()
    d = np.maximum(up-lo,0.)
    h = np.divide(-np.expm1(-d),d,out=np.ones_like(d),where=d>1e-6)
    E = np.where(d>1e-6,np.exp(up)*(1-h+h*np.log(h)),np.exp((lo+up)/2)*d*d/8)
    r = np.maximum(k-np.exp(lam),0.)
    rho = rf+(1-rf)*np.minimum(1,r/(E+eps))
    self.expsec_omega, self.expsec_rho, self.expsec_weights = omega, rho, omega*rho
    self.expsec_lcb_gradient, self.expsec_residual, self.expsec_maxgap = g, r, E
  

  
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
  def LCB_LB(self, l, u, kL, kU, opt_mode=2):
    return self.convex_relaxation(l, u, kL, kU, opt_mode=opt_mode)
  def sig_UB(self, l, u, kL, kU, opt_mode=2):
    return self.convex_relaxation(l, u, kL, kU, opt_mode=opt_mode, mode=1)
  def sig_LB(self, kL, kU, l = None, u =None):
    # lower-bound
    if l is None or u is None:
      s2_L = 0.0
    else:
      # TODO: replace with local minimizer routine
      x = np.atleast_2d( (l + u) / 2.)
      S2 = self.gpsurrogate.variance(x)
      s2_L = min(S2.flatten())
    return np.sqrt(s2_L)
  def convex_relaxation(self, l, u, kL, kU, opt_mode=2, mode=0):
    """
       mode: 0 --> convex relaxation for minimum of LCB acquisition function
       mode: 1 --> convex relaxation for maximum of variance        
    """
    assert mode in [0, 1], "mode can only be in 0, 1"
    assert opt_mode in [0, 1, 2, 3, 4, 5, 6], "opt mode can only be 0, 1, 2, 3, 4, or 5"
    assert not (opt_mode in [0, 1, 2, 3, 4] and self.kernel_spec != "pow_exp"), "opt mode 0,1,2,3, and 4 limited to pow_exp kernel"
    # opt_mode = 0 (previous baseline w ratio constraints)
    # opt_mode = 1 (Convex relaxation w no ratio constraints on k and no affine constraints)
    # opt_mode = 2 (Most recent relaxation w ratio constraints and affine constraints)
    # opt_mode = 3 (Relaxation in w)
    # opt_mode = 4 (opt_mode 3 but with alternative to ratio constraints on k)
    cons = [self.cons2 >= 0, self.en1 @ self.X >= 0]

    diagnostics_output = ""
    
    # Let us keep these for now even though they are redundant
    #if opt_mode != 5 and opt_mode != 6:
    cons.append(self.C2 @ self.X >= kL)
    cons.append(self.C2 @ self.X <= kU)
    if opt_mode != 0 and opt_mode != 5 and opt_mode != 6:
      # add x optimization variable constrained to box: l <= x <= u
      xvar = cp.Variable(self.x.shape[1])
      cons.append(l <= xvar)
      cons.append(xvar <= u)
      # add rho_i = \sum_j \theta_j |(x_j - x_j^(i)) / X_scale|^p variable
      ntrain = self.x.shape[0]
      rhovar = cp.Variable(ntrain) 
      # determine bounds for rho
      dmin = np.maximum(0.0, np.maximum((l - self.x) / self.X_scale, (self.x - u)/ self.X_scale))        # (nt,d)
      dmax = np.maximum(np.abs((l - self.x) / self.X_scale), np.abs((u - self.x) / self.X_scale))         # (nt,d)
      th  = self.theta.ravel()     # (d,)
      rhomin = (th * (dmin**self.p)).sum(axis=1)
      rhomax = (th * (dmax**self.p)).sum(axis=1) 
      assert len(rhomin) == ntrain
      # \rho_i = max_{x in box} { \rho_i(x) }
      # no need to include these constraints with equality constraint rho_i = ...
      cons.append(rhovar <= rhomax) # pre w-relaxation bounds
      #cons.append(rhomin <= rhovar)    
      # --- constraints coupling k and rho ---
      # k_i => exp(-\rho_i)
      for i in range(ntrain):
        cons.append((self.C2 @ self.X)[i] >= cp.atoms.exp(-rhovar[i]))
      kMax = np.exp(-rhomin)
      kMin = np.exp(-rhomax)
      #cons.append(kMin <= self.C2 @ self.X) implied by rhovar <= rhomax
      cons.append(self.C2 @ self.X <= kMax) # pre-secant relaxation
      # k_i <= secant of k_i(x) over kMin, kMax
      # //!: I suggest we check and handle small |rhomax-rhomin| 
      cons.append(self.C2 @ self.X <= kMax + cp.atoms.multiply((kMin - kMax) / (rhomax - rhomin),  (rhovar  - rhomin)))
      # ---
      

      # \sum_j theta_j * |(x_j - x_j^(i)) / X_scale|^p <= rho_i
      # for p = 1 or 2 is convex
      if opt_mode < 3:
        for i in range(ntrain):
          # Note that this is redundant when opt_mode is 3, being implied by the eq. constraint on rhovar
          cons.append(cp.atoms.norm((xvar-self.x[i]) / ((th**(-1./self.p)) * self.X_scale), p = self.p)**self.p <= rhovar[i])
      if opt_mode >= 3:
        assert self.p == 2.0, "opt_mode 3 only supports squared exponential kernel"
        wvar = cp.Variable(self.x.shape[1])
        for i in range(self.x.shape[1]):
          cons.append(wvar[i] >= xvar[i]**2.0)
          cons.append(wvar[i] <= (l[i] + u[i]) * xvar[i] - l[i] * u[i])
        for i in range(ntrain):
          cons.append(rhovar[i] == cp.atoms.sum(cp.atoms.multiply(th / self.X_scale**self.p, wvar - 2.0 * cp.atoms.multiply(self.x[i], xvar) + self.x[i] * self.x[i]))) 


      # --- constraints ---
      # affine constraints on \rho_i via reverse triangle inequality
      # no need for additional bound constraints when we have equality constraint
      if opt_mode >= 2:
        if self.p == 1.0:
          for i in range(ntrain):
            for j in range(i+1, ntrain):
              rhoij_bound = np.linalg.norm(th**(1./self.p) * (self.x[i] - self.x[j]) / self.X_scale, ord=self.p)**self.p
              cons.append(rhovar[i] - rhovar[j] <= rhoij_bound)
              cons.append(rhovar[i] - rhovar[j] >= -1.0 * rhoij_bound)
        elif self.p == 2.0:
          if opt_mode<3:
            for i in range(ntrain):
              for j in range(i+1, ntrain):
                dxji = self.x[j] - self.x[i]
                mult = 2. * th * dxji / (self.X_scale ** 2.0)
                shift = np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                assert opt_mode!=3, " constraint is redundant when opt_mode==3, being implied by the eq. constraint on rhovar"
                cons.append(rhovar[i] - rhovar[j] == mult.T @ xvar + shift)
          elif opt_mode == 3:
            for i in range(ntrain):
              for j in range(i+1, ntrain):
                dxji = self.x[j] - self.x[i]
                lo = np.where(dxji >= 0., l, u)
                hi = np.where(dxji >= 0., u, l)
                # rhoij_lbound <= rho_i - rho_j <= rhoij_ubound
                # k_i <= k_j * exp(-1 * rhoij_lbound)
                # k_i >= k_j * exp(-1 * rhoij_ubound)
                rhoij_ubound = 2. * np.inner((hi / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                rhoij_lbound = 2. * np.inner((lo / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
                kij_ubound = np.exp(-1. * rhoij_lbound)
                kij_lbound = np.exp(-1. * rhoij_ubound)
                if not ((kij_ubound <= 1.e3 and kij_ubound >= 1.e-3) and (kij_lbound <= 1.e3 and kij_lbound >= 1.e-3)):
                  continue
                # it is more important that we place bounds on k than rho
                cons.append((self.C2 @ self.X)[i] <= (self.C2 @ self.X)[j] * kij_ubound)
                cons.append((self.C2 @ self.X)[i] >= (self.C2 @ self.X)[j] * kij_lbound)
        else: # opt_mode 4
          qvar = cp.Variable(int((ntrain * (ntrain -1) ) /2))
          dvar = cp.Variable(int((ntrain * (ntrain -1) ) /2))
          k = 0
          for i in range(ntrain):
            for j in range(i+1, ntrain):
              # d = 2 x^\top \Theta (x^{(j)} - x^{(i)}) + ||x^{(i)}||_{\Theta}^2 - ||x^{(j)}||_{\Theta}^2 
              cons.append(dvar[k] == cp.atoms.sum(cp.atoms.multiply(th / self.X_scale**self.p, 2.0 * cp.atoms.multiply(xvar, self.x[j] - self.x[i]) + self.x[i] * self.x[i] - self.x[j] * self.x[j]))) 
              # q >= exp(d)
              cons.append(qvar[k] >= cp.atoms.exp(-1.0 * dvar[k]))
              # --- begin d secant constraint ---
              
              dxji = self.x[j] - self.x[i]
              lo = np.where(dxji >= 0., l, u)
              hi = np.where(dxji >= 0., u, l)
              # rhoij_lbound <= rho_i - rho_j <= rhoij_ubound
              # k_i <= k_j * exp(-1 * rhoij_lbound)
              # k_i >= k_j * exp(-1 * rhoij_ubound)
              d_ubound = 2. * np.inner((hi / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
              d_lbound = 2. * np.inner((lo / self.X_scale), th * dxji / self.X_scale) + np.inner(self.x[i] / self.X_scale, th * self.x[i] / self.X_scale) - np.inner(self.x[j] / self.X_scale, th * self.x[j] / self.X_scale)
              q_ubound = np.exp(-1.0 * d_lbound)
              q_lbound = np.exp(-1.0 * d_ubound)       
              cons.append(qvar[k] <= q_ubound + ((q_lbound - q_ubound) / (d_ubound - d_lbound)) * (dvar[k] - d_lbound))
              # --- end d secant constraint ---

              # McCormick relaxation on product k_i = q_k * k_j
              # z = x * y
              # z >= x_l y + x * y_l - x_l * y_l
              # z >= x_u y + x * y_u - x_u * y_u
              # z <= x_u y + x * y_l - x_u * y_l
              # z <= x_l * y + x * y_u - x_l * y_u
              cons.append((self.C2 @ self.X)[i] >= q_lbound * (self.C2 @ self.X)[j] + qvar[k] * kMin[j] - q_lbound * kMin[j])
              cons.append((self.C2 @ self.X)[i] >= q_ubound * (self.C2 @ self.X)[j] + qvar[k] * kMax[j] - q_ubound * kMax[j])
              cons.append((self.C2 @ self.X)[i] <= q_ubound * (self.C2 @ self.X)[j] + qvar[k] * kMin[j] - q_ubound * kMin[j])
              cons.append((self.C2 @ self.X)[i] <= q_lbound * (self.C2 @ self.X)[j] + qvar[k] * kMax[j] - q_lbound * kMax[j])
              # ---- end McCormick relaxation on product k_i = q_k * k_j
              
              # add additional bound constraints on on d
              cons.append(dvar[k] <= d_ubound)
              cons.append(d_lbound <= dvar[k])
              k = k + 1
    elif opt_mode == 5 or opt_mode == 6:
      ntrain = self.x.shape[0]
      dimx = self.x.shape[1]
      # add x optimization variable constrained to box: l <= x <= u
      xvar = cp.Variable(dimx)
      cons.append(l <= xvar)
      cons.append(xvar <= u)
      cons.append(cp.atoms.power(cp.atoms.norm(self.X[:-1]), 2) <= 1.0) # k^T R^-1 k = z^T z <= 1
      # determine bounds for k and lambda
      dmin = np.maximum(0.0, np.maximum((l - self.x) / self.X_scale, (self.x - u)/ self.X_scale))        # (nt,d)
      dmax = np.maximum(np.abs((l - self.x) / self.X_scale), np.abs((u - self.x) / self.X_scale))         # (nt,d)
      th  = self.theta.ravel()     # (d,)
      lamvar = cp.Variable(ntrain)
      lamU = np.log(kU)
      lamL = np.log(kL)
      cons.append(lamvar >= lamL)
      cons.append(lamvar <= lamU)
      for i in range(ntrain):
        cons.append((self.C2 @ self.X)[i] >= cp.atoms.exp(lamvar[i]))
      cons.append(self.C2 @ self.X <= kL + cp.atoms.multiply((kU - kL) / (lamU - lamL),  (lamvar  - lamL)))
      etavar = cp.Variable((ntrain, dimx))
      cons.append(lamvar == cp.atoms.sum(etavar, axis=1)) # sum along column of matrix-valued \eta
      if self.kernel_spec == "pow_exp":
        assert self.p in [1.0, 2.0], "opt_mode 5 only support matern 1/2 (a.k.a. pow exp) and SE kernels"
        if self.p == 2.0:
          wvar = cp.Variable(dimx)
          for i in range(ntrain):
            for j in range(dimx):
              cons.append(etavar[i,j] == (-1.0 * th[j] / (self.X_scale[j]**self.p)) * (wvar[j] - 2. * self.x[i][j] * xvar[j] + self.x[i][j]**2))
          for j in range(dimx):
            cons.append(cp.atoms.square(xvar[j]) <= wvar[j])
            cons.append(wvar[j] <= (l[j] + u[j]) * xvar[j] - l[j] * u[j])
        elif self.p == 1.0:
          # --- tau and alpha are ragged arrays
          taus = [[] for j in range(dimx)]
          for j in range(dimx):
            taus[j].append(l[j])
            taus[j].append(u[j])
            for i in range(ntrain):
              if self.x[i][j] < u[j] and l[j] < self.x[i][j]:
                taus[j].append(self.x[i][j])
          alphavars = [cp.Variable(len(taus[j])) for j in range(dimx)]
          for i in range(ntrain):
            for j in range(dimx):
              cons.append(etavar[i][j] == cp.atoms.sum(cp.atoms.multiply(-1.0 * th[j] / (self.X_scale[j]) * np.abs(taus[j] - self.x[i][j]), alphavars[j])))
          for j in range(dimx):
            cons.append(xvar[j] == cp.atoms.sum(cp.atoms.multiply(taus[j], alphavars[j])))
            cons.append(cp.atoms.sum(alphavars[j]) == 1.0)
            for i in range(len(taus[j])):
              cons.append(alphavars[j][i] >= 0.0)
          
      else: #matern32 or matern52
        nu = 1.5
        if self.kernel_spec != "matern32":
          nu = 2.5
        for j in range(dimx):
          component_phi = matern_phi(self.x[:,j].tolist(), th[j] / self.X_scale[j], nu)
          D_rs = component_phi.generate_alpha_beta_r(l[j], u[j])
          for k in range(len(D_rs)):
            # alpha_m xj + beta_m^T eta_(:, j) <= r_m
            cons.append(D_rs[k][0] * xvar[j] + cp.atoms.scalar_product(D_rs[k][1], etavar[:,j]) <= D_rs[k][2])
      # add constraints based on downselected nearest neighbor pairs
      # downselect on available pairs
      if opt_mode == 6:
        Ei_exp = np.zeros(ntrain)
        Ai     = np.zeros(ntrain)
        kvec = self.C2 @ self.X

        sensitivity_floor = 0.05
        x_ref = 0.5 * (np.asarray(l, dtype=float) + np.asarray(u, dtype=float))
        lcb_grad_k, k_ref, sigma_ref, mean_grad_k, variance_grad_k = lcb_gradient_at_single_reference(self, x_ref)
        abs_lcb_grad_k = np.abs(lcb_grad_k)
        normalized_lcb_sensitivity = abs_lcb_grad_k / max(float(np.max(abs_lcb_grad_k)), np.finfo(float).eps)

        for i in range(ntrain):
          #compute Ei_exp
          if lamU[i] > lamL[i]:
            # point where gap between exp and its secant is largest

            # original code misses  "- exp(lamstar)" for Ei_exp[i]
            # lamstar = np.log((np.exp(lamU[i]) - np.exp(lamL[i])) / (lamU[i] - lamL[i]))
            # Ei_exp[i] = np.exp(lamL[i]) + (np.exp(lamU[i]) - np.exp(lamL[i])) / (lamU[i] - lamL[i]) * (lamstar - lamL[i])
            exp_lamL = np.exp(lamL[i])
            lam_exp_diff = np.exp(lamU[i]) - exp_lamL
            slope = lam_exp_diff / (lamU[i] - lamL[i])
            lamstar = np.log(slope)
            sec_at_star = exp_lamL + slope*(lamstar - lamL[i])                                                          
            Ei_exp[i] = sec_at_star - slope
            if Ei_exp[i] < 0:
              raise RuntimeError("roundoff error: diff between exp and sec should be zero")            
          else:
            Ei_exp[i] = 0.
          #Ai[i] = Ei_exp[i] * (gamma_floor + (1. - gamma_floor) * np.abs(self.gamma[i]) / (np.max(np.abs(self.gamma)) + eps_gamma))
          Ai[i] = Ei_exp[i] * (sensitivity_floor + (1.0 - sensitivity_floor) * normalized_lcb_sensitivity[i])
        pair_selection_triplets = np.array([[pair[0], pair[1], Ai[pair[0]] + Ai[pair[1]]] for pair in self.nearest_neighbor_pairs]).reshape(-1, 3)
        args = np.argsort(pair_selection_triplets[:,-1])[::-1]
        pair_selection_triplets[:,:] = pair_selection_triplets[args,:]
        # now find c1 * p pairs
        c1 = 5
        ndownselect_pairs = min(len(self.nearest_neighbor_pairs), c1 * ntrain)
        for pair in pair_selection_triplets[:ndownselect_pairs]:
          i_idx = int(pair[0])
          r_idx = int(pair[1])
          lir_min = 0.
          lir_max = 0.
          if self.kernel_spec == "pow_exp":
            dphi_ijr = lambda t,j: -th[j] / (self.X_scale[j]**self.p) * (np.abs(t - self.x[i_idx][j])**self.p - np.abs(t - self.x[r_idx][j])**self.p)
            lijr_mins = [min([dphi_ijr(l[j], j), dphi_ijr(u[j],j)]) for j in range(dimx)]
            lijr_maxs = [max([dphi_ijr(l[j], j), dphi_ijr(u[j],j)]) for j in range(dimx)]
            lir_min = sum(lijr_mins)
            lir_max = sum(lijr_maxs)
          else:
            lir_min = 0.
            lir_max = 0.
            for j in range(dimx): 
              if self.kernel_spec == "matern32":
                _, _, lijr_min, lijr_max = dphir_minmax_threehalves(l[j], u[j], th[j] / self.X_scale[j], [self.x[i_idx][j], self.x[r_idx][j]])
              else: #matern 5/2
                _, _, lijr_min, lijr_max = dphir_minmax_fivehalves(l[j], u[j], th[j] / self.X_scale[j], [self.x[i_idx][j], self.x[r_idx][j]])
              lir_min += lijr_min
              lir_max += lijr_max

          #add_mccormick_ratio_constraints(cons=cons, ki=kvec[i_idx], kr=kvec[r_idx], lam_i=lamvar[i_idx], lam_r=lamvar[r_idx],
          #                                lir_min=lir_min, lir_max=lir_max, ki_min=kL[i_idx],
          #                                ki_max=kU[i_idx], kr_min=kL[r_idx], kr_max=kU[r_idx], name=f"{i_idx}_{r_idx}")
          add_ratio_constraints(cons, kvec[i_idx], kvec[r_idx], lir_min, lir_max)
          
          #sir_min, sir_max = compute_sigma_ir_bounds(l=l, u=u, theta=th, x_scale=self.X_scale, x_i=self.x[i_idx], x_r=self.x[r_idx],
          #                                           kernel_spec=self.kernel_spec, p=getattr(self, "p", 2.0))
          #add_ratio_informed_product_constraints(cons=cons, ki=kvec[i_idx], kr=kvec[r_idx], dirL=lir_min, dirU=lir_max, sirL=sir_min, sirU=sir_max)
          #add_mccormick_sum_product_constraints(cons, kvec[i_idx], kvec[r_idx], lamvar[i_idx], lamvar[r_idx], kL[i_idx], kU[i_idx],
          #                                      kL[r_idx], kU[r_idx], sir_min, sir_max)
          #add_product_constraints(cons, kvec[i_idx], kvec[r_idx], sir_min, sir_max)
          
    opt_tol = 1.e-8
    opt_rel_tol = 1.e-8
    for i in range(3):
      verbose = False
      if i > 0:
        max_iters = 1000
        #verbose = True
      else:
        max_iters = 300
      if i == 2:
        opt_rel_tol = 1.e-4
        
      try:
        if mode == 0:
          prob = cp.Problem(cp.Minimize(self.obj2), cons)
          if not prob.is_dcp():
            print("is not DCP")
            raise RuntimeError("LCB relaxation is not DCP")

          acqf_L = prob.solve(solver=cp.CLARABEL, verbose=verbose, tol_gap_abs=opt_tol, tol_gap_rel=opt_rel_tol, max_iter=max_iters)
          #acqf_L = cp.Problem(cp.Minimize(self.obj2), cons).solve(solver=cp.SCS, verbose=verbose, eps_abs=opt_tol, max_iters=max_iters)

          if prob.status != cp.OPTIMAL:
            if prob.status == cp.OPTIMAL_INACCURATE:
              # be conservative
              acqf_L -= 10*opt_tol
            else:
              raise RuntimeError("LCB relaxation solver did not return an optimal solution")

          if self.diagnostics and opt_mode==6:
            diagnostics_output = stats_common_se_point(owner=self, l=l, u=u, xvar=xvar, wvar=wvar, lamvar=lamvar)
            diagnostics_output = stats_lcb_relaxation_gap(owner=self, xvar=xvar, relaxation_value=acqf_L) + diagnostics_output

          if opt_mode in (5,6):
            assert mode == 0
            self.save_expsec_weights(cons, lamvar, lamL, lamU)

        else:
          sig_U = cp.Problem(cp.Maximize(self.obj3), cons).solve(solver=cp.CLARABEL, verbose=verbose, tol_gap_abs=opt_tol, tol_gap_rel=opt_rel_tol, max_iter=max_iters)
          if not (np.all(rhovar.value >= rhomin) and np.all(rhovar.value <= rhomax)):
            print("optimal rho not within rho bounds")
        pass
      except Exception as e:
        pass
        print(f"WARNING: convex solver at attempt {i+1} returned error: {e}", flush=True)
        if i == 0:
          opt_tol *= 1.e4
        if i == 1:
          opt_tol *= 1.e2
        print("WARNING: Loosening convex opt tolerance to ", opt_tol, flush=True)

        if mode == 0:
          acqf_L = -np.inf
        else:
          sig_U = np.inf
        continue
      else:
        break # exit loop acqf_L successfully computed :)
    if mode == 0:
      return acqf_L, diagnostics_output
    else:
      return sig_U, diagnostics_output
  
  def compute_acqf_bounds(self, l, u, skip_LB=False, skip_UB=False):
    diagnostics_str = ""
    # kernel bounds
    kL, kU = self.ker_bounds(l, u)
    if self.kernel_spec == "pow_exp":
      assert self.p == 1.0 or self.p == 2.0, "not supporting p not equal to 1 or 2"
    
    failed_LB_opt = False
    if isinstance(self.acqf, LCBacquisition):
      # opt_mode = 0 (previous baseline w ratio constraints)
      # opt_mode = 1 (Convex relaxation w no ratio constraints on k and no affine constraints)
      # opt_mode = 2 (Most recent relaxation w ratio constraints and affine constraints
      opt_mode = self.opt_mode
      
      if not skip_LB:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=UserWarning)
          #
          # Lower bound
          #
          acqf_L, d_str = self.LCB_LB(l, u, kL, kU, opt_mode=opt_mode)
          diagnostics_str += f"{d_str}"
        for i in range(self.opt_mode):
          if not np.isfinite(acqf_L):
            failed_LB_opt = True
            print("Warning: was not able to determine lower-bound in previous mode ", opt_mode, "... switching", flush=True)
            opt_mode -= 1
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              acqf_L, d_str = self.LCB_LB(l, u, kL, kU, opt_mode=opt_mode)
              diagnostics_str += f"{d_str}"
              print(f"finished in mode {opt_mode}!!!!!!!!!!!!!!!!!!")
          else:
            failed_LB_opt = False
      else:
        acqf_L = -np.inf 
        failed_LB_opt = False
    if not isinstance(self.acqf, LCBacquisition) or failed_LB_opt:
      # mean bounds
      mu_L, mu_U = self.mu_bounds(kL, kU)
      sig_L = self.sig_LB(kL, kU, l=l, u=u)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sig_U = self.sig_UB(l, u, kL, kU)
      if np.isfinite(sig_U):
        var_L = sig_L ** 2.
        var_U = sig_U ** 2.
      else:
        var_L, var_U = self.sigma2_bounds(kL, kU, l=l, u=u)
      # evaluate acquisition at (mu_L, var_U) and (mu_U, var_L)
      mu  = np.array([mu_L, mu_U])
      var = np.array([var_U, var_L])
      acqf_bounds = self.acqf.evaluate_meansig2(mu, var)
      acqf_L = acqf_bounds[0]

    if skip_UB:
      return float(np.asarray(acqf_L).reshape(-1)[0]), np.inf, None, diagnostics_str
    
    acqf_solve_success = False 
    if not self.acqf_UB_solver == "MINEVAL": # local gradient-based optimization method
      constraints = []
      box_bounds = np.array([l, u]).T
      acqf_callback = {'obj' : self.acqf.scalar_evaluate}
      if self.acqf.has_gradient:
        acqf_callback['grad'] = self.acqf.scalar_eval_g
      opt_evaluator = Evaluator()

      # We need to be carefull here since the errors in the gradient (compared to FD) are in the 1e-4 range
      # Relax tolerance for dual infeasibility/norm of gradient of the Lagrangian
      if self.acqf_UB_solver == "IPOPT": 
        opt_solver_options = {
          'max_iter': 100,
          'tol': 1.e-5,
          'honor_original_bounds': 'yes',
          'print_level': 0,
          'sb': 'yes',
          'acceptable_iter': 5,
          'acceptable_tol': 5e-4,
        }
      else: #SLSQP
        opt_solver_options = {'maxiter' : 100, 'tol' : 1.e-5}
      acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
      alpha = 0.5 #0.05 + 0.9 * self.rng.random(len(u)) # rand numbers in [0.05, 0.95)
      x0 = [alpha * l + (1. - alpha) * u]
      opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
      if not (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)):
        print(f"optimizer {opt_sol[0]} not within prescribed bounds: {l}, {u}")
      assert (np.all(opt_sol[0] >= l) and np.all(opt_sol[0] <= u)), f"acqf minimizer not within bounds"
      msg = opt_sol[3]
      acqf_solve_success = opt_sol[2]
      if not acqf_solve_success:
        print(self.acqf_UB_solver + " did not converge on BOX ... trying again with more verbosity and at another initial point", flush=True)
        print(self.acqf_UB_solver + " message: ", msg, flush=True)
        if self.acqf_UB_solver == "IPOPT":
          opt_solver_options = {
            'max_iter': 200,
            'tol': 1.e-3,
            'honor_original_bounds': 'yes',
            'print_level': 0,
            'sb': 'yes',
            'acceptable_iter': 5,
            'acceptable_tol': 1e-2
          }
        else: # SLSQP
          opt_solver_options = {'maxiter' : 200, 'tol' : 1.e-3, 'disp' : True}
        acqf_minimizer = minimizer_wrapper(acqf_callback, self.acqf_UB_solver, box_bounds, constraints, opt_solver_options)
        alpha = 0.5# 0.05 + 0.9 * self.rng.random(len(u)) # rand numbers in [0.05, 0.95)
        x0 = [alpha * l + (1. - alpha) * u]
        opt_sol = opt_evaluator.run(acqf_minimizer.minimizer_callback, x0)[0]
        acqf_solve_success = opt_sol[2]
        if not acqf_solve_success:
          print(self.acqf_UB_solver + " failed a second time. Will take the minimum of a small number of acqf function evaluations", flush=True)
      if acqf_solve_success:
        acqf_U = self.acqf.evaluate(np.atleast_2d(opt_sol[0])).flatten()[0]
        acqf_U_x = opt_sol[0]
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
      min_arg = np.argmin(acqf_eval.flatten())
      acqf_U_x = x_points[min_arg]
      acqf_U = acqf_eval[min_arg]
    if acqf_L > acqf_U:
      if abs(acqf_U - acqf_L) / abs(acqf_U) < 1.e-4:
        acqf_L = acqf_U - 1.e-8
      else:
        print("issue with upper and lower-bound computations...", flush=True)
        print("acqf_L = {0:1.12e}, acqf_U = {1:1.12e}".format(acqf_L, acqf_U), flush=True)
    #make sure output is flush out to get all the info in case code asserts
    sys.stdout.flush()
    sys.stderr.flush()
    assert acqf_L <= acqf_U, "error: computed acquisition function bounds: acqf_U < acqf_L"
    if isinstance(acqf_L, (list, np.ndarray)):
      acqf_L = acqf_L[0]
    if isinstance(acqf_U, (list, np.ndarray)):
      acqf_U = acqf_U[0]
    return acqf_L, acqf_U, acqf_U_x, diagnostics_str

  def restart_callback(self, nodes):
    nodes = np.asarray(nodes, dtype=object).reshape(-1)
    if nodes.size != 1:
      raise ValueError(f"restart_callback expects exactly one leaf, received {nodes.size}")

    old_leaf = nodes[0]
    node_id = int(old_leaf.node_id)

    try:
      if self.restart_lower_bound is None:
        aq_L, _, _, _ = self.compute_acqf_bounds(old_leaf.l, old_leaf.u, skip_UB=True)
      else:
        aq_L = float(self.restart_lower_bound(old_leaf))

      l = np.asarray(old_leaf.l, dtype=float).reshape(-1)
      u = np.asarray(old_leaf.u, dtype=float).reshape(-1)
      point = old_leaf.aq_U_x

      if point is not None:
        point = np.asarray(point, dtype=float).reshape(-1)

      if (point is None or point.size != l.size or not np.all(np.isfinite(point)) or np.any(point < l-1.e-12) or np.any(point > u+1.e-12)):
        point = 0.5*(l+u)

      point = point.copy()
      aq_U = float(np.asarray(self.acqf.evaluate(np.atleast_2d(point))).reshape(-1)[0])

      return [RestartResult(node_id=node_id, aq_L=float(aq_L), aq_U=aq_U, aq_U_x=point)]

    except Exception as exc:
      return [RestartResult(node_id=node_id, error=f"{type(exc).__name__}: {exc}")]
  
  def callback(self, nodes):
    parents = list(nodes.flatten())
    if len(parents) != 1:
      raise ValueError("Each asynchronous BnB task must contain exactly one parent")
    parent = parents[0]
    weights = (parent.metadata or {}).get("expsec_weights")
    weights = None
    started = time.time()
    try:
      child_boxes = minmax_expsec_branch(parent.l, parent.u, self, weights)
      #child_boxes = gradient_branch(parent.l, parent.u, self.acqf)
      #child_boxes = branch(parent.l, parent.u)
      if len(child_boxes) != 2:
        raise RuntimeError("branch() did not return exactly two child boxes")
      children = []
      for child_l, child_u in child_boxes:
        acqf_L, acqf_U, acqf_U_x, d_str = self.compute_acqf_bounds(child_l, child_u)

        metadata={"diagnostics":d_str.rstrip("\n"),
                  "expsec_weights":self.expsec_weights.copy()}
        child = BnBNode(child_l, child_u, acqf_L, acqf_U, acqf_U_x, metadata=metadata)
        children.append(child)
      result = BranchResult(
        parent_id=int(parent.node_id),
        generation=int(parent.generation),
        children=tuple(children),
        worker_metadata={"elapsed": time.time() - started},
      )
    except Exception as exc:
      result = BranchResult(
        parent_id=int(parent.node_id),
        generation=int(parent.generation),
        error=f"{type(exc).__name__}: {exc}",
        worker_metadata={"elapsed": time.time() - started},
      )
    # MPIEvaluator(function_mode=False) removes this one-element wrapper.
    return [result]
