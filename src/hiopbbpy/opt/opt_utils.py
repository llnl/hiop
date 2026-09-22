"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from .optproblem import IpoptProb
#import time

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
    msg = ""
    for x0 in x0s:
      #time.sleep(2.5)
      if self.method in ["SLSQP", "trust-constr"]:
        if 'tol' in self.solver_options:
          tol = self.solver_options['tol']
          solver_options = self.solver_options.copy()
          del solver_options['tol']
        else:
          solver_options = self.solver_options.copy()
          tol = None
        if 'grad' in self.fun:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, jac=self.fun['grad'], constraints=self.constraints, tol=tol, options=solver_options)
        else:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=self.constraints, tol=tol, options=solver_options)
        success = y.success
        if not success:
          msg = y.message
        xopt = y.x
        yopt = y.fun
      elif self.method == "trust-constr":
        #nonlinear_constraint = NonlinearConstraint(self.constraints['cons'], self.constraints['cl'], self.constraints['cu'], jac=self.constraints['jac'])
        bounds = Bounds(lb= self.bounds[0], ub=self.bounds[1], keep_feasible=True) 
        y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, tol=tol,options=self.solver_options)#constraints=[nonlinear_constraint], options=self.solver_options)
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
        if status == 0 or status == 1:
          # ipopt returns 0 as success, and 1 as acceptable success
          success = True
        else:
          msg = f"Ipopt failed to solve the problem. Status: {status}  Status msg: {msg}"
          success = False
    
        yopt = info['obj_val']
        xopt = sol
      output.append([xopt, yopt, success, msg])
    return output

from scipy.optimize import lsq_linear
import numpy as np
def fit_common_se_point_from_ratios(owner, k_values, l, u, anchor=0):
    """
    Fit one common point using all log-kernel ratios.

    Small pair_residuals mean that the relaxed kernel ratios are
    consistent with one point.  The absolute residuals then test
    whether the common multiplicative scale is also correct.
    """
    k_values = np.asarray(k_values, dtype=float).ravel()

    if np.any(~np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise ValueError("All kernel values must be finite and positive")

    theta = np.asarray(owner.theta, dtype=float).ravel()
    Xc = np.asarray(owner.Xc, dtype=float)

    l_c = (
        np.asarray(l, dtype=float).ravel()
        - owner.X_offset
    ) / owner.X_scale
    u_c = (
        np.asarray(u, dtype=float).ravel()
        - owner.X_offset
    ) / owner.X_scale

    rho_target = -np.log(k_values)
    center_norm2 = np.sum(
        theta[None, :] * Xc**2,
        axis=1,
    )

    p = len(k_values)
    other = np.asarray(
        [i for i in range(p) if i != anchor],
        dtype=int,
    )

    if len(other) == 0:
        raise ValueError(
            "At least two kernel components are required"
        )

    # rho_i - rho_r
    # = 2 (c_r - c_i)^T Theta y
    #   + ||c_i||_Theta^2 - ||c_r||_Theta^2.
    A = 2.0 * (Xc[anchor][None, :] - Xc[other]) * theta[None, :]
    b = rho_target[other] - rho_target[anchor] - (center_norm2[other] - center_norm2[anchor])

    result = lsq_linear(A, b, bounds=(l_c, u_c), method="trf", lsmr_tol="auto")

    y_common = result.x
    x_common = (
        owner.X_offset
        + owner.X_scale * y_common
    )

    pair_residual = A @ y_common - b

    rho_common = np.sum(
        theta[None, :]
        * (y_common[None, :] - Xc) ** 2,
        axis=1,
    )

    absolute_residual = rho_common - rho_target

    return {
        "x_common": x_common,
        "y_common": y_common,
        "rank": int(np.linalg.matrix_rank(A)),
        "pair_rms": float(
            np.sqrt(np.mean(pair_residual**2))
        ),
        "pair_max": float(
            np.max(np.abs(pair_residual))
        ),
        "absolute_rms": float(
            np.sqrt(np.mean(absolute_residual**2))
        ),
        "absolute_max": float(
            np.max(np.abs(absolute_residual))
        ),
        "absolute_mean": float(
            np.mean(absolute_residual)
        ),
        "absolute_std": float(
            np.std(absolute_residual)
        ),
        "pair_residual": pair_residual,
        "absolute_residual": absolute_residual,
    }
