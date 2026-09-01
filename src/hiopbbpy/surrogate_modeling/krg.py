'''
A subclass of GaussianProcess that implements a Kriging surrogate model using package SMT

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
'''

import numpy as np
from .gp import GaussianProcess
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace 


class smtKRG(GaussianProcess):
  def __init__(self, theta0, xlimits, ndim, corr="pow_exp", pow_exp_power=1.0, noise0=0.0, nugget = 100. * np.finfo(np.double).eps, random_state=None, hyper_opt="TNC", eval_noise=False, fix_theta = False, theta_bounds=[0.1,10.]):
    super().__init__(ndim, xlimits)
    if random_state is None:
      random_state = 42
    design_space = DesignSpace(xlimits, seed=random_state)
    if fix_theta:
      theta_bounds = [theta0-1.e-8, theta0+1.e-8]
      assert hyper_opt == "NoOp", "fix_theta=True should be used with hyper_opt='NoOp'"

    self._full_hyper_opt = hyper_opt
    self.surrogatesmt = KRG(design_space=design_space,
                            print_global=False,
                            noise0=[noise0],
                            eval_noise=False,
                            corr=corr,
                            pow_exp_power=pow_exp_power,
                            hyper_opt=hyper_opt,
                            theta0 = [theta0] * ndim,
                            nugget=nugget,
                            theta_bounds=theta_bounds)
    self.trained = False

  def mean(self, x):
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict mean or variances")
    return self.surrogatesmt.predict_values(x)

  def variance(self, x):
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict mean or variances")
    return self.surrogatesmt.predict_variances(x)

  def train(self, x, y, *, optimize_theta=True, theta_bounds=None):
    assert (theta_bounds is None) or (optimize_theta is True),  "Changing GP theta bounds requires reoptimizing theta"

    sm = self.surrogatesmt

    if theta_bounds is not None:
      bounds = np.asarray(theta_bounds, dtype=float).ravel()
      if bounds.size != 2 or not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f"Invalid theta_bounds: {theta_bounds}")
      sm.options["theta_bounds"] = bounds.tolist()
    else:
      bounds = np.asarray(sm.options["theta_bounds"], dtype=float)
      
    # Warm-start from the last (anisotropic) fit
    theta0 = np.asarray(sm.optimal_theta, dtype=float).ravel() if self.trained else np.asarray(sm.options["theta0"], dtype=float).ravel()
    sm.options["theta0"] = np.clip(theta0, bounds[0], bounds[1]).tolist()

    old_n_start = sm.options["n_start"]
    sm.options["hyper_opt"] = self._full_hyper_opt if optimize_theta else "NoOp"

    # NoOp uses only the first supplied theta; avoid unnecessary LHS starts.
    if not optimize_theta:
      sm.options["n_start"] = 1

    self.trained = False
    try:
      self.training_x = x
      self.training_y = y
      sm.set_training_values(x, y)
      sm.train()
      self.trained = True
    finally:
      # Keep the persistent configuration representing a full retrain.
      sm.options["hyper_opt"] = self._full_hyper_opt
      sm.options["n_start"] = old_n_start  

  def mean_gradient(self, x: np.ndarray) -> np.ndarray:
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict gradient")
    assert (np.size(x,-1) == self.ndim)
    gradient = [self.surrogatesmt._predict_derivatives(x, kx) for kx in range(self.ndim)]
    return np.atleast_2d(gradient).T

  def variance_gradient(self, x: np.ndarray) -> np.ndarray:
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict gradient")
    return self.surrogatesmt.predict_variance_gradient(x)

  def set_nugget(self, nugget):
    assert nugget >= 0., "nugget value must be non-negative"
    self.surrogatesmt.options["nugget"] = nugget

def smt_theta_bounds(S: int, N: int, corr: str = "squar_exp", pow_exp_power: float = 2., min_half_corr_spacing: float = 0.5, max_half_corr_widths: float = 2.0) -> np.ndarray:
    """
    Estimate common lower/upper bounds for SMT's per-dimension theta.

    The rule of thumb implemented here is very crude, does NOT consider the actual sample values, 
    and should be used under the assumptions below, for example, with the initial BO samples.

    Assumptions
    -----------
    - S locations are independent and approximately uniform in an N-dimensional hyperrectangle.
    - Input coordinate are standardized by the training-set std dev by SMT.

    The bounds allow the one-coordinate half-correlation distance to range from half the nominal 
    sample spacing to twice the standardized domain width.

    Returns: ndarray, shape (2,)
        Directly usable as SMT's ``theta_bounds``.
    """
    if S < 2:
        raise ValueError(f"S must be an integer >= 2; got {S}")

    if N < 1:
        raise ValueError(f"N must be an integer >= 1; got {N!r}")

    if min_half_corr_spacing <= 0.0 or max_half_corr_widths <= 0.0:
        raise ValueError("Half-correlation distance factors must be positive")

    corr = str(corr).lower()
    if corr == "pow_exp":
        p = float(pow_exp_power)
        if not np.isfinite(p) or not 0.0 < p <= 2.0:
            raise ValueError("pow_exp_power must be in (0, 2]")
        coefficient = np.log(2.0)
    elif corr in ("abs_exp", "matern12"):
        p = 1.0
        coefficient = np.log(2.0)
    elif corr == "squar_exp":
        p = 2.0
        coefficient = np.log(2.0)
    elif corr == "matern32":
        p = 1.0
        # theta * a when (1 + sqrt(3)*theta*a)
        # exp(-sqrt(3)*theta*a) == 0.5
        coefficient = 0.9689940864797172
    elif corr == "matern52":
        p = 1.0
        # theta * a when (1 + z + z**2/3) exp(-z) == 0.5,
        # with z = sqrt(5)*theta*a
        coefficient = 1.042122250130133
    else:
        raise ValueError("corr must be one of pow_exp, abs_exp/matern12, squar_exp, matern32, or matern52")

    # Nominal grid resolution of S points in N dimensions.
    points_per_dimension = np.exp(np.log(float(S)) / float(N))

    # A uniform coordinate of width L has std = L/sqrt(12), so after SMT standardization its domain width is
    # approximately sqrt(12).
    standardized_width = np.sqrt(12.0)

    shortest_half_corr_distance = min_half_corr_spacing * standardized_width / points_per_dimension
    longest_half_corr_distance = max_half_corr_widths * standardized_width

    # Small theta means long/smooth correlation; large theta means short/rough correlation.
    lower = coefficient / longest_half_corr_distance**p
    upper = coefficient / shortest_half_corr_distance**p

    return np.array([lower, upper], dtype=float)
