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
    design_space = DesignSpace(xlimits, random_state=random_state)

    if fix_theta:
      theta_bounds_ = [theta0-1.e-8, theta0+1.e-8]
    else:
      theta_bounds_ = theta_bounds
    self.surrogatesmt = KRG(design_space=design_space,
                            print_global=False,
                            noise0=[noise0],
                            eval_noise=False,
                            corr=corr,
                            pow_exp_power=pow_exp_power,
                            hyper_opt=hyper_opt,
                            theta0 = [theta0] * ndim,
                            nugget=nugget,
                            theta_bounds=theta_bounds_,
                            )
    self.trained = False

  def mean(self, x):
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict mean or variances")
    return self.surrogatesmt.predict_values(x)

  def variance(self, x):
    if not self.trained:
      raise ValueError("must train kriging model before utilizing it to predict mean or variances")
    return self.surrogatesmt.predict_variances(x)

  def train(self, x, y):
    self.training_x = x
    self.training_y = y
    self.surrogatesmt.set_training_values(x, y)
    self.surrogatesmt.train()
    self.trained = True

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
