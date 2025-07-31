"""
This file implements different acquisition functions, which are used in Bayesian optimization to decide where to sample next.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from scipy.stats import norm
from ..surrogate_modeling.gp import GaussianProcess

# A base class for acquisition functions
class acquisition(object):
  def __init__(self, gpsurrogate):
    assert isinstance(gpsurrogate, GaussianProcess) # add something here
    self.gpsurrogate = gpsurrogate
    self.has_gradient = False
  
  # Abstract method to evaluate the acquisition function at x.
  def evaluate(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Child class of acquisition should implement method evaluate")
  def scalar_evaluate(self, x: np.ndarray) -> float:
    assert len(x.shape) == 1, f"scalar_evaluate intended for use with vector inputs and not arrays of vector values x"
    return self.evaluate(np.atleast_2d(x))[0][0]  

  # Abstract method to evaluate the gradient of acquisition function at x.
  def eval_g(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Child class of acquisition should implement method evaluate")
  def scalar_eval_g(self, x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 1, f"scalar_eval_g intended for use with a single input and not an array of inputs x"
    return np.array(self.eval_g(np.atleast_2d(x))).flatten()

# A subclass of acquisition, implementing the Lower Confidence Bound (LCB) acquisition function.
class LCBacquisition(acquisition):
  def __init__(self, gpsurrogate, beta=3.0):
    super().__init__(gpsurrogate)
    self.beta = beta
    self.has_gradient = True

  # Method to evaluate the acquisition function at x.
  def evaluate(self, x : np.ndarray) -> np.ndarray:
    mu = self.gpsurrogate.mean(x)
    sig2 = self.gpsurrogate.variance(x)
    return mu - self.beta * np.sqrt(sig2)

  def eval_g(self, x: np.ndarray) -> np.ndarray:
    mu = self.gpsurrogate.mean(x)
    sig2 = self.gpsurrogate.variance(x)
    dsig2_dx = self.gpsurrogate.variance_gradient(x)
    dmu_dx = self.gpsurrogate.mean_gradient(x)
    return dmu_dx - 0.5 * self.beta * dsig2_dx / np.sqrt(sig2)

# A subclass of acquisition, implementing the Expected improvement (EI) acquisition function.
class EIacquisition(acquisition):
  def __init__(self, gpsurrogate):
      super().__init__(gpsurrogate)
      self.has_gradient = True

  # Method to evaluate the acquisition function at x.
  def evaluate(self, x : np.ndarray) -> np.ndarray:        
    y_data = self.gpsurrogate.training_y
    y_min = y_data[np.argmin(y_data[:, 0])]

    pred = self.gpsurrogate.mean(x)
    sig = np.sqrt(self.gpsurrogate.variance(x))

    retval = []
    if sig.size == 1 and np.abs(sig) > 1e-24:
      z = (y_min - pred) / sig
      retval = (y_min - pred) * norm.cdf(z) + sig * norm.pdf(z)
    elif sig.size == 1 and np.abs(sig) <= 1e-24:
      retval = 0.0
    elif sig.size > 1:
      raise NotImplementedError("TODO --- Not implemented yet!")

    ## instead of maximize EI, we minimize -EI
    retval *= -1.

    return retval

  def eval_g(self, x: np.ndarray) -> np.ndarray:
    y_data = self.gpsurrogate.training_y
    y_min = y_data[np.argmin(y_data[:, 0])]

    mean = self.gpsurrogate.mean(x)
    sig2 = self.gpsurrogate.variance(x)
    sig = np.sqrt(sig2)

    grad_EI = None
    if sig.size == 1 and np.abs(sig) > 1e-24:
      dmean_dx = self.gpsurrogate.mean_gradient(x)
      dsig2_dx = self.gpsurrogate.variance_gradient(x)
      dsig_dx = 0.5 * dsig2_dx / sig

      z = (y_min - mean) / sig
      ncdf = norm.cdf(z)
      npdf = norm.pdf(z)
      EI = (y_min - mean) * ncdf + sig * npdf

      dz_dx = -dmean_dx / sig - (y_min - mean) * dsig_dx / sig**2         
      grad_EI = -dmean_dx * ncdf + dsig_dx * npdf
      grad_EI *= -1.
    elif sig.size == 1 and np.abs(sig) <= 1e-24:
      grad_EI = 0.0
    elif sig.size > 1:
      raise NotImplementedError("TODO --- Not implemented yet!")

    return grad_EI
