"""
This is a base class for Gaussian Process (GP) models.
It defines methods for computing the mean, covariance, and variance of the GP.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np

class GaussianProcess:
  def __init__(self, ndim, xlimits=None):
    self.ndim = ndim
    self.xlimits = xlimits
    self.training_x = []
    self.training_y = []
    self.trained = False
  
  # Abstract method for computing the mean of the GP at a given input x
  def mean(self, x: np.ndarray) -> np.ndarray:
    """
    evaluation of the GP mean

    Parameters
    ---------
    x : ndarray[n, nx]

    Returns
    -------
    ndarray[n, 1]
       Mean of GP at x
    """
    raise NotImplementedError("Child class of GaussianProcess should implement method mean")
  
  # Abstract method for computing the covariance of the GP at a given input x
  def covariance(self, x: np.ndarray) -> np.ndarray:
    """
    evaluation of the GP covariance

    Parameters
    ---------
    x: ndarray[n, nx]

    Returns
    -------
    ndarray[n, n]
       Covariance of GP at w.r.t. x
    """
    raise NotImplementedError("Child class of GaussianProcess should implement method covariance")

  # Abstract method for computing the variance of the GP at a given input x
  def variance(self, x: np.ndarray) -> np.ndarray:
    """
    evaluation of the GP variance

    Parameters
    ---------
    x: ndarray[n, nx]

    Returns
    ------
    ndarray[n, 1]
       Variance of GP at x
    """
    y = np.ndarray((self.ndim, 1))
    for i in range(x.shape[1]):
      y[i][0] = covariance(np.atleast_2d(x[i,:]))[0][0]
    return y

  # Retrieves the bounds of the input space if xlimits is provided.
  def get_bounds(self):
    if self.xlimits is None:
      return None
    else:
      return [(self.xlimits[i][0], self.xlimits[i][1]) for i in range(self.ndim)]

  # Abstract method for training the GP
  def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    train the GP model

    Parameters
    ---------
    x : ndarray[n, nx]
    y : ndarray[n, 1]

    """
    NotImplementedError("Child class of GaussianProcess should implement method train")

  # Abstract method for computing the gradient of the mean of the GP at a given input x
  def mean_gradient(self, x: np.ndarray) -> np.ndarray:
    """
    evaluation of the gradien of GP mean

    Parameters
    ---------
    x : ndarray[n, nx]

    Returns
    -------
    ndarray[n, 1]
       Gradient of Mean of GP at x
    """
    raise NotImplementedError("Child class of GaussianProcess should implement method mean_gradient")

  # Abstract method for computing the gradient of variance of the GP at a given input x
  def variance_gradient(self, x: np.ndarray) -> np.ndarray:
    """
    evaluation of the gradient of GP variance

    Parameters
    ---------
    x: ndarray[n, nx]

    Returns
    -------
    ndarray[n, n]
       Gradient of variance of GP at w.r.t. x
    """
    raise NotImplementedError("Child class of GaussianProcess should implement method variance_gradient")
