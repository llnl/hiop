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
        #return np.atleast_2d(np.diag(covariance(x))).T

    # Retrieves the bounds of the input space if xlimits is provided.
    def get_bounds(self):
        if self.xlimits is None:
            return None
        else:
            return [(self.xlimits[i][0], self.xlimits[i][1]) for i in range(self.ndim)]

