"""
Implementation of the abstract optimization problem class

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
import numpy as np
from numpy.random import uniform

# define the general optimization problem class
class Problem:
    def __init__(self, ndim, xlimits, name=None):
        self.ndim = ndim
        self.xlimits = xlimits
        assert self.xlimits.shape[0] == ndim            
        if name is None:
            self.name = " "
        else:
            self.name = name
            
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        problem evaluation y = f(x) of
        a scalar valued function f

        Parameters
        ---------
        x : ndarray[n, nx]

        Returns
        -------
        ndarray[n, 1]
           Function values
        """
        raise NotImplementedError("Child class of hiopProblem should implement method _evaluate")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        problem callback y = f(x) of
        the scalar valued function  f

        Parameters
        ---------
        x : ndarray[n, nx]

        Returns
        -------
        ndarray[n, 1]
           Function values (cast to reals)
        """
        y = np.ndarray((x.shape[0], 1))
        y[:,:] = self._evaluate(x)
        return y

    def sample(self, nsample: int) -> np.ndarray:
        """
        generate nsample samples from domain defined
        by xlimits

        Parameters
        -------
        nsample : int

        Returns
        -------
        ndarray[nsample, nx]
           Samples from domain defined by xlimits
        """
        xsample = np.zeros((nsample, self.ndim))
        for j in range(self.ndim):
            xsample[:, j] = uniform(self.xlimits[j][0], self.xlimits[j][1], size=nsample)
        return xsample





