"""
Implementation of the abstract optimization problem class

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
import numpy as np
import collections.abc
from numpy.random import uniform
from scipy.stats import qmc

# define the general optimization problem class
class Problem:
    def __init__(self, ndim, xlimits, name=" ", constraints=[]):
        self.ndim = ndim
        self.xlimits = xlimits
        assert self.xlimits.shape[0] == ndim            
        assert isinstance(name, str)
        assert isinstance(constraints, collections.abc.Sequence)
        self.name = name
        self.sampler = qmc.LatinHypercube(ndim)
        self.constraints = constraints
            
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

        # uniform
        # xsample = np.zeros((nsample, self.ndim))
        # for j in range(self.ndim):
        #    xsample[:, j] = uniform(self.xlimits[j][0], self.xlimits[j][1], size=nsample)

        # from predefined sampler
        xsample = self.sampler.random(nsample)
        xsample = self.xlimits[:,0] + (self.xlimits[:,1] - self.xlimits[:,0]) * xsample

        return xsample

    def set_constraints(self, constraints):
        self.constraints = constraints





