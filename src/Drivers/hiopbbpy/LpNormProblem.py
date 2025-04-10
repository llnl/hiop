"""
Implementation of the LPNorm problem class f(x) = || x ||_p

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
import numpy as np
from hiopbbpy.problems.problem import Problem

class LpNormProblem(Problem):
    def __init__(self, ndim, xlimits, p=2.0, constraints=None):
        name = "LpNormProblem"
        super().__init__(ndim, xlimits, name=name, constraints=constraints)
        self.p = p

    def _evaluate(self, x):
        ne, nx = x.shape
        assert nx == self.ndim
        y = np.zeros((ne, 1))
        ytemp = np.linalg.norm(x, ord=self.p, axis=1)
        if len(ytemp.shape) == 1:
            y[:,0] = ytemp[:]
        elif len(ytemp.shape) == 2:
            y[:,:] = ytemp[:,:]
        return y
