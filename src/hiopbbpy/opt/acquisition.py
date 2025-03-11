"""
This file implements different acquisition functions, which are used in Bayesian optimization to decide where to sample next.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np
from ..surrogate_modeling.gp import GaussianProcess

# A base class for acquisition functions
class acquisition(object):
    def __init__(self, gpsurrogate):
        assert isinstance(gpsurrogate, GaussianProcess) # add something here
        self.gpsurrogate = gpsurrogate
    
    # Abstract method to evaluate the acquisition function at x.
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Child class of acquisition should implement method evaluate")


# A subclass of acquisition, implementing the Lower Confidence Bound (LCB) acquisition function.
class LCBacquisition(acquisition):
    def __init__(self, gpsurrogate, beta=3.0):
        super().__init__(gpsurrogate)
        self.beta = beta

    # Method to evaluate the acquisition function at x.
    def evaluate(self, x : np.ndarray) -> np.ndarray:
        mu = self.gpsurrogate.mean(x)
        sig = self.gpsurrogate.variance(x)
        return mu - self.beta * np.sqrt(sig)
