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

# A subclass of acquisition, implementing the Expected improvement (EI) acquisition function.
class EIacquisition(acquisition):
    def __init__(self, gpsurrogate):
        super().__init__(gpsurrogate)

    # Method to evaluate the acquisition function at x.
    def evaluate(self, x : np.ndarray) -> np.ndarray:        
        y_data = self.gpsurrogate.training_y
        y_min = y_data[np.argmin(y_data[:, 0])]

        pred = self.gpsurrogate.mean(x)
        sig = np.sqrt(self.gpsurrogate.variance(x))

        retval = []
        if sig.size == 1 and np.abs(sig) > 1e-12:
            arg0 = (y_min - pred) / sig
            retval = (y_min - pred) * norm.cdf(arg0) + sig * norm.pdf(arg0)
            retval *= -1.
        elif sig.size == 1 and np.abs(sig) <= 1e-12:
            retval = 0.0
        elif sig.size > 1:
            NotImplementedError("TODO --- Not implemented yet!")

        return retval
