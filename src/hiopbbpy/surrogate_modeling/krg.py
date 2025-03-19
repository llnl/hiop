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
    def __init__(self, theta, xlimits, ndim, corr="pow_exp", noise0=None, random_state=None):
        super().__init__(ndim, xlimits)
        if random_state is None:
            random_state = 42
        design_space = DesignSpace(xlimits, random_state=random_state)
        if noise0 is None:
            self.surrogatesmt = KRG(design_space=design_space,
                             print_global=False,
                             eval_noise=False,
                             corr=corr)
        else:
            self.surrogatesmt = KRG(design_space=design_space,
                             print_global=False,
                             noise0=noise0,
                             eval_noise=False,
                             corr=corr)
        self.trained = False

    def mean(self, x):
        if not self.trained:
            ValueError("must train kriging model before utilizing it to predict mean or variances")
        return self.surrogatesmt.predict_values(x)

    def variance(self, x):
        if not self.trained:
            ValueError("must train kriging model before utilizing it to predict mean or variances")
        return self.surrogatesmt.predict_variances(x)

    def train(self, x, y):
        self.training_x = x
        self.training_y = y
        self.surrogatesmt.set_training_values(x, y)
        self.surrogatesmt.train()
        self.trained = True





