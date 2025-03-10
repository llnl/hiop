import numpy as np
from .boalgorithm import BOAlgorithmBase
from ..surrogate_modeling.krg import smtKRG
from ..problems.problem import Problem

try:
    from smt.applications.ego import EGO as SMTEGO
    from smt.surrogate_models import KRG
except ImportError:
    ImportError("error importing smt")


# TODO: determine why xdoe is passed to the smt EGO
#       object.... if the surrogate is already trained
#       on doe data then maybe this isn't necessary

# TODO: if we set training data do xdoe and the associated SMTEGO object
#         need to be updated.
class smtBOAlgorithm(BOAlgorithmBase):
    def __init__(self, surrogate, xdoe, n_iter=10, n_start=10, acquisition_type="EI", save_opt_history=True):
        assert isinstance(surrogate, smtKRG)
        assert acquisition_type in ["EI", "SBO", "LCB"]
        super().__init__()
        self.surrogate = surrogate
        self.xdoe = xdoe
        self.n_iter = n_iter
        self.n_start = n_start
        self.ego = None
        self.save_opt_history = save_opt_history
        self.setAcquisitionType(acquisition_type)
    def setupBO(self):
        self.ego = SMTEGO(n_iter = self.n_iter,
                          criterion = self.acquisition_type,
                          xdoe = self.xdoe,
                          surrogate = self.surrogate.surrogatesmt,
                          n_start = self.n_start)
    def setTrainingData(self, xdata, ydata):
        self.ego.gpr.set_training_values(xdata, ydata)
        self.ego.gpr.train()
    def optimize(self, fun):
        if isinstance(fun, Problem):
            callback = fun.evaluate
        else:
            callback = fun
        if self.save_opt_history:
            self.x_opt, self.y_opt, self.idx_opt, self.x_hist, self.y_hist = self.ego.optimize(callback)
        else:
            self.x_opt, self.y_opt, _, _, _ = self.ego.optimize(callback)
