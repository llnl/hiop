"""
A GaussianProcess surrogate backed by MuyGPyS: predicts each point from its
nearest neighbors (nn), avoiding the dense n x n solve. Same train/mean/variance
interface as smtKRG.

Matern smoothness (smoothness) and length_scale are fixed; only sigma^2 is optimized (via
optimize_scale) since it sets the variance magnitude Thompson sampling uses.
Optimizing the kernel hyperparameters via MuyGPyS.optimize is the natural
next step.

"""
import numpy as np
from .gp import GaussianProcess

# MuyGPyS >= 0.9 API
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import Parameter, AnalyticScale
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper



# nn refers to nearest neighbor in this file
def _nn_indices(result):
  # NN_Wrapper.get_nns / get_batch_nns return either an index array or an
  # (indices, distances) tuple depending on version. this function normalizes both to just the indices.
  return result[0] if isinstance(result, tuple) else result


class muyGP(GaussianProcess):
  def __init__(self, ndim, xlimits, nn_count=None, smoothness=1.5,
               length_scale=0.2, noise=1e-5, nn_method="exact", normalize=True):
    super().__init__(ndim, xlimits)
    self.nn_count = nn_count if nn_count is not None else min(50, max(2 * ndim, 15))
    self.smoothness = smoothness
    self.length_scale = length_scale
    self.noise = noise
    self.nn_method = nn_method   # nearest neighbor method
    self.normalize = normalize
    self.muygps = None
    self.nbrs = None

  # Scale inputs to the unit cube (stabilizes the isotropic kernel across dims)
  def _unit(self, x):
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if not self.normalize:
      return X
    lb, ub = self.xlimits[:, 0], self.xlimits[:, 1]
    return (X - lb) / (ub - lb)

  def train(self, x, y):
    self.train_x = self._unit(x)
    self.train_y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = self.train_x.shape[0]
    nn = int(min(self.nn_count, max(2, n - 1)))   # can't request more NN than pts-1

    # Build the MuyGPS model. Matern smoothness/length_scale are fixed;
    # the scale (sigma^2) is optimized analytically below.
    self.muygps = MuyGPS(
        kernel=Matern(
            smoothness=Parameter(self.smoothness),
            deformation=Isotropy(metric=l2, length_scale=Parameter(self.length_scale)),
        ),
        noise=HomoscedasticNoise(self.noise),
        scale=AnalyticScale(),
    )

    # Nearest-neighbor index over the training features
    self.nbrs = NN_Wrapper(self.train_x, nn, nn_method=self.nn_method)

    # Set sigma^2 via MuyGPyS's analytic scale estimator (AnalyticScale)
    batch_indices = np.arange(n)
    batch_nn_indices = _nn_indices(self.nbrs.get_batch_nns(batch_indices))
    _, pairwise, _, batch_nn_targets = self.muygps.make_train_tensors(
        batch_indices, batch_nn_indices, self.train_x, self.train_y)
    self.muygps = self.muygps.optimize_scale(pairwise, batch_nn_targets)

    # Kernel hyperparameters (smoothness, length_scale) are fixed. To learn them,
    # run leave-one-out optimization via MuyGPyS.optimize on the train tensors and
    # reassign self.muygps with the optimized model.

    self.trained = True

  def mean(self, x):
    return self._predict(x)[0]

  def variance(self, x):
    return self._predict(x)[1]

  def _predict(self, x):
    if not self.trained:
      raise ValueError("must train muyGP before predicting mean or variance")
    Xt = self._unit(x)
    test_count = Xt.shape[0]
    batch_indices = np.arange(test_count)
    nn_indices = _nn_indices(self.nbrs.get_nns(Xt))                          
    crosswise, pairwise, batch_nn_targets = self.muygps.make_predict_tensors(
        batch_indices, nn_indices, Xt, self.train_x, self.train_y)
    Kcross = self.muygps.kernel(crosswise)
    Kin = self.muygps.kernel(pairwise)
    mu = np.asarray(self.muygps.posterior_mean(Kin, Kcross, batch_nn_targets)).reshape(-1, 1)
    var = np.asarray(self.muygps.posterior_variance(Kin, Kcross)).reshape(-1, 1)
    return mu, np.clip(var, 1e-12, None)
