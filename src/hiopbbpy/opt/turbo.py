"""
Implementation of the Trust-Region Bayesian Optimization algorithm (TuRBO / TuRBO-m).
Only basic version has been implemented so far. 

A localized alternative to BOAlgorithm: instead of optimizing a global acquisition
function with an NLP solver, TuRBO maintains one or more trust regions and selects
new points by Thompson sampling over a per-region candidate set. This needs only
per-point posterior mean/variance from the surrogate, so it drives 
GaussianProcess backend (smtKRG, MuyGPyS, ...) without modification.

Conventions (consistent with BOAlgorithm):
  * minimization
  * surrogate is a GaussianProcess (train / mean / variance)
  * batch objective evaluations go through obj_evaluator.run(problem.evaluate, X),
    so candidates from all trust regions are pooled into one parallel batch.

"""

import math
import numpy as np
from scipy.stats import qmc
from ..surrogate_modeling.gp import GaussianProcess
from ..problems.problem import Problem
from ..utils.util import Evaluator
from .boalgorithm import BOAlgorithmBase
from hiopbbpy.surrogate_modeling.muygp import muyGP

# Map real-space points into the unit cube defined by xlimits
def to_unit(x, lb, ub):
  return (x - lb) / (ub - lb)

# Map unit-cube points back into real space
def from_unit(u, lb, ub):
  return lb + (ub - lb) * u


# Generate Sobol perturbation candidates inside a trust-region box (unit cube)
def make_tr_candidates(center_u, length, n_candidates, rng, prob_perturb=None):
  dim = center_u.shape[0]
  lb = np.clip(center_u - length / 2.0, 0.0, 1.0)
  ub = np.clip(center_u + length / 2.0, 0.0, 1.0)

  # Sobol balance requires a power-of-two sample count; we draw 2^m and trim
  sob = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(1 << 31)))
  m2 = int(math.ceil(math.log2(max(n_candidates, 2))))
  pert = sob.random_base2(m=m2)[:n_candidates]
  pert = lb + (ub - lb) * pert

  # Perturb only a subset of coordinates per candidate (TuRBO mask)
  # Subject to algorithm design and change 
  if prob_perturb is None:
    prob_perturb = min(20.0 / dim, 1.0)
  mask = rng.random((n_candidates, dim)) <= prob_perturb
  empty = np.where(mask.sum(axis=1) == 0)[0]
  # Guarding against 0 perturbation in degenerate cases
  if len(empty):
    mask[empty, rng.integers(0, dim, size=len(empty))] = True

  X = np.tile(center_u, (n_candidates, 1))
  X[mask] = pert[mask]
  return X

# Posterior marginal statistics (mean, standard deviation) at the candidates.
# Computed once per iteration.
def posterior_stats(surrogate, X_real):
  mu = np.asarray(surrogate.mean(X_real)).reshape(-1)
  var = np.asarray(surrogate.variance(X_real)).reshape(-1)
  sd = np.sqrt(np.clip(var, 0.0, None))
  return mu, sd

# Select a batch by Thompson sampling: draw batch_size independent posterior
# realizations f_tilde = mu + sd * z over the pooled candidates, taking the
# argmin of each (minimization) without replacement. q independent draws give a
# decorrelated batch, rather than the q nearest neighbors of a single draw. 
# Important: this is diagonal (marginal) approximation due to integration with MuyGPy. 
def thompson_batch_select(mu, sd, batch_size, rng):
  n = mu.shape[0]
  q = min(batch_size, n)
  chosen = []
  taken = np.zeros(n, dtype=bool)
  for _ in range(q):
    f_tilde = mu + sd * rng.standard_normal(n)   # one posterior realization
    f_tilde[taken] = np.inf                      # exclude already-selected candidates
    i = int(np.argmin(f_tilde))
    chosen.append(i)
    taken[i] = True
  return np.array(chosen, dtype=int)


# TuRBO trust-region state (minimization). The box has side `length` in the unit
# cube; it doubles after `success_tolerance` consecutive improving batches and
# halves after `failure_tolerance` non-improving ones, restarting once it shrinks
# below `length_min`. Default values follow Eriksson et al. (2019).
#
# Note: register() is called only for a region that won >=1 batch point, so a region
# that stops winning stops updating (matches TuRBO-m's per-region update; restart of
# a collapsed region is handled in the main loop).
class TrustRegionState:
  def __init__(self, dim, batch_size, center_u, best_value, best_x,
               length_init=0.8, length_min=0.5 ** 7, length_max=1.6,
               success_tolerance=10, failure_tolerance=None):
    self.dim = dim
    self.batch_size = batch_size
    self.center_u = np.asarray(center_u, dtype=float)   # center in the unit cube
    self.best_value = float(best_value)
    self.best_x = np.asarray(best_x, dtype=float)       # center in real space
    self.length = float(length_init)
    self.length_min = float(length_min)
    self.length_max = float(length_max)
    self.success_counter = 0
    self.failure_counter = 0
    self.success_tolerance = success_tolerance  # grow the box after this many consecutive improving batches
    # Higher dim / smaller batch -> more failed batches allowed before shrinking:
    # a q-point batch is q chances to improve, so tolerate ~dim/q non-improving batches.
    if failure_tolerance is None:
      failure_tolerance = math.ceil(max(4.0 / batch_size, dim / batch_size))
    self.failure_tolerance = failure_tolerance
    self.restart_triggered = False

  # Update counters and side length given the best new value won by this region
  # A "success" = the region's best won point beats its incumbent by a relative
  # margin (not just any decrease, so noise doesn't count). Judged on the region's
  # single best won point, not each point in the batch.
  def register(self, y_min, x_at_min, u_at_min, tol=1e-3):
    improved = y_min < self.best_value - max(tol * abs(self.best_value), 1e-12) # 1e-12 is a safeguard when y_min is very small
    if improved:
      self.success_counter += 1
      self.failure_counter = 0   # any success resets the failure streak
      self.best_value = y_min
      self.best_x = np.asarray(x_at_min, dtype=float)
      self.center_u = np.asarray(u_at_min, dtype=float)
    else:
      self.success_counter = 0
      self.failure_counter += 1

    if self.success_counter == self.success_tolerance:
    # consistent improvement -> box may be too small, expand to move faster
      self.length = min(2.0 * self.length, self.length_max)
      self.success_counter = 0
    elif self.failure_counter == self.failure_tolerance:
      self.length /= 2.0
      self.failure_counter = 0

    if self.length < self.length_min: # box collapsed; main loop will reinitialize this region
      self.restart_triggered = True


# A subclass of BOAlgorithmBase implementing the TuRBO / TuRBO-m workflow
# options (dict, all optional):
#   n_trust_regions   regions; 1 = TuRBO-1, >1 = TuRBO-m         default=1
#   local_gp          one global GP (False) vs one GP per region default=False
#   batch_size        q points per iter, evaluated in parallel   default=1
#   bo_maxiter        number of BO iterations                    default from base
#   n_candidates      Sobol candidates scored per region         default=min(5000,max(2000,200*dim))
#   length_init       initial box side (unit cube)               default=0.8
#   length_max        cap on box growth                          default=1.6
#   length_min        restart threshold; box collapses below     default=0.5**7
#   success_tolerance improving batches before box doubles       default=10
#   obj_evaluator     Evaluator (serial) or MPIEvaluator         default from base
#   seed              RNG seed for reproducibility               default=0
#   log_level         logging verbosity                          default="INFO"

class TurboAlgorithm(BOAlgorithmBase):
  def __init__(self, prob:Problem, surrogate_factory, xtrain, ytrain,
               options = None):
    super().__init__()
    options = options or {}
    assert callable(surrogate_factory), "surrogate_factory must be callable() -> GaussianProcess"
    # surrogate_factory: callable() -> GaussianProcess. A factory (not a built model)
    # because TuRBO-m needs a fresh surrogate per region and on restart.1

    self.prob = prob
    self.surrogate_factory = surrogate_factory
    self.setTrainingData(np.array(xtrain, dtype=float), np.array(ytrain, dtype=float))

    self.xlimits = np.asarray(prob.xlimits, dtype=float)
    self.lb = self.xlimits[:, 0]
    self.ub = self.xlimits[:, 1]
    self.dim = prob.ndim

    logger_level = options.get('log_level', "INFO")
    self.logger.setlevel(logger_level)

    self.bo_maxiter = options.get('bo_maxiter', self.bo_maxiter)
    assert self.bo_maxiter > 0, f"Invalid bo_maxiter: {self.bo_maxiter}"

    batch_size = int(options.get('batch_size', 1))
    assert batch_size > 0, f"batch_size {batch_size} is not strictly positive"
    self.batch_size = batch_size

    self.n_trust_regions = options.get('n_trust_regions', 1)
    assert self.n_trust_regions > 0, f"Invalid n_trust_regions: {self.n_trust_regions}"

    # default scales candidates with dimension, capped to [2000, 5000]
    self.n_candidates = options.get('n_candidates', min(5000, max(2000, 200 * self.dim)))
    self.local_gp = options.get('local_gp', False)  # False: one global GP; True: one GP per region
    self.rng = np.random.default_rng(options.get('seed', 0))

    self.length_init = options.get('length_init', 0.8)
    self.length_min = options.get('length_min', 0.5 ** 7)
    self.length_max = options.get('length_max', 1.6)
    self.success_tolerance = options.get('success_tolerance', 10)

    self.obj_evaluator = options.get('obj_evaluator', self.obj_evaluator)
    assert isinstance(self.obj_evaluator, Evaluator)

    self.regions = []
    self.surrogates = []       # one surrogate (global mode) or one per region (local mode)

    self.logger.info(f"Problem name: {prob.name}")
    self.logger.info(f"Max BO iter: {self.bo_maxiter}")
    self.logger.info(f"Number of trust regions: {self.n_trust_regions}")
    self.logger.info(f"Batch size: {self.batch_size}")
    self.logger.info(f"Candidates per region: {self.n_candidates}")
    self.logger.info(f"Local GP per region: {self.local_gp}")
    self.logger.info(f"Initial training set: {xtrain.shape[0]} samples, {xtrain.shape[1]} dimensions")
    self.logger.info(f"Logger level: {logger_level}")

  # Method to (re)train the surrogate(s): one global model, or one per region
  def _train_surrogates(self):
    if self.local_gp:
      self.surrogates = []
      for r in range(len(self.regions)):
        gp = self.surrogate_factory()
        Xr, Yr = self._region_data(r)
        if Xr.shape[0] < 2:                        # too sparse: fall back to all data
          Xr, Yr = self.xtrain, self.ytrain
        gp.train(Xr, Yr)
        self.surrogates.append(gp)
    else:
      gp = self.surrogate_factory()
      gp.train(self.xtrain, self.ytrain)
      self.surrogates = [gp]

  # Return the surrogate that serves region r
  def _surrogate_for(self, r):
    return self.surrogates[r] if self.local_gp else self.surrogates[0]

  # Return the training data assigned to region r
  def _region_data(self, r):
    idx = self._region_idx[r]
    if len(idx) == 0:
      return np.empty((0, self.dim)), np.empty((0, 1))
    return self.xtrain[idx], self.ytrain[idx]

  # Seed the trust regions at the best distinct observed points
  def _init_regions(self):
    # hashable, noise-tolerant coordinate key so a set can de-duplicate repeated points
    order = np.argsort(self.ytrain.reshape(-1)) # sorting
    seeds, seen = [], set()
    for i in order: # from best to worst
      key = tuple(np.round(self.xtrain[i], 8))
      if key not in seen:
        seeds.append(int(i))
        seen.add(key)
      if len(seeds) == self.n_trust_regions:
        break
    while len(seeds) < self.n_trust_regions:       # pad with random points if not enough distinct points
      seeds.append(int(self.rng.integers(self.xtrain.shape[0]))) # 

    self.regions = []
    self._region_idx = [list() for _ in range(self.n_trust_regions)]
    for r, i in enumerate(seeds):
      x0 = self.xtrain[i]
      u0 = to_unit(x0, self.lb, self.ub)
      self.regions.append(TrustRegionState(
          self.dim, self.batch_size, u0, float(self.ytrain[i, 0]), x0,
          length_init=self.length_init, length_min=self.length_min,
          length_max=self.length_max, success_tolerance=self.success_tolerance))
      self._region_idx[r].append(int(i))

  # Reinitialize a collapsed region: draw a new random center anywhere in the domain (evaluated in parallel),
  # reset the box to full size, and clear the region's point set.
  def _restart_region(self, r):
    x0 = self.prob.sample(1)
    y0 = np.asarray(self.obj_evaluator.run(self.prob.evaluate, x0)).reshape(1, 1)
    self.xtrain = np.vstack([self.xtrain, x0])
    self.ytrain = np.vstack([self.ytrain, y0])
    self._record_history(x0, y0)
    i = self.xtrain.shape[0] - 1
    u0 = to_unit(x0[0], self.lb, self.ub)
    self.regions[r] = TrustRegionState(
        self.dim, self.batch_size, u0, float(y0[0, 0]), x0[0],
        length_init=self.length_init, length_min=self.length_min,
        length_max=self.length_max, success_tolerance=self.success_tolerance)
    self._region_idx[r] = [int(i)]
    self.logger.info(f"Region {r} restarted (trust-region length collapsed)")

  # Append evaluated points to the optimization history
  def _record_history(self, X, Y):
    for j in range(X.shape[0]):
      self.x_hist.append(X[j].flatten())
      self.y_hist.append(Y[j].flatten())

  # Method to perform Trust-Region Bayesian optimization
  def optimize(self):
    self.x_hist = []
    self.y_hist = []
    self._init_regions()

    best0 = float(np.min(self.ytrain))
    self.logger.iterations(f"Best objective from {self.ytrain.shape[0]} initial samples: {best0:.4e} ")

    for it in range(1, self.bo_maxiter + 1):
      self.logger.critical(f"*****************************")
      self.logger.critical(f"Iteration {it}/{self.bo_maxiter}")

      # Train the surrogate model(s)
      self._train_surrogates()

      # Generate candidates for each region and gather posterior statistics.
      # The GP posterior is evaluated once per candidate (mean, sd); the random
      # Thompson draws below reuse these, so mean()/variance() are not recomputed.
      pooled_u, pooled_x, pooled_mu, pooled_sd, pooled_region = [], [], [], [], []
      for r, st in enumerate(self.regions):
        U = make_tr_candidates(st.center_u, st.length, self.n_candidates, self.rng)
        Xr = from_unit(U, self.lb, self.ub)
        mu, sd = posterior_stats(self._surrogate_for(r), Xr)
        pooled_u.append(U)
        pooled_x.append(Xr)
        pooled_mu.append(mu)
        pooled_sd.append(sd)
        pooled_region.append(np.full(U.shape[0], r, dtype=int))

      U_all = np.vstack(pooled_u)
      X_all = np.vstack(pooled_x)
      MU_all = np.concatenate(pooled_mu)
      SD_all = np.concatenate(pooled_sd)
      R_all = np.concatenate(pooled_region)

      # Allocate the batch by Thompson sampling across the union of all regions:
      # q independent posterior draws, argmin of each (the implicit multi-armed
      # bandit that distributes the batch over trust regions).
      sel = thompson_batch_select(MU_all, SD_all, self.batch_size, self.rng)
      X_next = X_all[sel]
      U_next = U_all[sel]
      R_next = R_all[sel]

      # Evaluate the whole batch in parallel (the MPI evaluation-manager seam)
      Y_next = np.asarray(self.obj_evaluator.run(self.prob.evaluate, X_next)).reshape(-1, 1)

      # Update the global training set and history
      base = self.xtrain.shape[0]
      self.xtrain = np.vstack([self.xtrain, X_next])
      self.ytrain = np.vstack([self.ytrain, Y_next])
      self._record_history(X_next, Y_next)

      # Update each region from the points it won this iteration
      for r, st in enumerate(self.regions):
        mask = (R_next == r)
        if not np.any(mask):
          continue
        yr = Y_next[mask].reshape(-1)
        k = int(np.argmin(yr))
        where = np.where(mask)[0]
        self._region_idx[r].extend((base + where).tolist())
        st.register(float(yr[k]), X_next[where[k]], U_next[where[k]])
        if st.restart_triggered:
          self._restart_region(r)

      curr_best_y = float(np.min(self.ytrain))
      self.logger.iterations(f"Current best objective: {curr_best_y:.4e} ")
      self.logger.scalars(f"Training set size is now {self.xtrain.shape[0]}")
      self.logger.scalars("Trust-region lengths: [" +
                          ", ".join(f"{s.length:.2e}" for s in self.regions) + "]")

    # Save the optimal results and all the training data
    self.idx_opt = int(np.argmin(self.y_hist))
    self.x_opt = np.array(self.x_hist[self.idx_opt])
    self.y_opt = np.array(self.y_hist[self.idx_opt])
    self.setTrainingData(self.xtrain, self.ytrain)

    self.logger.critical("===================================")
    self.logger.critical("Trust-Region Bayesian Optimization completed")
    self.logger.critical(f"Total evaluations for BO iterations: {len(self.y_hist)}")
    self.logger.critical(f"Best value: {float(self.y_opt.reshape(-1)[0])}")
    self.logger.critical("===================================")
    return self.x_opt, self.y_opt
