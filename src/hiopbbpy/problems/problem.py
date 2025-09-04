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
    self.constraints = constraints          # a dict or a list of dict for constraints in scipy format
    self.n_con = 0                          # number of constraints
    if isinstance(constraints, dict):
      # vectorized case
      self.n_con = np.size(constraints['cl'], 0)
    elif isinstance(constraints, list):
      # each dict = one scalar constraint
      self.n_con = len(constraints)
        
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

  def con_evaluate(self, x: np.ndarray) -> np.ndarray:
    """
    problem callback to evaluate constraint

    Parameters
    ---------
    x : ndarray[n, nx]

    Returns
    -------
    ndarray[n, m]
       Function values (cast to reals)
    """
  
    if not self.constraints:
      return None
    
    n_samples = x.shape[0]
    y = np.zeros((n_samples, self.n_con))

    if isinstance(self.constraints, dict):
      # Vectorized constraints in one dict, returning all constraints at once
      con_eval = np.vstack([self.constraints['cons'](xi) for xi in x])
      y[:, :] = np.asarray(con_eval, dtype=float)
    else:
      # Multiple scalar constraints: list of dicts
      for j, con in enumerate(self.constraints):
        con_eval = np.vstack([con['fun'](xi) for xi in x])
        y[:, j] = np.asarray(con_eval, dtype=float).reshape(-1)

    return y

  def if_feasible(self, x: np.ndarray, y: np.ndarray = None, tol: float = 1e-8) -> np.ndarray:
    """
    Check feasibility of constraints for given x.

    Parameters
    ----------
    x : ndarray[n, nx]
        Batch of n samples, each of dimension nx.
    y : ndarray[n, m], optional
        Precomputed constraint values (from con_evaluate).
        If None, will be computed internally.
    tol : float
        Numerical tolerance for equality/inequality checks.

    Returns
    -------
    ndarray[n]
        Boolean feasibility array (True if feasible, False otherwise).
    """
    
    # Points are feasible if there are no constraints
    if not self.constraints:
      return np.array([True for _ in range(x.shape[0])]) 
    
    # Evaluate constraints if not provided
    if y is None:
      y = self.con_evaluate(x)

    # Collect bounds
    if isinstance(self.constraints, dict):
      cl = np.atleast_1d(self.constraints["cl"])
      cu = np.atleast_1d(self.constraints["cu"])
    else:
      # List of dicts: each has a 'type' field ('eq' or 'ineq')
      cl_list, cu_list = [], []
      for con in self.constraints:
        if con['type'] == 'eq':
          # equality: g(x) = 0
          cl_list.append(0.0)
          cu_list.append(0.0)
        elif con['type'] == 'ineq':
          # inequality: g(x) >= 0
          cl_list.append(0.0)
          cu_list.append(np.inf)
        else:
          raise ValueError(f"Unknown constraint type: {con['type']}")
      cl = np.array(cl_list, dtype=float)
      cu = np.array(cu_list, dtype=float)

    # Feasibility check: cl - tol <= y <= cu + tol
    feas_matrix = (y >= cl - tol) & (y <= cu + tol)
    
    # Each sample is feasible only if all constraints are satisfied
    return feas_matrix.all(axis=1)

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

    xsample = self.sampler.random(nsample)
    xsample = self.xlimits[:,0] + (self.xlimits[:,1] - self.xlimits[:,0]) * xsample

    return xsample

  def set_constraints(self, constraints):
    self.constraints = constraints
    if isinstance(constraints, dict):
      # vectorized case
      self.n_con = np.size(constraints['cl'], 0)
    elif isinstance(constraints, list):
      # each dict = one scalar constraint
      self.n_con = len(constraints)
    else:
      self.n_con = 0





