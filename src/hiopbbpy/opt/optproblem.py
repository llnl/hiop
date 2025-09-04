import numpy as np
from typing import Callable, Dict, List, Union, Tuple
from ..utils.util import check_required_keys
from importlib import import_module

"""
Convert a Scipy optimization problem to an Ipopt problem.
    
Parameters:
    objective (callable): Objective function.
    gradient (callable): Gradient of the objective.
    constraints_list: Scipy-style list of dicts with 'type', 'fun', and optional 'jac'.
    xbounds (list of tuple): Variable bounds [(x0_lb, x0_ub), ...]
    
Returns:
    Ipopt-compatible prob and bounds

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

def _require_cyipopt():
    """Import cyipopt only when actually needed, with a friendly error."""
    try:
        return import_module("cyipopt")
    except ImportError as e:
        raise ImportError(
            "This feature requires the optional dependency 'cyipopt'."
        ) from e

class IpoptProb:
  def __init__(self, objective, gradient, constraint:Union[Dict, List[Dict]], xbounds, solver_options=None):
    self.cons = constraint
    self.eval_f = objective
    self.eval_g  = gradient
    self.xl = [b[0] for b in xbounds]
    self.xu = [b[1] for b in xbounds]
    self.cl = []
    self.cu = []
    self.nvar = len(xbounds)

    self.ipopt_options = solver_options
    self.ipopt_options['sb'] = 'yes'

    if isinstance(self.cons, list):
      # constraints is provided as a list of dict, supported by SLSQP and Ipopt
      for con in self.cons:
        check_required_keys(con,['type', 'fun'])
        if con['type'] == 'eq':
          self.cl.append(0.0)
          self.cu.append(0.0)
        elif con['type'] == 'ineq':
          self.cl.append(0.0)
          self.cu.append(np.inf)
        else:
          raise ValueError(f"Unknown constraint type: {con['type']}")
    elif isinstance(self.cons, dict):
      check_required_keys(self.cons,['cons', 'jac', 'cl', 'cu'])
      # vectorized constraints are provided as a dict, supported by trust-constr and Ipopt
      self.cl = constraint['cl']
      self.cu = constraint['cu']
    else:
      raise ValueError("constraints must be provided as a dict of a list of dict.")
    self.ncon = len(self.cl)

    cyipopt = _require_cyipopt()
    self.nlp = cyipopt.Problem( n=self.nvar,
                                m=self.ncon,
                                problem_obj=self,
                                lb=self.xl,
                                ub=self.xu,
                                cl=self.cl,
                                cu=self.cu
                              )

  def objective(self, x):
    return self.eval_f(x)

  def gradient(self, x):
    return self.eval_g(x)

  def constraints(self, x):
    if isinstance(self.cons, list):
      return np.array([con['fun'](x) for con in self.cons])
    else:
      return self.cons['cons'](x)

  def jacobian(self, x):
    if isinstance(self.cons, list):
      jacs = []
      for con in self.cons:
        if 'jac' in con:
          jacs.append(con['jac'](x))
        else:
          raise ValueError("Jacobian not provided for constraint.")
      return np.vstack(jacs)
    else:
      return self.cons['jac'](x)

  def solve(self, x0, solver_options=None):
    ipopt_options = self.ipopt_options
    if solver_options is not None:
      ipopt_options = solver_options
    if ipopt_options is not None:
      for key, value in ipopt_options.items():
        self.nlp.add_option(key, value)

    # Solve the optimization problem
    return self.nlp.solve(x0)
