"""
Implementation of the Bayesian Optimization Algorithms

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from .optproblem import IpoptProb
#import time

class minimizer_wrapper:
  def __init__(self, fun, method, bounds, constraints, solver_options):
    self.fun = fun
    self.method = method
    self.bounds = bounds
    self.constraints = constraints
    self.solver_options = solver_options
  # Find the minimum of the input objective `fun`, using the minimize function from SciPy. 
  def minimizer_callback(self, x0s):
    output = []
    msg = ""
    for x0 in x0s:
      #time.sleep(2.5)
      if self.method in ["SLSQP", "trust-constr"]:
        if 'tol' in self.solver_options:
          tol = self.solver_options['tol']
          solver_options = self.solver_options.copy()
          del solver_options['tol']
        else:
          solver_options = self.solver_options.copy()
          tol = None
        if 'grad' in self.fun:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, jac=self.fun['grad'], constraints=self.constraints, tol=tol, options=solver_options)
        else:
          y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, constraints=self.constraints, tol=tol, options=solver_options)
        success = y.success
        if not success:
          msg = y.message
        xopt = y.x
        yopt = y.fun
      elif self.method == "trust-constr":
        #nonlinear_constraint = NonlinearConstraint(self.constraints['cons'], self.constraints['cl'], self.constraints['cu'], jac=self.constraints['jac'])
        bounds = Bounds(lb= self.bounds[0], ub=self.bounds[1], keep_feasible=True) 
        y = minimize(self.fun['obj'], x0, method=self.method, bounds=self.bounds, tol=tol,options=self.solver_options)#constraints=[nonlinear_constraint], options=self.solver_options)
        success = y.success
        if not success:
          msg = y.message
        xopt = y.x
        yopt = y.fun
      else:
        ipopt_prob = IpoptProb(self.fun['obj'], self.fun['grad'], self.constraints, self.bounds, self.solver_options)
        sol, info = ipopt_prob.solve(x0)
    
        status = info.get('status', -999)
        msg = info.get('status_msg', b'unknown error')
        if status == 0:
          # ipopt returns 0 as success
          success = True
        else:
          msg = f"Ipopt failed to solve the problem. Status msg: {msg}"
          success = False
    
        yopt = info['obj_val']
        xopt = sol
      output.append([xopt, yopt, success, msg])
    return output
