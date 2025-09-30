from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from ..problems.problem import Problem 
from .optproblem import IpoptProb
import warnings

# Moved the minimizer function to be used simultaneously by the BO algorithm and the BnB algorithm.

# Find the minimum of the input objective `fun`, using the minimize function from SciPy. 
def minimizer(fun, x0, method, bounds, constraints, solver_options):
    if method == "SLSQP":
        if 'grad' in fun:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, jac=fun['grad'], constraints=constraints, options=solver_options)
        else:
            y = minimize(fun['obj'], x0, method=method, bounds=bounds, constraints=constraints, options=solver_options)
        success = y.success
        if not success:
            print(y.message)
        xopt = y.x
        yopt = y.fun
    elif method == "trust-constr":
        nonlinear_constraint = NonlinearConstraint(constraints['cons'], constraints['cl'], constraints['cu'], jac=constraints['jac'])
        y = minimize(fun['obj'], x0, method=method, bounds=bounds, constraints=[nonlinear_constraint], options=solver_options)
        success = y.success
        if not success:
            print(y.message)
        xopt = y.x
        yopt = y.fun
    else:
        ipopt_prob = IpoptProb(fun['obj'], fun['grad'], constraints, bounds, solver_options)
        sol, info = ipopt_prob.solve(x0)

        status = info.get('status', -999)
        msg = info.get('status_msg', b'unknown error')
        if status == 0:
            # ipopt returns 0 as success
            success = True
        else:
            warnings.warn(f"Ipopt failed to solve the problem. Status msg: {msg}")
            success = False

        yopt = info['obj_val']
        xopt = sol

    return xopt, yopt, success
