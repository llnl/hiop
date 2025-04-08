import numpy as np

"""
Convert a Scipy optimization problem to an Ipopt problem.
    
Parameters:
    objective (callable): Objective function.
    gradient (callable): Gradient of the objective.
    constraints_list: Scipy-styple list of dicts with 'type', 'fun', and optional 'jac'.
    xbounds (list of tuple): Variable bounds [(x0_lb, x0_ub), ...]
    
Returns:
    Ipopt-compatible prob and bounds
"""
class IpoptProbFromScipy:
    def __init__(self, objective, gradient, constraints_list, xbounds):
        self.constraints_list = constraints_list
        self.eval_f = objective
        self.eval_g  = gradient
        self.xl = [b[0] for b in xbounds]
        self.xu = [b[1] for b in xbounds]
        self.cl = []
        self.cu = []
        self.nvar = len(xbounds)
        self.ncon = len(self.constraints_list)

        for con in constraints_list:
            if con['type'] == 'eq':
                self.cl.append(0.0)
                self.cu.append(0.0)
            elif con['type'] == 'ineq':
                self.cl.append(0.0)
                self.cu.append(np.inf)
            else:
                raise ValueError(f"Unknown constraint type: {con['type']}")

    def objective(self, x):
        return self.eval_f(x)

    def gradient(self, x):
        return self.eval_g(x)

    def constraints(self, x):
        return np.array([con['fun'](x) for con in self.constraints_list])

    def jacobian(self, x):
        jacs = []
        for con in self.constraints_list:
            if 'jac' in con:
                jacs.append(con['jac'](x))
            else:
                raise ValueError("Jacobian not provided for constraint.")
        return np.vstack(jacs)
