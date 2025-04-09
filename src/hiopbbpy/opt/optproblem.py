import numpy as np
import cyipopt

"""
Convert a Scipy optimization problem to an Ipopt problem.
    
Parameters:
    objective (callable): Objective function.
    gradient (callable): Gradient of the objective.
    constraints_list: Scipy-styple list of dicts with 'type', 'fun', and optional 'jac'.
    xbounds (list of tuple): Variable bounds [(x0_lb, x0_ub), ...]
    
Returns:
    Ipopt-compatible prob and bounds

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
class IpoptProb:
    def __init__(self, objective, gradient, constraints_list, xbounds, solver_options=None):
        self.constraints_list = constraints_list
        self.eval_f = objective
        self.eval_g  = gradient
        self.xl = [b[0] for b in xbounds]
        self.xu = [b[1] for b in xbounds]
        self.cl = []
        self.cu = []
        self.nvar = len(xbounds)
        self.ncon = len(self.constraints_list)
        self.ipopt_options = solver_options

        for con in constraints_list:
            if con['type'] == 'eq':
                self.cl.append(0.0)
                self.cu.append(0.0)
            elif con['type'] == 'ineq':
                self.cl.append(0.0)
                self.cu.append(np.inf)
            else:
                raise ValueError(f"Unknown constraint type: {con['type']}")

        self.nlp = cyipopt.Problem(
            n=self.nvar,
            m=self.ncon,
            problem_obj=self,
            lb=self.xl,
            ub=self.xu,
            cl=self.cl,
            cu=self.xu
        )

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

    def solve(self, x0, solver_options=None):
        ipopt_options = self.ipopt_options
        if solver_options is not None:
            ipopt_options = solver_options
        if ipopt_options is not None:
            for key, value in ipopt_options.items():
                self.nlp.add_option(key, value)

        # Solve the optimization problem
        return self.nlp.solve(x0)