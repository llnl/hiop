"""
Implementation of the Branch and Bound (BnB) algorithm for Optimization of Acquisition Functions in Bayesian optimization.
This module currently supports Convex Kernel Functions (Matern and SE) and Acquisition Functions EI and LCB.

Authors:    Natalia Rodriguez Figueroa <rodriguezfig1@llnl.gov>
"""

import numpy as np
import heapq
import cvxpy as cp
from scipy.stats import norm
from scipy.stats import uniform
from hiopbbpy.opt.acquisition import EIacquisition, LCBacquisition
from ..surrogate_modeling.gp import GaussianProcess
from hiopbbpy.opt.minimizer import minimizer

# A base class defining a general framework for Branch and Bound Optimization for Acquisition Functions
class BnBAlgorithmBase:
    
    def __init__(self):
        self.BnBNode = BnBNode
        self.BnB_LBmethod = "IPOPT"
        self.acquisition_type = "LCB"  # or "EI"
        self.kernel_spec = None
        self.kernel_func = None
        self.theta = 1.0  # length scale for the kernel function (refer to KRG for details)
        self.beta = 1.0  # exploration-exploitation trade-off parameter for LCB
        self.y_min = None # current best minimum of the objective function 
        self.epsilon_gap = 1e-3 # convergence tolerance for the gap
        self.epsilon_diam = 1e-2 # convergence tolerance for the diameter

        self.x_train = None
        self.y_train = None

        self.best_l = None
        self.best_u = None
        self.lower_bound = -np.inf
        self.total_nodes = 0
        self.final_gap = None
        self.final_diameter = None
        self.verbose = False

    def set_kernel(self, kernel_spec):
        assert kernel_spec in ["abs_exp", "squar_exp", "matern32", "matern52"]
        self.kernel_spec = kernel_spec

        if kernel_spec == "abs_exp":
            self.kernel_func = lambda d: np.exp(-np.sqrt(d) / self.theta)
        elif kernel_spec == "matern32":
            self.kernel_func = lambda d: (1 + np.sqrt(3) * np.sqrt(d)) * np.exp(-np.sqrt(3) * np.sqrt(d) / self.theta)
        elif kernel_spec == "matern52":
            self.kernel_func = lambda d: (1 + np.sqrt(5) * np.sqrt(d) + (5/3) * d) * np.exp(-np.sqrt(5) * np.sqrt(d) / self.theta)
        elif kernel_spec == "squar_exp":
            self.kernel_func = lambda d: np.exp(-d / (2 * self.theta ** 2))

    # You only have to run this once
    def set_covmatrix(self, x):
        n = x.shape[0]
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                d = np.sum((x[i] - x[j])**2)
                self.K[i, j] = self.kernel_func(d)
                self.K[j, i] = self.K[i, j]
        return self.K

    def set_acq_type(self, acq_type):
        assert acq_type in ["EI", "LCB"]
        self.acquisition_type = acq_type

    def set_tolerances(self, epsilon_gap=1e-3, epsilon_diam=1e-2):
        self.epsilon_gap = epsilon_gap
        self.epsilon_diam = epsilon_diam

    def set_y_min(self, y_min):
        self.y_min = y_min
    
    def ker_bounds(self, x, l, u):
        self.kL, self.kU = [], []
        for i in range(len(x)):
            d_L = np.sum(np.maximum(0, np.maximum(l - x[i], x[i] - u))**2)
            d_U = np.sum(np.maximum(np.abs(l - x[i]), np.abs(u - x[i]))**2)
            self.kL.append(self.kernel_func(d_U))
            self.kU.append(self.kernel_func(d_L))
        return self.kL, self.kU

    def mu_bounds(self):
        alpha = np.linalg.inv(self.K) @ self.y_train
        self.mu_U = np.sum(alpha * np.where(alpha >= 0, self.kU, self.kL))
        self.mu_L = np.sum(alpha * np.where(alpha >= 0, self.kL, self.kU))
        return self.mu_L, self.mu_U


    def var_U(self, regularize=True):

        # Regularize if necessary
        eigvals = np.linalg.eigvalsh(self.K)

        if np.any(eigvals <= 0):

            if not regularize:
                raise ValueError("K is not PD.")
            
            shift = 1e-8 - np.min(eigvals)
            self.K += shift * np.eye(self.K.shape[0])

        Q = np.linalg.inv(self.K)
        k_var = cp.Variable(len(self.kU))

        obj = cp.Maximize(1 - cp.quad_form(k_var, Q))
        constraints = [k_var >= self.kL, k_var <= self.kU]
        prob = cp.Problem(obj, constraints)
        prob.solve()

        self.var_U_val = prob.value
        return self.var_U_val

    def var_L(self):

        epsilon = 1e-6

        np.random.seed(1034)

        k = np.random.uniform(self.kL, self.kU)

        active_coords = set(range(len(k)))

        def f(k_vec): return 1 - k_vec @ self.K @ k_vec

        f_curr = f(k)

        while active_coords:

            improvement = False

            for i in list(active_coords):

                for val in [self.kL[i], self.kU[i]]:

                    k_new = k.copy()
                    k_new[i] = val
                    f_val = f(k_new)

                    if f_val < f_curr - epsilon:
                        k = k_new
                        f_curr = f_val
                        improvement = True
                        break
                if not improvement:
                    active_coords.remove(i)
            if not improvement:
                break
        self.var_L_val = f_curr
        return self.var_L_val



class BnBNode:
    def __init__(self, l, u, aq_L, aq_U):
        self.l = l
        self.u = u
        self.aq_L = aq_L
        self.aq_U = aq_U
        self.diam = np.max(u - l)
        self.midpoint = 0.5 * (l + u)

    def __lt__(self, other):
        return self.aq_U > other.aq_U

class BnBAlgorithm(BnBAlgorithmBase):
    def __init__(self, gpsurrogate:GaussianProcess):
        super().__init__()
        self.gpsurrogate = gpsurrogate  # Optional GP surrogate

    def _branch(self, l, u):
        idx = np.argmax(u - l)
        mid = (l[idx] + u[idx]) / 2
        l1, u1 = l.copy(), u.copy()
        l2, u2 = l.copy(), u.copy()
        u1[idx] = mid
        l2[idx] = mid
        return [(l1, u1), (l2, u2)]

    def compute_all_bounds(self, x, y, l, u):
        self.set_covmatrix(x)
        self.ker_bounds(x, l, u)
        mu_L, mu_U = self.mu_bounds()
        var_U_val = max(self.var_U(), 0)
        var_L_val = max(self.var_L(), 0)
        return mu_L, mu_U, var_L_val, var_U_val

    def compute_acq_upper_bound(self, mu_U, var_U_val, var_L_val):

        sigma = np.sqrt(var_L_val) if self.acquisition_type == "LCB" else np.sqrt(var_U_val)

        if self.acquisition_type == "LCB":

            return mu_U - self.beta * sigma
        
        elif self.acquisition_type == "EI":

            z = (mu_U - self.y_min) / sigma if sigma > 0 else 0.0
            ei = (mu_U - self.y_min) * norm.cdf(z) + sigma * norm.pdf(z)

            return max(ei, 0.0)
        else:
            raise ValueError("Invalid acquisition type")

    def compute_acq_lower_bound(self, mu_L, var_U_val, var_L_val):

        sigma = np.sqrt(var_U_val) if self.acquisition_type == "LCB" else np.sqrt(var_L_val)

        if self.acquisition_type == "LCB":
            return mu_L - self.beta * sigma
        
        elif self.acquisition_type == "EI":

            if self.BnB_LBmethod == "IPOPT":
                    
                    if self.acquisition_type == "LCB":

                        acqf = LCBacquisition(self.gpsurrogate)

                    elif self.acquisition_type == "EI":

                        acqf = EIacquisition(self.gpsurrogate)

                    else:
                        raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)

                    acqf_obj_callback = lambda x: float(np.array(acqf.evaluate(np.atleast_2d(x))).flat[0])
                    acqf_callback = {'obj': acqf_obj_callback}
                    if acqf.has_gradient == True:
                                acqf_grad_callback = lambda x: np.array(acqf.eval_g(np.atleast_2d(x)))
                                acqf_callback['grad'] = acqf_grad_callback

                    if x0 is None:
                        x0 = np.array([uniform(b[0], b[1]) for b in self.gpsurrogate.get_bounds()])

                    xopt, yout, success = self.acqf_minimizer_callback(acqf_callback, x0)

                    if not success:
                        raise RuntimeError("EI maximization failed")

                    return xopt, yout  

            else: 
                z = (mu_L - self.y_min) / sigma if sigma > 0 else 0.0
                ei = (mu_L - self.y_min) * norm.cdf(z) + sigma * norm.pdf(z)
                return max(ei, 0.0)
        
        else:
            raise ValueError("Invalid acquisition type")

    def optimize(self, x, y, l_init, u_init):

        self.x_train = x
        self.y_train = y
        self.y_min = np.min(y)

        # Compute bounds for root node
        mu_L, mu_U, var_L_val, var_U_val = self.compute_all_bounds(x, y, l_init, u_init)
        print(f"Initial bounds: mu_L={mu_L}, mu_U={mu_U}, var_L_val={var_L_val}, var_U_val={var_U_val}")
        aq_L_val = self.compute_acq_lower_bound(mu_L, var_U_val, var_L_val)
        print(f"Initial acquisition lower bound: {aq_L_val}")
        aq_U_val = self.compute_acq_upper_bound(mu_U, var_U_val, var_L_val)
        print(f"Initial acquisition upper bound: {aq_U_val}")

        # Initialize queue
        queue = [BnBNode(l_init, u_init, aq_L_val, aq_U_val)]
        diameters = [np.max(u_init - l_init)]
        self.best_l, self.best_u = l_init.copy(), u_init.copy()
        self.lower_bound = aq_L_val
        self.total_nodes = 1

        while queue:

            node = heapq.heappop(queue)

            if node.aq_U - self.lower_bound <= self.epsilon_gap or node.diam <= self.epsilon_diam:
                continue

            for l_child, u_child in self._branch(node.l, node.u):
                mu_L, mu_U, var_L_val, var_U_val = self.compute_all_bounds(x, y, l_child, u_child)
                aq_L_r = self.compute_acq_lower_bound(mu_L, var_U_val, var_L_val)
                aq_U_r = self.compute_acq_upper_bound(mu_U, var_U_val, var_L_val)
                self.total_nodes += 1
                diameters.append(np.max(u_child - l_child))

                if aq_U_r > self.lower_bound:
                    if aq_L_r > self.lower_bound:
                        self.lower_bound = aq_L_r
                        self.best_l, self.best_u = l_child, u_child
                    heapq.heappush(queue, BnBNode(l_child, u_child, aq_L_r, aq_U_r))

        self.final_gap = max([n.aq_U for n in queue], default=0.0) - self.lower_bound
        self.final_diameter = min(diameters)

        return self.best_l, self.best_u, self.lower_bound