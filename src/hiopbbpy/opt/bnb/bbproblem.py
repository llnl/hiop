"""
Implementation of the Branch and Bound (BnB) algorithm for Optimization of Acquisition Functions in Bayesian optimization.
This module currently supports Convex Kernel Functions (Matern and SE) and Acquisition Functions EI and LCB.

Authors:    Natalia Rodriguez Figueroa <rodriguezfig1@llnl.gov>
"""

import heapq
import itertools
import numpy as np
import cvxpy as cp
from scipy.stats import norm

from ..acquisition import LCBacquisition, EIacquisition

class BnBAlgorithmBase:
    def __init__(self, x=None, y=None):
        # Node class for priority queue
        self.BnBNode = BnBNode
        self.BnB_LBmethod = "IPOPT"
        #self.BnB_LBmethod = None  # Use CVXPY for lower bounds

        # Stopping criteria
        self.epsilon_gap = 1e-3
        self.epsilon_diam = 1e-2

        # Kernel info for bounds
        self.kernel_spec = None
        self.kernel_func = None
        self.y_min = None

        # Evaluation parameters
        self.K = None
        self.K_inv = None
        self.theta = None
        self.ell = None  # ARD scaling for distance

        # Test
        self.enable_debug_checks = False  

        # BnB search state
        self.best_l = None
        self.best_u = None
        self.upper_bound = np.inf
        self.final_gap = None
        self.final_diameter = None
        self.total_nodes = 0
        self.verbose = False

        # Training data
        self.x = x
        self.y = y

    def sync_kernel_from_smt(self):
        corr_map = {
            "pow_exp": "pow_exp",     # treat as Gaussian if power=2
            "squar_exp": "pow_exp",   # SMT sometimes uses this name for Gaussian
            "abs_exp": "matern12",    # ν = 1/2
            "matern32": "matern32",
            "matern52": "matern52",
        }
        corr_type = self.gpsurrogate.surrogatesmt.options["corr"]
        self.kernel_spec = corr_map[corr_type]
        self.set_kernel(self.kernel_spec)
        self.theta = np.asarray(self.gpsurrogate.surrogatesmt.corr.theta, dtype=float)

        # If pow_exp, ensure it's actually Gaussian (power=2); otherwise raise.
        if corr_type == "pow_exp":
            power = getattr(self.gpsurrogate.surrogatesmt.options, "pow_exp_power", 2)
            if power != 2:
                raise ValueError(f"pow_exp_power={power} not supported in isotropic form; need power=2 for SE.")

        # Map θ (SMT) -> ℓ for ARD scaling used in d
        if self.kernel_spec == "pow_exp":       # Gaussian
            self.ell = 1.0 / np.sqrt(self.theta)
        else:                                    # Matérn family
            self.ell = 1.0 / self.theta
    
    def sync_smt_parameterts(self):
        sm = self.gpsurrogate.surrogatesmt

        # --- map SMT corr name to the 5 supported kernels ---
        corr_map = {
            "pow_exp":  "pow_exp",    # power-exponential (Gaussian if power=2)
            "squar_exp":"pow_exp",    # SMT alias for Gaussian
            "abs_exp":  "matern12",   # exp(-sum θ|dx|)  == Matérn ν=1/2 product
            "matern32": "matern32",
            "matern52": "matern52",
        }
        corr_type = sm.options["corr"]
        if corr_type not in corr_map:
            raise ValueError(f"Unsupported SMT corr '{corr_type}' for kernel bounds.")
        self.kernel_spec = corr_map[corr_type]

        # --- pull trained hyperparams (prefer optimal_theta) ---
        theta = getattr(sm, "optimal_theta", None)
        if theta is None:
            theta = sm.corr.theta
        self.theta = np.asarray(theta, dtype=float).ravel()


    def set_kernel(self, kernel_spec):

        assert kernel_spec in ["abs_exp", "pow_exp", "matern32", "matern52"]
        self.kernel_spec = kernel_spec

        if kernel_spec == "abs_exp":  # ν = 1/2
            self.kernel_func = lambda d: np.exp(-np.sqrt(d))

        elif kernel_spec == "pow_exp":  # SE, ν = ∞
            self.kernel_func = lambda d: np.exp(-d)

        elif kernel_spec == "matern32":  # ν = 3/2
            self.kernel_func = lambda d: (
                (1 + np.sqrt(3) * np.sqrt(d)) *
                np.exp(-np.sqrt(3) * np.sqrt(d))
            )

        elif kernel_spec == "matern52":  # ν = 5/2
            self.kernel_func = lambda d: (
                (1 + np.sqrt(5) * np.sqrt(d) + (5/3) * d) *
                np.exp(-np.sqrt(5) * np.sqrt(d))
            )


    def set_covmatrix(self, x):

        n = x.shape[0]
        ell = np.asarray(self.ell, dtype=float)
        nugget = float(self.gpsurrogate.surrogatesmt.options['nugget'])
        self.K = np.zeros((n, n))
        
        # Compute symmetric kernel matrix
        for i in range(n):
            for j in range(i, n):

                d = np.sum(((x[i] - x[j]) / ell)**2) 
                
                self.K[i, j] = self.kernel_func(d)
                self.K[j, i] = self.K[i, j]

        # Add nugget for stability
        self.K += nugget * np.eye(n)

        # --- Compute inverse ---
        try:
            self.K_inv = np.linalg.inv(self.K)
            #print("K_inv:\n", self.K_inv)
        except np.linalg.LinAlgError:
            print("ERROR: Singular K detected, cannot invert.")

    def ker_bounds(self, x, l, u):

        ell = np.asarray(self.ell)
        kL, kU = [], []

        for xi in x:

            # per-dim nearest/farthest distances to the box
            dmin = np.maximum(0.0, np.maximum(l - xi, xi - u))     # (d,)
            dmax = np.maximum(np.abs(l - xi), np.abs(u - xi))      # (d,)

            d_L = np.sum((dmax / ell)**2)   # largest distance -> lower kernel
            d_U = np.sum((dmin / ell)**2)   # smallest distance -> upper kernel

            kL.append(self.kernel_func(d_L))
            kU.append(self.kernel_func(d_U))

        return np.array(kL), np.array(kU)


    def mu_bounds(self, y, kL, kU):

        alpha = self.K_inv @ y
        mu_U = np.sum(alpha * np.where(alpha >= 0, kU, kL))
        mu_L = np.sum(alpha * np.where(alpha >= 0, kL, kU))

        return mu_L, mu_U

    def sigma2_U(self, kL, kU):

        # Set up QP to solve for upper variance bound
        var = cp.Variable(len(kU))
        #Add constant values here constants = 
        obj = cp.Maximize(1 - cp.quad_form(var, self.K_inv))
        constraints = [var >= kL, var <= kU]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP)

        sigma2_U = prob.value
        return max(sigma2_U, 0)

    def sigma2_L(self,kL, kU, epsilon = 1e-6, random_seed= 42):

        # Randomly initialize a point in the bounds
        np.random.seed(random_seed)
        var = np.random.uniform(kL, kU)

        # Initialize active coordinates
        active_coords = set(range(len(kL)))

        # Define the function to minimize
        def f(k_vec): return 1 - k_vec @ self.K_inv @ k_vec
        f_curr = f(var)

        # Iteratively improve the point by evaluating each coordinate direction.
        while active_coords:
            improvement = False
            for i in list(active_coords):
                for val in [kL[i], kU[i]]:
                    var_new = var.copy()
                    var_new[i] = val
                    f_val = f(var_new)
                    if f_val < f_curr - epsilon:
                        var = var_new
                        f_curr = f_val
                        improvement = True
                        break
                if not improvement:
                    active_coords.remove(i)
            if not improvement:
                break

        sigma2_L = f_curr
        return max(sigma2_L, 0)

    def rs_ei(self, y, mu, sigma):
        
        y_min = np.min(y)

        if sigma > 1e-12:
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei
        else:
            # Deterministic case: EI = max(y_min - mu, 0)
            return -max(y_min - mu, 0.0)
    
    def rs_lcb(self, mu, sigma):

        return mu - self.beta * sigma


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
    def __init__(self, gpsurrogate, acqf_minimizer_callback=None):
        super().__init__()  # no args
        self.gpsurrogate = gpsurrogate
        self.acqf_minimizer_callback = acqf_minimizer_callback
        self.sync_kernel_from_smt()

    def _branch(self, l, u):

        # Force to float to avoid truncation issues
        l = l.astype(float)
        u = u.astype(float)

        # Pick the dimension with largest length
        d = np.argmax(u - l)
        mid = 0.5 * (l[d] + u[d])

        # If the midpoint is the same as one bound (degenerate split), return nothing
        if np.isclose(mid, l[d]) or np.isclose(mid, u[d]):
            return []

        # Generate child boxes
        l1, u1 = l.copy(), u.copy()
        l2, u2 = l.copy(), u.copy()
        
        # Split along midpoint
        u1[d] = mid
        l2[d] = mid

        return [(l1, u1), (l2, u2)]


    # For minimization, we find a feasible function value as the upper bound on the minimum value of the acquisition function.
    def compute_acq_upper_bound(self, x, y, l, u):

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

                    #if x0 is None: #Need to fix this.
                    x0 = np.array(np.uniform(l, u))
                    
                    xopt, yout, success = self.acqf_minimizer_callback(acqf_callback, x0)

                    if not success:
                        raise RuntimeError("EI maximization failed")

                    return float(yout)
            else:
                
                # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
                
                # Compute the kernel bounds with given x
                kL, kU = self.ker_bounds(x, l, u)
                # Compute the mean bounds
                mu_L, mu_U = self.mu_bounds(y,kL, kU)
                var_L = self.sigma2_L(kL, kU)

                if self.enable_debug_checks:

                    l_check = np.asarray(l).reshape(-1)
                    u_check = np.asarray(u).reshape(-1)
                    c_check = 0.5*(l+u)

                    # center + for each axis i: set x_i to l_i and u_i, others at center
                    Xchk = [c_check]
                    for i in range(l.size):
                        x_lo = c_check.copy(); x_lo[i] = l_check[i]; Xchk.append(x_lo)
                        x_hi = c_check.copy(); x_hi[i] = u_check[i]; Xchk.append(x_hi)
                    Xchk = np.vstack(Xchk)  # shape (1+2d, d)

                    # GP mean at those points (vector length 1+2d)
                    mu_vec = np.asarray(self.gpsurrogate.mean(Xchk)).reshape(-1)
                    print(self.gpsurrogate.surrogatesmt.options['poly'])
                    try:
                        nugget = float(self.gpsurrogate.surrogatesmt.options['nugget'])
                        print("nugget =", nugget)
                    except Exception:
                        nugget = 0.0

                    var_vec = np.asarray(self.gpsurrogate.variance(Xchk)).reshape(-1)

                    # check
                    tol = 1e-8
                    ok_mu = (mu_vec >= mu_L - tol) & (mu_vec <= mu_U + tol)
                    if not np.all(ok_mu):
                        bad = np.where(~ok_mu)[0].tolist()
                        print(f"[μ] bounds violated at indices {bad}: "
                            f"min={mu_vec.min():.6g}, max={mu_vec.max():.6g}, "
                            f"bounds=({mu_L:.6g},{mu_U:.6g})")
                    ok_var = (var_vec >= var_L - tol)

                    if not np.all(ok_var):
                        bad = np.where(~ok_var)[0].tolist()
                        print(f"[σ] bounds violated at indices {bad}: "
                            f"min={var_vec.min():.6g}, max={var_vec.max():.6g}, "
                            f"bounds=({var_L:.6g},inf)")

                
                if self.acquisition_type == "LCB":

                    lcb_U = self.rs_lcb(mu_U, np.sqrt(var_L))
                    return lcb_U
                
                elif self.acquisition_type == "EI":
                     
                    ei_U = self.rs_ei(y, mu_U, np.sqrt(var_L))
                    return ei_U
                
                else:
                    raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)

    # For minimization, we compute the lower bound explicitly using the acquisition function over mu, sigma.
    def compute_acq_lower_bound(self, x,y,l,u):

                # We compute the upper bound of the acquisition function based on bounds of the kernel, mu and sigma.
                
                # Compute the kernel bounds with given x
                kL, kU = self.ker_bounds(x, l, u)
                # Compute the mean bounds
                mu_L, mu_U = self.mu_bounds(y, kL, kU)
                var_U = self.sigma2_U(kL, kU)

                if self.enable_debug_checks:

                    l_check = np.asarray(l).reshape(-1)
                    u_check = np.asarray(u).reshape(-1)
                    c_check = 0.5*(l+u)

                    # center + for each axis i: set x_i to l_i and u_i, others at center
                    Xchk = [c_check]
                    for i in range(l.size):
                        x_lo = c_check.copy(); x_lo[i] = l_check[i]; Xchk.append(x_lo)
                        x_hi = c_check.copy(); x_hi[i] = u_check[i]; Xchk.append(x_hi)
                    Xchk = np.vstack(Xchk)  # shape (1+2d, d)

                    # GP mean at those points (vector length 1+2d)
                    mu_vec = np.asarray(self.gpsurrogate.mean(Xchk)).reshape(-1)
                    var_vec = np.asarray(self.gpsurrogate.variance(Xchk)).reshape(-1)

                    # check
                    tol = 1e-8
                    ok_mu = (mu_vec >= mu_L - tol) & (mu_vec <= mu_U + tol)
                    if not np.all(ok_mu):
                        bad = np.where(~ok_mu)[0].tolist()
                        print(f"[μ] bounds violated at indices {bad}: "
                            f"min={mu_vec.min():.6g}, max={mu_vec.max():.6g}, "
                            f"bounds=({mu_L:.6g},{mu_U:.6g})")
                    ok_var = (var_vec <= var_U + tol)

                    if not np.all(ok_var):
                        bad = np.where(~ok_var)[0].tolist()
                        print(f"[σ] bounds violated at indices {bad}: "
                            f"min={var_vec.min():.6g}, max={var_vec.max():.6g}, "
                            f"bounds=(0,{var_U:.6g})")

                if self.acquisition_type == "LCB":

                    lcb_U = self.rs_lcb(mu_L, np.sqrt(var_U))
                    return lcb_U
                
                elif self.acquisition_type == "EI":
                     
                    ei_U = self.rs_ei(y, mu_L, np.sqrt(var_U))
                    return ei_U
                
                else:
                    raise NotImplementedError("No implemented acquisition_type associated to" + self.acquisition_type)

    def _prune_queue(self, queue, gub, eps):
        """Keep only nodes that can beat current GUB within tolerance; then re-heapify."""
        # queue items are (L, counter, node)
        pruned = [(L, c, n) for (L, c, n) in queue if L < gub - eps]
        heapq.heapify(pruned)
        return pruned

    def optimize(self, x, y, l_init, u_init):
        """
        Branch & Bound minimization with tolerance stopping.
        Core logic only: correct heap order, pruning on GUB tightening,
        single global stop, diameter continue, consistent per-node prune.
        """
        print("=== Starting Branch & Bound Optimization (Minimization) ===")
        print(f"Initial bounds: l = {l_init}, u = {u_init}")
        print(f"Number of points: {x.shape[0]}, Dim = {x.shape[1]}")

        # Precompute covariance, etc.
        self.set_covmatrix(x)

        # Root bounds
        aq_L_val = self.compute_acq_lower_bound(x, y, l_init, u_init)
        aq_U_val = self.compute_acq_upper_bound(x, y, l_init, u_init)
        print(f"\nInitial acquisition lower bound: {aq_L_val}")
        print(f"Initial acquisition upper bound: {aq_U_val}")

        # Init root + heap ordered by aq_L
        root = BnBNode(l_init.astype(float), u_init.astype(float), aq_L_val, aq_U_val)
        
        # --- HEAP STORES TUPLES: (L, counter, node) ---
        self._ctr = getattr(self, "_ctr", itertools.count())
        queue = [(root.aq_L, next(self._ctr), root)]
        heapq.heapify(queue)

        # Global best (minimization GUB)
        self.best_val = aq_U_val
        self.best_l, self.best_u = l_init.copy(), u_init.copy()

        diameters = [float(np.max(u_init - l_init))]
        self.total_nodes = 1
        iteration = 0

        while queue:
            iteration += 1
            # pop the node with smallest L
            L_top, _, node = heapq.heappop(queue)

            # sanity: popped L must equal node.aq_L and be <= any remaining L
            assert abs(L_top - node.aq_L) <= 1e-12
            if queue:
                min_rest = min(L for (L,_,_) in queue)
                assert L_top <= min_rest + 1e-12, f"Heap not ordered by L (popped {L_top}, min_rest {min_rest})"

            # Update GUB
            if node.aq_U < self.best_val:
                self.best_val = node.aq_U
                self.best_l, self.best_u = node.l, node.u
                queue = self._prune_queue(queue, self.best_val, self.epsilon_gap)

            print(f"\n--- Iteration {iteration} ---")
            print(f"Node bounds: l={node.l}, u={node.u}")
            print(f"Node acquisition bounds: L={node.aq_L}, U={node.aq_U}")
            print(f"Current best feasible value (GUB): {self.best_val}")

            # Global stop: GUB - GLB <= eps  (GLB == node.aq_L == L_top)
            if self.best_val - node.aq_L <= self.epsilon_gap:
                print(f"STOP: GUB - Node.L = {self.best_val - node.aq_L} <= {self.epsilon_gap}")
                break

            # Diameter stop (node-local)
            node_diam = float(np.max(node.u - node.l))
            if node_diam <= self.epsilon_diam:
                print(f"Skip: Node diameter {node_diam} <= {self.epsilon_diam}")
                continue

            # Per-node prune (consistent with stop rule)
            if node.aq_L >= self.best_val - self.epsilon_gap:
                print("Pruned: Node cannot improve best within tolerance.")
                continue

            # --- Branch ---
            for l_child, u_child in self._branch(node.l, node.u):
                print(f"  Branching to child: l={l_child}, u={u_child}")

                aq_L_r = self.compute_acq_lower_bound(x, y, l_child, u_child)
                aq_U_r = self.compute_acq_upper_bound(x, y, l_child, u_child)
                print(f"  Child acquisition bounds: L={aq_L_r}, U={aq_U_r}")

                self.total_nodes += 1
                diameters.append(float(np.max(u_child - l_child)))

                # Update GUB from child and prune if improved
                if aq_U_r < self.best_val:
                    self.best_val = aq_U_r
                    self.best_l, self.best_u = l_child, u_child
                    queue = self._prune_queue(queue, self.best_val, self.epsilon_gap)

                # Child-level prune (same tolerance)
                if aq_L_r >= self.best_val - self.epsilon_gap:
                    print("  Child pruned: L ≥ GUB - eps.")
                    continue

                # PUSH AS TUPLE
                child = BnBNode(l_child, u_child, aq_L_r, aq_U_r)
                heapq.heappush(queue, (child.aq_L, next(self._ctr), child))

            if not queue:
                print("\nSTOP: Queue empty, no better nodes remain.")
                break

            # Optional visibility
            phi_GLB = min(L for (L,_,_) in queue)
            print(f"Queue size: {len(queue)} | GLB={phi_GLB} | GUB={self.best_val} | Gap={self.best_val - phi_GLB}")

        self.final_gap = self.best_val - min([L for (L,_,_) in queue], default=self.best_val)
        self.final_diameter = min(diameters) if diameters else float('inf')

        print("\n=== Optimization Finished ===")
        print(f"Total nodes explored: {self.total_nodes}")
        print(f"Best bounds: l={self.best_l}, u={self.best_u}")
        print(f"Best feasible acquisition value (GUB): {self.best_val}")
        print(f"Final gap: {self.final_gap}, final diameter: {self.final_diameter}")

        return self.best_l, self.best_u, self.best_val