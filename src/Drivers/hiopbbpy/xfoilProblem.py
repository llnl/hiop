import numpy as np
import sys
from pathlib import Path

import random

# Import the base optimization problem class.
from hiopbbpy.problems import Problem

# Import the ipopt problem class.
from cyipopt import Problem as cyProblem

from scipy.optimize import NonlinearConstraint

from ds4mems.airfoil import XFoilAirfoilPerformance

class XFoilSampler:
    def __init__(self, n, var_bounds, tighter_bounds=None, ref_x=None, rng=None, use_ref=False):
        self.n = n

        var_bounds = np.asarray(var_bounds, dtype=float)
        
        if var_bounds.shape != (self.n, 2):
            raise ValueError(f"var_bounds must have shape ({self.n}, 2), got {var_bounds.shape}")

        self.var_lb = var_bounds[:, 0]
        self.var_ub = var_bounds[:, 1]
        self.tighter_lb = None
        self.tighter_ub = None

        self.ref_x = None if ref_x is None else np.asarray(ref_x, dtype=float).ravel()
        if self.ref_x is not None and self.ref_x.shape != (self.n,):
            raise ValueError(f"ref_x must have shape ({self.n},), got {self.ref_x.shape}")
                
        if tighter_bounds is not None:
            #print(f"use small box")
            tighter_bounds = np.asarray(tighter_bounds, dtype=float)
            if tighter_bounds.shape != (self.n, 2):
                raise ValueError(f"tighter_bounds must have shape ({self.n}, 2), got {tighter_bounds.shape}")
            self.tighter_lb = tighter_bounds[:, 0]
            self.tighter_ub = tighter_bounds[:, 1]
    
        self.rng = np.random.default_rng() if rng is None else rng

    def rejection_sampling(self, n_samples, is_valid=None, max_tries=10000):
        n_samples = int(n_samples)
        print(f"Do rejection sampling! Find {n_samples} samples.")

        samples = []
        tries = 0

        while len(samples) < n_samples:
            if tries > max_tries:
                raise RuntimeError("Could not generate enough valid samples.")
            tries += 1

            # --- Generate one candidate ---
            if self.tighter_lb is None:
                x = self.rng.uniform(self.var_lb, self.var_ub)
            else:
                if self.rng.random() < 0.5:
                    print("sample from small box: ")
                    x = self.rng.uniform(self.tighter_lb, self.tighter_ub)
                else:
                    print("sample from big box:   ")
                    x = self.rng.uniform(self.var_lb, self.var_ub)

            # --- Check validity ---
            if is_valid is None:
                samples.append(x)
            else:
                try:
                    if is_valid(x,run_dir=f"temp_sample_{len(samples)}"):
                        samples.append(x)
                        print("")
                    else:
                        print("---reject!")
                except Exception:
                    pass  # reject if evaluation crashes

        return np.array(samples)


    def random(self, n_samples):
        n_samples = int(n_samples)

        # No small box --> sample everything from big box
        if self.tighter_lb is None:
            return self.rng.uniform(
                self.var_lb, self.var_ub, size=(n_samples, self.n)
            )

        # Split samples
        n_small = n_samples // 2
        n_full = n_samples - n_small

        samples = np.empty((n_samples, self.n))

        # Sample from smaller box
        print(f"generate {n_small} samples from small box")
        samples[:n_small] = self.rng.uniform(
            self.tighter_lb, self.tighter_ub, size=(n_small, self.n)
        )

        # Sample from full box
        print(f"generate {n_full} samples from big box")
        samples[n_small:] = self.rng.uniform(
            self.var_lb, self.var_ub, size=(n_full, self.n)
        )

        # Optional: shuffle so small-box samples aren’t grouped?
        #self.rng.shuffle(samples, axis=0)

        return samples

# -------------------------------------------------------------------
# Air Foil Optimization Problem Definition
# -------------------------------------------------------------------

class xfoilProblem(Problem):
    def __init__(self, ndim, xlimits, constraints=[], 
                 tighter_bounds=None, ref_x=None, use_ref=False,
                 xfoil_path=None, 
                 n_points=201, mach=0.2, reynolds_list=(1e6, 5e6),
                 penalty_weight=1e5, penalty_power=2, constr_eps=None):
        """
        Initializes the wind farm layout optimization problem using FLORIS.
        
        Parameters:
            ndim (int): Total number of decision variables
            xlimits: Limits for the decision variables.
            constraints (list): List of constraint definitions (optional).
        """
        name = 'xFoil'
        super().__init__(ndim, xlimits, name=name, constraints=constraints)

        # Derive the number of variables.
        self.n = int(ndim)
        self._eval_cache = {}   # key -> (f, c) or (f, None)
        self._cache_ndigits = 12
        # include config tag so cache is invalidated if settings change
        self._cache_tag = ("mach", mach, "Re", tuple(reynolds_list), "npts", n_points)
        self.initial_samples = True
        if xfoil_path is None:
            raise RuntimeError("Set a valid path to xfoil binary.")

        self.airfoil_perf = XFoilAirfoilPerformance(xfoil_path)

        # Define the objective function    
        self.base_obj_func = self.airfoil_perf_obj
        self.constr_func = self.airfoil_perf.constr

        self.penalty_weight = float(penalty_weight)
        self.penalty_power = int(penalty_power)
        self.constr_eps = np.finfo(np.float64).eps if constr_eps is None else float(constr_eps)

        self.obj_func = self._penalized_obj         # for unconstrained prob, set this to base_obj_func
        
        self.constraints = {}
        self.tighter_bounds = tighter_bounds
        self.sampler = XFoilSampler(n=self.n, var_bounds=xlimits, tighter_bounds=self.tighter_bounds, ref_x=ref_x, use_ref=use_ref)

    def airfoil_perf_obj(self, X, run_dir="./temp_output0"):
        aoa_for_xfoil = (0.0, 20.0, 1.0)  # (start, stop, step)
        re_values = np.linspace(1e6, 5e6, num=5)
        mach = 0.2

        #print(".", end="", flush=True)  # Progress indicator

        F = []
        for re in re_values:
            F.append(self.airfoil_perf(X, aoa=aoa_for_xfoil, re=re, mach=mach,run_dir=run_dir))

        return -np.mean(np.concat(F))  # Return negative mean performance for minimization
        

    def _cache_key(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).ravel()
        return (self._cache_tag, tuple(np.round(x, self._cache_ndigits)))

    def eval_cached(self, x: np.ndarray, run_dir="./temp_output2"):
        print("in eval_cached")
        k = self._cache_key(x)
        hit = self._eval_cache.get(k, None)
        if hit is not None:
            return hit  # (f, c)

        c = np.asarray(self.constr_func(x), dtype=float).ravel()
        if c > 0.0:
            f = float(self.base_obj_func(x, run_dir=run_dir))
        else:
            f = np.inf
        print(f"evaluation: f={f};  c={c}")
        self._eval_cache[k] = (f, c)
        return f, c

    def _penalized_obj(self, x: np.ndarray, run_dir="./temp_output3") -> float:
        #print("use penalty func!")
        print("in penalized_obj")
        
        f, con_arr = self.eval_cached(x,run_dir=run_dir)

        if not np.isfinite(f):
            print(f"Warning: base_obj_func returned {f} at x = {x}, replacing with 1e3")
            f = 1e3

        con_arr = np.atleast_1d(np.asarray(con_arr, dtype=float))
        if con_arr.shape != (1,):
            raise ValueError(
                f"xfoilProblem expects exactly 1 constraint, got shape {con_arr.shape}"
            )

        con_f = float(con_arr[0])
        # print(f"con_f = {con_f}")
        con_violation = max(0.0, self.constr_eps - con_f)  # scalar float

        # # adapting penalty multiplier (scalar)
        # v = con_violation
        # if v < 1e-4:
        #     penalty_mult = 0.0
        # elif v < 1e-2:
        #     penalty_mult = 10.0
        # elif v < 1e-1:
        #     penalty_mult = 100.0
        # elif v < 1.0:
        #     penalty_mult = 500.0
        # else:
        #     penalty_mult = 1000.0

        # penalty = penalty_mult * (con_violation ** self.penalty_power)  # scalar
        # print(f". obj = base obj + penalty: {f} + {penalty}", flush=True)
        # retval = f + penalty
    
        ### fixme: 
        #when con is large, skip evaluating f, just retrun big penalty term
        v = con_violation
        retval = 0.0
        if v < 1e-8:
            retval = f
        elif v < 1e-4:
            retval = -25
        elif v < 1:
            retval = -50
    
        print(f". penalty obj = {retval}; con_violation = {v}", flush=True)
        
        
        return retval


    def _evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the objective function
        
        Parameters:
            x (ndarray): An array 
        
        Returns:
            ndarray: The objective values 
        """
        print("in _evaluate")
        y = [float(self.obj_func(xi,**kwargs)) for xi in x]   # shape (k,)
        return np.asarray(y, dtype=float).reshape(-1, 1)    

    def sample(self, nsample: int) -> np.ndarray:
        def is_valid(x, run_dir=None):
            print("in is_valid")
            f, con_arr = self.eval_cached(x,run_dir=run_dir)
            
            con_arr = np.atleast_1d(np.asarray(con_arr, dtype=float))
            if con_arr.shape != (1,):
                raise ValueError(
                    f"xfoilProblem expects exactly 1 constraint, got shape {con_arr.shape}"
                )

            con_f = float(con_arr[0])
            
            #isvalid = np.isfinite(f)       # an alt way
            isvalid = (con_f > 0.0) and np.isfinite(f)
            return isvalid
        return self.sampler.rejection_sampling(nsample, is_valid=is_valid)