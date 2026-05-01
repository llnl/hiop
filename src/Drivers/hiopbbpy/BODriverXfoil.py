import logging
import os
import tempfile
from pathlib import Path
import numpy as np
import sys

from scipy.optimize import NonlinearConstraint

from ds4mems.airfoil import XFoilAirfoilPerformance
from xfoilProblem import xfoilProblem

from hiopbbpy.surrogate_modeling import smtKRG
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.utils import MPIEvaluator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

### Define problem and optimization parameters 
nx = 8                        # Number of design variables
use_ref = False
do_parallel = True
do_profiling = True
max_worker = 2


def main():
    xlimits_small = np.array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [0.0, 0.5],
                [-0.25, 0.25],
            ]
        )
    
    xlimits = np.array(
            [
                [-10.79733932, 39.28774442],
                [-22.81287711, 5.94493027],
                [-23.24153557, 12.15769847],
                [-5.47472985, 7.92991688],
                [-8.20327119, 12.44090178],
                [-1.50046976, 4.41520152],
                [-0.37926066, 4.19234019],
                [-1.61741561, 0.2987408],
            ]
        )
    # FIXME_NY --- use big box  
    xlimits = xlimits_small

    x_nan = np.array([[-1.2269792 , -1.40856   ,  0.56076634, -1.96206043,  1.44158607,
            0.22498941, -0.32848405,  0.04818669],
        [-1.42737423, -1.40397264,  0.36791669, -2.11149784,  1.54822502,
            0.07623255, -0.39513648, -0.05350951]], dtype=float)

    x_penalty = np.array([[-1.49019102, -1.37671764,  0.65196615, -2.38465265,  1.39195187,
            0.16935736, -0.40938738, -0.05497496],
        [-1.10599338, -1.10345512,  0.49594816, -1.93446173,  1.38383001,
            0.55640812, -0.41227115,  0.00824948]], dtype=float)

    x_opt = np.array([
                [-1.24028704, -1.31949567,  0.53528491, -2.16370978,  1.37588402,  0.32447433, -0.16926065, -0.20102103] 
            ], dtype=float)

    x_opt_2 = np.array([-1.46972556, -1.07491497,  0.32973962, -2.20769501,  1.26854236,
            0.3562351 , -0.397001  , -0.32209042])

    x_ref = x_opt[0]

    if use_ref:
        # --- dimension checks ---
        if x_ref is None:
            raise ValueError("x_ref must not be None when use_ref=True")

        if x_ref.shape != (nx,):
            raise ValueError(f"x_ref must have shape ({nx},), got {x_ref.shape}")

        # --- build new bounds ---
        delta = 0.1 * np.abs(x_ref)
        var_lb = x_ref - delta
        var_ub = x_ref + delta

        # replace xlimits
        xlimits = np.column_stack((var_lb, var_ub))
        xlimits_small = None
        print("built a box from the given reference point")


    ### Mac
    problem = xfoilProblem(nx, xlimits, tighter_bounds=xlimits_small, ref_x=x_ref, use_ref=use_ref, xfoil_path="/Users/chiang7/project/2025/scidac/other_lab/ornl/xfoil/bin/xfoil")

    ### LC
    #problem = xfoilProblem(nx, xlimits, tighter_bounds=xlimits_small, ref_x=x_ref, use_ref=use_ref, xfoil_path="/p/lustre1/chiang7/hiopbbpy/xfoil/bin/xfoil")

    print("Problem name: ", problem.name)  


    ### BO parameters
    n_samples = 4             # Number of initial design points
    theta = 1.e-2              # Hyperparameter for the Kriging (GP) model
    acq_type = "EI"            # Acquisition function: "EI" or "LCB"

    print("Acquisition type: ", acq_type)
        
    opt_solver = 'IPOPT'   #"SLSQP" "IPOPT" "trust-constr"
    if opt_solver == "SLSQP" or opt_solver == "trust-constr":
        solver_options = {"maxiter": 100}  #for scipy solvers
    elif opt_solver == "IPOPT":
        solver_options = {"max_iter": 200, "print_level": 1, "tol": 1e-4}
        
    options = {
            'acquisition_type': acq_type,
            'log_level': 'info',
            'bo_maxiter': 2,
            'opt_solver': opt_solver,
            'batch_size': 1,
            'n_start': 3,
            'solver_options': solver_options,
        }

    ### initial training set
    x_train = problem.sample(n_samples)


    if do_parallel:
        # ----- evaluator
        obj_evaluator = MPIEvaluator(
                            executor=ProcessPoolExecutor(max_workers=max_worker),
                            profiling=do_profiling,
                            task_name="PARALLEL_OBJ_EVAL",
                            use_run_dir=True,
                        )
        opt_evaluator = MPIEvaluator(
                            function_mode=False,
                            executor=ProcessPoolExecutor(max_workers=max_worker),
                            profiling=do_profiling,
                            task_name="PARALLEL_IPOPT_START",
                            use_run_dir=False,
                        )
        y_train = obj_evaluator.run(problem.evaluate, x_train)
        options['obj_evaluator'] = obj_evaluator
        options['opt_evaluator'] = opt_evaluator
    else:
        y_train = problem.evaluate(x_train)

    ### Define the GP surrogate model
    gp_model = smtKRG(theta, xlimits, nx)
    gp_model.train(x_train, y_train)

    # Instantiate and run Bayesian Optimization
    bo = BOAlgorithm(problem, gp_model, x_train, y_train, options = options) #EI or LCB
    bo.optimize()

    problem.obj_func(bo.x_opt, run_dir="final_opt")


if __name__ == "__main__":
    main()