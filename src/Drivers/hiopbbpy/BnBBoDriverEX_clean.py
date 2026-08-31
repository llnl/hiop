#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from hiopbbpy.problems import Problem, BraninProblem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithm, BOAlgorithm, LCBacquisition, EIacquisition
from hiopbbpy.utils import MPIEvaluator
import argparse
import random
import time as time      
from pathlib import Path

class QuadraticShift(Problem):
  """
  f(x) = ||x - c||^2
  - Global minimizer: x* = c
  - Global minimum:   f* = 0
  """
  def __init__(self, ndim=2, xlimits=None, c=None, constraints=[]):
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "QuadraticShift"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)

    # choose center if not provided
    if c is None:
      c = self.xlimits.mean(axis=1)
    self.c = np.asarray(c, dtype=float)
    assert self.c.shape == (ndim,), "c must be a {0:d}D point".format(ndim)

    # expose known solution for checking later
    # below is true when c is in xlimits
    self.x_star = self.c.copy()
    self.f_star = 0.0 

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    assert nx == self.ndim
    diff = x - self.c[None, :]
    y = np.sum(diff * diff, axis=1, dtype=float).reshape(ne, 1)
    return y

class PeriodicObjective(Problem):
  """
  f(x) = sum_{i in dim(x)} sin(2*pi * x_i)
  - Global minimizer: x* = (3/8,) * dim(x) + (1/2) * Z^{dim(x)}
  - Global minimum:   f* = -dim(x)
  """
  def __init__(self, ndim=2, xlimits=None, constraints=[], sleep_time=0.0):
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "Periodic"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    # expose known solution for checking later
        
    self.x_star = np.array([0.375,] * ndim)
    self.f_star = -1.0 * ndim

    # artificially increase the compute time required to 
    # evaluate the "black-box" periodic objective
    # sleep_time is the per-sample time required to evaluate
    # the "black-box" objective
    self.sleep_time = sleep_time

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, _ = x.shape
    time.sleep(ne * self.sleep_time) # sleep to artifically increase computational time to evaluate 
    y = np.sum(np.sin(4. * np.pi * x), axis=1).reshape(ne, 1)
    return y


class HartmannProblem(Problem):
  """
     Standard 6D Hartmann
  """
  def __init__(self, xlimits=None, constraints=[], sleep_time=0.0):
    ndim = 6
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "Hartmann"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    # expose known solution for checking later
    self.A = np.array([[10. , 3.0, 17. , 3.5, 1.7, 8.0],
                       [0.05, 10., 17. , 0.1, 8.0, 14.],
                       [3.0 , 3.5, 1.7 , 10., 17., 8.0],
                       [17. , 8.0, 0.05, 10., 0.1, 14.]])
    self.P = np.array([[1312., 1696., 5569., 124. , 8283., 5886.],
                       [2329., 4135., 8307., 3736., 1004., 9991.],
                       [2348., 1451., 3522., 2883., 3047., 6650.],
                       [4047., 8828., 8732., 5743., 1091., 381.0]])
    self.P = 1.e-4 * self.P
    self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
    self.sleep_time = sleep_time

    
    #self.x_star = np.array([0.375,] * ndim)
    #self.f_star = -1.0 * ndim 

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    #y = np.sum(np.sin(4. * np.pi * x), axis=1).reshape(ne, 1)
    y = np.zeros(ne)
    for k in range(ne):
      time.sleep(self.sleep_time)
      for i in range(len(self.alpha)):
        y[k] += -1.0 * self.alpha[i] * np.exp( -1.0 * np.inner(self.A[i,:], (x[k] - self.P[i,:])**2.))
    y = y.reshape(ne, 1)
    return y


class HartmannLikeProblem(Problem):
  """
     6D Hartmann-like anisotropic mixture
  """
  def __init__(self, xlimits=None, constraints=[]):
    ndim = 6
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "HartmannLike"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    self.c1 = np.array([0.20, 0.25, 0.70, 0.70, 0.30, 0.40])
    self.c2 = np.array([0.75, 0.75, 0.20, 0.25, 0.75, 0.70])
    

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    y = np.zeros(ne)
    for k in range(ne):
      y[k] += 0.88 * np.exp(-18. * np.linalg.norm(x[k] - self.c1, 2)**2.)
      y[k] += 0.84 * np.exp(-18. * np.linalg.norm(x[k] - self.c2, 2)**2.)
      arg = -450. * ((x[k][0] - 0.72)**2. + (x[k][1] - 0.14)**2.)
      arg += -70. * sum( (x[k][2:] - 0.63)**2.)
      y[k] += 1.20 * np.exp(arg)
      y[k] *= -1.0
    y = y.reshape(ne, 1)
    return y

"""
global minimum -0.801264 at 2.204461 for mode # 1
global minimum -0.999900 at 1.569224 for mode # 2
global minimum -0.959006 at 1.286198 for mode # 3
global minimum -0.937941 at 1.924579 for mode # 4
global minimum -0.988775 at 1.720171 for mode # 5
global minimum -0.999110 at 1.569224 for mode # 6
global minimum -0.992231 at 1.452869 for mode # 7
global minimum -0.981497 at 1.754763 for mode # 8
"""
class MichalewiczObjective(Problem):
  """
  f(x) = sum_{i in dim(x)} sin(x_i) * [sin( i * x_{i}^2 / pi)]
  """
  def __init__(self, ndim=2, xlimits=None, m=10, constraints=[]):
    if xlimits is None:
      xlimits = np.array([[0.0, np.pi]] * ndim, dtype=float)
    name = "Michalewicz"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    # expose known solution for checking later
    self.m = m
    
    #self.x_star = np.array([0.375,] * ndim)
    #self.f_star = -1.0 * ndim 

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    scale = [i+1. for i in range(nx)]
    x_scaled = np.array([[xj[i] * scale[i] for i in range(nx)] for xj in x])
    assert x_scaled.shape == x.shape, "error in shaping x_scaled"
    y = -1. * np.sum(np.sin(x) * np.sin(x_scaled * x / np.pi)**(2. * self.m), axis=1).reshape(ne, 1)
    return y

class ShekelProblem(Problem):
  """
     4D Shekel-like rational multi-peak function
  """
  def __init__(self, xlimits=None, constraints=[], sleep_time=0.0):
    ndim = 4
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "Shekel"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)

    self.sleep_time = sleep_time
    self.alpha = np.array([0.92, 0.89, 0.87, 0.85, 1.25])
    c0 = np.array([0.15, 0.20, 0.80, 0.60])
    c1 = np.array([0.70, 0.20, 0.25, 0.80])
    c2 = np.array([0.25, 0.75, 0.30, 0.20])
    c3 = np.array([0.80, 0.70, 0.70, 0.30])
    c4 = np.array([0.62, 0.58, 0.66, 0.61])
    self.cs = [c0, c1, c2, c3, c4]
    self.Ds = []
    for i in range(4):
      self.Ds.append(5. * np.identity(ndim))
    self.Ds.append(18. * np.identity(ndim))
    

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    y = np.zeros(ne)
    for k in range(ne):
      time.sleep(self.sleep_time)
      for i in range(5):
        y[k] += self.alpha[i] / (1. + np.linalg.norm(self.Ds[i].dot(x[k] - self.cs[i]))**2.)
    y *= -1.0
    y = y.reshape(ne, 1)
    return y


class SparseActiveProblem(Problem):
  """
     10D Sparse-active-variable function
  """
  def __init__(self, xlimits=None, constraints=[]):
    ndim = 10
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "SparseActive"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    y = np.zeros(ne)
    for k in range(ne):
      arg0 = -220. * ((x[k][0] - 0.64)**2. + (x[k][1] - 0.18) **2.0)
      arg1 = -18.0 * (x[k][0] + x[k][1] -1.)**2.0
      y[k] = 1.10 * np.exp(arg0) + 0.70 * np.exp(arg1) - 0.04 * sum((x[k][2:] - 0.5)**2.)
    y *= -1.0
    y = y.reshape(ne, 1)
    return y

class InvalidComputingModelError(RuntimeError):
    pass
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='myprogram')
  parser.add_argument("--nx", type=int, default=2, help="dimension of problem")
  parser.add_argument("--boiter", type=int, default=2, help="BO iterations") 
  parser.add_argument("--bnbtol", type=float, default=0.01, help="abs tolerance for bnb optimizer")
  parser.add_argument("--relbnbtol", type=float, default=0.01, help="rel tolerance for bnb optimizer")
  parser.add_argument("--bnbmaxiter", type=int, default=1000, help="maximum number of bnb iterations") 
  parser.add_argument("--bnbmaxtime", type=float, default=180., help="maximum time for bnb opt") 
  parser.add_argument("--bnb", action=argparse.BooleanOptionalAction, type=bool, default=True, help="BnB or multistart")
  parser.add_argument("--nsamples", type=int, default=6, help="number of initial samples")
  parser.add_argument("--seed", type=int, default=42, help="random seed")
  parser.add_argument("--problem", type=str, default="Periodic", help="black-box objective") 
  parser.add_argument("--make_plts", action=argparse.BooleanOptionalAction, type=bool, default=False, help="create plots or not")
  parser.add_argument("--save_data", action=argparse.BooleanOptionalAction, type=bool, default=False, help="save data or not")
  parser.add_argument("--optmode", type=int, default=5, help="LCB convex relaxation strategy")
  parser.add_argument("--mpimode", action=argparse.BooleanOptionalAction, type=bool, default=False, help="enable MPI parallelism and use MPI_COMM_WORLD.Get_size()-1 workers.")
  parser.add_argument("--num_workers", type=int, default=0, help="specify number of workers for non-mpimode with default value zero in which case multiprocessing.cpu_count()-1 will be used.") 
  parser.add_argument("--bnb_warmstart", action=argparse.BooleanOptionalAction, type=bool, default=False, help="use a partition of BnB node from previous iteration to initialize BnB search.")
  parser.add_argument("--bnb_warmstart_nodes", type=int, default=1, help="max number of BnB nodes the warmstart partition should have")
  parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, type=bool, default=False, help="build and print diagnostics")
  args = parser.parse_args()

  executor = None
  if args.mpimode:
    # for MPI mode the number of workers will be MPI_COMM_WORLD.Get_size()-1
    #
    # Examples of srun commands with 64 MPI workers on node and on two nodes
    # srun -N1 -n65 -u -m mpi4py.futures BnBBoDriverEX_clean.py --mpimode [remaining_options]
    # srun -N2 -n65 -u -m mpi4py.futures BnBBoDriverEX_clean.py --mpimode [remaining_options]
    
    if args.num_workers>0:
      raise ValueError("option --num_workers should not be used or should be zero with --mpimode")
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
    import sys
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
      MPIPoolExecutor()  # Workers enter executor loop
      sys.exit(0)

    # Master (rank 0) continues here
    executor = MPIPoolExecutor()
  else:
    # for non-MPI mode one can specify the number of workers. Ideally this should be set
    # to total number of cores/threads minus 1
    #
    # Example of srun command with 64 workers 
    # srun -n1 -c65 python -u BnBBoDriverEX_clean.py  --num_workers 64 [remaining_options]
    
    from concurrent.futures import ProcessPoolExecutor
    if args.num_workers>0:
      executor = ProcessPoolExecutor(max_workers=args.num_workers)

    try:
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      num_ranks = comm.Get_size()
      if num_ranks > 1:
        raise InvalidComputingModelError("Multiple MPI ranks detected without option --mpimode being specified.")
    except InvalidComputingModelError:
      raise
    except Exception:
      pass
    
  nx = args.nx # dimension of the problem
  boiter = args.boiter
  bnbtol = args.bnbtol # tolerance for bnb optimizer
  relbnbtol = args.relbnbtol
  bnbmaxiter = args.bnbmaxiter
  bnbmaxtime = args.bnbmaxtime
  batch_size = 1
  randseed = args.seed
  n_samples = args.nsamples 
  problem_name = args.problem
  make_plts = args.make_plts
  random.seed(randseed)
  np.random.seed(randseed)
  save_data = args.save_data

  acquisition_type = 'LCB' 
  BnB = args.bnb
  assert problem_name in ["QuadraticShift", "Periodic", "Michalewicz", "Branin", "Hartmann", "HartmannLike", "Shekel", "SparseActive"], "unrecognized problem name"
  
  if problem_name == "QuadraticShift":
    c = 0.5 * np.ones(nx) #np.linspace(0.25, 0.75, num=nx)
    problem = QuadraticShift(ndim=nx, c=c)
  elif problem_name == "Periodic":
    problem = PeriodicObjective(ndim=nx)
  elif problem_name == "Michalewicz":
    problem = MichalewiczObjective(ndim=nx)
  elif problem_name == "Branin":
    problem = BraninProblem()
  elif problem_name == "HartmannLike":
    problem = HartmannLikeProblem()
  elif problem_name == "Shekel":
    problem = ShekelProblem()
  elif problem_name == "SparseActive":
    problem = SparseActiveProblem()
  else:
    problem = HartmannProblem()
  
  # overwrite the problem dim if Branin or Hartmann
  if problem.name in ["Branin", "Hartmann", "HartmannLike", "Shekel", "SparseActive"]:
    nx = problem.ndim
  problem.set_constraints([])  

  problem.set_seed(randseed)
  x_train = problem.sample(n_samples)
  y_train = problem.evaluate(x_train)
  
  theta = 1.0  # hyperparameter for GP kernel
  fix_theta = False
  theta_bounds = [0.05, 1.5]
  pow_exp_power = 2.0 #1. or 2., only relevant for pow_exp kernel
  corr = "pow_exp" #"matern52" # "pow_exp", "matern32", "matern52"
  eval_noise = False

  hyper_opt="Cobyla" #More robust, derivative-free hyperparameter optimization
  #hyper_opt="TNC" # Faster/default, but more sensitive
  # hyper_opt="NoOp" # Freeze theta at theta0, useful for debugging BnB/Clarabel issues
  if fix_theta:
    hyper_opt="NoOp"
 
  nugget = 1e-12

  gp_model = smtKRG(theta, problem.xlimits, nx, corr=corr, pow_exp_power=pow_exp_power, eval_noise=eval_noise, fix_theta=fix_theta, theta_bounds=theta_bounds, hyper_opt=hyper_opt, nugget=nugget)
  gp_model.train(x_train, y_train)

  beta = 3
  if acquisition_type == 'LCB':
    acqf = LCBacquisition(gp_model, beta=beta)
  else:
    acqf = EIacquisition(gp_model)

  save_data_dir = 'data08122026/'
  # problems with fixed dim or not
  if problem.name in ["Branin", "Hartmann", "HartmannLike", "Shekel", "SparseActive"]:
    save_data_dir = save_data_dir + problem.name + '/'
  else:
    save_data_dir = save_data_dir + problem.name + 'dim' + str(nx) + '/'
  Path(save_data_dir).mkdir(parents=True, exist_ok=True)
  bnb_solver_options = {
      'epsilon_prune' : 1.e-12,
      'abs_tol' : bnbtol,
      'rel_tol' : relbnbtol,
      'epsilon_diam' : bnbtol / 100.,
      'max_iter': bnbmaxiter,
      'max_bnbtime': bnbmaxtime,
      'pure_BBS' : True,
      'synchronous' : False,
      'save_data' : save_data,
      'save_data_dir' : save_data_dir,
      'acqf_ub_solver': 'IPOPT',
      'min_diameter': 0.001,
      'opt_mode': args.optmode,
      'random_seed': randseed,
      'bnb_warmstart': args.bnb_warmstart,
      'bnb_warmstart_nodes': args.bnb_warmstart_nodes,
      'diagnostics' : args.diagnostics,
  }
  bnb_solver_options['node_evaluator'] = MPIEvaluator(function_mode=False, executor=executor, task_name="BO_BNB_NODE", profiling=False)
  
  if problem.name == "Michalewicz":
    bnb_solver_options['min_diameter'] *= np.pi

  options = {
      'acquisition_type' : acquisition_type,
      'LCB_beta': beta,
      'bo_maxiter' : boiter, 
      'batch_size' : batch_size,
      'opt_solver' : 'SLSQP',
      'bnb_warmstart' : args.bnb_warmstart,
  }
  if BnB:
    options['opt_solver'] = 'BnB'
    options['solver_options'] = bnb_solver_options 

  options['executor'] = executor
  options['obj_evaluator'] = MPIEvaluator(function_mode=True, executor=executor, task_name="BO_OBJ", profiling=False)
  options['opt_evaluator'] = MPIEvaluator(function_mode=True, executor=executor, task_name="BO_OPT", profiling=False)
  
    
  start_time = time.perf_counter()
  bo = BOAlgorithm(problem, gp_model, x_train, y_train, options=options)
  bo.optimize()
  end_time = time.perf_counter()
  print(f"Elapsed time: {end_time - start_time} seconds")
  x_bo = bo.getOptimizationHistory()[0]
  x_train_superset = np.concatenate((x_train, x_bo), axis=0)

  optimal_thetas = []
  
  if nx == 1:
    X = np.linspace(problem.xlimits[:,0], problem.xlimits[:,1], num=100)
    Yobj = problem.evaluate(X) 
  if not BnB:
    save_data_dir = save_data_dir + 'multistart_'

  if save_data:
    acqf_min_vals = []
    for i in range(boiter):
      x_train2 = x_train_superset[:(-boiter+i)*batch_size]
      y_train2 = problem.evaluate(x_train2)
      gp_model2 = smtKRG(theta, problem.xlimits, nx, pow_exp_power=pow_exp_power, eval_noise=eval_noise, fix_theta=fix_theta, theta_bounds=theta_bounds, hyper_opt=hyper_opt)
      gp_model2.train(x_train2, y_train2)
      optimal_thetas.append(gp_model2.surrogatesmt.optimal_theta)
      if acquisition_type == "LCB":
        acqf2 = LCBacquisition(gp_model2, beta=beta)
      else:
        acqf2 = EIacquisition(gp_model2)
      for k in range(batch_size):
        acqf_min_vals.append(acqf2.evaluate(np.atleast_2d(x_bo[i*batch_size + k]))[0][0])
      if not make_plts:
        continue
      if nx == 1 and save_data:
        Y_acqf2 = [acqf2.evaluate(x)[0] for x in X]
        x_batch = np.array([x_bo[k] for k in range(i*batch_size, (i+1)*batch_size)])
        y_batch = acqf2.evaluate(np.atleast_2d(x_batch))
        plt.plot(X, Y_acqf2,'k--', label=r''+acquisition_type+'$(x)$')
        plt.plot(x_batch, y_batch, r'r*', markersize=12, label=r'batch minimizer')
        plt.xlabel("x")
        plt.legend()
        plt.title(r""+acquisition_type+"$(x)$ at BO iteration {0:d}".format(i))
        plt.savefig(save_data_dir + "acqf_BOit"+str(i)+".png")
        plt.close()
        # plot the GP
        muX = gp_model2.mean(X)
        sigmaX = np.sqrt(gp_model2.variance(X))
        plt.plot(X, Yobj, 'k', label=r'$f(x)$')
        plt.plot(X, muX, 'r--', label=r'$\mu(x)$')
        plt.fill_between(X.flatten(), (muX-sigmaX).flatten(), (muX + sigmaX).flatten(),
                         label=r'$\tilde{f}$ confidence region', alpha=0.25)
        plt.scatter(x_train2, y_train2, marker='o', s=30, c='magenta', label='training points')
        plt.xlabel("x")
        plt.legend()
        plt.savefig(save_data_dir + "GP_BOit" + str(i) + ".png")
        plt.close()
        dmuX = gp_model2.mean_gradient(X)
        #dvarianceX = gp_model2.variance_gradient(X)
        plt.plot(X.flatten(), dmuX.flatten())
        plt.scatter(x_train2, y_train2, marker='o', s=30, c='magenta', label='training points')
        plt.xlabel("x")
        plt.savefig(save_data_dir + "dmean" + str(i) + ".png")
        plt.close()
      elif nx == 2: 
        l = problem.xlimits[:, 0].astype(float)
        u = problem.xlimits[:, 1].astype(float)
        X1D = [np.linspace(l[k], u[k],  100) for k in range(nx)]
        Xx, Xy = np.meshgrid(X1D[0], X1D[1])
        Z = np.array([[acqf2.evaluate(np.atleast_2d([Xx[k, j], Xy[k, j]])).flatten()[0] for j in range(Xx.shape[1])] for k in range(Xx.shape[0])])
        plt.contourf(Xx, Xy, Z, levels=40, cmap='viridis')
        # x, y points from BO batch
        xpts_batch = np.array([x_bo[j][0] for j in range(i*batch_size, (i+1)*batch_size)])
        ypts_batch = np.array([x_bo[j][1] for j in range(i*batch_size, (i+1)*batch_size)])
        plt.plot(xpts_batch, ypts_batch, r'r*', markersize=12)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.colorbar(label=r'$\varphi(x,y)$, acquisition function')
        plt.savefig(save_data_dir + "acqf_BOit"+str(i)+".png")
        plt.close()
    np.savetxt(save_data_dir + 'optimal_thetas.dat', optimal_thetas)
    np.savetxt(save_data_dir + 'bnb_branches.dat', bo.bnb_num_branch_hist)
    np.savetxt(save_data_dir + 'bohist_xpts.dat', x_bo)
    np.savetxt(save_data_dir + 'init_xtrain.dat', x_train)
    np.savetxt(save_data_dir + 'init_obj.dat', y_train)
    np.savetxt(save_data_dir + 'bohist_obj.dat', problem.evaluate(x_bo))
    np.savetxt(save_data_dir + 'acqf_min_vals.dat', acqf_min_vals)
    if nx == 2 and make_plts:
      l = problem.xlimits[:, 0].astype(float)
      u = problem.xlimits[:, 1].astype(float)
      X1D = [np.linspace(l[i], u[i], 100) for i in range(nx)]
      Xx, Xy = np.meshgrid(X1D[0], X1D[1])
      Z = np.array([[problem.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
      plt.contourf(Xx, Xy, Z, levels=40, cmap='viridis')
      plt.plot((x_bo.T)[0], (x_bo.T)[1], r'r*', markersize=12, label = "BO samples")
      plt.plot(x_train[:,0], x_train[:,1], r'k*', markersize=12, label= "Initial samples")
      plt.xlabel(r'$x$')
      plt.ylabel(r'$y$')
      plt.colorbar(label=r'$f(x,y)$, objective function')
      plt.legend()
      plt.savefig(save_data_dir + "obj.png")
      plt.close()
