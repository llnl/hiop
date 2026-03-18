#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from hiopbbpy.problems import Problem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithm, BOAlgorithm, LCBacquisition, EIacquisition
from hiopbbpy.utils import MPIEvaluator
import argparse
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#class acqf_bound_plotter):
#  def __init__(self, nodes):
#    self.nodes = nodes
#  def upper_bound_eval(x):
#    arg = 0
#    found_node = False
#    for i, node in enumerate(self.nodes):
#      if np.all(node.l <= x) and np.all(x <= node.u):
#        arg = i
#        found_node = True
#      if found_node:
#        break
#    return self.nodes[arg].aq_U
#  def upper_bound(X):
#    if len(X.shape) == 2:
      
  



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
  f(x) = \sum_{i in dim(x)} \sin(2\pi * x_i)
  - Global minimizer: x* = (3/8,) * dim(x) + (1/2) * \mathbb{Z}^{dim(x)}
  - Global minimum:   f* = -dim(x)
  """
  def __init__(self, ndim=2, xlimits=None, constraints=[]):
    if xlimits is None:
      xlimits = np.array([[0.0, 1.0]] * ndim, dtype=float)
    name = "PeriodicObjective"
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
    # expose known solution for checking later
    self.x_star = np.array([0.375,] * ndim)
    self.f_star = -1.0 * ndim 

  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, _ = x.shape
    y = np.sum(np.sin(4. * np.pi * x), axis=1).reshape(ne, 1)
    return y


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='myprogram')
  parser.add_argument("--nx", type=int, default=2, help="dimension of problem")
  parser.add_argument("--boiter", type=int, default=2, help="BO iterations") 
  parser.add_argument("--bnbtol", type=float, default=0.01, help="tolerance for bnb optimizer")
  parser.add_argument("--bnbmaxiter", type=int, default=1000, help="maximum number of bnb iterations") 
  parser.add_argument("--bnbmaxtime", type=float, default=180., help="maximum time for bnb opt") 
  args = parser.parse_args()
  nx = args.nx # dimension of the problem
  boiter = args.boiter
  bnbtol = args.bnbtol # tolerance for bnb optimizer
  bnbmaxiter = args.bnbmaxiter
  bnbmaxtime = args.bnbmaxtime
  random.seed(42)

  acquisition_type = 'LCB'  
  plot_acquisition = True
  ### parameters
  n_samples = 25  # number of the initial samples to train GP
  if nx == 1:
    n_samples = 5
  if nx == 2:
    n_samples = 8
  if nx == 3:
    n_samples = 14
  #n_samples = 10
  
  
  #c = 0.5 * np.ones(nx) #np.linspace(0.25, 0.75, num=nx)
  #problem = QuadraticShift(ndim=nx, c=c)
  problem = PeriodicObjective(ndim=nx)
  problem.set_constraints([])  
      
  x_train = problem.sample(n_samples)
  y_train = problem.evaluate(x_train)
  
  theta = 1.e0  # hyperparameter for GP kernel
  gp_model = smtKRG(theta, problem.xlimits, nx, pow_exp_power=1.0, eval_noise=False)
  gp_model.train(x_train, y_train)
  print("optimal theta = ", gp_model.surrogatesmt.optimal_theta) 
  
  if acquisition_type == 'LCB':
    acqf = LCBacquisition(gp_model)
  else:
    acqf = EIacquisition(gp_model)
  
  
  bnb_solver_options = {
      'epsilon_prune' : 1.e-12,
      'epsilon_gap' : bnbtol, 
      'epsilon_diam' : bnbtol / 100.,
      'max_iter': bnbmaxiter,
      'max_bnbtime': bnbmaxtime,
      'nodes_per_batch' : 32,
      'pure_BBS' : True,
      'sync_mode' : True,
  }

  batch_size = 1
  options = {
      'acquisition_type': acquisition_type,
      'bo_maxiter': boiter, 
      'batch_size': batch_size,
      'opt_solver': 'BnB',
      'solver_options' : bnb_solver_options,
  }
  bo = BOAlgorithm(problem, gp_model, x_train, y_train, options=options)
  bo.optimize()
  x_bo = bo.getOptimizationHistory()[0]
  x_train_superset = np.concatenate((x_train, x_bo), axis=0)

  optimal_thetas = []
  for i in range(boiter):
    x_train2 = x_train_superset[:-boiter+i]
    y_train2 = problem.evaluate(x_train2)
    gp_model2 = smtKRG(theta, problem.xlimits, nx)
    gp_model2.train(x_train2, y_train2)
    print("optimal theta = ", gp_model2.surrogatesmt.optimal_theta)
    optimal_thetas.append(gp_model2.surrogatesmt.optimal_theta)
    if acquisition_type == "LCB":
      acqf2 = LCBacquisition(gp_model2)
    else:
      acqf2 = EIacquisition(gp_model2)
    if nx == 1:
      X = np.linspace(problem.xlimits[:,0], problem.xlimits[:,1], num=100)
      Y_acqf2 = [acqf2.evaluate(x)[0] for x in X]
      y_star2 = acqf2.evaluate(np.atleast_2d(x_bo[i]))
      plt.plot(X, Y_acqf2,'k--', label=r''+acquisition_type+'$(x)$')
      plt.plot(x_bo[i], y_star2, r'r*', markersize=12, label=r'bnb minimizer')
      plt.xlabel("x")
      plt.legend()
      plt.title(r""+acquisition_type+"$(x)$ at BO iteration {0:d}".format(i))
      plt.savefig("acqf_BOit"+str(i)+".png")
      plt.close()
    elif nx == 2:
      l = problem.xlimits[:, 0].astype(float)
      u = problem.xlimits[:, 1].astype(float)
      X1D = [np.linspace(l[i], u[i],  200) for i in range(nx)]
      Xx, Xy = np.meshgrid(X1D[0], X1D[1])
      Z = np.array([[acqf2.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
      plt.contourf(Xx, Xy, Z, levels=40, cmap='viridis')
      plt.plot(x_bo[i][0], x_bo[i][1], r'r*', markersize=12)
      plt.xlabel(r'$x$')
      plt.ylabel(r'$y$')
      plt.colorbar(label=r'$\varphi(x,y)$, acquisition function')
      plt.savefig("acqf_BOit"+str(i)+".png")
      plt.close()
