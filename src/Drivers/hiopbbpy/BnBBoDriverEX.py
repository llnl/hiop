#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from hiopbbpy.problems import Problem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithm, BOAlgorithm, LCBacquisition, EIacquisition
from hiopbbpy.utils import MPIEvaluator
import argparse

class QuadraticShift(Problem):
  """
  f(x) = ||x - c||^2
  - Global minimizer: x* = c
  - Global minimum:   f* = 0
  """
  def __init__(self, ndim=2, xlimits=None, c=None, constraints=[]):
    if xlimits is None:
      xlimits = np.array([[-5.0, 5.0]] * ndim, dtype=float)
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='myprogram')
  parser.add_argument("--nx", type=int, default=2, help="dimension of problem")
  parser.add_argument("--bnbtol", type=float, default=0.01, help="tolerance for bnb optimizer")
  parser.add_argument("--bnbmaxiter", type=int, default=1000, help="maximum number of bnb iterations") 
  parser.add_argument("--bnbmaxtime", type=float, default=180., help="maximum time for bnb opt") 
  args = parser.parse_args()
  nx = args.nx # dimension of the problem
  bnbtol = args.bnbtol # tolerance for bnb optimizer
  bnbmaxiter = args.bnbmaxiter
  bnbmaxtime = args.bnbmaxtime

  acquisition_type = 'LCB'  
  ### parameters
  #n_samples = 50  # number of the initial samples to train GP
  n_samples = 10
  theta = 1.e-2  # hyperparameter for GP kernel
  c = np.linspace(1.25, 2.5, num=nx)
  #c2D=np.array([1.25, 2.5])
  #c1D=np.array([1.25])
  #if nx == 1:
  #  #delta = 10.0
  #  #xlimits = np.array([[0.75 - delta/2.,0.75 + delta/2.]])
  #  #c1D = np.array([0.75])
  #  #c = c1D
  #  c = np.array([0.75])
  #elif nx == 2:
  #  c = np.array([1.25, 2.5])
  #elif nx == 4:
  #  c = np.array([0.25, 0.75, 1.25, 1.75])
  #elif nx == 8:
  #  c = np.array([-3.5 + i for i in range(8)])
  #  #c = c2D
  
  
  problem = QuadraticShift(ndim=nx, c=c)
  problem.set_constraints([])  
      
  x_train = problem.sample(n_samples)
  y_train = problem.evaluate(x_train)
  
  gp_model = smtKRG(theta, problem.xlimits, nx)
  gp_model.train(x_train, y_train)
  
  
  if acquisition_type == 'LCB':
    acqf = LCBacquisition(gp_model)
  else:
    acqf = EIacquisition(gp_model)
  
  
  solver_options = {
      'epsilon_diam' : 1.e-14,
      'epsilon_gap' : bnbtol,    
      'max_iter': bnbmaxiter,
      'max_bnbtime': bnbmaxtime,
      'nodes_per_batch' : 32
  }
  bnb = BnBAlgorithm(acqf, options=solver_options)
  bnb.initialize()
  xstar = np.atleast_2d(bnb.optimize())
  ystar = acqf.evaluate(xstar)
  num_branches = bnb.num_branches
  print("number of branches in bnb algorithm = ", num_branches)

  
  #queue = bnb.queue
  ##print(queue)  
  ##print(queue[2][2].l, queue[2][2].u)
  #Xqueue = np.atleast_2d([(queue[i][2].l + queue[i][2].u) / 2. for i in range(len(queue))])
  #Yqueue = acqf.evaluate(Xqueue)
  ##print(Xqueue)
  ##from scipy.cluster.hierarchy import linkage, dendrogram
  ##
  ##linked = linkage(Xqueue, method='ward')
  ##for item in dir(linked):
  ##  print(item)
  ##from sklearn.cluster import DBSCAN
  ##clustering = DBSCAN(eps=0.05).fit(Xqueue)
  ##labels = np.unique(clustering.labels_)
  ##print("{0:d} unique clustering groups".format(len(labels)))
  #Xqueue = Xqueue.flatten()
  #Yqueue = Yqueue.flatten()
  if False and nx == 1:
    l = problem.xlimits[:, 0].astype(float)
    u = problem.xlimits[:, 1].astype(float)
    n_plot_pts = 1000
    X = np.atleast_2d(np.linspace(l[0], u[0], n_plot_pts)).transpose()
    Yacqf = [acqf.evaluate(x)[0] for x in X]
    
    plt.plot(X, Yacqf, 'k--', label=r'' + acquisition_type + '$(x)$')
    plt.plot(xstar, ystar, "r*", markersize=14, label=r'bnb minimizer')
    #for label in labels:
    #  args = np.argwhere(label == clustering.labels_)
    #  plt.plot(Xqueue[args], Yqueue[args], "*", markersize=12)
    plt.legend()
    plt.show()
  ##exit()

  
  
  #bo_maxiter = 1
  #batch_size = 1
  #options = {
  #    'acquisition_type': acquisition_type,
  #    'bo_maxiter': bo_maxiter, 
  #    'batch_size': batch_size,
  #    'opt_solver': 'BnB',
  #    'solver_options' : solver_options,
  #}
  #
  #bo = BOAlgorithm(problem, gp_model, x_train, y_train, options=options)
  #bo.optimize()
  #x_bo = bo.getOptimizationHistory()[0]
  #x_train_superset = np.concatenate((x_train, x_bo), axis=0)
  #for i in range(bo_maxiter):
  #  x_train2 = x_train_superset[:-bo_maxiter+i]
  #  y_train2 = problem.evaluate(x_train2)
  #  gp_model2 = smtKRG(theta, problem.xlimits, nx)
  #  gp_model2.train(x_train2, y_train2)
  #  if acquisition_type == "LCB":
  #    acqf2 = LCBacquisition(gp_model2)
  #  else:
  #    acqf2 = EIacquisition(gp_model2)
  #  Y_acqf2 = [acqf2.evaluate(x)[0] for x in X]
  #  y_star2 = acqf2.evaluate(np.atleast_2d(x_bo[i]))
  #  plt.plot(X, Y_acqf2,'k--', label=r''+acquisition_type+'$(x)$')
  #  plt.plot(x_bo[i], y_star2, r'r*', markersize=12, label=r'bnb minimizer')
  #  plt.xlabel("x")
  #  plt.legend()
  #  plt.title(r""+acquisition_type+"$(x)$ at BO iteration {0:d}".format(i))
  #  plt.show()
