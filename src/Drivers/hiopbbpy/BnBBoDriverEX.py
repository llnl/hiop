#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from hiopbbpy.problems import Problem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithm, LCBacquisition, EIacquisition
from hiopbbpy.opt import BOAlgorithm
from hiopbbpy.utils.util import Evaluator
from hiopbbpy.utils import MPIEvaluator


if __name__ == "__main__":
  nx = 1         # dimension of the problem
  acquisition_type = 'LCB'  
  ### parameters
  n_samples = 10  # number of the initial samples to train GP
  theta = 1.e-2  # hyperparameter for GP kernel
  c2D=np.array([1.25, 2.5])
  c1D=np.array([1.25])
  if nx == 1:
      delta = 10.0
      xlimits = np.array([[0.75 - delta/2.,0.75 + delta/2.]])
      c1D = np.array([0.75])
      c = c1D
  elif nx == 2:
      c = c2D
  
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
  
  if nx == 1:
      problem = QuadraticShift(ndim=nx, xlimits=xlimits, c=c)
  else:
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
  
  evaluator = MPIEvaluator(function_mode=False)
  
  solver_options = {
      'epsilon_diam' : 1.e-14,
      'epsilon_gap' : 1.e-2,    
      'max_iter': 300,
      'nodes_per_batch' : 4,
      'evaluator': evaluator
  }
  bnb = BnBAlgorithm(acqf, options=solver_options)
  xstar = bnb.optimize()
  ystar = acqf.evaluate(xstar)
  
  
  # In[10]:
  l = problem.xlimits[:, 0].astype(float)
  u = problem.xlimits[:, 1].astype(float)
  n_plot_pts = 1000
  X = np.atleast_2d(np.linspace(l[0], u[0], n_plot_pts)).transpose()
  Yacqf = [acqf.evaluate(x)[0] for x in X]
  
  plt.plot(X, Yacqf, 'k--', label=r'' + acquisition_type + '$(x)$')
  plt.plot(xstar, ystar, "r*", markersize=14, label=r'bnb minimizer')
  plt.legend()
  plt.show()
  
  
  # In[ ]:
  
  
  bo_maxiter = 3
  batch_size = 1
  options = {
      'acquisition_type': acquisition_type,
      'bo_maxiter': bo_maxiter, 
      'batch_size': batch_size,
      'opt_solver': 'BnB',
      'solver_options' : solver_options,
  }
  
  
  # In[ ]:
  
  
  bo = BOAlgorithm(problem, gp_model, x_train, y_train, options=options)
  bo.optimize()
  x_bo = bo.getOptimizationHistory()[0]
  x_train_superset = np.concatenate((x_train, x_bo), axis=0)
  for i in range(bo_maxiter):
      x_train2 = x_train_superset[:-bo_maxiter+i]
      y_train2 = problem.evaluate(x_train2)
      gp_model2 = smtKRG(theta, problem.xlimits, nx)
      gp_model2.train(x_train2, y_train2)
      if acquisition_type == "LCB":
        acqf2 = LCBacquisition(gp_model2)
      else:
        acqf2 = EIacquisition(gp_model2)
      Y_acqf2 = [acqf2.evaluate(x)[0] for x in X]
      y_star2 = acqf2.evaluate(np.atleast_2d(x_bo[i]))
      plt.plot(X, Y_acqf2,'k--', label=r''+acquisition_type+'$(x)$')
      plt.plot(x_bo[i], y_star2, r'r*', markersize=12, label=r'bnb minimizer')
      plt.xlabel("x")
      #plt.ylabel("LCB(x)")
      plt.legend()
      plt.title(r""+acquisition_type+"$(x)$ at BO iteration {0:d}".format(i))
      plt.show()
  #    
  #    
  #
  #
  ## In[ ]:




