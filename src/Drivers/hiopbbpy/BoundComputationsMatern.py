#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.size"] = 12
import argparse

from hiopbbpy.problems import Problem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithmBase, BnBAlgorithm, LCBacquisition, EIacquisition, BnBNode



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
    #in_range = [x >= 0.25 and x <= 0.75]
    in_range = [5./12. <= x[i,0] <= 11./12. for i in range(len(x))]
    y_update = np.zeros(y.shape)
    for i in range(len(y)):
      y_update[i,0] = y[i, 0]
      #if in_range[i]:
      #  y_update[i, 0] *= np.cos(24. * np.pi * x[i,0])
    return y_update


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='myprogram')
  parser.add_argument("--nx", type=int, default=1, help="dimension of problem")
  parser.add_argument("--ntrain", type=int, default=7, help="number of initial GP training points")
  parser.add_argument("--make_plts", type=int, default=1, help="create plots or not")
  parser.add_argument("--theta0", type=float, default=5.0, help="initial value of GP kernel hyperparameter theta")
  parser.add_argument("--optmode", type=int, default=5, help="LCB convex relaxation strategy")
  args = parser.parse_args()
  
  # parse arguments
  nx = args.nx # dimension of the problem
  make_plts = args.make_plts
  ntrain = args.ntrain
  theta0 = args.theta0
  

  # ---- black-box objective
  xlimits = np.array([[0., 1.]] * nx )
  problem = PeriodicObjective(ndim=nx, xlimits=xlimits)
  problem.set_constraints([])  
  
  # ---- generate GP training data
  problem.set_seed(42)
  x_train = problem.sample(ntrain)
  y_train = problem.evaluate(x_train)
  
  # ---- setup GP
  fix_theta = True
  corr = "matern52"
  #corr = "pow_exp"
  pow_exp_power = 2.0
  gp_model = smtKRG(theta0, problem.xlimits, nx, corr=corr, pow_exp_power=pow_exp_power, eval_noise=False, fix_theta=fix_theta)
  gp_model.train(x_train, y_train)
 
  # ---- acqf
  beta = 3.0  
  acqf = LCBacquisition(gp_model, beta=beta)

   
  l = problem.xlimits[:, 0].astype(float)
  u = problem.xlimits[:, 1].astype(float)
  
  if make_plts: 
    if nx == 1: 
      # evaluation points for plotting
      n_plot_pts = 400
      X = np.atleast_2d(np.linspace(l[0], u[0], n_plot_pts)).transpose()
      # ---- evaluate black-box objective
      Y_true = problem.evaluate(X)
      # ---- evaluate GP mean and pointwise variance
      muX = gp_model.mean(X)
      sigmaX = np.sqrt(gp_model.variance(X))
      # ---- evaluate aquisition function
      Y_acqf = acqf.evaluate(X)
      
      # ---- plot
      plt.plot(X, Y_acqf, label=r'$LCB(x)$')
      plt.plot(X, muX, "-.", label=r'$\mu(x)$')
      plt.plot(X, sigmaX, label=r'$\sigma(x)$')
      plt.xlabel("x")
      if isinstance(acqf, LCBacquisition):
        plt.ylabel("LCB(x)")
      elif isinstance(acqf, EIacquisition):
        plt.ylabel("EI(x)")
      plt.legend()
      plt.title("acquisition function")
      plt.show()
      plt.close()
      import matplotlib.pyplot as plt2
      plt.plot(X, Y_true, 'k', label=r'$f(x)$')
      plt.plot(X, muX, 'r--', label=r'$\mu(x)$')
      plt2.fill_between(X.flatten(), (muX-sigmaX).flatten(), (muX + sigmaX).flatten(),
                       label=r'$\tilde{f}$ confidence region', alpha=0.25)
      plt.scatter(x_train, y_train, marker='o', s=30, c='magenta', label='training points')
      plt.xlabel("x")
      plt.legend()
      plt.show()
      plt.close()
    elif nx == 2:
      X1D = [np.linspace(l[i], u[i],  100) for i in range(nx)]
      Xx, Xy = np.meshgrid(X1D[0], X1D[1])
      Z = np.array([[acqf.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
      plt.contourf(Xx, Xy, Z, levels=25, cmap='viridis')
      plt.colorbar(label='Acquisition function value')
      plt.scatter(x_train[:,0], x_train[:,1], c="red")
      plt.show()
      plt.close()
  
  # ---- changing the convex relaxation
  bnb_options = {
      'epsilon_prune' : 1.e-12,
      'pure_BBS' : True,
      'synchronous' : True,
      'opt_mode': args.optmode,
      'early_stopping_heuristics' : False,
      'save_data' : False,
      'acqf_ub_solver': 'IPOPT',
      'min_diameter': 0.001,
  }
 

  bnb = BnBAlgorithm(acqf, bnb_options)
  bnb.initialize()
  root = bnb.best_node
  
  from hiopbbpy.opt.bnbalgorithm import branch
  nodes = [root]
 
  num_divisions = 10
  if nx == 2: 
    num_divisions = 3
  if nx == 3:
    num_divisions = 3
  num_branches = 1
  LUBgaps = np.zeros(num_divisions)
  mingaps = np.zeros(num_divisions)
  avggaps = np.zeros(num_divisions)
  maxgaps = np.zeros(num_divisions)


  for j in range(num_divisions):
    for i in range(nx):
      children = []
      for node in nodes:
        for child_l, child_u in branch(node.l, node.u):
          output = bnb.compute_acqf_bounds(child_l, child_u)
          acqf_L = output[0]
          acqf_U = output[1]
          child = BnBNode(child_l, child_u, acqf_L, acqf_U)
          children.append(child)
          #gap_ratio = abs(acqf_U - acqf_L) / abs(acqf_U - acqf_L_old)
          #print("|acqf_U - acqf_L| / |acqf_U - acqf_L (old)| = ", gap_ratio)
      num_branches += len(children)
      nodes = children
      all_gaps = [abs(kid.aq_L - kid.aq_U) for kid in children]
      min_gap = min(all_gaps)
      max_gap = max(all_gaps)
      mingaps[j] = min_gap
      maxgaps[j] = max_gap
      avggaps[j] = np.mean(all_gaps)
      filename = "data08072026/all_gaps_"+corr
      if corr == "pow_exp":
        filename = filename + "_p"+str(pow_exp_power)
      filename = filename + "_div"+str(j)+".dat"
      np.savetxt(filename, all_gaps)
    if make_plts:
      plt.plot(X, Y_acqf, label=r'$LCB(x)$')
      for i in range(len(children)):
        child = children[i]
        nplt_pts = 10 # number of points to use for plotting straight lines
        xplt = np.linspace(child.l[0], child.u[0], num=nplt_pts)
        acqf_LB = child.aq_L * np.ones(nplt_pts)
        acqf_UB = child.aq_U * np.ones(nplt_pts)
        #acqf_LB_old = child_old.aq_L * np.ones(nplt_pts)
        # only include label for first child in children list
        if i == 0:
          plt.plot(xplt, acqf_LB, '--k', label=r'acqf LB ({0:d})'.format(args.optmode))
          plt.plot(xplt, acqf_UB, '-r', label=r'acqf UB')
          plt.title('min/max gap = {0:1.2e}/{1:1.2e}'.format(min_gap, max_gap))
          #plt.plot(xplt, acqf_LB_old, '-r', label=r'acqf LB ({0:d})'.format(opt_mode_old))
        else:
          plt.plot(xplt, acqf_LB, '--k')
          plt.plot(xplt, acqf_UB, '-r')
          #plt.plot(xplt, acqf_LB_old, '-r')
      plt.legend()
      plt.show()
    nodes = children
  plt.plot(mingaps, label='min gap')
  plt.plot(maxgaps, label='max gap')
  plt.plot(avggaps, label='avg gap')
  plt.xlabel('number of divisions')
  plt.legend()
  plt.show()
  plt.plot(mingaps, label='min gap')
  plt.plot(maxgaps, label='max gap')
  plt.plot(avggaps, label='avg gap')
  plt.xlabel('number of divisions')
  plt.yscale('log')
  plt.legend()
  plt.show()
