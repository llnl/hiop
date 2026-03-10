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
  parser.add_argument("--ntrain", type=int, default=3, help="number of initial GP training points")
  parser.add_argument("--make_plts", type=int, default=0, help="save plots or not")
  parser.add_argument("--opt_mode", type=int, default=2, help="optimization mode")
  parser.add_argument("--nugget", type=float, default=2.220446049250313e-14, help="nugget added to posterior GP for numerical stability")
  parser.add_argument("--theta0", type=float, default=1.0, help="initial value of GP kernel hyperparameter theta")
  parser.add_argument("--lengthscale", type=float, default=1.0, help="global length scale for the problem")
  args = parser.parse_args()
  
  # parse arguments
  nx = args.nx # dimension of the problem
  nugget = args.nugget
  opt_mode = args.opt_mode
  make_plts = args.make_plts
  ntrain = args.ntrain
  theta0 = args.theta0
  lengthscale = args.lengthscale
  

  save_dir = 'newdata/opt_mode'+str(opt_mode) + '/'
  
  # ---- black-box objective
  xlimits = np.array([[0., 1.]] * nx )
  xlimits *= lengthscale
  problem = PeriodicObjective(ndim=nx, xlimits=xlimits)
  problem.set_constraints([])  
  
  # ---- generate GP training data
  problem.set_seed(42)
  x_train = problem.sample(ntrain)
  y_train = problem.evaluate(x_train)
  
  # ---- setup GP
  gp_model = smtKRG(theta0, problem.xlimits, nx, pow_exp_power=1.0, eval_noise=False)
  gp_model.train(x_train, y_train)
   
 
  # ---- acqf
  beta = 3.0  
  acqf = LCBacquisition(gp_model, beta=beta)
     
  l = problem.xlimits[:, 0].astype(float)
  u = problem.xlimits[:, 1].astype(float)
  
  if make_plts: 
    if nx == 1: 
      # evaluation points for plotting
      n_plot_pts = 1000
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
      plt.ylabel("LCB(x)")
      plt.legend()
      plt.title("acquisition function")
      plt.savefig(save_dir + 'acqf.png')
      plt.close()
      import matplotlib.pyplot as plt2
      plt.plot(X, Y_true, 'k', label=r'$f(x)$')
      plt.plot(X, muX, 'r--', label=r'$\mu(x)$')
      plt2.fill_between(X.flatten(), (muX-sigmaX).flatten(), (muX + sigmaX).flatten(),
                       label=r'$\tilde{f}$ confidence region', alpha=0.25)
      plt.scatter(x_train, y_train, marker='o', s=30, c='magenta', label='training points')
      plt.xlabel("x")
      plt.legend()
      plt.savefig(save_dir + 'trainingdata.png')
      plt.close()
    elif nx == 2:
      X1D = [np.linspace(l[i], u[i],  100) for i in range(nx)]
      Xx, Xy = np.meshgrid(X1D[0], X1D[1])
      Z = np.array([[acqf.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
      plt.contourf(Xx, Xy, Z, levels=25, cmap='viridis')
      plt.colorbar(label='Acquisition function value')
      plt.scatter(x_train[:,0], x_train[:,1], c="red")
      plt.savefig(save_dir + 'acqf.png')
      plt.close()
  # In[ ]:
  
  
  bnb_options = {
      'opt_mode' : opt_mode,
      'acqf_ub_solver' : "SLSQP",
  }
  bnb = BnBAlgorithm(acqf, bnb_options)
  bnb.initialize()
  root = bnb.best_node
  
  
  # In[ ]:
  
  
  from hiopbbpy.opt.bnbalgorithm import branch
  nodes = [root]
 
  num_divisions = 7
  if nx == 2: 
    num_divisions = 3
  if nx == 3:
    num_divisions = 3
  num_branches = 1
  LUBgaps = np.zeros(num_divisions)
  min_gaps = np.zeros(num_divisions)
  max_gaps = np.zeros(num_divisions)
  avg_gaps = np.zeros(num_divisions)
  std_gaps = np.zeros(num_divisions)
  pruning_ratios = np.zeros(num_divisions)
  all_all_gaps = []
  
  for j in range(num_divisions):
    for i in range(nx):
      children = []
      for node in nodes:
        #TODO: dimension dependent splitting!
        # need to split nx times to take the hypercube
        # of side length L to 2^(nx) hypercubes, each of
        # side length L / 2
        for child_l, child_u in branch(node.l, node.u):
          acqf_L, acqf_U = bnb.compute_acqf_bounds(child_l, child_u)
          if acqf_U < acqf_L:
            acqf_L = acqf_U - 1.e-12
          child = BnBNode(child_l, child_u, acqf_L, acqf_U)
          children.append(child)
      num_branches += len(children)
      nodes = children
    min_idx = np.argmin([child.aq_U for child in children])        
    LUB = children[min_idx].aq_U
    LUBgap = children[min_idx].aq_U - children[min_idx].aq_L
    all_gaps = np.array([child.aq_U - child.aq_L for child in children])
    min_gaps[j] = min(all_gaps)
    max_gaps[j] = max(all_gaps)
    avg_gaps[j] = np.mean(all_gaps)
    std_gaps[j] = np.std(all_gaps)
    LUBgaps[j] = LUBgap
    all_all_gaps.append(all_gaps)
    
    pruning_ratio = 0.
    for i, child in enumerate(children):
      if child.aq_L > LUB:
        pruning_ratio += 1.
    pruning_ratio /= len(children)
    pruning_ratios[j] = pruning_ratio
    for i, child in enumerate(children):
      xplt = np.linspace(child.l[0], child.u[0], num=10)
      acqf_UB = child.aq_U * np.ones(10)
      acqf_LB = child.aq_L * np.ones(10)
      if i == min_idx:
        ub_color = 'r'
        lb_color = 'r'
      elif child.aq_L > LUB:
        ub_color = 'k'
        lb_color = 'g'
      else:
        ub_color = 'k'
        lb_color = 'm'
     
      if make_plts and nx == 1: 
        if i == 0:
          plt.plot(xplt, acqf_UB, "--" + ub_color, label="acqf UB")
          plt.plot(xplt, acqf_LB, "-" + lb_color, label="acqf LB")
        else:
          plt.plot(xplt, acqf_UB, "--" + ub_color)
          plt.plot(xplt, acqf_LB, "-" + lb_color)
      if make_plts and nx == 2:
        X1D = [np.linspace(child.l[i], child.u[i],  40) for i in range(nx)]
        Xx, Xy = np.meshgrid(X1D[0], X1D[1])
        Z = np.array([[acqf.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] - child.aq_L for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
        plt.contourf(Xx, Xy, Z, levels=25, cmap='viridis')
        print("lower bound = ", child.aq_L)
        print("upper bound = ", child.aq_U)
        print("minimum (sample) acqf = ", min(Z.flatten()) + child.aq_L)
        print("gap = ", child.aq_U - child.aq_L)
        plt.colorbar(label='Acquisition function - acqf LB')
        plt.scatter(x_train[:,0], x_train[:,1], c='red')
        plt.show()
        
         
    nodes = children
    if make_plts and nx == 1:
      plt.plot(X, Y_acqf, label=r'$LCB(x)$')
      plt.title('gap = {0:1.2e}, num_branches = {1:d}, pruning_ratio = {2:1.3f}'.format(
          LUBgap, num_branches, pruning_ratio))
      plt.legend()
      plt.savefig(save_dir + 'ublb'+str(j) + 'divisions.png')
      plt.close()
    #if make_plts and nx == 2:
    #  plt.colorbar(label='Acquisition function - acqf LB')
    #  plt.scatter(x_train[:,0], x_train[:,1], c='red')
    #  plt.show()
  np.savetxt(save_dir + 'LUBgapsvsdivisions.dat', LUBgaps)
  np.savetxt(save_dir + 'avggapsvsdivisions.dat', avg_gaps)
  np.savetxt(save_dir + 'stdgapsvsdivisions.dat', std_gaps)
  np.savetxt(save_dir + 'pruningratios.dat', pruning_ratios)
  for i in range(num_divisions):
    np.savetxt(save_dir + 'allgaps_'+str(i)+'.dat', all_all_gaps[i])
  np.savetxt(save_dir + 'gamma.dat', gp_model.surrogatesmt.optimal_par["gamma"])
  np.savetxt(save_dir + 'sigma2.dat', gp_model.surrogatesmt.optimal_par["sigma2"])
  np.savetxt(save_dir + 'opttheta.dat', gp_model.surrogatesmt.optimal_theta) 
  correlation_matrix = gp_model.surrogatesmt.optimal_par["C"] @ gp_model.surrogatesmt.optimal_par["C"]  
  corr_mat_cond = np.linalg.cond(correlation_matrix)
  np.savetxt(save_dir + 'Rcond.dat', np.array([corr_mat_cond]))
