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
  parser.add_argument("--bnbtol", type=float, default=0.01, help="tolerance for bnb optimizer")
  parser.add_argument("--bnbmaxiter", type=int, default=1000, help="maximum number of bnb iterations") 
  parser.add_argument("--bnbmaxtime", type=float, default=180., help="maximum time for bnb opt") 
  args = parser.parse_args()
  nx = args.nx # dimension of the problem
  bnbtol = args.bnbtol # tolerance for bnb optimizer
  bnbmaxiter = args.bnbmaxiter
  bnbmaxtime = args.bnbmaxtime
  random.seed(42)

  acquisition_type = 'LCB'  
  file_beg = ''
  ### parameters
  n_samples = 25  # number of the initial samples to train GP
  if nx == 1:
    n_samples = 20
  #n_samples = 10
  theta = 1.e-2  # hyperparameter for GP kernel
  c = 0.5 * np.ones(nx) #np.linspace(0.25, 0.75, num=nx)
  
  
  #problem = QuadraticShift(ndim=nx, c=c)
  problem = PeriodicObjective(ndim=nx)
  problem.set_constraints([])  
      
  x_train = problem.sample(n_samples)
  #train_pts = problem.sample(n_samples)
  #x_box = np.linspace(0.49, 0.51, num=3)
  #x_train = [[x, y] for x in x_box for y in x_box] + [X for X in train_pts]
  #x_train = np.array(x_train)
  ##for x in x_train:
  ##  print(x)
  ##print(x_train.shape)
  ##exit()
  y_train = problem.evaluate(x_train)
  
  gp_model = smtKRG(theta, problem.xlimits, nx)
  gp_model.train(x_train, y_train)
  
  
  if acquisition_type == 'LCB':
    acqf = LCBacquisition(gp_model)
  else:
    acqf = EIacquisition(gp_model)
  
  
  solver_options = {
      'epsilon_prune' : 1.e-12,
      'epsilon_gap' : bnbtol, 
      'epsilon_diam' : bnbtol / 100.,
      'max_iter': bnbmaxiter,
      'max_bnbtime': bnbmaxtime,
      'nodes_per_batch' : 32,
      'pure_BBS' : False,
      'sync_mode' : False,
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
  if nx == 1 and False:
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
  if nx == 2:
    l = problem.xlimits[:, 0].astype(float)
    u = problem.xlimits[:, 1].astype(float)
    X1D = [np.linspace(l[i], u[i],  200) for i in range(nx)]
    Xx, Xy = np.meshgrid(X1D[0], X1D[1])
    Z = np.array([[acqf.evaluate(np.atleast_2d([Xx[i, j], Xy[i, j]])).flatten()[0] for j in range(Xx.shape[1])] for i in range(Xx.shape[0])])
    plt.contourf(Xx, Xy, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Acquisition function value')
    plt.savefig('acqf.png')
    plt.close()

    # TODO: plot acqf upper bound - acqf lower bound on sequence of grids to see convergence

  plt.plot(bnb.branch_history, bnb.gap_history, label='gap')
  if len(bnb.branch_history) == len(bnb.prunedvol_history):
    plt.plot(bnb.branch_history, 1. - np.array(bnb.prunedvol_history), label='1. - pruned vol')
  if len(bnb.branch_history) == len(bnb.pruningratio_history):
    plt.plot(bnb.branch_history, bnb.pruningratio_history, label='pruning ratio')
  plt.xlabel('number of bnb nodes explored')
  plt.legend()
  plt.yscale('log')
  plt.savefig('gaphistory.png')
  plt.close()

  pruned_nodes = bnb.all_prunednodes 
  nonpruned_nodes = bnb.all_nonpruned_nodes
  all_nodes = pruned_nodes + nonpruned_nodes
  if nx == 2:
    # threshold plotting the pruned region
    # this plotting procedure takes time
    # when there are many pruned nodes
    if len(pruned_nodes) < 10000:
      for i, node in enumerate(pruned_nodes):
        Xnode = np.linspace(node.l[0], node.u[0], num=3)
        Ylower = node.l[1] * np.ones(len(Xnode))
        Yupper = node.u[1] * np.ones(len(Xnode))
        if i == 0:
          plt.fill_between(Xnode, Ylower, Yupper, color='lightblue', alpha=0.5, label='pruned region')
        else:
          plt.fill_between(Xnode, Ylower, Yupper, color='lightblue', alpha=0.5)
    nonpruned_node_midpoints = np.array([node.midpoint for node in nonpruned_nodes])
    print("shape of nonpruned_node midpoints = ", nonpruned_node_midpoints.shape)
    if len(nonpruned_node_midpoints) > 0:
      plt.scatter(nonpruned_node_midpoints[:,0], nonpruned_node_midpoints[:,1], color='red', marker='o', s=10, label='nonpruned midpoints')
      plt.legend()
      plt.savefig('prunedregion.png')
      plt.close()
  # plot acqf upper and lower bounds on the regions defined by nodes
  if nx == 2 and False:
    acqf_upper_bounds = [node.aq_U for node in all_nodes]
    acqf_lower_bounds = [node.aq_L for node in all_nodes]
    gaps = [all_nodes[i].aq_U - all_nodes[i].aq_L for i in range(len(all_nodes))]
    LUB = min(acqf_upper_bounds) # least upper bound
    GUB = max(acqf_upper_bounds) # greatest upper bound
    LLB = min(acqf_lower_bounds) # least lower bound
    GLB = max(acqf_upper_bounds) # greatest lower bound
    max_gap = max(gaps)
    min_gap = min(gaps)
    for node in all_nodes:
      Xnode = np.linspace(node.l[0], node.u[0], num=3)
      Ylower = node.l[1] * np.ones(len(Xnode))
      Yupper = node.u[1] * np.ones(len(Xnode))
      plt.fill_between(Xnode, Ylower, Yupper, color='black', alpha=1. - (node.aq_U - LUB) / (GUB - LUB))
    plt.title(f'(acqf ub - LUB) / (GUB - LUB), GUB = {GUB:1.2e}, LUB = {LUB:1.2e}')
    plt.savefig('acqf_ub.png')
    plt.close()
    for node in all_nodes:
      Xnode = np.linspace(node.l[0], node.u[0], num=3)
      Ylower = node.l[1] * np.ones(len(Xnode))
      Yupper = node.u[1] * np.ones(len(Xnode))
      plt.fill_between(Xnode, Ylower, Yupper, color='black', alpha=1. - (node.aq_L - LLB) / (GLB - LLB))
    plt.title(f'(acqf lb - LLB) / (GLB - LLB), GLB = {GLB:1.2e}, LLB = {LLB:1.2e}')
    plt.savefig('acqf_lb.png')
    plt.close()
    for i, node in enumerate(all_nodes):
      Xnode = np.linspace(node.l[0], node.u[0], num=3)
      Ylower = node.l[1] * np.ones(len(Xnode))
      Yupper = node.u[1] * np.ones(len(Xnode))
      plt.fill_between(Xnode, Ylower, Yupper, color='black', alpha=1. - gaps[i]/max_gap)
    plt.title(f'gap/max_gap, max_gap = {max_gap:1.2e}, min_gap = {min_gap:1.2e}')
    plt.savefig('acqf_gap.png')
    plt.close()


  # determine an optimal number of clusters via k-means + silhouette score
  # this is a dimension independent measure
  nonpruned_node_midpoints = np.array([node.midpoint for node in nonpruned_nodes])
  cluster_values = [k for k in range(2, min(10, len(nonpruned_node_midpoints)))]
  silhouette_scores = []
  for k in cluster_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(nonpruned_node_midpoints)
    score = silhouette_score(nonpruned_node_midpoints, cluster_labels)
    silhouette_scores.append(score)
    print(f"The Silhouette score of the nonpruned nodes on {k} clusters is {score}")
  plt.plot(cluster_values, silhouette_scores)
  plt.xlabel('k (# of clusters)')
  plt.ylabel('Silhouette score')
  plt.savefig('silhouettescore.png')
  plt.close()
  


  


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
