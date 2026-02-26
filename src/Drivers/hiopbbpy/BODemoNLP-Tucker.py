#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.size"] = 12

from hiopbbpy.problems import Problem
from hiopbbpy.surrogate_modeling import smtKRG 
from hiopbbpy.opt import BnBAlgorithmBase, BnBAlgorithm, LCBacquisition, EIacquisition, BnBNode

if len(sys.argv) > 1:
  opt_mode = int(sys.argv[1])
else:
  opt_mode = 2
save_dir = 'newdata/opt_mode'+str(opt_mode) + '/'
# #### Example 

# In[ ]:


### parameters
n_samples = 3  # number of the initial samples to train GP
theta = 1.e0# hyperparameter for GP kernel
nx = 1         # dimension of the problem
    
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
            if in_range[i]:
                y_update[i, 0] *= np.cos(24. * np.pi * x[i,0])
        #y = np.array(y[i,0] * 1.)
        return y_update

problem = PeriodicObjective(ndim=nx)
problem.set_constraints([])  

problem.set_seed(42)
x_train = problem.sample(n_samples)
y_train = problem.evaluate(x_train)

gp_model = smtKRG(theta, problem.xlimits, nx, pow_exp_power=1.0, eval_noise=False)#, hyper_opt="NoOp")
gp_model.train(x_train, y_train)

print("optimal theta = ", gp_model.surrogatesmt.optimal_theta)

# In[ ]:


l = problem.xlimits[:, 0].astype(float)
u = problem.xlimits[:, 1].astype(float)
n_plot_pts = 1000
X = np.atleast_2d(np.linspace(l[0], u[0], n_plot_pts)).transpose()

muX = gp_model.mean(X)
sigmaX = np.sqrt(gp_model.variance(X))
print("max pointwise standard dev = ", max(sigmaX))
beta = 3.0

acqf = LCBacquisition(gp_model, beta=beta)
Y_acqf = acqf.evaluate(X)
Y_true = problem.evaluate(X)


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
#plt.show()
plt.savefig(save_dir + 'trainingdata.png')
plt.close()

# In[ ]:


bnb_options = {
    'opt_mode' : opt_mode,
}
bnb = BnBAlgorithm(acqf, bnb_options)
bnb.initialize()
root = bnb.best_node


# In[ ]:


from hiopbbpy.opt.bnbalgorithm import branch
nodes = [root]

num_divisions = 7
num_branches = 1
LUBgaps = np.zeros(num_divisions)
min_gaps = np.zeros(num_divisions)
max_gaps = np.zeros(num_divisions)
avg_gaps = np.zeros(num_divisions)
std_gaps = np.zeros(num_divisions)
pruning_ratios = np.zeros(num_divisions)
all_all_gaps = []

for j in range(num_divisions):
    children = []
    for node in nodes:
        for child_l, child_u in branch(node.l, node.u):
            acqf_L, acqf_U = bnb.compute_acqf_bounds(child_l, child_u)
            child = BnBNode(child_l, child_u, acqf_L, acqf_U)
            children.append(child)
    num_branches += len(children)

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
        
        if i == 0:
            plt.plot(xplt, acqf_UB, "--" + ub_color, label="acqf UB")
            plt.plot(xplt, acqf_LB, "-" + lb_color, label="acqf LB")
        else:
            plt.plot(xplt, acqf_UB, "--" + ub_color)
            plt.plot(xplt, acqf_LB, "-" + lb_color)
    plt.plot(X, Y_acqf, label=r'$LCB(x)$')
    plt.title('gap = {0:1.2e}, num_branches = {1:d}, pruning_ratio = {2:1.3f}'.format(
        LUBgap, num_branches, pruning_ratio))
    plt.legend()
    plt.savefig(save_dir + 'ublb'+str(j) + 'divisions.png')
    plt.close()
    nodes = children


# In[ ]:


#plt.plot(range(1, num_divisions+1), LUBgaps)
#plt.yscale('log')
#plt.ylabel('LUB gap')
#plt.xlabel('num divisions')
#plt.show()
#
#plt.plot(range(1, num_divisions+1), avg_gaps, label='mean gap')
#plt.plot(range(1, num_divisions+1), avg_gaps + std_gaps)
#plt.plot(range(1, num_divisions+1), avg_gaps - std_gaps)
#plt.yscale('log')
#plt.ylabel('gap')
#plt.xlabel('num divisions')
#plt.show()


np.savetxt(save_dir + 'LUBgapsvsdivisions.dat', LUBgaps)
np.savetxt(save_dir + 'avggapsvsdivisions.dat', avg_gaps)
np.savetxt(save_dir + 'stdgapsvsdivisions.dat', std_gaps)
np.savetxt(save_dir + 'pruningratios.dat', pruning_ratios)
for i in range(num_divisions):
    np.savetxt(save_dir + 'allgaps_'+str(i)+'.dat', all_all_gaps[i])
