"""
Implementation of the Branin problem

min f(x_1, x_2) = \left( x_2 - \frac{5.1}{4\pi^2} x_1^2 + \frac{5}{\pi} x_1 - 6 \right)^2 + 10 \left(1 - \frac{1}{8\pi}\right) \cos(x_1) + 10.
s.t x_1 \in [-5, 10], \quad x_2 \in [0, 15].

It has three global minima at:
(x_1, x_2) = (-\pi, 12.275) 
             (\pi, 2.275)
             (9.42478, 2.475)
and the optimal objective 0.3979

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""
import numpy as np
from .problem import Problem
from numpy.random import uniform

# define the Branin problem class
class BraninProblem(Problem):
  def __init__(self, constraints=[]):
    ndim = 2
    xlimits = np.array([[-5.0, 10], [0.0, 15]]) 
    name = 'Branin'
    super().__init__(ndim, xlimits, name=name, constraints=constraints)
          
  def _evaluate(self, x: np.ndarray) -> np.ndarray:
    ne, nx = x.shape
    assert nx == self.ndim
    
    y = np.zeros((ne, 1), complex)
    b = 5.1 / (4.0 * (np.pi) ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    
    arg1 = (x[:,1] - b * x[:,0]**2 + c * x[:,0] - r)
    y[:,0] = arg1**2 + s * (1 - t) * np.cos(x[:,0]) + s
    
    return y






