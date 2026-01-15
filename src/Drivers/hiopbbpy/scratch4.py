import numpy as np

ndim = 3
l = np.zeros(ndim)
u = np.ones(ndim)
s_per_dim = 2 # samples per dimension
n_points = s_per_dim ** ndim
x_points = np.zeros((n_points, ndim))

for i in range(n_points):
  for j in range(ndim):
    x_points[i, j] = l[j] + (u[j] - l[j]) / (s_per_dim - 1.) * float(int(i / s_per_dim **j) % s_per_dim)
  print(x_points[i,:])
