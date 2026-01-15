import numpy as np

def f(x):
  ne, _ = x.shape
  return np.sum(np.sin(2* np.pi * x), axis=1).reshape(ne, 1)

if __name__ == "__main__":
  dimx = 2
  npts = 4
  X = np.array([np.random.randn(dimx) for i in range(npts)])
  Y = f(X)
  print(Y.shape)

