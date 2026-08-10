import numpy as np
import math # for math.comb function

def basis_conversion(alpha):
  # extract coefficients to express
  # (x - \alpha_{0}) * (x - \alpha_{1}) * ... * (x - \alpha_{k-1})
  # in the form
  # c_0 * x^k + c_1 * x^(k-1) + ... + c_k
  k = len(alpha)
  c = np.zeros(k + 1)
  c[0] = 1.
  idx_superset = []
  # initial set of indices
  idx_superset.append([[j] for j in range(k)])
  for n in range(1, k):
    idx_set = []
    for i in range(len(idx_superset[-1])):
      for j in range(k):
        if idx_superset[-1][i][-1] < j:
          idx_set.append(idx_superset[-1][i] + [j])
    idx_superset.append(idx_set)
  for k, idx_set in enumerate(idx_superset):
    for idx_pair in idx_set:
      temp = 1.
      for j in idx_pair:
        temp *= -alpha[j]
      c[k+1] += temp  
  return c

def basis_conversion_2(x0, x1, c):
  # extract coefficients to express
  # c_0 * (x - x0)^k + c_1 * (x - x0)^(k-1) + ... + c_k
  # in the form
  # b_0 * (x - x1)^k + b_1 * (x - x1)^(k-1) + ... + b_k
  k = len(c) - 1 # polynomial degree
  n = k + 1 # number of coefficients
  A = np.zeros((n, n)) # linear system matrix A b = c, dense lower-triangular
  delta = x1 - x0
  if np.allclose(x0, x1):
    b = np.zeros(n)
    b[:] = c[:]
    return b
  for i in range(n):
    for j in range(i+1):
      A[i, j] = (-delta) ** (i-j) * math.comb(k-j, i - j)
  b = np.linalg.solve(A, c)
  return b


def polynomial_multiply(c1, c2):
  c1deg = len(c1) - 1
  c2deg = len(c2) - 1
  c1_rev = c1[::-1]
  c2_rev = c2[::-1]
  # c1(x) = c1_0 + c1_1 (x - x0)^1 + c1_2 (x-x0)^2 + ... + c1_c1deg * (x-x0)^c1deg
  # c2(x) = c2_0 + c2_1 (x - x0)^1 + c2_2 (x-x0)^2 + ... + c2_c2deg * (x-x0)^c2deg
  # c3(x) = c1(x) * c2(x)
  c3_rev = np.zeros(c1deg + c2deg + 1)
  for k in range(len(c3_rev)):
    for i in range(k+1):
      j = k - i
      if i <= c1deg and j <= c2deg:
        c3_rev[k] += c1_rev[i] * c2_rev[j]
  c3 = c3_rev[::-1]
  return c3


class matern_phi:
  nu = 1.5
  X = []
  p = 0
  th = 1.0
  def __init__(self, X_, th, nu_=1.5):
    assert nu_ in [1.5, 2.5], "unsupported value of nu"
    self.nu = nu_
    assert len(X_) > 0, "must have at least one data point"
    self.X = X_ # deep copy?
    self.p = len(self.X)
    assert th > 0., "theta parameter must be positive"
    self.th = th
  def evaluate_i(self, s, i):
    assert i in range(self.p), "i out of range"
    if self.nu == 1.5:
      z = np.sqrt(3.) * self.th * np.abs(s - self.X[i])
      return np.log(1. + z) - z
    else:
      z = np.sqrt(5.) * self.th * np.abs(s - self.X[i])
      return np.log(1. + z + (z**2.) / 3.) - z
  def evaluate(self, s):
    y = self.evaluate_i(s, 0)
    for i in range(1, self.p):
      y += self.evaluate_i(s, i)
    return y
  def evaluate_deriv_i(self, s, i):
    assert i in range(self.p), "index i out of range"
    if self.nu == 1.5:
      c = np.sqrt(3.) * self.th
      z = c * np.abs(s - self.X[i])
      return c * np.sign(s - self.X[i]) * (1. / (1. + z) - 1.)
    else:
      c = np.sqrt(5.) * self.th
      z = c * np.abs(s - self.X[i])
      return c * np.sign(s - self.X[i]) * ((1. + 2.*z / 3.) / (1. + z + (z**2.) / 3.) - 1.)
  def evaluate_deriv(self, s):
    y = self.evaluate_deriv_i(s, 0)
    for i in range(1, self.p):
      y += self.evaluate_deriv_i(s, i)
    return y
  def generate_Djsec(self, l, u):
    Djsec = [] # list of cut directions (alpha, beta)
    for i in range(self.p):
      beta = np.zeros(self.p)
      beta[i] = -1.0
      alpha = (self.evaluate_i(u, i) - self.evaluate_i(l, i)) / (u - l)
      Djsec.append([alpha, beta])
    return Djsec
  def generate_Djtan(self, l, u):
    midpt = (l + u) / 2.
    Xbrk = [l, midpt, u] # "break points"
    for x in self.X:
      if x > l and x < u and not np.isclose(x, midpt, rtol=0.):
        Xbrk.append(x)
    Djtan = [] # list of cut directions (alpha, beta)
    for i in range(self.p):
      for t in Xbrk:
        alpha = -1.0 * self.evaluate_deriv_i(t, i)
        beta = np.zeros(self.p)
        beta[i] = 1.0
        Djtan.append([alpha, beta])
    return Djtan
  def generate_alpha_beta_r(self, l, u):
    Djsec = self.generate_Djsec(l, u)
    Djtan = self.generate_Djtan(l, u)
    Dj = Djsec + Djtan
    abrs = []
    for k in range(len(Dj)):
      alpham = Dj[k][0]
      betam  = Dj[k][1]
      if self.nu == 1.5:
        _, _, _, rm = Fjmax_threehalves(l, u, self.X, self.th, alpham, betam)
      else:
        _, _, _, rm = Fjmax_fivehalves(l, u, self.X, self.th, alpham, betam)
      print(rm)
      abrs.append([alpham, betam, rm])
    return abrs
   


# maximize Fj (\nu=3/2) on 1d interval [lj, uj]
# lj, uj scalars
# X a list
# thj a scalar
# alpha a scalar
# beta an array of length X
def Fjmax_threehalves(lj, uj, X, thj, alpha, beta):
  n = len(X)
  assert len(beta) == n, "beta not appropriate dimension"
  def phiii(s, i):
    assert i in range(n), "i out of range for phii call"
    z = np.sqrt(3.) * thj * np.abs(s - X[i])
    return np.log(1. + z) - z
  def Fj(s):
    y = alpha * s
    for i in range(n):
      y += beta[i] * phiii(s, i)
    return y
  # need to determine all breakpoints on interval [lj, uj]
  # data points for GP + end points
  Xbrkpts = [lj, uj]
  for x in X:
    if x > lj and x < uj:
      Xbrkpts.append(x)
  # need to do global maximization over each subinterval
  nintervals = len(Xbrkpts) - 1 
  xroots = []
  for i in range(nintervals): # for each interval between two (training + end) points
    xl = Xbrkpts[i]
    xu = Xbrkpts[i+1]
    omega = np.zeros(n)
    for k in range(n):
      # take midpoint to determine sign
      omega[k] = np.sign((xl + xu) / 2. - X[k]) / (np.sqrt(3.0) * thj)
    alphap = alpha - np.inner(beta, 1. / omega)
    sigma = X - omega
    coeffs = alphap * basis_conversion(sigma)
    for k in range(n):
      s_temp = np.array([sig for idx, sig in enumerate(sigma) if idx != k])
      coeffs[1:] += beta[k] * basis_conversion(s_temp)
    roots = np.polynomial.polynomial.polyroots(coeffs[::-1])
    for root in roots:
      if np.allclose(root.imag, 0.0) and root.real > xl and root.real < xu:
        xroots.append(root.real)
  xroots = np.array(xroots + [lj, uj] + X)
  yroots = Fj(xroots)
  i_min = np.argmin(yroots)
  i_max = np.argmax(yroots)
  x_max = xroots[i_max]
  y_max = yroots[i_max]
  x_min = xroots[i_min]
  y_min = yroots[i_min]
  return x_min, x_max, y_min, y_max





# maximize Fj (\nu=5/2) on 1d interval [lj, uj]
# lj, uj scalars
# X a list
# thj a scalar
# alpha a scalar
# beta an array of length X
def Fjmax_fivehalves(lj, uj, X, thj, alpha, beta):
  n = len(X)
  assert len(beta) == n, "beta not appropriate dimension"
  def phii(s, i):
    z = np.sqrt(5.) * thj * np.abs(s - X[i])
    return np.log(1. + z + (z**2.) / 3.) - z
  def Fj(s):
    y = alpha * s
    for i in range(n):
      y += beta[i] * phii(s, i)
    return y    
  # need to determine all breakpoints on interval [lj, uj]
  # data points for GP + end points
  Xbrkpts = [lj, uj]
  for x in X:
    if x > lj and x < uj:
      Xbrkpts.append(x)
  xroots = [] 
  # need to do global maximization over each subinterval
  nintervals = len(Xbrkpts) - 1 
  for i in range(nintervals):
    xl = Xbrkpts[i]
    xu = Xbrkpts[i+1]
    omega = np.zeros(n)
    zeta = np.zeros(n)
    for k in range(n):
      # take midpoint to determine sign
      omega[k] = np.sign((xl + xu) / 2. - X[k]) / (np.sqrt(5.0) * thj)
      zeta[k] = 5. / 3. * thj**2. * omega[k]
    alphap = alpha - np.inner(beta, 1. / omega) # alpha "prime"
    # Step 1: build up P(s) coeffs
    coeffs = [1.] # initialize as constant function
    for k in range(n):
      c_temp = basis_conversion_2(X[k], 0., [zeta[k], 1., omega[k]])
      coeffs = polynomial_multiply(coeffs, c_temp)  
    # Step 2: Scale P(s) by alpha'
    coeffs *= alphap 
    # Step 3: for each k: add 2 zeta_k * beta_k P_k(s) * (s - x_k + 1 / (2*zeta_k))
    for k in range(n):
      # initialize Pk as constant function
      # then update according to factors (omega_ii + (s - x_ii) + zeta_ii * (s - x_ii)^2
      # we convert the coefficients to common basis {1, s, s^2,...}
      Pk_coeffs = [1.]
      for ii in range(n):
        if ii == k:
          continue
        c_temp = basis_conversion_2(X[ii], 0., [zeta[ii], 1., omega[ii]]) # omega_ii + (s - x_ii) + zeta_ii (s - x_ii)^2
        Pk_coeffs = polynomial_multiply(Pk_coeffs, c_temp)
      # now with Pk_coeffs update coeffs of polynomial that we seek the root of
      # alpha' P(s) + \sum_k 2 zeta_k beta_k P_k(s) * (s + (-x_k + 1 / (2 zeta_k)))
      coeffs[1:] += 2 * zeta[k] * beta[k] * polynomial_multiply(Pk_coeffs, [1., -X[k] + 0.5 / zeta[k]])
    # re-order coefficients for polyroots call
    roots = np.polynomial.polynomial.polyroots(coeffs[::-1])
    # filter out non-real roots and roots that are not within [xl, xu]
    for root in roots:
      if np.allclose(root.imag, 0.0) and root.real > xl and root.real < xu:
        xroots.append(root.real)
  xroots = np.array(xroots + [lj, uj] + X)
  yroots = Fj(xroots)
  i_min = np.argmin(yroots)
  i_max = np.argmax(yroots)
  x_max = xroots[i_max]
  y_max = yroots[i_max]
  x_min = xroots[i_min]
  y_min = yroots[i_min]
  return x_min, x_max, y_min, y_max



