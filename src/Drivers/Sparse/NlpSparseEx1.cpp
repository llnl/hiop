#include "NlpSparseEx1.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum scal*1/4* { (x_{i}-1)^4 : i=1,...,n}
 *  s.t.
 *             scal * 4*x_1 + 2*x_2                     == scal*10
 *  scal * 5<= scal * 2*x_1         + x_3
 *  scal * 1<= scal * 2*x_1                 + 0.5*x_i   <= scal*2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 */
SparseEx1::SparseEx1(int n, double scal_input)
    : n_vars(n),
      n_cons{2},
      scal{scal_input}
{
  assert(n >= 3);
  if(n > 3) {
    n_cons += n - 3;
  }
}

SparseEx1::~SparseEx1() {}

bool SparseEx1::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars;
  m = n_cons;
  return true;
}

bool SparseEx1::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_vars);
  for(index_type i = 0; i < n; i++) {
    if(i == 0) {
      xlow[i] = -1e20;
      xupp[i] = 1e20;
      type[i] = hiopNonlinear;
      continue;
    }
    if(i == 1) {
      xlow[i] = 0.0;
      xupp[i] = 1e20;
      type[i] = hiopNonlinear;
      continue;
    }
    if(i == 2) {
      xlow[i] = 1.5;
      xupp[i] = 10.0;
      type[i] = hiopNonlinear;
      continue;
    }
    // this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
    xlow[i] = 0.5;
    xupp[i] = 1e20;
    type[i] = hiopNonlinear;
  }
  return true;
}

bool SparseEx1::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons);
  index_type conidx{0};
  clow[conidx] = scal * 10.0;
  cupp[conidx] = scal * 10.0;
  type[conidx++] = hiopInterfaceBase::hiopLinear;
  clow[conidx] = scal * 5.0;
  cupp[conidx] = 1e20;
  type[conidx++] = hiopInterfaceBase::hiopLinear;
  for(index_type i = 3; i < n_vars; i++) {
    clow[conidx] = scal * 1.0;
    cupp[conidx] = scal * 2 * n_vars;
    type[conidx++] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}

bool SparseEx1::get_sparse_blocks_info(size_type& nx,
                                       size_type& nnz_sparse_Jaceq,
                                       size_type& nnz_sparse_Jacineq,
                                       size_type& nnz_sparse_Hess_Lagr)
{
  nx = n_vars;
  ;
  nnz_sparse_Jaceq = 2;
  nnz_sparse_Jacineq = 2 + 2 * (n_vars - 3);
  nnz_sparse_Hess_Lagr = n_vars;
  return true;
}

bool SparseEx1::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_vars);
  obj_value = 0.;
  for(auto i = 0; i < n; i++) {
    obj_value += scal * 0.25 * pow(x[i] - 1., 4);
  }

  return true;
}

bool SparseEx1::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_vars);
  for(auto i = 0; i < n; i++) {
    gradf[i] = scal * pow(x[i] - 1., 3);
  }
  return true;
}

bool SparseEx1::eval_cons(const size_type& n,
                          const size_type& m,
                          const size_type& num_cons,
                          const index_type* idx_cons,
                          const double* x,
                          bool new_x,
                          double* cons)
{
  return false;
}

/* Four constraints no matter how large n is */
bool SparseEx1::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_vars);
  assert(m == n_cons);
  assert(n_cons == 2 + n - 3);

  // local contributions to the constraints in cons are reset
  for(auto j = 0; j < m; j++) {
    cons[j] = 0.;
  }

  index_type conidx{0};
  // compute the constraint one by one.
  //  --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
  cons[conidx++] += scal * (4 * x[0] + 2 * x[1]);

  // --- constraint 2 body ---> 2*x_1 + x_3
  cons[conidx++] += scal * (2 * x[0] + 1 * x[2]);

  // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
  for(auto i = 3; i < n; i++) {
    cons[conidx++] += scal * (2 * x[0] + 0.5 * x[i]);
  }

  return true;
}

bool SparseEx1::eval_Jac_cons(const size_type& n,
                              const size_type& m,
                              const size_type& num_cons,
                              const index_type* idx_cons,
                              const double* x,
                              bool new_x,
                              const size_type& nnzJacS,
                              index_type* iJacS,
                              index_type* jJacS,
                              double* MJacS)
{
  return false;
}

bool SparseEx1::eval_Jac_cons(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const size_type& nnzJacS,
                              index_type* iJacS,
                              index_type* jJacS,
                              double* MJacS)
{
  assert(n == n_vars);
  assert(m == n_cons);
  assert(n >= 3);

  assert(nnzJacS == 4 + 2 * (n - 3));

  int nnzit{0};
  index_type conidx{0};

  if(iJacS != NULL && jJacS != NULL) {
    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 1;
    conidx++;

    // --- constraint 2 body ---> 2*x_1 + x_3
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 2;
    conidx++;

    // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
    for(auto i = 3; i < n; i++) {
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = i;
      conidx++;
    }
    assert(nnzit == nnzJacS);
  }

  // values for sparse Jacobian if requested by the solver
  nnzit = 0;
  if(MJacS != NULL) {
    // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
    MJacS[nnzit++] = scal * 4;
    MJacS[nnzit++] = scal * 2;

    // --- constraint 2 body ---> 2*x_1 + x_3
    MJacS[nnzit++] = scal * 2;
    MJacS[nnzit++] = scal * 1;

    // --- constraint 3 body --->   2*x_1 + 0.5*x_4
    for(auto i = 3; i < n; i++) {
      MJacS[nnzit++] = scal * 2;
      MJacS[nnzit++] = scal * 0.5;
    }
    assert(nnzit == nnzJacS);
  }
  return true;
}

bool SparseEx1::eval_Hess_Lagr(const size_type& n,
                               const size_type& m,
                               const double* x,
                               bool new_x,
                               const double& obj_factor,
                               const double* lambda,
                               bool new_lambda,
                               const size_type& nnzHSS,
                               index_type* iHSS,
                               index_type* jHSS,
                               double* MHSS)
{
  // Note: lambda is not used since all the constraints are linear and, therefore, do
  // not contribute to the Hessian of the Lagrangian
  assert(nnzHSS == n);

  if(iHSS != NULL && jHSS != NULL) {
    for(int i = 0; i < n; i++) {
      iHSS[i] = jHSS[i] = i;
    }
  }

  if(MHSS != NULL) {
    for(int i = 0; i < n; i++) {
      MHSS[i] = scal * obj_factor * 3 * pow(x[i] - 1., 2);
    }
  }
  return true;
}

bool SparseEx1::get_starting_point(const size_type& n, double* x0)
{
  assert(n == n_vars);
  for(auto i = 0; i < n; i++) {
    x0[i] = 0.0;
  }
  return true;
}
