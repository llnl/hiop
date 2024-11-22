#include "NlpSparseEx4.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

/* Test problem from a tiny concave example
 *  min   -3*x*x-2*y*y
 *  s.t.
 *   y >= 0.06*x*x
 *   y <= 10 - 0.05*x*x
 *   y*y <= 64
 *   x*x <= 100
 *   0 <= x <= 11
 *   0 <= y <= 11
 */
SparseEx4::SparseEx4(double scal_input)
    : n_vars(2),
      n_cons{4},
      scal{scal_input}
{
  assert(n_vars == 2);
  assert(n_cons == 4);
}

SparseEx4::~SparseEx4() {}

bool SparseEx4::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars;
  m = n_cons;
  return true;
}

bool SparseEx4::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_vars);
  xlow[0] = 0.;
  xupp[0] = 11.;
  type[0] = hiopNonlinear;
  xlow[1] = 0.;
  xupp[1] = 11.;
  type[1] = hiopNonlinear;
  return true;
}

bool SparseEx4::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons);
  clow[0] = 0.0;
  cupp[0] = 1e20;
  type[0] = hiopInterfaceBase::hiopLinear;

  clow[1] = -1e20;
  cupp[1] = 10.0;
  type[1] = hiopInterfaceBase::hiopLinear;

  clow[2] = -1e20;
  cupp[2] = 64.0;
  type[2] = hiopInterfaceBase::hiopLinear;

  clow[3] = -1e20;
  cupp[3] = 100.;
  type[3] = hiopInterfaceBase::hiopLinear;

  return true;
}

bool SparseEx4::get_sparse_blocks_info(size_type& nx,
                                       size_type& nnz_sparse_Jaceq,
                                       size_type& nnz_sparse_Jacineq,
                                       size_type& nnz_sparse_Hess_Lagr)
{
  nx = n_vars;
  ;
  nnz_sparse_Jaceq = 0;
  nnz_sparse_Jacineq = 6;
  nnz_sparse_Hess_Lagr = 2;
  return true;
}

bool SparseEx4::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_vars);
  obj_value = 0.;
  obj_value += -3. * x[0] * x[0] - 2. * x[1] * x[1];

  return true;
}

bool SparseEx4::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_vars);
  gradf[0] = -6. * x[0];
  gradf[1] = -4. * x[1];
  return true;
}

bool SparseEx4::eval_cons(const size_type& n,
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
bool SparseEx4::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_vars);
  assert(m == n_cons);

  // local contributions to the constraints in cons are reset
  for(auto j = 0; j < m; j++) {
    cons[j] = 0.;
  }

  index_type conidx{0};
  // compute the constraint one by one.
  //  --- constraint 1 body --->  y - 0.06*x*x >= 0.0
  cons[conidx++] += scal * (x[1] - 0.06 * x[0] * x[0]);

  // --- constraint 2 body ---> y + 0.05*x*x <= 10.0
  cons[conidx++] += scal * (x[1] + 0.05 * x[0] * x[0]);

  // --- constraint 3 body ---> y*y <= -64.
  cons[conidx++] += scal * (x[1] * x[1]);

  // --- constraint 4 body ---> x*x <= -100.
  cons[conidx++] += scal * (x[0] * x[0]);

  return true;
}

bool SparseEx4::eval_Jac_cons(const size_type& n,
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

bool SparseEx4::eval_Jac_cons(const size_type& n,
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

  int nnzit{0};
  index_type conidx{0};

  if(iJacS != NULL && jJacS != NULL) {
    // --- constraint 1 body --->  y - 0.06*x*x >= 0.0
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 1;
    conidx++;

    // --- constraint 2 body ---> y + 0.05*x*x <= 10.0
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 1;
    conidx++;

    // --- constraint 3 body ---> y*y <= -64.
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 1;
    conidx++;

    // --- constraint 4 body ---> x*x <= -100.
    iJacS[nnzit] = conidx;
    jJacS[nnzit++] = 0;
    conidx++;

    assert(nnzit == nnzJacS);
  }

  // values for sparse Jacobian if requested by the solver
  nnzit = 0;
  if(MJacS != NULL) {
    // --- constraint 1 body --->  y - 0.06*x*x >= 0.0
    MJacS[nnzit++] = -0.12 * x[0];
    MJacS[nnzit++] = 1.;
    conidx++;

    // --- constraint 2 body ---> y + 0.05*x*x <= 10.0
    MJacS[nnzit++] = 0.1 * x[0];
    MJacS[nnzit++] = 1.;
    conidx++;

    // --- constraint 3 body ---> y*y <= -64.
    MJacS[nnzit++] = 2. * x[1];
    conidx++;

    // --- constraint 4 body ---> x*x <= -100.
    MJacS[nnzit++] = 2. * x[0];
    conidx++;
    assert(nnzit == nnzJacS);
  }
  return true;
}

bool SparseEx4::eval_Hess_Lagr(const size_type& n,
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
    MHSS[0] = obj_factor * (-6.) + lambda[0] * (-0.12) + lambda[1] * (-0.1) + lambda[3] * (2.);
    MHSS[1] = obj_factor * (-4.) + lambda[2] * (2.);
  }
  return true;
}

bool SparseEx4::get_starting_point(const size_type& n, double* x0)
{
  assert(n == n_vars);
  for(auto i = 0; i < n; i++) x0[i] = 0.0;
  return true;
}
