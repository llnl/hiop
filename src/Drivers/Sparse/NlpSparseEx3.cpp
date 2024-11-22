#include "NlpSparseEx3.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum { x_{i} : i=1,...,n}
 *  s.t.
 *        x_1 + x_n == 10,              , if eq_feas == true or eq_infeas == true
 *        x_1 + x_n   == 10,   i=3,...,n, if eq_feas == true
 *        x_1 + x_n   == 15,   i=3,...,n, if eq_infeas == true
 *        10-a <= x_1 + x_n  <= 10+a,           , if ineq_feas == true or ineq_infeas == true
 *        10-a <= x_1 + x_n  <= 15+a, i=3,...,n , if ineq_feas == true
 *         3-a <= x_1 + x_n  <= 5-a,  i=3,...,n,   if ineq_infeas == true
 *        x_i >= 0, i=1,...,n
 *
 *  a >= 0 , by default a = 1e-6
 *  n >= 3;
 */
SparseEx3::SparseEx3(int n, double scala_a, bool eq_feas, bool eq_infeas, bool ineq_feas, bool ineq_infeas)
    : n_vars_{n},
      n_cons_{0},
      scala_a_{scala_a},
      eq_feas_{eq_feas},
      eq_infeas_{eq_infeas},
      ineq_feas_{ineq_feas},
      ineq_infeas_{ineq_infeas}
{
  assert(n >= 3);
  assert(scala_a >= 0);
  if(eq_feas_ || eq_infeas_) {
    n_cons_++;
  }
  if(eq_feas_) {
    n_cons_ += n - 2;
  }
  if(eq_infeas_) {
    n_cons_ += n - 2;
  }
  if(ineq_feas_ || ineq_infeas_) {
    n_cons_++;
  }
  if(ineq_feas_) {
    n_cons_ += n - 2;
  }
  if(ineq_infeas_) {
    n_cons_ += n - 2;
  }
}

SparseEx3::~SparseEx3() {}

bool SparseEx3::get_prob_info(NonlinearityType& type)
{
  type = hiopInterfaceBase::hiopLinear;
  return true;
}

bool SparseEx3::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

bool SparseEx3::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_vars_);
  for(index_type i = 0; i < n; i++) {
    xlow[i] = 0.0;
    xupp[i] = 1e20;
    type[i] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}

bool SparseEx3::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons_);
  index_type conidx{0};
  if(eq_feas_ || eq_infeas_) {
    clow[conidx] = 10.0;
    cupp[conidx] = 10.0;
    type[conidx++] = hiopInterfaceBase::hiopLinear;
  }
  if(eq_feas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      clow[conidx] = 10.0;
      cupp[conidx] = 10.0;
      type[conidx++] = hiopInterfaceBase::hiopLinear;
    }
  }
  if(eq_infeas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      clow[conidx] = 15.0;
      cupp[conidx] = 15.0;
      type[conidx++] = hiopInterfaceBase::hiopLinear;
    }
  }
  if(ineq_feas_ || ineq_infeas_) {
    clow[conidx] = 10.0 - scala_a_;
    cupp[conidx] = 10.0 + scala_a_;
    type[conidx++] = hiopInterfaceBase::hiopLinear;
  }
  if(ineq_feas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      clow[conidx] = 10.0 - scala_a_;
      cupp[conidx] = 15.0 + scala_a_;
      type[conidx++] = hiopInterfaceBase::hiopLinear;
    }
  }
  if(ineq_infeas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      clow[conidx] = 3.0 - scala_a_;
      cupp[conidx] = 5.0 - scala_a_;
      type[conidx++] = hiopInterfaceBase::hiopLinear;
    }
  }
  return true;
}

bool SparseEx3::get_sparse_blocks_info(size_type& nx,
                                       size_type& nnz_sparse_Jaceq,
                                       size_type& nnz_sparse_Jacineq,
                                       size_type& nnz_sparse_Hess_Lagr)
{
  nx = n_vars_;
  nnz_sparse_Jaceq = 0;
  nnz_sparse_Jacineq = 0;
  nnz_sparse_Hess_Lagr = 0;

  if(eq_feas_ || eq_infeas_) {
    nnz_sparse_Jaceq += 2;
  }
  if(eq_feas_) {
    nnz_sparse_Jaceq += (n_vars_ - 2) * 2;
  }
  if(eq_infeas_) {
    nnz_sparse_Jaceq += (n_vars_ - 2) * 2;
  }
  if(ineq_feas_ || ineq_infeas_) {
    nnz_sparse_Jacineq += 2;
  }
  if(ineq_feas_) {
    nnz_sparse_Jacineq += (n_vars_ - 2) * 2;
  }
  if(ineq_infeas_) {
    nnz_sparse_Jacineq += (n_vars_ - 2) * 2;
  }
  nnzJac_ = nnz_sparse_Jaceq + nnz_sparse_Jacineq;
  return true;
}

bool SparseEx3::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_vars_);
  obj_value = 0.;
  for(auto i = 0; i < n; i++) {
    obj_value += x[i];
  }
  return true;
}

bool SparseEx3::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_vars_);
  for(auto i = 0; i < n; i++) {
    gradf[i] = 1.0;
  }
  return true;
}

bool SparseEx3::eval_cons(const size_type& n,
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
bool SparseEx3::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_vars_);
  assert(m == n_cons_);
  index_type conidx{0};

  // local contributions to the constraints in cons are reset
  for(auto j = 0; j < m; j++) {
    cons[j] = 0.;
  }

  if(eq_feas_ || eq_infeas_) {
    cons[conidx++] += x[0] + x[n - 1];
  }
  if(eq_feas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      cons[conidx++] += x[0] + x[n - 1];
    }
  }
  if(eq_infeas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      cons[conidx++] += x[0] + x[n - 1];
    }
  }
  if(ineq_feas_ || ineq_infeas_) {
    cons[conidx++] += x[0] + x[n - 1];
  }
  if(ineq_feas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      cons[conidx++] += x[0] + x[n - 1];
    }
  }
  if(ineq_infeas_) {
    for(index_type i = 0; i < n_vars_ - 2; i++) {
      cons[conidx++] += x[0] + x[n - 1];
    }
  }
  return true;
}

bool SparseEx3::eval_Jac_cons(const size_type& n,
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

bool SparseEx3::eval_Jac_cons(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const size_type& nnzJacS,
                              index_type* iJacS,
                              index_type* jJacS,
                              double* MJacS)
{
  assert(n == n_vars_);
  assert(m == n_cons_);
  assert(n >= 3);

  assert(nnzJacS == nnzJac_);

  int nnzit{0};
  index_type conidx{0};

  if(iJacS != NULL && jJacS != NULL) {
    if(eq_feas_ || eq_infeas_) {
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = n_vars_ - 1;
      conidx++;
    }
    if(eq_feas_) {
      for(index_type i = 0; i < n_vars_ - 2; i++) {
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = n_vars_ - 1;
        conidx++;
      }
    }
    if(eq_infeas_) {
      for(index_type i = 0; i < n_vars_ - 2; i++) {
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = n_vars_ - 1;
        conidx++;
      }
    }
    if(ineq_feas_ || ineq_infeas_) {
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = 0;
      iJacS[nnzit] = conidx;
      jJacS[nnzit++] = n_vars_ - 1;
      conidx++;
    }
    if(ineq_feas_) {
      for(index_type i = 0; i < n_vars_ - 2; i++) {
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = n_vars_ - 1;
        conidx++;
      }
    }
    if(ineq_infeas_) {
      for(index_type i = 0; i < n_vars_ - 2; i++) {
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;
        jJacS[nnzit++] = n_vars_ - 1;
        conidx++;
      }
    }
    assert(nnzit == nnzJac_);
  }

  // values for sparse Jacobian if requested by the solver
  if(MJacS != NULL) {
    nnzit = 0;
    for(index_type k = 0; k < nnzJac_; ++k) {
      MJacS[nnzit++] = 1.0;
    }
  }
  return true;
}

bool SparseEx3::eval_Hess_Lagr(const size_type& n,
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
  assert(nnzHSS == 0);

  return true;
}

bool SparseEx3::get_starting_point(const size_type& n, double* x0)
{
  assert(n == n_vars_);
  for(auto i = 0; i < n; i++) {
    x0[i] = 0.0;
  }
  return true;
}
