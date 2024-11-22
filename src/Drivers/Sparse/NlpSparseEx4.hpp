#ifndef HIOP_EXAMPLE_SPARSE_EX4
#define HIOP_EXAMPLE_SPARSE_EX4

#include "hiopInterface.hpp"

#include <cassert>

using size_type = hiop::size_type;
using index_type = hiop::index_type;

/* Test problem from a tiny concave example
 *  min   -3*x*x-2*y*y
 *  s.t.
 *   y - 0.06*x*x >= 0.0
 *   y - 0.05*x*x <= 10.
 *   y*y <= 64
 *   x*x <= 100
 *   0 <= x <= 11
 *   0 <= y <= 11
 */
class SparseEx4 : public hiop::hiopInterfaceSparse
{
public:
  SparseEx4(double scal_in);
  virtual ~SparseEx4();

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  virtual bool get_sparse_blocks_info(size_type& nx,
                                      size_type& nnz_sparse_Jaceq,
                                      size_type& nnz_sparse_Jacineq,
                                      size_type& nnz_sparse_Hess_Lagr);

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS);
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS);
  virtual bool get_starting_point(const size_type& n, double* x0);
  virtual bool eval_Hess_Lagr(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const size_type& nnzHSS,
                              index_type* iHSS,
                              index_type* jHSS,
                              double* MHSS);

  // not implemented
  virtual bool
  get_starting_point(const size_type&, const size_type&, double*, bool&, double*, double*, double*, bool&, double*)
  {
    return false;
  }

  virtual bool
  get_warmstart_point(const size_type&, const size_type&, double*, double*, double*, double*, double*, double*, double*)
  {
    return false;
  }

private:
  size_type n_vars;
  size_type n_cons;
  double scal;
};
#endif
