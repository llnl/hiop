#ifndef HIOP_EXAMPLE_SPARSE_EX3
#define HIOP_EXAMPLE_SPARSE_EX3

#include "hiopInterface.hpp"

#include <cassert>

using size_type = hiop::size_type;
using index_type = hiop::index_type;

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum { x_{i} : i=1,...,n}
 *  s.t.
 *        x_1 + x_n == 10,              , if eq_feas == true or eq_infeas == true
 *        x_1 + x_n   == 10,   i=3,...,n, if eq_feas == true
 *        x_1 + x_n   == 15,   i=3,...,n, if eq_infeas == true
 *        10-a <= x_1 + x_n  <= 10+a,           , if ineq_feas == true or ineq_infeas == true
 *        10-a <= x_1 + x_n  <= 15+a, i=3,...,n , if ineq_feas == true
 *         3-a <= x_1 + x_n  <= 5-a, i=3,...,n,   if ineq_infeas == true
 *        x_i >= 0, i=1,...,n
 *
 *  a >= 0 , by default a = 1e-6
 *  n >= 3
 */
class SparseEx3 : public hiop::hiopInterfaceSparse
{
public:
  SparseEx3(int n,
            double scala_a = 1e-6,
            bool eq_feas = false,
            bool eq_infeas = false,
            bool ineq_feas = false,
            bool ineq_infeas = false);
  virtual ~SparseEx3();

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_prob_info(NonlinearityType& type);
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

private:
  size_type n_vars_;
  size_type n_cons_;
  double scala_a_;
  bool eq_feas_;
  bool eq_infeas_;
  bool ineq_feas_;
  bool ineq_infeas_;
  size_type nnzJac_;
};
#endif
