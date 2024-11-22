#ifndef HIOP_EXAMPLE_SPARSE_EX2
#define HIOP_EXAMPLE_SPARSE_EX2

#include "hiopInterface.hpp"

// this include is not needed in general
// we use hiopMatrixSparse in this particular example for convenience
#include "hiopMatrixSparse.hpp"
#include "LinAlgFactory.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <cassert>
#include <cstring>  //for memcpy
#include <cstdio>
#include <cmath>

using size_type = hiop::size_type;
using index_type = hiop::index_type;

/** Nonlinear *highly nonconvex* and *rank deficient* problem test for the Filter IPM
 * Newton of HiOp. It uses a Sparse NLP formulation. The problem is based on SparseEx1.
 *
 *  min   (2*convex_obj-1)*scal*sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + 0.5x^Tx
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.0 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]                  (rnkdef-con1)
 *        4*x_1 + 2*x_2 == 10                                (rnkdef-con2)
 *
 *  other parameters are:
 *  convex_obj: set to 1 to have a convex problem, otherwise set it to 0
 *  scale_quartic_obj_term: scaling factor for the quartic term in the objective (1.0 by default).
 *
 */
class SparseEx2 : public hiop::hiopInterfaceSparse
{
public:
  SparseEx2(int n, bool convex_obj, bool rankdefic_Jac_eq, bool rankdefic_Jac_ineq, double scal_neg_obj = 1.0);
  virtual ~SparseEx2();

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);
  virtual bool get_sparse_blocks_info(index_type& nx,
                                      index_type& nnz_sparse_Jaceq,
                                      index_type& nnz_sparse_Jacineq,
                                      index_type& nnz_sparse_Hess_Lagr);

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
  int n_vars_;
  int n_cons_;
  bool convex_obj_;
  bool rankdefic_eq_;
  bool rankdefic_ineq_;
  double scal_neg_obj_;
};

#endif
