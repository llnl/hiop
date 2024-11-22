#ifndef HIOP_EXAMPLE_DENSE_EX4
#define HIOP_EXAMPLE_DENSE_EX4

#include "hiopInterface.hpp"

#include <cassert>

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

using size_type = hiop::size_type;
using index_type = hiop::index_type;

/* Test problem from a tiny concave example
 *  min   -3*x*x-2*y*y
 *  s.t.
 *   y - 0.06*x*x >= 0
 *   y + 0.05*x*x <= 10
 *   y*y <= 64
 *   x*x <= 100
 *   0 <= x <= 11
 *   0 <= y <= 11
 */
class DenseConsEx4 : public hiop::hiopInterfaceDenseConstraints
{
public:
  DenseConsEx4();
  virtual ~DenseConsEx4();

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             double* Jac);

  virtual bool get_vecdistrib_info(size_type global_n, index_type* cols);

  virtual bool get_starting_point(const size_type& n, double* x0);

  /*
    void solution_callback(hiop::hiopSolveStatus status,
                           int n, const double* x,
                           const double* z_L,
                           const double* z_U,
                           int m, const double* g,
                           const double* lambda,
                           double obj_value)
    {
      printf("solution_callback with optimal value: %g. Also x[1]=%22.14f\n", obj_value, x[1]);
    };


    virtual bool iterate_callback(int iter, double obj_value, double logbar_obj_value,
                                  int n, const double* x,
                                  const double* z_L,
                                  const double* z_U,
                                  int m, const double* g,
                                  const double* lambda,
                                  double inf_pr, double inf_du, double onenorm_pr,
                                  double mu,
                                  double alpha_du, double alpha_pr,
                                  int ls_trials)
    {
      if(iter==3) return false;
      printf("%g %g\n", x[0], x[1]);
      return true;
    }
  */
private:
  size_type n_vars_, n_cons_;
#ifdef HIOP_USE_MPI
  MPI_Comm comm;
#endif
  int my_rank;
  int comm_size;
  index_type* col_partition_;
  bool unconstrained_;

public:
  inline index_type idx_local2global(size_type global_n, index_type idx_local)
  {
    assert(idx_local + col_partition_[my_rank] < col_partition_[my_rank + 1]);
    if(global_n == n_vars_) {
      return idx_local + col_partition_[my_rank];
    }
    assert(false && "you shouldn't need global index for a vector of this size.");
    return -1;
  }
  inline index_type idx_global2local(size_type global_n, index_type idx_global)
  {
    assert(idx_global >= col_partition_[my_rank] && "global index does not belong to this rank");
    assert(idx_global < col_partition_[my_rank + 1] && "global index does not belong to this rank");
    assert(global_n == n_vars_ && "your global_n does not match the number of variables?");
    return (idx_global - col_partition_[my_rank]);
  }
};
#endif
