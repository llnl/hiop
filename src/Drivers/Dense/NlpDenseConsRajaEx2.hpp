#ifndef HIOP_EXAMPLE_DENSE_RAJA_EX2
#define HIOP_EXAMPLE_DENSE_RAJA_EX2

#include "hiopInterface.hpp"
#include <RAJA/RAJA.hpp>
#include <cmath>

using size_type = hiop::size_type;
using index_type = hiop::index_type;

#if defined(HIOP_USE_MPI)
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

// Execution policies for CPU fallback (sequential)
using exec_pol   = RAJA::seq_exec;
using reduce_pol = RAJA::seq_reduce;

class DenseConsRajaEx2 : public hiop::hiopInterfaceDenseConstraints {
public:
  DenseConsRajaEx2(int n, bool unconstrained = false);
  virtual ~DenseConsRajaEx2();

  bool get_prob_sizes(size_type& n, size_type& m) override;
  bool get_vecdistrib_info(size_type global_n, index_type* cols) override;
  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type) override;
  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type) override;
  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value) override;
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf) override;
  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons) override;
  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const index_type* idx_cons,
                     const double* x,
                     bool new_x,
                     double* Jac) override;
  bool get_starting_point(const size_type& n, double* x0) override;

private:
  size_type    n_vars_, n_cons_;
#if defined(HIOP_USE_MPI)
  MPI_Comm     comm_;
#endif
  int          my_rank_, comm_size_;
  index_type*  col_partition_;
  bool         unconstrained_;
};

#endif // HIOP_EXAMPLE_DENSE_RAJA_EX2