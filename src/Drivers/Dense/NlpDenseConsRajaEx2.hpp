#ifndef HIOP_EXAMPLE_DENSE_RAJA_EX2
#define HIOP_EXAMPLE_DENSE_RAJA_EX2

// HiOp interface base
#include "hiopInterface.hpp"

// MPI includes
#ifdef HIOP_USE_MPI
#include <mpi.h>
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

// Type aliases for clarity
using size_type  = hiop::size_type;
using index_type = hiop::index_type;

/**
 * @brief RAJA-enabled implementation of DenseConsEx2 example.
 *
 * Problem:
 *   minimize   sum_{i=1}^n [1/4 * (x_i - 1)^4]
 *   subject to four linear constraints:
 *     x_1 free
 *     x_2 >= 0
 *     x_3 in [1.5, 10]
 *     x_i >= 0.5 for i >= 4
 */
class DenseConsRajaEx2 : public hiop::hiopInterfaceDenseConstraints
{
public:
  // === Constructors & Destructor ===

  /**
   * @brief Constructor.
   * @param n Number of variables.
   * @param unconstrained Whether to disable constraints.
   */
  DenseConsRajaEx2(int n, bool unconstrained = false);

  /**
   * @brief Destructor.
   */
  virtual ~DenseConsRajaEx2();

  // === Problem definition interface ===

  bool get_prob_sizes(size_type& n, size_type& m) override;

  bool get_vecdistrib_info(size_type global_n, index_type* cols) override;

  bool get_vars_info(const size_type& n,
                     double* xlow,
                     double* xupp,
                     NonlinearityType* type) override;

  bool get_cons_info(const size_type& m,
                     double* clow,
                     double* cupp,
                     NonlinearityType* type) override;

  // === Objective and derivative evaluations ===

  bool eval_f(const size_type& n,
              const double* x,
              bool new_x,
              double& obj_value) override;

  bool eval_grad_f(const size_type& n,
                   const double* x,
                   bool new_x,
                   double* gradf) override;

  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons) override;

  bool eval_cons(const size_type& n,
		 const size_type& m,
		 const double* x,
		 bool new_x,
 		 double* cons) override
  {
    return false;
  }

  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const index_type* idx_cons,
                     const double* x,
                     bool new_x,
                     double* Jac) override;

  bool eval_Jac_cons(const size_type& n,
		     const size_type& m,
		     const double* x,
		     bool new_x,
		     double* Jac) override;

  bool get_starting_point(const size_type& n,
                          double* x0) override;

  bool get_starting_point(const size_type& n,
			  const size_type& m,
			  double* x0,
			  bool& duals_avail,
			  double* z_bndL0,
			  double* z_bndU0,
			  double* lamda0,
			  bool& slacks_avail,
			  double* ineq_slack) override
  {
    duals_avail = false;
    slacks_avail = false;
    return false;
  }

private:
  // === Problem dimensions ===
  size_type    n_vars_;        ///< Number of variables
  size_type    n_cons_;        ///< Number of constraints

#ifdef HIOP_USE_MPI
  MPI_Comm     comm_;          ///< MPI communicator
#endif

  // === MPI-related ===
  int          my_rank_;       ///< MPI rank
  int          comm_size_;     ///< Number of MPI ranks

  // === Partitioning ===
  index_type*  col_partition_; ///< Column partitioning array

  // === Problem options ===
  bool         unconstrained_; ///< Unconstrained flag
};

#endif // HIOP_EXAMPLE_DENSE_RAJA_EX2
