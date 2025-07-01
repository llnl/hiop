#include "NlpDenseConsRajaEx2.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// Umpire
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

// RAJA
#include <RAJA/RAJA.hpp>

// HiOp matrix implementations
#include <hiopMatrixDenseRowMajor.hpp>
#include <hiopMatrixDenseRaja.hpp>

#ifdef HIOP_USE_CUDA
  #include "ExecPoliciesRajaCudaImpl.hpp"
  using exec_pol   = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_exec;
  using reduce_pol = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_reduce;
#elif defined(HIOP_USE_HIP)
  #include "ExecPoliciesRajaHipImpl.hpp"
  using exec_pol   = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_exec;
  using reduce_pol = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_reduce;
#elif defined(HIOP_USE_RAJA)
  using exec_pol   = RAJA::seq_exec;
  using reduce_pol = RAJA::seq_reduce;
#endif

// Type aliases for clarity
using size_type  = hiop::size_type;
using index_type = hiop::index_type;

constexpr double INF = 1e20;

// === Constructor ===
DenseConsRajaEx2::DenseConsRajaEx2(int n, bool unconstrained)
  : n_vars_{n},
    n_cons_{unconstrained ? 0 : 4},
    unconstrained_{unconstrained}
{
#ifdef HIOP_USE_MPI
  comm_ = MPI_COMM_WORLD;
  MPI_Comm_size(comm_, &comm_size_);
  MPI_Comm_rank(comm_, &my_rank_);
#else
  my_rank_   = 0;
  comm_size_ = 1;
#endif

  // Partition variables for each rank
  col_partition_ = new index_type[comm_size_ + 1];
  index_type q = n_vars_ / comm_size_;
  index_type r = n_vars_ % comm_size_;

  for(int i = 0; i <= comm_size_; ++i) {
    col_partition_[i] = i * q + (i < r ? i : r);
  }
}

// === Destructor ===
DenseConsRajaEx2::~DenseConsRajaEx2()
{
  delete[] col_partition_;
}

// === Problem sizes ===
bool DenseConsRajaEx2::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

// === Vector distribution info ===
bool DenseConsRajaEx2::get_vecdistrib_info(size_type global_n, index_type* cols)
{
  std::memcpy(cols, col_partition_, (comm_size_ + 1) * sizeof(index_type));
  return true;
}

// === Variable bounds and types ===
bool DenseConsRajaEx2::get_vars_info(const size_type& n,
                                     double* xlow,
                                     double* xupp,
                                     NonlinearityType* type)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  RAJA::forall<exec_pol>(RAJA::RangeSegment(start, end), RAJA_LAMBDA(index_type i) {
    index_type li = i - start;
    if (i == 0) {
      xlow[li] = -INF; xupp[li] = INF;
    } else if (i == 1) {
      xlow[li] = 0.0;  xupp[li] = INF;
    } else if (i == 2) {
      xlow[li] = 1.5;  xupp[li] = 10.0;
    } else {
      xlow[li] = 0.5;  xupp[li] = INF;
    }
    type[li] = hiopNonlinear;
  });

  return true;
}

// === Constraint bounds and types ===
bool DenseConsRajaEx2::get_cons_info(const size_type& m,
                                     double* clow,
                                     double* cupp,
                                     NonlinearityType* type)
{
  if (unconstrained_) return true;

  clow[0] = n_vars_ + 1; cupp[0] = n_vars_ + 1;
  clow[1] = 5.0;         cupp[1] = INF;
  clow[2] = 1.0;         cupp[2] = 2.0 * n_vars_;
  clow[3] = -INF;        cupp[3] = 4.0 * n_vars_;

  std::fill(type, type + m, hiopInterfaceBase::hiopLinear);
  return true;
}

// === Objective evaluation ===
bool DenseConsRajaEx2::eval_f(const size_type& n,
                              const double* x,
                              bool new_x,
                              double& obj_value)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  RAJA::ReduceSum<reduce_pol, double> sum(0.0);

  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(start, end),
    RAJA_LAMBDA(index_type i) {
      index_type li = i - start;
      double xi = x[li];
      sum += 0.25 * std::pow(xi - 1.0, 4);
    });

  obj_value = sum.get();

#ifdef HIOP_USE_MPI
  double tmp = obj_value;
  MPI_Allreduce(&tmp, &obj_value, 1, MPI_DOUBLE, MPI_SUM, comm_);
#endif

  return true;
}

// === Gradient of objective ===
bool DenseConsRajaEx2::eval_grad_f(const size_type& n,
                                   const double* x,
                                   bool new_x,
                                   double* gradf)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(start, end),
    RAJA_LAMBDA(index_type i) {
      index_type li = i - start;
      double xi = x[li];
      gradf[li] = std::pow(xi - 1.0, 3);
    });

  return true;
}

// === Constraint evaluation ===
bool DenseConsRajaEx2::eval_cons(const size_type& n,
                                 const size_type& m,
                                 const size_type& num_cons,
                                 const index_type* idx_cons,
                                 const double* x,
                                 bool new_x,
                                 double* cons)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  std::fill(cons, cons + num_cons, 0.0);

  for (size_type j = 0; j < num_cons; ++j) {
    index_type ci = idx_cons[j];
    RAJA::ReduceSum<reduce_pol, double> sum(0.0);

    RAJA::forall<exec_pol>(
      RAJA::RangeSegment(start, end),
      RAJA_LAMBDA(index_type i) {
        index_type li = i - start;
        double xi = x[li];
        switch (ci) {
          case 0: sum += xi; break;
          case 1: sum += (i == 0 ? 2.0 * xi : xi); break;
          case 2: sum += (i == 0 ? 2.0 * xi : (i == 1 ? 0.5 * xi : xi)); break;
          case 3: sum += (i == 0 ? 4.0 * xi : (i <= 2 ? 2.0 * xi : xi)); break;
        }
      });

    cons[j] = sum.get();
  }

#ifdef HIOP_USE_MPI
  if (num_cons > 0) {
    std::vector<double> tmp(num_cons);
    MPI_Allreduce(cons, tmp.data(), num_cons, MPI_DOUBLE, MPI_SUM, comm_);
    std::memcpy(cons, tmp.data(), num_cons * sizeof(double));
  }
#endif

  return true;
}

// === Jacobian of constraints ===
bool DenseConsRajaEx2::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const size_type& num_cons,
                                     const index_type* idx_cons,
                                     const double* /*x*/,
                                     bool /*new_x*/,
                                     double* Jac)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(0, num_cons * nloc),
    RAJA_LAMBDA(index_type idx) {
      index_type ci = idx / nloc;
      index_type lj = idx % nloc;
      index_type gi = idx_cons[ci];
      double v = 1.0;

      if (gi == 1) {
        v = (lj == 0 ? 2.0 : 1.0);
      } else if (gi == 2) {
        v = (lj == 0 ? 2.0 : (lj == 1 ? 0.5 : 1.0));
      } else if (gi == 3) {
        v = (lj == 0 ? 4.0 : (lj <= 2 ? 2.0 : 1.0));
      }

      Jac[ci * nloc + lj] = v;
    });

  return true;
}

// === Starting point ===
bool DenseConsRajaEx2::get_starting_point(const size_type& n,
                                          double* x0)
{
  index_type start = col_partition_[my_rank_];
  index_type end   = col_partition_[my_rank_ + 1];
  index_type nloc  = end - start;

  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(0, nloc),
    RAJA_LAMBDA(index_type li) {
      x0[li] = 1.0;
    });

  return true;
}