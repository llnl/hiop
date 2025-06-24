#include "NlpDenseConsRajaEx2.hpp"
#include <cmath>
#include <cstring>
#include <RAJA/RAJA.hpp>

DenseConsRajaEx2::DenseConsRajaEx2(int n, bool unconstrained)
  : n_vars_(n), n_cons_(4), unconstrained_(unconstrained)
{
#ifdef HIOP_USE_MPI
  comm_ = MPI_COMM_WORLD;
  MPI_Comm_size(comm_, &comm_size_);
  MPI_Comm_rank(comm_, &my_rank_);
#else
  my_rank_   = 0;
  comm_size_ = 1;
#endif

  if(unconstrained_) n_cons_ = 0;

  // partition variables for each rank
  col_partition_ = new index_type[comm_size_+1];
  index_type q = n_vars_ / comm_size_;
  index_type r = n_vars_ - comm_size_ * q;
  for(int i = 0; i <= comm_size_; ++i) {
    col_partition_[i] = i * q + (i < r ? i : r);
  }
}

DenseConsRajaEx2::~DenseConsRajaEx2()
{
  delete[] col_partition_;
}

bool DenseConsRajaEx2::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

bool DenseConsRajaEx2::get_vecdistrib_info(size_type global_n, index_type* cols)
{
  for(int i = 0; i <= comm_size_; ++i) {
    cols[i] = col_partition_[i];
  }
  return true;
}

bool DenseConsRajaEx2::get_vars_info(const size_type& n,
                                     double* xlow,
                                     double* xupp,
                                     NonlinearityType* type)
{
  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(col_partition_[my_rank_], col_partition_[my_rank_+1]),
    RAJA_LAMBDA(index_type i) {
      index_type li = i - col_partition_[my_rank_];
      if(i == 0) {
        xlow[li] = -1e20; xupp[li] =  1e20;
      } else if(i == 1) {
        xlow[li] =  0.0;  xupp[li] =  1e20;
      } else if(i == 2) {
        xlow[li] =  1.5;  xupp[li] = 10.0;
      } else {
        xlow[li] =  0.5;  xupp[li] =  1e20;
      }
      type[li] = hiopNonlinear;
    });
  return true;
}

bool DenseConsRajaEx2::get_cons_info(const size_type& m,
                                     double* clow,
                                     double* cupp,
                                     NonlinearityType* type)
{
  if(unconstrained_) return true;
  // constraint 0: sum x_i == n_vars_ + 1
  clow[0] = n_vars_ + 1; cupp[0] = n_vars_ + 1;
  // constraint 1: 2*x0 + sum_{i=1..} x_i >= 5
  clow[1] = 5.0;         cupp[1] = 1e20;
  // constraint 2: 2*x0 + 0.5*x1 + sum_{i=2..} x_i in [1,2*n_vars_]
  clow[2] = 1.0;         cupp[2] = 2.0 * n_vars_;
  // constraint 3: 4*x0 + 2*x1 + 2*x2 + sum_{i=3..} x_i <= 4*n_vars_
  clow[3] = -1e20;       cupp[3] = 4.0 * n_vars_;

  for(int j = 0; j < m; ++j) {
    type[j] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}

bool DenseConsRajaEx2::eval_f(const size_type& n,
                              const double* x,
                              bool new_x,
                              double& obj_value)
{
  RAJA::ReduceSum<reduce_pol, double> red(0.0);
  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(col_partition_[my_rank_], col_partition_[my_rank_+1]),
    RAJA_LAMBDA(index_type i) {
      double xi = x[i];
      red += 0.25 * (xi - 1.0) * (xi - 1.0) * (xi - 1.0) * (xi - 1.0);
    });
  obj_value = red.get();

#ifdef HIOP_USE_MPI
  double tmp = obj_value;
  MPI_Allreduce(&tmp, &obj_value, 1, MPI_DOUBLE, MPI_SUM, comm_);
#endif
  return true;
}

bool DenseConsRajaEx2::eval_grad_f(const size_type& n,
                                    const double* x,
                                    bool new_x,
                                    double* gradf)
{
  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(col_partition_[my_rank_], col_partition_[my_rank_+1]),
    RAJA_LAMBDA(index_type i) {
      double xi = x[i];
      gradf[i - col_partition_[my_rank_]] = (xi - 1.0) * (xi - 1.0) * (xi - 1.0);
    });
  return true;
}

bool DenseConsRajaEx2::eval_cons(const size_type& n,
                                  const size_type& m,
                                  const size_type& num_cons,
                                  const index_type* idx_cons,
                                  const double* x,
                                  bool new_x,
                                  double* cons)
{
  // initialize
  for(int j = 0; j < num_cons; ++j) cons[j] = 0.0;

  for(int j = 0; j < num_cons; ++j) {
    int ci = idx_cons[j];
    RAJA::ReduceSum<reduce_pol, double> red(0.0);
    RAJA::forall<exec_pol>(
      RAJA::RangeSegment(col_partition_[my_rank_], col_partition_[my_rank_+1]),
      RAJA_LAMBDA(index_type i) {
        double xi = x[i];
        switch(ci) {
          case 0: red += xi; break;
          case 1: red += (i == col_partition_[my_rank_] ? 2.0 * xi : xi); break;
          case 2:
            red += (i == col_partition_[my_rank_] ? 2.0 * xi :
                    (i == col_partition_[my_rank_] + 1 ? 0.5 * xi : xi));
            break;
          default:
            red += (i == col_partition_[my_rank_]     ? 4.0 * xi :
                    (i <= col_partition_[my_rank_] + 2 ? 2.0 * xi : xi));
        }
      });
    cons[j] = red.get();
  }

#ifdef HIOP_USE_MPI
  if(num_cons > 0) {
    std::vector<double> tmp(num_cons);
    MPI_Allreduce(cons, tmp.data(), num_cons, MPI_DOUBLE, MPI_SUM, comm_);
    std::memcpy(cons, tmp.data(), num_cons * sizeof(double));
  }
#endif
  return true;
}

bool DenseConsRajaEx2::eval_Jac_cons(const size_type& n,
                                      const size_type& m,
                                      const size_type& num_cons,
                                      const index_type* idx_cons,
                                      const double* x,
                                      bool new_x,
                                      double* Jac)
{
  index_type nloc = n_vars_;
  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(0, num_cons * nloc),
    RAJA_LAMBDA(index_type idx) {
      int ii = idx / nloc;
      int jj = idx % nloc;
      double v = 1.0;
      int ci = idx_cons[ii];
      if(ci == 1) v = (jj == 0 ? 2.0 : 1.0);
      else if(ci == 2) v = (jj == 0 ? 2.0 : (jj == 1 ? 0.5 : 1.0));
      else if(ci == 3) v = (jj == 0 ? 4.0 : (jj <= 2 ? 2.0 : 1.0));
      Jac[ii * nloc + jj] = v;
    });
  return true;
}

bool DenseConsRajaEx2::get_starting_point(const size_type& n,
                                           double* x0)
{
  RAJA::forall<exec_pol>(
    RAJA::RangeSegment(0, n_vars_),
    RAJA_LAMBDA(index_type i) { x0[i] = 1.0; });
  return true;
}
