#include "NlpDenseConsRajaEx2.hpp"

DenseConsRajaEx2::DenseConsRajaEx2(int n, bool unconstrained)
  : n_vars_(n), n_cons_(4), unconstrained_(unconstrained)
{
#if defined(HIOP_USE_MPI)
  comm_ = MPI_COMM_WORLD;
  MPI_Comm_size(comm_, &comm_size_);
  MPI_Comm_rank(comm_, &my_rank_);
#else
  my_rank_    = 0;
  comm_size_  = 1;
#endif
  if(unconstrained_) n_cons_ = 0;

  // partition variables for each rank
  col_partition_ = new index_type[comm_size_+1];
  index_type q = n_vars_ / comm_size_;
  index_type r = n_vars_ - comm_size_*q;
  for(int i=0; i<=comm_size_; ++i) {
    col_partition_[i] = i*q + (i<r ? i : r);
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
  for(int i=0; i<=comm_size_; ++i) cols[i] = col_partition_[i];
  return true;
}

bool DenseConsRajaEx2::get_vars_info(const size_type& n,
                                     double* xlow,
                                     double* xupp,
                                     NonlinearityType* type)
{
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, n_vars_), RAJA_LAMBDA(RAJA::Index_type i) {
    if(i==0) {
      xlow[i] = -1e20; xupp[i] =  1e20;
    } else if(i==1) {
      xlow[i] =  0.0;  xupp[i] =  1e20;
    } else if(i==2) {
      xlow[i] =  1.5;  xupp[i] = 10.0;
    } else {
      xlow[i] =  0.5;  xupp[i] =  1e20;
    }
    type[i] = hiopNonlinear;
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
  // constraint 2: 2*x0 + 0.5*x1 + sum_{i=2..} x_i in [1,2*n]
  clow[2] = 1.0;         cupp[2] = 2.0 * n_vars_;
  // constraint 3: 4*x0 + 2*x1 + 2*x2 + sum_{i=3..} x_i <= 4*n
  clow[3] = -1e20;       cupp[3] = 4.0 * n_vars_;
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, n_cons_), RAJA_LAMBDA(RAJA::Index_type i) {
    type[i] = hiopNonlinear;
  });
  return true;
}

bool DenseConsRajaEx2::eval_f(const size_type& n,
                               const double* x,
                               bool new_x,
                               double& obj_value)
{
  RAJA::ReduceSum<reduce_pol, double> red(0.0);
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, n_vars_), RAJA_LAMBDA(RAJA::Index_type i) {
    red += 0.25 * pow(x[i] - 1.0, 4);
  });
  obj_value = red.get();
  return true;
}

bool DenseConsRajaEx2::eval_grad_f(const size_type& n,
                                    const double* x,
                                    bool new_x,
                                    double* gradf)
{
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, n_vars_), RAJA_LAMBDA(RAJA::Index_type i) {
    gradf[i] = pow(x[i] - 1.0, 3);
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
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, num_cons), RAJA_LAMBDA(RAJA::Index_type ii) {
    int ci = idx_cons[ii];
    if(ci == 0) {
      double sum = 0.0;
      for(int j = 0; j < n_vars_; ++j) sum += x[j];
      cons[ii] = sum;
    } else if(ci == 1) {
      double sum = 2.0 * x[0];
      for(int j = 1; j < n_vars_; ++j) sum += x[j];
      cons[ii] = sum;
    } else if(ci == 2) {
      double sum = 2.0 * x[0] + 0.5 * x[1];
      for(int j = 2; j < n_vars_; ++j) sum += x[j];
      cons[ii] = sum;
    } else if(ci == 3) {
      double sum = 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
      for(int j = 3; j < n_vars_; ++j) sum += x[j];
      cons[ii] = sum;
    }
  });
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
  // dense Jacobian: num_cons rows by n_vars_ cols
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, num_cons * n_vars_), RAJA_LAMBDA(RAJA::Index_type idx) {
    int ii = idx / n_vars_;
    int jj = idx % n_vars_;
    double v = 0.0;
    if(ii == 0) {
      v = 1.0;
    } else if(ii == 1) {
      v = (jj == 0 ? 2.0 : 1.0);
    } else if(ii == 2) {
      v = (jj == 0 ? 2.0 : (jj == 1 ? 0.5 : 1.0));
    } else if(ii == 3) {
      v = (jj == 0 ? 4.0 : (jj == 1 ? 2.0 : (jj == 2 ? 2.0 : 1.0)));
    }
    Jac[ii * n_vars_ + jj] = v;
  });
  return true;
}

bool DenseConsRajaEx2::get_starting_point(const size_type& n,
                                           double* x0)
{
  RAJA::forall<exec_pol>(RAJA::RangeSegment(0, n_vars_), RAJA_LAMBDA(RAJA::Index_type i) {
    x0[i] = 1.0;
  });
  return true;
}