#include "NlpDenseConsEx4.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

DenseConsEx4::DenseConsEx4()
    : n_vars_(2),
      n_cons_(4),
      unconstrained_(false)
{
  comm_size = 1;
  my_rank = 0;
#ifdef HIOP_USE_MPI
  comm = MPI_COMM_WORLD;
  int ierr = MPI_Comm_size(comm, &comm_size);
  assert(MPI_SUCCESS == ierr);
  ierr = MPI_Comm_rank(comm, &my_rank);
  assert(MPI_SUCCESS == ierr);
#endif

  if(unconstrained_) {
    n_cons_ = 0;
  }

  // set up vector distribution for primal variables - easier to store it as a member in this simple example
  col_partition_ = new index_type[comm_size + 1];
  index_type quotient = n_vars_ / comm_size;
  index_type remainder = n_vars_ - comm_size * quotient;
  // if(my_rank==0) printf("reminder=%llu quotient=%llu\n", remainder, quotient);
  int i = 0;
  col_partition_[i++] = 0;
  while(i <= remainder) {
    col_partition_[i] = col_partition_[i - 1] + quotient + 1;
    i++;
  }
  while(i <= comm_size) {
    col_partition_[i] = col_partition_[i - 1] + quotient;
    i++;
  }
}
DenseConsEx4::~DenseConsEx4() { delete[] col_partition_; }

bool DenseConsEx4::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

bool DenseConsEx4::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  index_type i_local;
  for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
    i_local = idx_global2local(n, i);
    if(i == 0) {
      xlow[i_local] = 0.;
      xupp[i_local] = 11.;
      type[i_local] = hiopNonlinear;
      continue;
    }
    if(i == 1) {
      xlow[i_local] = 0.;
      xupp[i_local] = 11.;
      type[i_local] = hiopNonlinear;
      continue;
    }
  }
  return true;
}

bool DenseConsEx4::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons_);
  if(!unconstrained_) {
    clow[0] = 0.0;
    cupp[0] = 1e20;
    type[0] = hiopInterfaceBase::hiopLinear;

    clow[1] = -1e20;
    cupp[1] = 10.0;
    type[1] = hiopInterfaceBase::hiopLinear;

    clow[2] = -1e20;
    cupp[2] = 64.0;
    type[2] = hiopInterfaceBase::hiopLinear;

    clow[3] = -1e20;
    cupp[3] = 100.;
    type[3] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}

bool DenseConsEx4::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  obj_value = 0.;

  index_type i_local;
  for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
    i_local = idx_global2local(n, i);

    if(i == 0) {
      obj_value += -3. * x[i_local] * x[i_local];
      continue;
    }
    if(i == 1) {
      obj_value += -2. * x[i_local] * x[i_local];
      continue;
    }
  }

#ifdef HIOP_USE_MPI
  double obj_global;
  int ierr = MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm);
  assert(ierr == MPI_SUCCESS);
  obj_value = obj_global;
#endif
  return true;
}

bool DenseConsEx4::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  index_type i_local;
  for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
    i_local = idx_global2local(n, i);

    if(i == 0) {
      gradf[i_local] = -6. * x[i_local];
      continue;
    }
    if(i == 1) {
      gradf[i_local] = -4. * x[i_local];
      continue;
    }
  }

  return true;
}

/* Four constraints no matter how large n is */
bool DenseConsEx4::eval_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             double* cons)
{
  if(unconstrained_) {
    assert(m == 0);
    return true;
  }

  assert(n == n_vars_);
  assert(m == n_cons_);
  assert(n_cons_ == 4);
  assert(num_cons <= m);
  assert(num_cons >= 0);
  // local contributions to the constraints in cons are reset
  for(int j = 0; j < num_cons; j++) {
    cons[j] = 0.;
  }
  // compute the constraint one by one.
  for(int itcon = 0; itcon < num_cons; itcon++) {
    // --- constraint 1 body ---> sum x_i = n+1
    if(idx_cons[itcon] == 0) {
      index_type i_local;
      for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
        i_local = idx_global2local(n, i);
        if(i == 0) {
          cons[itcon] += -0.06 * x[i_local] * x[i_local];
        }
        if(i == 1) {
          cons[itcon] += x[i_local];
        }
      }
      continue;  // done with this constraint
    }

    // --- constraint 2 body ---> 2*x_1 + sum {x_i : i=2,...,n}
    if(idx_cons[itcon] == 1) {
      index_type i_local;
      for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
        i_local = idx_global2local(n, i);
        if(i == 0) {
          cons[itcon] += 0.05 * x[i_local] * x[i_local];
        }
        if(i == 1) {
          cons[itcon] += x[i_local];
        }
      }
      continue;
    }

    // --- constraint 3 body ---> 2*x_1 + 0.5*x_2 + sum{x_i : i=3,...,n}
    if(idx_cons[itcon] == 2) {
      index_type i_local;
      for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
        i_local = idx_global2local(n, i);
        if(i == 1) {
          cons[itcon] += x[i_local] * x[i_local];
        }
      }
      continue;
    }

    // --- constraint 4 body ---> 4*x_1 + 2*x_2 + 2*x_3 + sum{x_i : i=4,...,n}
    if(idx_cons[itcon] == 3) {
      index_type i_local;
      for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
        i_local = idx_global2local(n, i);
        if(i == 0) {
          cons[itcon] += x[i_local] * x[i_local];
        }
      }
      continue;
    }
  }  // end for loop over constraints

#ifdef HIOP_USE_MPI
  if(num_cons > 0) {
    double* cons_global = new double[num_cons];
    int ierr = MPI_Allreduce(cons, cons_global, num_cons, MPI_DOUBLE, MPI_SUM, comm);
    assert(ierr == MPI_SUCCESS);
    memcpy(cons, cons_global, num_cons * sizeof(double));
    delete[] cons_global;
  }
#endif

  return true;
}

bool DenseConsEx4::eval_Jac_cons(const size_type& n,
                                 const size_type& m,
                                 const size_type& num_cons,
                                 const index_type* idx_cons,
                                 const double* x,
                                 bool new_x,
                                 double* Jac)
{
  if(unconstrained_) {
    assert(m == 0);
    return true;
  }

  assert(n == n_vars_);
  assert(m == n_cons_);
  size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
  // here we will iterate over the local indexes, however we still need to work with the
  // global indexes to correctly determine the entries in the Jacobian corresponding
  // to the 'rebels' variables x_1, x_2, x_3

  for(int itcon = 0; itcon < num_cons; itcon++) {
    assert(itcon * n_local + n_local <= n_local * num_cons);

    // Jacobian of constraint 1
    if(idx_cons[itcon] == 0) {
      for(int i = 0; i < n_local; ++i) {
        if(idx_local2global(n, i) == 0) {
          Jac[itcon * n_local + i] = -0.12 * x[i];
        } else if(idx_local2global(n, i) == 1) {
          Jac[itcon * n_local + i] = 1.0;
        }
      }
      continue;
    }

    // Jacobian of constraint 2
    if(idx_cons[itcon] == 1) {
      for(int i = 0; i < n_local; ++i) {
        if(idx_local2global(n, i) == 0) {
          Jac[itcon * n_local + i] = 0.1 * x[i];
        } else if(idx_local2global(n, i) == 1) {
          Jac[itcon * n_local + i] = 1.0;
        }
      }
      continue;
    }

    // Jacobian of constraint 3
    if(idx_cons[itcon] == 2) {
      for(int i = 0; i < n_local; ++i) {
        if(idx_local2global(n, i) == 0) {
          Jac[itcon * n_local + i] = 0.0;
        } else if(idx_local2global(n, i) == 1) {
          Jac[itcon * n_local + i] = 2. * x[i];
        }
      }
      continue;
    }

    // Jacobian of constraint  4
    if(idx_cons[itcon] == 3) {
      for(int i = 0; i < n_local; ++i) {
        if(idx_local2global(n, i) == 0) {
          Jac[itcon * n_local + i] = 2. * x[i];
        } else if(idx_local2global(n, i) == 1) {
          Jac[itcon * n_local + i] = 0.0;
        }
      }
    }
  }
  return true;
}

bool DenseConsEx4::get_vecdistrib_info(size_type global_n, index_type* cols)
{
  if(global_n == n_vars_) {
    for(int i = 0; i <= comm_size; i++) {
      cols[i] = col_partition_[i];
    }
  } else {
    assert(false && "You shouldn't need distrib info for this size.");
  }
  return true;
}

bool DenseConsEx4::get_starting_point(const size_type& global_n, double* x0)
{
  assert(global_n == n_vars_);
  size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
  for(index_type i = 0; i < n_local; i++) {
    x0[i] = 0.0;
  }
  return true;
}
