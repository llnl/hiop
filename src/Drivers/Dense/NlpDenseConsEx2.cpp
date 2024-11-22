#include "NlpDenseConsEx2.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

DenseConsEx2::DenseConsEx2(int n, bool unconstrained)
    : n_vars_(n),
      n_cons_(4),
      unconstrained_(unconstrained)
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
DenseConsEx2::~DenseConsEx2() { delete[] col_partition_; }

bool DenseConsEx2::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

bool DenseConsEx2::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  index_type i_local;
  for(index_type i = col_partition_[my_rank]; i < col_partition_[my_rank + 1]; i++) {
    i_local = idx_global2local(n, i);
    if(i == 0) {
      xlow[i_local] = -1e20;
      xupp[i_local] = 1e20;
      type[i_local] = hiopNonlinear;
      continue;
    }
    if(i == 1) {
      xlow[i_local] = 0.0;
      xupp[i_local] = 1e20;
      type[i_local] = hiopNonlinear;
      continue;
    }
    if(i == 2) {
      xlow[i_local] = 1.5;
      xupp[i_local] = 10.0;
      type[i_local] = hiopNonlinear;
      continue;
    }
    // this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
    xlow[i_local] = 0.5;
    xupp[i_local] = 1e20;
    type[i_local] = hiopNonlinear;
  }
  return true;
}
bool DenseConsEx2::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons_);
  if(!unconstrained_) {
    clow[0] = n_vars_ + 1;
    cupp[0] = n_vars_ + 1;
    type[0] = hiopInterfaceBase::hiopLinear;

    clow[1] = 5.0;
    cupp[1] = 1e20;
    type[1] = hiopInterfaceBase::hiopLinear;

    clow[2] = 1.0;
    cupp[2] = 2 * n_vars_;
    type[2] = hiopInterfaceBase::hiopLinear;

    clow[3] = -1e20;
    cupp[3] = 4 * n_vars_;
    type[3] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}
bool DenseConsEx2::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
  obj_value = 0.;
  for(int i = 0; i < n_local; i++) {
    obj_value += 0.25 * pow(x[i] - 1., 4);
  }
#ifdef HIOP_USE_MPI
  double obj_global;
  int ierr = MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm);
  assert(ierr == MPI_SUCCESS);
  obj_value = obj_global;
#endif
  return true;
}
bool DenseConsEx2::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
  for(int i = 0; i < n_local; i++) {
    gradf[i] = pow(x[i] - 1., 3);
  }
  return true;
}

/* Four constraints no matter how large n is */
bool DenseConsEx2::eval_cons(const size_type& n,
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
      size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
      // loop over local x in local indexes and add its entries to the result
      for(int i = 0; i < n_local; i++) {
        cons[itcon] += x[i];
      }
      continue;  // done with this constraint
    }

    // --- constraint 2 body ---> 2*x_1 + sum {x_i : i=2,...,n}
    if(idx_cons[itcon] == 1) {
      int i_local;
      // loop over local x in global indexes
      for(size_type i_global = col_partition_[my_rank]; i_global < col_partition_[my_rank + 1]; i_global++) {
        i_local = idx_global2local(n, i_global);
        // x_1 has a different contribution to constraint 2 than the rest
        if(i_global == 0) {
          cons[itcon] += 2 * x[i_local];
        } else {
          cons[itcon] += x[i_local];
        }
      }
      continue;
    }
    // --- constraint 3 body ---> 2*x_1 + 0.5*x_2 + sum{x_i : i=3,...,n}
    if(idx_cons[itcon] == 2) {
      int i_local;
      // loop over x in global indexes
      for(size_type i_global = col_partition_[my_rank]; i_global < col_partition_[my_rank + 1]; i_global++) {
        i_local = idx_global2local(n, i_global);
        // x_1 and x_2 have a different contributions to constraint 3 than the rest
        if(i_global == 0) {
          cons[itcon] += 2.0 * x[i_local];
        } else if(i_global == 1) {
          cons[itcon] += 0.5 * x[i_local];
        } else {
          cons[itcon] += x[i_local];
        }
      }
      continue;
    }
    // --- constraint 4 body ---> 4*x_1 + 2*x_2 + 2*x_3 + sum{x_i : i=4,...,n}
    if(idx_cons[itcon] == 3) {
      int i_local;
      // loop over x in global indexes
      for(size_type i_global = col_partition_[my_rank]; i_global < col_partition_[my_rank + 1]; i_global++) {
        i_local = idx_global2local(n, i_global);
        // x_1, x_2, and x_3 have a different contributions to constraint 3 than the rest
        if(i_global == 0) {
          cons[itcon] += 4 * x[i_local];
        } else if(i_global == 1 || i_global == 2) {
          cons[itcon] += 2 * x[i_local];
        } else {
          cons[itcon] += x[i_local];
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

bool DenseConsEx2::eval_Jac_cons(const size_type& n,
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
  int i;
  // here we will iterate over the local indexes, however we still need to work with the
  // global indexes to correctly determine the entries in the Jacobian corresponding
  // to the 'rebels' variables x_1, x_2, x_3

  for(int itcon = 0; itcon < num_cons; itcon++) {
    assert(itcon * n_local + n_local <= n_local * num_cons);

    // Jacobian of constraint 1 is all ones
    if(idx_cons[itcon] == 0) {
      for(i = 0; i < n_local; i++) {
        Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
      }
      continue;
    }

    // Jacobian of constraint 2 is all ones except the first entry, which is 2
    if(idx_cons[itcon] == 1) {
      for(i = 1; i < n_local; i++) {
        Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
      }
      // this is an overkill, but finding it useful for educational purposes
      // is local index 0 the global index 0 (x_1)? If yes the Jac should be 2.0

      //! Jac[itcon][0] = idx_local2global(n,0)==0?2.:1.;
      Jac[itcon * n_local + 0] = idx_local2global(n, 0) == 0 ? 2. : 1.;
      continue;
    }

    // Jacobian of constraint 3
    if(idx_cons[itcon] == 2) {
      for(i = 2; i < n_local; i++) {
        Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
      }
      //! Jac[itcon][0] = idx_local2global(n,0)==0?2.:1.;
      Jac[itcon * n_local + 0] = idx_local2global(n, 0) == 0 ? 2. : 1.;
      //! Jac[itcon][1] = idx_local2global(n,1)==1?0.5:1.;
      Jac[itcon * n_local + 1] = idx_local2global(n, 1) == 1 ? 0.5 : 1.;
      continue;
    }

    // Jacobian of constraint  4
    if(idx_cons[itcon] == 3) {
      for(i = 2; i < n_local; i++) {
        Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
      }
      //! Jac[itcon][0] = idx_local2global(n,0)==0?4.:1.;
      Jac[itcon * n_local + 0] = idx_local2global(n, 0) == 0 ? 4. : 1.;
      //! Jac[itcon][1] = idx_local2global(n,1)==1?2.:1.;
      Jac[itcon * n_local + 1] = idx_local2global(n, 1) == 1 ? 2. : 1.;
      //! Jac[itcon][2] = idx_local2global(n,2)==2?2.:1.;
      Jac[itcon * n_local + 2] = idx_local2global(n, 2) == 2 ? 2. : 1.;
    }
  }
  return true;
}

bool DenseConsEx2::get_vecdistrib_info(size_type global_n, index_type* cols)
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

bool DenseConsEx2::get_starting_point(const size_type& global_n, double* x0)
{
  assert(global_n == n_vars_);
  size_type n_local = col_partition_[my_rank + 1] - col_partition_[my_rank];
  for(index_type i = 0; i < n_local; i++) {
    x0[i] = 0.0;
  }
  return true;
}
