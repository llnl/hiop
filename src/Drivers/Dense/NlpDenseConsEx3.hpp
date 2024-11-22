#ifndef HIOP_EXAMPLE_DENSE_EX3
#define HIOP_EXAMPLE_DENSE_EX3

#include "hiopInterface.hpp"

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

/* Problem test with fixed variables and related corner cases.
 *  min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
 *  s.t.
 *        sum x_i = n+1
 *        5<= 2*x_1 + sum {x_i : i=2,...,n}
 *        x_1=0 fixed
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *        x_i <=0.5, i=3n/4+1,...,n (additional fixed variables)
 */
class DenseConsEx3 : public hiop::hiopInterfaceDenseConstraints
{
public:
  DenseConsEx3(int n)
      : n_vars(n),
        n_cons(2)
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

    // set up vector distribution for primal variables - easier to store it as a member in this simple example
    col_partition = new index_type[comm_size + 1];
    size_type quotient = n_vars / comm_size;
    size_type remainder = n_vars - comm_size * quotient;

    int i = 0;
    col_partition[i++] = 0;
    while(i <= remainder) {
      col_partition[i] = col_partition[i - 1] + quotient + 1;
      i++;
    }
    while(i <= comm_size) {
      col_partition[i] = col_partition[i - 1] + quotient;
      i++;
    }
  };

  virtual ~DenseConsEx3() { delete[] col_partition; };

  virtual bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = n_vars;
    m = n_cons;
    return true;
  }

  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    assert(n >= 4 && "number of variables should be greater than 4 for this example");
    index_type i_local;
    for(index_type i = col_partition[my_rank]; i < col_partition[my_rank + 1]; i++) {
      i_local = idx_global2local(n, i);
      if(i == 0) {
        xlow[i_local] = 1.5;
        xupp[i_local] = 1.50;
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
      // this is for x_4, x_5, ... , x_n (i>=4), which are bounded till i=3/n4 and fixed after that
      xlow[i_local] = 0.5;
      type[i_local] = hiopNonlinear;
      if(i + 1 <= 3 * (n / 4.0))
        xupp[i_local] = 1e20;
      else
        xupp[i_local] = 0.50;
    }
    return true;
  }

  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == n_cons);
    clow[0] = n_vars + 1;
    cupp[0] = n_vars + 1;
    type[0] = hiopInterfaceBase::hiopLinear;
    clow[1] = 5.0;
    cupp[1] = 1e20;
    type[1] = hiopInterfaceBase::hiopLinear;
    return true;
  }

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    size_type n_local = col_partition[my_rank + 1] - col_partition[my_rank];
    obj_value = 0.;
    for(int i = 0; i < n_local; i++) obj_value += 0.25 * pow(x[i] - 1., 4);
#ifdef HIOP_USE_MPI
    double obj_global;
    int ierr = MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    assert(ierr == MPI_SUCCESS);
    obj_value = obj_global;
#endif
    return true;
  }

  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons)
  {
    assert(n == n_vars);
    assert(m == n_cons);
    assert(n_cons == 2);
    assert(num_cons <= m);
    assert(num_cons >= 0);
    // local contributions to the constraints in cons are reset
    for(int j = 0; j < num_cons; j++) cons[j] = 0.;

    // compute the constraint one by one.
    for(int itcon = 0; itcon < num_cons; itcon++) {
      // --- constraint 1 body ---> sum x_i = n+1
      if(idx_cons[itcon] == 0) {
        size_type n_local = col_partition[my_rank + 1] - col_partition[my_rank];
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
        for(size_type i_global = col_partition[my_rank]; i_global < col_partition[my_rank + 1]; i_global++) {
          i_local = idx_global2local(n, i_global);
          // x_1 has a different contribution to constraint 2 than the rest
          if(i_global == 0)
            cons[itcon] += 2 * x[i_local];
          else
            cons[itcon] += x[i_local];
        }
        continue;
      }
    }  // end for loop over constraints

#ifdef HIOP_USE_MPI
    double* cons_global = new double[num_cons];
    int ierr = MPI_Allreduce(cons, cons_global, num_cons, MPI_DOUBLE, MPI_SUM, comm);
    assert(ierr == MPI_SUCCESS);
    memcpy(cons, cons_global, num_cons * sizeof(double));
    delete[] cons_global;
#endif

    return true;
  }

  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    size_type n_local = col_partition[my_rank + 1] - col_partition[my_rank];
    for(int i = 0; i < n_local; i++) gradf[i] = pow(x[i] - 1., 3);
    return true;
  }

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             double* Jac)
  {
    assert(n == n_vars);
    assert(m == n_cons);
    size_type n_local = col_partition[my_rank + 1] - col_partition[my_rank];
    int i;
    // here we will iterate over the local indexes, however we still need to work with the
    // global indexes to correctly determine the entries in the Jacobian corresponding
    // to the 'rebels' variables x_1, x_2, x_3

    for(int itcon = 0; itcon < num_cons; itcon++) {
      // Jacobian of constraint 1 is all ones
      if(idx_cons[itcon] == 0) {
        for(i = 0; i < n_local; i++) Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
        continue;
      }

      // Jacobian of constraint 2 is all ones except the first entry, which is 2
      if(idx_cons[itcon] == 1) {
        for(i = 1; i < n_local; i++) Jac[itcon * n_local + i] = 1.0;  //! Jac[itcon][i]=1.0;
        // this is an overkill, but finding it useful for educational purposes
        // is local index 0 the global index 0 (x_1)? If yes the Jac should be 2.0
        //! Jac[itcon][0] = idx_local2global(n,0)==0?2.:1.;
        Jac[itcon * n_local + 0] = idx_local2global(n, 0) == 0 ? 2. : 1.;
        continue;
      }
    }
    return true;
  };

  virtual bool get_vecdistrib_info(size_type global_n, index_type* cols)
  {
    if(global_n == n_vars) {
      for(int i = 0; i <= comm_size; i++) {
        cols[i] = col_partition[i];
      }
    } else {
      assert(false && "You shouldn't need distrib info for this size.");
      return false;
    }
    return true;
  }

  virtual bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == n_vars);
    size_type n_local = col_partition[my_rank + 1] - col_partition[my_rank];
    for(int i = 0; i < n_local; i++) x0[i] = 0.0;
    return true;
  }

private:
  int n_vars, n_cons;
#ifdef HIOP_USE_MPI
  MPI_Comm comm;
#endif
  int my_rank, comm_size;
  index_type* col_partition;

public:
  inline size_type idx_local2global(size_type global_n, index_type idx_local)
  {
    assert(idx_local + col_partition[my_rank] < col_partition[my_rank + 1]);
    if(global_n == n_vars) return idx_local + col_partition[my_rank];
    assert(false && "you shouldn't need global index for a vector of this size.");
    return -1;
  }
  inline index_type idx_global2local(size_type global_n, index_type idx_global)
  {
    assert(idx_global >= col_partition[my_rank] && "global index does not belong to this rank");
    assert(idx_global < col_partition[my_rank + 1] && "global index does not belong to this rank");
    assert(global_n == n_vars && "your global_n does not match the number of variables?");
    return (idx_global - col_partition[my_rank]);
  }
};
#endif
