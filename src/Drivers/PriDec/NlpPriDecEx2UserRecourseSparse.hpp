#ifndef HIOP_EXAMPLE_PRIDEC_EX2_SPARSE_RECOURSE
#define HIOP_EXAMPLE_PRIDEC_EX2_SPARSE_RECOURSE

#include <cstring>  //for memcpy
#include <cstdio>

using size_type = hiop::size_type;
using index_type = hiop::index_type;
/** This class provide an example of what a user of hiop::hiopInterfacePriDecProblem
 * should implement in order to provide the recourse problem to
 * hiop::hiopAlgPrimalDecomposition solver
 *
 * For a given vector x\in R^n and \xi \in R^{n_S}, this example class implements
 *
 *     r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that
 *
 *                   (1-y_1 + \xi^i_1)^2 + \sum_{k=2}^{n_S} (y_k+\xi^i_k)^2
 *
 *                                       + \sum_{k=n_S+1}^{n_y} y_k^2 >= 1   //last one in the constraint implementation
 *
 *                   y_k - y_{k-1} >=0, k=2, ..., n_y
 *
 *                   y_1 >=0
 *
 * The recourse problem is of Sparse HiOp input.
 */

class PriDecRecourseProbleEx2Sparse : public hiop::hiopInterfaceSparse
{
public:
  PriDecRecourseProbleEx2Sparse(int n, int nS, int S)
      : x_(nullptr),
        xi_(nullptr),
        nx_(n),
        nS_(nS),
        S_(S)
  {
    assert(nS_ >= 1);
    assert(nx_ >= nS_);  // ny = nx = n
    assert(S_ >= 1);
    ny_ = nx_;
  }

  PriDecRecourseProbleEx2Sparse(int n, int nS, int S, const double* x, const double* xi)
      : nx_(n),
        nS_(nS),
        S_(S)
  {
    assert(nS_ >= 1);
    assert(nx_ >= nS_);  // ny = nx = n
    assert(S_ >= 1);

    ny_ = nx_;
    xi_ = new double[nS_];
    memcpy(xi_, xi, nS_ * sizeof(double));
    x_ = new double[nx_];

    memcpy(x_, x, nx_ * sizeof(double));
  }

  PriDecRecourseProbleEx2Sparse(int n, int nS, int S, int idx, const double* x, const double* xi)
      : nx_(n),
        nS_(nS),
        S_(S)
  {
    assert(nS_ >= 1);
    assert(nx_ >= nS_);  // ny=nx=n
    assert(S_ >= 1);

    ny_ = nx_;
    xi_ = new double[nS_];
    memcpy(xi_, xi, nS_ * sizeof(double));
    x_ = new double[nx_];

    memcpy(x_, x, nx_ * sizeof(double));
    idx_ = idx;
  }

  virtual ~PriDecRecourseProbleEx2Sparse()
  {
    delete[] x_;
    delete[] xi_;
  }

  /// Set the basecase solution `x`
  void set_x(const double* x)
  {
    if(x_ == NULL) {
      x_ = new double[nx_];
    }
    memcpy(x_, x, nx_ * sizeof(double));
  }

  /// Set the "sample" vector \xi
  void set_center(const double* xi)
  {
    if(xi_ == NULL) {
      xi_ = new double[nS_];
    }
    memcpy(xi_, xi, nS_ * sizeof(double));
  }

  bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = ny_;
    m = ny_;
    return true;
  }

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    // y_1 bounded
    xlow[0] = 0.;
    xupp[0] = 1e20;
    for(int i = 1; i < n; ++i) {
      xlow[i] = -1e+20;
    }
    for(int i = 1; i < n; ++i) {
      xupp[i] = +1e+20;
    }
    for(int i = 0; i < n; ++i) {
      type[i] = hiopNonlinear;
    }
    return true;
  }

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == ny_);
    for(int i = 0; i < ny_ - 1; i++) {
      clow[i] = 0.;
      cupp[i] = 1e20;
    }

    // clow[ny_-1] = 0.;
    clow[ny_ - 1] = 1.;
    cupp[ny_ - 1] = 1e20;
    return true;
  }

  bool get_sparse_blocks_info(size_type& nx,
                              size_type& nnz_sparse_Jaceq,
                              size_type& nnz_sparse_Jacineq,
                              size_type& nnz_sparse_Hess_Lagr)
  {
    nx = ny_;
    assert(nx > 0);

    nnz_sparse_Jaceq = 0;
    nnz_sparse_Jacineq = ny_ + (ny_ - 1) * 2;

    nnz_sparse_Hess_Lagr = ny_;  // Lagrangian
    return true;
  }

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    assert(ny_ == n);
    obj_value = 0.;
    for(int i = 0; i < n; i++) {
      obj_value += (x[i] - x_[i]) * (x[i] - x_[i]);
    }
    obj_value *= 0.5;
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
    assert(n == ny_);
    assert(m == ny_);
    assert(num_cons == ny_ || num_cons == 0);
    if(num_cons == 0) {
      return true;
    }

    for(auto j = 0; j < m; j++) {
      cons[j] = 0.;
    }

    for(int irow = 0; irow < num_cons; irow++) {
      const int con_idx = (int)idx_cons[irow];
      if(con_idx < m - 1) {
        cons[con_idx] = x[con_idx + 1] - x[con_idx];
      } else {
        assert(con_idx == m - 1);
        cons[m - 1] = (1 - x[0] + xi_[0]) * (1 - x[0] + xi_[0]);
        for(int i = 1; i < nS_; i++) {
          cons[m - 1] += (x[i] + xi_[i]) * (x[i] + xi_[i]);
        }
        for(int i = nS_; i < nx_; i++) {
          cons[m - 1] += x[i] * x[i];
        }
      }
    }
    return true;
  }

  // r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    assert(ny_ == n);
    for(int i = 0; i < nx_; i++) {
      gradf[i] = (x[i] - x_[i]);
    }
    return true;
  }

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS)
  {
    assert(num_cons == nx_ || num_cons == 0);
    // indexes for sparse part
    if(num_cons == 0) {
      return true;
    }

    if(iJacS != NULL && jJacS != NULL) {
      int nnzit = 0;
      for(int itrow = 0; itrow < num_cons; itrow++) {
        const int con_idx = (int)idx_cons[itrow];
        if(con_idx < ny_ - 1) {
          // sparse Jacobian eq w.r.t. x and s
          // y_k
          iJacS[nnzit] = con_idx;
          jJacS[nnzit] = con_idx;  //-1
          nnzit++;

          // y_{k+1}
          iJacS[nnzit] = con_idx;
          jJacS[nnzit] = con_idx + 1;  // 1
          nnzit++;
        } else if(con_idx == m - 1) {
          iJacS[nnzit] = m - 1;
          jJacS[nnzit] = 0;
          nnzit++;
          // cons[m-1] = (1-x[0]+xi_[0]);

          for(int i = 1; i < m; i++) {
            iJacS[nnzit] = m - 1;
            jJacS[nnzit] = i;
            nnzit++;
            // cons[m-1] += x[i]*x[i];
          }
          // sparse Jacobian ineq w.r.t x and s
        }
      }
      assert(nnzit == nnzJacS);
    }
    // values for sparse Jacobian if requested by the solver
    if(MJacS != NULL) {
      int nnzit = 0;
      for(int itrow = 0; itrow < num_cons; itrow++) {
        const int con_idx = (int)idx_cons[itrow];
        if(con_idx < m - 1) {
          // sparse Jacobian eq w.r.t. x and s
          // y_{k+1}
          MJacS[nnzit] = -1.;
          nnzit++;
          // y_k
          MJacS[nnzit] = 1.;
          nnzit++;
        } else if(con_idx == m - 1) {
          MJacS[nnzit] = -2 * (1 - x[0] + xi_[0]);
          nnzit++;
          // cons[m-1] = (1-x[0]+xi_[0])^2;
          assert(m >= nS_);
          for(int i = 1; i < nS_; i++) {
            MJacS[nnzit] = 2 * (x[i] + xi_[i]);
            nnzit++;
            // cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
          }
          for(int i = nS_; i < m; i++) {
            MJacS[nnzit] = 2 * x[i];
            nnzit++;
            // cons[m-1] += x[i]*x[i];
          }
          // sparse Jacobian ineq w.r.t x and s
        }
      }
      assert(nnzit == nnzJacS);
    }
    // assert("for debugging" && false); //for debugging purpose
    return true;
  }

  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const int& nnzHSS,
                      int* iHSS,
                      int* jHSS,
                      double* MHSS)
  {
    assert(nnzHSS == m);
    // r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that
    if(iHSS != NULL && jHSS != NULL) {
      for(int i = 0; i < m; i++) {
        iHSS[i] = jHSS[i] = i;
      }
    }

    if(MHSS != NULL) {
      for(int i = 0; i < m; i++) {
        MHSS[i] = obj_factor;
      }
      MHSS[0] += 2 * lambda[m - 1];
      for(int i = 1; i < m; i++) {
        MHSS[i] += lambda[m - 1] * 2.;
      }
    }
    return true;
  }

  // Implementation of the primal starting point specification
  bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == nx_);
    for(int i = 0; i < global_n; i++) {
      x0[i] = 1.;
    }
    return true;
  }

  bool get_starting_point(const size_type& n,
                          const size_type& m,
                          double* x0,
                          bool& duals_avail,
                          double* z_bndL0,
                          double* z_bndU0,
                          double* lambda0,
                          bool& slacks_avail,
                          double* ineq_slack)
  {
    return false;
  }

  /**
   * This function computes the derivative of the recourse function with respect to x in the problem description,
   * which is the x_ in the protected variable, while x in the function implementation
   * represents y in the problem description.
   */
  bool compute_gradx(const int n, const double* y, double* gradx)
  {
    assert(nx_ == n);
    for(int i = 0; i < nx_; i++) {
      gradx[i] = (x_[i] - y[i]);
    }
    return true;
  };

  /**
   * Returns COMM_SELF communicator since this example is only intended to run
   * on one MPI process
   */
  bool get_MPI_comm(MPI_Comm& comm_out)
  {
    comm_out = MPI_COMM_SELF;
    return true;
  }

protected:
  double* x_;
  double* xi_;
  int nx_;  // In this example, n_ = nx_ = ny_.
  int ny_;
  int nS_;
  int S_;
  int idx_;
};

#endif
