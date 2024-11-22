#ifndef HIOP_EXAMPLE_PRIDEC_EX2_SPARSE_RAJA_RECOURSE
#define HIOP_EXAMPLE_PRIDEC_EX2_SPARSE_RAJA_RECOURSE

#include <cstring>  //for memcpy
#include <cstdio>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>
using size_type = hiop::size_type;
using index_type = hiop::index_type;

#if defined(HIOP_USE_CUDA)
#include <ExecPoliciesRajaCudaImpl.hpp>
using ex9_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_exec;
using ex9_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_reduce;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using ex9_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_exec;
using ex9_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_reduce;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using ex9_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaOmp>::hiop_raja_exec;
using ex9_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaOmp>::hiop_raja_reduce;
#endif

using namespace hiop;

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
 * The recourse problems are implemented in hiop sparse class with Raja to work in memory space.
 */

class PriDecRecourseProbleEx2Sparse : public hiop::hiopInterfaceSparse
{
public:
  PriDecRecourseProbleEx2Sparse(int n, int nS, int S, std::string mem_space)
      : x_(nullptr),
        xi_(nullptr),
        nx_(n),
        nS_(nS),
        S_(S),
        mem_space_(mem_space)
  {
    // Make sure mem_space_ is uppercase
    transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);

    // auto& resmgr = umpire::ResourceManager::getInstance();
    // umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    //  umpire::Allocator allocator
    //   = resmgr.getAllocator(mem_space_ == "DEFAULT" ? "HOST" : mem_space_);

    assert(nS_ >= 1);
    assert(nx_ >= nS_);  // ny = nx = n
    assert(S_ >= 1);
    ny_ = nx_;
  }

  PriDecRecourseProbleEx2Sparse(int n, int nS, int S, const double* x, const double* xi, std::string mem_space)
      : nx_(n),
        nS_(nS),
        S_(S),
        mem_space_(mem_space)
  {
    assert(nS_ >= 1);
    assert(nx_ >= nS_);
    assert(S_ >= 1);

    ny_ = nx_;
    // for(int i=0;i<nx_;i++) printf("x %d %18.12e ",i,x[i]);

    auto* x_vec = const_cast<double*>(x);
    auto* xi_vec = const_cast<double*>(xi);
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);

    xi_ = static_cast<double*>(allocator.allocate(nS_ * sizeof(double)));
    x_ = static_cast<double*>(allocator.allocate(nx_ * sizeof(double)));

    auto* str = allocator.getAllocationStrategy();
    resmgr.registerAllocation(x_vec, {x_vec, nx_ * sizeof(double), str});
    resmgr.registerAllocation(xi_vec, {xi_vec, nS_ * sizeof(double), str});

    resmgr.copy(x_, x_vec);
    resmgr.copy(xi_, xi_vec);

    resmgr.deregisterAllocation(x_vec);
    resmgr.deregisterAllocation(xi_vec);
  }

  PriDecRecourseProbleEx2Sparse(int n, int nS, int S, int idx, const double* x, const double* xi, std::string mem_space)
      : nx_(n),
        nS_(nS),
        S_(S),
        mem_space_(mem_space)
  {
    assert(nS_ >= 1);
    assert(nx_ >= nS_);  // ny=nx=n
    assert(S_ >= 1);

    ny_ = nx_;

    auto* x_vec = const_cast<double*>(x);
    auto* xi_vec = const_cast<double*>(xi);
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);

    xi_ = static_cast<double*>(allocator.allocate(nS_ * sizeof(double)));
    x_ = static_cast<double*>(allocator.allocate(nx_ * sizeof(double)));

    auto* str = allocator.getAllocationStrategy();
    resmgr.registerAllocation(x_vec, {x_vec, nx_ * sizeof(double), str});
    resmgr.registerAllocation(xi_vec, {xi_vec, nS_ * sizeof(double), str});

    resmgr.copy(x_, x_vec);
    resmgr.copy(xi_, xi_vec);

    resmgr.deregisterAllocation(x_vec);
    resmgr.deregisterAllocation(xi_vec);
    idx_ = idx;
  }

  virtual ~PriDecRecourseProbleEx2Sparse()
  {
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator allocator;
    if(mem_space_ == "DEFAULT") {
      allocator = resmgr.getAllocator("HOST");
    } else {
      allocator = resmgr.getAllocator(mem_space_);
    }

    allocator.deallocate(xi_);
    allocator.deallocate(x_);
  }

  // Set the basecase solution `x`
  // Assuming 'x' is not assigned by umpire
  void set_x(const double* x)
  {
    auto* x_vec = const_cast<double*>(x);

    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    if(x_ == NULL) {
      x_ = static_cast<double*>(allocator.allocate(nx_ * sizeof(double)));
    }
    auto* str = allocator.getAllocationStrategy();
    resmgr.registerAllocation(x_vec, {x_vec, nx_ * sizeof(double), str});
    resmgr.copy(x_, x_vec);
    resmgr.deregisterAllocation(x_vec);
  }

  /// Set the "sample" vector \xi
  void set_center(const double* xi)
  {
    auto* xi_vec = const_cast<double*>(xi);
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator allocator = resmgr.getAllocator(mem_space_);

    if(xi_ == NULL) {
      xi_ = static_cast<double*>(allocator.allocate(nS_ * sizeof(double)));
    }
    auto* str = allocator.getAllocationStrategy();
    resmgr.registerAllocation(xi_vec, {xi_vec, nS_ * sizeof(double), str});
    resmgr.copy(xi_, xi_vec);
    resmgr.deregisterAllocation(xi_vec);
  }

  bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = ny_;
    m = ny_;
    return true;
  }

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    RAJA::forall<ex9_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if(i == 0) {
            xlow[i] = 0.;  // y_1 bounded
          } else {
            xlow[i] = -1e+20;
          }
        });

    RAJA::forall<ex9_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if(i == 0) {
            xupp[i] = 1e20;
          } else {
            xupp[i] = 1e+20;
          }
        });

    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, n), [=](RAJA::Index_type i) { type[i] = hiopNonlinear; });
    return true;
  }

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == ny_);
    const auto d_ny = ny_;
    RAJA::forall<ex9_raja_exec>(
        RAJA::RangeSegment(0, ny_),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if(i == d_ny - 1) {
            clow[d_ny - 1] = 1.;
            cupp[d_ny - 1] = 1e20;
          } else {
            clow[i] = 0.;
            cupp[i] = 1e20;
          }
        });

    // clow[ny_-1] = 0.;

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
    RAJA::ReduceSum<ex9_raja_reduce, double> aux(0);  // why do we need reducesum?
    const auto d_x = x_;
    RAJA::forall<ex9_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) { aux += (x[i] - d_x[i]) * (x[i] - d_x[i]); });

    obj_value += aux.get();
    obj_value *= 0.5;
    return true;
  }

  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    // return false so that HiOp will rely on the constraint evaluator defined above
    return false;
  }
  bool eval_cons(const size_type& n,
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

    RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, m), RAJA_LAMBDA(RAJA::Index_type i) { cons[i] = 0.; });

    const auto* d_xi = xi_;
    const auto d_nx = nx_;
    const auto d_nS = nS_;
    RAJA::forall<ex9_raja_exec>(
        RAJA::RangeSegment(0, num_cons),
        RAJA_LAMBDA(RAJA::Index_type irow) {
          const int con_idx = (int)idx_cons[irow];
          if(con_idx < m - 1) {
            cons[con_idx] = x[con_idx + 1] - x[con_idx];
          } else {
            assert(con_idx == m - 1);
            cons[m - 1] = (1 - x[0] + d_xi[0]) * (1 - x[0] + d_xi[0]);
            //   RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(1, nS_),
            //     RAJA_LAMBDA(RAJA::Index_type i)
            //        {
            //        Using a for loop inside a RAJA loop:
            for(int i = 1; i < d_nS; i++) {
              cons[m - 1] += (x[i] + d_xi[i]) * (x[i] + d_xi[i]);
            }
            //});
            // RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(nS_, nx_),
            //  RAJA_LAMBDA(RAJA::Index_type i)
            //{
            for(int i = d_nS; i < d_nx; i++) {
              cons[m - 1] += x[i] * x[i];
            }
            //});
          }
        });
    return true;
  }

  // r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    assert(ny_ == n);
    const auto d_x = x_;
    RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, nx_), RAJA_LAMBDA(RAJA::Index_type i) { gradf[i] = (x[i] - d_x[i]); });
    return true;
  }

  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const double* x,
                     bool new_x,
                     const size_type& nnzJacS,
                     index_type* iJacS,
                     index_type* jJacS,
                     double* MJacS)
  {
    return false;
  }

  bool eval_Jac_cons(const size_type& n,
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

    // auto& resmgr = umpire::ResourceManager::getInstance();
    // umpire::Allocator allocator = resmgr.getAllocator(mem_space_);
    // index_type*  nnzit = static_cast<index_type*>(allocator.allocate(1 * sizeof(index_type)));
    // resmgr.memset(nnzit, 0);

    if(iJacS != NULL && jJacS != NULL) {
      const auto d_ny = ny_;
      RAJA::forall<ex9_raja_exec>(
          RAJA::RangeSegment(0, num_cons),
          RAJA_LAMBDA(RAJA::Index_type itrow) {
            const int con_idx = (int)idx_cons[itrow];
            if(con_idx < d_ny - 1) {
              // sparse Jacobian eq w.r.t. x and s
              // y_k
              iJacS[2 * itrow] = con_idx;
              jJacS[2 * itrow] = con_idx;  //-1
              // nnzit[0] = nnzit[0] + 1;

              // y_{k+1}
              iJacS[2 * itrow + 1] = con_idx;
              jJacS[2 * itrow + 1] = con_idx + 1;  // 1
              // nnzit[0] += 1;    // nnzit does nto work inside RAJA loop
            } else if(con_idx == m - 1) {
              assert(itrow == m - 1);
              iJacS[2 * (m - 1)] = m - 1;
              jJacS[2 * (m - 1)] = 0;
              // nnzit[0] += 1;
              // cons[m-1] = (1-x[0]+xi_[0]);

              // RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(1, m),
              //  [=] __device__ (RAJA::Index_type i)
              //{
              for(int i = 1; i < m; i++) {
                iJacS[2 * (m - 1) + i] = m - 1;
                jJacS[2 * (m - 1) + i] = i;
                // nnzit[0] += 1;
              }
              // cons[m-1] += x[i]*x[i];
              // sparse Jacobian ineq w.r.t x and s
            }
          });
      assert(2 * (m - 1) + m == nnzJacS);  // utilizing the structure of the nonzeros directly
      // assert(nnzit[0]==nnzJacS);
    }
    // values for sparse Jacobian if requested by the solver
    if(MJacS != NULL) {
      // nnzit[0] = 0;
      const auto* d_xi = xi_;
      const auto d_nS = nS_;
      RAJA::forall<ex9_raja_exec>(
          RAJA::RangeSegment(0, num_cons),
          RAJA_LAMBDA(RAJA::Index_type itrow) {
            const int con_idx = (int)idx_cons[itrow];
            if(con_idx < m - 1) {
              // sparse Jacobian eq w.r.t. x and s
              // yk+1
              MJacS[2 * itrow] = -1.;
              // nnzit[0] += 1;
              // yk
              MJacS[2 * itrow + 1] = 1.;
              // nnzit[0] += 1;
            } else if(con_idx == m - 1) {
              assert(itrow == m - 1);
              MJacS[2 * (m - 1)] = -2 * (1 - x[0] + d_xi[0]);
              // nnzit[0] += 1;
              // cons[m-1] = (1-x[0]+xi_[0])^2;
              assert(m >= d_nS);
              for(int i = 1; i < d_nS; i++) {
                MJacS[2 * (m - 1) + i] = 2 * (x[i] + d_xi[i]);
                // nnzit[0] += 1;
              }
              // cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
              for(int i = d_nS; i < m; i++) {
                MJacS[2 * (m - 1) + i] = 2 * x[i];
                // nnzit[0] += 1;
                // cons[m-1] += x[i]*x[i];
              }
              // sparse Jacobian ineq w.r.t x and s
            }
          });
      assert(2 * (m - 1) + m == nnzJacS);
    }
    // allocator.deallocate(nnzit);
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
    if(iHSS != NULL && jHSS != NULL) {
      RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, m), RAJA_LAMBDA(RAJA::Index_type i) { iHSS[i] = jHSS[i] = i; });
    }
    if(MHSS != NULL) {
      RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, m), RAJA_LAMBDA(RAJA::Index_type i) { MHSS[i] = obj_factor; });
      MHSS[0] += 2 * lambda[m - 1];
      RAJA::forall<ex9_raja_exec>(
          RAJA::RangeSegment(1, m),
          RAJA_LAMBDA(RAJA::Index_type i) { MHSS[i] += lambda[m - 1] * 2.; });
    }
    return true;
  }

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == nx_);
    RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, global_n), RAJA_LAMBDA(RAJA::Index_type i) { x0[i] = 1.; });
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
    duals_avail = false;
    slacks_avail = false;
    return false;
  }

  bool get_warmstart_point(const size_type& n,
                           const size_type& m,
                           double* x0,
                           double* z_bndL0,
                           double* z_bndU0,
                           double* lambda0,
                           double* ineq_slack,
                           double* vl0,
                           double* vu0)
  {
    return false;
  }

  /*
   * This function computes the derivative of the recourse function with respect to x in the problem description,
   * which is the x_ in the protected variable, while x in the function implementation
   * represents y in the problem description.
   */
  bool compute_gradx(const int n, const double* y, double* gradx)
  {
    assert(nx_ == n);
    const auto* d_x = x_;
    RAJA::forall<ex9_raja_exec>(RAJA::RangeSegment(0, nx_), RAJA_LAMBDA(RAJA::Index_type i) { gradx[i] = (d_x[i] - y[i]); });
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
  int nx_;  // n_ = nx = ny
  int ny_;
  int nS_;
  int S_;
  int idx_;
  std::string mem_space_;
};

#endif
