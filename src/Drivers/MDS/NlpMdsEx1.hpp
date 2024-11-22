#ifndef HIOP_EXAMPLE_MDS_EX1
#define HIOP_EXAMPLE_MDS_EX1

#include "hiopInterface.hpp"

// this include is not needed in general
// we use hiopMatrixDense in this particular example for convienience
#include "hiopMatrixDenseRowMajor.hpp"
#include "LinAlgFactory.hpp"

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

/* Problem test for the linear algebra of Mixed Dense-Sparse NLPs
 * if 'empty_sp_row' is set to true:
 *  min   sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s + Md y = 0, i=1,...,ns
 *        [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
 *        [-inf] <= [            ] + [e^T] y <= [ 2 ]
 *        [-2  ]    [ x_3        ]   [e^T]      [inf]
 *        x <= 3
 *        s>=0
 *        -4 <=y_1 <=4, the rest of y are free
 * otherwise:
 *  min   sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s + Md y = 0, i=1,...,ns
 *        [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
 *        [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
 *        [-2  ]    [ x_3        ]   [e^T]      [inf]
 *        x <= 3
 *        s>=0
 *        -4 <=y_1 <=4, the rest of y are free
 *
 * The vector 'y' is of dimension nd = ns (can be changed in the constructor)
 * Dense matrices Qd and Md are such that
 * Qd  = two on the diagonal, one on the first offdiagonals, zero elsewhere
 * Md  = minus one everywhere
 * e   = vector of all ones
 *
 * Coding of the problem in MDS HiOp input: order of variables need to be [x,s,y]
 * since [x,s] are the so-called sparse variables and y are the dense variables
 */

class MdsEx1 : public hiop::hiopInterfaceMDS
{
public:
  MdsEx1(int ns_, bool empty_sp_row = false)
      : MdsEx1(ns_, ns_, empty_sp_row)
  {}

  MdsEx1(int ns_, int nd_, bool empty_sp_row = false)
      : ns(ns_),
        sol_x_(NULL),
        sol_zl_(NULL),
        sol_zu_(NULL),
        sol_lambda_(NULL),
        empty_sp_row_(empty_sp_row)
  {
    if(ns < 0) {
      ns = 0;
    } else {
      if(4 * (ns / 4) != ns) {
        ns = 4 * ((4 + ns) / 4);
        printf("[warning] number (%d) of sparse vars is not a multiple ->was altered to %d\n", ns_, ns);
      }
    }

    if(nd_ < 0)
      nd = 0;
    else
      nd = nd_;

    Q = hiop::LinearAlgebraFactory::create_matrix_dense("DEFAULT", nd, nd);
    Q->setToConstant(1e-8);
    Q->addDiagonal(2.);
    double* Qa = Q->local_data();
    for(int i = 1; i < nd - 1; i++) {
      // Qa[i][i+1] = 1.;
      Qa[i * nd + i + 1] = 1.;
      // Qa[i+1][i] = 1.;
      Qa[(i + 1) * nd + i] = 1.;
    }

    Md = hiop::LinearAlgebraFactory::create_matrix_dense("DEFAULT", ns, nd);
    Md->setToConstant(-1.0);

    _buf_y = new double[nd];

    haveIneq = true;
  }

  virtual ~MdsEx1()
  {
    delete[] _buf_y;
    delete Md;
    delete Q;
    delete[] sol_lambda_;
    delete[] sol_zu_;
    delete[] sol_zl_;
    delete[] sol_x_;
  }

  bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = 2 * ns + nd;
    m = ns + 3 * haveIneq;
    return true;
  }

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    // assert(n>=4 && "number of variables should be greater than 4 for this example");
    assert(n == 2 * ns + nd);

    // x
    for(int i = 0; i < ns; ++i) xlow[i] = -1e+20;
    // s
    for(int i = ns; i < 2 * ns; ++i) xlow[i] = 0.;
    // y
    xlow[2 * ns] = -4.;
    for(int i = 2 * ns + 1; i < n; ++i) xlow[i] = -1e+20;

    // x
    for(int i = 0; i < ns; ++i) xupp[i] = 3.;
    // s
    for(int i = ns; i < 2 * ns; ++i) xupp[i] = +1e+20;
    // y
    xupp[2 * ns] = 4.;
    for(int i = 2 * ns + 1; i < n; ++i) xupp[i] = +1e+20;

    for(int i = 0; i < n; ++i) type[i] = hiopNonlinear;
    return true;
  }

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == ns + 3 * haveIneq);
    int i;
    // x+s - Md y = 0, i=1,...,ns
    for(i = 0; i < ns; i++) clow[i] = cupp[i] = 0.;

    if(haveIneq) {
      // [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
      clow[i] = -2;
      cupp[i++] = 2.;
      // [-inf] <= [ x_2        ] + [e^T] y <= [ 2 ]
      clow[i] = -1e+20;
      cupp[i++] = 2.;
      // [-2  ]    [ x_3        ]   [e^T]      [inf]
      clow[i] = -2;
      cupp[i++] = 1e+20;
    }
    assert(i == m);

    for(i = 0; i < m; ++i) type[i] = hiopNonlinear;
    return true;
  }

  bool get_sparse_dense_blocks_info(int& nx_sparse,
                                    int& nx_dense,
                                    int& nnz_sparse_Jace,
                                    int& nnz_sparse_Jaci,
                                    int& nnz_sparse_Hess_Lagr_SS,
                                    int& nnz_sparse_Hess_Lagr_SD)
  {
    nx_sparse = 2 * ns;
    nx_dense = nd;
    nnz_sparse_Jace = 2 * ns;
    if(empty_sp_row_) {
      nnz_sparse_Jaci = (ns == 0 || !haveIneq) ? 0 : 2 + ns;
    } else {
      nnz_sparse_Jaci = (ns == 0 || !haveIneq) ? 0 : 3 + ns;
    }
    nnz_sparse_Hess_Lagr_SS = 2 * ns;
    nnz_sparse_Hess_Lagr_SD = 0.;
    return true;
  }

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    // assert(ns>=4);
    assert(Q->n() == nd);
    assert(Q->m() == nd);
    obj_value = 0.;  // x[0]*(x[0]-1.);
    // sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    for(int i = 0; i < ns; i++) obj_value += x[i] * (x[i] - 1.);
    obj_value *= 0.5;

    double term2 = 0.;
    const double* y = x + 2 * ns;
    Q->timesVec(0.0, _buf_y, 1., y);
    for(int i = 0; i < nd; i++) term2 += _buf_y[i] * y[i];
    obj_value += 0.5 * term2;

    const double* s = x + ns;
    double term3 = 0.;  // s[0]*s[0];
    for(int i = 0; i < ns; i++) term3 += s[i] * s[i];
    obj_value += 0.5 * term3;

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
    const double* s = x + ns;
    const double* y = x + 2 * ns;

    assert(num_cons == ns || num_cons == 3 * haveIneq);

    bool isEq = false;
    for(int irow = 0; irow < num_cons; irow++) {
      const int con_idx = (int)idx_cons[irow];
      if(con_idx < ns) {
        // equalities: x+s - Md y = 0
        cons[con_idx] = x[con_idx] + s[con_idx];
        isEq = true;
      } else if(haveIneq) {
        assert(con_idx < ns + 3);
        // inequality
        const int conineq_idx = con_idx - ns;
        if(conineq_idx == 0) {
          cons[conineq_idx] = x[0];
          for(int i = 0; i < ns; i++) cons[conineq_idx] += s[i];
          for(int i = 0; i < nd; i++) cons[conineq_idx] += y[i];
        } else if(conineq_idx == 1) {
          if(empty_sp_row_) {
            cons[conineq_idx] = 0.0;
          } else {
            cons[conineq_idx] = x[1];
          }
          for(int i = 0; i < nd; i++) cons[conineq_idx] += y[i];
        } else if(conineq_idx == 2) {
          cons[conineq_idx] = x[2];
          for(int i = 0; i < nd; i++) cons[conineq_idx] += y[i];
        } else {
          assert(false);
        }
      }
    }
    if(isEq) {
      Md->timesVec(1.0, cons, 1.0, y);
    }
    return true;
  }

  // sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    // x_i - 0.5
    for(int i = 0; i < ns; i++) gradf[i] = x[i] - 0.5;

    // Qd*y
    const double* y = x + 2 * ns;
    double* gradf_y = gradf + 2 * ns;
    Q->timesVec(0.0, gradf_y, 1., y);

    // s
    const double* s = x + ns;
    double* gradf_s = gradf + ns;
    for(int i = 0; i < ns; i++) gradf_s[i] = s[i];

    return true;
  }

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nsparse,
                             const size_type& ndense,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS,
                             double* JacD)
  {
    assert(num_cons == ns || num_cons == 3 * haveIneq);

    if(iJacS != NULL && jJacS != NULL) {
      int nnzit = 0;
      for(int itrow = 0; itrow < num_cons; itrow++) {
        const int con_idx = (int)idx_cons[itrow];
        if(con_idx < ns && ns > 0) {
          // sparse Jacobian eq w.r.t. x and s
          // x
          iJacS[nnzit] = con_idx;
          jJacS[nnzit] = con_idx;
          nnzit++;

          // s
          iJacS[nnzit] = con_idx;
          jJacS[nnzit] = con_idx + ns;
          nnzit++;

        } else if(haveIneq) {
          // sparse Jacobian ineq w.r.t x and s
          if(con_idx - ns == 0 && ns > 0) {
            // w.r.t x_1
            iJacS[nnzit] = 0;
            jJacS[nnzit] = 0;
            nnzit++;
            // w.r.t s
            for(int i = 0; i < ns; i++) {
              iJacS[nnzit] = 0;
              jJacS[nnzit] = ns + i;
              nnzit++;
            }
          } else {
            if(((con_idx - ns == 1 && !empty_sp_row_) || con_idx - ns == 2) && ns > 0) {
              // w.r.t x_2 or x_3
              iJacS[nnzit] = con_idx - ns;
              jJacS[nnzit] = con_idx - ns;
              nnzit++;
            }
          }
        }
      }
      assert(nnzit == nnzJacS);
    }
    // values for sparse Jacobian if requested by the solver
    if(MJacS != NULL) {
      int nnzit = 0;
      for(int itrow = 0; itrow < num_cons; itrow++) {
        const int con_idx = (int)idx_cons[itrow];
        if(con_idx < ns && ns > 0) {
          // sparse Jacobian EQ w.r.t. x and s
          // x
          MJacS[nnzit] = 1.;
          nnzit++;

          // s
          MJacS[nnzit] = 1.;
          nnzit++;

        } else if(haveIneq) {
          // sparse Jacobian INEQ w.r.t x and s
          if(con_idx - ns == 0 && ns > 0) {
            // w.r.t x_1
            MJacS[nnzit] = 1.;
            nnzit++;
            // w.r.t s
            for(int i = 0; i < ns; i++) {
              MJacS[nnzit] = 1.;
              nnzit++;
            }
          } else {
            if(((con_idx - ns == 1 && !empty_sp_row_) || con_idx - ns == 2) && ns > 0) {
              // w.r.t x_2 or x_3
              MJacS[nnzit] = 1.;
              nnzit++;
            }
          }
        }
      }
      assert(nnzit == nnzJacS);
    }

    // dense Jacobian w.r.t y
    if(JacD != NULL) {
      bool isEq = false;
      for(int itrow = 0; itrow < num_cons; itrow++) {
        const int con_idx = (int)idx_cons[itrow];
        if(con_idx < ns) {
          isEq = true;
          assert(num_cons == ns);
          continue;
        } else if(haveIneq) {
          // do an in place fill-in for the ineq Jacobian corresponding to e^T
          assert(con_idx - ns == 0 || con_idx - ns == 1 || con_idx - ns == 2);
          assert(num_cons == 3);
          for(int i = 0; i < nd; i++) {
            //! JacD[con_idx-ns][i] = 1.;
            JacD[(con_idx - ns) * nd + i] = 1.;
          }
        }
      }
      if(isEq) {
        memcpy(JacD, Md->local_data(), ns * nd * sizeof(double));
      }
    }

    return true;
  }

  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const size_type& nsparse,
                      const size_type& ndense,
                      const size_type& nnzHSS,
                      index_type* iHSS,
                      index_type* jHSS,
                      double* MHSS,
                      double* HDD,
                      size_type& nnzHSD,
                      index_type* iHSD,
                      index_type* jHSD,
                      double* MHSD)
  {
    // Note: lambda is not used since all the constraints are linear and, therefore, do
    // not contribute to the Hessian of the Lagrangian

    assert(nnzHSS == 2 * ns);
    assert(nnzHSD == 0);
    assert(iHSD == NULL);
    assert(jHSD == NULL);
    assert(MHSD == NULL);

    if(iHSS != NULL && jHSS != NULL) {
      for(int i = 0; i < 2 * ns; i++) iHSS[i] = jHSS[i] = i;
    }

    if(MHSS != NULL) {
      for(int i = 0; i < 2 * ns; i++) MHSS[i] = obj_factor;
    }

    if(HDD != NULL) {
      const int nx_dense_squared = nd * nd;
      // memcpy(HDD[0], Q->local_buffer(), nx_dense_squared*sizeof(double));
      const double* Qv = Q->local_data();
      for(int i = 0; i < nx_dense_squared; i++) HDD[i] = obj_factor * Qv[i];
    }
    return true;
  }

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == 2 * ns + nd);
    for(int i = 0; i < global_n; i++) x0[i] = 1.;
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
    slacks_avail = false;

    if(sol_x_ && sol_zl_ && sol_zu_ && sol_lambda_) {
      duals_avail = true;

      memcpy(x0, sol_x_, n * sizeof(double));
      memcpy(z_bndL0, sol_zl_, n * sizeof(double));
      memcpy(z_bndU0, sol_zu_, n * sizeof(double));
      memcpy(lambda0, sol_lambda_, m * sizeof(double));

    } else {
      duals_avail = false;
      return false;
    }
    return true;
  }

  /* The public methods below are not part of hiopInterface. They are a proxy
   * for user's (front end) code to set solutions from a previous solve.
   *
   * Same behaviour can be achieved internally (in this class ) if desired by
   * overriding @solution_callback and @get_starting_point
   */
  void set_solution_primal(const double* x)
  {
    int n = 2 * ns + nd;
    if(NULL == sol_x_) {
      sol_x_ = new double[n];
    }
    memcpy(sol_x_, x, n * sizeof(double));
  }
  void set_solution_duals(const double* zl, const double* zu, const double* lambda)
  {
    int m = ns + 3 * haveIneq;
    int n = 2 * ns + nd;
    if(NULL == sol_zl_) {
      sol_zl_ = new double[n];
    }
    memcpy(sol_zl_, zl, n * sizeof(double));

    if(NULL == sol_zu_) {
      sol_zu_ = new double[n];
    }
    memcpy(sol_zu_, zu, n * sizeof(double));

    if(NULL == sol_lambda_) {
      sol_lambda_ = new double[m];
    }
    memcpy(sol_lambda_, lambda, m * sizeof(double));
  }

  /** pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process */
  virtual bool get_MPI_comm(MPI_Comm& comm_out)
  {
    comm_out = MPI_COMM_SELF;
    return true;
  }

protected:
  int ns, nd;
  hiop::hiopMatrixDense *Q, *Md;
  double* _buf_y;
  bool haveIneq;

  /* Internal buffers to store primal-dual solution */
  double* sol_x_;
  double* sol_zl_;
  double* sol_zu_;
  double* sol_lambda_;

  /* indicate if problem has empty row in constraint Jacobian */
  bool empty_sp_row_;
};

class MdsEx1OneCallCons : public MdsEx1
{
public:
  MdsEx1OneCallCons(int ns_in, bool empty_sp_row = false)
      : MdsEx1(ns_in, empty_sp_row)
  {}

  MdsEx1OneCallCons(int ns_in, int nd_in, bool empty_sp_row = false)
      : MdsEx1(ns_in, nd_in, empty_sp_row)
  {}

  virtual ~MdsEx1OneCallCons() {}

  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons)
  {
    // return false so that HiOp will rely on the one-call constraint evaluator defined below
    return false;
  }
  /** all constraints evaluated in here */
  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    assert(3 * haveIneq + ns == m);
    const double* s = x + ns;
    const double* y = x + 2 * ns;

    for(int con_idx = 0; con_idx < m; ++con_idx) {
      if(con_idx < ns) {
        // equalities
        cons[con_idx] = x[con_idx] + s[con_idx];
      } else if(haveIneq) {
        // inequalties
        assert(con_idx < ns + 3);
        if(con_idx == ns) {
          cons[con_idx] = x[0];
          for(int i = 0; i < ns; i++) cons[con_idx] += s[i];
          for(int i = 0; i < nd; i++) cons[con_idx] += y[i];

        } else if(con_idx == ns + 1) {
          if(empty_sp_row_) {
            cons[con_idx] = 0.0;
          } else {
            cons[con_idx] = x[1];
          }
          for(int i = 0; i < nd; i++) cons[con_idx] += y[i];
        } else if(con_idx == ns + 2) {
          cons[con_idx] = x[2];
          for(int i = 0; i < nd; i++) cons[con_idx] += y[i];
        } else {
          assert(false);
        }
      }
    }

    // apply Md to y and add the result to equality part of 'cons'

    // we know that equalities are the first ns constraints so this should work
    Md->timesVec(1.0, cons, 1.0, y);
    return true;
  }

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const size_type& nsparse,
                             const size_type& ndense,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS,
                             double* JacD)
  {
    return false;  // so that HiOp will call the one-call full-Jacob function below
  }

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const size_type& nsparse,
                             const size_type& ndense,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS,
                             double* JacD)
  {
    assert(m == ns + 3 * haveIneq);

    if(iJacS != NULL && jJacS != NULL) {
      int nnzit = 0;
      for(int con_idx = 0; con_idx < ns; ++con_idx) {
        // sparse Jacobian eq w.r.t. x and s
        // x
        iJacS[nnzit] = con_idx;
        jJacS[nnzit] = con_idx;
        nnzit++;

        // s
        iJacS[nnzit] = con_idx;
        jJacS[nnzit] = con_idx + ns;
        nnzit++;
      }
      if(haveIneq && ns > 0) {
        for(int con_idx = ns; con_idx < m; ++con_idx) {
          // sparse Jacobian ineq w.r.t x and s
          if(con_idx == ns) {
            // w.r.t x_1
            iJacS[nnzit] = con_idx;
            jJacS[nnzit] = 0;
            nnzit++;
            // w.r.t s
            for(int i = 0; i < ns; i++) {
              iJacS[nnzit] = con_idx;
              jJacS[nnzit] = ns + i;
              nnzit++;
            }
          } else {
            if((con_idx - ns == 1 && !empty_sp_row_) || con_idx - ns == 2) {
              // w.r.t x_2 or x_3
              iJacS[nnzit] = con_idx;
              jJacS[nnzit] = con_idx - ns;
              nnzit++;
            }
          }
        }
      }
      assert(nnzit == nnzJacS);
    }
    // values for sparse Jacobian if requested by the solver
    if(MJacS != NULL) {
      int nnzit = 0;
      for(int con_idx = 0; con_idx < ns; ++con_idx) {
        // sparse Jacobian EQ w.r.t. x and s
        // x
        MJacS[nnzit] = 1.;
        nnzit++;

        // s
        MJacS[nnzit] = 1.;
        nnzit++;
      }

      if(haveIneq && ns > 0) {
        for(int con_idx = ns; con_idx < m; ++con_idx) {
          // sparse Jacobian INEQ w.r.t x and s
          if(con_idx - ns == 0) {
            // w.r.t x_1
            MJacS[nnzit] = 1.;
            nnzit++;
            // w.r.t s
            for(int i = 0; i < ns; i++) {
              MJacS[nnzit] = 1.;
              nnzit++;
            }
          } else {
            if((con_idx - ns == 1 && !empty_sp_row_) || con_idx - ns == 2) {
              // w.r.t x_2 or x_3
              MJacS[nnzit] = 1.;
              nnzit++;
            }
          }
        }
      }
      assert(nnzit == nnzJacS);
    }

    // dense Jacobian w.r.t y
    if(JacD != NULL) {
      // just copy the dense Jacobian corresponding to equalities
      memcpy(JacD, Md->local_data(), ns * nd * sizeof(double));

      if(haveIneq) {
        assert(ns + 3 == m);
        // do an in place fill-in for the ineq Jacobian corresponding to e^T
        for(int i = 0; i < 3 * nd; ++i) JacD[ns * nd + i] = 1.;
      }
    }
    return true;
  }
};

#endif
