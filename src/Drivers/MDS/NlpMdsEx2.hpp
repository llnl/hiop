#ifndef HIOP_EXAMPLE_MDS_EX2
#define HIOP_EXAMPLE_MDS_EX2

#include "hiopInterface.hpp"

// this include is not needed in general
// we use hiopMatrixDense in this particular example for convienience
#include "hiopMatrixDense.hpp"
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

/** Nonlinear *highly nonconvex* and *rank deficient* problem test for the Filter IPM
 * Newton of HiOp. It uses a mixed Dense-Sparse NLP formulation. The problem is based
 * on MDS Ex1.
 *
 *  min   - sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} - 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s + Md y = 0, i=1,...,ns
 *        [ -2 ]    [ x_1 + e^T s]   [e^T]          [ 2 ]
 *        [-inf] <= [ x_2        ] + [e^T] y     <= [ 2 ]
 *        [ -2 ]    [ x_3        ]   [e^T]          [inf]
 *        -10 <= x <= 3
 *        s>=0
 *        -4 <=y <=4, the rest of y are free
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= [ x_1 + e^T s + x_2 + 2e^T y] <= [ 4 ]   (rnkdef-con1.1)
 *        [ -4 ] <= [ x_1 + e^T s + x_3 + 2e^T y] <= [inf]   (rnkdef-con1.2)
 *        x+s + Md y = 0                                     (rnkdef-con2)
 *
 * The vector 'y' is of dimension nd = ns (can be changed on construction)
 * Dense matrices Qd and Md are such that
 * Qd  = two on the diagonal, one on the first offdiagonals, zero elsewhere
 * Md  = minus one everywhere, matrix ns x nd
 * e   = vector of all ones
 *
 * Coding of the problem in MDS HiOp input: order of variables need to be [x,s,y]
 * since [x,s] are the so-called sparse variables and y are the dense variables
 */
class MdsEx2 : public hiop::hiopInterfaceMDS
{
public:
  MdsEx2(int ns)
      : MdsEx2(ns, ns, true, true, true)
  {}

  MdsEx2(int ns, int nd, bool convex_obj, bool rankdefic_Jac_eq, bool rankdefic_Jac_ineq)
      : ns_(ns),
        rankdefic_eq_(rankdefic_Jac_eq),
        rankdefic_ineq_(rankdefic_Jac_ineq),
        convex_obj_(convex_obj)
  {
    if(ns_ < 0) {
      ns_ = 0;
    } else {
      if(4 * (ns_ / 4) != ns_) {
        ns_ = 4 * ((4 + ns_) / 4);
        printf("[warning] number (%d) of sparse vars is not a multiple of n; was altered to %d\n", ns, ns_);
      }
    }

    if(nd < 0)
      nd_ = 0;
    else
      nd_ = nd;

    Q_ = hiop::LinearAlgebraFactory::create_matrix_dense("DEFAULT", nd_, nd_);
    Q_->setToConstant(0.);
    Q_->addDiagonal(2. * (2 * convex_obj_ - 1));  //-2 or 2
    double* Qa = Q_->local_data();
    for(int i = 1; i < nd - 1; i++) {
      // Qa[i][i+1] = 1.;
      Qa[i * nd_ + i + 1] = 1.;
      // Qa[i+1][i] = 1.;
      Qa[(i + 1) * nd_ + i] = 1.;
    }

    Md_ = hiop::LinearAlgebraFactory::create_matrix_dense("DEFAULT", ns_, nd_);
    Md_->setToConstant(-1.0);

    _buf_y_ = new double[nd_];
  }

  virtual ~MdsEx2()
  {
    delete[] _buf_y_;
    delete Md_;
    delete Q_;
  }

  bool get_prob_sizes(size_type& n, size_type& m)
  {
    n = 2 * ns_ + nd_;
    m = ns_ + 3 + 2 * rankdefic_ineq_ + ns_ * rankdefic_eq_;
    return true;
  }

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    // assert(n>=4 && "number of variables should be greater than 4 for this example");
    assert(n == 2 * ns_ + nd_);

    // x
    for(int i = 0; i < ns_; ++i) xlow[i] = -10.;
    // s
    for(int i = ns_; i < 2 * ns_; ++i) xlow[i] = 0.;
    // y
    for(int i = 2 * ns_; i < n; ++i) xlow[i] = -4.;

    // x
    for(int i = 0; i < ns_; ++i) xupp[i] = 3.;
    // s
    for(int i = ns_; i < 2 * ns_; ++i) xupp[i] = +1e+20;
    // y
    xupp[2 * ns_] = 4.;
    for(int i = 2 * ns_ + 1; i < n; ++i) xupp[i] = 4.;

    for(int i = 0; i < n; ++i) type[i] = hiopNonlinear;
    return true;
  }

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == ns_ + 3 + 2 * rankdefic_ineq_ + ns_ * rankdefic_eq_);
    int i;
    // x+s - Md y = 0, i=1,...,ns
    for(i = 0; i < ns_; i++) clow[i] = cupp[i] = 0.;

    {
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
    if(rankdefic_ineq_) {
      // [-inf] <= [ x_1 + e^T s + x_2 + 2e^T y] <= [ 4 ]
      clow[i] = -1e+20;
      cupp[i++] = 4.;
      // [ -4 ] <= [ x_1 + e^T s + x_3 + 2e^T y] <= [inf]
      clow[i] = -4;
      cupp[i++] = 1e+20;
    }

    if(rankdefic_eq_) {
      for(; i < m;) {
        assert(i >= ns_ + 3 + 2 * rankdefic_ineq_);
        clow[i] = 0.;
        cupp[i++] = 0.;
      }
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
    nx_sparse = 2 * ns_;
    nx_dense = nd_;
    nnz_sparse_Jace = 2 * ns_ + rankdefic_eq_ * 2 * ns_;
    nnz_sparse_Jaci = ns_ == 0 ? 0 : 3 + ns_ + rankdefic_ineq_ * 2 * (2 + ns_);
    nnz_sparse_Hess_Lagr_SS = 2 * ns_;
    nnz_sparse_Hess_Lagr_SD = 0.;
    return true;
  }

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    // assert(ns>=4);
    assert(Q_->n() == nd_);
    assert(Q_->m() == nd_);
    obj_value = 0.;  // x[0]*(x[0]-1.);
    // sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
    for(int i = 0; i < ns_; i++) obj_value += x[i] * (x[i] - 1.);
    obj_value *= 0.5;
    obj_value *= (2 * convex_obj_ - 1);  // switch sign if non-convex problem is desired

    double term2 = 0.;
    const double* y = x + 2 * ns_;
    Q_->timesVec(0.0, _buf_y_, 1., y);
    for(int i = 0; i < nd_; i++) term2 += _buf_y_[i] * y[i];
    obj_value += 0.5 * term2;

    const double* s = x + ns_;
    double term3 = 0.;  // s[0]*s[0];
    for(int i = 0; i < ns_; i++) term3 += s[i] * s[i];
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
    // return false so that HiOp will rely on the on-call constraint evaluator defined below
    return false;
  }
  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    const double* s = x + ns_;
    const double* y = x + 2 * ns_;

    assert(n == 2 * ns_ + nd_);
    assert(m == ns_ + 3 + 2 * rankdefic_ineq_ + ns_ * rankdefic_eq_);

    int con_idx = 0;

    // equalities: x+s - Md y = 0
    for(; con_idx < ns_; con_idx++) {
      cons[con_idx] = x[con_idx] + s[con_idx];
    }
    Md_->timesVec(1.0, cons, 1.0, y);

    //[ -2 ]  <=  [ x_1 + e^T s]   [e^T] y     <=     [ 2 ]
    cons[con_idx] = x[0];
    for(int i = 0; i < ns_; i++) cons[con_idx] += s[i];
    for(int i = 0; i < nd_; i++) cons[con_idx] += y[i];
    con_idx++;

    //[-inf] <= [ x_2        ] + [e^T] y     <= [ 2 ]
    cons[con_idx] = x[1];
    for(int i = 0; i < nd_; i++) cons[con_idx] += y[i];
    con_idx++;

    //[ -2 ] <=   [ x_3        ]   [e^T] y     <= [inf]
    cons[con_idx] = x[2];
    for(int i = 0; i < nd_; i++) cons[con_idx] += y[i];
    con_idx++;

    assert(con_idx == ns_ + 3);

    if(rankdefic_ineq_) {
      // [-inf] <= [ x_1 + e^T s + x_2 + 2e^T y] <= [ 4 ]
      cons[con_idx] = x[0] + x[1];
      for(int i = 0; i < ns_; i++) cons[con_idx] += s[i];
      for(int i = 0; i < nd_; i++) cons[con_idx] += 2 * y[i];
      con_idx++;

      // [ -4 ] <= [ x_1 + e^T s + x_3 + 2e^T y] <= [inf]
      cons[con_idx] = x[0] + x[2];
      for(int i = 0; i < ns_; i++) cons[con_idx] += s[i];
      for(int i = 0; i < nd_; i++) cons[con_idx] += 2 * y[i];
      con_idx++;
    }

    if(rankdefic_eq_) {
      for(int i = 0; i < ns_; i++) {
        cons[con_idx++] = x[i] + s[i];
      }
      Md_->timesVec(1.0, cons + (m - ns_), 1.0, y);
    }
    assert(m == con_idx);

    return true;
  }

  // sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    //! assert(ns>=4); assert(Q->n()==ns/4); assert(Q->m()==ns/4);
    // x_i - 0.5
    for(int i = 0; i < ns_; i++) gradf[i] = (x[i] - 0.5) * (2 * convex_obj_ - 1);

    // Qd*y
    const double* y = x + 2 * ns_;
    double* gradf_y = gradf + 2 * ns_;
    Q_->timesVec(0.0, gradf_y, 1., y);

    // s
    const double* s = x + ns_;
    double* gradf_s = gradf + ns_;
    for(int i = 0; i < ns_; i++) gradf_s[i] = s[i];

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
    // return false so that HiOp will rely on the on-call constraint evaluator defined below
    return false;
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
    assert(m == ns_ + 3 + 2 * rankdefic_ineq_ + ns_ * rankdefic_eq_);
    assert(nnzJacS == 2 * ns_ + rankdefic_eq_ * 2 * ns_ + (ns_ == 0) ? 0 : 3 + ns_ + rankdefic_ineq_ * 2 * (2 + ns_));
    // sparse part first
    if(iJacS != NULL && jJacS != NULL) {
      int nnzit = 0;
      int con_idx = 0;
      for(; con_idx < ns_; con_idx++) {
        // sparse Jacobian eq w.r.t. x and s
        // x
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = con_idx;

        // s
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = con_idx + ns_;
      }
      assert(con_idx == ns_);
      assert(nnzit == 2 * ns_);

      // sparse Jacobian ineq w.r.t x and s
      if(ns_ > 0) {
        // w.r.t x_1
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 0;
        // w.r.t s
        for(int i = 0; i < ns_; i++) {
          iJacS[nnzit] = con_idx;
          jJacS[nnzit++] = ns_ + i;
        }
        con_idx++;

        // w.r.t x_2
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 1;
        con_idx++;

        // w.r.t x_3
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 2;
        con_idx++;
      }  // end of if(ns>0)
      assert(nnzit == 2 * ns_ + 3 * (ns_ > 0) + ns_);

      if(rankdefic_ineq_ && ns_ > 0) {
        // [-inf] <= [ x_1 + e^T s + x_2 + 2e^T y] <= [ 4 ]
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 0;  // x1

        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 1;  // x2

        for(int i = 0; i < ns_; i++) {
          iJacS[nnzit] = con_idx;
          jJacS[nnzit++] = ns_ + i;  // s
        }
        con_idx++;

        // [ -4 ] <= [ x_1 + e^T s + x_3 + 2e^T y] <= [inf]
        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 0;  // x1

        iJacS[nnzit] = con_idx;
        jJacS[nnzit++] = 2;  // x3

        for(int i = 0; i < ns_; i++) {
          iJacS[nnzit] = con_idx;
          jJacS[nnzit++] = ns_ + i;  // s
        }
        con_idx++;
      }
      assert(nnzit == 2 * ns_ + 3 * (ns_ > 0) + ns_ + rankdefic_ineq_ * 2 * (2 + ns_) * (ns_ > 0));

      if(rankdefic_eq_) {
        // x+s - Md y = 0, i=1,...,ns
        for(int i = 0; i < ns_; i++) {
          iJacS[nnzit] = con_idx;
          jJacS[nnzit++] = i;  // x
          iJacS[nnzit] = con_idx;
          jJacS[nnzit++] = i + ns_;  // s
          con_idx++;
        }
      }
      assert(nnzit == nnzJacS);
    }

    // values for sparse Jacobian if requested by the solver
    if(MJacS != NULL) {
      int nnzit = 0;
      int con_idx = 0;

      // sparse Jacobian EQ w.r.t. x and s
      for(int i = 0; i < ns_; i++) {
        MJacS[nnzit++] = 1.;  // x
        MJacS[nnzit++] = 1.;  // s
        con_idx++;
      }
      if(ns_ > 0) {
        // sparse Jacobian INEQ w.r.t x and s

        // w.r.t x_1
        MJacS[nnzit++] = 1.;
        // w.r.t s
        for(int i = 0; i < ns_; i++) {
          MJacS[nnzit++] = 1.;
        }
        con_idx++;

        // w.r.t x_2
        MJacS[nnzit++] = 1.;
        con_idx++;

        // w.r.t. x_3
        MJacS[nnzit++] = 1.;
        con_idx++;
      }
      assert(nnzit == 2 * ns_ + 3 * (ns_ > 0) + ns_);
      assert(con_idx == ns_ + 3 * (ns_ > 0));

      if(rankdefic_ineq_ && ns_ > 0) {
        // [-inf] <= [ x_1 + e^T s + x_2 + 2e^T y] <= [ 4 ]
        MJacS[nnzit++] = 1.;  // x1
        MJacS[nnzit++] = 1.;  // x2
        for(int i = 0; i < ns_; i++) {
          MJacS[nnzit++] = 1.;  // s
        }
        con_idx++;

        // [ -4 ] <= [ x_1 + e^T s + x_3 + 2e^T y] <= [inf]
        MJacS[nnzit++] = 1.;  // x1
        MJacS[nnzit++] = 1.;  // x3
        for(int i = 0; i < ns_; i++) {
          MJacS[nnzit++] = 1.;  // s
        }
        con_idx++;
      }
      assert(nnzit == 2 * ns_ + 3 * (ns_ > 0) + ns_ + rankdefic_ineq_ * 2 * (2 + ns_) * (ns_ > 0));

      // x+s - Md y = 0, i=1,...,ns
      if(rankdefic_eq_) {
        for(int i = 0; i < ns_; i++) {
          MJacS[nnzit++] = 1.;  // x
          MJacS[nnzit++] = 1.;  // s
          con_idx++;
        }
      }
      assert(nnzit == nnzJacS);
    }

    //
    // dense Jacobian w.r.t y
    //
    if(JacD != NULL) {
      // eq
      memcpy(JacD, Md_->local_data(), ns_ * nd_ * sizeof(double));

      // ineq
      for(int i = 0; i < 3 * nd_; i++) {
        //! JacD[ns_][i] = 1.;
        JacD[ns_ * nd_ + i] = 1.;
      }

      int con_idx = ns_ + 3;
      if(rankdefic_ineq_) {
        for(int i = 0; i < 2 * nd_; i++) {
          //! JacD[con_idx][i] = 2.;
          JacD[con_idx * nd_ + i] = 2.;
        }
        con_idx += 2;
      }

      if(rankdefic_eq_) {
        memcpy(JacD + con_idx * nd_, Md_->local_data(), ns_ * nd_ * sizeof(double));
        con_idx += ns_;
      }

      assert(con_idx == m);
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

    assert(nnzHSS == 2 * ns_);
    assert(nnzHSD == 0);
    assert(iHSD == NULL);
    assert(jHSD == NULL);
    assert(MHSD == NULL);

    if(iHSS != NULL && jHSS != NULL) {
      for(int i = 0; i < 2 * ns_; i++) iHSS[i] = jHSS[i] = i;
    }

    if(MHSS != NULL) {
      for(int i = 0; i < ns_; i++) MHSS[i] = obj_factor * (2 * convex_obj_ - 1);
      for(int i = ns_; i < 2 * ns_; i++) MHSS[i] = obj_factor;
    }

    if(HDD != NULL) {
      const int nx_dense_squared = nd_ * nd_;
      // memcpy(HDD[0], Q->local_buffer(), nx_dense_squared*sizeof(double));
      const double* Qv = Q_->local_data();
      for(int i = 0; i < nx_dense_squared; i++) HDD[i] = obj_factor * Qv[i];
    }
    return true;
  }

  bool get_starting_point(const size_type& global_n, double* x0)
  {
    assert(global_n == 2 * ns_ + nd_);
    for(int i = 0; i < global_n; i++) x0[i] = 10.;
    return true;
  }

  /** pass the COMM_SELF communicator since this example is only intended to run inside 1 MPI process */
  virtual bool get_MPI_comm(MPI_Comm& comm_out)
  {
    comm_out = MPI_COMM_SELF;
    return true;
  }

protected:
  int ns_, nd_;
  hiop::hiopMatrixDense *Q_, *Md_;
  double* _buf_y_;
  bool rankdefic_eq_, rankdefic_ineq_;
  bool convex_obj_;
};

#endif
