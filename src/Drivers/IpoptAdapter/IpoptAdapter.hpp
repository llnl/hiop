/* Adapter that converts  HiOP interface to IPOPT TNLP interface.
 * Can be easily used with existing IPOPT drivers (to solve problems in HiOP's input), for example
 *    DenseConsEx2 hiop_interface(7);
 *    // create a new instance of your nlp by using the adapter offered by hiOP.
 *    SmartPtr<TNLP> mynlp = new hiop2IpoptTNLP(&hiop_interface);
 *    // from now on everything is compatible with Ipopt
 *
 * An example IPOPT driver that solve HiOP NLPs using Ipopt is available upon request.
 *
 */

/*
 * Author: Cosmin G. Petra, LLNL, 2016. Updated 2019.
 */

#ifndef HIOP_IPOPT_ADAPTER
#define HIOP_IPOPT_ADAPTER

#include "IpTNLP.hpp"
#include "hiopInterface.hpp"

#include "hiopMatrixDenseRowMajor.hpp"

#include <cassert>
#include <cstring>
using namespace Ipopt;

namespace hiop
{

/* Addapts HiOp DenseConstraints interface to Ipopt TNLP interface */
// TO DO: call eval_cons (and Jacob) separately for Eq and Ineq as per documentation of these methods
class hiopDenseCons2IpoptTNLP : public TNLP
{
public:
  hiopDenseCons2IpoptTNLP(hiopInterfaceDenseConstraints* hiopNLP_)
      : hiopNLP(hiopNLP_) {};
  virtual ~hiopDenseCons2IpoptTNLP() {};

  /* Overloads from TNLP */
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    size_type nvars, ncons;
    if(false == hiopNLP->get_prob_sizes(nvars, ncons)) return false;
    n = (int)nvars;
    m = (int)ncons;
    nnz_jac_g = n * m;
    nnz_h_lag = 0;
    index_style = TNLP::C_STYLE;
    return true;
  }

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u)
  {
    bool bSuccess = true;
    size_type nll = n, mll = m;
    hiopInterfaceBase::NonlinearityType* types = new hiopInterfaceBase::NonlinearityType[n];
    bSuccess = hiopNLP->get_vars_info(nll, x_l, x_u, types);
    delete[] types;

    if(bSuccess) {
      types = new hiopInterfaceBase::NonlinearityType[m];
      bSuccess = hiopNLP->get_cons_info(mll, g_l, g_u, types);
      delete[] types;
    }
    return bSuccess;
  }

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n,
                                  bool init_x,
                                  Number* x,
                                  bool init_z,
                                  Number* z_L,
                                  Number* z_U,
                                  Index m,
                                  bool init_lambda,
                                  Number* lambda)
  {
    assert(false == init_z);
    assert(false == init_lambda);
    size_type nll = n;
    return hiopNLP->get_starting_point(nll, x);
  }

  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
  {
    size_type nll = n;
    return hiopNLP->eval_f(nll, x, new_x, obj_value);
  }

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
  {
    size_type nll = n;
    return hiopNLP->eval_grad_f(nll, x, new_x, grad_f);
  }

  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
  {
    size_type nll = n, mll = m;
    index_type* idx_cons = new index_type[m];
    for(int i = 0; i < m; i++) idx_cons[i] = i;
    bool bret = hiopNLP->eval_cons(nll, mll, mll, idx_cons, x, new_x, g);
    delete[] idx_cons;
    return bret;
  }

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool
  eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values)
  {
    bool bret = true;
    size_type nll = n, mll = m, onell = 1;
    double* constraint = new double[n];
    size_type nz = 0;
    for(size_type i = 0; i < m && bret; i++) {
      if(values) {
        bret = hiopNLP->eval_Jac_cons(nll, mll, onell, &i, x, new_x, constraint);
        if(!bret) break;

        memcpy(values + i * n, constraint, ((size_t)n) * sizeof(double));

      } else {  // this is only for iRow and jCol

        for(size_type j = 0; j < n; j++) {
          iRow[nz] = (int)i;
          jCol[nz] = (int)j;
          nz++;
        }
      }
    }
    delete[] constraint;
    return bret;
  }

  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n,
                      const Number* x,
                      bool new_x,
                      Number obj_factor,
                      Index m,
                      const Number* lambda,
                      bool new_lambda,
                      Index nele_hess,
                      Index* iRow,
                      Index* jCol,
                      Number* values)
  {
    return false;
  }

  /* This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Index n,
                                 const Number* x,
                                 const Number* z_L,
                                 const Number* z_U,
                                 Index m,
                                 const Number* g,
                                 const Number* lambda,
                                 Number obj_value,
                                 const IpoptData* ip_data,
                                 IpoptCalculatedQuantities* ip_cq) {};

private:
  hiopInterfaceDenseConstraints* hiopNLP;

  /* Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *
   */
  hiopDenseCons2IpoptTNLP() {};
  hiopDenseCons2IpoptTNLP(const hiopDenseCons2IpoptTNLP&) {};
  hiopDenseCons2IpoptTNLP& operator=(const hiopDenseCons2IpoptTNLP&);
  //@}
};

// we use hiopMatrixDenseRowMajor for the MDS adapter to enable double indexing,
//[i][j] on the double**
// buffers hiopInterfaceMDS implementations expect
#include "hiopMatrixDenseRowMajor.hpp"

/* Adapter from MixedDenseSparse NLP formulation to Ipopt's general TNLP */
class hiopMDS2IpoptTNLP : public TNLP
{
public:
  hiopMDS2IpoptTNLP(hiopInterfaceMDS* hiopNLP_)
      : hiopNLP(hiopNLP_)
  {
    nx_sparse = nx_dense = nnz_sparse_Jaceq = nnz_sparse_Jacineq = 0;
    nnz_sparse_Hess_Lagr_SS = nnz_sparse_Hess_Lagr_SD = 0;
    cons_eq_idxs = cons_ineq_idxs = NULL;
    JacDeq = JacDineq = HessDL = JacDeqineq = NULL;
    onecall_Jac_detected_ = false;
  };
  virtual ~hiopMDS2IpoptTNLP()
  {
    delete[] cons_eq_idxs;
    delete[] cons_ineq_idxs;
    delete JacDeq;
    delete JacDineq;
    delete HessDL;
    delete JacDeqineq;
  };

  /* Overloads from TNLP */
  /** Method to return some info about the nlp */
  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    size_type nvars, ncons;
    if(false == hiopNLP->get_prob_sizes(nvars, ncons)) return false;

    if(false == hiopNLP->get_sparse_dense_blocks_info(nx_sparse,
                                                      nx_dense,
                                                      nnz_sparse_Jaceq,
                                                      nnz_sparse_Jacineq,
                                                      nnz_sparse_Hess_Lagr_SS,
                                                      nnz_sparse_Hess_Lagr_SD)) {
      return false;
    }

    nnz_jac_g = nnz_sparse_Jaceq + nnz_sparse_Jacineq;
    // also put the dense part
    nnz_jac_g += (int)ncons * nx_dense;

    n = (int)nvars;
    m = (int)ncons;
    nnz_h_lag = nnz_sparse_Hess_Lagr_SS;
    assert(nnz_sparse_Hess_Lagr_SD == 0);
    // plus the dense part
    nnz_h_lag += nx_dense * (nx_dense + 1) / 2;

    index_style = TNLP::C_STYLE;
    return true;
  }

  /** Method to return the bounds for my problem */
  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u)
  {
    bool bSuccess = true;
    size_type nll = n, mll = m;
    hiopInterfaceBase::NonlinearityType* types = new hiopInterfaceBase::NonlinearityType[n];
    bSuccess = hiopNLP->get_vars_info(nll, x_l, x_u, types);
    delete[] types;

    if(bSuccess) {
      types = new hiopInterfaceBase::NonlinearityType[m];
      bSuccess = hiopNLP->get_cons_info(mll, g_l, g_u, types);
      delete[] types;
    }

    n_eq = n_ineq = 0;
    for(int it = 0; it < m; it++) {
      if(g_l[it] == g_u[it])
        n_eq++;
      else
        n_ineq++;
    }
    assert(n_eq + n_ineq == m);
    if(cons_eq_idxs != NULL) delete[] cons_eq_idxs;
    if(cons_ineq_idxs != NULL) delete[] cons_ineq_idxs;

    cons_eq_idxs = new index_type[n_eq];
    cons_ineq_idxs = new index_type[n_ineq];

    int it_eq = 0, it_ineq = 0;
    for(int it = 0; it < m; it++) {
      if(g_l[it] == g_u[it])
        cons_eq_idxs[it_eq++] = it;
      else
        cons_ineq_idxs[it_ineq++] = it;
    }
    assert(it_eq == n_eq);
    assert(it_ineq == n_ineq);

    return bSuccess;
  }

  /** Method to return the starting point for the algorithm */
  bool get_starting_point(Index n,
                          bool init_x,
                          Number* x,
                          bool init_z,
                          Number* z_L,
                          Number* z_U,
                          Index m,
                          bool init_lambda,
                          Number* lambda)
  {
    assert(false == init_z && "primal-dual restart not supported by the addapter");
    assert(false == init_lambda && "primal-dual restart not supported by the addapter");
    size_type nll = n;
    return hiopNLP->get_starting_point(nll, x);
  }

  /** Method to return the objective value */
  bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
  {
    size_type nll = n;
    return hiopNLP->eval_f(nll, x, new_x, obj_value);
  }

  /** Method to return the gradient of the objective */
  bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
  {
    size_type nll = n;
    return hiopNLP->eval_grad_f(nll, x, new_x, grad_f);
  }

  /** Method to return the constraint residuals */
  // HiOp calls Eq and Ineq separately -> the interface expects that so we have to
  // mimic it
  bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
  {
    size_type nll = n, mll = m;
    bool bret = false;
    bool eq_call_failed = false;
    bool try_onecall_Jac = false;
    {
      double g_eq[n_eq];
      size_type num_cons = n_eq;
      bret = hiopNLP->eval_cons(nll, mll, num_cons, cons_eq_idxs, x, new_x, g_eq);
      if(bret) {
        for(int i = 0; i < n_eq; i++) g[cons_eq_idxs[i]] = g_eq[i];
      } else {
        eq_call_failed = true;
      }
    }
    {
      double g_ineq[n_ineq];
      size_type num_cons = n_ineq;
      bret = hiopNLP->eval_cons(nll, mll, num_cons, cons_ineq_idxs, x, new_x, g_ineq);
      if(bret) {
        for(int i = 0; i < n_ineq; i++) g[cons_ineq_idxs[i]] = g_ineq[i];
      } else {
        if(!eq_call_failed)
          return false;
        else
          try_onecall_Jac = true;
      }
    }

    if(try_onecall_Jac) {
      bret = hiopNLP->eval_cons(nll, mll, x, new_x, g);
      // for(int i=0; i<mll; i++) printf("%.6e ", g[i]); printf("\n");
    }

    return bret;
  }

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values)
  {
    bool bret = true;
    size_type nll = n, mll = m;
    bool eq_call_failed = false;
    bool try_onecall_Jac = false;
    if(values == NULL) {
      int nnzit = 0;
      // Sparse Jac for Eq
      {
        size_type num_cons = n_eq;
        bret = hiopNLP->eval_Jac_cons(nll,
                                      mll,
                                      num_cons,
                                      cons_eq_idxs,
                                      x,
                                      new_x,
                                      nx_sparse,
                                      nx_dense,
                                      nnz_sparse_Jaceq,
                                      iRow,
                                      jCol,
                                      NULL,
                                      NULL);
        if(bret) {
          nnzit += nnz_sparse_Jaceq;
          for(int i = 0; i < n_eq; i++) {
            for(int j = 0; j < nx_dense; j++) {
              assert(nnzit < nele_jac);
              iRow[nnzit] = (int)cons_eq_idxs[i];
              jCol[nnzit] = j + nx_sparse;
              nnzit++;
            }
          }
        } else {
          eq_call_failed = true;
        }
      }

      // Sparse Jac for Ineq
      {
        size_type num_cons = n_ineq;
        bret = hiopNLP->eval_Jac_cons(nll,
                                      mll,
                                      num_cons,
                                      cons_ineq_idxs,
                                      x,
                                      new_x,
                                      nx_sparse,
                                      nx_dense,
                                      nnz_sparse_Jacineq,
                                      iRow + nnzit,
                                      jCol + nnzit,
                                      NULL,
                                      NULL);
        if(bret) {
          // in-place shift of iRow and jCol for Jacineq
          for(int it = nnzit; it < nnzit + nnz_sparse_Jacineq; it++) iRow[it] += n_eq;

          nnzit += nnz_sparse_Jacineq;
          assert(nnzit <= nele_jac);

          for(int i = 0; i < n_ineq; i++) {
            for(int j = 0; j < nx_dense; j++) {
              assert(nnzit < nele_jac);
              iRow[nnzit] = (int)cons_ineq_idxs[i];
              jCol[nnzit] = j + nx_sparse;
              nnzit++;
            }
          }
        } else {
          if(eq_call_failed)
            try_onecall_Jac = true;
          else
            return false;
        }
      }
      assert(try_onecall_Jac || nnzit == nele_jac);

      if(try_onecall_Jac) {
        bret = hiopNLP->eval_Jac_cons(nll,
                                      mll,
                                      x,
                                      new_x,
                                      nx_sparse,
                                      nx_dense,
                                      nnz_sparse_Jaceq + nnz_sparse_Jacineq,
                                      iRow,
                                      jCol,
                                      values,
                                      NULL);
        if(!bret) return false;

        nnzit = nnz_sparse_Jaceq + nnz_sparse_Jacineq;
        // put the dense part of the MDS in the Ipopt sparse Jac matrix
        for(int i = 0; i < m; i++) {
          for(int j = 0; j < nx_dense; j++) {
            assert(nnzit < nele_jac);
            iRow[nnzit] = i;
            jCol[nnzit] = j + nx_sparse;
            nnzit++;
          }
        }
        assert(nnzit == nele_jac);
      }

    } else {
      assert(values != NULL);

      // avoid unnecessary reallocations when one-call constraints/Jacobian is active
      if(false == onecall_Jac_detected_) {
        if(JacDeq == NULL) {
          JacDeq = new hiopMatrixDenseRowMajor(n_eq, nx_dense);
          assert(JacDineq == NULL);
        } else {
          // this for the case when the problem (constraints) sizes changed
          if(JacDeq->m() != n_eq || JacDeq->n() != nx_dense) {
            delete JacDeq;
            JacDeq = new hiopMatrixDenseRowMajor(n_eq, nx_dense);
          }
        }
        if(JacDineq == NULL) {
          JacDineq = new hiopMatrixDenseRowMajor(n_ineq, nx_dense);
        } else {
          // this for the case when the problem (constraints) sizes changed
          if(JacDineq->m() != n_ineq || JacDineq->n() != nx_dense) {
            delete JacDineq;
            JacDineq = new hiopMatrixDenseRowMajor(n_ineq, nx_dense);
          }
        }

        int nnzit = 0;
        // sparse Jac Eq
        {
          size_type num_cons = n_eq;
          bret = hiopNLP->eval_Jac_cons(nll,
                                        mll,
                                        num_cons,
                                        cons_eq_idxs,
                                        x,
                                        new_x,
                                        nx_sparse,
                                        nx_dense,
                                        nnz_sparse_Jaceq,
                                        NULL,
                                        NULL,
                                        values,
                                        JacDeq->local_data());
          if(bret) {
            nnzit += nnz_sparse_Jaceq;
            assert(nnzit <= nele_jac);

            // the dense part
            const size_t len = (size_t)(n_eq * nx_dense);
            memcpy(values + nnzit, JacDeq->local_data(), len * sizeof(double));

            nnzit += n_eq * nx_dense;
            assert(nnzit <= nele_jac);
          } else {
            eq_call_failed = true;
            delete JacDeq;
            JacDeq = NULL;
          }
        }

        // sparse Jac Ineq
        {
          size_type num_cons = n_ineq;
          bret = hiopNLP->eval_Jac_cons(nll,
                                        mll,
                                        num_cons,
                                        cons_ineq_idxs,
                                        x,
                                        new_x,
                                        nx_sparse,
                                        nx_dense,
                                        nnz_sparse_Jacineq,
                                        NULL,
                                        NULL,
                                        values + nnzit,
                                        JacDineq->local_data());
          if(bret) {
            nnzit += nnz_sparse_Jacineq;
            assert(nnzit <= nele_jac);

            const size_t len = (size_t)(n_ineq * nx_dense);
            // the dense part
            memcpy(values + nnzit, JacDineq->local_data(), len * sizeof(double));
            nnzit += n_ineq * nx_dense;
            assert(nnzit == nele_jac);
          } else {
            delete JacDineq;
            JacDineq = NULL;
            if(!eq_call_failed) {
              return false;
            } else {
              onecall_Jac_detected_ = true;
              try_onecall_Jac = true;
            }
          }
        }
      } else {  // if(true == onecall_Jac_detected_) {
        try_onecall_Jac = true;
      }

      // try one call Jacobian
      if(try_onecall_Jac) {
        if(JacDeqineq == NULL) {
          JacDeqineq = new hiopMatrixDenseRowMajor(m, nx_dense);
          assert(JacDeq == NULL);
          assert(JacDineq == NULL);
        }

        bret = hiopNLP->eval_Jac_cons(nll,
                                      mll,
                                      x,
                                      new_x,
                                      nx_sparse,
                                      nx_dense,
                                      nnz_sparse_Jaceq + nnz_sparse_Jacineq,
                                      NULL,
                                      NULL,
                                      values,
                                      JacDeqineq->local_data());
        if(!bret) return false;

        int nnzit = nnz_sparse_Jaceq + nnz_sparse_Jacineq;
        // put the dense part of the MDS in the Ipopt sparse Jac matrix
        memcpy(values + nnzit, JacDeqineq->local_data(), ((size_t)m * nx_dense) * sizeof(double));
        nnzit += m * nx_dense;

        assert(nnzit == nele_jac);
      }
    }
    return true;
  }

  bool eval_h(Index n,
              const Number* x,
              bool new_x,
              Number obj_factor,
              Index m,
              const Number* lambda,
              bool new_lambda,
              Index nele_hess,
              Index* iRow,
              Index* jCol,
              Number* values)
  {
    bool bret = true;
    size_type nll = n, mll = m;
    assert(nnz_sparse_Hess_Lagr_SD == 0 && "not yet supported");

    if(values == NULL) {
      int nnzit = 0;

      bret = hiopNLP->eval_Hess_Lagr(nll,
                                     mll,
                                     x,
                                     new_x,
                                     obj_factor,
                                     lambda,
                                     new_lambda,
                                     nx_sparse,
                                     nx_dense,
                                     nnz_sparse_Hess_Lagr_SS,
                                     iRow,
                                     jCol,
                                     NULL,
                                     NULL,
                                     nnz_sparse_Hess_Lagr_SD,
                                     NULL,
                                     NULL,
                                     NULL);
      if(!bret) return false;
      nnzit += nnz_sparse_Hess_Lagr_SS;

      // dense part
      for(int i = 0; i < nx_dense; i++) {
        const int row = nx_sparse + i;
        for(int j = i; j < nx_dense; j++) {
          iRow[nnzit] = row;
          jCol[nnzit] = nx_sparse + j;
          nnzit++;
        }
      }
#ifndef NDEBUG
      // nnzit += nx_dense*nx_dense;
      assert(nnzit == nele_hess);
#endif

    } else {
      assert(values != NULL);

      int nnzit = 0;
      if(HessDL == NULL) {
        HessDL = new hiopMatrixDenseRowMajor(nx_dense, nx_dense);
      } else {
        // this for the case when the problem (constraints) sizes changed
        if(HessDL->m() != nx_dense) {
          delete HessDL;
          HessDL = new hiopMatrixDenseRowMajor(nx_dense, nx_dense);
        }
      }
      double* HessMat = HessDL->local_data();

      bret = hiopNLP->eval_Hess_Lagr(nll,
                                     mll,
                                     x,
                                     new_x,
                                     obj_factor,
                                     lambda,
                                     new_lambda,
                                     nx_sparse,
                                     nx_dense,
                                     nnz_sparse_Hess_Lagr_SS,
                                     NULL,
                                     NULL,
                                     values,
                                     HessMat,
                                     nnz_sparse_Hess_Lagr_SD,
                                     NULL,
                                     NULL,
                                     NULL);
      if(!bret) return false;
      nnzit += nnz_sparse_Hess_Lagr_SS;

      // dense part
      for(int i = 0; i < nx_dense; ++i) {
        for(int j = i; j < nx_dense; ++j) {
          values[nnzit] = HessMat[i * nx_dense + j];
          nnzit++;
        }
      }
#ifndef NDEBUG
      // nnzit += nx_dense*nx_dense;
      assert(nnzit == nele_hess);
#endif
    }

    return true;
  }

  /* This method is called when the algorithm terminates. */
  void finalize_solution(SolverReturn status,
                         Index n,
                         const Number* x,
                         const Number* z_L,
                         const Number* z_U,
                         Index m,
                         const Number* g,
                         const Number* lambda,
                         Number obj_value,
                         const IpoptData* ip_data,
                         IpoptCalculatedQuantities* ip_cq)
  {
    //! we use hiop::Solve_Success -> //! TODO: convert between IPOPT and HiOp err codes
    hiopNLP->solution_callback(hiop::Solve_Success, n, x, z_L, z_U, m, g, lambda, obj_value);

    // free auxiliary buffers that may have been used by this adapter
    delete JacDeq;
    delete JacDineq;
    delete HessDL;
    delete JacDeqineq;
    JacDeq = JacDineq = HessDL = JacDeqineq = NULL;

    delete[] cons_eq_idxs;
    delete[] cons_ineq_idxs;
    cons_eq_idxs = cons_ineq_idxs = NULL;

    onecall_Jac_detected_ = false;
  };

private:
  hiopInterfaceMDS* hiopNLP;
  int nx_sparse, nx_dense;  // by convention, sparse variables comes first
  int nnz_sparse_Jaceq, nnz_sparse_Jacineq;
  int nnz_sparse_Hess_Lagr_SS, nnz_sparse_Hess_Lagr_SD;
  int n_eq, n_ineq;
  index_type *cons_eq_idxs, *cons_ineq_idxs;
  hiopMatrixDenseRowMajor *JacDeq, *JacDineq, *HessDL;
  hiopMatrixDenseRowMajor* JacDeqineq;  // this holds the full Jacobian when one-call Jacobian is activated

  bool onecall_Jac_detected_;

  /* Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *
   */
  hiopMDS2IpoptTNLP() {};
  hiopMDS2IpoptTNLP(const hiopMDS2IpoptTNLP&) {};
  hiopMDS2IpoptTNLP& operator=(const hiopMDS2IpoptTNLP&);
  //@}
};

/* Adapter from Sparse NLP formulation to Ipopt's general TNLP */
class hiopSparse2IpoptTNLP : public TNLP
{
public:
  hiopSparse2IpoptTNLP(hiopInterfaceSparse* hiopNLP_)
      : hiopNLP(hiopNLP_),
        m_nx{0},
        m_nnz_sparse_Jaceq{0},
        m_nnz_sparse_Jacineq{0},
        m_nnz_sparse_Hess_Lagr{0},
        m_cons_eq_idxs{nullptr},
        m_cons_ineq_idxs{nullptr},
        //     m_Jac_eq{nullptr}, m_Jac_ineq{nullptr}, m_Hess{nullptr}, m_Jac{nullptr},
        m_onecall_Jac_detected_{false} {};
  virtual ~hiopSparse2IpoptTNLP()
  {
    delete[] m_cons_eq_idxs;
    delete[] m_cons_ineq_idxs;
    //    delete [] m_Jac_eq;
    //    delete [] m_Jac_ineq;
    //    delete [] m_Hess;
    //    delete [] m_Jac;
  };

  /* Overloads from TNLP */
  /** Method to return some info about the nlp */
  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    size_type nvars, ncons;
    if(false == hiopNLP->get_prob_sizes(nvars, ncons)) return false;

    if(false == hiopNLP->get_sparse_blocks_info(m_nx, m_nnz_sparse_Jaceq, m_nnz_sparse_Jacineq, m_nnz_sparse_Hess_Lagr)) {
      return false;
    }

    nnz_jac_g = m_nnz_sparse_Jaceq + m_nnz_sparse_Jacineq;

    n = (int)nvars;
    m = (int)ncons;
    nnz_h_lag = m_nnz_sparse_Hess_Lagr;

    index_style = TNLP::C_STYLE;
    return true;
  }

  /** Method to return the bounds for my problem */
  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u)
  {
    bool bSuccess = true;
    size_type nll = n, mll = m;
    hiopInterfaceBase::NonlinearityType* types = new hiopInterfaceBase::NonlinearityType[n];
    bSuccess = hiopNLP->get_vars_info(nll, x_l, x_u, types);
    delete[] types;

    if(bSuccess) {
      types = new hiopInterfaceBase::NonlinearityType[m];
      bSuccess = hiopNLP->get_cons_info(mll, g_l, g_u, types);
      delete[] types;
    }

    m_n_eq = m_n_ineq = 0;
    for(int it = 0; it < m; it++) {
      if(g_l[it] == g_u[it])
        m_n_eq++;
      else
        m_n_ineq++;
    }
    assert(m_n_eq + m_n_ineq == m);
    if(m_cons_eq_idxs != NULL) delete[] m_cons_eq_idxs;
    if(m_cons_ineq_idxs != NULL) delete[] m_cons_ineq_idxs;

    m_cons_eq_idxs = new index_type[m_n_eq];
    m_cons_ineq_idxs = new index_type[m_n_ineq];

    int it_eq = 0, it_ineq = 0;
    for(int it = 0; it < m; it++) {
      if(g_l[it] == g_u[it])
        m_cons_eq_idxs[it_eq++] = it;
      else
        m_cons_ineq_idxs[it_ineq++] = it;
    }
    assert(it_eq == m_n_eq);
    assert(it_ineq == m_n_ineq);

    return bSuccess;
  }

  /** Method to return the starting point for the algorithm */
  bool get_starting_point(Index n,
                          bool init_x,
                          Number* x,
                          bool init_z,
                          Number* z_L,
                          Number* z_U,
                          Index m,
                          bool init_lambda,
                          Number* lambda)
  {
    assert(false == init_z && "primal-dual restart not supported by the addapter");
    assert(false == init_lambda && "primal-dual restart not supported by the addapter");
    size_type nll = n;
    return hiopNLP->get_starting_point(nll, x);
  }

  /** Method to return the objective value */
  bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
  {
    size_type nll = n;
    return hiopNLP->eval_f(nll, x, new_x, obj_value);
  }

  /** Method to return the gradient of the objective */
  bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
  {
    size_type nll = n;
    return hiopNLP->eval_grad_f(nll, x, new_x, grad_f);
  }

  /** Method to return the constraint residuals */
  // HiOp calls Eq and Ineq separately -> the interface expects that so we have to
  // mimic it
  bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
  {
    size_type nll = n, mll = m;
    bool bret = false;
    bool eq_call_failed = false;
    bool try_onecall_Jac = false;

    {
      double g_eq[m_n_eq];
      size_type num_cons = m_n_eq;
      bret = hiopNLP->eval_cons(nll, mll, num_cons, m_cons_eq_idxs, x, new_x, g_eq);
      if(bret) {
        for(int i = 0; i < m_n_eq; i++) g[m_cons_eq_idxs[i]] = g_eq[i];
      } else {
        eq_call_failed = true;
      }
    }

    {
      double g_ineq[m_n_ineq];
      size_type num_cons = m_n_ineq;
      bret = hiopNLP->eval_cons(nll, mll, num_cons, m_cons_ineq_idxs, x, new_x, g_ineq);
      if(bret) {
        for(int i = 0; i < m_n_ineq; i++) g[m_cons_ineq_idxs[i]] = g_ineq[i];
      } else {
        if(!eq_call_failed)
          return false;
        else
          try_onecall_Jac = true;
      }
    }

    if(try_onecall_Jac) {
      bret = hiopNLP->eval_cons(nll, mll, x, new_x, g);
    }

    return bret;
  }

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values)
  {
    bool bret = true;
    size_type nll = n, mll = m;
    bool eq_call_failed = false;
    bool try_onecall_Jac = false;
    if(values == NULL) {
      int nnzit = 0;
      // Sparse Jac for Eq
      {
        size_type num_cons = m_n_eq;
        bret = hiopNLP->eval_Jac_cons(nll, mll, num_cons, m_cons_eq_idxs, x, new_x, m_nnz_sparse_Jaceq, iRow, jCol, NULL);

        if(bret) {
          nnzit += m_nnz_sparse_Jaceq;
        } else {
          eq_call_failed = true;
        }
      }

      // Sparse Jac for Ineq
      {
        size_type num_cons = m_n_ineq;
        bret = hiopNLP->eval_Jac_cons(nll,
                                      mll,
                                      num_cons,
                                      m_cons_ineq_idxs,
                                      x,
                                      new_x,
                                      m_nnz_sparse_Jacineq,
                                      iRow + nnzit,
                                      jCol + nnzit,
                                      NULL);
        if(bret) {
          // in-place shift of iRow and jCol for Jacineq
          for(int it = nnzit; it < nnzit + m_nnz_sparse_Jacineq; it++) iRow[it] += m_n_eq;

          nnzit += m_nnz_sparse_Jacineq;
          assert(nnzit <= nele_jac);
        } else {
          if(eq_call_failed)
            try_onecall_Jac = true;
          else
            return false;
        }
      }
      assert(try_onecall_Jac || nnzit == nele_jac);

      if(try_onecall_Jac) {
        bret = hiopNLP->eval_Jac_cons(nll, mll, x, new_x, m_nnz_sparse_Jaceq + m_nnz_sparse_Jacineq, iRow, jCol, values);
        if(!bret) {
          return false;
        }
        nnzit = m_nnz_sparse_Jaceq + m_nnz_sparse_Jacineq;
        assert(nnzit == nele_jac);
      }
    } else {
      assert(values != NULL);

      // avoid unnecessary reallocations when one-call constraints/Jacobian is active

      if(false == m_onecall_Jac_detected_) {
        int nnzit = 0;
        // sparse Jac Eq
        {
          size_type num_cons = m_n_eq;
          bret =
              hiopNLP->eval_Jac_cons(nll, mll, num_cons, m_cons_eq_idxs, x, new_x, m_nnz_sparse_Jaceq, NULL, NULL, values);
          if(bret) {
            nnzit += m_nnz_sparse_Jaceq;
            assert(nnzit <= nele_jac);
          } else {
            eq_call_failed = true;
          }
        }

        // sparse Jac Ineq
        {
          size_type num_cons = m_n_ineq;
          bret = hiopNLP->eval_Jac_cons(nll,
                                        mll,
                                        num_cons,
                                        m_cons_ineq_idxs,
                                        x,
                                        new_x,
                                        m_nnz_sparse_Jacineq,
                                        NULL,
                                        NULL,
                                        values + nnzit);
          if(bret) {
            nnzit += m_nnz_sparse_Jacineq;
            assert(nnzit <= nele_jac);
            assert(nnzit == nele_jac);
          } else {
            if(!eq_call_failed) {
              return false;
            } else {
              m_onecall_Jac_detected_ = true;
              try_onecall_Jac = true;
            }
          }
        }
      } else {  // if(true == onecall_Jac_detected_) {
        try_onecall_Jac = true;
      }

      // try one call Jacobian
      if(try_onecall_Jac) {
        bret = hiopNLP->eval_Jac_cons(nll, mll, x, new_x, m_nnz_sparse_Jaceq + m_nnz_sparse_Jacineq, NULL, NULL, values);
        if(!bret) return false;

        int nnzit = m_nnz_sparse_Jaceq + m_nnz_sparse_Jacineq;
        assert(nnzit == nele_jac);
      }
    }
    return true;
  }

  bool eval_h(Index n,
              const Number* x,
              bool new_x,
              Number obj_factor,
              Index m,
              const Number* lambda,
              bool new_lambda,
              Index nele_hess,
              Index* iRow,
              Index* jCol,
              Number* values)
  {
    bool bret = true;
    size_type nll = n, mll = m;

    if(values == NULL) {
      int nnzit = 0;

      bret =
          hiopNLP
              ->eval_Hess_Lagr(nll, mll, x, new_x, obj_factor, lambda, new_lambda, m_nnz_sparse_Hess_Lagr, iRow, jCol, NULL);
      if(!bret) return false;
      nnzit += m_nnz_sparse_Hess_Lagr;

#ifndef NDEBUG
      // nnzit += nx_dense*nx_dense;
      assert(nnzit == nele_hess);
#endif

    } else {
      assert(values != NULL);

      int nnzit = 0;

      bret = hiopNLP->eval_Hess_Lagr(nll,
                                     mll,
                                     x,
                                     new_x,
                                     obj_factor,
                                     lambda,
                                     new_lambda,
                                     m_nnz_sparse_Hess_Lagr,
                                     iRow,
                                     jCol,
                                     values);
      if(!bret) return false;
      nnzit += m_nnz_sparse_Hess_Lagr;

#ifndef NDEBUG
      // nnzit += nx_dense*nx_dense;
      assert(nnzit == nele_hess);
#endif
    }
    return true;
  }

  /* This method is called when the algorithm terminates. */
  void finalize_solution(SolverReturn status,
                         Index n,
                         const Number* x,
                         const Number* z_L,
                         const Number* z_U,
                         Index m,
                         const Number* g,
                         const Number* lambda,
                         Number obj_value,
                         const IpoptData* ip_data,
                         IpoptCalculatedQuantities* ip_cq)
  {
    //! we use hiop::Solve_Success -> //! TODO: convert between IPOPT and HiOp err codes
    hiopNLP->solution_callback(hiop::Solve_Success, n, x, z_L, z_U, m, g, lambda, obj_value);

    // free auxiliary buffers that may have been used by this adapter
    //    delete m_Jac_eq; delete m_Jac_ineq;  delete m_Hess; delete m_Jac;
    //    JacDeq = JacDineq = HessDL = JacDeqineq = NULL;

    delete[] m_cons_eq_idxs;
    delete[] m_cons_ineq_idxs;
    m_cons_eq_idxs = m_cons_ineq_idxs = NULL;

    m_onecall_Jac_detected_ = false;
  };

private:
  hiopInterfaceSparse* hiopNLP;
  int m_nx;  // by convention, sparse variables comes first
  int m_nnz_sparse_Jaceq, m_nnz_sparse_Jacineq, m_nnz_sparse_Hess_Lagr;
  int m_n_eq, m_n_ineq;
  index_type *m_cons_eq_idxs, *m_cons_ineq_idxs;
  //  hiopMatrixSparseTriplet *m_Jac_eq, *m_Jac_ineq, *m_Hess;
  //  hiopMatrixSparseTriplet *m_Jac; //this holds the full Jacobian when one-call Jacobian is activated

  bool m_onecall_Jac_detected_;

  /* Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *
   */
  hiopSparse2IpoptTNLP() {};
  hiopSparse2IpoptTNLP(const hiopSparse2IpoptTNLP&) {};
  hiopSparse2IpoptTNLP& operator=(const hiopSparse2IpoptTNLP&);
  //@}
};

}  // end of namespace hiop
#endif
