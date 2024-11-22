// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to
// endorse or promote products derived from this software without specific prior written
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC
// nor any of their employees, makes any warranty, express or implied, or assumes any
// liability or responsibility for the accuracy, completeness, or usefulness of any
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or
// imply its endorsement, recommendation, or favoring by the United States Government or
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed
// herein do not necessarily state or reflect those of the United States Government or
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or
// product endorsement purposes.

/**
 * @file chiopInterface.hpp
 *
 * @author Michel Schanen <mschanen@anl.gov>, ANL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#ifndef CHIOP_INTERFACE_HPP
#define CHIOP_INTERFACE_HPP
#include "hiop_defs.hpp"
#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

/** Light C interface that wraps around the mixed-dense nlp class in HiOp. Its initial motivation
 * was to serve as an interface to Julia
 */

using namespace hiop;
class cppUserProblemMDS;
extern "C" {
// C struct with HiOp function callbacks
typedef struct cHiopMDSProblem
{
  hiopNlpMDS* refcppHiop;
  cppUserProblemMDS* hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void* user_data;
  // Used by hiop_mds_solve_problem() to store the final state. The duals should be added here.
  double* solution;
  double obj_value;
  // HiOp callback function wrappers
  int (*get_starting_point)(size_type n_, double* x0, void* user_data);
  int (*get_prob_sizes)(size_type* n_, size_type* m_, void* user_data);
  int (*get_vars_info)(size_type n, double* xlow_, double* xupp_, void* user_data);
  int (*get_cons_info)(size_type m, double* clow_, double* cupp_, void* user_data);
  int (*eval_f)(size_type n, double* x, int new_x, double* obj, void* user_data);
  int (*eval_grad_f)(size_type n, double* x, int new_x, double* gradf, void* user_data);
  int (*eval_cons)(size_type n, size_type m, double* x, int new_x, double* cons, void* user_data);
  int (*get_sparse_dense_blocks_info)(hiop_size_type* nx_sparse,
                                      hiop_size_type* nx_dense,
                                      hiop_size_type* nnz_sparse_Jaceq,
                                      hiop_size_type* nnz_sparse_Jacineq,
                                      hiop_size_type* nnz_sparse_Hess_Lagr_SS,
                                      hiop_size_type* nnz_sparse_Hess_Lagr_SD,
                                      void* user_data);
  int (*eval_Jac_cons)(size_type n,
                       size_type m,
                       double* x,
                       int new_x,
                       size_type nsparse,
                       size_type ndense,
                       int nnzJacS,
                       hiop_index_type* iJacS,
                       hiop_index_type* jJacS,
                       double* MJacS,
                       double* JacD,
                       void* user_data);
  int (*eval_Hess_Lagr)(size_type n,
                        size_type m,
                        double* x,
                        int new_x,
                        double obj_factor,
                        double* lambda,
                        int new_lambda,
                        size_type nsparse,
                        size_type ndense,
                        hiop_size_type nnzHSS,
                        hiop_index_type* iHSS,
                        hiop_index_type* jHSS,
                        double* MHSS,
                        double* HDD,
                        hiop_size_type nnzHSD,
                        hiop_index_type* iHSD,
                        hiop_index_type* jHSD,
                        double* MHSD,
                        void* user_data);
} cHiopMDSProblem;
}

// The cpp object used in the C interface
class cppUserProblemMDS : public hiopInterfaceMDS
{
public:
  cppUserProblemMDS(cHiopMDSProblem* cprob_)
      : cprob(cprob_)
  {}

  virtual ~cppUserProblemMDS() {}
  // HiOp callbacks calling the C wrappers
  bool get_prob_sizes(size_type& n_, size_type& m_)
  {
    cprob->get_prob_sizes(&n_, &m_, cprob->user_data);
    return true;
  };
  bool get_starting_point(const size_type& n, double* x0)
  {
    cprob->get_starting_point(n, x0, cprob->user_data);
    return true;
  };
  bool get_vars_info(const size_type& n, double* xlow_, double* xupp_, NonlinearityType* type)
  {
    for(size_type i = 0; i < n; ++i) type[i] = hiopNonlinear;
    cprob->get_vars_info(n, xlow_, xupp_, cprob->user_data);
    return true;
  };
  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    for(size_type i = 0; i < m; ++i) type[i] = hiopNonlinear;
    cprob->get_cons_info(m, clow, cupp, cprob->user_data);
    return true;
  };
  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    cprob->eval_f(n, (double*)x, 0, &obj_value, cprob->user_data);
    return true;
  };

  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    cprob->eval_grad_f(n, (double*)x, 0, gradf, cprob->user_data);

    return true;
  };
  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const hiop_index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons)
  {
    return false;
  };
  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    cprob->eval_cons(n, m, (double*)x, new_x, cons, cprob->user_data);
    return true;
  };
  bool get_sparse_dense_blocks_info(hiop_size_type& nx_sparse,
                                    hiop_size_type& nx_dense,
                                    hiop_size_type& nnz_sparse_Jaceq,
                                    hiop_size_type& nnz_sparse_Jacineq,
                                    hiop_size_type& nnz_sparse_Hess_Lagr_SS,
                                    hiop_size_type& nnz_sparse_Hess_Lagr_SD)
  {
    cprob->get_sparse_dense_blocks_info(&nx_sparse,
                                        &nx_dense,
                                        &nnz_sparse_Jaceq,
                                        &nnz_sparse_Jacineq,
                                        &nnz_sparse_Hess_Lagr_SS,
                                        &nnz_sparse_Hess_Lagr_SD,
                                        cprob->user_data);
    return true;
  };
  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const hiop_index_type* idx_cons,
                     const double* x,
                     bool new_x,
                     const size_type& nsparse,
                     const size_type& ndense,
                     const hiop_size_type& nnzJacS,
                     hiop_index_type* iJacS,
                     hiop_index_type* jJacS,
                     double* MJacS,
                     double* JacD)
  {
    return false;
  };
  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const double* x,
                     bool new_x,
                     const size_type& nsparse,
                     const size_type& ndense,
                     const hiop_size_type& nnzJacS,
                     hiop_index_type* iJacS,
                     hiop_index_type* jJacS,
                     double* MJacS,
                     double* JacD)
  {
    cprob->eval_Jac_cons(n, m, (double*)x, new_x, nsparse, ndense, nnzJacS, iJacS, jJacS, MJacS, JacD, cprob->user_data);
    return true;
  };
  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const size_type& nsparse,
                      const size_type& ndense,
                      const hiop_size_type& nnzHSS,
                      hiop_index_type* iHSS,
                      hiop_index_type* jHSS,
                      double* MHSS,
                      double* HDD,
                      hiop_size_type& nnzHSD,
                      hiop_index_type* iHSD,
                      hiop_index_type* jHSD,
                      double* MHSD)
  {
    // Note: lambda is not used since all the constraints are linear and, therefore, do
    // not contribute to the Hessian of the Lagrangian
    cprob->eval_Hess_Lagr(n,
                          m,
                          (double*)x,
                          new_x,
                          obj_factor,
                          (double*)lambda,
                          new_lambda,
                          nsparse,
                          ndense,
                          nnzHSS,
                          iHSS,
                          jHSS,
                          MHSS,
                          HDD,
                          nnzHSD,
                          iHSD,
                          jHSD,
                          MHSD,
                          cprob->user_data);
    return true;
  };

private:
  // Storing the C struct in the CPP object
  cHiopMDSProblem* cprob;
};

/** The 3 essential function calls to create and destroy a problem object in addition to solve a problem.
 * Some option setters will be added in the future.
 */
extern "C" int hiop_mds_create_problem(cHiopMDSProblem* problem);
extern "C" int hiop_mds_solve_problem(cHiopMDSProblem* problem);
extern "C" int hiop_mds_destroy_problem(cHiopMDSProblem* problem);

#ifdef HIOP_SPARSE
class cppUserProblemSparse;
extern "C" {
// C struct with HiOp function callbacks
typedef struct cHiopSparseProblem
{
  hiopNlpSparse* refcppHiop_;
  cppUserProblemSparse* hiopinterface_;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void* user_data_;
  // Used by hiop_sparse_createProblemsolveProblem() to store the final state. The duals should be added here.
  double* solution_;
  double obj_value_;
  int niters_;
  int status_;
  // HiOp callback function wrappers
  int (*get_starting_point_)(hiop_size_type n, double* x0, void* user_data);
  int (*get_prob_sizes_)(hiop_size_type* n, hiop_size_type* m, void* user_data);
  int (*get_vars_info_)(hiop_size_type n, double* xlow, double* xupp, void* user_data);
  int (*get_cons_info_)(hiop_size_type m, double* clow, double* cupp, void* user_data);
  int (*eval_f_)(hiop_size_type n, double* x, int new_x, double* obj, void* user_data);
  int (*eval_grad_f_)(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data);
  int (*eval_cons_)(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* user_data);
  int (*get_sparse_blocks_info_)(hiop_size_type* nx,
                                 hiop_size_type* nnz_sparse_Jaceq,
                                 hiop_size_type* nnz_sparse_Jacineq,
                                 hiop_size_type* nnz_sparse_Hess_Lagr,
                                 void* user_data);
  int (*eval_Jac_cons_)(size_type n,
                        size_type m,
                        double* x,
                        int new_x,
                        int nnzJacS,
                        hiop_index_type* iJacS,
                        hiop_index_type* jJacS,
                        double* MJacS,
                        void* user_data);
  int (*eval_Hess_Lagr_)(size_type n,
                         size_type m,
                         double* x,
                         int new_x,
                         double obj_factor,
                         double* lambda,
                         int new_lambda,
                         hiop_size_type nnzHSS,
                         hiop_index_type* iHSS,
                         hiop_index_type* jHSS,
                         double* MHSS,
                         void* user_data);
} cHiopSparseProblem;
}

// The cpp object used in the C interface
class cppUserProblemSparse : public hiopInterfaceSparse
{
public:
  cppUserProblemSparse(cHiopSparseProblem* cprob)
      : cprob_(cprob)
  {}

  virtual ~cppUserProblemSparse() {}

  // HiOp callbacks calling the C wrappers
  bool get_prob_sizes(size_type& n, size_type& m)
  {
    cprob_->get_prob_sizes_(&n, &m, cprob_->user_data_);
    return true;
  };

  bool get_starting_point(const size_type& n, double* x0)
  {
    cprob_->get_starting_point_(n, x0, cprob_->user_data_);
    return true;
  };

  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
  {
    for(size_type i = 0; i < n; ++i) {
      type[i] = hiopNonlinear;
    }
    cprob_->get_vars_info_(n, xlow, xupp, cprob_->user_data_);
    return true;
  };

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    for(size_type i = 0; i < m; ++i) {
      type[i] = hiopNonlinear;
    }
    cprob_->get_cons_info_(m, clow, cupp, cprob_->user_data_);
    return true;
  };

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    cprob_->eval_f_(n, (double*)x, 0, &obj_value, cprob_->user_data_);
    return true;
  };

  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    cprob_->eval_grad_f_(n, (double*)x, 0, gradf, cprob_->user_data_);

    return true;
  };

  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons)
  {
    return false;
  };

  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    cprob_->eval_cons_(n, m, (double*)x, new_x, cons, cprob_->user_data_);
    return true;
  };

  bool get_sparse_blocks_info(size_type& nx,
                              size_type& nnz_sparse_Jaceq,
                              size_type& nnz_sparse_Jacineq,
                              size_type& nnz_sparse_Hess_Lagr)
  {
    cprob_->get_sparse_blocks_info_(&nx, &nnz_sparse_Jaceq, &nnz_sparse_Jacineq, &nnz_sparse_Hess_Lagr, cprob_->user_data_);
    return true;
  };

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
    return false;
  };

  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const double* x,
                     bool new_x,
                     const size_type& nnzJacS,
                     index_type* iJacS,
                     index_type* jJacS,
                     double* MJacS)
  {
    cprob_->eval_Jac_cons_(n, m, (double*)x, new_x, nnzJacS, iJacS, jJacS, MJacS, cprob_->user_data_);
    return true;
  };
  bool eval_Hess_Lagr(const size_type& n,
                      const size_type& m,
                      const double* x,
                      bool new_x,
                      const double& obj_factor,
                      const double* lambda,
                      bool new_lambda,
                      const size_type& nnzHSS,
                      index_type* iHSS,
                      index_type* jHSS,
                      double* MHSS)
  {
    // Note: lambda is not used since all the constraints are linear and, therefore, do
    // not contribute to the Hessian of the Lagrangian
    cprob_->eval_Hess_Lagr_(n,
                            m,
                            (double*)x,
                            new_x,
                            obj_factor,
                            (double*)lambda,
                            new_lambda,
                            nnzHSS,
                            iHSS,
                            jHSS,
                            MHSS,
                            cprob_->user_data_);
    return true;
  };

private:
  // Storing the C struct in the CPP object
  cHiopSparseProblem* cprob_;
};

/** The 3 essential function calls to create and destroy a problem object in addition to solve a problem.
 * Some option setters will be added in the future.
 */
extern "C" int hiop_sparse_create_problem(cHiopSparseProblem* problem);
extern "C" int hiop_sparse_solve_problem(cHiopSparseProblem* problem);
extern "C" int hiop_sparse_destroy_problem(cHiopSparseProblem* problem);

#endif  // #ifdef HIOP_SPARSE

class cppUserProblemDense;
extern "C" {
// C struct with HiOp function callbacks
typedef struct cHiopDenseProblem
{
  hiopNlpDenseConstraints* refcppHiop;
  cppUserProblemDense* hiopinterface;
  // user_data similar to the Ipopt interface. In case of Julia pointer to the Julia problem object.
  void* user_data;
  // Used by hiop_sparse_createProblemsolveProblem() to store the final state. The duals should be added here.
  double* solution;
  double obj_value;
  int niters;
  int status;
  // HiOp callback function wrappers
  int (*get_starting_point)(hiop_size_type n_, double* x0, void* user_data);
  int (*get_prob_sizes)(hiop_size_type* n_, hiop_size_type* m_, void* user_data);
  int (*get_vars_info)(hiop_size_type n, double* xlow_, double* xupp_, void* user_data);
  int (*get_cons_info)(hiop_size_type m, double* clow_, double* cupp_, void* user_data);
  int (*eval_f)(hiop_size_type n, double* x, int new_x, double* obj, void* user_data);
  int (*eval_grad_f)(hiop_size_type n, double* x, int new_x, double* gradf, void* user_data);
  int (*eval_cons)(hiop_size_type n, hiop_size_type m, double* x, int new_x, double* cons, void* user_data);
  int (*eval_Jac_cons)(size_type n, size_type m, double* x, int new_x, double* Jac, void* user_data);
} cHiopDenseProblem;
}

// The cpp object used in the C interface
class cppUserProblemDense : public hiopInterfaceDenseConstraints
{
public:
  cppUserProblemDense(cHiopDenseProblem* cprob_)
      : cprob(cprob_)
  {}

  virtual ~cppUserProblemDense() {}

  // HiOp callbacks calling the C wrappers
  bool get_prob_sizes(size_type& n_, size_type& m_)
  {
    cprob->get_prob_sizes(&n_, &m_, cprob->user_data);
    return true;
  };

  bool get_starting_point(const size_type& n, double* x0)
  {
    cprob->get_starting_point(n, x0, cprob->user_data);
    return true;
  };

  bool get_vars_info(const size_type& n, double* xlow_, double* xupp_, NonlinearityType* type)
  {
    for(size_type i = 0; i < n; ++i) {
      type[i] = hiopNonlinear;
    }
    cprob->get_vars_info(n, xlow_, xupp_, cprob->user_data);
    return true;
  };

  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
  {
    for(size_type i = 0; i < m; ++i) {
      type[i] = hiopNonlinear;
    }
    cprob->get_cons_info(m, clow, cupp, cprob->user_data);
    return true;
  };

  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
  {
    cprob->eval_f(n, (double*)x, 0, &obj_value, cprob->user_data);
    return true;
  };

  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
  {
    cprob->eval_grad_f(n, (double*)x, 0, gradf, cprob->user_data);

    return true;
  };

  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons)
  {
    return false;
  };

  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    cprob->eval_cons(n, m, (double*)x, new_x, cons, cprob->user_data);
    return true;
  };

  bool eval_Jac_cons(const size_type& n,
                     const size_type& m,
                     const size_type& num_cons,
                     const index_type* idx_cons,
                     const double* x,
                     bool new_x,
                     double* Jac)
  {
    return false;
  };

  bool eval_Jac_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* Jac)
  {
    cprob->eval_Jac_cons(n, m, (double*)x, new_x, Jac, cprob->user_data);
    return true;
  };

private:
  // Storing the C struct in the CPP object
  cHiopDenseProblem* cprob;
};

/** The 3 essential function calls to create and destroy a problem object in addition to solve a problem.
 * Some option setters will be added in the future.
 */
extern "C" int hiop_dense_create_problem(cHiopDenseProblem* problem);
extern "C" int hiop_dense_solve_problem(cHiopDenseProblem* problem);
extern "C" int hiop_dense_destroy_problem(cHiopDenseProblem* problem);

#endif
