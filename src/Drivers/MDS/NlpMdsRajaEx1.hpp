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
 * @file NlpMdsRajaEx1.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#ifndef HIOP_EXAMPLE_MDS_RAJA_EX1
#define HIOP_EXAMPLE_MDS_RAJA_EX1

#include "hiopInterface.hpp"

// this include is not needed in general
// we use hiopMatrixDense in this particular example for convienience
#include <hiopMatrixDense.hpp>
#include <LinAlgFactory.hpp>

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

/**
 *  Problem test for the linear algebra of Mixed Dense-Sparse NLPs
 *
 * If 'empty_sp_row' is set to true, the following MDS NLP is solved
 *
 *  min   sum 0.5 {x_i*(x_{i}-1) : i=1,...,ns} + 0.5 y'*Qd*y + 0.5 s^T s
 *  s.t.  x+s + Md y = 0, i=1,...,ns
 *        [-2  ]    [ x_1 + e^T s]   [e^T]      [ 2 ]
 *        [-inf] <= [            ] + [e^T] y <= [ 2 ]
 *        [-2  ]    [ x_3        ]   [e^T]      [inf]
 *        x <= 3
 *        s>=0
 *        -4 <=y_1 <=4, the rest of y are free
 *
 * Otherwise (second inequality involves a sparse variable):
 *
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
 *
 * @note All pointers marked as "managed by Umpire" are allocated by HiOp using the
 * Umpire's API. They all are addresses in the same memory space; however, the memory
 * space can be host (typically CPU), device (typically GPU), or unified memory (um)
 * spaces as per Umpire specification. The selection of the memory space is done via
 * the option "mem_space" of HiOp. It is the responsibility of the implementers of
 * the HiOp's interfaces (such as the hiop::hiopInterfaceMDS used in this example) to
 * work with the "managed by Umpire" pointers in the same memory space as the one
 * specified by the "mem_space" option.
 *
 */

class MdsEx1 : public hiop::hiopInterfaceMDS
{
public:
  MdsEx1(int ns_in, std::string mem_space, bool empty_sp_row = false)
      : MdsEx1(ns_in, ns_in, mem_space, empty_sp_row)
  {}

  MdsEx1(int ns_in, int nd_in, std::string mem_space, bool empty_sp_row = false);

  virtual ~MdsEx1();

  /**
   * @brief Number of variables and constraints.
   */
  bool get_prob_sizes(size_type& n, size_type& m);

  /**
   * @brief Get types and bounds on the variables.
   *
   * @param[in] n number of variables
   * @param[out] ixlow array with lower bounds (managed by Umpire)
   * @param[out] ixupp array with upper bounds (managed by Umpire)
   * @param[out] type array with the variable types (on host)
   */
  bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);

  /**
   * Get types and bounds corresponding to constraints. An equality constraint is specified
   * by setting the lower and upper bounds equal.
   *
   * @param[in] m Number of constraints
   * @param[out] iclow array with lower bounds (managed by Umpire)
   * @param[out] icupp array with upper bounds (managed by Umpire)
   * @param[out] type array with the variable types (on host)
   */
  bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  /**
   *  Returns the sizes and number of nonzeros of the sparse and dense blocks within MDS
   *
   * @param[out] nx_sparse number of sparse variables
   * @param[out] nx_dense number of dense variables
   * @param[out] nnz_sparse_Jace number of nonzeros in the Jacobian of the equalities w.r.t.
   *                             sparse variables
   * @param[out] nnz_sparse_Jaci number of nonzeros in the Jacobian of the inequalities w.r.t.
   *                             sparse variables
   * @param[out] nnz_sparse_Hess_Lagr_SS number of nonzeros in the (sparse) Hessian w.r.t.
   * sparse variables
   * @param[out] nnz_sparse_Hess_Lagr_SD reserved, always set to 0
   */
  bool get_sparse_dense_blocks_info(int& nx_sparse,
                                    int& nx_dense,
                                    int& nnz_sparse_Jace,
                                    int& nnz_sparse_Jaci,
                                    int& nnz_sparse_Hess_Lagr_SS,
                                    int& nnz_sparse_Hess_Lagr_SD);

  /**
   * Evaluate objective.
   *
   * @param[in] n number of variables
   * @param[in] x array with the optimization variables or point at which to evaluate
   *              (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been
   * evaluated previously (false) or not (true) at x
   * @param[out] obj_value the objective function value.
   */
  bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);

  /**
   * Evaluate a subset of the constraints specified by idx_cons
   *
   * @param[in] num_cons number of constraints to evaluate (size of idx_cons array)
   * @param[in] idx_cons indexes of the constraints to evaluate (managed by Umpire)
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *            previously (false) or not (true) at x
   * @param[out] cons array with values of the constraints (managed by Umpire, size num_cons)
   */
  bool eval_cons(const size_type& n,
                 const size_type& m,
                 const size_type& num_cons,
                 const index_type* idx_cons,
                 const double* x,
                 bool new_x,
                 double* cons);

  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
  {
    // return false so that HiOp will rely on the constraint evaluator defined above
    return false;
  }

  /**
   * Evaluation of the gradient of the objective.
   *
   * @param[in] n number of variables
   * @param[in] x array with the optimization variables or point at which to evaluate
   *              (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *                  previously (false) or not (true) at x
   * @param[out] gradf array with the values of the gradient (managed by Umpire)
   */
  bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);

  /**
   * Evaluates the Jacobian of the constraints. Please check the user manual and the
   * documentation of hiop::hiopInterfaceMDS for a detailed discussion of how the last
   * four arguments are expected to behave.
   *
   * @param[in] n number of variables
   * @param[in] m Number of constraints
   * @param[in] num_cons number of constraints to evaluate (size of idx_cons array)
   * @param[in] idx_cons indexes of the constraints to evaluate (managed by Umpire)
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *                  previously (false) or not (true) at x
   * @param[in] nsparse number of sparse variables
   * @param[in] ndense number of dense variables
   * @param[in] nnzJacS number of nonzeros in the sparse Jacobian
   * @param[out] iJacS array of row indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] jJacS array of column indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] MJacS array of nonzero values in the sparse Jacobian (managed by Umpire)
   * @param[out] JacD array with the values of the dense Jacobian (managed by Umpire)
   */
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
                             double* JacD);

  /// Similar to the above, but not used in this example.
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
    // return false so that HiOp will rely on the Jacobian evaluator defined above
    return false;
  }

  /**
   * Evaluate the Hessian of the Lagrangian function. Please consult the user manual for a
   * detailed discussion of the form the Lagrangian function takes.
   *
   * @param[in] n number of variables
   * @param[in] m Number of constraints
   * @param[in] x the point at which to evaluate (managed by Umpire)
   * @param[in] new_x indicates whether any of the other eval functions have been evaluated
   *                  previously (false) or not (true) at x
   * @param[in] obj_factor scalar that multiplies the objective term in the Lagrangian function
   * @param[in] lambda array with values of the multipliers used by the Lagrangian function
   * @param[in] new_lambda indicates whether lambda  values changed since last call
   * @param[in] nsparse number of sparse variables
   * @param[in] ndense number of dense variables
   * @param[in] nnzHSS number of nonzeros in the (sparse) Hessian w.r.t. sparse variables
   * @param[out] iHSS array of row indexes in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] jHSS array of column indexes in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] MHSS array of nonzero values in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] HDDD array with the values of the Hessian w.r.t. to dense variables
   *                  (managed by Umpire)
   * @param[out] iHSD is reserved and should not be accessed
   * @param[out] jHSD is reserved and should not be accessed
   * @param[out] MHSD is reserved and should not be accessed
   * @param[out] HHSD is reserved and should not be accessed

   */
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
                      double* MHSD);

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const size_type& global_n, double* x0);

  bool get_starting_point(const size_type& n,
                          const size_type& m,
                          double* x0,
                          bool& duals_avail,
                          double* z_bndL0,
                          double* z_bndU0,
                          double* lambda0,
                          bool& slacks_avail,
                          double* ineq_slack);

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

  /* The public methods below are not part of hiopInterface. They are a proxy
   * for user's (front end) code to set solutions from a previous solve.
   *
   * Same behaviour can be achieved internally (in this class) if desired by
   * overriding @solution_callback and @get_starting_point
   */
  void set_solution_primal(const double* x_vec);

  void set_solution_duals(const double* zl_vec, const double* zu_vec, const double* lambda_vec);

  void initialize();

protected:
  int ns_, nd_;
  hiop::hiopMatrixDense* Q_;
  hiop::hiopMatrixDense* Md_;
  double* buf_y_;
  bool haveIneq_;
  std::string mem_space_;

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
  MdsEx1OneCallCons(int ns_in, std::string mem_space, bool empty_sp_row = false)
      : MdsEx1(ns_in, mem_space, empty_sp_row)
  {}

  MdsEx1OneCallCons(int ns_in, int nd_in, std::string mem_space)
      : MdsEx1(ns_in, nd_in, mem_space)
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
  bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);

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
                             double* JacD);
};

#endif
