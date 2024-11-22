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
 * @file NlpSparseRajaEx2.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#ifndef HIOP_EXAMPLE_SPARSE_RAJA_EX2
#define HIOP_EXAMPLE_SPARSE_RAJA_EX2

#include "hiopInterface.hpp"
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
 * Newton of HiOp. It uses a Sparse NLP formulation. The problem is based on SparseEx2.
 *
 *  min   (2*convex_obj-1)*scal*sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + 0.5x^Tx
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.0 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]                  (rnkdef-con1 --- ineq con)
 *        4*x_1 + 2*x_2 == 10                                (rnkdef-con2 ---   eq con)
 *
 *  other parameters are:
 *  convex_obj: set to 1 to have a convex problem, otherwise set it to 0
 *  scale_quartic_obj_term: scaling factor for the quartic term in the objective (1.0 by default).
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
class SparseRajaEx2 : public hiop::hiopInterfaceSparse
{
public:
  SparseRajaEx2(std::string mem_space,
                int n,
                bool convex_obj,
                bool rankdefic_Jac_eq,
                bool rankdefic_Jac_ineq,
                double scal_neg_obj = 1.0);
  virtual ~SparseRajaEx2();

  /**
   * @brief Number of variables and constraints.
   */
  virtual bool get_prob_sizes(size_type& n, size_type& m);

  /**
   * @brief Get types and bounds on the variables.
   *
   * @param[in] n number of variables
   * @param[out] ixlow array with lower bounds (managed by Umpire)
   * @param[out] ixupp array with upper bounds (managed by Umpire)
   * @param[out] type array with the variable types (on host)
   */
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);

  /**
   * Get types and bounds corresponding to constraints. An equality constraint is specified
   * by setting the lower and upper bounds equal.
   *
   * @param[in] m Number of constraints
   * @param[out] iclow array with lower bounds (managed by Umpire)
   * @param[out] icupp array with upper bounds (managed by Umpire)
   * @param[out] type array with the variable types (on host)
   */
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  /**
   *  Returns the sizes and number of nonzeros of the sparse blocks
   *
   * @param[out] nx number of variables
   * @param[out] nnz_sparse_Jace number of nonzeros in the Jacobian of the equalities w.r.t.
   *                             sparse variables
   * @param[out] nnz_sparse_Jaci number of nonzeros in the Jacobian of the inequalities w.r.t.
   *                             sparse variables
   * @param[out] nnz_sparse_Hess_Lagr number of nonzeros in the (sparse) Hessian
   */
  virtual bool get_sparse_blocks_info(index_type& nx,
                                      index_type& nnz_sparse_Jace,
                                      index_type& nnz_sparse_Jaci,
                                      index_type& nnz_sparse_Hess_Lagr);

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
  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);

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
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons)
  {
    // return false so that HiOp will rely on the constraint evaluator defined below
    return false;
  }

  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);

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
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);

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
   * @param[in] nnzJacS number of nonzeros in the sparse Jacobian
   * @param[out] iJacS array of row indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] jJacS array of column indexes in the sparse Jacobian (managed by Umpire)
   * @param[out] MJacS array of nonzero values in the sparse Jacobian (managed by Umpire)
   */
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
    // return false so that HiOp will rely on the Jacobian evaluator defined below
    return false;
  }

  /// Similar to the above, but not used in this example.
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const size_type& nnzJacS,
                             index_type* iJacS,
                             index_type* jJacS,
                             double* MJacS);

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
   * @param[in] nnzHSS number of nonzeros in the (sparse) Hessian w.r.t. sparse variables
   * @param[out] iHSS array of row indexes in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] jHSS array of column indexes in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   * @param[out] MHSS array of nonzero values in the Hessian w.r.t. sparse variables
   *                  (managed by Umpire)
   */
  virtual bool eval_Hess_Lagr(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const size_type& nnzHSS,
                              index_type* iHSS,
                              index_type* jHSS,
                              double* MHSS);

  /**
   * Implementation of the primal starting point specification
   *
   * @param[in] n number of variables
   * @param[in] x0 the primal starting point(managed by Umpire)
   */
  virtual bool get_starting_point(const size_type& n, double* x0);

private:
  int n_vars_;
  int n_cons_;
  bool convex_obj_;
  bool rankdefic_eq_;
  bool rankdefic_ineq_;
  double scal_neg_obj_;

  std::string mem_space_;
};

#endif
