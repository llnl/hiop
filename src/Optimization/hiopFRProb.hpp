// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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
 * @file hiopFRProb.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_FR_INTERFACE
#define HIOP_FR_INTERFACE

#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

namespace hiop
{

/** Specialized interface for feasibility restoration problem with sparse blocks in the Jacobian and Hessian.
 *
 * More specifically, this interface is for specifying optimization problem:
 *
 * min f(x)
 *  s.t. g(x) <= or = 0, lb<=x<=ub
 *
 * such that Jacobian w.r.t. x and Hessian of the Lagrangian w.r.t. x are sparse
 *
 * @note this interface is 'local' in the sense that data is not assumed to be
 * distributed across MPI ranks ('get_vecdistrib_info' should return 'false').
 * Acceleration can be however obtained using OpenMP and CUDA via Raja
 * abstraction layer that HiOp uses and via linear solver.
 *
 */
class hiopFRProbSparse : public hiopInterfaceSparse
{
public:
  hiopFRProbSparse(hiopAlgFilterIPMBase& solver_base);
  virtual ~hiopFRProbSparse();
  virtual bool get_MPI_comm(MPI_Comm& comm_out);

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_prob_info(NonlinearityType& type);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);
  virtual bool get_sparse_blocks_info(int& nx, int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq, int& nnz_sparse_Hess_Lagr);

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             const int& nnzJacS,
                             int* iJacS,
                             int* jJacS,
                             double* MJacS);
  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const double* x,
                             bool new_x,
                             const int& nnzJacS,
                             int* iJacS,
                             int* jJacS,
                             double* MJacS);

  virtual bool get_warmstart_point(const size_type& n,
                                   const size_type& m,
                                   double* x0,
                                   double* z_bndL0,
                                   double* z_bndU0,
                                   double* lambda0,
                                   double* ineq_slack,
                                   double* vl0,
                                   double* vu0);

  virtual bool eval_Hess_Lagr(const size_type& n,
                              const size_type& m,
                              const double* x,
                              bool new_x,
                              const double& obj_factor,
                              const double* lambda,
                              bool new_lambda,
                              const int& nnzHSS,
                              int* iHSS,
                              int* jHSS,
                              double* MHSS);

  virtual bool iterate_callback(int iter,
                                double obj_value,
                                double logbar_obj_value,
                                int n,
                                const double* x,
                                const double* z_L,
                                const double* z_U,
                                int m_ineq,
                                const double* s,
                                int m,
                                const double* g,
                                const double* lambda,
                                double inf_pr,
                                double inf_du,
                                double onenorm_pr_,
                                double mu,
                                double alpha_du,
                                double alpha_pr,
                                int ls_trials);

  virtual bool force_update_x(const int n, double* x);

  virtual const hiopVector& get_fr_sol_x() const { return *last_x_; }
  virtual const hiopVector& get_fr_sol_d() const { return *last_d_; }

private:
  size_type n_;
  size_type m_;

  size_type n_x_;
  size_type m_eq_;
  size_type m_ineq_;

  size_type nnz_Jac_c_;
  size_type nnz_Jac_d_;
  size_type nnz_Hess_Lag_;

  hiopAlgFilterIPMBase& solver_base_;
  hiopNlpSparse* nlp_base_;

  hiopVector* x_ref_;
  hiopVector* DR_;
  hiopVector* wrk_x_;
  hiopVector* wrk_c_;
  hiopVector* wrk_d_;
  hiopVector* wrk_eq_;
  hiopVector* wrk_ineq_;
  hiopVector* wrk_cbody_;
  hiopVector* wrk_dbody_;
  hiopVector* wrk_primal_;  // [x pe ne pi ni]
  hiopVector* wrk_dual_;    // [c d]

  hiopMatrixSparse* Jac_cd_;
  hiopMatrixSparse* Hess_cd_;

  hiopVector* last_x_;
  hiopVector* last_d_;

  double zeta_;
  double theta_ref_;
  double nrmInf_feas_ref_;
  double mu_;
  double rho_;
  double obj_base_;

  int pe_st_;  // the 1st index of pe in the full primal space
  int ne_st_;  // the 1st index of ne in the full primal space
  int pi_st_;  // the 1st index of pi in the full primal space
  int ni_st_;  // the 1st index of ni in the full primal space
};

/** Specialized interface for feasibility restoration problem with MDS blocks in the Jacobian and Hessian.
 *
 * More specifically, this interface is for specifying optimization problem:
 *
 * min f(x)
 *  s.t. g(x) <= or = 0, lb<=x<=ub
 *
 * such that Jacobian w.r.t. x and Hessian of the Lagrangian w.r.t. x are MDS
 *
 */
class hiopFRProbMDS : public hiopInterfaceMDS
{
public:
  hiopFRProbMDS(hiopAlgFilterIPMBase& solver_base);
  virtual ~hiopFRProbMDS();
  virtual bool get_MPI_comm(MPI_Comm& comm_out);

  virtual bool get_sparse_dense_blocks_info(int& nx_sparse,
                                            int& nx_dense,
                                            int& nnz_sparse_Jaceq,
                                            int& nnz_sparse_Jacineq,
                                            int& nnz_sparse_Hess_Lagr_SS,
                                            int& nnz_sparse_Hess_Lagr_SD);

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

  virtual bool eval_Hess_Lagr(const size_type& n,
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

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_prob_info(NonlinearityType& type);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons);
  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);

  virtual bool get_warmstart_point(const size_type& n,
                                   const size_type& m,
                                   double* x0,
                                   double* z_bndL0,
                                   double* z_bndU0,
                                   double* lambda0,
                                   double* ineq_slack,
                                   double* vl0,
                                   double* vu0);

  virtual bool iterate_callback(int iter,
                                double obj_value,
                                double logbar_obj_value,
                                int n,
                                const double* x,
                                const double* z_L,
                                const double* z_U,
                                int m_ineq,
                                const double* s,
                                int m,
                                const double* g,
                                const double* lambda,
                                double inf_pr,
                                double inf_du,
                                double onenorm_pr_,
                                double mu,
                                double alpha_du,
                                double alpha_pr,
                                int ls_trials);

  virtual bool force_update_x(const int n, double* x);

  virtual const hiopVector& get_fr_sol_x() const { return *last_x_; }
  virtual const hiopVector& get_fr_sol_d() const { return *last_d_; }

private:
  size_type n_;
  size_type n_sp_;
  size_type n_de_;
  size_type m_;

  size_type n_x_;
  size_type n_x_sp_;
  size_type n_x_de_;
  size_type m_eq_;
  size_type m_ineq_;

  size_type nnz_sp_Jac_c_;
  size_type nnz_sp_Jac_d_;
  size_type nnz_sp_Hess_Lagr_SS_;
  size_type nnz_sp_Hess_Lagr_SD_;

  hiopAlgFilterIPMBase& solver_base_;
  hiopNlpMDS* nlp_base_;

  hiopVector* x_ref_;
  hiopVector* DR_;
  hiopVector* wrk_x_;
  hiopVector* wrk_c_;
  hiopVector* wrk_d_;
  hiopVector* wrk_eq_;
  hiopVector* wrk_ineq_;
  hiopVector* wrk_cbody_;
  hiopVector* wrk_dbody_;
  hiopVector* wrk_primal_;  // [xsp pe ne pi ni xde]
  hiopVector* wrk_dual_;    // [c d]

  hiopVector* wrk_x_sp_;  // the sparse part of x, xsp
  hiopVector* wrk_x_de_;  // the dense part of x, xde

  hiopMatrixMDS* Jac_cd_;
  hiopMatrixSymBlockDiagMDS* Hess_cd_;

  hiopVector* last_x_;
  hiopVector* last_d_;

  double zeta_;
  double theta_ref_;
  double nrmInf_feas_ref_;
  double mu_;
  double rho_;
  double obj_base_;

  int x_sp_st_;  // the 1st index of x_sp in the full primal space
  int pe_st_;    // the 1st index of pe in the full primal space
  int ne_st_;    // the 1st index of ne in the full primal space
  int pi_st_;    // the 1st index of pi in the full primal space
  int ni_st_;    // the 1st index of ni in the full primal space
  int x_de_st_;  // the 1st index of x_de in the full primal space
};

/** Specialized interface for feasibility restoration problem with dense blocks in the Jacobian and Hessian.
 *
 * More specifically, this interface is for specifying optimization problem:
 *
 * min f(x)
 *  s.t. g(x) <= or = 0, lb<=x<=ub
 *
 * such that Jacobian w.r.t. x and Hessian of the Lagrangian w.r.t. x are dense
 *
 * @note this interface is 'local' in the sense that data is not assumed to be
 * distributed across MPI ranks ('get_vecdistrib_info' should return 'false').
 * Acceleration can be however obtained using OpenMP and CUDA via Raja
 * abstraction layer that HiOp uses and via linear solver.
 *
 */
class hiopFRProbDense : public hiopInterfaceDenseConstraints
{
public:
  hiopFRProbDense(hiopAlgFilterIPMBase& solver_base);
  virtual ~hiopFRProbDense();
  virtual bool get_MPI_comm(MPI_Comm& comm_out);
  virtual bool get_vecdistrib_info(size_type global_n, index_type* cols);

  virtual bool eval_Jac_cons(const size_type& n,
                             const size_type& m,
                             const size_type& num_cons,
                             const index_type* idx_cons,
                             const double* x,
                             bool new_x,
                             double* Jac)
  {
    return false;
  }

  virtual bool eval_Jac_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* Jac);

  virtual bool get_prob_sizes(size_type& n, size_type& m);
  virtual bool get_prob_info(NonlinearityType& type);
  virtual bool get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type);
  virtual bool get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type);

  virtual bool eval_f(const size_type& n, const double* x, bool new_x, double& obj_value);
  virtual bool eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf);
  virtual bool eval_cons(const size_type& n,
                         const size_type& m,
                         const size_type& num_cons,
                         const index_type* idx_cons,
                         const double* x,
                         bool new_x,
                         double* cons)
  {
    return false;
  }
  virtual bool eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons);
  virtual bool get_warmstart_point(const size_type& n,
                                   const size_type& m,
                                   double* x0,
                                   double* z_bndL0,
                                   double* z_bndU0,
                                   double* lambda0,
                                   double* ineq_slack,
                                   double* vl0,
                                   double* vu0);

  virtual bool iterate_callback(int iter,
                                double obj_value,
                                double logbar_obj_value,
                                int n,
                                const double* x,
                                const double* z_L,
                                const double* z_U,
                                int m_ineq,
                                const double* s,
                                int m,
                                const double* g,
                                const double* lambda,
                                double inf_pr,
                                double inf_du,
                                double onenorm_pr_,
                                double mu,
                                double alpha_du,
                                double alpha_pr,
                                int ls_trials);

  virtual bool force_update_x(const int n, double* x);

  virtual const hiopVector& get_fr_sol_x() const { return *last_x_; }
  virtual const hiopVector& get_fr_sol_d() const { return *last_d_; }

private:
  size_type n_;
  size_type m_;

  size_type n_x_;
  size_type m_eq_;
  size_type m_ineq_;

  hiopAlgFilterIPMBase& solver_base_;
  hiopNlpDenseConstraints* nlp_base_;

  hiopVector* x_ref_;
  hiopVector* DR_;
  hiopVector* wrk_x_;
  hiopVector* wrk_c_;
  hiopVector* wrk_d_;
  hiopVector* wrk_eq_;
  hiopVector* wrk_ineq_;
  hiopVector* wrk_cbody_;
  hiopVector* wrk_dbody_;
  hiopVector* wrk_primal_;  // [xde pe ne pi ni]
  hiopVector* wrk_dual_;    // [c d]

  hiopMatrixDense* Jac_cd_;

  hiopVector* last_x_;
  hiopVector* last_d_;

  double zeta_;
  double theta_ref_;
  double nrmInf_feas_ref_;
  double mu_;
  double rho_;
  double obj_base_;

  index_type* col_partition_;
  MPI_Comm comm_;
  size_type rank_;
  size_type comm_size_;
#ifdef HIOP_USE_MPI
  index_type* vec_distrib_base_;
#endif

  int pe_st_;  // the 1st index of pe in the full primal space
  int ne_st_;  // the 1st index of ne in the full primal space
  int pi_st_;  // the 1st index of pi in the full primal space
  int ni_st_;  // the 1st index of ni in the full primal space
};

}  // namespace hiop
#endif
