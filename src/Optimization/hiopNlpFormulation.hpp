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
 * @file hiopNlpFormulation.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_NLP_FORMULATION
#define HIOP_NLP_FORMULATION

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "hiopMatrix.hpp"
#include "hiopMatrixMDS.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#endif

#include "hiopNlpTransforms.hpp"

#include "hiopRunStats.hpp"
#include "hiopLogger.hpp"
#include "hiopOptions.hpp"

#include "hiopVectorInt.hpp"

#include <cstring>

namespace hiop
{

// some forward decls
class hiopDualsLsqUpdate;

/** Class for a general NlpFormulation with general constraints and bounds on the variables.
 * This class also  acts as a factory for linear algebra objects (derivative
 * matrices, KKT system) whose types are decided based on the hiopInterfaceXXX object passed in the
 * constructor.
 *
 * This formulation assumes that optimiz variables, rhs, and gradient are VECTORS: contiguous
 * double arrays for which only local part is accessed (no inter-process comm).
 * Derivatives are generic MATRICES, whose type depend on
 *    i.  the NLP formulation (sparse general or NLP with few dense constraints)
 *   ii. the interface provided (general sparse (not yet supported), mixed sparse-dense, or dense
 * constraints).
 * Exact matching of MATRICES and hiopInterface is to be done by specializations of this class.
 */
class hiopNlpFormulation
{
public:
  hiopNlpFormulation(hiopInterfaceBase& interface, const char* option_file = nullptr);
  virtual ~hiopNlpFormulation();

  virtual bool finalizeInitialization();
  virtual bool apply_scaling(hiopVector& c, hiopVector& d, hiopVector& gradf, hiopMatrix& Jac_c, hiopMatrix& Jac_d);

  /**
   * Wrappers for the interface calls.
   * Can be overridden for specialized formulations required by the algorithm.
   */
  virtual bool eval_f(hiopVector& x, bool new_x, double& f);
  virtual bool eval_grad_f(hiopVector& x, bool new_x, hiopVector& gradf);

  virtual bool eval_c(hiopVector& x, bool new_x, hiopVector& c);
  virtual bool eval_d(hiopVector& x, bool new_x, hiopVector& d);
  virtual bool eval_c_d(hiopVector& x, bool new_x, hiopVector& c, hiopVector& d);
  /* the implementation of the next two methods depends both on the interface and on the formulation */
  virtual bool eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c) = 0;
  virtual bool eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d) = 0;
  virtual bool eval_Jac_c_d(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);

protected:
  // calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d) = 0;

public:
  virtual bool eval_Hess_Lagr(const hiopVector& x,
                              bool new_x,
                              const double& obj_factor,
                              const hiopVector& lambda_eq,
                              const hiopVector& lambda_ineq,
                              bool new_lambdas,
                              hiopMatrix& Hess_L) = 0;
  /* starting point */
  virtual bool get_starting_point(hiopVector& x0,
                                  bool& duals_avail,
                                  hiopVector& zL0,
                                  hiopVector& zU0,
                                  hiopVector& yc0,
                                  hiopVector& yd0,
                                  bool& slacks_avail,
                                  hiopVector& d0);

  virtual bool get_warmstart_point(hiopVector& x0,
                                   hiopVector& zL0,
                                   hiopVector& zU0,
                                   hiopVector& yc0,
                                   hiopVector& yd0,
                                   hiopVector& d0,
                                   hiopVector& vl0,
                                   hiopVector& vu0);

  /* Allocates the LSQ duals update class. */
  virtual hiopDualsLsqUpdate* alloc_duals_lsq_updater() = 0;

  /** linear algebra factory */
  virtual hiopVector* alloc_primal_vec() const;
  virtual hiopVector* alloc_dual_eq_vec() const;
  virtual hiopVector* alloc_dual_ineq_vec() const;
  virtual hiopVector* alloc_dual_vec() const;
  /* the implementation of the next two methods depends both on the interface and on the formulation */
  virtual hiopMatrix* alloc_Jac_c() = 0;
  virtual hiopMatrix* alloc_Jac_d() = 0;
  virtual hiopMatrix* alloc_Jac_cons() = 0;
  virtual hiopMatrix* alloc_Hess_Lagr() = 0;

  virtual void user_callback_solution(hiopSolveStatus status,
                                      const hiopVector& x,
                                      hiopVector& z_L,
                                      hiopVector& z_U,
                                      hiopVector& c,
                                      hiopVector& d,
                                      hiopVector& y_c,
                                      hiopVector& y_d,
                                      double obj_value);

  virtual bool user_callback_iterate(int iter,
                                     double obj_value,
                                     double logbar_obj_value,
                                     const hiopVector& x,
                                     const hiopVector& z_L,
                                     const hiopVector& z_U,
                                     const hiopVector& s,  // the slack for inequalities
                                     const hiopVector& c,
                                     const hiopVector& d,
                                     const hiopVector& yc,
                                     const hiopVector& yd,
                                     double inf_pr,
                                     double inf_du,
                                     double onenorm_pr,
                                     double mu,
                                     double alpha_du,
                                     double alpha_pr,
                                     int ls_trials);

  virtual bool user_callback_full_iterate(hiopVector& x,
                                          hiopVector& z_L,
                                          hiopVector& z_U,
                                          hiopVector& y_c,
                                          hiopVector& y_d,
                                          hiopVector& s,
                                          hiopVector& v_L,
                                          hiopVector& v_U);

  virtual bool user_force_update(int iter,
                                 double& obj_value,
                                 hiopVector& x,
                                 hiopVector& z_L,
                                 hiopVector& z_U,
                                 hiopVector& c,
                                 hiopVector& d,
                                 hiopVector& y_c,
                                 hiopVector& y_d,
                                 double& mu,
                                 double& alpha_du,
                                 double& alpha_pr);

  /** const accessors */
  inline const hiopVector& get_xl() const { return *xl_; }
  inline const hiopVector& get_xu() const { return *xu_; }
  inline const hiopVector& get_ixl() const { return *ixl_; }
  inline const hiopVector& get_ixu() const { return *ixu_; }
  inline const hiopVector& get_dl() const { return *dl_; }
  inline const hiopVector& get_du() const { return *du_; }
  inline const hiopVector& get_idl() const { return *idl_; }
  inline const hiopVector& get_idu() const { return *idu_; }
  inline const hiopVector& get_crhs() const { return *c_rhs_; }

  inline hiopInterfaceBase::NonlinearityType* get_var_type() const { return vars_type_; }
  inline hiopInterfaceBase::NonlinearityType* get_cons_eq_type() const { return cons_eq_type_; }
  inline hiopInterfaceBase::NonlinearityType* get_cons_ineq_type() const { return cons_ineq_type_; }
  inline hiopInterfaceBase::NonlinearityType get_prob_type() const { return prob_type_; }

  /** const accessors */
  inline size_type n() const { return n_vars_; }
  inline size_type m() const { return n_cons_; }
  inline size_type m_eq() const { return n_cons_eq_; }
  inline size_type m_ineq() const { return n_cons_ineq_; }
  inline size_type n_low() const { return n_bnds_low_; }
  inline size_type n_upp() const { return n_bnds_upp_; }
  inline size_type m_ineq_low() const { return n_ineq_low_; }
  inline size_type m_ineq_upp() const { return n_ineq_upp_; }
  inline size_type n_complem() const { return m_ineq_low() + m_ineq_upp() + n_low() + n_upp(); }

  inline size_type n_local() const { return xl_->get_local_size(); }
  inline size_type n_low_local() const { return n_bnds_low_local_; }
  inline size_type n_upp_local() const { return n_bnds_upp_local_; }

  /* methods for transforming the internal objects to corresponding user objects */
  inline double user_obj(double hiop_f) { return nlp_transformations_.apply_inv_to_obj(hiop_f); }
  inline void user_x(hiopVector& hiop_x, double* user_x)
  {
    // double *hiop_xa = hiop_x.local_data();
    hiopVector* x = nlp_transformations_.apply_inv_to_x(hiop_x, /*new_x=*/true);
    // memcpy(user_x, user_xa, hiop_x.get_local_size()*sizeof(double));
    memcpy(user_x, x->local_data(), nlp_transformations_.n_post_local() * sizeof(double));
  }

  /* copies/unpacks duals of the bounds and of constraints from 'it' to the three arrays */
  void get_dual_solutions(const hiopIterate& it, double* zl_a, double* zu_a, double* lambda_a);

  /// @brief return the scaling fact for objective
  double get_obj_scale() const;

  /// @brief adjust variable/constraint bounds according to the given iteration.
  void adjust_bounds(const hiopIterate& it);

  /// @brief reset variable/constraint bounds in the elastic_mode
  void reset_bounds(double bound_relax_perturb);

  /* outputing and debug-related functionality*/
  hiopLogger* log;
  hiopRunStats runStats;
  hiopOptions* options;
  // prints a summary of the problem
  virtual void print(FILE* f = NULL, const char* msg = NULL, int rank = -1) const;
#ifdef HIOP_USE_MPI
  inline MPI_Comm get_comm() const { return comm_; }
  inline int get_rank() const { return rank_; }
  inline int get_num_ranks() const { return num_ranks_; }
  inline index_type* getVecDistInfo() { return vec_distrib_; }
#endif
protected:
  /* Preprocess bounds in a form supported by the NLP formulation. Returns counts of
   * the variables with lower, upper, and lower and lower bounds, as well of the fixed
   * variables.
   */
  virtual bool process_bounds(size_type& n_bnds_low, size_type& n_bnds_upp, size_type& n_bnds_lu, size_type& nfixed_vars);
  /* Preprocess constraints in a form supported the NLP formulation. */
  virtual bool process_constraints();

protected:
#ifdef HIOP_USE_MPI
  MPI_Comm comm_;
  int rank_;
  int num_ranks_;
  bool mpi_init_called;
#endif

  /* Problem data and dimensions */
  size_type n_vars_;
  size_type n_cons_;
  size_type n_cons_eq_;
  size_type n_cons_ineq_;
  size_type n_bnds_low_;
  size_type n_bnds_low_local_;
  size_type n_bnds_upp_;
  size_type n_bnds_upp_local_;
  size_type n_ineq_low_;
  size_type n_ineq_upp_;
  size_type n_bnds_lu_;
  size_type n_ineq_lu_;
  hiopVector *xl_, *xu_, *ixu_, *ixl_;              // these will/can be global, memory distributed
  hiopInterfaceBase::NonlinearityType* vars_type_;  // C array containing the types for local vars

  hiopVector* c_rhs_;  // local
  hiopInterfaceBase::NonlinearityType* cons_eq_type_;

  hiopVector *dl_, *du_, *idl_, *idu_;  // these will be local
  hiopInterfaceBase::NonlinearityType* cons_ineq_type_;

  /**
   * Flag to indicate whether problem is LP, QP or NLP
   */
  hiopInterfaceBase::NonlinearityType prob_type_;

  // flag to indicate whether f/grad/con/Jac/Hes has been evaluated once
  bool nlp_evaluated_;

  // keep track of the constraints indexes in the original, user's formulation
  hiopVectorInt *cons_eq_mapping_, *cons_ineq_mapping_;

  // options for which this class was setup
  std::string strFixedVars_;  //"none", "fixed", "relax"
  double dFixedVarsTol_;

  /**
   * @brief Internal NLP transformations that supports fixing and relaxing variables as well as
   * problem rescalings.
   */
  hiopNlpTransformations nlp_transformations_;

  // internal NLP transformations (currently gradient scaling implemented)
  hiopNLPObjGradScaling* nlp_scaling_;

  /// @brief internal NLP transformations that relaxes the bounds
  hiopBoundsRelaxer* relax_bounds_;

#ifdef HIOP_USE_MPI
  // inter-process distribution of vectors
  index_type* vec_distrib_;
#endif

  /* User provided interface */
  hiopInterfaceBase& interface_base;

  /**
   * Flag to indicate whether to use evaluate all constraints once or separately for equalities
   * or inequalities. Possible values
   * -1 : not initialized/not decided
   *  0 : separately
   *  1 : at once
   */
  int cons_eval_type_;

  /**
   * Internal buffer for constraints. Used only when constraints and Jacobian are evaluated at
   * once (cons_eval_type_==1), otherwise NULL.
   */
  hiopVector* cons_body_;

  /**
   * Internal buffer for the Jacobian. Used only when constraints and Jacobian are evaluated at
   * once (cons_eval_type_==1), otherwise NULL.
   */
  hiopMatrix* cons_Jac_;

  /**
   * Internal buffer for the multipliers of the constraints use to copy the multipliers of eq. and
   * ineq. into and to return it to the user via @user_callback_solution and @user_callback_iterate
   */
  hiopVector* cons_lambdas_;

  /**
   * Internal buffers. These vectors are used in unscaling the corresponding values.
   */
  hiopVector* temp_eq_;
  hiopVector* temp_ineq_;
  hiopVector* temp_x_;

private:
  hiopNlpFormulation(const hiopNlpFormulation& s)
      : nlp_transformations_(this),
        interface_base(s.interface_base)

  {};
};

/* *************************************************************************
 * Class is for NLPs that has a small number of general/dense constraints *
 * Splits the constraints in ineq and eq.
 * *************************************************************************
 */
class hiopNlpDenseConstraints : public hiopNlpFormulation
{
public:
  hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface, const char* option_file = nullptr);
  virtual ~hiopNlpDenseConstraints();

  virtual bool finalizeInitialization();

  virtual bool eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c);
  virtual bool eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d);
  /* specialized evals to avoid overhead of dynamic cast. Generic variants available above. */
  virtual bool eval_Jac_c(hiopVector& x, bool new_x, double* Jac_c);
  virtual bool eval_Jac_d(hiopVector& x, bool new_x, double* Jac_d);

protected:
  // calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of
  // hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);

public:
  virtual bool eval_Hess_Lagr(const hiopVector& x,
                              bool new_x,
                              const double& obj_factor,
                              const hiopVector& lambda_eq,
                              const hiopVector& lambda_ineq,
                              bool new_lambda,
                              hiopMatrix& Hess_L)
  {
    // silently ignore the call since we're in the quasi-Newton case
    return true;
  }

  /* Allocates the LSQ duals update class. */
  virtual hiopDualsLsqUpdate* alloc_duals_lsq_updater();

  virtual hiopMatrixDense* alloc_Jac_c();
  virtual hiopMatrixDense* alloc_Jac_d();
  virtual hiopMatrixDense* alloc_Jac_cons();
  // returns HessianDiagPlusRowRank which (fakely) inherits from hiopMatrix
  virtual hiopMatrix* alloc_Hess_Lagr();

  /* this is in general for a dense matrix with n_vars cols and a small number of
   * 'nrows' rows. The second argument indicates how much total memory should the
   * matrix (pre)allocate.
   */
  virtual hiopMatrixDense* alloc_multivector_primal(int nrows, int max_rows = -1) const;

private:
  /* interface implemented and provided by the user */
  hiopInterfaceDenseConstraints& interface;
};

/* *************************************************************************
 * Class is for general NLPs that have mixed sparse-dense (MDS) derivatives
 * blocks.
 * *************************************************************************
 */
class hiopNlpMDS : public hiopNlpFormulation
{
public:
  hiopNlpMDS(hiopInterfaceMDS& interface_, const char* option_file = nullptr)
      : hiopNlpFormulation(interface_, option_file),
        interface(interface_)
  {
    buf_lambda_ = LinearAlgebraFactory::create_vector(options->GetString("mem_space"), 0);
  }
  virtual ~hiopNlpMDS() { delete buf_lambda_; }

  virtual bool finalizeInitialization();

  virtual bool eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c);
  virtual bool eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d);

protected:
  // calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);

public:
  virtual bool eval_Hess_Lagr(const hiopVector& x,
                              bool new_x,
                              const double& obj_factor,
                              const hiopVector& lambda_eq,
                              const hiopVector& lambda_ineq,
                              bool new_lambdas,
                              hiopMatrix& Hess_L);

  /* Allocates the LSQ duals update class. */
  virtual hiopDualsLsqUpdate* alloc_duals_lsq_updater();

  virtual hiopMatrix* alloc_Jac_c()
  {
    assert(n_vars_ == nx_sparse + nx_dense);
    return new hiopMatrixMDS(n_cons_eq_, nx_sparse, nx_dense, nnz_sparse_Jaceq, options->GetString("mem_space"));
  }
  virtual hiopMatrix* alloc_Jac_d()
  {
    assert(n_vars_ == nx_sparse + nx_dense);
    return new hiopMatrixMDS(n_cons_ineq_, nx_sparse, nx_dense, nnz_sparse_Jacineq, options->GetString("mem_space"));
  }
  virtual hiopMatrix* alloc_Jac_cons()
  {
    assert(n_vars_ == nx_sparse + nx_dense);
    return new hiopMatrixMDS(n_cons_,
                             nx_sparse,
                             nx_dense,
                             nnz_sparse_Jaceq + nnz_sparse_Jacineq,
                             options->GetString("mem_space"));
  }
  virtual hiopMatrix* alloc_Hess_Lagr()
  {
    assert(0 == nnz_sparse_Hess_Lagr_SD);
    return new hiopMatrixSymBlockDiagMDS(nx_sparse, nx_dense, nnz_sparse_Hess_Lagr_SS, options->GetString("mem_space"));
  }

  /** const accessors */
  virtual size_type nx_sp() const { return nx_sparse; }
  virtual size_type nx_de() const { return nx_dense; }
  inline int get_nnz_sp_Jaceq() const { return nnz_sparse_Jaceq; }
  inline int get_nnz_sp_Jacineq() const { return nnz_sparse_Jacineq; }
  inline int get_nnz_sp_Hess_Lagr_SS() const { return nnz_sparse_Hess_Lagr_SS; }
  inline int get_nnz_sp_Hess_Lagr_SD() const { return nnz_sparse_Hess_Lagr_SD; }

private:
  hiopInterfaceMDS& interface;
  int nx_sparse, nx_dense;
  int nnz_sparse_Jaceq, nnz_sparse_Jacineq;
  int nnz_sparse_Hess_Lagr_SS, nnz_sparse_Hess_Lagr_SD;

  hiopVector* buf_lambda_;
};

/* *************************************************************************
 * Class is for general NLPs that have sparse derivatives blocks.
 * *************************************************************************
 */
class hiopNlpSparse : public hiopNlpFormulation
{
public:
  hiopNlpSparse(hiopInterfaceSparse& interface_, const char* option_file = nullptr)
      : hiopNlpFormulation(interface_, option_file),
        interface(interface_),
        num_jac_eval_{0},
        num_hess_eval_{0}
  {
    buf_lambda_ = LinearAlgebraFactory::create_vector(options->GetString("mem_space"), 0);
  }
  virtual ~hiopNlpSparse() { delete buf_lambda_; }

  virtual bool finalizeInitialization();

  virtual bool eval_Jac_c(hiopVector& x, bool new_x, hiopMatrix& Jac_c);
  virtual bool eval_Jac_d(hiopVector& x, bool new_x, hiopMatrix& Jac_d);

protected:
  // calls specific hiopInterfaceXXX::eval_Jac_cons and deals with specializations of hiopMatrix arguments
  virtual bool eval_Jac_c_d_interface_impl(hiopVector& x, bool new_x, hiopMatrix& Jac_c, hiopMatrix& Jac_d);

public:
  virtual bool eval_Hess_Lagr(const hiopVector& x,
                              bool new_x,
                              const double& obj_factor,
                              const hiopVector& lambda_eq,
                              const hiopVector& lambda_ineq,
                              bool new_lambdas,
                              hiopMatrix& Hess_L);
  /* Allocates the LSQ duals update class. */
  virtual hiopDualsLsqUpdate* alloc_duals_lsq_updater();

  virtual hiopMatrix* alloc_Jac_c()
  {
    return LinearAlgebraFactory::create_matrix_sparse(options->GetString("mem_space"),
                                                      n_cons_eq_,
                                                      n_vars_,
                                                      nnz_sparse_Jaceq_);
    // return new hiopMatrixSparseTriplet(n_cons_eq_, n_vars_, nnz_sparse_Jaceq_);
  }
  virtual hiopMatrix* alloc_Jac_d()
  {
    return LinearAlgebraFactory::create_matrix_sparse(options->GetString("mem_space"),
                                                      n_cons_ineq_,
                                                      n_vars_,
                                                      nnz_sparse_Jacineq_);
    // return new hiopMatrixSparseTriplet(n_cons_ineq_, n_vars_, nnz_sparse_Jacineq_);
  }
  virtual hiopMatrix* alloc_Jac_cons()
  {
    return LinearAlgebraFactory::create_matrix_sparse(options->GetString("mem_space"),
                                                      n_cons_,
                                                      n_vars_,
                                                      nnz_sparse_Jaceq_ + nnz_sparse_Jacineq_);
    // return new hiopMatrixSparseTriplet(n_cons_, n_vars_, nnz_sparse_Jaceq_ + nnz_sparse_Jacineq_);
  }
  virtual hiopMatrix* alloc_Hess_Lagr()
  {
    return LinearAlgebraFactory::create_matrix_sym_sparse(options->GetString("mem_space"), n_vars_, nnz_sparse_Hess_Lagr_);
    // return new hiopMatrixSymSparseTriplet(n_vars_, nnz_sparse_Hess_Lagr_);
  }
  virtual size_type nx() const { return n_vars_; }

  // not inherited from NlpFormulation

  /**
   * @brief Allocates a non-MPI vector with size given by the size of primal plus dual spaces.
   * The dual space corresponds to  both equality and inequality constraints.
   */
  virtual hiopVector* alloc_primal_dual_vec() const
  {
    assert(n_cons_ == n_cons_eq_ + n_cons_ineq_);
    return LinearAlgebraFactory::create_vector(options->GetString("mem_space"), n_vars_ + n_cons_);
  }

  /** const accessors */
  inline int get_nnz_Jaceq() const { return nnz_sparse_Jaceq_; }
  inline int get_nnz_Jacineq() const { return nnz_sparse_Jacineq_; }
  inline int get_nnz_Hess_Lagr() const { return nnz_sparse_Hess_Lagr_; }

protected:
  hiopInterfaceSparse& interface;
  int nnz_sparse_Jaceq_;
  int nnz_sparse_Jacineq_;
  int nnz_sparse_Hess_Lagr_;
  int num_jac_eval_;
  int num_hess_eval_;

  hiopVector* buf_lambda_;
};

/**
 * Specialized NLP formulation class that poses equalities as relaxed two-sided
 * inequalities
 */
class hiopNlpSparseIneq : public hiopNlpSparse
{
public:
  hiopNlpSparseIneq(hiopInterfaceSparse& interface_, const char* option_file = nullptr)
      : hiopNlpSparse(interface_, option_file),
        n_cons_eq_origNLP_(0),
        eq_relax_value_(1e-8)
  {}
  virtual ~hiopNlpSparseIneq() {}
  /* Preprocess constraints so that equalities are posed as relaxed two-sided inequalities. */
  virtual bool process_constraints();

  /* Perform initialization and preprocessing. */
  virtual bool finalizeInitialization();

protected:
  /* Number of equalities in the original NLP formulation. */
  size_type n_cons_eq_origNLP_;

  /* Maximum violation of the equalities relative to the magnitude of the right-hand side. */
  double eq_relax_value_;
};
}  // namespace hiop
#endif
