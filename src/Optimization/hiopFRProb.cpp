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
 * @file hiopFRProb.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopFRProb.hpp"

#include "hiopVector.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>
#include <math.h>

namespace hiop
{

/*
 *  Specialized interface for feasibility restoration problem with sparse blocks in the Jacobian and Hessian.
 */
hiopFRProbSparse::hiopFRProbSparse(hiopAlgFilterIPMBase& solver_base)
    : solver_base_(solver_base),
      last_x_{nullptr},
      last_d_{nullptr}
{
  nlp_base_ = dynamic_cast<hiopNlpSparse*>(solver_base.get_nlp());
  n_x_ = nlp_base_->n();
  m_eq_ = nlp_base_->m_eq();
  m_ineq_ = nlp_base_->m_ineq();

  n_ = n_x_ + 2 * m_eq_ + 2 * m_ineq_;
  m_ = m_eq_ + m_ineq_;

  pe_st_ = n_x_;
  ne_st_ = pe_st_ + m_eq_;
  pi_st_ = ne_st_ + m_eq_;
  ni_st_ = pi_st_ + m_ineq_;

  x_ref_ = solver_base.get_it_curr()->get_x();

  // build vector VR
  DR_ = x_ref_->new_copy();
  DR_->component_abs();
  DR_->invert();
  DR_->component_min(1.0);

  wrk_x_ = x_ref_->alloc_clone();
  wrk_c_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_d_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_eq_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_ineq_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_cbody_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_dbody_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_primal_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_);
  wrk_dual_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_);
  last_x_ = x_ref_->alloc_clone();
  last_d_ = wrk_d_->alloc_clone();

  // nnz for sparse matrices;
  nnz_Jac_c_ = nlp_base_->get_nnz_Jaceq() + 2 * m_eq_;
  nnz_Jac_d_ = nlp_base_->get_nnz_Jacineq() + 2 * m_ineq_;

  // not sure if Hess has diagonal terms, compute nnz_hess here
  // assuming hess is in upper_triangular form
  hiopMatrixSparse* Hess_base = dynamic_cast<hiopMatrixSparse*>(solver_base_.get_Hess_Lagr());
  nnz_Hess_Lag_ = n_x_ + Hess_base->numberOfOffDiagNonzeros();

  Jac_cd_ = LinearAlgebraFactory::create_matrix_sparse(nlp_base_->options->GetString("mem_space"),
                                                       m_,
                                                       n_,
                                                       nnz_Jac_c_ + nnz_Jac_d_);
  Hess_cd_ = LinearAlgebraFactory::create_matrix_sym_sparse(nlp_base_->options->GetString("mem_space"), n_, nnz_Hess_Lag_);

  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  theta_ref_ = solver_base_.get_resid()->get_theta();  // at current point, i.e., reference point
  nrmInf_feas_ref_ = solver_base_.get_resid()->get_nrmInf_bar_feasib();
  mu_ = solver_base.get_mu();
  mu_ = std::max(mu_, nrmInf_feas_ref_);

  zeta_ = std::sqrt(mu_);
  rho_ = 1000;  // FIXME: make this as an user option
}

hiopFRProbSparse::~hiopFRProbSparse()
{
  delete wrk_x_;
  delete wrk_c_;
  delete wrk_d_;
  delete wrk_eq_;
  delete wrk_ineq_;
  delete wrk_cbody_;
  delete wrk_dbody_;
  delete wrk_primal_;
  delete wrk_dual_;
  delete DR_;
  delete Jac_cd_;
  delete Hess_cd_;
  if(last_x_) {
    delete last_x_;
  }
  if(last_d_) {
    delete last_d_;
  }
}

bool hiopFRProbSparse::get_MPI_comm(MPI_Comm& comm_out)
{
#ifdef HIOP_USE_MPI
  comm_out = nlp_base_->get_comm();
#else
  comm_out = MPI_COMM_WORLD;
#endif
  return true;
}

bool hiopFRProbSparse::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_;
  m = m_;
  return true;
}

bool hiopFRProbSparse::get_prob_info(NonlinearityType& type)
{
  // the objective of FR problem is quadratic. constraint c(x) depends on the origianl problem
  if(nlp_base_->get_prob_type() != hiopNonlinear) {
    type = hiopQuadratic;
  }
  return true;
}

bool hiopFRProbSparse::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_);

  const hiopVector& xl = nlp_base_->get_xl();
  const hiopVector& xu = nlp_base_->get_xu();
  const NonlinearityType* var_type = nlp_base_->get_var_type();

  // x, p and n
  wrk_primal_->setToConstant(0.0);
  xl.copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(xlow);

  wrk_primal_->setToConstant(1e+20);
  xu.copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(xupp);

  wrk_primal_->set_array_from_to(type, 0, n_x_, var_type, 0);
  wrk_primal_->set_array_from_to(type, n_x_, n_, hiopLinear);

  return true;
}

bool hiopFRProbSparse::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == m_);
  assert(m_eq_ + m_ineq_ == m_);
  const hiopVector& crhs = nlp_base_->get_crhs();
  const hiopVector& dl = nlp_base_->get_dl();
  const hiopVector& du = nlp_base_->get_du();
  const NonlinearityType* cons_eq_type = nlp_base_->get_cons_eq_type();
  const NonlinearityType* cons_ineq_type = nlp_base_->get_cons_ineq_type();

  wrk_dual_->setToConstant(0.0);

  // assemble wrk_dual_ = [crhs; dl] for lower bounds
  crhs.copyToStarting(*wrk_dual_, 0);
  dl.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(clow);

  // assemble wrk_dual_ = [crhs; du] for upper bounds
  du.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(cupp);

  wrk_dual_->set_array_from_to(type, 0, m_eq_, cons_eq_type, 0);
  wrk_dual_->set_array_from_to(type, m_eq_, m_, cons_ineq_type, 0);

  return true;
}

bool hiopFRProbSparse::get_sparse_blocks_info(int& nx,
                                              int& nnz_sparse_Jaceq,
                                              int& nnz_sparse_Jacineq,
                                              int& nnz_sparse_Hess_Lagr)
{
  nx = n_;
  nnz_sparse_Jaceq = nnz_Jac_c_;
  nnz_sparse_Jacineq = nnz_Jac_d_;
  nnz_sparse_Hess_Lagr = nnz_Hess_Lag_;
  return true;
}

bool hiopFRProbSparse::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_);
  obj_value = 0.;

  wrk_primal_->copy_from_starting_at(x, 0, n_);  // [x pe ne pi ni]
  wrk_x_->copy_from_starting_at(x, 0, n_x_);     // [x]

  // rho*sum(p+n)
  obj_value += rho_ * (wrk_primal_->sum_local() - wrk_x_->sum_local());

  // zeta/2*[DR*(x-x_ref)]^2
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  double wrk_db = wrk_x_->twonorm();

  obj_value += 0.5 * zeta_ * wrk_db * wrk_db;

  // keep a copy of the original objective value
  nlp_base_->eval_f(*wrk_x_, new_x, obj_base_);

  return true;
}

bool hiopFRProbSparse::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_);

  // p and n
  wrk_primal_->setToConstant(rho_);

  // x: zeta*DR^2*(x-x_ref)
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->scale(zeta_);
  wrk_x_->copyToStarting(*wrk_primal_, 0);

  wrk_primal_->copyTo(gradf);

  return true;
}

bool hiopFRProbSparse::eval_cons(const size_type& n,
                                 const size_type& m,
                                 const size_type& num_cons,
                                 const index_type* idx_cons,
                                 const double* x,
                                 bool new_x,
                                 double* cons)
{
  return false;
}

bool hiopFRProbSparse::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_);
  assert(m == m_);

  // evaluate c and d
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  nlp_base_->eval_c_d(*wrk_x_, new_x, *wrk_c_, *wrk_d_);

  // compute FR equality constratint body c-pe+ne
  wrk_eq_->copy_from_starting_at(x, pe_st_, m_eq_);  // pe
  wrk_c_->axpy(-1.0, *wrk_eq_);
  wrk_eq_->copy_from_starting_at(x, ne_st_, m_eq_);  // ne
  wrk_c_->axpy(1.0, *wrk_eq_);

  // compute FR inequality constratint body d-pi+ni
  wrk_ineq_->copy_from_starting_at(x, pi_st_, m_ineq_);  // pi
  wrk_d_->axpy(-1.0, *wrk_ineq_);
  wrk_ineq_->copy_from_starting_at(x, ni_st_, m_ineq_);  // ni
  wrk_d_->axpy(1.0, *wrk_ineq_);

  // assemble the full vector
  wrk_c_->copyToStarting(*wrk_dual_, 0);
  wrk_d_->copyToStarting(*wrk_dual_, m_eq_);

  wrk_dual_->copyTo(cons);

  return true;
}

bool hiopFRProbSparse::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const size_type& num_cons,
                                     const index_type* idx_cons,
                                     const double* x,
                                     bool new_x,
                                     const int& nnzJacS,
                                     int* iJacS,
                                     int* jJacS,
                                     double* MJacS)
{
  return false;
}

/// @pre assuming Jac of the original prob is sorted
bool hiopFRProbSparse::eval_Jac_cons(const size_type& n,
                                     const size_type& m,
                                     const double* x,
                                     bool new_x,
                                     const int& nnzJacS,
                                     int* iJacS,
                                     int* jJacS,
                                     double* MJacS)
{
  assert(n == n_);
  assert(m == m_);

  assert(nnzJacS == nlp_base_->get_nnz_Jaceq() + nlp_base_->get_nnz_Jacineq() + 2 * (m_));

  hiopMatrixSparse& Jac_c = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Jac_c());
  hiopMatrixSparse& Jac_d = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Jac_d());

  // extend Jac to the p and n parts
  if(MJacS != nullptr) {
    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
    nlp_base_->eval_Jac_c_d(*wrk_x_, new_x, Jac_c, Jac_d);
  }

  Jac_cd_->set_Jac_FR(Jac_c, Jac_d, iJacS, jJacS, MJacS);

  return true;
}

bool hiopFRProbSparse::eval_Hess_Lagr(const size_type& n,
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
  assert(nnzHSS == nnz_Hess_Lag_);

  // shortcut to the original Hess
  hiopMatrixSparse& Hess = dynamic_cast<hiopMatrixSparse&>(*solver_base_.get_Hess_Lagr());

  if(MHSS != nullptr) {
    // get x for the original problem
    wrk_x_->copy_from_starting_at(x, 0, n_x_);

    // split lambda
    wrk_eq_->copy_from_starting_at(lambda, 0, m_eq_);
    wrk_ineq_->copy_from_starting_at(lambda, m_eq_, m_ineq_);

    double obj_factor = 0.0;
    // get Hess for the x part --- use original Hess as buffers
    nlp_base_->eval_Hess_Lagr(*wrk_x_, new_x, obj_factor, *wrk_eq_, *wrk_ineq_, new_lambda, Hess);

    // additional diag Hess for x:  zeta*DR^2
    wrk_x_->setToConstant(zeta_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->componentMult(*DR_);
  }

  // extend Hes to the p and n parts
  Hess_cd_->set_Hess_FR(Hess, iHSS, jHSS, MHSS, *wrk_x_);

  return true;
}

bool hiopFRProbSparse::get_warmstart_point(const size_type& n,
                                           const size_type& m,
                                           double* x0,
                                           double* z_bndL0,
                                           double* z_bndU0,
                                           double* lambda0,
                                           double* ineq_slack,
                                           double* vl0,
                                           double* vu0)
{
  assert(n == n_);
  assert(m == m_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  hiopVector* zl = solver_base_.get_it_curr()->get_zl();
  hiopVector* zu = solver_base_.get_it_curr()->get_zu();
  hiopVector* vl = solver_base_.get_it_curr()->get_vl();
  hiopVector* vu = solver_base_.get_it_curr()->get_vu();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x0 = x_ref
  wrk_x_->copyFrom(*x_ref_);

  // s = curr_s
  s->copyTo(ineq_slack);

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x0
   */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                           // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                  // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);  // ni

  wrk_primal_->copyTo(x0);

  /* initialize the dual variables for the variable bounds*/
  // get z = min(rho, z_base)
  wrk_x_->copyFrom(*zl);
  wrk_x_->component_min(rho_);

  // compute zl for p and n = mu*(p0)^-1
  wrk_c_->invert();
  wrk_c_->scale(mu_);
  wrk_eq_->invert();
  wrk_eq_->scale(mu_);
  wrk_d_->invert();
  wrk_d_->scale(mu_);
  wrk_ineq_->invert();
  wrk_ineq_->scale(mu_);

  // assemble zl
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                           // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                  // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);  // ni
  wrk_primal_->copyTo(z_bndL0);

  // get zu
  wrk_primal_->setToZero();
  wrk_x_->copyFrom(*zu);
  wrk_x_->component_min(rho_);
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(z_bndU0);

  // compute vl vu
  wrk_ineq_->copyFrom(*vl);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vl0);

  wrk_ineq_->copyFrom(*vu);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vu0);

  // set lambda to 0 --- this will be updated by lsq later.
  // Need to have this since we set duals_avail to true, in order to initialize zl and zu
  wrk_dual_->setToZero();
  wrk_dual_->copyTo(lambda0);
  return true;
}

bool hiopFRProbSparse::iterate_callback(int iter,
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
                                        int ls_trials)
{
  assert(n_ == n);
  assert(m_ineq_ == m_ineq);

  const hiopVector& crhs = nlp_base_->get_crhs();

  // evaluate c_body and d_body in base problem
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
  nlp_base_->eval_c_d(*wrk_x_, true, *wrk_cbody_, *wrk_dbody_);

  // compute theta for base problem
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_dbody_->axpy(-1.0, *wrk_d_);  // wrk_dbody_ = (d-s)

  double theta_ori = 0.0;
  theta_ori += wrk_cbody_->onenorm();
  theta_ori += wrk_dbody_->onenorm();

  double nrmInf_feas_ori = 0.0;
  nrmInf_feas_ori = fmax(wrk_cbody_->infnorm(), wrk_dbody_->infnorm());

  // check if restoration phase should be discontinued
  double max_nrmInf_feas = nlp_base_->options->GetNumeric("kappa_resto") * nrmInf_feas_ref_;

  // termination condition 1) theta_curr <= kappa_resto*theta_ref
  if(nrmInf_feas_ori <= max_nrmInf_feas && iter > 0) {
    // termination condition 2) (theta and logbar) are not in the original filter
    // check (original) filter condition

    // compute the original logbar objective from the trial point given by the FR problem
    // Note that this function will updates the slack and dual variables

    // set original trial (x,d) to the soltion from FR problem
    hiopIterate* it_base_trial = solver_base_.get_it_trial_nonconst();
    const hiopIterate* it_base_curr = solver_base_.get_it_curr();
    it_base_trial->get_x()->copyFrom(*wrk_x_);
    it_base_trial->get_d()->copyFrom(*wrk_d_);

    // compute other slacks in the base problem
    [[maybe_unused]] const size_type n_adjusted_slacks =
        it_base_trial->compute_safe_slacks(*it_base_curr, solver_base_.get_mu());

    // evaluate base problem log barr
    solver_base_.get_logbar()->updateWithNlpInfo_trial_funcOnly(*it_base_trial, obj_base_, *wrk_cbody_, *wrk_dbody_);

    double trial_bar_obj_ori = solver_base_.get_logbar()->f_logbar_trial;

    if(!solver_base_.filter_contains(theta_ori, trial_bar_obj_ori)) {
      // terminate FR
      last_x_->copyFrom(*wrk_x_);
      last_d_->copyFrom(*wrk_d_);
      return false;
    }
  }

  mu_ = mu;
  zeta_ = std::sqrt(mu_);

  return true;
}

bool hiopFRProbSparse::force_update_x(const int n, double* x)
{
  // this function is used in FR in FR, see eq (33)
  assert(n == n_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x is fixed
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x = [x pe ne pi ni
   */
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);

  wrk_primal_->copyTo(x);

  return true;
}

/*
 *  Specialized interface for feasibility restoration problem with MDS blocks in the Jacobian and Hessian.
 */
hiopFRProbMDS::hiopFRProbMDS(hiopAlgFilterIPMBase& solver_base)
    : solver_base_(solver_base),
      last_x_{nullptr},
      last_d_{nullptr}
{
  nlp_base_ = dynamic_cast<hiopNlpMDS*>(solver_base.get_nlp());
  n_x_ = nlp_base_->n();
  n_x_sp_ = nlp_base_->nx_sp();
  n_x_de_ = nlp_base_->nx_de();
  m_eq_ = nlp_base_->m_eq();
  m_ineq_ = nlp_base_->m_ineq();

  n_ = n_x_ + 2 * m_eq_ + 2 * m_ineq_;
  n_sp_ = n_x_sp_ + 2 * m_eq_ + 2 * m_ineq_;
  n_de_ = n_x_de_;
  m_ = m_eq_ + m_ineq_;

  x_sp_st_ = 0;
  pe_st_ = n_x_sp_;
  ne_st_ = pe_st_ + m_eq_;
  pi_st_ = ne_st_ + m_eq_;
  ni_st_ = pi_st_ + m_ineq_;
  x_de_st_ = ni_st_ + m_ineq_;

  x_ref_ = solver_base.get_it_curr()->get_x();

  // build vector VR
  DR_ = x_ref_->new_copy();
  DR_->component_abs();
  DR_->invert();
  DR_->component_min(1.0);

  wrk_x_ = x_ref_->alloc_clone();
  wrk_c_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_d_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_eq_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_ineq_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_cbody_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_dbody_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_primal_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_);
  wrk_dual_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_);
  last_x_ = x_ref_->alloc_clone();
  last_d_ = wrk_d_->alloc_clone();

  wrk_x_sp_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_x_sp_);
  wrk_x_de_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_x_de_);

  // nnz for sparse matrices;
  nnz_sp_Jac_c_ = nlp_base_->get_nnz_sp_Jaceq() + 2 * m_eq_;
  nnz_sp_Jac_d_ = nlp_base_->get_nnz_sp_Jacineq() + 2 * m_ineq_;

  // not sure if Hess has diagonal terms, compute nnz_hess here
  // assuming hess is in upper_triangular form
  hiopMatrixSymBlockDiagMDS* Hess_SS = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(solver_base_.get_Hess_Lagr());
  nnz_sp_Hess_Lagr_SS_ = n_x_sp_ + Hess_SS->sp_mat()->numberOfOffDiagNonzeros();
  nnz_sp_Hess_Lagr_SD_ = 0;

  Jac_cd_ = new hiopMatrixMDS(m_, n_sp_, n_de_, nnz_sp_Jac_c_ + nnz_sp_Jac_d_, nlp_base_->options->GetString("mem_space"));
  Hess_cd_ = new hiopMatrixSymBlockDiagMDS(n_sp_, n_de_, nnz_sp_Hess_Lagr_SS_, nlp_base_->options->GetString("mem_space"));

  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  theta_ref_ = solver_base_.get_resid()->get_theta();  // at current point, i.e., reference point
  nrmInf_feas_ref_ = solver_base_.get_resid()->get_nrmInf_bar_feasib();
  mu_ = solver_base.get_mu();
  mu_ = std::max(mu_, nrmInf_feas_ref_);

  zeta_ = std::sqrt(mu_);
  rho_ = 1000;  // FIXME: make this as an user option
}

hiopFRProbMDS::~hiopFRProbMDS()
{
  delete wrk_x_;
  delete wrk_c_;
  delete wrk_d_;
  delete wrk_eq_;
  delete wrk_ineq_;
  delete wrk_cbody_;
  delete wrk_dbody_;
  delete wrk_primal_;
  delete wrk_dual_;
  delete DR_;

  delete wrk_x_sp_;
  delete wrk_x_de_;

  delete Jac_cd_;
  delete Hess_cd_;
  if(last_x_) {
    delete last_x_;
  }
  if(last_d_) {
    delete last_d_;
  }
}

bool hiopFRProbMDS::get_MPI_comm(MPI_Comm& comm_out)
{
#ifdef HIOP_USE_MPI
  comm_out = nlp_base_->get_comm();
#else
  comm_out = MPI_COMM_WORLD;
#endif
  return true;
}

bool hiopFRProbMDS::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_;
  m = m_;
  return true;
}

bool hiopFRProbMDS::get_prob_info(NonlinearityType& type)
{
  // the objective of FR problem is quadratic. constraint c(x) depends on the origianl problem
  if(nlp_base_->get_prob_type() != hiopNonlinear) {
    type = hiopQuadratic;
  }
  return true;
}

bool hiopFRProbMDS::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_);

  const hiopVector& xl = nlp_base_->get_xl();
  const hiopVector& xu = nlp_base_->get_xu();
  const NonlinearityType* var_type = nlp_base_->get_var_type();

  // x, p and n, in the order of [xsp pe ne pi ni xde]
  wrk_primal_->setToConstant(0.0);
  xl.startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);
  xl.startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);
  wrk_primal_->copyTo(xlow);

  wrk_primal_->setToConstant(1e+20);
  xu.startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);
  xu.startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);
  wrk_primal_->copyTo(xupp);

  wrk_primal_->set_array_from_to(type, 0, n_, hiopLinear);
  wrk_primal_->set_array_from_to(type, x_sp_st_, x_sp_st_ + n_x_sp_, var_type, 0);
  wrk_primal_->set_array_from_to(type, x_de_st_, x_de_st_ + n_x_de_, var_type, n_x_sp_);

  return true;
}

bool hiopFRProbMDS::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == m_);
  assert(m_eq_ + m_ineq_ == m_);
  const hiopVector& crhs = nlp_base_->get_crhs();
  const hiopVector& dl = nlp_base_->get_dl();
  const hiopVector& du = nlp_base_->get_du();
  const NonlinearityType* cons_eq_type = nlp_base_->get_cons_eq_type();
  const NonlinearityType* cons_ineq_type = nlp_base_->get_cons_ineq_type();

  wrk_dual_->setToConstant(0.0);

  // assemble wrk_dual_ = [crhs; dl] for lower bounds
  crhs.copyToStarting(*wrk_dual_, 0);
  dl.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(clow);

  // assemble wrk_dual_ = [crhs; du] for upper bounds
  du.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(cupp);

  wrk_dual_->set_array_from_to(type, 0, m_eq_, cons_eq_type, 0);
  wrk_dual_->set_array_from_to(type, m_eq_, m_, cons_ineq_type, 0);

  return true;
}

bool hiopFRProbMDS::get_sparse_dense_blocks_info(int& nx_sparse,
                                                 int& nx_dense,
                                                 int& nnz_sparse_Jaceq,
                                                 int& nnz_sparse_Jacineq,
                                                 int& nnz_sparse_Hess_Lagr_SS,
                                                 int& nnz_sparse_Hess_Lagr_SD)
{
  nx_sparse = n_sp_;
  nx_dense = n_de_;
  nnz_sparse_Jaceq = nnz_sp_Jac_c_;
  nnz_sparse_Jacineq = nnz_sp_Jac_d_;
  nnz_sparse_Hess_Lagr_SS = nnz_sp_Hess_Lagr_SS_;
  nnz_sparse_Hess_Lagr_SD = nnz_sp_Hess_Lagr_SD_;
  return true;
}

bool hiopFRProbMDS::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_);
  obj_value = 0.;

  wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
  wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // [xsp]
  wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // [xde]

  // rho*sum(p+n)
  obj_value += rho_ * (wrk_primal_->sum_local() - wrk_x_->sum_local());

  // zeta/2*[DR*(x-x_ref)]^2
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  double wrk_db = wrk_x_->twonorm();

  obj_value += 0.5 * zeta_ * wrk_db * wrk_db;

  nlp_base_->eval_f(*wrk_x_, new_x, obj_base_);

  return true;
}

bool hiopFRProbMDS::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_);

  // x
  wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
  wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
  wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]

  // p and n
  wrk_primal_->setToConstant(rho_);

  // x
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->scale(zeta_);
  wrk_x_->startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);        // x = [xsp pe ne pi ni xde]
  wrk_x_->startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);  // x = [xsp pe ne pi ni xde]

  wrk_primal_->copyTo(gradf);

  return true;
}

bool hiopFRProbMDS::eval_cons(const size_type& n,
                              const size_type& m,
                              const size_type& num_cons,
                              const index_type* idx_cons,
                              const double* x,
                              bool new_x,
                              double* cons)
{
  return false;
}

bool hiopFRProbMDS::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_);
  assert(m == m_);

  // evaluate base case c and d
  wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
  wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
  wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]
  nlp_base_->eval_c_d(*wrk_x_, new_x, *wrk_c_, *wrk_d_);

  // compute FR equality constratint body c-pe+ne
  wrk_eq_->copy_from_starting_at(x, pe_st_, m_eq_);  // pe
  wrk_c_->axpy(-1.0, *wrk_eq_);
  wrk_eq_->copy_from_starting_at(x, ne_st_, m_eq_);  // ne
  wrk_c_->axpy(1.0, *wrk_eq_);

  // compute FR inequality constratint body d-pi+ni
  wrk_ineq_->copy_from_starting_at(x, pi_st_, m_ineq_);  // pi
  wrk_d_->axpy(-1.0, *wrk_ineq_);
  wrk_ineq_->copy_from_starting_at(x, ni_st_, m_ineq_);  // ni
  wrk_d_->axpy(1.0, *wrk_ineq_);

  // assemble the full vector
  wrk_c_->copyToStarting(*wrk_dual_, 0);
  wrk_d_->copyToStarting(*wrk_dual_, m_eq_);

  wrk_dual_->copyTo(cons);

  return true;
}

bool hiopFRProbMDS::eval_Jac_cons(const size_type& n,
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
  return false;
}

/// @pre assuming Jac of the original prob is sorted
bool hiopFRProbMDS::eval_Jac_cons(const size_type& n,
                                  const size_type& m,
                                  const double* x,
                                  bool new_x,
                                  const size_type& nsparse,
                                  const size_type& ndense,
                                  const size_type& nnzJacS,
                                  int* iJacS,
                                  int* jJacS,
                                  double* MJacS,
                                  double* JacD)
{
  assert(n == n_);
  assert(m == m_);
  assert(nsparse == n_sp_);
  assert(ndense == n_de_);
  assert(nnzJacS == nlp_base_->get_nnz_sp_Jaceq() + nlp_base_->get_nnz_sp_Jacineq() + 2 * (m_));

  hiopMatrixMDS* Jac_c = dynamic_cast<hiopMatrixMDS*>(solver_base_.get_Jac_c());
  hiopMatrixMDS* Jac_d = dynamic_cast<hiopMatrixMDS*>(solver_base_.get_Jac_d());
  assert(Jac_c && Jac_d);

  // extend Jac to the p and n parts
  if(MJacS != nullptr) {
    // get x for the original problem
    wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
    wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
    wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]

    // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
    nlp_base_->eval_Jac_c_d(*wrk_x_, new_x, *Jac_c, *Jac_d);
  }

  Jac_cd_->set_Jac_FR(*Jac_c, *Jac_d, iJacS, jJacS, MJacS, JacD);

  return true;
}

bool hiopFRProbMDS::get_warmstart_point(const size_type& n,
                                        const size_type& m,
                                        double* x0,
                                        double* z_bndL0,
                                        double* z_bndU0,
                                        double* lambda0,
                                        double* ineq_slack,
                                        double* vl0,
                                        double* vu0)
{
  assert(n == n_);
  assert(m == m_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  hiopVector* zl = solver_base_.get_it_curr()->get_zl();
  hiopVector* zu = solver_base_.get_it_curr()->get_zu();

  hiopVector* vl = solver_base_.get_it_curr()->get_vl();
  hiopVector* vu = solver_base_.get_it_curr()->get_vu();

  const hiopVector& crhs = nlp_base_->get_crhs();

  // x0 = x_ref
  wrk_x_->copyFrom(*x_ref_);

  // s = curr_s
  s->copyTo(ineq_slack);

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x0
   */
  wrk_x_->startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);        // [xsp pe ne pi ni xde]
  wrk_c_->copyToStarting(*wrk_primal_, pe_st_);                                  // pe
  wrk_eq_->copyToStarting(*wrk_primal_, ne_st_);                                 // ne
  wrk_d_->copyToStarting(*wrk_primal_, pi_st_);                                  // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, ni_st_);                               // ni
  wrk_x_->startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);  // [xsp pe ne pi ni xde]

  wrk_primal_->copyTo(x0);

  /* initialize the dual variables for the variable bounds*/
  // get z = min(rho, z_base)
  wrk_x_->copyFrom(*zl);
  wrk_x_->component_min(rho_);

  // compute zl for p and n = mu*(p0)^-1
  wrk_c_->invert();
  wrk_c_->scale(mu_);
  wrk_eq_->invert();
  wrk_eq_->scale(mu_);
  wrk_d_->invert();
  wrk_d_->scale(mu_);
  wrk_ineq_->invert();
  wrk_ineq_->scale(mu_);

  // assemble zl
  wrk_x_->startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);        // [xsp pe ne pi ni xde]
  wrk_c_->copyToStarting(*wrk_primal_, pe_st_);                                  // pe
  wrk_eq_->copyToStarting(*wrk_primal_, ne_st_);                                 // ne
  wrk_d_->copyToStarting(*wrk_primal_, pi_st_);                                  // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, ni_st_);                               // ni
  wrk_x_->startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);  // [xsp pe ne pi ni xde]
  wrk_primal_->copyTo(z_bndL0);

  // get zu
  wrk_primal_->setToZero();
  wrk_x_->copyFrom(*zu);
  wrk_x_->component_min(rho_);
  wrk_x_->startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);        // [xsp pe ne pi ni xde]
  wrk_x_->startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);  // [xsp pe ne pi ni xde]
  wrk_primal_->copyTo(z_bndU0);

  // compute vl vu
  wrk_ineq_->copyFrom(*vl);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vl0);

  wrk_ineq_->copyFrom(*vu);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vu0);

  // set lambda to 0 --- this will be updated by lsq later.
  // Need to have this since we set duals_avail to true, in order to initialize zl and zu
  wrk_dual_->setToZero();
  wrk_dual_->copyTo(lambda0);
  return true;
}

bool hiopFRProbMDS::iterate_callback(int iter,
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
                                     int ls_trials)
{
  assert(n_ == n);
  assert(m_ineq_ == m_ineq);

  const hiopVector& crhs = nlp_base_->get_crhs();

  // evaluate c_body and d_body in base problem
  wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
  wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
  wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]
  wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
  nlp_base_->eval_c_d(*wrk_x_, true, *wrk_cbody_, *wrk_dbody_);

  // compute theta for base problem
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_dbody_->axpy(-1.0, *wrk_d_);  // wrk_dbody_ = (d-s)

  double theta_ori = 0.0;
  theta_ori += wrk_cbody_->onenorm();
  theta_ori += wrk_dbody_->onenorm();

  double nrmInf_feas_ori = 0.0;
  nrmInf_feas_ori = fmax(wrk_cbody_->infnorm(), wrk_dbody_->infnorm());

  // check if restoration phase should be discontinued
  double max_nrmInf_feas = nlp_base_->options->GetNumeric("kappa_resto") * nrmInf_feas_ref_;

  // termination condition 1) theta_curr <= kappa_resto*theta_ref
  if(nrmInf_feas_ori <= max_nrmInf_feas && iter > 0) {
    // termination condition 2) (theta and logbar) are not in the original filter
    // check (original) filter condition

    // compute the original logbar objective from the trial point given by the FR problem
    // Note that this function will updates the slack and dual variables

    // set original trial (x,d) to the soltion from FR problem
    hiopIterate* it_base_trial = solver_base_.get_it_trial_nonconst();
    const hiopIterate* it_base_curr = solver_base_.get_it_curr();
    it_base_trial->get_x()->copyFrom(*wrk_x_);
    it_base_trial->get_d()->copyFrom(*wrk_d_);

    // compute other slacks in the base problem
    [[maybe_unused]] const size_type n_adjusted_slacks =
        it_base_trial->compute_safe_slacks(*it_base_curr, solver_base_.get_mu());

    // evaluate base problem log barr
    solver_base_.get_logbar()->updateWithNlpInfo_trial_funcOnly(*it_base_trial, obj_base_, *wrk_cbody_, *wrk_dbody_);

    double trial_bar_obj_ori = solver_base_.get_logbar()->f_logbar_trial;

    if(!solver_base_.filter_contains(theta_ori, trial_bar_obj_ori)) {
      // terminate FR
      last_x_->copyFrom(*wrk_x_);
      last_d_->copyFrom(*wrk_d_);
      return false;
    }
  }

  mu_ = mu;
  zeta_ = std::sqrt(mu_);

  return true;
}

bool hiopFRProbMDS::force_update_x(const int n, double* x)
{
  // this function is used in FR in FR, see eq (33)
  assert(n == n_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x is fixed
  wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
  wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
  wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x
   */
  wrk_x_->startingAtCopyToStartingAt(0, *wrk_primal_, x_sp_st_, n_x_sp_);        // [xsp pe ne pi ni xde]
  wrk_c_->copyToStarting(*wrk_primal_, pe_st_);                                  // pe
  wrk_eq_->copyToStarting(*wrk_primal_, ne_st_);                                 // ne
  wrk_d_->copyToStarting(*wrk_primal_, pi_st_);                                  // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, ni_st_);                               // ni
  wrk_x_->startingAtCopyToStartingAt(n_x_sp_, *wrk_primal_, x_de_st_, n_x_de_);  // [xsp pe ne pi ni xde]

  wrk_primal_->copyTo(x);

  return true;
}

bool hiopFRProbMDS::eval_Hess_Lagr(const size_type& n,
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
  assert(nnzHSS == nnz_sp_Hess_Lagr_SS_);
  assert(nnzHSD == 0);

  // shortcut to the original Hess
  hiopMatrixSymBlockDiagMDS* Hess = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(solver_base_.get_Hess_Lagr());
  assert(Hess);

  if(MHSS != nullptr) {
    // get x for the original problem
    wrk_primal_->copy_from_starting_at(x, 0, n_);                                  // [xsp pe ne pi ni xde]
    wrk_primal_->startingAtCopyToStartingAt(x_sp_st_, *wrk_x_, 0, n_x_sp_);        // x = [xsp xde]
    wrk_primal_->startingAtCopyToStartingAt(x_de_st_, *wrk_x_, n_x_sp_, n_x_de_);  // x = [xsp xde]

    // split lambda
    wrk_eq_->copy_from_starting_at(lambda, 0, m_eq_);
    wrk_ineq_->copy_from_starting_at(lambda, m_eq_, m_ineq_);

    double obj_factor = 0.0;
    // get Hess for the x part --- use original Hess as buffers
    nlp_base_->eval_Hess_Lagr(*wrk_x_, new_x, obj_factor, *wrk_eq_, *wrk_ineq_, new_lambda, *Hess);

    // additional diag Hess for x:  zeta*DR^2
    wrk_x_->setToConstant(zeta_);
    wrk_x_->componentMult(*DR_);
    wrk_x_->componentMult(*DR_);

    wrk_x_->copyToStarting(0, *wrk_x_sp_);
    wrk_x_->copyToStarting(n_x_sp_, *wrk_x_de_);
  }

  // extend Hes to the p and n parts
  Hess_cd_->set_Hess_FR(*Hess, iHSS, jHSS, MHSS, HDD, *wrk_x_sp_, *wrk_x_de_);

  return true;
}

/*
index_type hiopFRProbDense::denseVecBase2FR(hiopVector* vec_base, hiopVector* vec_fr)
{
  assert(vec_base->get_size() == n_x_);
  assert(vec_fr->get_size() == n_);

   {
      return idx_local + col_partition_[my_rank];
    }
    assert(false && "you shouldn't need global index for a vector of this size.");
    return -1;
  }
}
*/

/*
 *  Specialized interface for feasibility restoration problem with MDS blocks in the Jacobian and Hessian.
 */
hiopFRProbDense::hiopFRProbDense(hiopAlgFilterIPMBase& solver_base)
    : solver_base_(solver_base),
      last_x_{nullptr},
      last_d_{nullptr}
{
  nlp_base_ = dynamic_cast<hiopNlpDenseConstraints*>(solver_base.get_nlp());
  n_x_ = nlp_base_->n();
  m_eq_ = nlp_base_->m_eq();
  m_ineq_ = nlp_base_->m_ineq();
#ifdef HIOP_USE_MPI
  vec_distrib_base_ = nlp_base_->getVecDistInfo();
#endif
  n_ = n_x_ + 2 * m_eq_ + 2 * m_ineq_;
  m_ = m_eq_ + m_ineq_;

  pe_st_ = n_x_;
  ne_st_ = pe_st_ + m_eq_;
  pi_st_ = ne_st_ + m_eq_;
  ni_st_ = pi_st_ + m_ineq_;

  x_ref_ = solver_base.get_it_curr()->get_x();

  // build vector VR
  DR_ = x_ref_->new_copy();
  DR_->component_abs();
  DR_->invert();
  DR_->component_min(1.0);

  comm_ = MPI_COMM_WORLD;
  comm_size_ = 1;
  rank_ = 0;
#ifdef HIOP_USE_MPI
  comm_ = nlp_base_->get_comm();
  comm_size_ = nlp_base_->get_num_ranks();
  rank_ = nlp_base_->get_rank();
#endif

  // assign col_partition_
  col_partition_ = new index_type[comm_size_ + 1];
  col_partition_[0] = 0;
  col_partition_[comm_size_] = n_;

#ifdef HIOP_USE_MPI
  if(vec_distrib_base_) {
    for(int i = 0; i < comm_size_; ++i) {
      col_partition_[i] = nlp_base_->getVecDistInfo()[i];
    }
  }
  if(col_partition_) {
    wrk_primal_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_, col_partition_, comm_);
    Jac_cd_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", m_, n_, col_partition_, comm_);
  } else {
    wrk_primal_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_);
    Jac_cd_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", m_, n_);
  }
#else
  wrk_primal_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), n_);
  Jac_cd_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", m_, n_);
#endif

  wrk_c_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_eq_);
  wrk_d_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_ineq_);
  wrk_dual_ = LinearAlgebraFactory::create_vector(nlp_base_->options->GetString("mem_space"), m_);

  wrk_x_ = x_ref_->alloc_clone();
  wrk_eq_ = wrk_c_->alloc_clone();
  wrk_ineq_ = wrk_d_->alloc_clone();
  wrk_cbody_ = wrk_c_->alloc_clone();
  wrk_dbody_ = wrk_d_->alloc_clone();
  last_x_ = x_ref_->alloc_clone();
  last_d_ = wrk_d_->alloc_clone();

  // set mu0 to be the maximun of the current barrier parameter mu and norm_inf(|c|)*/
  theta_ref_ = solver_base_.get_resid()->get_theta();  // at current point, i.e., reference point
  nrmInf_feas_ref_ = solver_base_.get_resid()->get_nrmInf_bar_feasib();
  mu_ = solver_base.get_mu();
  mu_ = std::max(mu_, nrmInf_feas_ref_);

  zeta_ = std::sqrt(mu_);
  rho_ = 1000;  // FIXME: make this as an user option
}

hiopFRProbDense::~hiopFRProbDense()
{
  delete wrk_x_;
  delete wrk_c_;
  delete wrk_d_;
  delete wrk_eq_;
  delete wrk_ineq_;
  delete wrk_cbody_;
  delete wrk_dbody_;
  delete wrk_primal_;
  delete wrk_dual_;
  delete DR_;

  delete Jac_cd_;

  if(last_x_) {
    delete last_x_;
  }
  if(last_d_) {
    delete last_d_;
  }
  delete[] col_partition_;
}

bool hiopFRProbDense::get_MPI_comm(MPI_Comm& comm_out)
{
  comm_out = comm_;
  return true;
}

bool hiopFRProbDense::get_vecdistrib_info(size_type global_n, index_type* cols)
{
  if(global_n == n_) {
    for(int i = 0; i <= comm_size_; i++) {
      cols[i] = col_partition_[i];
    }
  } else {
    assert(false && "You shouldn't need distrib info for this size.");
  }
  return true;
}

bool hiopFRProbDense::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_;
  m = m_;
  return true;
}

bool hiopFRProbDense::get_prob_info(NonlinearityType& type)
{
  // the objective of FR problem is quadratic. constraint c(x) depends on the origianl problem
  if(nlp_base_->get_prob_type() != hiopNonlinear) {
    type = hiopQuadratic;
  }
  return true;
}

bool hiopFRProbDense::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_);

  const hiopVector& xl = nlp_base_->get_xl();
  const hiopVector& xu = nlp_base_->get_xu();

  // x, p and n, in the order of [x pe ne pi ni]

  // build new lower bound
  wrk_primal_->setToConstant(0.0);
  // FIXME; add global methed. now we only have local method
  xl.copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(xlow);

  // build new upper bound
  wrk_primal_->setToConstant(1e+20);
  xu.copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(xupp);

  for(index_type i_local = 0; i_local < xl.get_local_size(); ++i_local) {
    type[i_local] = hiopNonlinear;
  }

  return true;
}

bool hiopFRProbDense::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == m_);
  assert(m_eq_ + m_ineq_ == m_);
  const hiopVector& crhs = nlp_base_->get_crhs();
  const hiopVector& dl = nlp_base_->get_dl();
  const hiopVector& du = nlp_base_->get_du();

  wrk_dual_->setToConstant(0.0);

  // assemble wrk_dual_ = [crhs; dl] for lower bounds

  // FIXME; add global methed. now we only have local method
  crhs.copyToStarting(*wrk_dual_, 0);
  dl.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(clow);

  // assemble wrk_dual_ = [crhs; du] for upper bounds
  du.copyToStarting(*wrk_dual_, (int)m_eq_);
  wrk_dual_->copyTo(cupp);

  for(index_type i_local = 0; i_local < dl.get_local_size(); ++i_local) {
    type[i_local] = hiopInterfaceBase::hiopLinear;
  }
  return true;
}

bool hiopFRProbDense::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_);
  obj_value = 0.;

  // FIXME; add global methed. now we only have local method, or sync mpi results here
  wrk_primal_->copy_from_starting_at(x, 0, n_);  // [x pe ne pi ni]
  wrk_x_->copy_from_starting_at(x, 0, n_x_);     // [x]

  // rho*sum(p+n)
  // FIXME; add global methed. now we only have local method, or sync mpi results here
  obj_value += rho_ * (wrk_primal_->sum_local() - wrk_x_->sum_local());

#ifdef HIOP_USE_MPI
  double obj_global;
  int ierr = MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(ierr == MPI_SUCCESS);
  obj_value = obj_global;
#endif

  // zeta/2*[DR*(x-x_ref)]^2
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  double wrk_db = wrk_x_->twonorm();

  obj_value += 0.5 * zeta_ * wrk_db * wrk_db;

  // keep a copy of the original objective value
  nlp_base_->eval_f(*wrk_x_, new_x, obj_base_);

  return true;
}

bool hiopFRProbDense::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_);

  // p and n
  wrk_primal_->setToConstant(rho_);

  // x: zeta*DR^2*(x-x_ref)
  // FIXME; add global methed. now we only have local method
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_x_->axpy(-1.0, *x_ref_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->componentMult(*DR_);
  wrk_x_->scale(zeta_);

  // build [x p n]
  // FIXME; add global methed. now we only have local method
  wrk_x_->copyToStarting(*wrk_primal_, 0);

  wrk_primal_->copyTo(gradf);

  return true;
}

bool hiopFRProbDense::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_);
  assert(m == m_);

  // assemble vector x
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  // evaluate base case c and d
  nlp_base_->eval_c_d(*wrk_x_, new_x, *wrk_c_, *wrk_d_);

#ifdef HIOP_USE_MPI
  // p and n are only located in the last rank
  // compute FR equality constratint body c-pe+ne
  if(m_eq_ > 0) {
    if(rank_ == comm_size_ - 1) {
      wrk_eq_->copy_from_starting_at(x + pe_st_ - col_partition_[rank_], 0, m_eq_);  // pe
      wrk_c_->axpy(-1.0, *wrk_eq_);
      wrk_eq_->copy_from_starting_at(x + ne_st_ - col_partition_[rank_], 0, m_eq_);  // ne
      wrk_c_->axpy(1.0, *wrk_eq_);
    }
    int ierr = MPI_Bcast(wrk_c_->local_data(), m_eq_, MPI_DOUBLE, comm_size_ - 1, comm_);
    assert(ierr == MPI_SUCCESS);
  }

  // compute FR equality constratint body d-pi+ni
  if(m_ineq_ > 0) {
    if(rank_ == comm_size_ - 1) {
      wrk_ineq_->copy_from_starting_at(x + pi_st_ - col_partition_[rank_], 0, m_ineq_);  // pi
      wrk_d_->axpy(-1.0, *wrk_ineq_);
      wrk_ineq_->copy_from_starting_at(x + ni_st_ - col_partition_[rank_], 0, m_ineq_);  // ni
      wrk_d_->axpy(1.0, *wrk_ineq_);
    }
    int ierr = MPI_Bcast(wrk_d_->local_data(), m_ineq_, MPI_DOUBLE, comm_size_ - 1, comm_);
    assert(ierr == MPI_SUCCESS);
  }
#else
  // compute FR equality constratint body c-pe+ne
  if(m_eq_ > 0) {
    wrk_eq_->copy_from_starting_at(x + pe_st_, 0, m_eq_);  // pe
    wrk_c_->axpy(-1.0, *wrk_eq_);
    wrk_eq_->copy_from_starting_at(x + ne_st_, 0, m_eq_);  // ne
    wrk_c_->axpy(1.0, *wrk_eq_);
  }

  // compute FR equality constratint body d-pi+ni
  if(m_ineq_ > 0) {
    wrk_ineq_->copy_from_starting_at(x + pi_st_, 0, m_ineq_);  // pi
    wrk_d_->axpy(-1.0, *wrk_ineq_);
    wrk_ineq_->copy_from_starting_at(x + ni_st_, 0, m_ineq_);  // ni
    wrk_d_->axpy(1.0, *wrk_ineq_);
  }
#endif

  // assemble the full vector
  // FIXME; add global methed. now we only have local method
  wrk_c_->copyToStarting(*wrk_dual_, 0);
  wrk_d_->copyToStarting(*wrk_dual_, m_eq_);

  wrk_dual_->copyTo(cons);

  return true;
}

/// @pre assuming Jac of the original prob is sorted
bool hiopFRProbDense::eval_Jac_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* Jac)
{
  assert(n == n_);
  assert(m == m_);

  hiopMatrixDense* Jac_c = dynamic_cast<hiopMatrixDense*>(solver_base_.get_Jac_c());
  hiopMatrixDense* Jac_d = dynamic_cast<hiopMatrixDense*>(solver_base_.get_Jac_d());
  assert(Jac_c && Jac_d);

  // extend Jac to the p and n parts
  assert(Jac != nullptr);
  // get x for the original problem
  // FIXME; add global methed. now we only have local method
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  // get Jac_c and Jac_d for the x part --- use original Jac_c/Jac_d as buffers
  nlp_base_->eval_Jac_c_d(*wrk_x_, new_x, *Jac_c, *Jac_d);

  // FIXME add set_Jac_FR in hiopMatrixDense
  Jac_cd_->set_Jac_FR(*Jac_c, *Jac_d);

  Jac_cd_->copy_to(Jac);

  return true;
}

bool hiopFRProbDense::get_warmstart_point(const size_type& n,
                                          const size_type& m,
                                          double* x0,
                                          double* z_bndL0,
                                          double* z_bndU0,
                                          double* lambda0,
                                          double* ineq_slack,
                                          double* vl0,
                                          double* vu0)
{
  assert(n == n_);
  assert(m == m_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  hiopVector* zl = solver_base_.get_it_curr()->get_zl();
  hiopVector* zu = solver_base_.get_it_curr()->get_zu();
  hiopVector* vl = solver_base_.get_it_curr()->get_vl();
  hiopVector* vu = solver_base_.get_it_curr()->get_vu();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x0 = x_ref
  wrk_x_->copyFrom(*x_ref_);

  // s = curr_s
  s->copyTo(ineq_slack);

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x0
   */
  // FIXME; add global methed. now we only have local method
  wrk_primal_->setToConstant(0.0);
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                           // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                  // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);  // ni

  wrk_primal_->copyTo(x0);

  /* initialize the dual variables for the variable bounds*/
  // get z = min(rho, z_base)
  wrk_x_->copyFrom(*zl);
  wrk_x_->component_min(rho_);

  // compute zl for p and n = mu*(p0)^-1
  wrk_c_->invert();
  wrk_c_->scale(mu_);
  wrk_eq_->invert();
  wrk_eq_->scale(mu_);
  wrk_d_->invert();
  wrk_d_->scale(mu_);
  wrk_ineq_->invert();
  wrk_ineq_->scale(mu_);

  // assemble zl
  // FIXME; add global methed. now we only have local method
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);                           // pe
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);                  // ne
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);               // pi
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);  // ni
  wrk_primal_->copyTo(z_bndL0);

  // get zu
  wrk_primal_->setToZero();
  wrk_x_->copyFrom(*zu);
  wrk_x_->component_min(rho_);
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_primal_->copyTo(z_bndU0);

  // compute vl vu
  wrk_ineq_->copyFrom(*vl);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vl0);

  wrk_ineq_->copyFrom(*vu);
  wrk_ineq_->component_min(rho_);
  wrk_ineq_->copyTo(vu0);

  // set lambda to 0 --- this will be updated by lsq later.
  // Need to have this since we set duals_avail to true, in order to initialize zl and zu
  wrk_dual_->setToZero();
  wrk_dual_->copyTo(lambda0);
  return true;
}

bool hiopFRProbDense::iterate_callback(int iter,
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
                                       int ls_trials)
{
  assert(n_ == n);
  assert(m_ineq_ == m_ineq);

  const hiopVector& crhs = nlp_base_->get_crhs();

  // evaluate c_body and d_body in base problem
  // FIXME; add global methed. now we only have local method
  wrk_x_->copy_from_starting_at(x, 0, n_x_);
  wrk_d_->copy_from_starting_at(s, 0, m_ineq_);
  nlp_base_->eval_c_d(*wrk_x_, true, *wrk_cbody_, *wrk_dbody_);

  // compute theta for base problem
  wrk_cbody_->axpy(-1.0, crhs);     // wrk_cbody_ = (c-crhs)
  wrk_dbody_->axpy(-1.0, *wrk_d_);  // wrk_dbody_ = (d-s)

  double theta_ori = 0.0;
  theta_ori += wrk_cbody_->onenorm();
  theta_ori += wrk_dbody_->onenorm();

  double nrmInf_feas_ori = 0.0;
  nrmInf_feas_ori = fmax(wrk_cbody_->infnorm(), wrk_dbody_->infnorm());

  // check if restoration phase should be discontinued
  double max_nrmInf_feas = nlp_base_->options->GetNumeric("kappa_resto") * nrmInf_feas_ref_;

  // termination condition 1) theta_curr <= kappa_resto*theta_ref
  if(nrmInf_feas_ori <= max_nrmInf_feas && iter > 0) {
    // termination condition 2) (theta and logbar) are not in the original filter
    // check (original) filter condition
    [[maybe_unused]] double trial_obj_ori = obj_base_;  // obj_base_ has been updated in the FR loop when we evaluate obj

    // compute the original logbar objective from the trial point given by the FR problem
    // Note that this function will updates the slack and dual variables

    // set original trial (x,d) to the soltion from FR problem
    hiopIterate* it_base_trial = solver_base_.get_it_trial_nonconst();
    const hiopIterate* it_base_curr = solver_base_.get_it_curr();
    it_base_trial->get_x()->copyFrom(*wrk_x_);
    it_base_trial->get_d()->copyFrom(*wrk_d_);

    // compute other slacks in the base problem
    [[maybe_unused]] size_type n_adjusted_slacks = it_base_trial->compute_safe_slacks(*it_base_curr, solver_base_.get_mu());

    // evaluate base problem log barr
    solver_base_.get_logbar()->updateWithNlpInfo_trial_funcOnly(*it_base_trial, obj_base_, *wrk_cbody_, *wrk_dbody_);

    double trial_bar_obj_ori = solver_base_.get_logbar()->f_logbar_trial;

    if(!solver_base_.filter_contains(theta_ori, trial_bar_obj_ori)) {
      // terminate FR
      last_x_->copyFrom(*wrk_x_);
      last_d_->copyFrom(*wrk_d_);
      return false;
    }
  }

  mu_ = mu;
  zeta_ = std::sqrt(mu_);

  return true;
}

bool hiopFRProbDense::force_update_x(const int n, double* x)
{
  // this function is used in FR in FR, see eq (33)
  assert(n == n_);

  hiopVector* c = solver_base_.get_c();
  hiopVector* d = solver_base_.get_d();
  hiopVector* s = solver_base_.get_it_curr()->get_d();
  const hiopVector& crhs = nlp_base_->get_crhs();

  // x is fixed
  // FIXME; add global methed. now we only have local method
  wrk_x_->copy_from_starting_at(x, 0, n_x_);

  /*
   * compute pe (wrk_c_) and ne (wrk_eq_) rom equation (33)
   */
  // firstly use pe as a temp vec
  double tmp_db = mu_ / (2 * rho_);
  wrk_cbody_->copyFrom(*c);
  wrk_cbody_->axpy(-1.0, crhs);  // wrk_cbody_ = (c-crhs)
  wrk_c_->setToConstant(tmp_db);
  wrk_c_->axpy(-0.5, *wrk_cbody_);  // wrk_c_ = (mu-rho*(c-crhs))/(2*rho)

  // compute ne (wrk_eq_)
  wrk_eq_->copyFrom(*wrk_c_);
  wrk_eq_->componentMult(*wrk_c_);
  wrk_eq_->axpy(tmp_db, *wrk_cbody_);
  wrk_eq_->component_sqrt();
  wrk_eq_->axpy(1.0, *wrk_c_);

  // compute pe (wrk_c_)
  wrk_c_->copyFrom(*wrk_cbody_);
  wrk_c_->axpy(1.0, *wrk_eq_);

  /*
   * compute pi (wrk_d_) and ni (wrk_ineq_) rom equation (33)
   */
  // firstly use pi as a temp vec
  wrk_dbody_->copyFrom(*d);
  wrk_dbody_->axpy(-1.0, *s);  // wrk_dbody_ = (d-s)
  wrk_d_->setToConstant(tmp_db);
  wrk_d_->axpy(-0.5, *wrk_dbody_);  // wrk_c_ = (mu-rho*(d-s))/(2*rho)

  // compute ni (wrk_ineq_)
  wrk_ineq_->copyFrom(*wrk_d_);
  wrk_ineq_->componentMult(*wrk_d_);
  wrk_ineq_->axpy(tmp_db, *wrk_dbody_);
  wrk_ineq_->component_sqrt();
  wrk_ineq_->axpy(1.0, *wrk_d_);

  // compute pi (wrk_d_)
  wrk_d_->copyFrom(*wrk_dbody_);
  wrk_d_->axpy(1.0, *wrk_ineq_);

  /*
   * assemble x = [x pe ne pi ni]
   */
  // FIXME; add global methed. now we only have local method
  wrk_x_->copyToStarting(*wrk_primal_, 0);
  wrk_c_->copyToStarting(*wrk_primal_, n_x_);
  wrk_eq_->copyToStarting(*wrk_primal_, n_x_ + m_eq_);
  wrk_d_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_);
  wrk_ineq_->copyToStarting(*wrk_primal_, n_x_ + 2 * m_eq_ + m_ineq_);

  wrk_primal_->copyTo(x);

  return true;
}

};  // namespace hiop
