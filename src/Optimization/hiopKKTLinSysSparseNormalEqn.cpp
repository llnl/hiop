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
 * @file hiopKKTLinSysSparseNormalEqn.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 */

#include "hiopKKTLinSysSparseNormalEqn.hpp"
#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverSymSparseMA57.hpp"
#endif

#ifdef HIOP_USE_CUDA
#include "hiopLinSolverCholCuSparse.hpp"
#include "hiopVectorCuda.hpp"
#endif  // HIOP_USE_CUDA

#include "hiopMatrixSparseCSRSeq.hpp"

namespace hiop
{
hiopKKTLinSysSparseNormalEqn::hiopKKTLinSysSparseNormalEqn(hiopNlpFormulation* nlp)
    : hiopKKTLinSysNormalEquation(nlp),
      rhs_{nullptr},
      Hess_diag_{nullptr},
      deltawx_{nullptr},
      deltawd_{nullptr},
      deltacc_{nullptr},
      deltacd_{nullptr},
      dual_reg_copy_{nullptr},
      Hess_diag_copy_{nullptr},
      Hx_copy_{nullptr},
      Hd_copy_{nullptr},
      Hxd_inv_copy_{nullptr},
      write_linsys_counter_(-1),
      csr_writer_(nlp),
      nlpSp_{nullptr},
      HessSp_{nullptr},
      Jac_cSp_{nullptr},
      Jac_dSp_{nullptr},
      JacD_{nullptr},
      JacDt_{nullptr},
      JDiagJt_{nullptr},
      Diag_dualreg_{nullptr},
      M_normaleqn_{nullptr}
{
  nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
  assert(nlpSp_);
}

hiopKKTLinSysSparseNormalEqn::~hiopKKTLinSysSparseNormalEqn()
{
  delete rhs_;
  delete Hess_diag_;
  delete deltawx_;
  delete deltawd_;
  delete deltacc_;
  delete deltacd_;
  delete dual_reg_copy_;
  delete Hess_diag_copy_;
  delete Hx_copy_;
  delete Hd_copy_;
  delete Hxd_inv_copy_;
  delete JacD_;
  delete JacDt_;
  delete JDiagJt_;
  delete Diag_dualreg_;
  delete M_normaleqn_;
}

bool hiopKKTLinSysSparseNormalEqn::build_kkt_matrix(const hiopPDPerturbation& pdreg)
{
#ifdef HIOP_DEEPCHECKS
  assert(perturb_calc_->check_consistency() && "something went wrong with IC");
#endif

  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  HessSp_ = dynamic_cast<hiopMatrixSparse*>(Hess_);
  if(!HessSp_) {
    assert(false);
    return false;
  }

  Jac_cSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_c_);
  if(!Jac_cSp_) {
    assert(false);
    return false;
  }

  Jac_dSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_d_);
  if(!Jac_dSp_) {
    assert(false);
    return false;
  }

  nlp_->runStats.kkt.tmUpdateInit.start();

  /* TODO: here we assume Hess is diagonal!*/
  assert(HessSp_->is_diagonal());

  hiopMatrixSymSparseTriplet* Hess_triplet = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);

  assert(HessSp_ && Jac_cSp_ && Jac_dSp_);
  if(nullptr == Jac_dSp_ || nullptr == HessSp_) {
    nlp_->runStats.kkt.tmUpdateInit.stop();
    // incorrect linear algebra objects were provided to this class
    return false;
  }

  size_type nx = HessSp_->n();
  size_type neq = Jac_cSp_->m();
  size_type nineq = Jac_dSp_->m();
  assert(nineq == Dd_->get_size());
  assert(nx == Dx_->get_size());

  // NOTE:
  // hybrid compute mode -> linear algebra objects used internally by the class will be allocated on the device. Most of the
  // inputs to this class will be however on HOST under hybrid mode, so some objects are copied/replicated/transfered to
  // device gpu compute mode -> not yet supported cpu compute mode -> all objects on HOST, however, some objects will still
  // be copied (e.g., Hd_) to ensure code homogeneity
  //
  // REMARK: The objects that are copied/replicated are temporary and will be removed later on as the remaining sparse KKT
  // computations will be ported to device

  // determine the "internal" memory space, see above note
  std::string mem_space_internal = determine_memory_space_internal(nlp_->options->GetString("compute_mode"));

  // allocate on the first call
  if(nullptr == Hess_diag_) {
    // HOST
    Hess_diag_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nineq);

    Hess_triplet->extract_diagonal(*Hess_diag_);

    assert(nullptr == Hd_copy_);  // should be also not allocated
    // temporary: make a copy of Hd on the "internal" mem_space
    Hess_diag_copy_ = LinearAlgebraFactory::create_vector(mem_space_internal, nx);
    Hx_copy_ = LinearAlgebraFactory::create_vector(mem_space_internal, nx);
    Hd_copy_ = LinearAlgebraFactory::create_vector(mem_space_internal, nineq);
    Hxd_inv_copy_ = LinearAlgebraFactory::create_vector(mem_space_internal, nx + nineq);

    assert(nullptr == deltawx_);  // should be also not allocated
    // allocate this internal vector on the device if hybrid compute mode
    deltawx_ = LinearAlgebraFactory::create_vector(mem_space_internal, nx);
    deltawd_ = LinearAlgebraFactory::create_vector(mem_space_internal, nineq);
    deltacc_ = LinearAlgebraFactory::create_vector(mem_space_internal, neq);
    deltacd_ = LinearAlgebraFactory::create_vector(mem_space_internal, nineq);
    dual_reg_copy_ = LinearAlgebraFactory::create_vector(mem_space_internal, neq + nineq);
  }
  // build the diagonal Hx = Dx + delta_wx + diag(Hess)
  Hx_->copyFrom(*Dx_);
  Hx_->axpy(1., *delta_wx_);
  Hx_->axpy(1., *Hess_diag_);

  // HD = Dd_ + delta_wd
  Hd_->copyFrom(*Dd_);
  Hd_->axpy(1., *delta_wd_);

  // temporary code, see above note
  {
    if(mem_space_internal == "CUDA") {
#ifdef HIOP_USE_CUDA
      auto Hess_diag_cuda = dynamic_cast<hiopVectorCuda*>(Hess_diag_copy_);
      auto Hess_diag_par = dynamic_cast<hiopVectorPar*>(Hess_diag_);
      auto Hx_cuda = dynamic_cast<hiopVectorCuda*>(Hx_copy_);
      auto Hx_par = dynamic_cast<hiopVectorPar*>(Hx_);
      auto Hd_cuda = dynamic_cast<hiopVectorCuda*>(Hd_copy_);
      auto Hd_par = dynamic_cast<hiopVectorPar*>(Hd_);
      assert(Hx_cuda && "incorrect type for vector class");
      assert(Hx_par && "incorrect type for vector class");
      Hess_diag_cuda->copy_from_vectorpar(*Hess_diag_par);
      Hx_cuda->copy_from_vectorpar(*Hx_par);
      Hd_cuda->copy_from_vectorpar(*Hd_par);

      auto deltawx_cuda = dynamic_cast<hiopVectorCuda*>(deltawx_);
      auto deltawd_cuda = dynamic_cast<hiopVectorCuda*>(deltawd_);
      auto deltacc_cuda = dynamic_cast<hiopVectorCuda*>(deltacc_);
      auto deltacd_cuda = dynamic_cast<hiopVectorCuda*>(deltacd_);
      const hiopVectorPar& deltawx_host = dynamic_cast<const hiopVectorPar&>(*delta_wx_);
      const hiopVectorPar& deltawd_host = dynamic_cast<const hiopVectorPar&>(*delta_wd_);
      const hiopVectorPar& deltacc_host = dynamic_cast<const hiopVectorPar&>(*delta_cc_);
      const hiopVectorPar& deltacd_host = dynamic_cast<const hiopVectorPar&>(*delta_cd_);

      deltawx_cuda->copy_from_vectorpar(deltawx_host);
      deltawd_cuda->copy_from_vectorpar(deltawd_host);
      deltacc_cuda->copy_from_vectorpar(deltacc_host);
      deltacd_cuda->copy_from_vectorpar(deltacd_host);
#else
      assert(false && "compute mode not available under current build: enable CUDA.");
#endif
    } else {
      assert(dynamic_cast<hiopVectorPar*>(Hd_) && "incorrect type for vector class");
      Hess_diag_copy_->copyFrom(*Hess_diag_);
      Hx_copy_->copyFrom(*Hx_);
      Hd_copy_->copyFrom(*Hd_);
      deltawx_->copyFrom(*delta_wx_);
      deltawd_->copyFrom(*delta_wd_);
      deltacc_->copyFrom(*delta_cc_);
      deltacd_->copyFrom(*delta_cd_);
    }
  }

  nlp_->runStats.kkt.tmUpdateInit.stop();
  nlp_->runStats.kkt.tmUpdateLinsys.start();

  /*
   *  compute condensed linear system
   * ( [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1} [ Jc^T  Jd^T ] + [ delta_cc     0     ] )
   * ( [ Jd -I ] [   0           Dd+delta_wd ]      [  0     -I  ]   [    0      delta_cd ] )
   */

  // TODO: jump to the steps where we add dual regularization, if delta_wx is not changed and this function is called due to
  // refactorization
  hiopTimer t;

  if(nullptr == JDiagJt_) {
    t.reset();
    t.start();
    // first time this is called
    // form sparse matrix in triplet form on HOST
    size_type nnz_jac_con = Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros() + nineq;
    auto* Jac_triplet_tmp = new hiopMatrixSparseTriplet(neq + nineq, nx + nineq, nnz_jac_con);
    Jac_triplet_tmp->setToZero();

    // build  [ Jc  0 ]
    //        [ Jd -I ]
    // copy Jac to the full iterate matrix
    size_type dest_nnz_st{0};
    Jac_triplet_tmp->copyRowsBlockFrom(*Jac_cSp_, 0, neq, 0, dest_nnz_st);
    dest_nnz_st += Jac_cSp_->numberOfNonzeros();
    Jac_triplet_tmp->copyRowsBlockFrom(*Jac_dSp_, 0, nineq, neq, dest_nnz_st);
    dest_nnz_st += Jac_dSp_->numberOfNonzeros();

    // minus identity matrix for slack variables
    Jac_triplet_tmp->copyDiagMatrixToSubblock(-1., neq, nx, dest_nnz_st, nineq);
    dest_nnz_st += nineq;

    /// TODO: now we assume Jc and Jd won't change, i.e., LP or QP. hence we build JacD_ and JacDt_ once and save them
    Jac_triplet_tmp->sort();

    assert(nullptr == JacD_ && nullptr == JacDt_ && nullptr == JDiagJt_);

    JacD_ = LinearAlgebraFactory::create_matrix_sparse_csr(mem_space_internal);
    JacD_->form_from_symbolic(*Jac_triplet_tmp);
    JacD_->form_from_numeric(*Jac_triplet_tmp);

    JacDt_ = LinearAlgebraFactory::create_matrix_sparse_csr(mem_space_internal);
    JacDt_->form_transpose_from_symbolic(*JacD_);
    JacDt_->form_transpose_from_numeric(*JacD_);  // need this line before calling JacD_->times_mat_alloc(*JacDt_)

    // symbolic multiplication for JacD*Diag*JacDt
    //  J * (D*Jt)  (D is not used since it does not change the sparsity pattern)
    JDiagJt_ = JacD_->times_mat_alloc(*JacDt_);
    JacD_->times_mat_symbolic(*JDiagJt_, *JacDt_);

    delete Jac_triplet_tmp;
  }

  t.reset();
  t.start();

  if(pdreg.get_curr_delta_type() != hiopPDPerturbation::DeltasUpdateType::DualUpdate) {
    // build the diagonal Hxd_inv_copy_ = [H+Dx+delta_wx, Dd+delta_wd ]^{-1}
    Hx_copy_->copyToStarting(*Hxd_inv_copy_, 0);
    Hd_copy_->copyToStarting(*Hxd_inv_copy_, nx);
    Hxd_inv_copy_->invert();
    // J * D * Jt
    JacDt_->form_transpose_from_numeric(*JacD_);
    JacDt_->scale_rows(*Hxd_inv_copy_);
    JacD_->times_mat_numeric(0.0, *JDiagJt_, 1.0, *JacDt_);

#ifdef HIOP_DEEPCHECKS
    JDiagJt_->check_csr_is_ordered();
#endif
  }

  if(nullptr == M_normaleqn_) {
    t.reset();
    t.start();
    Diag_dualreg_ = LinearAlgebraFactory::create_matrix_sparse_csr(mem_space_internal);
    Diag_dualreg_->form_diag_from_symbolic(*dual_reg_copy_);

    // form sparsity pattern of M_normaleqn_ = JacD*Diag*JacDt + delta_dual*I
    M_normaleqn_ = Diag_dualreg_->add_matrix_alloc(*JDiagJt_);
    Diag_dualreg_->add_matrix_symbolic(*M_normaleqn_, *JDiagJt_);
  }

  t.reset();
  t.start();
  Diag_dualreg_->set_diagonal(0.0);
  // if(!delta_cc_in.is_zero() || !delta_cd_in.is_zero()) // TODO: for efficiency?
  {
    deltacc_->copyToStarting(*dual_reg_copy_, 0);
    deltacd_->copyToStarting(*dual_reg_copy_, neq);
    Diag_dualreg_->form_diag_from_numeric(*dual_reg_copy_);
  }

  Diag_dualreg_->add_matrix_numeric(*M_normaleqn_, 1.0, *JDiagJt_, 1.0);

  //
  // instantiate linear solver class based on values of compute_mode, safe mode, and other options
  //
  linSys_ = determine_and_create_linsys();

  nlp_->runStats.kkt.tmUpdateLinsys.stop();

  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "KKT_SPARSE_NormalEqn linsys: Low-level linear system size %d nnz %d\n",
                      neq + nineq,
                      M_normaleqn_->numberOfNonzeros());
  }

  // write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_ >= 0) {
    // TODO csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nx, 0, nineq);
  }

  return true;
}

hiopLinSolverSymSparse* hiopKKTLinSysSparseNormalEqn::determine_and_create_linsys()
{
  if(linSys_) {
    return dynamic_cast<hiopLinSolverSymSparse*>(linSys_);
  }

  size_type n = M_normaleqn_->m();
  auto linsolv = nlp_->options->GetString("linear_solver_sparse");
  if(nlp_->options->GetString("compute_mode") == "cpu") {
    // TODO:
    // add support for linear_solver == "cholmod"
    //  maybe add pardiso as an option in the future
    //
    assert((linsolv == "ma57" || linsolv == "auto") && "Only MA57 or auto is supported on cpu.");

#ifdef HIOP_USE_COINHSL
    nlp_->log->printf(hovWarning, "KKT_SPARSE_NormalEqn linsys: alloc MA57 for matrix of size %d (0 cons)\n", n);

    // we need to get CPU CSR matrix
    auto* M_csr = dynamic_cast<hiopMatrixSparseCSRSeq*>(M_normaleqn_);
    assert(M_csr);
    linSys_ = new hiopLinSolverSparseCsrMa57(M_csr, nlp_);
#else
    assert(false && "HiOp was built without a sparse linear solver needed by the condensed KKT approach");
#endif  // HIOP_USE_COINHSL

  } else {
    //
    // on device: compute_mode is hybrid, auto, or gpu
    //
    assert(nullptr == linSys_);

    assert((linsolv == "cusolver-chol" || linsolv == "auto") && "Only cusolver-chol or auto is supported on gpu.");

#ifdef HIOP_USE_CUDA
    nlp_->log->printf(hovWarning, "KKT_SPARSE_NormalEqn linsys: alloc cuSOLVER-chol matrix size %d\n", n);
    assert(M_normaleqn_);
    linSys_ = new hiopLinSolverCholCuSparse(M_normaleqn_, nlp_);
#endif

    // Return NULL (and assert) if a GPU sparse linear solver is not present
    assert(linSys_ != nullptr &&
           "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
           "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to 'cpu'");
  }
  assert(linSys_ && "KKT_SPARSE_NormalEqn linsys: cannot instantiate backend linear solver");

  return dynamic_cast<hiopLinSolverSymSparse*>(linSys_);
}

bool hiopKKTLinSysSparseNormalEqn::solveCompressed(hiopVector& ryc, hiopVector& ryd, hiopVector& dyc, hiopVector& dyd)
{
  bool bret{false};

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  size_type nyc = ryc.get_size();
  size_type nyd = ryd.get_size();

  // this is rhs used by the direct "condensed" solve
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nyc + nyd);
  }

  nlp_->log->write("RHS KKT_SPARSE_NormalEqn ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_NormalEqn ryd:", ryd, hovIteration);

  //
  // form the rhs for the sparse linSys
  //
  ryc.copyToStarting(*rhs_, 0);
  ryd.copyToStarting(*rhs_, nyc);

  if(write_linsys_counter_ >= 0) {
    csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);
  }
  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  //
  // solve
  //
  nlp_->runStats.kkt.tmSolveInner.start();
  bret = linSys_->solve(*rhs_);
  nlp_->runStats.kkt.tmSolveInner.stop();

  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "(summary for linear solver from KKT_SPARSE_NormalEqn(direct))\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }
  if(write_linsys_counter_ >= 0) {
    csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
  }

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  //
  // unpack
  //
  rhs_->startingAtCopyToStartingAt(0, dyc, 0);
  rhs_->startingAtCopyToStartingAt(nyc, dyd, 0);
  nlp_->log->write("SOL KKT_SPARSE_NormalEqn dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_NormalEqn dyd:", dyd, hovMatrices);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  return bret;
}

int hiopKKTLinSysSparseNormalEqn::factorizeWithCurvCheck()
{
  // factorization
  size_type n_neg_eig = hiopKKTLinSysCurvCheck::factorizeWithCurvCheck();

  if(n_neg_eig == -1) {
    nlp_->log->printf(hovScalars, "KKT_SPARSE_NormalEqn linsys: Detected null eigenvalues.\n");
    n_neg_eig = -1;
  } else {
    // Cholesky factorization succeeds. Matrix is PD and hence the corresponding Augmented system has correct inertia
    n_neg_eig = Jac_c_->m() + Jac_d_->m();
    ;
  }

  return n_neg_eig;
}

}  // namespace hiop
