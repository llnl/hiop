// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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

#include "hiopKKTLinSysMDS.hpp"
#include "hiopLinSolverSymDenseLapack.hpp"

#ifdef HIOP_USE_MAGMA
#include "hiopLinSolverSymDenseMagma.hpp"
#endif

namespace hiop
{

hiopKKTLinSysCompressedMDSXYcYd::hiopKKTLinSysCompressedMDSXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp),
      rhs_(NULL),
      _buff_xs_(NULL),
      Hxs_(NULL),
      Hxs_wrk_(nullptr),
      HessMDS_(NULL),
      Jac_cMDS_(NULL),
      Jac_dMDS_(NULL),
      write_linsys_counter_(-1),
      csr_writer_(nlp)
{
  nlpMDS_ = dynamic_cast<hiopNlpMDS*>(nlp_);
  assert(nlpMDS_);
}

hiopKKTLinSysCompressedMDSXYcYd::~hiopKKTLinSysCompressedMDSXYcYd()
{
  delete rhs_;
  delete _buff_xs_;
  delete Hxs_;
  delete Hxs_wrk_;
}

int hiopKKTLinSysCompressedMDSXYcYd::factorizeWithCurvCheck()
{
  // factorization
  int n_neg_eig = hiopKKTLinSysCurvCheck::factorizeWithCurvCheck();

  int n_neg_eig_11 = 0;
  if(n_neg_eig >= 0) {
    // 'n_neg_eig' is the number of negative eigenvalues of the "dense" (reduced) KKT
    //
    // One can compute the number of negative eigenvalues of the whole MDS or XYcYd
    // linear system using Haynsworth inertia additivity formula, namely,
    // count the negative eigenvalues of the sparse Hessian block.
    int n_neg_eig_Hxs = Hxs_->numOfElemsLessThan(-1e-14);
    int n_zero_eig_Hxs = Hxs_->numOfElemsAbsLessThan(1e-14);
    n_neg_eig_11 += n_neg_eig_Hxs;
    if(n_zero_eig_Hxs > 0) {
      n_neg_eig_11 = -1;
    }
  }

  if(n_neg_eig_11 < 0) {
    nlp_->log->printf(hovWarning, "KKT_MDS_XYcYd linsys: Detected null eigenvalues in (1,1) sparse block.\n");
    assert(n_neg_eig_11 == -1);
    n_neg_eig = -1;
  } else if(n_neg_eig_11 > 0) {
    n_neg_eig += n_neg_eig_11;
    nlp_->log->printf(hovScalars, "KKT_MDS_XYcYd linsys: Detected negative eigenvalues in (1,1) sparse block.\n");
  }
  return n_neg_eig;
}

bool hiopKKTLinSysCompressedMDSXYcYd::update(const hiopIterate* iter,
                                             const hiopVector* grad_f,
                                             const hiopMatrix* Jac_c,
                                             const hiopMatrix* Jac_d,
                                             hiopMatrix* Hess)
{
  if(!nlpMDS_) {
    assert(false);
    return false;
  }

  nlp_->runStats.linsolv.reset();
  nlp_->runStats.tmSolverInternal.start();
  nlp_->runStats.kkt.tmUpdateInit.start();

  iter_ = iter;
  grad_f_ = grad_f;
  Jac_c_ = Jac_c;
  Jac_d_ = Jac_d;
  Hess_ = Hess;

  HessMDS_ = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(Hess);
  if(!HessMDS_) {
    assert(false);
    return false;
  }

  Jac_cMDS_ = dynamic_cast<const hiopMatrixMDS*>(Jac_c);
  if(!Jac_cMDS_) {
    assert(false);
    return false;
  }

  Jac_dMDS_ = dynamic_cast<const hiopMatrixMDS*>(Jac_d);
  if(!Jac_dMDS_) {
    assert(false);
    return false;
  }

  int nxs = HessMDS_->n_sp(), nxd = HessMDS_->n_de(), nx = HessMDS_->n();
  int neq = Jac_cMDS_->m(), nineq = Jac_dMDS_->m();

  assert(nx == nxs + nxd);
  assert(nx == Jac_cMDS_->n_sp() + Jac_cMDS_->n_de());
  assert(nx == Jac_dMDS_->n_sp() + Jac_dMDS_->n_de());

  //
  // based on safe_mode_, decide whether to go with the nopiv (fast) or Bunch-Kaufman (stable) linear solve
  //
  linSys_ = determineAndCreateLinsys(nxd, neq, nineq);

  //
  // update/compute KKT
  //

  // Dx (<-- log-barrier diagonal, for both sparse (Dxs) and dense (Dxd)
  assert(Dx_->get_local_size() == nxs + nxd);
  Dx_->setToZero();
  Dx_->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp_->get_ixl());
  Dx_->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp_->get_ixu());
  nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

  nlp_->runStats.kkt.tmUpdateInit.stop();

  //
  // factorization + inertia correction if needed
  //
  const bool retval = factorize();

  nlp_->runStats.tmSolverInternal.stop();
  return retval;
}

bool hiopKKTLinSysCompressedMDSXYcYd::build_kkt_matrix(const hiopPDPerturbation& pdreg)
{
  assert(linSys_);
  hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);
  assert(linSys);

  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  int nxs = HessMDS_->n_sp(), nxd = HessMDS_->n_de();
  int neq = Jac_cMDS_->m(), nineq = Jac_dMDS_->m();

  hiopMatrixDense& Msys = linSys->sysMatrix();
  if(perf_report_) {
    nlp_->log->printf(hovSummary, "KKT_MDS_XYcYd linsys: Low-level linear system size: %d\n", Msys.n());
  }

  nlp_->runStats.kkt.tmUpdateLinsys.start();

  // update linSys system matrix, including IC perturbations
  Msys.setToZero();

  int alpha = 1.;

  // perf eval
  // hiopTimer tm;
  // tm.start();

  HessMDS_->de_mat()->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);
  Jac_cMDS_->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd, alpha, Msys);
  Jac_dMDS_->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd + neq, alpha, Msys);

  // tm.stop();
  // printf("the three add methods took %g sec\n", tm.getElapsedTime());
  // tm.reset();

  // update -> add Dxd to (1,1) block of KKT matrix (Hd = HessMDS_->de_mat already added above)
  Msys.addSubDiagonal(0, alpha, *Dx_, nxs, nxd);
  // add perturbation 'delta_wx' for xd
  Msys.addSubDiagonal(0, alpha, *delta_wx_, nxs, nxd);

  // build the diagonal Hxs = Hsparse+Dxs
  if(NULL == Hxs_) {
    Hxs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nxs);
    Hxs_wrk_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nxs);
    assert(Hxs_);
  }
  Hxs_->startingAtCopyFromStartingAt(0, *Dx_, 0);

  // a good time to add the IC 'delta_wx' perturbation
  Hxs_wrk_->startingAtCopyFromStartingAt(0, *delta_wx_, 0);
  Hxs_->axpy(1., *Hxs_wrk_);

  // Hxs +=  diag(HessMDS->sp_mat());
  // todo: make sure we check that the HessMDS->sp_mat() is a diagonal
  HessMDS_->sp_mat()->startingAtAddSubDiagonalToStartingAt(0, alpha, *Hxs_, 0);
  nlp_->log->write("Hxs in KKT_MDS_X", *Hxs_, hovMatrices);

  // add - Jac_c_sp * (Hxs)^{-1} Jac_c_sp^T to diagonal block linSys starting at (nxd, nxd)
  alpha = -1.;

  // perf eval
  // tm.start();
  Jac_cMDS_->sp_mat()->addMDinvMtransToDiagBlockOfSymDeMatUTri(nxd, alpha, *Hxs_, Msys);

  // tm.stop();
  // printf("addMDinvMtransToDiagBlockOfSymDeMatUTri 111 took %g sec\n", tm.getElapsedTime());
  // tm.reset();

  Msys.addSubDiagonal(-1., nxd, *delta_cc_);

  /* we've just done above the (1,1) and (2,2) blocks of
   *
   * [ Hd+Dxd+delta_wx*I           Jcd^T                                 Jdd^T  ]
   * [  Jcd              -Jcs(Hs+Dxs+delta_wx*I)^{-1}Jcs^T-delta_cc*I    K_21   ]
   * [  Jdd                        K_21                                  M_{33} ]
   *
   * where
   * K_21 = - Jcs * (Hs+Dxs+delta_wx)^{-1} * Jds^T
   *
   * M_{33} = -Jds(Hs+Dxs+delta_wx)^{-1}Jds^T - (Dd+delta_wd)*I^{-1} - delta_cd*I
   *   is performed below
   */

  alpha = -1.;
  // add   - Jac_d_sp * (Hxs+Dxs+delta_wx*I)^{-1} * Jac_d_sp^T   to diagonal block
  // linSys starting at (nxd+neq, nxd+neq)

  // perf eval
  // tm.start();

  Jac_dMDS_->sp_mat()->addMDinvMtransToDiagBlockOfSymDeMatUTri(nxd + neq, alpha, *Hxs_, Msys);

  // tm.stop();
  // printf("addMDinvMtransToDiagBlockOfSymDeMatUTri 222 took %g sec\n", tm.getElapsedTime());

  // K_21 = - Jcs * (Hs+Dxs+delta_wx)^{-1} * Jds^T
  alpha = -1.;
  Jac_cMDS_->sp_mat()->addMDinvNtransToSymDeMatUTri(nxd, nxd + neq, alpha, *Hxs_, *Jac_dMDS_->sp_mat(), Msys);

  // add -{Dd}^{-1}
  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu + delta_wd * I
  Dd_inv_->copyFrom(*delta_wd_);
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
  Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
  assert(true == Dd_inv_->allPositive());
#endif
  Dd_inv_->invert();

  alpha = -1.;
  Msys.addSubDiagonal(alpha, nxd + neq, *Dd_inv_);
  Msys.addSubDiagonal(alpha, nxd + neq, *delta_cd_);

  nlp_->log->write("KKT_MDS_XYcYd linsys:", Msys, hovMatrices);

  nlp_->runStats.kkt.tmUpdateLinsys.stop();

  // write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_ >= 0) {
    csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nxd + nxs, neq, nineq);
  }

  return true;
}

bool hiopKKTLinSysCompressedMDSXYcYd::solveCompressed(hiopVector& rx,
                                                      hiopVector& ryc,
                                                      hiopVector& ryd,
                                                      hiopVector& dx,
                                                      hiopVector& dyc,
                                                      hiopVector& dyd)
{
  hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);

  if(!nlpMDS_) {
    assert(false);
    return false;
  }
  if(!HessMDS_) {
    assert(false);
    return false;
  }
  if(!Jac_cMDS_) {
    assert(false);
    return false;
  }
  if(!Jac_dMDS_) {
    assert(false);
    return false;
  }

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  int nx = rx.get_size(), nyc = ryc.get_size(), nyd = ryd.get_size();
  int nxsp = Hxs_->get_size();
  assert(nxsp <= nx);
  int nxde = nlpMDS_->nx_de();
  assert(nxsp + nxde == nx);
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nxde + nyc + nyd);
  }
  if(_buff_xs_ == NULL) {
    _buff_xs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nxsp);
  }

  nlp_->log->write("RHS KKT_MDS_XYcYd rx: ", rx, hovIteration);
  nlp_->log->write("RHS KKT_MDS_XYcYd ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_MDS_XYcYd ryd:", ryd, hovIteration);

  hiopVector& rxs = *_buff_xs_;
  // rxs = Hxs^{-1} * rx_sparse
  rx.startingAtCopyToStartingAt(0, rxs, 0, nxsp);
  rxs.componentDiv(*Hxs_);

  // ryc = ryc - Jac_c_sp * Hxs^{-1} * rxs
  // use dyc as working buffer to avoid altering ryc, which refers directly in the hiopResidual class
  assert(dyc.get_size() == ryc.get_size());
  dyc.copyFrom(ryc);
  Jac_cMDS_->sp_mat()->timesVec(1.0, dyc, -1., rxs);

  // ryd = ryd - Jac_d_sp * Hxs^{-1} * rxs
  Jac_dMDS_->sp_mat()->timesVec(1.0, ryd, -1., rxs);

  //
  // form the rhs for the MDS linSys
  //
  // rhs[0:nxde-1] = rx[nxs:(nxsp+nxde-1)]
  rx.startingAtCopyToStartingAt(nxsp, *rhs_, 0, nxde);
  // rhs[nxde:nxde+nyc-1] = ryc
  dyc.copyToStarting(*rhs_, nxde);
  // ths[nxde+nyc:nxde+nyc+nyd-1] = ryd
  ryd.copyToStarting(*rhs_, nxde + nyc);

  if(write_linsys_counter_ >= 0) {
    csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);
  }

  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  nlp_->runStats.kkt.tmSolveInner.start();

  // solve
  bool linsol_ok = linSys->solve(*rhs_);
  nlp_->runStats.kkt.tmSolveInner.stop();

  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "(summary for linear solver from KKT_MDS_XYcYd)\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }

  if(write_linsys_counter_ >= 0) {
    csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
  }
  if(false == linsol_ok) return false;

  nlp_->runStats.kkt.tmSolveRhsManip.start();

  // unpack
  rhs_->startingAtCopyToStartingAt(0, dx, nxsp, nxde);
  rhs_->startingAtCopyToStartingAt(nxde, dyc, 0);
  rhs_->startingAtCopyToStartingAt(nxde + nyc, dyd, 0);

  // compute dxs
  hiopVector& dxs = *_buff_xs_;
  // dxs = (Hxs)^{-1} ( rxs - Jac_c_sp^T dyc - Jac_d_sp^T dyd)
  rx.startingAtCopyToStartingAt(0, dxs, 0, nxsp);
  Jac_cMDS_->sp_mat()->transTimesVec(1., dxs, -1., dyc);
  Jac_dMDS_->sp_mat()->transTimesVec(1., dxs, -1., dyd);
  dxs.componentDiv(*Hxs_);
  // copy to dx
  dxs.startingAtCopyToStartingAt(0, dx, 0);

  nlp_->log->write("SOL KKT_MDS_XYcYd dx: ", dx, hovMatrices);
  nlp_->log->write("SOL KKT_MDS_XYcYd dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_MDS_XYcYd dyd:", dyd, hovMatrices);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  return true;
}

hiopLinSolverSymDense* hiopKKTLinSysCompressedMDSXYcYd::determineAndCreateLinsys(int nxd, int neq, int nineq)
{
#ifdef HIOP_USE_MAGMA
  bool switched_linsolvers = false;
  if(safe_mode_) {
    hiopLinSolverSymDenseMagmaBuKa* p = dynamic_cast<hiopLinSolverSymDenseMagmaBuKa*>(linSys_);
    if(p == NULL) {
      // we have a nopiv linear solver or linear solver has not been created yet
      if(linSys_) switched_linsolvers = true;
      delete linSys_;
      linSys_ = NULL;
    } else {
      return p;
    }
  } else {
    hiopLinSolverSymDenseMagmaNopiv* p = dynamic_cast<hiopLinSolverSymDenseMagmaNopiv*>(linSys_);
    if(p == NULL) {
      // we have a BuKa linear solver or linear solver has not been created yet
      if(linSys_) switched_linsolvers = true;
      delete linSys_;
      linSys_ = NULL;
    } else {
      return p;
    }
  }
#endif

  if(NULL == linSys_) {
    int n = nxd + neq + nineq;

    if("cpu" == nlp_->options->GetString("compute_mode")) {
      nlp_->log->printf(hovScalars, "KKT_MDS_XYcYd linsys: Lapack for a matrix of size %d [1]\n", n);
      linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
      return dynamic_cast<hiopLinSolverSymDense*>(linSys_);
    }

#ifdef HIOP_USE_MAGMA
    if(nlp_->options->GetString("compute_mode") == "hybrid" || nlp_->options->GetString("compute_mode") == "gpu" ||
       nlp_->options->GetString("compute_mode") == "auto") {
      if(safe_mode_) {
        auto hovLevel = hovScalars;
        if(switched_linsolvers) hovLevel = hovWarning;

        nlp_->log->printf(hovLevel,
                          "KKT_MDS_XYcYd linsys: MagmaBuKa size %d (%d cons) (safe_mode=%d)\n",
                          n,
                          neq + nineq,
                          safe_mode_);

        linSys_ = new hiopLinSolverSymDenseMagmaBuKa(n, nlp_);
      } else {
        auto hovLevel = hovScalars;
        if(switched_linsolvers) hovLevel = hovWarning;

        nlp_->log->printf(hovLevel,
                          "KKT_MDS_XYcYd linsys: MagmaNopiv size %d (%d cons) (safe_mode=%d)\n",
                          n,
                          neq + nineq,
                          safe_mode_);

        linSys_ = new hiopLinSolverSymDenseMagmaNopiv(n, nlp_);
        // hiopLinSolverSymDenseMagmaNopiv* p = new hiopLinSolverSymDenseMagmaNopiv(n, nlp_);
        // linSys_ = p;
        // p->set_fake_inertia(neq + nineq);
      }
    } else {
      nlp_->log->printf(hovScalars, "KKT_MDS_XYcYd linsys: Lapack for a matrix of size %d [2]\n", n);
      linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
      return dynamic_cast<hiopLinSolverSymDense*>(linSys_);
    }
#else
    nlp_->log->printf(hovScalars, "KKT_MDS_XYcYd linsys: Lapack for a matrix of size %d [3]\n", n);
    linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
    return dynamic_cast<hiopLinSolverSymDense*>(linSys_);
#endif
  }
  return dynamic_cast<hiopLinSolverSymDense*>(linSys_);
}

}  // namespace hiop
