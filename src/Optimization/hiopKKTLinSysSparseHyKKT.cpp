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
 * @file hiopKKTLinSysSparseHyKKT.cpp
 *
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 * @author Slaven Peles <peless@ornl.gov>
 *
 * @brief XDYcYd KKT system solved with ReSolve's HyKKT.
 */

#include "hiopKKTLinSysSparseHyKKT.hpp"

#include "hiopLinSolverSparseHyKKT.hpp"

namespace hiop
{

/* *************************************************************************
 * For class hiopKKTLinSysCompressedSparseXDYcYdHyKKT
 * *************************************************************************
 */
hiopKKTLinSysCompressedSparseXDYcYdHyKKT::hiopKKTLinSysCompressedSparseXDYcYdHyKKT(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedSparseXDYcYd(nlp)
{}

hiopKKTLinSysCompressedSparseXDYcYdHyKKT::~hiopKKTLinSysCompressedSparseXDYcYdHyKKT() {}

bool hiopKKTLinSysCompressedSparseXDYcYdHyKKT::build_kkt_matrix(const hiopPDPerturbation& pdreg)
{
  delta_wx_ = perturb_calc_->get_curr_delta_wx();
  delta_wd_ = perturb_calc_->get_curr_delta_wd();
  delta_cc_ = perturb_calc_->get_curr_delta_cc();
  delta_cd_ = perturb_calc_->get_curr_delta_cd();

  HessSp_ = dynamic_cast<hiopMatrixSparse*>(Hess_);
  Jac_cSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_c_);
  Jac_dSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_d_);

  if(!HessSp_ || !Jac_cSp_ || !Jac_dSp_) {
    assert(false);
    return false;
  }

  const size_type nx = HessSp_->n();
  const size_type nd = Jac_dSp_->m();

  nlp_->runStats.kkt.tmUpdateLinsys.start();

  // HyKKT consumes the KKT blocks directly; only the regularized diagonals
  // are assembled here.
  if(nullptr == Hx_) {
    Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
    assert(Hx_);
  }
  Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);
  Hx_->axpy(1., *delta_wx_);

  if(nullptr == Hd_) {
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nd);
    assert(Hd_);
  }
  Hd_->startingAtCopyFromStartingAt(0, *Dd_, 0);
  Hd_->axpy(1., *delta_wd_);

  nlp_->runStats.kkt.tmUpdateLinsys.stop();

  // HyKKT currently has no matrix blocks for the dual regularization terms.
  if(delta_cc_->infnorm() != 0.0 || delta_cd_->infnorm() != 0.0) {
    nlp_->log->printf(hovError, "ReSolve HyKKT does not support nonzero dual regularization.\n");
    return false;
  }

  if(nullptr == linSys_) {
    auto* fact_acceptor_ic = dynamic_cast<hiopFactAcceptorIC*>(fact_acceptor_);
    if(fact_acceptor_ic) {
      nlp_->log->printf(hovError,
                        "KKT_SPARSE_XDYcYd linsys with HyKKT does not support inertia correction. "
                        "Please set option 'fact_acceptor' to 'inertia_free'.\n");
      assert(false);
      return false;
    }

    auto* hykkt = new hiopLinSolverSparseHyKKT(nlp_);
    hykkt->set_kkt_blocks(HessSp_, Jac_cSp_, Jac_dSp_, Hx_, Hd_);
    linSys_ = hykkt;

    nlp_->log->printf(hovScalars,
                      "KKT_SPARSE_XDYcYd linsys: alloc [HyKKT] size %d (%d cons)(%s)\n",
                      static_cast<int>(nx + nd + Jac_cSp_->m() + nd),
                      static_cast<int>(Jac_cSp_->m() + nd),
                      nlp_->options->GetString("compute_mode").c_str());
  }

  return true;
}

int hiopKKTLinSysCompressedSparseXDYcYdHyKKT::factorizeWithCurvCheck()
{
  assert(linSys_);
  // HyKKT factorizes inside solve(); matrixChanged() refreshes the block values.
  return linSys_->matrixChanged();
}

}  // namespace hiop
