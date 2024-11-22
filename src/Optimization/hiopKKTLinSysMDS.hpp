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

#ifndef HIOP_KKTLINSYSMDS
#define HIOP_KKTLINSYSMDS

#include "hiopKKTLinSys.hpp"
#include "hiopLinSolver.hpp"

#include "hiopCSR_IO.hpp"

namespace hiop
{

/*
 * Solves KKTLinSysCompressedXYcYd by exploiting the mixed dense-sparse (MDS)
 * structure of the problem
 *
 * In general, the so-called XYcYd system has the form
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 *
 * For MDS structure, the above linear system is exactly
 * [  Hs  +  Dxs    0       Jcs^T   Jds^T   ] [dxs]   [ rxs_tilde ]
 * [     0        Hd+Dxd    Jcd^T   Jdd^T   ] [dxd]   [ rxd_tilde ]
 * [    Jcs        Jcd        0       0     ] [dyc] = [   ryc    ]
 * [    Jds        Jdd        0    -Dd^{-1} ] [dyd]   [ ryd_tilde]
 * where
 *  - Jcs and Jds contain the sparse columns of the Jacobians Jc and Jd
 *  - Jcd and Jdd contain the dense  columns of the Jacobians Jc and Jd
 *  - Hs is a diagonal matrix (sparse part of the Hessian)
 *  - Hd is the dense part of the Hessian
 *  - Dxs and Dxd are diagonals corresponding to sparse (xs) and dense (xd)
 * variables in the log-barrier diagonal Dx, respectively
 *
 * 'solveCompressed' performs a reduction to
 * [ Hd+Dxd               Jcd^T                          Jdd^T              ] [dxd]
 * [  Jcd       -Jcs(Hs+Dxs)^{-1}Jcs^T                   K_21               ] [dyc] =
 * [  Jdd                 K_21^T             -Jds(Hs+Dxs)^{-1}Jds^T-Dd^{-1} ] [dyd]
 *
 *                                              [ rxd_tilde                             ]
 *                                          =   [ ryc       - Jcs(Hs+Dxs)^{-1}rxs_tilde ]
 *                                              [ ryd_tilde - Jds(Hs+Dxs)^{-1}rxs_tilde ]
 * where
 * K_21 = - Jcs * (Hs+Dxs)^{-1} * Jds^T
 *
 * Then get dxs from
 * dxs = (Hs+Dxs)^{-1}[rxs_tilde - Jcs^T dyc - Jds^T dyd]
 */
class hiopKKTLinSysCompressedMDSXYcYd : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysCompressedMDSXYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedMDSXYcYd();

  virtual int factorizeWithCurvCheck();

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c,
                      const hiopMatrix* Jac_d,
                      hiopMatrix* Hess);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dyc,
                               hiopVector& dyd);

protected:
  hiopVector* rhs_;       //[rxdense, ryc, ryd]
  hiopVector* _buff_xs_;  // an auxiliary buffer

  //
  // from the parent class we also use
  //
  //  hiopVectorPar *Dd_inv;
  //  hiopVectorPar *ryd_tilde;

  // from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //   hiopVectorPar *Dx;
  //   hiopVectorPar *rx_tilde;

  // Keeps Hxs = HessMDS->sp_mat() + Dxs (Dx=log-barrier diagonal for xs)
  hiopVector* Hxs_;
  hiopVector* Hxs_wrk_;

  // just dynamic_cast-ed pointers
  hiopNlpMDS* nlpMDS_;
  hiopMatrixSymBlockDiagMDS* HessMDS_;
  const hiopMatrixMDS* Jac_cMDS_;
  const hiopMatrixMDS* Jac_dMDS_;

  // -1 when disabled; otherwise acts like a counter, 0,1,... incremented each time
  // 'solveCompressed' is called; activated by the 'write_kkt' option
  int write_linsys_counter_;
  hiopCSR_IO csr_writer_;

private:
  // placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverSymDense* determineAndCreateLinsys(int nxd, int neq, int nineq);
};

}  // namespace hiop

#endif
