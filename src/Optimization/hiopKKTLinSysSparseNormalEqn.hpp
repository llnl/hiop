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
 * @file hiopKKTLinSysSparseNormalEqn.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 */

#ifndef HIOP_KKTLINSYSSPARSE_NORMALEQ
#define HIOP_KKTLINSYSSPARSE_NORMALEQ

#include "hiopKKTLinSysSparse.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "hiopMatrixSparseCSR.hpp"
#include "hiopKrylovSolver.hpp"

namespace hiop
{

/**
 * @brief Provides the functionality for reducing the KKT linear system to the
 * normal equation system below in dyc and dyd variables and then to perform
 * the basic ops needed to compute the remaining directions
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed
 * linear system
 * ( [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1} [ Jc^T  Jd^T ] + [ delta_cc     0     ] ) [dyc] = [ ryc_tilde ]
 * ( [ Jd -I ] [   0           Dd+delta_wd ]      [  0     -I  ]   [    0      delta_cd ] ) [dyd]   [ ryd_tilde ]
 *
 * [ ryc_tilde ] = [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1}  [ rx_tilde ] - [ ryc ]
 * [ ryd_tilde ]   [ Jd -I ] [   0           Dd+delta_wd ]       [ rd_tilde ]   [ ryd ]
 *
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *  - Dx is diagonal corresponding to x variable in the log-barrier diagonal Dx, respectively
 *
 * REMARK: This linear system fits LP/QP best, where H is empty and hence only diagonal matrices are inversed.
 * If H is diagonal, the normal equation matrix becomes:
 *   [ Jc(H+Dx+delta_wx)^{-1}Jc^T    Jc(H+Dx+delta_wx)^{-1}Jd^T ]                        + [ delta_cc     0     ]
 *   [ Jd(H+Dx+delta_wx)^{-1}Jc^T    Jd(H+Dx+delta_wx)^{-1}Jd^T + ( Dd+delta_wd)^{-1} ]    [    0      delta_cd ]
 *
 */
class hiopKKTLinSysSparseNormalEqn : public hiopKKTLinSysNormalEquation
{
public:
  hiopKKTLinSysSparseNormalEqn(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysSparseNormalEqn();

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

  virtual bool solveCompressed(hiopVector& ryc_tilde, hiopVector& ryd_tilde, hiopVector& dyc, hiopVector& dyd);

  /**
   * @brief factorize the matrix and check curvature
   */
  virtual int factorizeWithCurvCheck();

protected:
  hiopVector* rhs_;  // [ryc_tilde, ryd_tilde]

  // diagOf(Hess)
  hiopVector* Hess_diag_;

  /// Stores Dx plus delta_wx for more efficient updates of the condensed system matrix
  hiopVector* deltawx_;
  hiopVector* deltawd_;
  hiopVector* deltacc_;
  hiopVector* deltacd_;
  hiopVector* dual_reg_copy_;  // a vector for dual regularizations

  /// Stores some copies on the device (to be later removed)
  hiopVector* Hess_diag_copy_;
  hiopVector* Hx_copy_;
  hiopVector* Hd_copy_;
  hiopVector* Hxd_inv_copy_;  // [H+Dx+delta_wx, Dd+delta_wd ]^-1

  // -1 when disabled; otherwise acts like a counter, 0,1,... incremented each time
  // 'solveCompressed' is called; activated by the 'write_kkt' option
  int write_linsys_counter_;
  hiopCSR_IO csr_writer_;

  // just dynamic_cast-ed pointers
  hiopNlpSparse* nlpSp_;
  hiopMatrixSparse* HessSp_;
  const hiopMatrixSparse* Jac_cSp_;
  const hiopMatrixSparse* Jac_dSp_;

  /**
   * Member for ( [ Jc  0 ] [ H+Dx+delta_wx     0       ]^{-1} [ Jc^T  Jd^T ] + [ delta_cc     0     ] )
   *            ( [ Jd -I ] [   0           Dd+delta_wd ]      [  0     -I  ]   [    0      delta_cd ] )
   * let JacD_ = [Jc 0; Jd -I]
   * @pre: now we assume Jc and Jd won't change, i.e., LP or QP. hence we build JacD_ and JacDt_ once and save them
   */

  /// Member for JacD in CSR format
  hiopMatrixSparseCSR* JacD_;

  /// Member for JacD' in CSR format
  hiopMatrixSparseCSR* JacDt_;

  /// Member for JacD*Dd*JacD'
  hiopMatrixSparseCSR* JDiagJt_;

  /// Member for delta_cc*I
  hiopMatrixSparseCSR* Diag_dualreg_;

  /// Member forJacD*Dd*JacD' - delta_cc*I
  hiopMatrixSparseCSR* M_normaleqn_;

private:
  // placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverSymSparse* determine_and_create_linsys();

  /// Determines memory space used internally based on the "mem_space" and "compute_mode" options. This is temporary
  /// functionality and will be removed later on when all the objects will be in the same memory space.
  inline std::string determine_memory_space_internal(const std::string& opt_compute_mode)
  {
    if(opt_compute_mode == "cpu" || opt_compute_mode == "auto") {
      return "DEFAULT";
    } else {
      //(opt_compute_mode == "hybrid" || opt_compute_mode == "gpu") {
#ifdef HIOP_USE_CUDA
      assert(opt_compute_mode != "gpu" && "When code is GPU-ready, remove this method");
      return "CUDA";
#else
      assert(false && "compute mode not supported without HIOP_USE_CUDA build");
      return "DEFAULT";
#endif  // HIOP_USE_CUDA
    }
  }
};

}  // namespace hiop

#endif
