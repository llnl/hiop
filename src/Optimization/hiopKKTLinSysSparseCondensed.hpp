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
 * @file hiopKKTLinSysSparseCondensed.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 */

#ifndef HIOP_KKTLINSYSSPARSECONDENSED
#define HIOP_KKTLINSYSSPARSECONDENSED

#include "hiopKKTLinSysSparse.hpp"
#include "hiopMatrixSparseTriplet.hpp"
#include "hiopMatrixSparseCSR.hpp"
#include "hiopKrylovSolver.hpp"

namespace hiop
{

/**
 * Solves a sparse KKT linear system by exploiting the sparse structure, namely reduces
 * the so-called XDYcYd KKT system
 * [  H  +  Dx    0    Jd^T ] [ dx]   [ rx_tilde ]
 * [    0         Dd   -I   ] [ dd] = [ rd_tilde ]
 * [    Jd       -I     0   ] [dyd]   [   ryd    ]
 * into the condensed KKT system
 * (H+Dx+Jd^T*Dd*Jd)dx = rx_tilde + Jd^T*Dd*ryd + Jd^T*rd_tilde
 * dd = Jd*dx - ryd
 * dyd = Dd*dd - rd_tilde = Dd*Jd*dx - Dd*ryd - rd_tilde

 * Here Jd is sparse Jacobians for inequalities, H is a sparse Hessian matrix, Dx is
 * log-barrier diagonal corresponding to x variables, Dd is the log-barrier diagonal
 * corresponding to the inequality slacks, and I is the identity matrix.
 *
 * @note: the NLP is assumed to have no equality constraints (or have been relaxed to
 * two-sided inequality constraints).
 *
 * Dual regularization may be not enforced as it requires repeated divisions that are
 * prone to round-off error accumulation. When/If the class is going to be updated to
 * use dual regularization, the regularized XDYcYd KKT system reads:
 * [  H+Dx+delta_wx*I         0         Jd^T     ] [ dx]   [ rx_tilde ]
 * [          0         Dd+delta_wd*I   -I       ] [ dd] = [ rd_tilde ]
 * [          Jd             -I         -delta_cd] [dyd]   [   ryd    ]
 *
 * (notation) Dd2 = [ I+delta_cd*(Dd+delta_wd*I) ]^{-1}
 * (notation) Dd3 = Dd2*(Dd+delta_wd*I)
 *
 * dd = Jd*dx - delta_cd*dyd - ryd
 *
 * From (Dd+delta_wd*I)*dd - dyd = rd_tilde one can write
 *   ->   (Dd+delta_wd*I)*(Jd*dx - delta_cd*dyd - ryd) - dyd = rd_tilde
 *   ->   [I+delta_cd*(Dd+delta_wd*I)] dyd = (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde
 * dyd = (I+delta_cd*(Dd+delta_wd*I))^{-1} [ (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde ]
 * dyd =               Dd2                 [ (Dd+delta_wd*I)*(Jd*dx - ryd) - rd_tilde ]
 * dyd = Dd3*Jd*dx - Dd3*ryd - Dd2 rd_tilde
 *
 * (H+Dx+delta_wx*I + Jd^T * Dd3 * Jd) dx = rx_tilde + Jd^T*Dd3*ryd +  Jd^T*Dd2*rd_tilde
 */

class hiopKKTLinSysCondensedSparse : public hiopKKTLinSysCompressedSparseXDYcYd
{
public:
  hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp);
  hiopKKTLinSysCondensedSparse() = delete;
  virtual ~hiopKKTLinSysCondensedSparse();

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& rd,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dd,
                               hiopVector& dyc,
                               hiopVector& dyd);

protected:
  /**
   * Solves the compressed XDYcYd system by using direct solves with Cholesky factors of the
   * condensed linear system and appropriately manipulate the XDYcYD rhs/sol to condensed rhs/sol.
   *
   * The method is used as a preconditioner solve in the Krylov-based iterative refinement from
   * solve_compressed method.
   */
  virtual bool solve_compressed_direct(hiopVector& rx,
                                       hiopVector& rd,
                                       hiopVector& ryc,
                                       hiopVector& ryd,
                                       hiopVector& dx,
                                       hiopVector& dd,
                                       hiopVector& dyc,
                                       hiopVector& dyd);

protected:
  ////
  ////from the parent class and its parents we also use
  ////
  // right-hand side [rx_tilde, rd_tilde, ((ryc->empty)), ryd]
  //   hiopVector *rhs_;

  //  hiopVectorPar *Dd;
  //  hiopVectorPar *ryd_tilde;

  // from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //   hiopVectorPar *Dx;
  //   hiopVectorPar *rx_tilde;

  // keep Hx = Dx (Dx=log-barrier diagonal for x) + regularization
  // keep Hd = Dd (Dd=log-barrier diagonal for slack variable) + regularization
  //   hiopVector *Hx_, *Hd_;

  //
  //  hiopNlpSparse* nlpSp_;
  //  hiopMatrixSparse* HessSp_;
  //  const hiopMatrixSparse* Jac_cSp_;
  //  const hiopMatrixSparse* Jac_dSp_;

  // int write_linsys_counter_;
  //  hiopCSR_IO csr_writer_;

  /// Member for JacD in CSR format
  hiopMatrixSparseCSR* JacD_;

  /// Member for JacD' in CSR format
  hiopMatrixSparseCSR* JacDt_;

  /// Member for lower triangular part of Hess in CSR
  hiopMatrixSparseCSR* Hess_lower_csr_;

  /// Member for upper triangular part of Hess
  hiopMatrixSparseCSR* Hess_upper_csr_;

  /// Member for Hess
  hiopMatrixSparseCSR* Hess_csr_;

  /// Member for JacD'*Dd*JacD
  hiopMatrixSparseCSR* JtDiagJ_;

  /// Member for JacD'*Dd*JacD + H + Dx + delta_wx*I
  hiopMatrixSparseCSR* M_condensed_;

  /// Member for storing auxiliary sum of upper triangle of H + Dx + delta_wx*I
  hiopMatrixSparseCSR* Hess_upper_plus_diag_;

  /// Member for storing the auxiliary sum of Dx + delta_wx*I
  hiopMatrixSparseCSR* Diag_Dx_deltawx_;

  /// Stores Dx plus delta_wx for more efficient updates of the condensed system matrix
  hiopVector* Dx_plus_deltawx_;
  hiopVector* deltawx_;

  /// Stores a copy of Hd_ on the device (to be later removed)
  hiopVector* Hd_copy_;

private:
  /// Decides which linear solver to be used. Call only after `M_condended_` has been computed.
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
