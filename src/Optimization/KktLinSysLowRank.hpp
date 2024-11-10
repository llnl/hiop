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
 * @file KktLinSysLowRank.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>,  LLNL
 *
 */

#ifndef HIOP_KKTLINSYSY_LOWRANK
#define HIOP_KKTLINSYSY_LOWRANK

#include "hiopKKTLinSys.hpp"

namespace hiop
{

/**
 * @brief Encapsulates solves with the KKT system of IPM filter.
 *
 * This class is for problems where the Hessian of the Lagrangian is a or is approximated 
 * by low-rank matrix plus a multiple of identity and the number of the constraints is not
 * too large. 
 * 
 * It works with Hessian being a HessianDiagPlusLowRank class and the constraints Jacobian 
 * being hiopMatrixDense. 
 *
 * This class solves the XYcYd compression of the full KKT. See solveCompressed method 
 * for details on the approach used to solve the linear system. 
 */

class KktLinSysLowRank : public hiopKKTLinSysCompressedXYcYd
{
public:
  KktLinSysLowRank(hiopNlpFormulation* nlp);
  virtual ~KktLinSysLowRank();

  /// @brief Updates the KKT system with new info at current iteration
  bool update(const hiopIterate* iter,
	      const hiopVector* grad_f,
	      const hiopMatrix* Jac_c,
              const hiopMatrix* Jac_d,
	      hiopMatrix* Hess)
  {
    const hiopMatrixDense* Jac_c_ = dynamic_cast<const hiopMatrixDense*>(Jac_c);
    const hiopMatrixDense* Jac_d_ = dynamic_cast<const hiopMatrixDense*>(Jac_d);
    HessianDiagPlusRowRank* Hess_ = dynamic_cast<HessianDiagPlusRowRank*>(Hess);
    if(Jac_c_==nullptr || Jac_d_==nullptr || Hess_==nullptr) {
      assert(false);
      return false;
    }
    return update(iter, grad_f_, Jac_c_, Jac_d_, Hess_);
  }

  /// @brief Updates the KKT system with new info at current iteration
  virtual bool update(const hiopIterate* iter,
		      const hiopVector* grad_f,
		      const hiopMatrixDense* Jac_c,
                      const hiopMatrixDense* Jac_d,
		      HessianDiagPlusRowRank* Hess);

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg)
  {
    assert(false && "not yet implemented");
    return false;
  }

  /**
   * Solves the compressed linear system, part of the KKT Linear System interface
   * 
   * Solves the system corresponding to directions for x, yc, and yd, namely
   * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx ]
   * [    Jc          0     0     ] [dyc] = [ ryc]
   * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd]
   *
   * This is done by forming and solving
   * [ Jc*(H+Dx)^{-1}*Jc^T   Jc*(H+Dx)^{-1}*Jd^T          ] [dyc] = [ Jc(H+Dx)^{-1} rx - ryc ]
   * [ Jd*(H+Dx)^{-1}*Jc^T   Jd*(H+Dx)^{-1}*Jd^T + Dd^{-1}] [dyd]   [ Jd(H+dx)^{-1} rx - ryd ]
   * and then solving for dx from
   *  dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
   *
   */
  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dyc,
                               hiopVector& dyd);

  //LAPACK wrappers
  int solve(hiopMatrixDense& M, hiopVector& rhs);
  int solveWithRefin(hiopMatrixDense& M, hiopVector& rhs);
#ifdef HIOP_DEEPCHECKS
  static double solveError(const hiopMatrixDense& M,  const hiopVector& x, hiopVector& rhs);
  double errorCompressedLinsys(const hiopVector& rx,
                               const hiopVector& ryc,
                               const hiopVector& ryd,
			       const hiopVector& dx,
                               const hiopVector& dyc,
                               const hiopVector& dyd);
protected:
  /// @brief perform y=beta*y+alpha*H*x without the log barrier term from H
  void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector& x)
  {
    HessianDiagPlusRowRank* hesslowrank = dynamic_cast<HessianDiagPlusRowRank*>(Hess_);
    assert(nullptr != hesslowrank);
    hesslowrank->times_vec_no_logbar_term(beta, y, alpha, x);
  }
#endif

private:
  /// The kxk reduced matrix
  hiopMatrixDense* N_; 
#ifdef HIOP_DEEPCHECKS
  /// A copy of the above to compute the residual
  hiopMatrixDense* Nmat_; 
#endif
  //internal buffers: k is usually 2 x quasi-Newton memory; n is the size of primal variable vector
  hiopMatrixDense* kxn_mat_; 
  hiopVector* k_vec1_;
};
} //end namespace

#endif // HIOP_KKTLINSYSY_LOWRANK
