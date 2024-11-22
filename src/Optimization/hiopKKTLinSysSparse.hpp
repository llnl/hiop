// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Nai-Yuan Chiang, chiang7@llnl.gov and Cosmin G. Petra, petra1@llnl.gov.
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

#ifndef HIOP_KKTLINSYSSPARSE
#define HIOP_KKTLINSYSSPARSE

#include "hiopKKTLinSys.hpp"
#include "hiopLinSolver.hpp"

#include "hiopCSR_IO.hpp"

namespace hiop
{

/*
 * Solves KKTLinSysCompressedXYcYd by exploiting the sparse structure
 *
 * In general, the so-called XYcYd system has the form
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 *
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *  - Dx is diagonal corresponding to x variable in the log-barrier diagonal Dx, respectively
 *
 */
class hiopKKTLinSysCompressedSparseXYcYd : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysCompressedSparseXYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedSparseXYcYd();

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dyc,
                               hiopVector& dyd);

protected:
  hiopVector* rhs_;  //[rx_tilde, ryc_tilde, ryd_tilde]

  //
  // from the parent class we also use
  //
  //  hiopVectorPar *Dd_inv;
  //  hiopVectorPar *ryd_tilde;

  // from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //   hiopVectorPar *Dx;
  //   hiopVectorPar *rx_tilde;

  // Keeps Hx = HessSp_->sp_mat() + Dxs (Dx=log-barrier diagonal for x)
  hiopVector* Hx_;

  // just dynamic_cast-ed pointers
  hiopNlpSparse* nlpSp_;
  hiopMatrixSparse* HessSp_;
  const hiopMatrixSparse* Jac_cSp_;
  const hiopMatrixSparse* Jac_dSp_;

  // -1 when disabled; otherwise acts like a counter, 0,1,... incremented each time
  // 'solveCompressed' is called; activated by the 'write_kkt' option
  int write_linsys_counter_;
  hiopCSR_IO csr_writer_;

private:
  // placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverSymSparse* determineAndCreateLinsys(int nxd, int neq, int nineq, int nnz);
};

/*
 * Solves KKTLinSysCompressedXDYcYd by exploiting the sparse structure
 *
 * In general, the so-called XDYcYd system has the form
 * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
 * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
 * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]
 *
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *  - Dx is diagonal corresponding to x variable in the log-barrier diagonal Dx, respectively
 *
 */
class hiopKKTLinSysCompressedSparseXDYcYd : public hiopKKTLinSysCompressedXDYcYd
{
public:
  hiopKKTLinSysCompressedSparseXDYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedSparseXDYcYd();

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
  hiopVector* rhs_;  //[rx_tilde, rd_tilde, ryc, ryd]

  //
  // from the parent class we also use
  //
  //  hiopVectorPar *Dd;
  //  hiopVectorPar *ryd_tilde;

  // from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //   hiopVectorPar *Dx;
  //   hiopVectorPar *rx_tilde;

  // Keeps Hx = Dx (Dx=log-barrier diagonal for x) + regularization
  // Keeps Hd = Dd (Dd=log-barrier diagonal for slack variable) + regularization
  hiopVector *Hx_, *Hd_;

  // just dynamic_cast-ed pointers
  hiopNlpSparse* nlpSp_;
  hiopMatrixSparse* HessSp_;
  const hiopMatrixSparse* Jac_cSp_;
  const hiopMatrixSparse* Jac_dSp_;

  // -1 when disabled; otherwise acts like a counter, 0,1,... incremented each time
  // 'solveCompressed' is called; activated by the 'write_kkt' option
  int write_linsys_counter_;
  hiopCSR_IO csr_writer_;

private:
  // placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverSymSparse* determineAndCreateLinsys(int nxd, int neq, int nineq, int nnz);
};

/*
 * Solves KKTLinSysCompressedXYcYd by exploiting the sparse structure
 *
 * In general, the so-called XYcYd system has the form
 * [   H   Jc^T  Jd^T | 0 |  0   0  -I   I   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  Jc    0     0   | 0 |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    0     0   |-I |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0    -I   | 0 |  -I  I   0   0   |  0   0   0   0  ] [  dd]   [    rd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   |-I |  0   0   0   0   |  I   0   0   0  ] [ dvl]   [   rvl    ]
 * [  0     0     0   | I |  0   0   0   0   |  0   I   0   0  ] [ dvu]   [   rvu    ]
 * [ -I     0     0   | 0 |  0   0   0   0   |  0   0   I   0  ] [ dzl]   [   rzl    ]
 * [  I     0     0   | 0 |  0   0   0   0   |  0   0   0   I  ] [ dzu]   [   rzu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   | 0 | Sl^d 0   0   0   | Vl   0   0   0  ] [dsdl]   [  rsdl    ]
 * [  0     0     0   | 0 |  0  Su^d 0   0   |  0  Vu   0   0  ] [dsdu]   [  rsdu    ]
 * [  0     0     0   | 0 |  0   0  Sl^x 0   |  0   0  Zl   0  ] [dsxl]   [  rsxl    ]
 * [  0     0     0   | 0 |  0   0   0  Su^x |  0   0   0  Zu  ] [dsxu]   [  rsxu    ]
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *
 */
class hiopKKTLinSysSparseFull : public hiopKKTLinSysFull
{
public:
  hiopKKTLinSysSparseFull(hiopNlpFormulation* nlp);

  virtual ~hiopKKTLinSysSparseFull();

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

  bool solve(hiopVector& rx,
             hiopVector& ryc,
             hiopVector& ryd,
             hiopVector& rd,
             hiopVector& rvl,
             hiopVector& rvu,
             hiopVector& rzl,
             hiopVector& rzu,
             hiopVector& rsdl,
             hiopVector& rsdu,
             hiopVector& rsxl,
             hiopVector& rsxu,
             hiopVector& dx,
             hiopVector& dyc,
             hiopVector& dyd,
             hiopVector& dd,
             hiopVector& dvl,
             hiopVector& dvu,
             hiopVector& dzl,
             hiopVector& dzu,
             hiopVector& dsdl,
             hiopVector& dsdu,
             hiopVector& dsxl,
             hiopVector& dsxu);

protected:
  hiopVector* rhs_;

  hiopVector *Hx_, *Hd_;

  // just dynamic_cast-ed pointers
  hiopNlpSparse* nlpSp_;
  hiopMatrixSparse* HessSp_;
  const hiopMatrixSparse* Jac_cSp_;
  const hiopMatrixSparse* Jac_dSp_;

  // -1 when disabled; otherwise acts like a counter, 0,1,... incremented each time
  // 'solve' is called; activated by the 'write_kkt' option
  int write_linsys_counter_;
  hiopCSR_IO csr_writer_;

private:
  // placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverNonSymSparse* determineAndCreateLinsys(const int& n, const int& n_con, const int& nnz);
};

}  // namespace hiop

#endif
