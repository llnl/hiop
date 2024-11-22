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

#ifndef HIOP_KKTLINSYSY_DENSE
#define HIOP_KKTLINSYSY_DENSE

#include "hiopKKTLinSys.hpp"
#include "hiopLinSolver.hpp"
#include "hiopLinSolverSymDenseLapack.hpp"

#ifdef HIOP_USE_MAGMA
#include "hiopLinSolverSymDenseMagma.hpp"
#endif

#include "hiopCSR_IO.hpp"

namespace hiop
{

/* KKT system treated as dense; used for development/testing purposes mainly
 * updates the parts in KKT system that are dependent on the iterate.
 * Triggers a refactorization for the dense linear system
 * Forms the linear system
 * [  H  +  Dx   Jc^T   Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc        0       0    ] [dyc] = [   ryc    ]
 * [    Jd        0   -Dd^{-1} ] [dyd]   [   ryd    ]
 */
class hiopKKTLinSysDenseXYcYd : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysDenseXYcYd(hiopNlpFormulation* nlp)
      : hiopKKTLinSysCompressedXYcYd(nlp),
        rhsXYcYd(NULL),
        write_linsys_counter(-1),
        csr_writer(nlp)
  {}
  virtual ~hiopKKTLinSysDenseXYcYd() { delete rhsXYcYd; }

  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg)
  {
    delta_wx_ = perturb_calc_->get_curr_delta_wx();
    delta_wd_ = perturb_calc_->get_curr_delta_wd();
    delta_cc_ = perturb_calc_->get_curr_delta_cc();
    delta_cd_ = perturb_calc_->get_curr_delta_cd();

    assert(nlp_);

    int nx = Hess_->m();
    assert(nx == Hess_->n());
    assert(nx == Jac_c_->n());
    assert(nx == Jac_d_->n());
    int neq = Jac_c_->m(), nineq = Jac_d_->m();

    if(NULL == linSys_) {
      int n = Jac_c_->m() + Jac_d_->m() + Hess_->m();

      if(nlp_->options->GetString("compute_mode") == "hybrid" || nlp_->options->GetString("compute_mode") == "gpu") {
#ifdef HIOP_USE_MAGMA
        linSys_ = new hiopLinSolverSymDenseMagmaNopiv(n, nlp_);
        nlp_->log->printf(hovScalars, "LinSysDenseXYcYd: instantiating Magma for a matrix of size %d\n", n);
#else
        linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
        nlp_->log->printf(hovScalars, "LinSysDenseXYcYd: instantiating Lapack for a matrix of size %d\n", n);
#endif
      } else {
        linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
        nlp_->log->printf(hovScalars, "LinSysDenseXYcYd: instantiating Lapack for a matrix of size %d\n", n);
      }
    }

    hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);
    hiopMatrixDense& Msys = linSys->sysMatrix();

    //
    // update linSys system matrix, including IC perturbations
    //
    nlp_->runStats.kkt.tmUpdateLinsys.start();

    Msys.setToZero();

    int alpha = 1.;
    Hess_->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);

    Jac_c_->transAddToSymDenseMatrixUpperTriangle(0, nx, alpha, Msys);
    Jac_d_->transAddToSymDenseMatrixUpperTriangle(0, nx + neq, alpha, Msys);

    Msys.addSubDiagonal(alpha, 0, *Dx_);
    Msys.addSubDiagonal(alpha, 0, *delta_wx_);

    // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu + delta_wd*I
    Dd_inv_->copyFrom(*delta_wd_);
    Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
    Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
    assert(true == Dd_inv_->allPositive());
#endif
    Dd_inv_->invert();

    alpha = -1.;
    Msys.addSubDiagonal(alpha, nx + neq, *Dd_inv_);

#ifdef HIOP_DEEPCHECKS
    assert(perturb_calc_->check_consistency() && "something went wrong with IC");
#endif
    Msys.addSubDiagonal(alpha, nx, *delta_cd_);

    nlp_->log->write("KKT Linsys:", Msys, hovMatrices);

    // write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") {
      write_linsys_counter++;
    }
    if(write_linsys_counter >= 0) {
      csr_writer.writeMatToFile(Msys, write_linsys_counter, nx, neq, nineq);
    }

    return true;
  }

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dyc,
                               hiopVector& dyd)
  {
    hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);
    assert(linSys && "fail to get an object for correct linear system");

    int nx = rx.get_size(), nyc = ryc.get_size(), nyd = ryd.get_size();
    if(rhsXYcYd == nullptr) {
      rhsXYcYd = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx + nyc + nyd);
    }
    nlp_->log->write("RHS KKT XYcYd rx: ", rx, hovIteration);
    nlp_->log->write("RHS KKT XYcYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT XYcYd ryd:", ryd, hovIteration);

    rx.copyToStarting(*rhsXYcYd, 0);
    ryc.copyToStarting(*rhsXYcYd, nx);
    ryd.copyToStarting(*rhsXYcYd, nx + nyc);

    if(write_linsys_counter >= 0) csr_writer.writeRhsToFile(*rhsXYcYd, write_linsys_counter);

    //! todo: iterative refinement
    bool sol_ok = linSys->solve(*rhsXYcYd);

    if(write_linsys_counter >= 0) csr_writer.writeSolToFile(*rhsXYcYd, write_linsys_counter);

    if(false == sol_ok) return false;

    rhsXYcYd->copyToStarting(0, dx);
    rhsXYcYd->copyToStarting(nx, dyc);
    rhsXYcYd->copyToStarting(nx + nyc, dyd);

    nlp_->log->write("SOL KKT XYcYd dx: ", dx, hovMatrices);
    nlp_->log->write("SOL KKT XYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT XYcYd dyd:", dyd, hovMatrices);
    return true;
  }

protected:
  hiopVector* rhsXYcYd;

  /** -1 when disabled; otherwise acts like a counter, 0,1,...
   * incremented each time 'solveCompressed' is called depends on the 'write_kkt' option
   */
  int write_linsys_counter;
  hiopCSR_IO csr_writer;

private:
  hiopKKTLinSysDenseXYcYd()
      : hiopKKTLinSysCompressedXYcYd(NULL),
        write_linsys_counter(-1),
        csr_writer(NULL)
  {
    assert(false);
  }
};

/** KKT system treated as dense; used for developement/testing purposes mainly */
class hiopKKTLinSysDenseXDYcYd : public hiopKKTLinSysCompressedXDYcYd
{
public:
  hiopKKTLinSysDenseXDYcYd(hiopNlpFormulation* nlp)
      : hiopKKTLinSysCompressedXDYcYd(nlp),
        rhsXDYcYd(NULL),
        write_linsys_counter(-1),
        csr_writer(nlp)
  {}
  virtual ~hiopKKTLinSysDenseXDYcYd() { delete rhsXDYcYd; }

  /* Updates the parts in KKT system that are dependent on the iterate.
   * Triggers a refactorization for the dense linear system
   * Forms the linear system
   * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
   * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
   * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]
   */
  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg)
  {
    assert(nlp_);
    delta_wx_ = perturb_calc_->get_curr_delta_wx();
    delta_wd_ = perturb_calc_->get_curr_delta_wd();
    delta_cc_ = perturb_calc_->get_curr_delta_cc();
    delta_cd_ = perturb_calc_->get_curr_delta_cd();

    int nx = Hess_->m();
    assert(nx == Hess_->n());
    assert(nx == Jac_c_->n());
    assert(nx == Jac_d_->n());
    int neq = Jac_c_->m(), nineq = Jac_d_->m();
    assert(nx == Hess_->n());
    assert(nx == Jac_c_->n());
    assert(nx == Jac_d_->n());

    if(NULL == linSys_) {
      int n = nx + neq + 2 * nineq;

      if(nlp_->options->GetString("compute_mode") == "hybrid" || nlp_->options->GetString("compute_mode") == "gpu") {
#ifdef HIOP_USE_MAGMA
        nlp_->log->printf(hovScalars, "LinSysDenseDXYcYd: instantiating Magma for a matrix of size %d\n", n);
        linSys_ = new hiopLinSolverSymDenseMagmaNopiv(n, nlp_);
#else
        nlp_->log->printf(hovScalars, "LinSysDenseXDYcYd: instantiating Lapack for a matrix of size %d\n", n);
        linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
#endif
      } else {
        nlp_->log->printf(hovScalars, "LinSysDenseXDYcYd instantiating Lapack for a matrix of size %d\n", n);
        linSys_ = new hiopLinSolverSymDenseLapack(n, nlp_);
      }
    }

    hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);
    hiopMatrixDense& Msys = linSys->sysMatrix();

    //
    // update linSys system matrix, including IC perturbations
    //
    Msys.setToZero();

    const int alpha = 1.;
    Hess_->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);

    Jac_c_->transAddToSymDenseMatrixUpperTriangle(0, nx + nineq, alpha, Msys);
    Jac_d_->transAddToSymDenseMatrixUpperTriangle(0, nx + nineq + neq, alpha, Msys);

    // add diagonals and IC perturbations
    Msys.addSubDiagonal(alpha, 0, *Dx_);
    Msys.addSubDiagonal(alpha, 0, *delta_wx_);

    Msys.addSubDiagonal(alpha, nx, *Dd_);
    Msys.addSubDiagonal(alpha, nx, *delta_wd_);

    // add -I (of size nineq) starting at index (nx, nx+nineq+neq)
    int col_start = nx + nineq + neq;
    double* MsysM = Msys.local_data();
    int m_Msys = Msys.m();
    assert(m_Msys == Msys.n());
    for(int i = nx; i < nx + nineq; i++) {
      // MsysM[i][col_start++] -= 1.;
      assert(i * m_Msys + col_start < m_Msys * m_Msys);
      MsysM[i * m_Msys + col_start] -= 1.;
      col_start++;
    }

#ifdef HIOP_DEEPCHECKS
    assert(delta_cc_->is_equal(*delta_cd_));
#endif
    Msys.addSubDiagonal(-alpha, nx + nineq, *delta_cd_);

    nlp_->log->write("KKT Linsys:", Msys, hovMatrices);

    // write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") {
      write_linsys_counter++;
    }
    if(write_linsys_counter >= 0) {
      csr_writer.writeMatToFile(Msys, write_linsys_counter, nx, neq, nineq);
    }

    nlp_->log->write("KKT XDYcYd Linsys (to be factorized):", Msys, hovMatrices);
    return true;
  }

  virtual bool solveCompressed(hiopVector& rx,
                               hiopVector& rd,
                               hiopVector& ryc,
                               hiopVector& ryd,
                               hiopVector& dx,
                               hiopVector& dd,
                               hiopVector& dyc,
                               hiopVector& dyd)
  {
    hiopLinSolverSymDense* linSys = dynamic_cast<hiopLinSolverSymDense*>(linSys_);

    int nx = rx.get_size(), nyc = ryc.get_size(), nyd = ryd.get_size();
    if(rhsXDYcYd == nullptr) {
      rhsXDYcYd = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx + nyc + 2 * nyd);
    }

    nlp_->log->write("RHS KKT XDycYd rx: ", rx, hovMatrices);
    nlp_->log->write("RHS KKT XDycYd rd: ", rd, hovMatrices);
    nlp_->log->write("RHS KKT XDycYd ryc:", ryc, hovMatrices);
    nlp_->log->write("RHS KKT XDycYd ryd:", ryd, hovMatrices);

    rx.copyToStarting(*rhsXDYcYd, 0);
    rd.copyToStarting(*rhsXDYcYd, nx);
    ryc.copyToStarting(*rhsXDYcYd, nx + nyd);
    ryd.copyToStarting(*rhsXDYcYd, nx + nyd + nyc);

    if(write_linsys_counter >= 0) csr_writer.writeRhsToFile(*rhsXDYcYd, write_linsys_counter);

    bool sol_ok = linSys->solve(*rhsXDYcYd);

    if(write_linsys_counter >= 0) csr_writer.writeSolToFile(*rhsXDYcYd, write_linsys_counter);

    if(false == sol_ok) return false;

    rhsXDYcYd->copyToStarting(0, dx);
    rhsXDYcYd->copyToStarting(nx, dd);
    rhsXDYcYd->copyToStarting(nx + nyd, dyc);
    rhsXDYcYd->copyToStarting(nx + nyd + nyc, dyd);

    nlp_->log->write("SOL KKT XDYcYd dx: ", dx, hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dd: ", dd, hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dyd:", dyd, hovMatrices);
    return true;
  }

protected:
  hiopVector* rhsXDYcYd;
  //-1 when disabled; otherwise acts like a counter, 0,1,... incremented each time 'solveCompressed' is called
  // depends on the 'write_kkt' option
  int write_linsys_counter;
  hiopCSR_IO csr_writer;

private:
  hiopKKTLinSysDenseXDYcYd()
      : hiopKKTLinSysCompressedXDYcYd(NULL),
        write_linsys_counter(-1),
        csr_writer(NULL)
  {
    assert(false && "not intended to be used");
  }
};

}  // namespace hiop

#endif
