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
 * @file hiopKKTLinSysSparseHyKKT.hpp
 *
 * @author Slaven Peles <peless@ornl.gov>
 *
 * @brief XDYcYd KKT system solved with ReSolve's HyKKT.
 */

#ifndef HIOP_KKTLINSYSSPARSEHYKKT
#define HIOP_KKTLINSYSSPARSEHYKKT

#include "hiopKKTLinSysSparse.hpp"

namespace hiop
{

/**
 * Solves the sparse XDYcYd KKT system
 * [  H+Dx+delta_wx*I       0           Jc^T   Jd^T ] [ dx ]   [ rx_tilde ]
 * [        0         Dd+delta_wd*I      0     -I   ] [ dd ] = [ rd_tilde ]
 * [       Jc               0            0      0   ] [dyc ]   [   ryc    ]
 * [       Jd              -I            0      0   ] [dyd ]   [   ryd    ]
 * using ReSolve's HyKKT solver.
 *
 * HyKKT operates directly on the KKT matrix blocks instead of a fully assembled
 * sparse KKT matrix. This class only assembles the regularized diagonals
 * Hx = Dx + delta_wx*I and Hd = Dd + delta_wd*I and hands the blocks (H, Jc, Jd,
 * Hx, Hd) to hiopLinSolverSparseHyKKT, which owns all ReSolve objects. The
 * right-hand side assembly and the solve are inherited from
 * hiopKKTLinSysCompressedSparseXDYcYd.
 *
 * @note Dual regularization (delta_cc, delta_cd) is not supported; build_kkt_matrix
 * fails if either is nonzero.
 * @note Inertia correction is not supported; option 'fact_acceptor' must be
 * 'inertia_free'.
 */
class hiopKKTLinSysCompressedSparseXDYcYdHyKKT : public hiopKKTLinSysCompressedSparseXDYcYd
{
public:
  hiopKKTLinSysCompressedSparseXDYcYdHyKKT(hiopNlpFormulation* nlp);
  hiopKKTLinSysCompressedSparseXDYcYdHyKKT() = delete;
  virtual ~hiopKKTLinSysCompressedSparseXDYcYdHyKKT();

  /**
   * Assembles the regularized diagonals Hx and Hd and, on first call, creates the
   * HyKKT linear solver with the KKT blocks. Does not assemble a KKT matrix.
   */
  virtual bool build_kkt_matrix(const hiopPDPerturbation& pdreg);

protected:
  /// HyKKT factorizes inside solve(); this only refreshes the block values.
  virtual int factorizeWithCurvCheck();
};

}  // end of namespace

#endif
