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
 * @file hiopLinSolverSparseHyKKT.hpp
 *
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 *
 */

#ifndef HIOP_LINSOLVER_HYKKT
#define HIOP_LINSOLVER_HYKKT

#include "hiopLinSolver.hpp"

#include <cassert>

namespace ReSolve
{
class MatrixHandler;
class VectorHandler;

class LinAlgWorkspaceCpu;

#ifdef HIOP_USE_CUDA
class LinAlgWorkspaceCUDA;
#endif

#ifdef HIOP_USE_HIP
class LinAlgWorkspaceHIP;
#endif

namespace hykkt
{
class HyKKTSolver;
}

namespace matrix
{
class Csr;
}

namespace vector
{
class Vector;
}
}  // namespace ReSolve

namespace hiop
{

class hiopMatrixSparse;

/**
 * @brief Sparse KKT solver adapter for ReSolve's HyKKT.
 *
 * Unlike the other sparse solvers, HyKKT operates on the KKT blocks rather
 * than on an assembled KKT matrix, so this class does not own a system
 * matrix. The KKT class hands over the Hessian, Jacobian, and diagonal
 * blocks through set_kkt_blocks(). matrixChanged() converts them to
 * ReSolve CSR on the first call and refreshes the numerical values on
 * later calls. solve() takes the stacked right-hand side
 * [rx; rd; ryc; ryd] assembled by the XDYcYd KKT class and overwrites it
 * with the solution.
 *
 * @ingroup LinearSolvers
 */
class hiopLinSolverSparseHyKKT : public hiopLinSolverSymSparse
{
public:
  hiopLinSolverSparseHyKKT(hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverSparseHyKKT();

  /**
   * @brief Set the KKT blocks HyKKT operates on.
   *
   * The pointers are retained, not copied, and must stay valid for the
   * lifetime of this object. `Hx` holds the Hessian diagonal contribution
   * Dx + delta_wx and `Hd` the slack diagonal Dd + delta_wd.
   */
  void set_kkt_blocks(hiopMatrixSparse* Hess,
                      const hiopMatrixSparse* Jac_c,
                      const hiopMatrixSparse* Jac_d,
                      const hiopVector* Hx,
                      const hiopVector* Hd);

  /**
   * @brief Build the ReSolve blocks on the first call and update their values afterwards.
   *
   * HyKKT factorizes inside solve(), so this only prepares the matrix data.
   *
   * @return Zero on success, negative on failure.
   */
  virtual int matrixChanged();

  /**
   * @brief Solve the KKT system.
   *
   * @param x On entry, the stacked right-hand side [rx; rd; ryc; ryd].
   *
   * @post On exit, `x` is overwritten with [dx; dd; dyc; dyd].
   */
  virtual bool solve(hiopVector& x);

  /** Multiple right-hand sides are not supported. */
  virtual bool solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

protected:
  bool initialize_matrix_blocks();
  bool initialize_vector_blocks();
  bool initialize_solver();
  bool update_matrix_blocks();

  /// True when refactorization and solves run on the device.
  bool solve_on_device() const;

  /// True when the HiOp blocks live in device memory.
  bool blocks_on_device() const;

  hiopMatrixSparse* HessSp_;
  const hiopMatrixSparse* Jac_cSp_;
  const hiopMatrixSparse* Jac_dSp_;
  const hiopVector* Hx_;
  const hiopVector* Hd_;

  ReSolve::hykkt::HyKKTSolver* hykkt_solver_;

  ReSolve::matrix::Csr* H_;
  ReSolve::matrix::Csr* D_s_;
  ReSolve::matrix::Csr* J_;
  ReSolve::matrix::Csr* J_d_;

  // HiOp rd/dd correspond to HyKKT's slack blocks r_s/s.
  ReSolve::vector::Vector* r_x_;
  ReSolve::vector::Vector* r_s_;
  ReSolve::vector::Vector* r_y_;
  ReSolve::vector::Vector* r_yd_;

  ReSolve::vector::Vector* x_;
  ReSolve::vector::Vector* s_;
  ReSolve::vector::Vector* y_;
  ReSolve::vector::Vector* y_d_;

  ReSolve::LinAlgWorkspaceCpu* cpu_workspace_;
#ifdef HIOP_USE_CUDA
  ReSolve::LinAlgWorkspaceCUDA* cuda_workspace_;
#endif
#ifdef HIOP_USE_HIP
  ReSolve::LinAlgWorkspaceHIP* hip_workspace_;
#endif

  ReSolve::MatrixHandler* matrix_handler_;
  ReSolve::VectorHandler* vector_handler_;

  // Maps persistent ReSolve CSR entries to HiOp triplet entries for
  // numerical updates; H_diag_to_csr_host_ locates the added Hx diagonal.
  int* H_csr_to_triplet_host_;
  int* H_diag_to_csr_host_;
  int* J_csr_to_triplet_host_;
  int* J_d_csr_to_triplet_host_;

#ifdef HIOP_USE_GPU
  int* H_csr_to_triplet_device_;
  int* H_diag_to_csr_device_;
  int* J_csr_to_triplet_device_;
  int* J_d_csr_to_triplet_device_;
#endif
};

}  // namespace hiop

#endif
