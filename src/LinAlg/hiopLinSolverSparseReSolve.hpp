//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file hiopLinSolverSparseReSolve.hpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 *
 */

#ifndef HIOP_LINSOLVER_RESOLVE
#define HIOP_LINSOLVER_RESOLVE

#include "hiopLinSolver.hpp"

#include <cassert>
#include <string>

namespace ReSolve
{
class SystemSolver;

class LinAlgWorkspaceCpu;

#ifdef HIOP_USE_CUDA
class LinAlgWorkspaceCUDA;
#endif

#ifdef HIOP_USE_HIP
class LinAlgWorkspaceHIP;
#endif

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
 * @brief Sparse symmetric linear solver adapter for ReSolve's SystemSolver.
 *
 * HiOp converts its symmetric triplet matrix to CSR and hands it to
 * ReSolve::SystemSolver, which owns the KLU factorization, the optional
 * accelerator refactorization backend, and optional FGMRES iterative
 * refinement.
 *
 * @ingroup LinearSolvers
 */
class hiopLinSolverSymSparseReSolve : public hiopLinSolverSymSparse
{
public:
  hiopLinSolverSymSparseReSolve(const int& n, const int& nnz, hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverSymSparseReSolve();

  /**
   * @brief Update matrix values and factorize or refactorize the selected
   * ReSolve backend.
   *
   * @return Zero on success; negative when setup or numerical factorization
   * fails and HiOp should regularize the system.
   */
  virtual int matrixChanged();

  /**
   * @brief Solves a linear system.
   *
   * @param x On entry, the right-hand side of the system.
   *
   * @post On exit, `x` is overwritten with the solution.
   */
  virtual bool solve(hiopVector& x);

  /** Multiple right-hand sides are not supported yet. */
  virtual bool solve(hiopMatrix& /* x */)
  {
    assert(false && "not yet supported");
    return false;
  }

protected:
  /** Build the CSR matrix and perform one-time KLU setup and symbolic analysis. */
  int firstCall();

  /** Update CSR values from the current HiOp matrix. */
  int update_matrix_values();

  /** Count CSR entries after expanding the symmetric triplet matrix. */
  void compute_nnz();

  /**
   * @brief Build CSR structure and triplet-to-CSR update mappings.
   * @return Zero on success; nonzero on allocation or copy failure.
   */
  int set_csr_indices_values();

  /** Return the host-visible triplet matrix used to construct CSR. */
  hiopMatrixSparse* host_matrix() const;

  /** Configure FGMRES iterative refinement on the system solver. */
  void setup_iterative_refinement();

  hiopMatrixSparse* M_host_;

  int n_;
  int nnz_;

  int* index_convert_CSR2Triplet_host_;
  int* index_convert_extra_Diag2CSR_host_;

  int* index_convert_CSR2Triplet_device_;
  int* index_convert_extra_Diag2CSR_device_;

  /// Nonzero when the selected backend has a valid numerical factorization.
  int factorizationSetupSucc_;

  /// True after one-time accelerator setup from the initial KLU factors.
  bool refactorization_setup_complete_;

  bool is_first_call_;
  bool use_ir_;

  /// True when refactorization and triangular solves run on the device.
  bool solve_on_device_;

  /// ReSolve refactorization method ID: "klu", "glu", "cusolverrf", or "rocsolverrf".
  std::string refactorization_method_;

  ReSolve::matrix::Csr* matrix_;
  ReSolve::vector::Vector* rhs_;
  ReSolve::vector::Vector* solution_;

  ReSolve::LinAlgWorkspaceCpu* cpu_workspace_;
#ifdef HIOP_USE_CUDA
  ReSolve::LinAlgWorkspaceCUDA* cuda_workspace_;
#endif
#ifdef HIOP_USE_HIP
  ReSolve::LinAlgWorkspaceHIP* hip_workspace_;
#endif

  ReSolve::SystemSolver* solver_;
};

}  // namespace hiop

#endif
