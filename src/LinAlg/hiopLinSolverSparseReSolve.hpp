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
 * @author Tamar DeWilde <dewildetc@ornl.gov>
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#ifndef HIOP_LINSOLVER_RESOLVE
#define HIOP_LINSOLVER_RESOLVE

#include "hiopLinSolver.hpp"

#include <cassert>

namespace ReSolve
{
class LinSolverDirectKLU;

class MatrixHandler;
class VectorHandler;
class GramSchmidt;
class LinSolverIterativeFGMRES;
class PreconditionerLU;

#ifdef HIOP_USE_CUDA
class LinSolverDirectCuSolverGLU;
class LinSolverDirectCuSolverRf;
class LinAlgWorkspaceCUDA;
#endif

#ifdef HIOP_USE_HIP
class LinSolverDirectRocSolverRf;
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
 * @brief Sparse symmetric linear solver adapter for ReSolve's public API.
 *
 * HiOp converts its symmetric triplet matrix to CSR and coordinates the
 * selected ReSolve factorization, refactorization, solve, and optional
 * iterative-refinement components.
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
  /**
   * @brief Numerical backend used after the initial matrix setup.
   *
   * KLU is used directly for host execution and supplies the factors needed
   * to initialize an accelerator refactorization backend.
   */
  enum class RefactorizationMode
  {
    CPU_KLU,

  #ifdef HIOP_USE_CUDA
    CUDA_GLU,
    CUDA_RF,
  #endif

  #ifdef HIOP_USE_HIP
    HIP_RF,
  #endif
  };

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

  /** Initialize the selected accelerator backend from the KLU factors. */
  int setup_refactorization_solver();

  /** Run numerical refactorization with the selected backend. */
  int refactorize_selected_solver();

  /** Solve the current system with the selected backend. */
  int solve_selected_solver();

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

  ReSolve::matrix::Csr* matrix_;
  ReSolve::vector::Vector* rhs_;
  ReSolve::vector::Vector* solution_;

  ReSolve::LinSolverDirectKLU* factorization_solver_;

#ifdef HIOP_USE_CUDA
  ReSolve::LinAlgWorkspaceCUDA* cuda_workspace_;
  ReSolve::LinSolverDirectCuSolverGLU* cuda_glu_solver_;
  ReSolve::LinSolverDirectCuSolverRf* cuda_rf_solver_;
#endif

#ifdef HIOP_USE_HIP
  ReSolve::LinAlgWorkspaceHIP* hip_workspace_;
  ReSolve::LinSolverDirectRocSolverRf* hip_rf_solver_;
#endif
  ReSolve::MatrixHandler* ir_matrix_handler_;
  ReSolve::VectorHandler* ir_vector_handler_;
  ReSolve::GramSchmidt* ir_gram_schmidt_;
  ReSolve::LinSolverIterativeFGMRES* ir_solver_;
  ReSolve::PreconditionerLU* ir_preconditioner_;

  RefactorizationMode refactorization_mode_;
};

}  // namespace hiop

#endif
