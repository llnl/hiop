// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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
 * @file hiopLinSolverSparseSuperLU.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 * @brief Wrapper for SuperLU_DIST sparse direct solver
 */

#ifndef HIOP_LINSOLVER_SUPERLU
#define HIOP_LINSOLVER_SUPERLU

#include "hiopLinSolver.hpp"
#include "hiopMatrixSparseTriplet.hpp"

// SuperLU_DIST headers
#include "superlu_ddefs.h"

/**
 * Implements the linear solver class using SuperLU_DIST
 *
 * @ingroup LinearSolvers
 */

namespace hiop
{

/**
 * Wrapper for SuperLU_DIST distributed sparse direct solver.
 *
 * This class uses a triplet sparse matrix (member `M_`) to store the KKT linear system,
 * which is converted internally to CSR format required by SuperLU_DIST.
 *
 * Note: SuperLU_DIST does not provide inertia information directly, so this solver
 * should be used with the 'inertia_free' factorization acceptor option.
 */
class hiopLinSolverSymSparseSuperLU : public hiopLinSolverSymSparse
{
public:
  hiopLinSolverSymSparseSuperLU(const int& n, const int& nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymSparseSuperLU();

  /**
   * Triggers a refactorization of the matrix, if necessary.
   * Returns -1 to indicate inertia information is not available.
   */
  int matrixChanged() override;

  /**
   * Solves a linear system.
   * @param x is on entry the right hand side(s) of the system to be solved. On
   * exit it contains the solution(s).
   */
  bool solve(hiopVector& x) override;

protected:
  hiopLinSolverSymSparseSuperLU() = delete;

  /**
   * Called the very first time a matrix is factorized. Allocates space
   * for the factorization and performs ordering.
   */
  virtual void firstCall();

  /**
   * Convert triplet format from HiOp to CSR format for SuperLU_DIST.
   * For symmetric matrices, this converts the full matrix.
   */
  void convertTripletToCSR();

private:
  int m_;    // number of rows of the whole matrix
  int n_;    // number of cols of the whole matrix
  int nnz_;  // number of nonzeros in the matrix

  // CSR storage arrays (SuperLU_DIST format)
  int* rowptr_;   // Row pointers (size n_+1)
  int* colind_;   // Column indices (size nnz_)
  double* values_; // Matrix values (size nnz_)

  // SuperLU_DIST data structures
  SuperMatrix A_;                    // Matrix descriptor
  dScalePermstruct_t ScalePermstruct_;  // Scaling and permutation
  dLUstruct_t LUstruct_;             // LU factors
  dSOLVEstruct_t SOLVEstruct_;       // Solve structures
  gridinfo_t grid_;                  // Process grid
  superlu_dist_options_t options_;   // Solver options
  SuperLUStat_t stat_;               // Statistics

  // Status flags
  bool is_first_call_;    // First factorization call
  bool is_factored_;      // Whether matrix has been factored

  // Temporary storage
  hiopVector* rhs_;       // RHS working array
  double* berr_;          // Backward error bound

  int info_;              // Return status from SuperLU
};

} // namespace hiop

#endif // HIOP_LINSOLVER_SUPERLU
