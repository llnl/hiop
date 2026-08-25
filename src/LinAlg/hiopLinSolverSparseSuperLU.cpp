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
 * @file hiopLinSolverSparseSuperLU.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 */

// Include SuperLU headers FIRST
#include "superlu_ddefs.h"

// After SuperLU is included, prevent HiOp from redefining BLAS/FC_GLOBAL
// Mark the header guards so HiOp's versions are skipped
#define HIOP_BLASDEFS  // Prevent hiop_blasdefs.hpp content
#define FC_HEADER_INCLUDED  // Prevent HiOp's FortranCInterface.hpp

// Also undef FC_GLOBAL so it doesn't interfere with HiOp's usage
// SuperLU already used it, now we prevent redefinition conflicts
#undef FC_GLOBAL
#undef FC_GLOBAL_

// Now include HiOp headers
#include "hiopLinSolverSparseSuperLU.hpp"
#include "hiopMatrixSparseTriplet.hpp"

// Redefine FC_GLOBAL to match what HiOp expects (name##_)
// This allows any HiOp code that uses FC_GLOBAL to work
#define FC_GLOBAL(name,NAME) name##_
#define FC_GLOBAL_(name,NAME) name##_

#include <cassert>
#include <cstring>
#include <algorithm>
#include <vector>

namespace hiop
{

// PIMPL struct to hide SuperLU types from header
struct SuperLUData {
  SuperMatrix A;                     // Matrix descriptor
  dScalePermstruct_t ScalePermstruct;  // Scaling and permutation
  dLUstruct_t LUstruct;              // LU factors
  dSOLVEstruct_t SOLVEstruct;        // Solve structures
  gridinfo_t grid;                   // Process grid
  superlu_dist_options_t options;    // Solver options
  SuperLUStat_t stat;                // Statistics

  SuperLUData() = default;
};

hiopLinSolverSymSparseSuperLU::hiopLinSolverSymSparseSuperLU(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverSymSparse(n, nnz, nlp),
      m_(n),
      n_(n),
      nnz_(nnz),
      rowptr_(nullptr),
      colind_(nullptr),
      values_(nullptr),
      is_first_call_(true),
      is_factored_(false),
      rhs_(nullptr),
      berr_(nullptr),
      info_(0),
      row_perm_method_("auto")  // Default to automatic selection
{
  // Allocate CSR arrays
  rowptr_ = new int[n_ + 1];
  colind_ = new int[nnz_];
  values_ = new double[nnz_];

  // Allocate error bound array
  berr_ = new double[1];

  // Allocate SuperLU PIMPL structure (will be initialized in firstCall())
  superlu_data_ = new hiop::SuperLUData;
}

hiopLinSolverSymSparseSuperLU::~hiopLinSolverSymSparseSuperLU()
{
  // Clean up SuperLU structures if they were initialized
  if(!is_first_call_) {
    // Destroy SuperMatrix
    Destroy_CompRowLoc_Matrix_dist(&superlu_data_->A);

    // Free scaling and permutation structures
    dScalePermstructFree(&superlu_data_->ScalePermstruct);

    // Free LU structures
    dLUstructFree(&superlu_data_->LUstruct);

    // Free solve structures
    if(is_factored_) {
      dSolveFinalize(&superlu_data_->options, &superlu_data_->SOLVEstruct);
    }

    // Free statistics
    PStatFree(&superlu_data_->stat);

    // Exit process grid
    superlu_gridexit(&superlu_data_->grid);
  }

  // Free SuperLU PIMPL structure
  delete superlu_data_;

  // Free CSR arrays
  delete[] rowptr_;
  delete[] colind_;
  delete[] values_;
  delete[] berr_;

  // Free working arrays
  delete rhs_;

  // Free matrix if owned
  if(sys_mat_owned_) {
    delete M_;
  }
}

void hiopLinSolverSymSparseSuperLU::firstCall()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  // Initialize MPI process grid
  // For now, use all processes in a 1D grid (nprow=1, npcol=nprocs)
  // This can be optimized later for better load balancing
  int nprow = 1;
  int npcol = 1;

#ifdef HIOP_USE_MPI
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  npcol = nprocs;
#endif

  superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &superlu_data_->grid);

  // Set default options
  set_default_options_dist(&superlu_data_->options);

  // Configure options for symmetric indefinite system
  superlu_data_->options.Fact = DOFACT;                // First factorization
  superlu_data_->options.Equil = YES;                  // Equilibrate the matrix
  superlu_data_->options.ParSymbFact = NO;             // Symbolic factorization (NO=sequential)
  superlu_data_->options.ColPerm = MMD_AT_PLUS_A;      // Column ordering: minimum degree on A'+A

  // Configure row permutation based on user selection or automatic choice
  if(row_perm_method_ == "auto") {
    // Automatic selection: SUMAC for GPU builds, SUITOR otherwise
#ifdef HIOP_USE_CUDA
    superlu_data_->options.RowPerm = SUMAC;            // GPU-accelerated symmetric matching
    nlp_->log->printf(hovSummary,
                      "hiopLinSolverSymSparseSuperLU: Using SUMAC (GPU) symmetric matching\n");
#else
    superlu_data_->options.RowPerm = SUITOR;           // CPU-based symmetric matching
    nlp_->log->printf(hovSummary,
                      "hiopLinSolverSymSparseSuperLU: Using SUITOR (CPU) symmetric matching\n");
#endif
  } else if(row_perm_method_ == "sumac") {
    superlu_data_->options.RowPerm = SUMAC;
    nlp_->log->printf(hovSummary,
                      "hiopLinSolverSymSparseSuperLU: Using SUMAC (GPU) symmetric matching\n");
  } else if(row_perm_method_ == "suitor") {
    superlu_data_->options.RowPerm = SUITOR;
    nlp_->log->printf(hovSummary,
                      "hiopLinSolverSymSparseSuperLU: Using SUITOR (CPU) symmetric matching\n");
  } else if(row_perm_method_ == "mc80") {
    superlu_data_->options.RowPerm = MC80;
    nlp_->log->printf(hovSummary,
                      "hiopLinSolverSymSparseSuperLU: Using MC80 (HSL) symmetric matching\n");
  } else if(row_perm_method_ == "mc64") {
    superlu_data_->options.RowPerm = LargeDiag_MC64;
    nlp_->log->printf(hovWarning,
                      "hiopLinSolverSymSparseSuperLU: Using MC64 (generic, not optimized for symmetric systems)\n");
  } else {
    // Invalid option - default to SUITOR with warning
    superlu_data_->options.RowPerm = SUITOR;
    nlp_->log->printf(hovWarning,
                      "hiopLinSolverSymSparseSuperLU: Unknown row permutation method '%s', using SUITOR\n",
                      row_perm_method_.c_str());
  }

  superlu_data_->options.ReplaceTinyPivot = YES;       // Replace tiny pivots
  superlu_data_->options.IterRefine = SLU_DOUBLE;      // Iterative refinement
  superlu_data_->options.Trans = NOTRANS;              // No transpose
  superlu_data_->options.PrintStat = NO;               // Don't print statistics
  superlu_data_->options.SymPattern = NO;              // Not symmetric pattern only

  // Initialize scaling and permutation structure
  dScalePermstructInit(n_, n_, &superlu_data_->ScalePermstruct);

  // Initialize LU structure
  dLUstructInit(n_, &superlu_data_->LUstruct);

  // Initialize statistics
  PStatInit(&superlu_data_->stat);

  is_first_call_ = false;
}

void hiopLinSolverSymSparseSuperLU::convertTripletToCSR()
{
  assert(M_ != nullptr);
  hiopMatrixSparseTriplet* M_triplet = dynamic_cast<hiopMatrixSparseTriplet*>(M_);
  assert(M_triplet != nullptr && "Matrix must be in triplet format");

  const index_type* iRow = M_triplet->i_row();
  const index_type* jCol = M_triplet->j_col();
  const double* vals = M_triplet->M();
  const int nnz_triplet = M_triplet->numberOfNonzeros();

  // Create a vector of tuples for sorting
  std::vector<std::tuple<int, int, double>> entries;
  entries.reserve(nnz_triplet);

  // Collect all entries (convert to 0-based indexing if needed)
  for(int k = 0; k < nnz_triplet; k++) {
    entries.push_back(std::make_tuple(static_cast<int>(iRow[k]),
                                       static_cast<int>(jCol[k]),
                                       vals[k]));
  }

  // Sort by row, then by column
  std::sort(entries.begin(), entries.end(),
            [](const std::tuple<int, int, double>& a, const std::tuple<int, int, double>& b) {
              if(std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) < std::get<0>(b);
              }
              return std::get<1>(a) < std::get<1>(b);
            });

  // Build CSR structure
  std::fill(rowptr_, rowptr_ + n_ + 1, 0);

  // Count entries per row
  for(const auto& entry : entries) {
    int row = std::get<0>(entry);
    rowptr_[row + 1]++;
  }

  // Compute cumulative sum for row pointers
  for(int i = 0; i < n_; i++) {
    rowptr_[i + 1] += rowptr_[i];
  }

  // Fill column indices and values
  int idx = 0;
  for(const auto& entry : entries) {
    colind_[idx] = std::get<1>(entry);
    values_[idx] = std::get<2>(entry);
    idx++;
  }

  assert(idx == nnz_ && "Number of nonzeros mismatch");
  assert(rowptr_[n_] == nnz_ && "CSR row pointer error");
}

int hiopLinSolverSymSparseSuperLU::matrixChanged()
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);

  nlp_->runStats.linsolv.tmFactTime.start();

  // First call initialization
  if(is_first_call_) {
    firstCall();
  }

  // Convert matrix from triplet to CSR format
  convertTripletToCSR();

  // Create SuperMatrix in CSR format
  // For distributed SuperLU, we use the local CSR format
  int_t m_loc = n_;  // Number of local rows
  int_t fst_row = 0; // First row index

  // Create compressed row storage matrix
  dCreate_CompRowLoc_Matrix_dist(&superlu_data_->A,
                                  n_, n_, nnz_, m_loc, fst_row,
                                  values_, colind_, rowptr_,
                                  SLU_NR_loc, SLU_D, SLU_GE);

  // Set factorization mode
  if(is_factored_) {
    // Subsequent factorization with same sparsity pattern
    superlu_data_->options.Fact = SamePattern;
  } else {
    superlu_data_->options.Fact = DOFACT;
  }

  // Allocate right-hand side (needed for pdgssvx interface)
  // We'll create it on first solve, for now just allocate a temporary
  if(rhs_ == nullptr) {
    // Create a vector of the right size - will be allocated properly in solve()
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), n_);
  }
  double* rhs_data = rhs_->local_data();
  std::fill(rhs_data, rhs_data + n_, 0.0); // Zero RHS for factorization

  // Perform factorization
  int nrhs = 1;
  pdgssvx(&superlu_data_->options, &superlu_data_->A, &superlu_data_->ScalePermstruct, rhs_data,
          m_loc, nrhs, &superlu_data_->grid, &superlu_data_->LUstruct, &superlu_data_->SOLVEstruct,
          berr_, &superlu_data_->stat, &info_);

  // Check for errors
  if(info_ != 0) {
    nlp_->log->printf(hovError,
                      "hiopLinSolverSymSparseSuperLU: pdgssvx factorization returned error %d\n",
                      info_);
  }

  is_factored_ = (info_ == 0);

  nlp_->runStats.linsolv.tmFactTime.stop();

  // SuperLU_DIST does not provide inertia information
  // Return -1 to indicate inertia is not available
  // The solver should be used with 'inertia_free' factorization acceptor
  nlp_->runStats.linsolv.tmInertiaComp.start();
  int negEigVal = -1;
  nlp_->runStats.linsolv.tmInertiaComp.stop();

  return negEigVal;
}

bool hiopLinSolverSymSparseSuperLU::solve(hiopVector& x_in)
{
  assert(n_ == M_->n() && M_->n() == M_->m());
  assert(n_ > 0);
  assert(x_in.get_size() == static_cast<size_type>(n_));
  assert(is_factored_ && "Matrix must be factored before solve");

  nlp_->runStats.linsolv.tmTriuSolves.start();

  hiopVector* x = dynamic_cast<hiopVector*>(&x_in);
  assert(x != nullptr);

  // Copy RHS into working array
  if(rhs_ == nullptr) {
    rhs_ = x->new_copy();
  } else {
    rhs_->copyFrom(*x);
  }

  double* rhs_data = rhs_->local_data();

  // Solve the system using existing factorization
  superlu_data_->options.Fact = FACTORED;

  int_t m_loc = n_;
  int nrhs = 1;

  pdgssvx(&superlu_data_->options, &superlu_data_->A, &superlu_data_->ScalePermstruct, rhs_data,
          m_loc, nrhs, &superlu_data_->grid, &superlu_data_->LUstruct, &superlu_data_->SOLVEstruct,
          berr_, &superlu_data_->stat, &info_);

  // Copy solution back
  x->copyFrom(*rhs_);

  if(info_ != 0) {
    nlp_->log->printf(hovError,
                      "hiopLinSolverSymSparseSuperLU: pdgssvx solve returned error %d\n",
                      info_);
  }

  nlp_->runStats.linsolv.tmTriuSolves.stop();

  return (info_ == 0);
}

void hiopLinSolverSymSparseSuperLU::setRowPermutationMethod(const std::string& method)
{
  if(method == "auto" || method == "sumac" || method == "suitor" ||
     method == "mc80" || method == "mc64") {
    row_perm_method_ = method;

    if(!is_first_call_) {
      nlp_->log->printf(hovWarning,
                        "hiopLinSolverSymSparseSuperLU: Row permutation method changed to '%s', "
                        "but solver already initialized. Change will take effect on next solve.\n",
                        method.c_str());
    }
  } else {
    nlp_->log->printf(hovError,
                      "hiopLinSolverSymSparseSuperLU: Invalid row permutation method '%s'. "
                      "Valid options: auto, sumac, suitor, mc80, mc64\n",
                      method.c_str());
  }
}

} // namespace hiop
