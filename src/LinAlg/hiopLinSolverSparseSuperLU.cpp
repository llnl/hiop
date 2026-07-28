#include "hiopLinSolverSparseSuperLU.hpp"
#include "hiop_blasdefs.hpp"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <vector>

namespace hiop
{

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
      info_(0)
{
  // Allocate CSR arrays
  rowptr_ = new int[n_ + 1];
  colind_ = new int[nnz_];
  values_ = new double[nnz_];

  // Allocate error bound array
  berr_ = new double[1];

  // Initialize SuperLU_DIST structures will be done in firstCall()
}

hiopLinSolverSymSparseSuperLU::~hiopLinSolverSymSparseSuperLU()
{
  // Clean up SuperLU structures if they were initialized
  if(!is_first_call_) {
    // Destroy SuperMatrix
    Destroy_CompRowLoc_Matrix_dist(&A_);

    // Free scaling and permutation structures
    dScalePermstructFree(&ScalePermstruct_);

    // Free LU structures
    dLUstructFree(&LUstruct_);

    // Free solve structures
    if(is_factored_) {
      dSolveFinalize(&options_, &SOLVEstruct_);
    }

    // Free statistics
    PStatFree(&stat_);

    // Exit process grid
    superlu_gridexit(&grid_);
  }

  // Free CSR arrays
  delete[] rowptr_;
  delete[] colind_;
  delete[] values_;
  delete[] berr_;

  // Free working arrays
  delete rhs_;
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

  superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid_);

  // Set default options
  set_default_options_dist(&options_);

  // Configure options for symmetric indefinite system
  options_.Fact = DOFACT;                // First factorization
  options_.Equil = YES;                  // Equilibrate the matrix
  options_.ParSymbFact = NO;             // Symbolic factorization (NO=sequential)
  options_.ColPerm = MMD_AT_PLUS_A;      // Column ordering: minimum degree on A'+A
  options_.RowPerm = LargeDiag_MC64;     // Row permutation for numerical stability
  options_.ReplaceTinyPivot = YES;       // Replace tiny pivots
  options_.IterRefine = DOUBLE;          // Iterative refinement
  options_.Trans = NOTRANS;              // No transpose
  options_.PrintStat = NO;               // Don't print statistics
  options_.SymPattern = NO;              // Not symmetric pattern only

  // Initialize scaling and permutation structure
  dScalePermstructInit(n_, n_, &ScalePermstruct_);

  // Initialize LU structure
  dLUstructInit(n_, &LUstruct_);

  // Initialize statistics
  PStatInit(&stat_);

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
  dCreate_CompRowLoc_Matrix_dist(&A_,
                                  n_, n_, nnz_, m_loc, fst_row,
                                  values_, colind_, rowptr_,
                                  SLU_NR_loc, SLU_D, SLU_GE);

  // Set factorization mode
  if(is_factored_) {
    // Subsequent factorization with same sparsity pattern
    options_.Fact = SamePattern;
  } else {
    options_.Fact = DOFACT;
  }

  // Allocate right-hand side (needed for pdgssvx interface)
  if(rhs_ == nullptr) {
    rhs_ = M_->alloc_clone_vec();
  }
  double* rhs_data = rhs_->local_data();
  std::fill(rhs_data, rhs_data + n_, 0.0); // Zero RHS for factorization

  // Perform factorization
  int nrhs = 1;
  pdgssvx(&options_, &A_, &ScalePermstruct_, rhs_data,
          m_loc, nrhs, &grid_, &LUstruct_, &SOLVEstruct_,
          berr_, &stat_, &info_);

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
  options_.Fact = FACTORED;

  int_t m_loc = n_;
  int nrhs = 1;

  pdgssvx(&options_, &A_, &ScalePermstruct_, rhs_data,
          m_loc, nrhs, &grid_, &LUstruct_, &SOLVEstruct_,
          berr_, &stat_, &info_);

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

} // namespace hiop
