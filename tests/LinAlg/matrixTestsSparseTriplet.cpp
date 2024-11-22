// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
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
 * @file matrixTestsSparseTriplet.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */

#include <cstring>
#include <hiopMatrix.hpp>
#include "matrixTestsSparseTriplet.hpp"
#include <hiopVectorIntSeq.hpp>

namespace hiop
{
namespace tests
{

/// Set `i`th element of vector `x`
void MatrixTestsSparseTriplet::setLocalElement(hiop::hiopVector* xvec, const local_ordinal_type i, const real_type val)
{
  auto x = dynamic_cast<hiop::hiopVectorPar*>(xvec);
  if(x != nullptr) {
    real_type* data = x->local_data();
    data[i] = val;
  } else
    THROW_NULL_DEREF;
}

/// Returns element (i,j) of a dense matrix `A`.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsSparseTriplet::getLocalElement(const hiop::hiopMatrix* A,
                                                    local_ordinal_type row,
                                                    local_ordinal_type col)
{
  auto mat = dynamic_cast<const hiop::hiopMatrixDense*>(A);

  if(mat != nullptr) {
    const double* M = mat->local_data_const();
    // return M[row][col];
    return M[row * mat->get_local_size_n() + col];
  }

  else
    THROW_NULL_DEREF;
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
real_type MatrixTestsSparseTriplet::getLocalElement(const hiop::hiopVector* x, local_ordinal_type i)
{
  const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
  if(xvec != nullptr)
    return xvec->local_data_const()[i];
  else
    THROW_NULL_DEREF;
}

real_type* MatrixTestsSparseTriplet::getMatrixData(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  return mat->M();
}

real_type MatrixTestsSparseTriplet::getMatrixData(hiop::hiopMatrixSparse* A, local_ordinal_type i, local_ordinal_type j)
{
  auto* mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  auto* val = mat->M();
  auto* iRow = mat->i_row();
  auto* jCol = mat->j_col();
  auto nnz = mat->numberOfNonzeros();

  for(auto k = 0; k < nnz; i++) {
    if(iRow[k] == i && jCol[k] == j) {
      return val[k];
    }
    // assume elements are row-major ordered.
    if(iRow[k] >= i) break;
  }
  return zero;
}

const local_ordinal_type* MatrixTestsSparseTriplet::getRowIndices(const hiop::hiopMatrixSparse* A)
{
  const auto* mat = dynamic_cast<const hiop::hiopMatrixSparseTriplet*>(A);
  return mat->i_row();
}

const local_ordinal_type* MatrixTestsSparseTriplet::getColumnIndices(const hiop::hiopMatrixSparse* A)
{
  const auto* mat = dynamic_cast<const hiop::hiopMatrixSparseTriplet*>(A);
  return mat->j_col();
}

/// Returns size of local data array for vector `x`
int MatrixTestsSparseTriplet::getLocalSize(const hiop::hiopVector* x)
{
  const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
  if(xvec != nullptr)
    return static_cast<int>(xvec->get_local_size());
  else
    THROW_NULL_DEREF;
}

/**
 * @brief Verifies values of the sparse matrix *only at indices already defined by the sparsity pattern*
 * This may seem misleading, but verify answer does not check *every* value of the matrix,
 * but only `nnz` elements.
 *
 */
[[nodiscard]]
int MatrixTestsSparseTriplet::verifyAnswer(hiop::hiopMatrixSparse* A, const double answer)
{
  if(A == nullptr) return 1;
  auto mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  const local_ordinal_type nnz = mat->numberOfNonzeros();
  const real_type* values = mat->M();
  int fail = 0;
  for(local_ordinal_type i = 0; i < nnz; i++) {
    if(!isEqual(values[i], answer)) {
      fail++;
    }
  }
  return fail;
}

/**
 * @brief Verifies values of the sparse matrix *only at indices already defined by the sparsity pattern*
 * This may seem misleading, but verify answer does not check *every* value of the matrix,
 * but only `nnz` elements with index from nnz_st to nnz_ed
 *
 */
[[nodiscard]]
int MatrixTestsSparseTriplet::verifyAnswer(hiop::hiopMatrix* A,
                                           local_ordinal_type nnz_st,
                                           local_ordinal_type nnz_ed,
                                           const double answer)
{
  if(A == nullptr) return 1;
  auto mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  const real_type* values = mat->M();
  int fail = 0;
  for(local_ordinal_type i = nnz_st; i < nnz_ed; i++) {
    if(!isEqual(values[i], answer)) {
      fail++;
    }
  }
  return fail;
}

/*
 * Pass a function-like object to calculate the expected
 * answer dynamically, based on the row and column
 */
[[nodiscard]]
int MatrixTestsSparseTriplet::verifyAnswer(hiop::hiopMatrixDense* A,
                                           std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
  // auto A = dynamic_cast<hiop::hiopMatrixDense*>(Amat);
  assert(A->get_local_size_n() == A->n() && "Matrix should not be distributed");
  const local_ordinal_type M = A->get_local_size_m();
  const local_ordinal_type N = A->get_local_size_n();
  int fail = 0;
  for(local_ordinal_type i = 0; i < M; i++) {
    for(local_ordinal_type j = 0; j < N; j++) {
      if(!isEqual(getLocalElement(A, i, j), expect(i, j))) {
        // printf("(%d, %d) failed. %f != %f.\n", i, j, getLocalElement(A, i, j), expect(i, j));
        fail++;
      }
    }
  }
  return fail;
}

/// Checks if _local_ vector elements are set to `answer`.
[[nodiscard]]
int MatrixTestsSparseTriplet::verifyAnswer(hiop::hiopVector* x, double answer)
{
  const local_ordinal_type N = getLocalSize(x);

  int local_fail = 0;
  for(local_ordinal_type i = 0; i < N; ++i) {
    if(!isEqual(getLocalElement(x, i), answer)) {
      printf("Failed. %f != %f.\n", getLocalElement(x, i), answer);
      ++local_fail;
    }
  }
  return local_fail;
}

[[nodiscard]]
int MatrixTestsSparseTriplet::verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect)
{
  const local_ordinal_type N = getLocalSize(x);

  int local_fail = 0;
  for(int i = 0; i < N; i++) {
    if(!isEqual(getLocalElement(x, i), expect(i))) {
      printf("Failed. %f != %f.\n", getLocalElement(x, i), expect(i));
      ++local_fail;
    }
  }
  return local_fail;
}

local_ordinal_type* MatrixTestsSparseTriplet::numNonzerosPerRow(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  auto nnz = mat->numberOfNonzeros();
  auto iRow = mat->i_row();
  auto sparsity_pattern = new local_ordinal_type[mat->m()];
  std::memset(sparsity_pattern, 0, sizeof(local_ordinal_type) * mat->m());

  for(local_ordinal_type i = 0; i < nnz; i++) {
    sparsity_pattern[iRow[i]]++;
  }
  return sparsity_pattern;
}

local_ordinal_type* MatrixTestsSparseTriplet::numNonzerosPerCol(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(A);
  auto nnz = mat->numberOfNonzeros();
  auto jCol = mat->j_col();
  auto sparsity_pattern = new local_ordinal_type[mat->n()];
  std::memset(sparsity_pattern, 0, sizeof(local_ordinal_type) * mat->n());

  for(local_ordinal_type i = 0; i < nnz; i++) {
    sparsity_pattern[jCol[i]]++;
  }
  return sparsity_pattern;
}

void MatrixTestsSparseTriplet::initializeMatrix(hiop::hiopMatrixSparse* mat, local_ordinal_type entries_per_row)
{
  auto* A = dynamic_cast<hiop::hiopMatrixSparseTriplet*>(mat);
  local_ordinal_type* iRow = A->i_row();
  local_ordinal_type* jCol = A->j_col();
  double* val = A->M();

  local_ordinal_type m = A->m();
  local_ordinal_type n = A->n();

  assert(A->numberOfNonzeros() == m * entries_per_row && "Matrix initialized with insufficent number of non-zero entries");

  for(local_ordinal_type row = 0, col = 0, i = 0; row < m && i < A->numberOfNonzeros(); row++, col = 0) {
    for(local_ordinal_type j = 0; j < entries_per_row - 1; i++, j++, col += n / entries_per_row) {
      iRow[i] = row;
      jCol[i] = col;
      val[i] = one;
    }

    iRow[i] = row;
    jCol[i] = n - 1;
    val[i++] = one;
  }
}

/**
 * @brief Since some classes will have to copy data from device, this method is
 * a placeholder to keep tests implementation-agnostic; classes that have
 * device memory will copy from device when this is called, CPU-bound classes
 * will no-op.
 */
void MatrixTestsSparseTriplet::maybeCopyToDev(hiop::hiopMatrixSparse*) {}

/**
 * @brief placeholder on CPU-bound classes.
 * @see MatrixTestsSparseTriplet::maybeCopyToDev
 */
void MatrixTestsSparseTriplet::maybeCopyFromDev(hiop::hiopMatrixSparse*) {}

int MatrixTestsSparseTriplet::getLocalElement(hiop::hiopVectorInt* xvec, int idx) const
{
  if(auto* x = dynamic_cast<hiop::hiopVectorIntSeq*>(xvec)) {
    return x->local_data_host_const()[idx];
  } else {
    assert(false && "Wrong type of vector passed into `MatrixTestsSparseTriplet::getLocalElement`!");
    return 0;
  }
}

void MatrixTestsSparseTriplet::setLocalElement(hiop::hiopVectorInt* xvec, int idx, int value) const
{
  if(auto* x = dynamic_cast<hiop::hiopVectorIntSeq*>(xvec)) {
    x->local_data_host()[idx] = value;
  } else {
    assert(false && "Wrong type of vector passed into `MatrixTestsSparseTriplet::setLocalElement`!");
  }
}

}  // namespace tests
}  // namespace hiop
