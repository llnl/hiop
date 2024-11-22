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
 * @file MatrixTestsRajaSymSparseTriplet.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */

#include <cstring>
#include <hiopMatrixDenseRaja.hpp>
#include <hiopVectorRaja.hpp>
#include <hiopMatrixRajaSparseTriplet.hpp>
#include "matrixTestsRajaSymSparseTriplet.hpp"

// TODO: this is a quick hack. Will need to modify this class to be aware of the instantiated
//  template parameters for vector and matrix RAJA classes. Likely a better approach would be
//  to revise the tests to try out multiple configurations of the memory backends and execution
//  policies for RAJA dense matrix.
#if defined(HIOP_USE_CUDA)
#include <ExecPoliciesRajaCudaImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
using hiopMatrixSymSparseTripletRajaT =
    hiop::hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
#endif

namespace hiop
{
namespace tests
{

/// Returns element (i,j) of a dense matrix `A`.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsRajaSymSparseTriplet::getLocalElement(const hiop::hiopMatrix* A,
                                                           local_ordinal_type row,
                                                           local_ordinal_type col)
{
  const auto* mat = dynamic_cast<const hiopMatrixRajaDense*>(A);

  if(mat != nullptr) {
    auto* amat = const_cast<hiopMatrixRajaDense*>(mat);
    amat->copyFromDev();
    double* M = amat->local_data_const();
    return M[row * amat->get_local_size_n() + col];
  } else
    THROW_NULL_DEREF;
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
real_type MatrixTestsRajaSymSparseTriplet::getLocalElement(const hiop::hiopVector* x, local_ordinal_type i)
{
  const auto* xvec = dynamic_cast<const hiopVectorRajaT*>(x);
  if(xvec != nullptr) {
    auto* axvec = const_cast<hiopVectorRajaT*>(xvec);
    axvec->copyFromDev();
    return xvec->local_data_host_const()[i];
  } else
    THROW_NULL_DEREF;
}

real_type* MatrixTestsRajaSymSparseTriplet::getMatrixData(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiopMatrixSymSparseTripletRajaT*>(A);
  mat->copyFromDev();
  return mat->M_host();
}

const local_ordinal_type* MatrixTestsRajaSymSparseTriplet::getRowIndices(const hiop::hiopMatrixSparse* A)
{
  const auto* mat = dynamic_cast<const hiopMatrixSymSparseTripletRajaT*>(A);
  const_cast<hiopMatrixSymSparseTripletRajaT*>(mat)->copyFromDev();  // UB?
  return mat->i_row_host();
}

const local_ordinal_type* MatrixTestsRajaSymSparseTriplet::getColumnIndices(const hiop::hiopMatrixSparse* A)
{
  const auto* mat = dynamic_cast<const hiopMatrixSymSparseTripletRajaT*>(A);
  const_cast<hiopMatrixSymSparseTripletRajaT*>(mat)->copyFromDev();  // UB?
  return mat->j_col_host();
}

/// Returns size of local data array for vector `x`
int MatrixTestsRajaSymSparseTriplet::getLocalSize(const hiop::hiopVector* x)
{
  const auto* xvec = dynamic_cast<const hiopVectorRajaT*>(x);
  if(xvec != nullptr)
    return static_cast<int>(xvec->get_local_size());
  else
    THROW_NULL_DEREF;
}

/*
 * Pass a function-like object to calculate the expected
 * answer dynamically, based on the row and column
 */
[[nodiscard]]
int MatrixTestsRajaSymSparseTriplet::verifyAnswer(hiop::hiopMatrixDense* Amat,
                                                  std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
  auto* A = dynamic_cast<hiopMatrixRajaDense*>(Amat);
  assert(A->get_local_size_n() == A->n() && "Matrix should not be distributed");
  const local_ordinal_type M = A->get_local_size_m();
  const local_ordinal_type N = A->get_local_size_n();
  int fail = 0;
  A->copyFromDev();
  double* mat = A->local_data_host();
  for(local_ordinal_type i = 0; i < M; i++) {
    for(local_ordinal_type j = 0; j < N; j++) {
      if(!isEqual(mat[i * N + j], expect(i, j))) {
        // printf("(%d, %d) failed. %f != %f.\n", i, j, mat[i][j], expect(i, j));
        fail++;
      }
    }
  }
  return fail;
}

[[nodiscard]]
int MatrixTestsRajaSymSparseTriplet::verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect)
{
  const local_ordinal_type N = getLocalSize(x);

  auto* xvec = dynamic_cast<hiopVectorRajaT*>(x);
  xvec->copyFromDev();
  const auto* vec = xvec->local_data_host_const();

  int local_fail = 0;
  for(int i = 0; i < N; i++) {
    if(!isEqual(vec[i], expect(i))) {
      // printf("%d failed. %f != %f (exp.)\n", i, vec[i], expect(i));
      ++local_fail;
    }
  }
  return local_fail;
}

local_ordinal_type* MatrixTestsRajaSymSparseTriplet::numNonzerosPerRow(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiopMatrixSymSparseTripletRajaT*>(A);
  mat->copyFromDev();
  auto nnz = mat->numberOfNonzeros();
  auto iRow = mat->i_row_host();
  auto jCol = mat->j_col_host();
  auto sparsity_pattern = new local_ordinal_type[mat->m()];
  std::memset(sparsity_pattern, 0, sizeof(local_ordinal_type) * mat->m());

  for(local_ordinal_type i = 0; i < nnz; i++) {
    sparsity_pattern[iRow[i]]++;
    if(iRow[i] != jCol[i]) {
      sparsity_pattern[jCol[i]]++;
    }
  }
  return sparsity_pattern;
}

local_ordinal_type* MatrixTestsRajaSymSparseTriplet::numNonzerosPerCol(hiop::hiopMatrixSparse* A)
{
  auto* mat = dynamic_cast<hiopMatrixSymSparseTripletRajaT*>(A);
  mat->copyFromDev();
  auto nnz = mat->numberOfNonzeros();
  auto iRow = mat->i_row_host();
  auto jCol = mat->j_col_host();
  auto sparsity_pattern = new local_ordinal_type[mat->n()];
  std::memset(sparsity_pattern, 0, sizeof(local_ordinal_type) * mat->n());

  for(local_ordinal_type i = 0; i < nnz; i++) {
    sparsity_pattern[jCol[i]]++;
    if(iRow[i] != jCol[i]) {
      sparsity_pattern[iRow[i]]++;
    }
  }
  return sparsity_pattern;
}

}  // namespace tests
}  // namespace hiop
