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
 * @file MatrixTestsRajaDense.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Robert Rutherford <robert.rutherford@pnnl.gov>, PNNL
 *
 */

#include <hiopMatrixDenseRaja.hpp>
#include <hiopVectorRaja.hpp>
#include "matrixTestsRajaDense.hpp"

// TODO: this is a quick hack. Will need to modify this class to be aware of the instantiated
//  template parameters for vector and matrix RAJA classes. Likely a better approach would be
//  to revise the tests to try out multiple configurations of the memory backends and execution
//  policies for RAJA dense matrix.

#if defined(HIOP_USE_CUDA)
#include <ExecPoliciesRajaCudaImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
using hiopMatrixRajaDense = hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
#endif

namespace hiop
{
namespace tests
{

//
// Matrix helper methods
//

/// Get number of rows in local data block of matrix _A_
local_ordinal_type MatrixTestsRajaDense::getNumLocRows(const hiop::hiopMatrixDense* A)
{
  const auto* amat = dynamic_cast<const hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  return amat->get_local_size_m();
  //                         ^^^
}

/// Get number of columns in local data block of matrix _A_
local_ordinal_type MatrixTestsRajaDense::getNumLocCols(const hiop::hiopMatrixDense* A)
{
  const auto* amat = dynamic_cast<const hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  return amat->get_local_size_n();
  //                         ^^^
}

/// Set local data element (i,j) of matrix _A_ to _val_ in current memory space.
void MatrixTestsRajaDense::setLocalElement(hiop::hiopMatrixDense* A,
                                           local_ordinal_type i,
                                           local_ordinal_type j,
                                           real_type val)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  amat->copyFromDev();
  real_type* data = amat->local_data_host();
  local_ordinal_type ncols = getNumLocCols(A);
  data[i * ncols + j] = val;
  amat->copyToDev();
}

/// Set a single local row of matrix to a constant value in current memory space.
void MatrixTestsRajaDense::setLocalRow(hiop::hiopMatrixDense* A, const local_ordinal_type row, const real_type val)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  const local_ordinal_type N = getNumLocCols(A);
  amat->copyFromDev();
  real_type* data = amat->local_data_host();
  for(int j = 0; j < N; j++) {
    data[row * N + j] = val;
  }
  amat->copyToDev();
}

/// Returns by value local element (i,j) of matrix _A_.
real_type MatrixTestsRajaDense::getLocalElement(const hiop::hiopMatrixDense* A, local_ordinal_type i, local_ordinal_type j)
{
  const auto* am = dynamic_cast<const hiopMatrixRajaDense*>(A);
  if(am == nullptr) THROW_NULL_DEREF;
  auto* amat = const_cast<hiopMatrixRajaDense*>(am);
  if(amat == nullptr) THROW_NULL_DEREF;

  amat->copyFromDev();
  const real_type* data = amat->local_data_host();
  local_ordinal_type ncols = getNumLocCols(A);
  return data[i * ncols + j];
}

/// Get MPI communicator of matrix _A_
MPI_Comm MatrixTestsRajaDense::getMPIComm(hiop::hiopMatrixDense* A)
{
  const auto* amat = dynamic_cast<const hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;
  return amat->get_mpi_comm();
}

/// Returns pointer to local data block of matrix _A_ in current memory space.
const real_type* MatrixTestsRajaDense::getLocalDataConst(hiop::hiopMatrixDense* A)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;
  return amat->local_data_const();
}

/// Returns pointer to local data block of matrix _A_ in current memory space.
real_type* MatrixTestsRajaDense::getLocalData(hiopMatrixDense* A)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;
  return amat->local_data();
}

/// Reduce return output: Every rank returns failure if any individual rank fails
bool MatrixTestsRajaDense::reduceReturn(int failures, hiopMatrixDense* A)
{
  int fail = 0;

#ifdef HIOP_USE_MPI
  MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(A));
#else
  (void)A;
  fail = failures;
#endif

  return (fail != 0);
}

/// Verify matrix elements are set to `answer`.
[[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopMatrixDense* A, const double answer)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;
  const local_ordinal_type M = getNumLocRows(amat);
  const local_ordinal_type N = getNumLocCols(amat);

  // Copy data to the host mirror
  amat->copyFromDev();
  // Get pointer to dense matrix local data on the host
  const real_type* local_matrix_data = amat->local_data_host();

  int fail = 0;
  // RAJA matrix is stored in row-major format
  for(local_ordinal_type i = 0; i < M; i++) {
    for(local_ordinal_type j = 0; j < N; j++) {
      if(!isEqual(local_matrix_data[i * N + j], answer)) {
        // std::cout << i << " " << j << " = "
        //           << local_matrix_data[i*N + j] << " != "
        //           << answer << "\n";
        fail++;
      }
    }
  }
  return fail;
}

/**
 * Verify matrix elements are set as defined in `expect`.
 */
[[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopMatrixDense* A,
                                       std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
  auto* amat = dynamic_cast<hiopMatrixRajaDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;
  const local_ordinal_type M = getNumLocRows(amat);
  const local_ordinal_type N = getNumLocCols(amat);

  // Copy data to the host mirror
  amat->copyFromDev();
  // Get pointer to dense matrix local data on the host
  const real_type* local_matrix_data = amat->local_data_host();

  int fail = 0;
  for(local_ordinal_type i = 0; i < M; i++) {
    for(local_ordinal_type j = 0; j < N; j++) {
      if(!isEqual(local_matrix_data[i * N + j], expect(i, j))) {
        std::cout << i << ", " << j << ": = " << local_matrix_data[i * N + j] << " != " << expect(i, j) << "\n";
        fail++;
      }
    }
  }
  return fail;
}

//
// Vector helper methods
//

/// Returns size of local data array for vector _x_
local_ordinal_type MatrixTestsRajaDense::getLocalSize(const hiop::hiopVector* x)
{
  const auto* xvec = dynamic_cast<const hiopVectorRajaT*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  return xvec->get_local_size();
}

/// Sets a local data element of vector _x_ in current memory space
void MatrixTestsRajaDense::setLocalElement(hiop::hiopVector* x, const local_ordinal_type i, const real_type val)
{
  auto* xvec = dynamic_cast<hiopVectorRajaT*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  xvec->copyFromDev();
  real_type* data = xvec->local_data_host();
  data[i] = val;
  xvec->copyToDev();
}

/// Returns local data element _i_ of vector _x_ by value on the host.
real_type MatrixTestsRajaDense::getLocalElement(const hiop::hiopVector* x, local_ordinal_type i)
{
  const auto* xv = dynamic_cast<const hiopVectorRajaT*>(x);
  if(xv == nullptr) THROW_NULL_DEREF;
  auto* xvec = const_cast<hiopVectorRajaT*>(xv);
  if(xvec == nullptr) THROW_NULL_DEREF;

  xvec->copyFromDev();
  return xvec->local_data_host_const()[i];
}

/// Checks if _local_ vector elements are set to `answer`.
[[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopVector* x, double answer)
{
  auto* xvec = dynamic_cast<hiopVectorRajaT*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  const local_ordinal_type N = getLocalSize(xvec);

  // Copy vector local data to the host mirror
  xvec->copyFromDev();
  // Get raw pointer to the host mirror
  const real_type* local_data = xvec->local_data_host_const();

  int local_fail = 0;
  for(local_ordinal_type i = 0; i < N; ++i) {
    if(!isEqual(local_data[i], answer)) {
      ++local_fail;
    }
  }

  return local_fail;
}

/// Checks if _local_ vector elements match `expected` values.
[[nodiscard]]
int MatrixTestsRajaDense::verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect)
{
  auto* xvec = dynamic_cast<hiopVectorRajaT*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  const local_ordinal_type N = getLocalSize(xvec);

  // Copy vector local data to the host mirror
  xvec->copyFromDev();
  // Get raw pointer to the host mirror
  const real_type* local_data = xvec->local_data_host_const();

  int local_fail = 0;
  for(local_ordinal_type i = 0; i < N; ++i) {
    if(!isEqual(local_data[i], expect(i))) {
      ++local_fail;
    }
  }
  return local_fail;
}

// End helper methods

}  // namespace tests
}  // namespace hiop
