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
 * @file matrixTestsDenseRowMajor.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Robert Rutherford <robert.rutherford@pnnl.gov>, PNNL
 *
 */
#include <hiopMatrixDenseRowMajor.hpp>
#include "matrixTestsDenseRowMajor.hpp"

namespace hiop
{
namespace tests
{

//
// Matrix helper methods
//

/// Get number of rows in local data block of matrix _A_
local_ordinal_type MatrixTestsDenseRowMajor::getNumLocRows(const hiop::hiopMatrixDense* A)
{
  const auto* amat = dynamic_cast<const hiop::hiopMatrixDenseRowMajor*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  // get_local_size_m returns global ordinal type! HiOp issue?
  return static_cast<local_ordinal_type>(amat->get_local_size_m());
  //                                                         ^^^
}

/// Get number of columns in local data block of matrix _A_
local_ordinal_type MatrixTestsDenseRowMajor::getNumLocCols(const hiop::hiopMatrixDense* A)
{
  const auto* amat = dynamic_cast<const hiop::hiopMatrixDenseRowMajor*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  // Local sizes should be returned as local ordinal type
  return static_cast<local_ordinal_type>(amat->get_local_size_n());
  //                                                         ^^^
}

/// Set local data element (i,j) of matrix _A_ to _val_.
void MatrixTestsDenseRowMajor::setLocalElement(hiop::hiopMatrixDense* A,
                                               local_ordinal_type i,
                                               local_ordinal_type j,
                                               real_type val)
{
  auto* amat = dynamic_cast<hiop::hiopMatrixDenseRowMajor*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  real_type* data = amat->local_data();
  local_ordinal_type ncols = getNumLocCols(A);
  // data[i][j] = val;
  data[i * ncols + j] = val;
}

/// Method to set a single local row of matrix to a constant value
void MatrixTestsDenseRowMajor::setLocalRow(hiop::hiopMatrixDense* A, const local_ordinal_type row, const real_type val)
{
  auto* amat = dynamic_cast<hiop::hiopMatrixDenseRowMajor*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  const local_ordinal_type N = getNumLocCols(amat);
  for(int i = 0; i < N; i++) {
    setLocalElement(amat, row, i, val);
  }
}

/// Returns by value local element (i,j) of matrix _A_.
real_type MatrixTestsDenseRowMajor::getLocalElement(const hiop::hiopMatrixDense* A,
                                                    local_ordinal_type i,
                                                    local_ordinal_type j)
{
  const auto* amat = dynamic_cast<const hiop::hiopMatrixDenseRowMajor*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  const real_type* data = amat->local_data_const();
  local_ordinal_type ncols = getNumLocCols(A);
  return data[i * ncols + j];
}

/// Get MPI communicator of matrix _A_
MPI_Comm MatrixTestsDenseRowMajor::getMPIComm(hiop::hiopMatrixDense* A)
{
  auto* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  return amat->get_mpi_comm();
}

/// Returns const pointer to local data block of matrix _A_.
const real_type* MatrixTestsDenseRowMajor::getLocalDataConst(hiop::hiopMatrixDense* A)
{
  auto* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  return amat->local_data_const();
}

/// Returns const pointer to local data block of matrix _A_.
real_type* MatrixTestsDenseRowMajor::getLocalData(hiop::hiopMatrixDense* A)
{
  auto* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
  if(amat == nullptr) THROW_NULL_DEREF;

  return amat->local_data();
}

/// Reduce return output: Every rank returns failure if any individual rank fails
bool MatrixTestsDenseRowMajor::reduceReturn(int failures, hiop::hiopMatrixDense* A)
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
int MatrixTestsDenseRowMajor::verifyAnswer(hiop::hiopMatrixDense* A, const double answer)
{
  local_ordinal_type mrows = getNumLocRows(A);
  local_ordinal_type ncols = getNumLocCols(A);

  int fail = 0;
  for(local_ordinal_type i = 0; i < mrows; i++) {
    for(local_ordinal_type j = 0; j < ncols; j++) {
      if(!isEqual(getLocalElement(A, i, j), answer)) {
        // std::cout << i << " " << j << " : "
        //           << getLocalElement(A, i, j) << " != "
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
int MatrixTestsDenseRowMajor::verifyAnswer(hiop::hiopMatrixDense* A,
                                           std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
  local_ordinal_type mrows = getNumLocRows(A);
  local_ordinal_type ncols = getNumLocCols(A);

  int fail = 0;
  for(local_ordinal_type i = 0; i < mrows; i++) {
    for(local_ordinal_type j = 0; j < ncols; j++) {
      if(!isEqual(getLocalElement(A, i, j), expect(i, j))) {
        // std::cout << i << " " << j << " : "
        //           << getLocalElement(A, i, j) << " != "
        //           << expect(i, j) << "\n";
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
int MatrixTestsDenseRowMajor::getLocalSize(const hiop::hiopVector* x)
{
  const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  return static_cast<int>(xvec->get_local_size());
}

/// Sets a local data element of vector _x_
void MatrixTestsDenseRowMajor::setLocalElement(hiop::hiopVector* x, const local_ordinal_type i, const real_type val)
{
  auto* xvec = dynamic_cast<hiop::hiopVectorPar*>(x);
  if(xvec == nullptr) {
    THROW_NULL_DEREF;
  }

  real_type* data = x->local_data();
  data[i] = val;
}

/// Returns local data element _i_ of vector _x_.
real_type MatrixTestsDenseRowMajor::getLocalElement(const hiop::hiopVector* x, local_ordinal_type i)
{
  const auto* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
  if(xvec == nullptr) THROW_NULL_DEREF;

  return xvec->local_data_const()[i];
}

/// Checks if _local_ vector elements are set to `answer`.
[[nodiscard]]
int MatrixTestsDenseRowMajor::verifyAnswer(hiop::hiopVector* x, double answer)
{
  const local_ordinal_type N = getLocalSize(x);

  int local_fail = 0;
  for(local_ordinal_type i = 0; i < N; ++i) {
    if(!isEqual(getLocalElement(x, i), answer)) {
      ++local_fail;
    }
  }

  return local_fail;
}

/// Checks if _local_ vector elements match `expected` values.
[[nodiscard]]
int MatrixTestsDenseRowMajor::verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect)
{
  const local_ordinal_type N = getLocalSize(x);

  int local_fail = 0;
  for(int i = 0; i < N; i++) {
    if(!isEqual(getLocalElement(x, i), expect(i))) {
      ++local_fail;
    }
  }
  return local_fail;
}

// End helper methods

}  // namespace tests
}  // namespace hiop
