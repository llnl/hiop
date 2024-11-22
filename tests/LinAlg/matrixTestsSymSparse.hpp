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
 * @file matrixTestsSymSparse.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */

#pragma once

#include <iostream>
#include <functional>

#include <hiopMatrixSparseTriplet.hpp>
#include <hiopMatrixSparse.hpp>
#include <hiopMatrixDense.hpp>
#include <hiopVector.hpp>
#include "testBase.hpp"

namespace hiop
{
namespace tests
{

/**
 * @brief Tests are re-implemented here if necessary for SparseTriplet Matrices,
 * as the data layout is significantly different compares to dense matrices.
 *
 * Any tests that would modify the sparsity pattern are not implemented.
 * Any tests that would make calls to non-implemented/needed functions are not implemented.
 *
 */
class MatrixTestsSymSparse : public TestBase
{
public:
  MatrixTestsSymSparse() {}
  virtual ~MatrixTestsSymSparse() {}

  /// @brief Test y <- beta * y + alpha * A * x
  bool matrixTimesVec(hiop::hiopMatrixSparse& A, hiop::hiopVector& y, hiop::hiopVector& x)
  {
    assert(y.get_size() == A.m() && "Did you pass in vectors of the correct sizes?");
    assert(x.get_size() == A.n() && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = two, beta = half, A_val = half, y_val = two, x_val = three;
    int fail = 0;

    y.setToConstant(y_val);
    x.setToConstant(x_val);
    A.setToConstant(A_val);
    local_ordinal_type* sparsity_pattern = numNonzerosPerRow(&A);

    A.timesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y, [=](local_ordinal_type i) {
      const local_ordinal_type numValuesInRow = sparsity_pattern[i];
      return (beta * y_val) + (alpha * A_val * x_val * numValuesInRow);
    });

    delete[] sparsity_pattern;
    printMessage(fail, __func__);
    return fail;
  }

  /**
   * Block of W += alpha*A
   *
   * @pre W is square
   * @pre A is symmetric sparse matrix
   *
   */
  bool matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(hiop::hiopMatrixDense& W,
                                                           hiop::hiopMatrixSparse& A,  // sym sparse matrix
                                                           const int rank = 0)
  {
    const local_ordinal_type N_loc = W.get_local_size_n();
    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    assert(W.m() == W.n());
    assert(W.m() >= A.m());
    assert(W.n() >= A.n());

    // The offset must be on dense matrix diagonal
    const local_ordinal_type start_diag = N_loc - A_N_loc;
    const real_type alpha = half, A_val = half, W_val = one;
    int fail = 0;

    // Check with non-1 alpha
    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.addUpperTriangleToSymDenseMatrixUpperTriangle(start_diag, alpha, W);

    // get sparsity pattern
    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    fail += verifyAnswer(&W, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // check if (i, j) within bounds of A
      // then check if (i, j) within upper triangle of W
      const bool isUpperTriangle =
          (i >= start_diag && i < start_diag + A_M && j >= start_diag && j < start_diag + A_N_loc && j >= i);

      int i_sp = i - start_diag;
      int j_sp = j - start_diag;
      // only nonzero entries in A will be added
      const bool indexExists =
          (find_unsorted_pair(i_sp, j_sp, iRow, jCol, nnz) || find_unsorted_pair(j_sp, i_sp, iRow, jCol, nnz));
      real_type ans = (isUpperTriangle && indexExists) ? W_val + A_val * alpha : W_val;  // 1 + .5 * .5 = 1.25
      return ans;
    });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /// @todo Document this test for `startingAtAddSubDiagonalToStartingAt`
  bool matrixStartingAtAddSubDiagonalToStartingAt(hiop::hiopVector& W, hiop::hiopMatrixSparse& A, const int rank = 0)
  {
    assert(W.get_size() == A.m());  // A is square matrix

    const auto start_src_idx = 0;
    const auto start_dest_idx = 0;
    const auto num_elems = W.get_size();
    const auto A_val = half;
    const auto W_val = one;
    const auto alpha = half;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.startingAtAddSubDiagonalToStartingAt(start_src_idx, alpha, W, start_dest_idx, num_elems);

    const auto* iRow = getRowIndices(&A);
    const auto* jCol = getColumnIndices(&A);
    auto nnz = A.numberOfNonzeros();
    const auto fail = verifyAnswer(&W, [=](local_ordinal_type i) -> real_type {
      const bool indexExists = find_unsorted_pair(i, i, iRow, jCol, nnz);
      return (indexExists) ? (W_val + A_val * alpha) : W_val;
    });

    printMessage(fail, __func__, rank);
    return fail;
  }

  /// @brief copy sublok `B` into `A`, and add diagonal `diag` to `A`, `W` is a buffer used to verify solution
  bool matrix_set_Hess_FR(hiop::hiopMatrixDense& W,
                          hiop::hiopMatrixSparse& A,
                          hiop::hiopMatrixSparse& B,
                          hiop::hiopVector& diag,
                          const int rank = 0)
  {
    assert(A.m() == A.n());            // A is square matrix
    assert(diag.get_size() == B.m());  // B is square matrix
    assert(A.m() >= B.m());            // A is larger or equal to B
    assert(W.m() == W.n());            // W is square matrix
    assert(W.m() == A.m());            // W has same dim as A

    const local_ordinal_type A_M = A.m();
    const local_ordinal_type A_N_loc = A.n();
    const local_ordinal_type B_M = B.m();

    const auto B_val = one;
    const auto W_val = zero;
    const auto D_val = two;
    const real_type alpha = one;
    const local_ordinal_type start_diag = 0;
    int fail = 0;

    B.setToConstant(B_val);
    diag.setToConstant(D_val);

    A.set_Hess_FR(B, A.i_row(), A.j_col(), A.M(), diag);

    // copy to a dense matrix
    W.setToConstant(W_val);
    A.addUpperTriangleToSymDenseMatrixUpperTriangle(start_diag, alpha, W);

    // get sparsity pattern
    const auto* iRowB = getRowIndices(&B);
    const auto* jColB = getColumnIndices(&B);
    auto nnzB = B.numberOfNonzeros();
    fail += verifyAnswer(&W, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // check if (i, j) within bounds of A
      // then check if (i, j) within upper triangle of W
      const bool isUpperTriangle =
          (i >= start_diag && i < start_diag + A_M && j >= start_diag && j < start_diag + A_N_loc && j >= i);

      // only nonzero entries in A will be added to W
      int i_sp = i - start_diag;
      int j_sp = j - start_diag;
      const bool sp_index_B = (i < B_M && j < B_M) && (find_unsorted_pair(i_sp, j_sp, iRowB, jColB, nnzB) ||
                                                       find_unsorted_pair(j_sp, i_sp, iRowB, jColB, nnzB));
      const bool diag_index = (i < B_M && j < B_M) && (i == j);

      real_type ans;

      if(isUpperTriangle && sp_index_B && diag_index) {
        // found in sparse matirx B and it is a diagonal entry in B
        ans = B_val + D_val;
      } else if(isUpperTriangle && sp_index_B) {
        // found in sparse matirx B
        ans = B_val;
      } else if(isUpperTriangle && diag_index) {
        // NOT found in sparse matirx B. It comes from extra diag term
        ans = D_val;
      } else {
        ans = W_val;
      }
      return ans;
    });

    printMessage(fail, __func__, rank);
    return fail;
  }

protected:
  /// TODO: The sparse matrix is not distributed - all is local.
  virtual real_type getLocalElement(const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) = 0;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) = 0;
  virtual real_type* getMatrixData(hiop::hiopMatrixSparse* a) = 0;
  virtual const local_ordinal_type* getRowIndices(const hiop::hiopMatrixSparse* a) = 0;
  virtual const local_ordinal_type* getColumnIndices(const hiop::hiopMatrixSparse* a) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopMatrixDense* A,
                           std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect) = 0;
  virtual local_ordinal_type* numNonzerosPerRow(hiop::hiopMatrixSparse* mat) = 0;
  virtual local_ordinal_type* numNonzerosPerCol(hiop::hiopMatrixSparse* mat) = 0;

private:
  // linearly scans an unsorted array
  static bool find_unsorted_pair(int valA, int valB, const int* arrA, const int* arrB, size_t arrslen)
  {
    for(int i = 0; i < static_cast<int>(arrslen); i++) {
      if(arrA[i] == valA && arrB[i] == valB) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace tests
}  // namespace hiop
