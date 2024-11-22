// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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
 * @file matrixTestsDense.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Jake Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Robert Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#pragma once

#include <iostream>
#include <functional>
#include <cassert>
#include <utility>
#include <hiopVector.hpp>
#include <hiopMatrixDense.hpp>
#include "testBase.hpp"

// for processing indexes/ints arrays on host
#include "hiopVectorIntSeq.hpp"

namespace hiop
{
namespace tests
{

/**
 * @brief Collection of tests for abstract hiopMatrixDense implementations.
 *
 * This class contains implementation of all dense matrix unit tests and abstract
 * interface for testing utility functions, which are specific to the particular
 * matrix and vector implementations.
 *
 * To add a new test, simply add a new public method to this class and call it
 * from function runTests implemented in file testMatrixDense.cpp. Use helper
 * functions to abstract implementation specific details such as local data
 * size and memory space, accessing local data elements, etc.
 *
 * If you want to add tests for a new dense matrix implementation (e.g.
 * column-major), you will need to reimplement helper functions, as well.
 *
 * @warning HiOp distributed memory partitioning is 1-D and some of the unit
 * tests here implicitly assume that. When and if HiOp MPI partitioning
 * changes, these tests will have to be rewritten.
 */

class MatrixTestsDense : public TestBase
{
public:
  MatrixTestsDense() {}
  virtual ~MatrixTestsDense() {}

  int matrixSetToZero(hiop::hiopMatrixDense& A, const int rank)
  {
    A.setToConstant(one);
    A.setToZero();
    const int fail = verifyAnswer(&A, zero);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixSetToConstant(hiop::hiopMatrixDense& A, const int rank)
  {
    A.setToConstant(one);
    int fail = verifyAnswer(&A, one);
    A.setToConstant(two);
    fail += verifyAnswer(&A, two);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixCopyFrom(hiopMatrixDense& dst, hiopMatrixDense& src, const int rank)
  {
    assert(dst.n() == src.n() && "Did you pass in matrices of the same size?");
    assert(dst.m() == src.m() && "Did you pass in matrices of the same size?");
    assert(getNumLocRows(&dst) == getNumLocRows(&src) && "Did you pass in matrices of the same size?");
    assert(getNumLocCols(&dst) == getNumLocCols(&src) && "Did you pass in matrices of the same size?");
    const real_type src_val = one;

    // Test copying src another matrix
    src.setToConstant(src_val);
    dst.setToZero();

    dst.copyFrom(src);
    int fail = verifyAnswer(&dst, src_val);

    // test copying src a raw buffer
    const real_type* src_buf = getLocalDataConst(&src);
    dst.setToZero();

    dst.copyFrom(src_buf);
    fail += verifyAnswer(&dst, src_val);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  int matrix_copy_to(hiopMatrixDense& dst, hiopMatrixDense& src, const int rank)
  {
    assert(dst.n() == src.n() && "Did you pass in matrices of the same size?");
    assert(dst.m() == src.m() && "Did you pass in matrices of the same size?");
    assert(getNumLocRows(&dst) == getNumLocRows(&src) && "Did you pass in matrices of the same size?");
    assert(getNumLocCols(&dst) == getNumLocCols(&src) && "Did you pass in matrices of the same size?");
    const real_type src_val = one;

    // Test copying to dest
    src.setToConstant(src_val);
    dst.setToZero();

    // test copying src a raw buffer
    real_type* dst_buf = getLocalData(&dst);
    dst.setToZero();

    src.copy_to(dst_buf);
    int fail = verifyAnswer(&dst, src_val);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  /*
   * y_{glob} \leftarrow \beta y_{glob} + \alpha A_{glob \times loc} x_{loc}
   */
  int matrixTimesVec(hiop::hiopMatrixDense& A, hiop::hiopVector& y, hiop::hiopVector& x, const int rank = 0)
  {
    assert(getLocalSize(&y) == getNumLocRows(&A) && "Did you pass in vectors of the correct sizes?");
    assert(getLocalSize(&x) == getNumLocCols(&A) && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one;
    const real_type beta = one;
    const real_type A_val = one;
    const real_type y_val = three;
    const real_type x_val = three;
    const real_type N_glob = static_cast<real_type>(A.n());
    int fail = 0;

    y.setToConstant(y_val);
    x.setToConstant(x_val);
    A.setToConstant(A_val);

    A.timesVec(beta, y, alpha, x);

    real_type expected = (beta * y_val) + (alpha * A_val * x_val * N_glob);
    fail += verifyAnswer(&y, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * y = beta * y + alpha * A^T * x
   *
   * Notice that since A^T, x must not be distributed in this case, whereas
   * the plain `timesVec' nessecitated that x be distributed and y not be.
   */
  int matrixTransTimesVec(hiop::hiopMatrixDense& A, hiop::hiopVector& x, hiop::hiopVector& y, const int rank = 0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);

    assert(getLocalSize(&x) == getNumLocRows(&A) && "Did you pass in vectors of the correct sizes?");
    assert(getLocalSize(&y) == getNumLocCols(&A) && "Did you pass in vectors of the correct sizes?");
    const real_type alpha = one;
    const real_type beta = one;
    const real_type A_val = one;
    const real_type y_val = three;
    const real_type x_val = three;
    // Take m() because A will be transposed
    const real_type N_glob = static_cast<real_type>(A.m());
    int fail = 0;

    // Index of row of A that will be set to zero,
    // and index of y that will be beta * y_val
    const local_ordinal_type index_to_zero = N - 1;

    A.setToConstant(A_val);
    y.setToConstant(y_val);
    x.setToConstant(x_val);

    /*
     * Zero a row of A^T to test that the resulting vector
     * has its initial value as the first element, ensuring that
     * the matrix is correctly transposed.
     */
    for(int i = 0; i < M; i++) {
      setLocalElement(&A, i, index_to_zero, zero);
    }

    A.transTimesVec(beta, y, alpha, x);

    fail += verifyAnswer(&y, [=](local_ordinal_type i) -> real_type {
      const bool isZerodRow = (i == index_to_zero);
      return isZerodRow ? beta * y_val : (beta * y_val) + (alpha * A_val * x_val * N_glob);
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /**
   *  W = beta * W + alpha * A * M
   *
   * Shapes:
   *   A: KxM
   *   M: MxN
   *   W: KxN
   *   all local
   */
  int matrixTimesMat(hiop::hiopMatrixDense& A, hiop::hiopMatrixDense& X, hiop::hiopMatrixDense& W, const int rank = 0)
  {
    const local_ordinal_type K = getNumLocCols(&A);
    assert(K == A.n());
    assert(getNumLocCols(&X) == X.n());
    assert(K == getNumLocRows(&X));
    assert(getNumLocRows(&A) == getNumLocRows(&W));
    assert(getNumLocCols(&X) == getNumLocCols(&W));
    const real_type A_val = two, X_val = three, W_val = two, alpha = two, beta = two;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    X.setToConstant(X_val);
    A.timesMat(beta, W, alpha, X);
    real_type expected = (beta * W_val) + (alpha * A_val * X_val * K);

    const int fail = verifyAnswer(&W, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   *  W = beta * W + alpha * this^T * X
   *
   *  A: kxm local
   *  W: mxn
   *  X: kxn
   *
   */
  int matrixTransTimesMat(hiop::hiopMatrixDense& A_local, hiop::hiopMatrixDense& W, hiop::hiopMatrixDense& X, const int rank)
  {
    const local_ordinal_type K = getNumLocRows(&A_local);
    const global_ordinal_type N_loc = getNumLocCols(&X);
    assert(getNumLocCols(&A_local) == getNumLocRows(&W) && "Matrices have mismatched shapes");
    assert(X.n() == W.n() && "Matrices have mismatched shapes");
    assert(N_loc == getNumLocCols(&W) && "Matrices have mismatched shapes");
    assert(K == getNumLocRows(&X) && "Matrices have mismatched shapes");
    const real_type A_val = two, X_val = three, W_val = two, alpha = two, beta = two;

    /*
     * One row of X will be set to zero to ensure
     * the matrix multiply and transpose operations are
     * working correctly.
     */
    const int idx_of_zero_row = K - 1;

    A_local.setToConstant(A_val);
    W.setToConstant(W_val);
    X.setToConstant(X_val);

    // X[idx][:] = 0
    for(int i = 0; i < N_loc; i++) {
      setLocalElement(&X, idx_of_zero_row, i, zero);
    }

    A_local.transTimesMat(beta, W, alpha, X);

    real_type expected = (beta * W_val) + (alpha * A_val * X_val * (K - 1));
    const int fail = verifyAnswer(&W, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &W);
  }

  /*
   *  W = beta * W + alpha * this * X^T
   *
   *  A: mxn
   *  W: mxk local
   *  X: kxn
   *
   */
  int matrixTimesMatTrans(hiop::hiopMatrixDense& A, hiop::hiopMatrixDense& W_local, hiop::hiopMatrixDense& X, const int rank)
  {
    assert(getNumLocRows(&A) == getNumLocRows(&W_local) && "Matrices have mismatched sizes");
    assert(getNumLocRows(&X) == getNumLocCols(&W_local) && "Matrices have mismatched sizes");
    assert(getNumLocCols(&A) == getNumLocCols(&X) && "Matrices have mismatched sizes");
    const real_type A_val = two, X_val = three, W_val = two, alpha = two, beta = two;
    const real_type Nglob = static_cast<real_type>(A.n());

    A.setToConstant(A_val);
    W_local.setToConstant(W_val);
    X.setToConstant(X_val);

    // Set a row of X to zero
    local_ordinal_type idx_of_zero_row = getNumLocRows(&X) - 1;
    setLocalRow(&X, idx_of_zero_row, zero);

    A.timesMatTrans(beta, W_local, alpha, X);

    // Column of W with second term equal to zero
    local_ordinal_type idx_of_zero_col = getNumLocCols(&W_local) - 1;
    int fail = verifyAnswer(&W_local, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      return j == idx_of_zero_col ? (beta * W_val) : (beta * W_val) + (alpha * A_val * X_val * Nglob);
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * this += alpha * diag
   */
  int matrixAddDiagonal(hiop::hiopMatrixDense& A, hiop::hiopVector& x, const int rank = 0)
  {
    int fail = 0;
    assert(getNumLocCols(&A) == getLocalSize(&x));
    assert(getNumLocRows(&A) == getLocalSize(&x));
    assert(getNumLocRows(&A) == A.n());
    assert(A.n() == x.get_size());
    assert(A.m() == x.get_size());
    static const real_type alpha = two, A_val = quarter, x_val = half;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addDiagonal(alpha, x);
    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isOnDiagonal = (i == j);
      return isOnDiagonal ? A_val + x_val * alpha : A_val;
    });

    A.setToConstant(A_val);
    A.addDiagonal(alpha);
    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isOnDiagonal = (i == j);
      return isOnDiagonal ? A_val + alpha : A_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /**
   * @breif this += alpha * subdiag
   *
   * @note this test checks all three overloads:
   *   - addSubDiagonal(const double&, int_type, const hiopVector&)
   *   - addSubDiagonal(int, const double&, const hiopVector&, int, int)
   *   - addSubDiagonal(int, int, const double&)
   */
  int matrixAddSubDiagonal(hiop::hiopMatrixDense& A, hiop::hiopVector& x, const int rank = 0)
  {
    int fail = 0;
    const local_ordinal_type N = getNumLocCols(&A);
    const local_ordinal_type x_len = getLocalSize(&x);
    const real_type alpha = half;
    const real_type A_val = half;
    const real_type x_val = one;
    assert(N == A.n() && "Test should only be ran sequentially.");
    assert(N == A.m() && "Test should only run with symmetric matrices.");

    // Test the overload that assumes the entire source vector
    // will be added to the subdiagonal
    local_ordinal_type start_idx = N - x_len;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addSubDiagonal(alpha, start_idx, x);
    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isOnSubDiagonal = (i >= start_idx && i == j);
      return isOnSubDiagonal ? A_val + x_val * alpha : A_val;
    });

    // We're only going to add n-1 elements of the vector
    // Test the overload that specifies subset of the vector
    // to be added to subdiagonal
    local_ordinal_type start_idx_src = 1;
    local_ordinal_type num_elements_to_add = x_len - start_idx_src;
    local_ordinal_type start_idx_dest = (N - x_len) + start_idx_src;

    A.setToConstant(A_val);
    x.setToConstant(x_val);
    A.addSubDiagonal(start_idx_dest, alpha, x, start_idx_src, num_elements_to_add);

    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isOnSubDiagonal = (i >= start_idx_dest && i == j);
      return isOnSubDiagonal ? A_val + x_val * alpha : A_val;
    });

    // Operating on N-2 elements s.t. the first and last elements of the sub
    // diagonal are not operated on.
    start_idx_dest = 1;
    const double c = two;
    const int num_elems = N - 2;
    A.setToConstant(A_val);
    A.addSubDiagonal(start_idx_dest, num_elems, c);
    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isOperatedOn = i >= start_idx_dest && i == j && i < start_idx_dest + num_elems;
      return isOperatedOn ? A_val + c : A_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * A += alpha * B
   */
  int matrixAddMatrix(hiop::hiopMatrixDense& A, hiop::hiopMatrixDense& B, const int rank)
  {
    assert(getNumLocRows(&A) == getNumLocRows(&B));
    assert(getNumLocCols(&A) == getNumLocCols(&B));
    const real_type alpha = half, A_val = half, B_val = one;

    A.setToConstant(A_val);
    B.setToConstant(B_val);
    A.addMatrix(alpha, B);
    const int fail = verifyAnswer(&A, A_val + B_val * alpha);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Block of W += alpha*A
   *
   * Block of W summed with A is in the trasposed
   * location of the same call to addToSymDenseMatrixUpperTriangle
   *
   * Precondition: W is square
   */
  int matrixTransAddToSymDenseMatrixUpperTriangle(hiop::hiopMatrixDense& W, hiop::hiopMatrixDense& A, const int rank = 0)
  {
    const local_ordinal_type N_loc = getNumLocCols(&W);
    const local_ordinal_type A_M = getNumLocRows(&A);
    const local_ordinal_type A_N_loc = getNumLocCols(&A);
    assert(W.m() == W.n());
    assert(getNumLocRows(&W) >= getNumLocRows(&A));
    assert(W.n() >= A.n());

    const local_ordinal_type start_idx_row = 0;
    const local_ordinal_type start_idx_col = N_loc - A_M;
    const real_type alpha = half, A_val = half, W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.transAddToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, W);
    const int fail = verifyAnswer(&W, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isTransUpperTriangle =
          (i >= start_idx_row && i < start_idx_row + A_N_loc && j >= start_idx_col && j < start_idx_col + A_M);

      return isTransUpperTriangle ? W_val + A_val * alpha : W_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Upper diagonal block of W += alpha * A
   *
   * Preconditions:
   * W is square
   * A is square
   * degree of A <= degree of W
   */
  int matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(hiop::hiopMatrixDense& W,
                                                          hiop::hiopMatrixDense& A,
                                                          const int rank = 0)
  {
    const local_ordinal_type A_M = getNumLocRows(&A);
    const local_ordinal_type A_N = getNumLocCols(&A);
    assert(W.m() == W.n());
    assert(A.m() == A.n());
    assert(W.n() >= A.n());
    assert(getNumLocCols(&A) <= getNumLocCols(&W));

    // Map the upper triangle of A to W starting
    // at W's upper left corner
    const local_ordinal_type diag_start = 0;
    int fail = 0;
    const real_type alpha = half, A_val = half, W_val = one;

    A.setToConstant(A_val);
    W.setToConstant(W_val);
    A.addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start, alpha, W);
    fail += verifyAnswer(&W, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      bool isUpperTriangle = (i >= diag_start && i < diag_start + A_N && j >= i && j < diag_start + A_M);
      return isUpperTriangle ? W_val + A_val * alpha : W_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Set bottom right value to ensure that all values
   * are checked.
   */
  int matrixMaxAbsValue(hiop::hiopMatrixDense& A, const int rank)
  {
    const local_ordinal_type last_row_idx = getNumLocRows(&A) - 1;
    const local_ordinal_type last_col_idx = getNumLocCols(&A) - 1;
    int fail = 0;

    // Positive largest value
    A.setToConstant(zero);
    if(rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, one);
    fail += A.max_abs_value() != one;

    // Negative largest value
    A.setToConstant(zero);
    if(rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, -one);
    fail += A.max_abs_value() != one;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Set bottom right value to ensure that all values
   * are checked.
   */
  int matrix_row_max_abs_value(hiop::hiopMatrixDense& A, hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type last_row_idx = getNumLocRows(&A) - 1;
    const local_ordinal_type last_col_idx = getNumLocCols(&A) - 1;
    int fail = 0;

    // set the last element to -2, others are set to 1
    A.setToConstant(one);
    if(rank == 0) {
      setLocalElement(&A, last_row_idx, last_col_idx, -two);
    }

    A.row_max_abs_value(x);

    fail += verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool is_last_row = (i == last_row_idx);
      return is_last_row ? two : one;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * scale each row of A
   */
  int matrix_scale_row(hiop::hiopMatrixDense& A, hiop::hiopVector& x, const int rank)
  {
    const real_type A_val = two;
    const real_type x_val = three;
    int fail = 0;

    x.setToConstant(x_val);
    A.setToConstant(A_val);

    A.scale_row(x, false);

    real_type expected = A_val * x_val;
    fail += verifyAnswer(&A, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /*
   * Set bottom right value to ensure that all values
   * are checked.
   */
  int matrixIsFinite(hiop::hiopMatrixDense& A, const int rank)
  {
    const local_ordinal_type last_row_idx = getNumLocRows(&A) - 1;
    const local_ordinal_type last_col_idx = getNumLocCols(&A) - 1;
    int fail = 0;

    A.setToConstant(zero);
    if(!A.isfinite()) fail++;

    A.setToConstant(zero);
    if(rank == 0) setLocalElement(&A, last_row_idx, last_col_idx, INFINITY);
    if(!A.isfinite() && rank != 0) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  //////////////////////////////////////////////////////////////////////
  // Test for methods from hiopMatrixDense that are not part of the
  // abstract class hiopMatrix
  //////////////////////////////////////////////////////////////////////

  /**
   * @brief Test method for appending matrix row.
   *
   * @pre Matrix `A` must have space allocated for appending the row.
   */
  int matrixAppendRow(hiopMatrixDense& A, hiopVector& vec, const int rank)
  {
    assert(A.n() == vec.get_size() &&
           "Did you pass in a vector with the same length as the number of columns of the matrix?");
    assert(getNumLocCols(&A) == vec.get_local_size() &&
           "Did you pass in a vector with the same length as the number of columns of the matrix?");
    const global_ordinal_type init_num_rows = A.m();
    const real_type A_val = one;
    const real_type vec_val = two;
    int fail = 0;

    A.setToConstant(A_val);
    vec.setToConstant(vec_val);
    A.appendRow(vec);

    // Ensure A's num rows is updated
    if(A.m() != init_num_rows + 1) fail++;

    // Ensure vec's values are copied over to A's last row
    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // Rows are always global in HiOp (for now)
      auto irow = static_cast<global_ordinal_type>(i);
      (void)j;  // j is unused
      const bool isLastRow = (irow == init_num_rows);
      return isLastRow ? vec_val : A_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  /**
   * Tests function that copies rows from source to destination starting from
   * `dst_start_idx` in the same order.
   *
   */
  int matrixCopyRowsFrom(hiopMatrixDense& dst, hiopMatrixDense& src, const int rank)
  {
    assert(dst.n() == src.n());
    assert(dst.m() > src.m());
    assert(getNumLocCols(&dst) == getNumLocCols(&src));
    assert(getNumLocRows(&dst) > getNumLocRows(&src));
    const real_type dst_val = one;
    const real_type src_val = two;
    const local_ordinal_type src_num_rows = getNumLocRows(&src);
    local_ordinal_type num_rows_to_copy = src_num_rows;
    const local_ordinal_type dst_start_idx = getNumLocRows(&dst) - src_num_rows;

    // Test copying continuous rows from matrix
    dst.setToConstant(dst_val);
    src.setToConstant(src_val);

    dst.copyRowsFrom(src, num_rows_to_copy, dst_start_idx);

    int fail = verifyAnswer(&dst, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      (void)j;  // j is unused
      const bool isRowCopiedOver = (i >= dst_start_idx && i < dst_start_idx + src_num_rows);
      return isRowCopiedOver ? src_val : dst_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  /**
   * Tests function that copies rows from source to destination in order
   * specified by index array `row_idxs`.
   *
   */
  int matrixCopyRowsFromSelect(hiopMatrixDense& dst, hiopMatrixDense& src, hiopVectorInt& rows_idxs, const int rank)
  {
    assert(dst.n() == src.n());
    assert(getNumLocCols(&dst) == getNumLocCols(&src));
    const real_type dst_val = one;
    const real_type src_val = two;
    const local_ordinal_type num_rows_to_copy = getNumLocRows(&dst);
    assert(num_rows_to_copy == rows_idxs.get_local_size());
    assert(num_rows_to_copy <= src.m());

    // Test copying continuous rows from matrix
    dst.setToConstant(dst_val);
    src.setToConstant(src_val);

    hiopVectorIntSeq rows_idxs_host(rows_idxs.get_local_size());
    index_type* rows_idxs_arr = rows_idxs_host.local_data();
    for(index_type i = 0; i < num_rows_to_copy; ++i) {
      rows_idxs_arr[i] = i;
    }
    rows_idxs_arr[0] = num_rows_to_copy - 1;
    rows_idxs_arr[num_rows_to_copy - 1] = 0;
    rows_idxs.copy_from_vectorseq(rows_idxs_host);

    setLocalRow(&src, num_rows_to_copy - 1, zero);

    dst.copyRowsFrom(src, rows_idxs.local_data(), num_rows_to_copy);

    int fail = verifyAnswer(&dst, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      (void)j;  // j is unused
      return i == 0 ? zero : src_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  int matrixCopyBlockFromMatrix(hiopMatrixDense& src, hiopMatrixDense& dst, const int rank = 0)
  {
    assert(src.n() < dst.n() && "Src mat must be smaller than dst mat");
    assert(src.m() < dst.m() && "Src mat must be smaller than dst mat");
    assert(getNumLocCols(&src) < getNumLocCols(&dst) && "Src mat must be smaller than dst mat");
    const real_type src_val = one;
    const real_type dst_val = two;

    // Copy matrix block into downmost and rightmost location
    // possible
    const local_ordinal_type src_num_rows = getNumLocRows(&src);
    const local_ordinal_type src_num_cols = getNumLocCols(&src);
    const local_ordinal_type dst_start_row = getNumLocRows(&dst) - src_num_rows;
    const local_ordinal_type dst_start_col = getNumLocCols(&dst) - src_num_cols;

    src.setToConstant(src_val);
    dst.setToConstant(dst_val);
    dst.copyBlockFromMatrix(dst_start_row, dst_start_col, src);

    const int fail = verifyAnswer(&dst, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      const bool isIdxCopiedFromSource =
          (i >= dst_start_row && i < dst_start_row + src_num_rows && j >= dst_start_col && j < dst_start_col + src_num_cols);
      return isIdxCopiedFromSource ? src_val : dst_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  int matrixCopyFromMatrixBlock(hiopMatrixDense& src, hiopMatrixDense& dst, const int rank = 0)
  {
    assert(src.n() > dst.n() && "Src mat must be larger than dst mat");
    assert(src.m() > dst.m() && "Src mat must be larger than dst mat");
    assert(getNumLocCols(&src) > getNumLocCols(&dst) && "Src mat must be larger than dst mat");
    const local_ordinal_type dst_m = getNumLocRows(&dst);
    const local_ordinal_type dst_n = getNumLocCols(&dst);
    const local_ordinal_type src_m = getNumLocRows(&src);
    const local_ordinal_type src_n = getNumLocCols(&src);
    const local_ordinal_type block_start_row = (src_m - getNumLocRows(&dst)) - 1;
    const local_ordinal_type block_start_col = (src_n - getNumLocCols(&dst)) - 1;

    const real_type src_val = one;
    const real_type dst_val = two;
    src.setToConstant(src_val);
    if(rank == 0) setLocalElement(&src, src_m - 1, src_n - 1, zero);
    dst.setToConstant(dst_val);

    dst.copyFromMatrixBlock(src, block_start_row, block_start_col);

    const int fail = verifyAnswer(&dst, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // This is the element set to zero in src
      // before being copied over
      if(i == dst_m && j == dst_n && rank == 0)
        return zero;
      else
        return src_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  int matrix_set_Hess_FR(hiopMatrixDense& src, hiopMatrixDense& dst, hiopVector& diag, const int rank = 0)
  {
    assert(src.n() == src.m() && "Src mat must be square mat");
    assert(src.n() == dst.n() && "Src mat must be equal to dst mat");
    assert(src.m() == dst.m() && "Src mat must be equal to dst mat");
    assert(src.m() == diag.get_size() && "Wrong vec size");

    const real_type src_val = one;
    const real_type diag_val = two;
    src.setToConstant(src_val);
    diag.setToConstant(diag_val);

    dst.set_Hess_FR(src, diag);

    const int fail = verifyAnswer(&dst, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // This is the element set to zero in src
      // before being copied over
      if(i == j && rank == 0)
        return src_val + diag_val;
      else
        return src_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
  }

  /**
   * shiftRows does not overwrite rows in the opposite direction
   * they are shifted. For example
   * @verbatim
   *    2 2 2                   2 2 2
   *    1 1 1                   1 1 1
   *    1 1 1  shiftRows(2) ->  2 2 2
   * @endverbatim
   * The uppermost row is not overwritten by the 1-row that would
   * wrap around and replace it.
   */
  int matrixShiftRows(hiopMatrixDense& A, const int rank)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    local_ordinal_type uniq_row_idx = 0;
    local_ordinal_type shift = M - 1;
    int fail = 0;

    const real_type A_val = one;
    const real_type uniq_row_val = two;
    A.setToConstant(A_val);

    // Set one row to a unique value
    setLocalRow(&A, uniq_row_idx, uniq_row_val);
    A.shiftRows(shift);

    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      (void)j;  // j is unused
      const bool isUniqueRow = (i == (uniq_row_idx + shift) || i == uniq_row_idx);
      return isUniqueRow ? uniq_row_val : A_val;
    });

    // Now check negative shift
    shift *= -1;
    uniq_row_idx = M - 1;
    A.setToConstant(A_val);
    setLocalRow(&A, uniq_row_idx, uniq_row_val);
    A.shiftRows(shift);

    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      (void)j;  // j is unused
      const bool isUniqueRow = (i == (uniq_row_idx + shift) || i == uniq_row_idx);
      return isUniqueRow ? uniq_row_val : A_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixReplaceRow(hiopMatrixDense& A, hiopVector& vec, const int rank)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    assert(getNumLocCols(&A) == vec.get_local_size() && "Did you pass a vector and matrix of compatible lengths?");
    assert(A.n() == vec.get_size() && "Did you pass a vector and matrix of compatible lengths?");

    const local_ordinal_type row_idx = M - 1;
    const local_ordinal_type col_idx = 1;
    const real_type A_val = one;
    const real_type vec_val = two;
    A.setToConstant(A_val);
    vec.setToConstant(vec_val);
    setLocalElement(&vec, col_idx, zero);

    A.replaceRow(row_idx, vec);
    const int fail = verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      // Was the row replaced?
      if(i == row_idx) {
        // Was the value at col_idx set to zero?
        if(j == col_idx)
          return zero;
        else
          return vec_val;
      }
      // The matrix should be otherwise unchanged.
      else {
        return A_val;
      }
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixGetRow(hiopMatrixDense& A, hiopVector& vec, const int rank)
  {
    const local_ordinal_type N = getNumLocCols(&A);
    const local_ordinal_type M = getNumLocRows(&A);
    assert(N == vec.get_local_size() && "Did you pass a vector and matrix of compatible lengths?");
    assert(A.n() == vec.get_size() && "Did you pass a vector and matrix of compatible lengths?");

    // Set one value in the matrix row to be retrieved to be a unique value
    const local_ordinal_type row_idx = M - 1;
    const local_ordinal_type col_idx = N - 1;
    const real_type A_val = one;
    const real_type vec_val = two;
    A.setToConstant(A_val);
    if(rank == 0) setLocalElement(&A, row_idx, col_idx, zero);
    vec.setToConstant(vec_val);
    A.getRow(row_idx, vec);

    const int fail = verifyAnswer(&vec, [=](local_ordinal_type i) -> real_type {
      if(rank == 0 && i == col_idx)
        return zero;
      else
        return A_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrix_symmetrize(hiop::hiopMatrixDense& A, const int rank = 0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);
    int fail = 0;
    const real_type upper_val = one;
    const real_type diag_val = zero;

    assert(A.m() == A.n());
    A.setToZero();

    // Set the upper triangular part to one
    for(int i = 0; i < N; i++) {
      for(int j = i + 1; j < M; j++) {
        if(rank == 0) {
          setLocalElement(&A, i, j, upper_val);
        }
      }
    }

    // copy the upper triangular part to the lower triangylar part
    A.symmetrize();

    fail += verifyAnswer(&A, [=](local_ordinal_type i, local_ordinal_type j) -> real_type {
      bool is_diagonal = (j == i);
      return is_diagonal ? diag_val : upper_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

#ifdef HIOP_DEEPCHECKS
  int matrixAssertSymmetry(hiop::hiopMatrixDense& A, const int rank = 0)
  {
    const local_ordinal_type M = getNumLocRows(&A);
    const local_ordinal_type N = getNumLocCols(&A);
    int fail = 0;

    assert(A.m() == A.n());
    A.setToConstant(one);
    fail += !A.assertSymmetry(eps);

    // Set first row and column to zero globally
    for(int i = 0; i < N; i++) {
      setLocalElement(&A, 0, i, zero);
    }
    if(rank == 0) {
      for(int i = 0; i < M; i++) {
        setLocalElement(&A, i, 0, zero);
      }
    }
    fail += !A.assertSymmetry(eps);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixOverwriteUpperTriangleWithLower(hiop::hiopMatrixDense& A, const int rank = 0)
  {
    const local_ordinal_type M = getNumLocRows(&A);

    for(int i = 0; i < M; i++) {
      setLocalRow(&A, i, i);
    }

    A.overwriteUpperTriangleWithLower();

    const int fail = verifyAnswer(&A, [=](int i, int j) { return j <= i ? i : j; });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixOverwriteLowerTriangleWithUpper(hiop::hiopMatrixDense& A, const int rank = 0)
  {
    const local_ordinal_type M = getNumLocRows(&A);

    for(int i = 0; i < M; i++) {
      setLocalRow(&A, i, i);
    }

    A.overwriteLowerTriangleWithUpper();

    const int fail = verifyAnswer(&A, [=](int i, int j) { return j < i ? j : i; });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }
#endif

  int matrixNumRows(hiop::hiopMatrixDense& A, global_ordinal_type M, const int rank)
  {
    const bool fail = A.m() == M ? 0 : 1;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

  int matrixNumCols(hiop::hiopMatrixDense& A, global_ordinal_type N, const int rank)
  {
    const bool fail = A.n() == N ? 0 : 1;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
  }

protected:
  // Matrix helper functions
  virtual local_ordinal_type getNumLocRows(const hiop::hiopMatrixDense* a) = 0;
  virtual local_ordinal_type getNumLocCols(const hiop::hiopMatrixDense* a) = 0;
  virtual void setLocalElement(hiop::hiopMatrixDense* a, local_ordinal_type i, local_ordinal_type j, real_type val) = 0;
  virtual void setLocalRow(hiop::hiopMatrixDense* A, const local_ordinal_type row, const real_type val) = 0;
  virtual real_type getLocalElement(const hiop::hiopMatrixDense* a, local_ordinal_type i, local_ordinal_type j) = 0;
  virtual const real_type* getLocalDataConst(hiop::hiopMatrixDense* a) = 0;
  virtual real_type* getLocalData(hiop::hiopMatrixDense* a) = 0;
  virtual int verifyAnswer(hiop::hiopMatrixDense* A, real_type answer) = 0;
  virtual int verifyAnswer(hiop::hiopMatrixDense* A,
                           std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
  virtual bool reduceReturn(int failures, hiop::hiopMatrixDense* A) = 0;

  // Vector helper function
  virtual void setLocalElement(hiop::hiopVector* _x, const local_ordinal_type i, const real_type val) = 0;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) = 0;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
  virtual int verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect) = 0;
};

}  // namespace tests
}  // namespace hiop
