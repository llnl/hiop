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
 * @file testMatrixSparse.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#include <iostream>
#include <cassert>
#include <cstring>

#include <hiopOptions.hpp>
#include <LinAlgFactory.hpp>
#include <hiopVector.hpp>
#include <hiopVectorIntSeq.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include "LinAlg/matrixTestsSparseTriplet.hpp"

#ifdef HIOP_USE_RAJA
#include <hiopVectorIntRaja.hpp>
#include <hiopVectorRaja.hpp>
#include <hiopMatrixDenseRaja.hpp>
#include "LinAlg/matrixTestsRajaSparseTriplet.hpp"
#endif

using namespace hiop::tests;

int main(int argc, char** argv)
{
  if(argc > 1) {
    std::cout << "Executable " << argv[0] << " doesn't take any input.";
  }
  hiop::hiopOptionsNLP options;

  local_ordinal_type M_local = 5;
  local_ordinal_type N_local = 10 * M_local;

  // Sparse matrix is not distributed
  global_ordinal_type M_global = M_local;
  global_ordinal_type N_global = N_local;

  int fail = 0;

  // Test sparse matrix
  {
    const std::string mem_space = "DEFAULT";
    std::cout << "\nTesting hiopMatrixSparseTriplet\n";
    hiop::tests::MatrixTestsSparseTriplet test;
    test.set_mem_space(mem_space);

    // Establishing sparsity pattern and initializing Matrix
    local_ordinal_type entries_per_row = 5;
    local_ordinal_type nnz = M_local * entries_per_row;

    hiop::hiopMatrixSparse* mxn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_local, N_local, nnz);
    test.initializeMatrix(mxn_sparse, entries_per_row);

    hiop::hiopMatrixSparse* mxn_empty = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_local, N_global, 0);
    test.initializeMatrix(mxn_empty, 0);

    hiop::hiopMatrixSparse* nullxn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, 0, N_global, 0);
    test.initializeMatrix(nullxn_sparse, 0);

    hiop::hiopVectorPar vec_m(M_global);
    hiop::hiopVectorPar vec_n(N_global);

    /// @see LinAlg/matrixTestsSparseTriplet.hpp for reasons why some tests are implemented/not implemented
    fail += test.matrixNumRows(*mxn_sparse, M_global);
    fail += test.matrixNumCols(*mxn_sparse, N_global);
    fail += test.matrixSetToZero(*mxn_sparse);
    fail += test.matrixSetToConstant(*mxn_sparse);
    fail += test.matrixTimesVec(*mxn_sparse, vec_m, vec_n);
    fail += test.matrixTransTimesVec(*mxn_sparse, vec_m, vec_n);
    fail += test.matrixMaxAbsValue(*mxn_sparse);
    fail += test.matrix_row_max_abs_value(*mxn_sparse, vec_m);
    fail += test.matrix_scale_row(*mxn_sparse, vec_m);
    fail += test.matrixIsFinite(*mxn_sparse);

    // Need a dense matrix to store the output of the following tests
    global_ordinal_type W_delta = M_global * 10;
    hiop::hiopMatrixDenseRowMajor W_dense(N_global + W_delta, N_global + W_delta);

    // local_ordinal_type test_offset = 10;
    local_ordinal_type test_offset = 4;
    fail += test.matrixAddMDinvMtransToDiagBlockOfSymDeMatUTri(*mxn_sparse, vec_n, W_dense, test_offset);

    // Need a dense matrix that is big enough for the sparse matrix to map inside the upper triangular part of it
    // hiop::hiopMatrixDenseRowMajor n2xn2_dense(2 * N_global, 2 * N_global);
    // fail += test.addToSymDenseMatrixUpperTriangle(W_dense, *mxn_sparse);
    fail += test.transAddToSymDenseMatrixUpperTriangle(W_dense, *mxn_sparse);

    // Initialise another sparse Matrix
    local_ordinal_type M2 = M_global * 2;
    local_ordinal_type nnz2 = M2 * (entries_per_row);

    hiop::hiopMatrixSparse* m2xn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M2, N_global, nnz2);
    test.initializeMatrix(m2xn_sparse, entries_per_row);

    hiop::hiopMatrixDenseRowMajor mxm2_dense(M_global, M2);

    // Set offsets where to insert sparse matrix
    local_ordinal_type i_offset = 1;
    local_ordinal_type j_offset = M2 + 1;

    fail += test.matrixTimesMatTrans(*mxn_sparse, *m2xn_sparse, mxm2_dense);
    fail += test.matrixAddMDinvNtransToSymDeMatUTri(*mxn_sparse, *m2xn_sparse, vec_n, W_dense, i_offset, j_offset);

    // copy sparse matrix to a dense matrix
    hiop::hiopMatrixDenseRowMajor mxn_dense(M_global, N_global);
    fail += test.matrix_copy_to(mxn_dense, *mxn_sparse);

    // extend a sparse matrix [C;D] to [C -I I 0 0; D 0 0 -I I]
    hiop::hiopMatrixDenseRowMajor m3xn3_dense(M_global + M2, N_global + 2 * (M_global + M2));
    local_ordinal_type nnz3 = nnz + nnz2 + 2 * M_global + 2 * M2;
    hiop::hiopMatrixSparse* m3xn3_sparse =
        hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_global + M2, N_global + 2 * (M_global + M2), nnz3);
    fail += test.matrix_set_Jac_FR(m3xn3_dense, *m3xn3_sparse, *mxn_sparse, *m2xn_sparse);

    // functions used to build large sparse matrix from small pieces
    fail += test.matrix_copy_subdiagonal_from(m3xn3_dense, *m3xn3_sparse, vec_m);
    fail += test.matrix_set_subdiagonal_to(m3xn3_dense, *m3xn3_sparse);

    hiop::hiopVectorIntSeq select(M_local);
    fail += test.matrix_copy_rows_from(*mxn_sparse, *m2xn_sparse, select);

    // copy the 1st row of mxn_sparse to the last row in m2xn_sparse
    // replace the nonzero index from "nnz-entries_per_row"
    fail += test.copy_rows_block_from(*mxn_sparse,
                                      *m2xn_sparse,
                                      0,
                                      1,
                                      M_global - 1,
                                      mxn_sparse->numberOfNonzeros() - entries_per_row);
    fail += test.copy_rows_block_from(*mxn_empty, *m2xn_sparse, 0, 1, M_global - 1, mxn_sparse->numberOfNonzeros());
    fail += test.copy_rows_block_from(*nullxn_sparse, *m2xn_sparse, 0, 0, M_global - 1, mxn_sparse->numberOfNonzeros());
    fail += test.copy_rows_block_from(*nullxn_sparse, *nullxn_sparse, 0, 0, 0, nullxn_sparse->numberOfNonzeros());

    // create a bigger matrix, to test copy_submatrix_from and opy_submatrix_from_trans
    hiop::hiopMatrixDenseRowMajor m4xn4_dense(2 * M_global + N_global, 2 * M_global + N_global);
    local_ordinal_type nnz4 = entries_per_row * (2 * M_global + N_global);
    assert(nnz4 < (2 * M_global + N_global) * (2 * M_global + N_global));
    hiop::hiopMatrixSparse* m4xn4_sparse =
        hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, 2 * M_global + N_global, 2 * M_global + N_global, nnz4);
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_copy_submatrix_from(m4xn4_dense, *m4xn4_sparse, *mxn_sparse, M_global, 2 * M_global, nnz4 - nnz);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail +=
        test.matrix_copy_submatrix_from_trans(m4xn4_dense, *m4xn4_sparse, *mxn_sparse, M_global, 2 * (M_global), nnz4 - nnz);

    hiop::hiopVectorPar v_patern(N_global);
    local_ordinal_type nnz_to_replace = M_global;
    assert(N_global > nnz_to_replace);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail +=
        test.matrix_copy_diag_matrix_to_subblock(m4xn4_dense, *m4xn4_sparse, M_global, 2 * M_global, nnz4 - 2 * nnz, nnz);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_copy_diag_matrix_to_subblock_w_pattern(m4xn4_dense,
                                                               *m4xn4_sparse,
                                                               vec_n,
                                                               v_patern,
                                                               M_global,
                                                               2 * M_global,
                                                               nnz4 - 2 * nnz,
                                                               nnz_to_replace);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_set_submatrix_to_constant_diag_w_colpattern(m4xn4_dense,
                                                                    *m4xn4_sparse,
                                                                    v_patern,
                                                                    M_global,
                                                                    2 * M_global,
                                                                    nnz4 - 2 * nnz,
                                                                    nnz_to_replace);
    fail += test.matrix_set_submatrix_to_constant_diag_w_rowpattern(m4xn4_dense,
                                                                    *m4xn4_sparse,
                                                                    v_patern,
                                                                    M_global,
                                                                    2 * M_global,
                                                                    nnz4 - 2 * nnz,
                                                                    nnz_to_replace);

    // Remove testing objects
    delete mxn_sparse;
    delete mxn_empty;
    delete nullxn_sparse;
    delete m2xn_sparse;
    delete m3xn3_sparse;
    delete m4xn4_sparse;
  }

#ifdef HIOP_USE_RAJA
  // Test RAJA sparse matrix
  {
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
    const std::string mem_space = "HOST";
#else
    const std::string mem_space = "DEVICE";
#endif
    std::cout << "\nTesting hiopMatrixRajaSparseTriplet mem_space=" << mem_space << "\n";

    hiop::tests::MatrixTestsRajaSparseTriplet test;
    test.set_mem_space(mem_space);

    // Establishing sparsity pattern and initializing Matrix
    local_ordinal_type entries_per_row = 5;
    local_ordinal_type nnz = M_local * entries_per_row;

    hiop::hiopMatrixSparse* mxn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_local, N_local, nnz);

    test.initializeMatrix(mxn_sparse, entries_per_row);

    hiop::hiopMatrixSparse* mxn_empty = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_local, N_global, 0);
    test.initializeMatrix(mxn_empty, 0);

    hiop::hiopMatrixSparse* nullxn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, 0, N_global, 0);
    test.initializeMatrix(nullxn_sparse, 0);

    // hiop::hiopVectorRajaPar vec_m(M_global, mem_space);
    hiop::hiopVector* vec_m = hiop::LinearAlgebraFactory::create_vector(mem_space, M_global);
    hiop::hiopVector* vec_m_2 = hiop::LinearAlgebraFactory::create_vector(mem_space, M_global);
    hiop::hiopVector* vec_n = hiop::LinearAlgebraFactory::create_vector(mem_space, N_global);

    /// @see LinAlg/matrixTestsSparseTriplet.hpp for reasons why some tests are implemented/not implemented
    fail += test.matrixNumRows(*mxn_sparse, M_global);
    fail += test.matrixNumCols(*mxn_sparse, N_global);
    fail += test.matrixSetToZero(*mxn_sparse);
    fail += test.matrixSetToConstant(*mxn_sparse);
    fail += test.matrixMaxAbsValue(*mxn_sparse);
    fail += test.matrix_row_max_abs_value(*mxn_sparse, *vec_m);
    fail += test.matrix_scale_row(*mxn_sparse, *vec_m);
    fail += test.matrixIsFinite(*mxn_sparse);
    fail += test.matrixTimesVec(*mxn_sparse, *vec_m, *vec_n);
    fail += test.matrixTransTimesVec(*mxn_sparse, *vec_m, *vec_n);

    // Need a dense matrix to store the output of the following tests
    global_ordinal_type W_delta = M_global * 10;

    hiop::hiopMatrixDense* W_dense =
        hiop::LinearAlgebraFactory::create_matrix_dense(mem_space, N_global + W_delta, N_global + W_delta);

    // local_ordinal_type test_offset = 10;
    local_ordinal_type test_offset = 4;
    fail += test.matrixAddMDinvMtransToDiagBlockOfSymDeMatUTri(*mxn_sparse, *vec_n, *W_dense, test_offset);

    // testing adding sparse matrix to the upper triangular area of a symmetric dense matrix
    // fail += test.addToSymDenseMatrixUpperTriangle(W_dense, *mxn_sparse);
    fail += test.transAddToSymDenseMatrixUpperTriangle(*W_dense, *mxn_sparse);

    // Initialise another sparse Matrix
    local_ordinal_type M2 = M_global * 2;
    local_ordinal_type nnz2 = M2 * (entries_per_row);

    /// @todo: use linear algebra factory for this
    hiop::hiopMatrixSparse* m2xn_sparse = hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M2, N_global, nnz2);
    test.initializeMatrix(m2xn_sparse, entries_per_row);

    hiop::hiopMatrixDense* mxm2_dense = hiop::LinearAlgebraFactory::create_matrix_dense(mem_space, M_global, M2);

    // Set offsets where to insert sparse matrix
    local_ordinal_type i_offset = 1;
    local_ordinal_type j_offset = M2 + 1;

    fail += test.matrixTimesMatTrans(*mxn_sparse, *m2xn_sparse, *mxm2_dense);
    fail += test.matrixAddMDinvNtransToSymDeMatUTri(*mxn_sparse, *m2xn_sparse, *vec_n, *W_dense, i_offset, j_offset);

    // copy sparse matrix to dense matrix
    hiop::hiopMatrixDense* mxn_dense = hiop::LinearAlgebraFactory::create_matrix_dense(mem_space, M_global, N_global);

    fail += test.matrix_copy_to(*mxn_dense, *mxn_sparse);

    // extend a sparse matrix [C;D] to [C -I I 0 0; D 0 0 -I I]
    hiop::hiopMatrixDense* m3xn3_dense =
        hiop::LinearAlgebraFactory::create_matrix_dense(mem_space, M_global + M2, N_global + 2 * (M_global + M2));

    local_ordinal_type nnz3 = nnz + nnz2 + 2 * M_global + 2 * M2;
    hiop::hiopMatrixSparse* m3xn3_sparse =
        hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_global + M2, N_global + 2 * (M_global + M2), nnz3);
    fail += test.matrix_set_Jac_FR(*m3xn3_dense, *m3xn3_sparse, *mxn_sparse, *m2xn_sparse);

    // functions used to build large sparse matrix from small pieces
    fail += test.matrix_copy_subdiagonal_from(*m3xn3_dense, *m3xn3_sparse, *vec_m);
    fail += test.matrix_set_subdiagonal_to(*m3xn3_dense, *m3xn3_sparse);

    hiop::hiopVectorInt* select = hiop::LinearAlgebraFactory::create_vector_int(mem_space, M_local);
    hiop::hiopMatrixSparse* mxn_sparse_2 =
        hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, M_local, N_local, nnz);
    fail += test.matrix_copy_rows_from(*mxn_sparse_2, *m2xn_sparse, *select);
    delete select;

    // copy the 1st row of mxn_sparse to the last row in m2xn_sparse
    // replace the nonzero index from "nnz-entries_per_row"
    fail += test.copy_rows_block_from(*mxn_sparse,
                                      *m2xn_sparse,
                                      0,
                                      1,
                                      M_global - 1,
                                      mxn_sparse->numberOfNonzeros() - entries_per_row);
    fail += test.copy_rows_block_from(*mxn_empty, *m2xn_sparse, 0, 1, M_global - 1, mxn_sparse->numberOfNonzeros());
    fail += test.copy_rows_block_from(*nullxn_sparse, *m2xn_sparse, 0, 0, M_global - 1, mxn_sparse->numberOfNonzeros());
    fail += test.copy_rows_block_from(*nullxn_sparse, *nullxn_sparse, 0, 0, 0, nullxn_sparse->numberOfNonzeros());

    // create a bigger matrix, to test copy_submatrix_from and opy_submatrix_from_trans
    // hiop::hiopMatrixRajaDense m4xn4_dense(2*M_global+N_global, 2*M_global+N_global,mem_space);
    hiop::hiopMatrixDense* m4xn4_dense =
        hiop::LinearAlgebraFactory::create_matrix_dense(mem_space, 2 * M_global + N_global, 2 * M_global + N_global);

    local_ordinal_type nnz4 = entries_per_row * (2 * M_global + N_global);
    assert(nnz4 < (2 * M_global + N_global) * (2 * M_global + N_global));
    hiop::hiopMatrixSparse* m4xn4_sparse =
        hiop::LinearAlgebraFactory::create_matrix_sparse(mem_space, 2 * M_global + N_global, 2 * M_global + N_global, nnz4);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_copy_submatrix_from(*m4xn4_dense, *m4xn4_sparse, *mxn_sparse, M_global, 2 * M_global, nnz4 - nnz);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_copy_submatrix_from_trans(*m4xn4_dense,
                                                  *m4xn4_sparse,
                                                  *mxn_sparse,
                                                  M_global,
                                                  2 * (M_global),
                                                  nnz4 - nnz);

    hiop::hiopVector* v_pattern = hiop::LinearAlgebraFactory::create_vector(mem_space, N_global);
    local_ordinal_type nnz_to_replace = M_global;
    assert(N_global > nnz_to_replace);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail +=
        test.matrix_copy_diag_matrix_to_subblock(*m4xn4_dense, *m4xn4_sparse, M_global, 2 * M_global, nnz4 - 2 * nnz, nnz);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_copy_diag_matrix_to_subblock_w_pattern(*m4xn4_dense,
                                                               *m4xn4_sparse,
                                                               *vec_n,
                                                               *v_pattern,
                                                               M_global,
                                                               2 * M_global,
                                                               nnz4 - 2 * nnz,
                                                               nnz_to_replace);

    // reset the sparsity, since previous function may change the sparsity
    test.initializeMatrix(m4xn4_sparse, entries_per_row);
    fail += test.matrix_set_submatrix_to_constant_diag_w_colpattern(*m4xn4_dense,
                                                                    *m4xn4_sparse,
                                                                    *v_pattern,
                                                                    M_global,
                                                                    2 * M_global,
                                                                    nnz4 - 2 * nnz,
                                                                    nnz_to_replace);
    fail += test.matrix_set_submatrix_to_constant_diag_w_rowpattern(*m4xn4_dense,
                                                                    *m4xn4_sparse,
                                                                    *v_pattern,
                                                                    M_global,
                                                                    2 * M_global,
                                                                    nnz4 - 2 * nnz,
                                                                    nnz_to_replace);

    // Remove testing objects
    delete mxn_sparse;
    delete mxn_empty;
    delete nullxn_sparse;
    delete mxn_sparse_2;
    delete m2xn_sparse;
    delete m3xn3_sparse;
    delete m4xn4_sparse;

    delete W_dense;
    delete mxm2_dense;
    delete mxn_dense;
    delete m3xn3_dense;
    delete m4xn4_dense;

    delete vec_m;
    delete vec_m_2;
    delete vec_n;
    delete v_pattern;
  }
#endif

  if(fail) {
    std::cout << "\n" << fail << " sparse matrix tests failed!\n\n";
  } else {
    std::cout << "\nAll sparse matrix tests passed!\n\n";
  }

  return fail;
}
