// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
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
 * @file MatrixSparseCSRCudaKernels.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */
#ifndef HIOP_SPARSE_MATRIX_CSRCUDA_KER
#define HIOP_SPARSE_MATRIX_CSRCUDA_KER

#include "ExecSpace.hpp"

namespace hiop
{
namespace cuda
{

/**
 * Set diagonal of the CSR matrix to `val` by performing a binary search on the column indexes
 * for each row. Assumes pointers are on the device and parallelizes over rows.
 *
 * @pre CSR matrix must be square.
 * @pre Diagonal entries must appear explicitly among the nonzeros.
 * @pre Column indexes must be sorted for any given row.
 */
void csr_set_diag_kernel(int n, int nnz, int* irowptr, int* jcoldind, double* values, double val, int block_size);

/**
 * Add the constant `val` to the diagonal of the CSR matrix. Performs a binary search on the column indexes
 * for each row. Assumes pointers are on the device and parallelizes over rows.
 *
 * @pre CSR matrix must be square.
 * @pre Diagonal entries must appear explicitly among the nonzeros.
 * @pre Column indexes must be sorted for any given row.
 */
void csr_add_diag_kernel(int n, int nnz, int* irowptr, int* jcoldind, double* values, double Dval, int block_size);

/**
 * Add entries of the array `values` to the diagonal of the CSR matrix. Performs a binary search on the column indexes
 * for each row. Assumes pointers are on the device and parallelizes over rows.
 *
 * @pre CSR matrix must be square.
 * @pre Diagonal entries must appear explicitly among the nonzeros.
 * @pre Column indexes must be sorted for any given row.
 * @pre
 */
void csr_add_diag_kernel(int n,
                         int nnz,
                         int* irowptr,
                         int* jcoldind,
                         double* values,
                         double alpha,
                         const double* Dvalues,
                         int block_size);

/**
 * Copies the diagonal of a CSR matrix into the array `diag_out`. All pointers are on the device. The
 * output array should be allocated to hold `n` doubles.
 *
 * @pre CSR matrix must be square.
 * @pre Column indexes must be sorted for any given row.
 */
void csr_get_diag_kernel(int n,
                         int nnz,
                         const int* irowptr,
                         const int* jcoldind,
                         const double* values,
                         double* diag_out,
                         int block_size);

/**
 * Populates the row pointers and column indexes array to hold a CSR diagonal matrix of size `n`.
 */
void csr_form_diag_symbolic_kernel(int n, int* irowptr, int* jcolind, int block_size);

/**
 * Scales rows of the sparse CSR matrix with the diagonal matrix given by array `D`
 *
 * @pre All pointers should be on the device.
 * @pre Column indexes must be sorted for any given row.
 */
void csr_scalerows_kernel(int nrows,
                          int ncols,
                          int nnz,
                          int* irowptr,
                          int* jcoldind,
                          double* values,
                          const double* D,
                          int block_size);
}  // end of namespace cuda
}  // end of namespace hiop

#endif
