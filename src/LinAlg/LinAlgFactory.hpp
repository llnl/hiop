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
#pragma once

#include <string>
#include <iostream>
#include <ExecSpace.hpp>
#include <hiopMPI.hpp>
#include <hiopVector.hpp>
#include <hiopMatrixDense.hpp>
#include <hiopMatrixSparse.hpp>
#include <hiopMatrixSparseCSR.hpp>
#include <hiopVectorInt.hpp>

namespace hiop
{

/**
 * @brief Factory for HiOp's linear algebra objects
 *
 */
class LinearAlgebraFactory
{
public:
  LinearAlgebraFactory() = delete;
  ~LinearAlgebraFactory() = delete;

  /**
   * @brief Static method to create vector
   */
  static hiopVector* create_vector(const ExecSpaceInfo& hi,
                                   const size_type& glob_n,
                                   index_type* col_part = nullptr,
                                   MPI_Comm comm = MPI_COMM_SELF);
  /**
   * @brief Static method to create local int vector.
   */
  static hiopVectorInt* create_vector_int(const ExecSpaceInfo& hi, size_type size);

  /**
   * @brief Static method to create a dense matrix.
   *
   */
  static hiopMatrixDense* create_matrix_dense(const ExecSpaceInfo& hi,
                                              const size_type& m,
                                              const size_type& glob_n,
                                              index_type* col_part = NULL,
                                              MPI_Comm comm = MPI_COMM_SELF,
                                              const size_type& m_max_alloc = -1);
  /**
   * @brief Static method to create the default, triplet sparse matrix
   */
  static hiopMatrixSparse* create_matrix_sparse(const ExecSpaceInfo& hi, size_type rows, size_type cols, size_type nnz);

  /**
   * @brief Static method to create an empty CSR sparse matrix of the type that supports the
   * memory space passed as argument.
   */
  static hiopMatrixSparseCSR* create_matrix_sparse_csr(const std::string& mem_space);

  /**
   * @brief Static method to create a CSR sparse matrix of the type that supports the
   * memory space passed as argument.
   */
  static hiopMatrixSparseCSR* create_matrix_sparse_csr(const std::string& mem_space,
                                                       size_type rows,
                                                       size_type cols,
                                                       size_type nnz);

  /**
   * @brief Static method to create a symmetric sparse matrix
   */
  static hiopMatrixSparse* create_matrix_sym_sparse(const ExecSpaceInfo& hi, size_type size, size_type nnz);

  /**
   * @brief Static method to create a raw C array
   */
  static double* create_raw_array(const std::string& mem_space, size_type n);

  /**
   * @brief Static method to delete a raw C array
   */
  static void delete_raw_array(const std::string& mem_space, double* a);
};

}  // namespace hiop
