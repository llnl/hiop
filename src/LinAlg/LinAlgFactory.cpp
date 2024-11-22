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
 * @file LinAlgFactory.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */
#include <algorithm>
#include <iostream>

#include <hiop_defs.hpp>

#include <ExecSpace.hpp>

#ifdef HIOP_USE_RAJA
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <hiopVectorIntRaja.hpp>
#include <hiopVectorRaja.hpp>
#include <hiopMatrixDenseRaja.hpp>
#include <hiopMatrixRajaSparseTriplet.hpp>
#endif  // HIOP_USE_RAJA
#ifdef HIOP_USE_CUDA
#include <hiopVectorCuda.hpp>
#include <hiopVectorIntCuda.hpp>
#endif
#ifdef HIOP_USE_HIP
#include <hiopVectorHip.hpp>
#include <hiopVectorIntHip.hpp>
#endif

#include <hiopMatrixSparseCsrCuda.hpp>

#include <hiopVectorIntSeq.hpp>
#include <hiopVectorPar.hpp>
#include <hiopMatrixDenseRowMajor.hpp>
#include <hiopMatrixSparseTriplet.hpp>
#include <hiopMatrixSparseCSRSeq.hpp>
#include "LinAlgFactory.hpp"

#include "hiopCppStdUtils.hpp"

using namespace hiop;

/**
 * @brief Method to create vector.
 *
 * Creates legacy HiOp vector by default, RAJA vector when memory space
 * is specified.
 */
hiopVector* LinearAlgebraFactory::create_vector(const ExecSpaceInfo& hi,  // const std::string& mem_space,
                                                const size_type& glob_n,
                                                index_type* col_part,
                                                MPI_Comm comm)
{
  const std::string mem_space_upper = toupper(hi.mem_space_);
  if(mem_space_upper == "DEFAULT") {
    return new hiopVectorPar(glob_n, col_part, comm);
  } else {
    if(hi.exec_backend_ == "RAJA") {
#ifdef HIOP_USE_RAJA
      if(hi.mem_backend_ == "UMPIRE") {
#ifdef HIOP_USE_CUDA
        return new hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>(glob_n,
                                                                                          mem_space_upper,
                                                                                          col_part,
                                                                                          comm);
#endif
#ifdef HIOP_USE_HIP
        return new hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>(glob_n,
                                                                                         mem_space_upper,
                                                                                         col_part,
                                                                                         comm);
#endif
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        return new hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>(glob_n,
                                                                                         mem_space_upper,
                                                                                         col_part,
                                                                                         comm);
#endif

      } else {
        // RAJA exec policy with non-Umpire memory backend
        // work in progress
        assert(false && "work in progress");
        if(hi.mem_backend_ == "cuda") {
#ifdef HIOP_USE_CUDA
          assert(mem_space_upper == "DEVICE");
          return new hiop::hiopVectorRaja<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(glob_n,
                                                                                          mem_space_upper,
                                                                                          col_part,
                                                                                          comm);
#endif
        }
        if(hi.mem_backend_ == "hip") {
#ifdef HIOP_USE_HIP
          assert(mem_space_upper == "DEVICE");
          return new hiop::hiopVectorRaja<hiop::MemBackendHip, hiop::ExecPolicyRajaHip>(glob_n,
                                                                                        mem_space_upper,
                                                                                        col_part,
                                                                                        comm);
#endif
        } else {
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
          assert(hi.mem_backend_ == "stdcpp" || hi.mem_backend_ == "auto");
          return new hiop::hiopVectorRaja<hiop::MemBackendCpp, hiop::ExecPolicyRajaOmp>(glob_n,
                                                                                        mem_space_upper,
                                                                                        col_part,
                                                                                        comm);
#endif
        }
        return nullptr;
      }
#else         // non RAJA
      assert(false &&
             "requested execution space not available because Hiop was not"
             "built with RAJA support");
#endif        // #ifdef HIOP_USE_RAJA
    } else {  // else for if(hi.exec_backend_ == "RAJA")
      if(mem_space_upper == "CUDA") {
#ifdef HIOP_USE_CUDA
        return new hiop::hiopVectorCuda(glob_n, col_part, comm);
#else   // ifdef HIOP_USE_CUDA
        assert(false &&
               "requested memory space not available because Hiop was not"
               "built with CUDA support");
#endif  // ifdef HIOP_USE_CUDA
      } else {
        if(mem_space_upper == "HIP") {
#ifdef HIOP_USE_HIP
          return new hiop::hiopVectorHip(glob_n, col_part, comm);
#else   // ifdef HIOP_USE_HIP
          assert(false &&
                 "requested memory space not available because HiOp was not"
                 "built with HIP support");
#endif  // ifdef HIOP_USE_HIP
        } else {
          assert(false && "to be implemented");
        }
      }
    }
  }
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Method to create local int vector.
 *
 * Creates int vector with operator new by default, RAJA vector when memory space
 * is specified.
 */
hiopVectorInt* LinearAlgebraFactory::create_vector_int(const ExecSpaceInfo& hi, size_type n)
{
  const std::string ms = toupper(hi.mem_space_);
  if(ms == "DEFAULT") {
    return new hiopVectorIntSeq(n);
  } else {
    if(hi.exec_backend_ == "RAJA") {
#ifdef HIOP_USE_RAJA
      if(hi.mem_backend_ == "UMPIRE") {
#ifdef HIOP_USE_CUDA
        return new hiop::hiopVectorIntRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>(n, ms);
#endif
#ifdef HIOP_USE_HIP
        return new hiop::hiopVectorIntRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>(n, ms);
#endif
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        return new hiop::hiopVectorIntRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>(n, ms);
#endif

      } else {
        // RAJA exec policy with non-Umpire memory backend
        // work in progress
        assert(false && "work in progress");
        if(hi.mem_backend_ == "cuda") {
#ifdef HIOP_USE_CUDA
          assert(ms == "DEVICE");
          return new hiop::hiopVectorIntRaja<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(n, ms);
#endif
        }
        if(hi.mem_backend_ == "hip") {
#ifdef HIOP_USE_HIP
          assert(ms == "DEVICE");
          return new hiop::hiopVectorIntRaja<hiop::MemBackendHip, hiop::ExecPolicyRajaHip>(n, ms);
#endif
        } else {
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
          assert(hi.mem_backend_ == "stdcpp" || hi.mem_backend_ == "auto");
          return new hiop::hiopVectorIntRaja<hiop::MemBackendCpp, hiop::ExecPolicyRajaOmp>(n, ms);
#endif
        }
        return nullptr;
      }
#else  // non RAJA
      assert(false &&
             "requested execution space not available because Hiop was not"
             "built with RAJA support");
#endif
    } else {  // else for if(hi.exec_backend_ == "RAJA")
      if(ms == "CUDA") {
#ifdef HIOP_USE_CUDA
        return new hiop::hiopVectorIntCuda(n, ms);
#else   // ifdef HIOP_USE_CUDA
        assert(false &&
               "requested memory space not available because Hiop was not"
               "built with CUDA support");
#endif  // ifdef HIOP_USE_CUDA
      } else {
        if(ms == "HIP") {
#ifdef HIOP_USE_HIP
          return new hiop::hiopVectorIntHip(n, ms);
#else   // ifdef HIOP_USE_HIP
          assert(false &&
                 "requested memory space not available because HiOp was not"
                 "built with HIP support");
#endif  // ifdef HIOP_USE_CUDA
        } else {
          assert(false && "to be implemented");
        }
      }
    }
  }
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Method to create matrix.
 *
 * Creates legacy HiOp dense matrix by default, RAJA vector when memory space
 * is specified.
 */
hiopMatrixDense* LinearAlgebraFactory::create_matrix_dense(const ExecSpaceInfo& hi,  // const std::string& mem_space,
                                                           const size_type& m,
                                                           const size_type& glob_n,
                                                           index_type* col_part,
                                                           MPI_Comm comm,
                                                           const size_type& m_max_alloc)
{
  const std::string mem_space_upper = toupper(hi.mem_space_);
  if(mem_space_upper == "DEFAULT") {
    return new hiopMatrixDenseRowMajor(m, glob_n, col_part, comm, m_max_alloc);
  } else {
    if(hi.exec_backend_ == "RAJA") {
#ifdef HIOP_USE_RAJA
      if(hi.mem_backend_ == "UMPIRE") {
#if defined(HIOP_USE_CUDA)
        return new hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>(m,
                                                                                               glob_n,
                                                                                               mem_space_upper,
                                                                                               col_part,
                                                                                               comm,
                                                                                               m_max_alloc);
#elif defined(HIOP_USE_HIP)
        return new hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>(m,
                                                                                              glob_n,
                                                                                              mem_space_upper,
                                                                                              col_part,
                                                                                              comm,
                                                                                              m_max_alloc);
#else   // this is for #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        return new hiop::hiopMatrixDenseRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>(m,
                                                                                              glob_n,
                                                                                              mem_space_upper,
                                                                                              col_part,
                                                                                              comm,
                                                                                              m_max_alloc);
#endif  // HIOP_USE_CUDA

      } else {  // if(hi.mem_backend_ == "UMPIRE")
        // RAJA exec policy but memory backend not based on Umpire
        assert(false && "work in progress");
        if(hi.mem_backend_ == "CUDA") {
#ifdef HIOP_USE_CUDA
          assert(mem_space_upper == "DEVICE");
          return new hiop::hiopMatrixDenseRaja<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(m,
                                                                                               glob_n,
                                                                                               mem_space_upper,
                                                                                               col_part,
                                                                                               comm,
                                                                                               m_max_alloc);
#else
          assert(false && "cuda memory backend not available because HiOp was not built with CUDA");
#endif  // HIOP_USE_CUDA

        } else if(hi.mem_backend_ == "HIP") {
#if defined(HIOP_USE_HIP)
          assert(mem_space_upper == "DEVICE");
          return new hiop::hiopMatrixDenseRaja<hiop::MemBackendHip, hiop::ExecPolicyRajaHip>(m,
                                                                                             glob_n,
                                                                                             mem_space_upper,
                                                                                             col_part,
                                                                                             comm,
                                                                                             m_max_alloc);
#else
          assert(false && "hip memory backend not available because HiOp was not built with HIP");
          return nullptr;
#endif  // HIOP_USE_HIP

        } else {
          // RAJA-OMP exec policy with non-Umpire memory backend

          assert(false && "work in progress");
          assert(mem_space_upper == "host");

#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
          return new hiop::hiopMatrixDenseRaja<hiop::MemBackendCpp, hiop::ExecPolicyRajaOmp>(m,
                                                                                             glob_n,
                                                                                             mem_space_upper,
                                                                                             col_part,
                                                                                             comm,
                                                                                             m_max_alloc);
#else
          assert(false && "cuda memory backend not available because HiOp was not built with RAJA-OMP");
#endif  //! defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        }
      }  // end of else if(hi.mem_backend_ == "UMPIRE")
#else         // else of ifdef HIOP_USE_RAJA
      assert(false &&
             "requested memory space not available because Hiop was not"
             "built with RAJA support");
      return nullptr;
#endif        // HIOP_USE_RAJA
    } else {  // for if(hi.exec_backend_ == "RAJA")
      assert(false && "device memory backend not available because HiOp was not built with RAJA");
      if(mem_space_upper == "CUDA") {
#ifdef HIOP_USE_CUDA
        assert(mem_space_upper == "DEVICE");
        return new hiop::hiopMatrixDenseRaja<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(m,
                                                                                             glob_n,
                                                                                             mem_space_upper,
                                                                                             col_part,
                                                                                             comm,
                                                                                             m_max_alloc);
#else
        assert(false && "requested memory space not available because Hiop was not built with CUDA");
#endif  // HIOP_USE_CUDA
      } else if(mem_space_upper == "HIP") {
        assert(false && "to be implemented");
      } else {
        assert(false && "to be implemented");
      }
    }  // end of else for if(hi.exec_backend_ == "RAJA")
  }  // end of else if(mem_space_upper == "DEFAULT")
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Creates an instance of a sparse matrix of the appropriate implementation
 * depending on the build.
 */
hiopMatrixSparse* LinearAlgebraFactory::create_matrix_sparse(const ExecSpaceInfo& hi,
                                                             size_type rows,
                                                             size_type cols,
                                                             size_type nnz)
{
  const std::string mem_space_upper = toupper(hi.mem_space_);
  if(mem_space_upper == "DEFAULT") {
    return new hiopMatrixSparseTriplet(rows, cols, nnz);
  } else {
    if(hi.exec_backend_ == "RAJA") {
#ifdef HIOP_USE_RAJA
      if(hi.mem_backend_ == "UMPIRE") {
#if defined(HIOP_USE_CUDA)
        return new hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>(rows,
                                                                                                 cols,
                                                                                                 nnz,
                                                                                                 mem_space_upper);
#elif defined(HIOP_USE_HIP)
        return new hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>(rows,
                                                                                                cols,
                                                                                                nnz,
                                                                                                mem_space_upper);
#else           // this is for #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        return new hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>(rows,
                                                                                                cols,
                                                                                                nnz,
                                                                                                mem_space_upper);
#endif          // HIOP_USE_CUDA
      } else {  // if(hi.mem_backend_ == "UMPIRE")
        // RAJA exec policy but memory backend not based on Umpire
        assert(false && "work in progress");
        if(hi.mem_backend_ == "CUDA") {
#ifdef HIOP_USE_CUDA
          assert(mem_space_upper == "DEVICE");
          return new hiopMatrixRajaSparseTriplet<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(rows,
                                                                                                 cols,
                                                                                                 nnz,
                                                                                                 mem_space_upper);
#else
          assert(false && "cuda memory backend not available because HiOp was not built with CUDA");
          return nullptr;
#endif  // HIOP_USE_CUDA
        } else if(hi.mem_backend_ == "HIP") {
#if defined(HIOP_USE_HIP)
          assert(mem_space_upper == "DEVICE");
          return new hiopMatrixRajaSparseTriplet<hiop::MemBackendHip, hiop::ExecPolicyRajaHip>(rows,
                                                                                               cols,
                                                                                               nnz,
                                                                                               mem_space_upper);
#else
          assert(false && "hip memory backend not available because HiOp was not built with HIP");
          return nullptr;
#endif  // HIOP_USE_HIP
        } else {
          // RAJA-OMP exec policy with non-Umpire memory backend

          assert(false && "work in progress");
          assert(mem_space_upper == "host");

#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
          return new hiopMatrixRajaSparseTriplet<hiop::MemBackendCpp, hiop::ExecPolicyRajaOmp>(rows,
                                                                                               cols,
                                                                                               nnz,
                                                                                               mem_space_upper);
#else
          return new hiopMatrixSparseTriplet(rows, cols, nnz);
#endif  //! defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        }
      }  // end of else if(hi.mem_backend_ == "UMPIRE")
#else
      assert(false &&
             "requested memory space not available because Hiop was not"
             "built with RAJA support");
      return nullptr;
#endif        // HIOP_USE_RAJA
    } else {  // for if(hi.exec_backend_ == "RAJA")
      if(mem_space_upper == "CUDA") {
#ifdef HIOP_USE_CUDA
        assert(mem_space_upper == "DEVICE");
        assert(false && "not supported yet");
        return nullptr;
#else
        assert(false && "requested memory space not available because Hiop was not built with CUDA");
        return nullptr;
#endif  // HIOP_USE_CUDA
      } else if(mem_space_upper == "HIP") {
        assert(false && "to be implemented");
        return nullptr;
      } else {
        assert(false && "to be implemented");
        return nullptr;
      }
    }  // end of else for if(hi.exec_backend_ == "RAJA")
  }  // end of else if(mem_space_upper == "DEFAULT")
  assert(false && "should not reach here");
  return nullptr;
}

hiopMatrixSparseCSR* LinearAlgebraFactory::create_matrix_sparse_csr(const std::string& mem_space)
{
  const std::string mem_space_upper = toupper(mem_space);
  if(mem_space_upper == "DEFAULT") {
    return new hiopMatrixSparseCSRSeq();
  } else {
#ifdef HIOP_USE_CUDA
    return new hiopMatrixSparseCSRCUDA();
#else
    assert(false && "requested memory space not available because Hiop was not built with CUDA support");
#endif
  }
  assert(false && "should not reach here");
  return nullptr;
}

hiopMatrixSparseCSR* LinearAlgebraFactory::create_matrix_sparse_csr(const std::string& mem_space,
                                                                    size_type rows,
                                                                    size_type cols,
                                                                    size_type nnz)
{
  const std::string mem_space_upper = toupper(mem_space);
  if(mem_space_upper == "DEFAULT") {
    return new hiopMatrixSparseCSRSeq(rows, cols, nnz);
  } else {
#ifdef HIOP_USE_CUDA
    return new hiopMatrixSparseCSRCUDA(rows, cols, nnz);
#else
    assert(false && "requested memory space not available because Hiop was not built with CUDA support");
#endif
  }
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Creates an instance of a symmetric sparse matrix of the appropriate
 * implementation depending on the build.
 */
hiopMatrixSparse* LinearAlgebraFactory::create_matrix_sym_sparse(const ExecSpaceInfo& hi, size_type size, size_type nnz)
{
  const std::string mem_space_upper = toupper(hi.mem_space_);
  if(mem_space_upper == "DEFAULT") {
    return new hiopMatrixSymSparseTriplet(size, nnz);
  } else {
    if(hi.exec_backend_ == "RAJA") {
#ifdef HIOP_USE_RAJA
      if(hi.mem_backend_ == "UMPIRE") {
#if defined(HIOP_USE_CUDA)
        return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>(size,
                                                                                                    nnz,
                                                                                                    mem_space_upper);
#elif defined(HIOP_USE_HIP)
        return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>(size,
                                                                                                   nnz,
                                                                                                   mem_space_upper);
#else           // this is for #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>(size,
                                                                                                   nnz,
                                                                                                   mem_space_upper);
#endif          // HIOP_USE_CUDA
      } else {  // if(hi.mem_backend_ == "UMPIRE")
        // RAJA exec policy but memory backend not based on Umpire
        assert(false && "work in progress");
        if(hi.mem_backend_ == "CUDA") {
#ifdef HIOP_USE_CUDA
          assert(mem_space_upper == "DEVICE");
          return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendCuda, hiop::ExecPolicyRajaCuda>(size,
                                                                                                    nnz,
                                                                                                    mem_space_upper);
#else
          assert(false && "cuda memory backend not available because HiOp was not built with CUDA");
          return nullptr;
#endif  // HIOP_USE_CUDA
        } else if(hi.mem_backend_ == "HIP") {
#if defined(HIOP_USE_HIP)
          assert(mem_space_upper == "DEVICE");
          return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendHip, hiop::ExecPolicyRajaHip>(size,
                                                                                                  nnz,
                                                                                                  mem_space_upper);
#else
          assert(false && "hip memory backend not available because HiOp was not built with HIP");
          return nullptr;
#endif  // HIOP_USE_HIP
        } else {
          // RAJA-OMP exec policy with non-Umpire memory backend

          assert(false && "work in progress");
          assert(mem_space_upper == "host");

#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
          return new hiopMatrixRajaSymSparseTriplet<hiop::MemBackendCpp, hiop::ExecPolicyRajaOmp>(size,
                                                                                                  nnz,
                                                                                                  mem_space_upper);
#else
          return new hiopMatrixSymSparseTriplet(size, nnz);
#endif  //! defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
        }
      }  // end of else if(hi.mem_backend_ == "UMPIRE")
#else
      assert(false &&
             "requested memory space not available because Hiop was not"
             "built with RAJA support");
      return nullptr;
#endif        // HIOP_USE_RAJA
    } else {  // for if(hi.exec_backend_ == "RAJA")
      if(mem_space_upper == "CUDA") {
#ifdef HIOP_USE_CUDA
        assert(mem_space_upper == "DEVICE");
        assert(false && "not supported yet");
        return nullptr;
#else
        assert(false && "requested memory space not available because Hiop was not built with CUDA");
        return nullptr;
#endif  // HIOP_USE_CUDA
      } else if(mem_space_upper == "HIP") {
        assert(false && "to be implemented");
        return nullptr;
      } else {
        assert(false && "to be implemented");
        return nullptr;
      }
    }  // end of else for if(hi.exec_backend_ == "RAJA")
  }  // end of else if(mem_space_upper == "DEFAULT")
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Static method to create a raw C array
 */
double* LinearAlgebraFactory::create_raw_array(const std::string& mem_space, size_type n)
{
  const std::string mem_space_upper = toupper(mem_space);
  if(mem_space_upper == "DEFAULT") {
    return new double[n];
  } else {
#ifdef HIOP_USE_RAJA
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator al = resmgr.getAllocator(mem_space_upper);
    return static_cast<double*>(al.allocate(n * sizeof(double)));
#else
    assert(false &&
           "requested memory space not available because Hiop was not"
           "built with RAJA support");
#endif
  }
  assert(false && "should not reach here");
  return nullptr;
}

/**
 * @brief Static method to delete a raw C array
 */
void LinearAlgebraFactory::delete_raw_array(const std::string& mem_space, double* a)
{
  const std::string mem_space_upper = toupper(mem_space);
  if(mem_space_upper == "DEFAULT") {
    delete[] a;
  } else {
#ifdef HIOP_USE_RAJA
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator al = resmgr.getAllocator(mem_space_upper);
    al.deallocate(a);
#endif
  }
}
