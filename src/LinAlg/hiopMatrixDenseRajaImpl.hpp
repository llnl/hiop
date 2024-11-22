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
 * @file hiopMatrixDenseRajaImpl.hpp
 *
 * @author Jake Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Robert Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */
#include "hiopMatrixDenseRaja.hpp"

#include <cstdio>
#include <cassert>

#include "hiopVectorRaja.hpp"
#include "hiopVectorPar.hpp"
#include "hiop_blasdefs.hpp"

namespace hiop
{
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::hiopMatrixDenseRaja(const size_type& m,
                                                                  const size_type& glob_n,
                                                                  std::string mem_space,
                                                                  index_type* col_part /* = nullptr */,
                                                                  MPI_Comm comm /* = MPI_COMM_SELF */,
                                                                  const size_type& m_max_alloc /* = -1 */)
    : hiopMatrixDense(m, glob_n, comm),
      exec_space_(ExecSpace<MEMBACKEND, RAJAEXECPOL>(MEMBACKEND(mem_space))),
      exec_space_host_(ExecSpace<MEMBACKENDHOST, EXECPOLICYHOST>(MEMBACKENDHOST::new_backend_host())),
      mem_space_(mem_space),
      buff_mxnlocal_host_(nullptr)
{
  m_local_ = m;
  n_global_ = glob_n;
  comm_ = comm;
  int P = 0;
  if(col_part) {
#ifdef HIOP_USE_MPI
    int ierr = MPI_Comm_rank(comm_, &P);
    assert(ierr == MPI_SUCCESS);
#endif
    glob_jl_ = col_part[P];
    glob_ju_ = col_part[P + 1];
  } else {
    glob_jl_ = 0;
    glob_ju_ = n_global_;
  }
  n_local_ = glob_ju_ - glob_jl_;

  myrank_ = P;

  max_rows_ = m_max_alloc;
  if(max_rows_ == -1) max_rows_ = m_local_;
  assert(max_rows_ >= m_local_ && "the requested extra allocation is smaller than the allocation needed by the matrix");

#ifndef HIOP_USE_GPU
  assert(mem_space_ == "HOST");
#endif

  data_dev_ = exec_space_.template alloc_array<double>(n_local_ * max_rows_);

  if(exec_space_.mem_backend().is_device()) {
    // If memory space is on device, create a host mirror
    data_host_ = exec_space_host_.template alloc_array<double>(n_local_ * max_rows_);
    yglob_host_ = exec_space_host_.template alloc_array<double>(m_local_);
    ya_host_ = exec_space_host_.template alloc_array<double>(m_local_);
  } else {
    data_host_ = data_dev_;
    // If memory space is not on device, these buffers are allocated in memory space
    yglob_host_ = exec_space_.template alloc_array<double>(m_local_);
    ya_host_ = exec_space_.template alloc_array<double>(m_local_);
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::~hiopMatrixDenseRaja()
{
  if(data_dev_ != data_host_) {
    exec_space_host_.dealloc_array(data_host_);
    exec_space_host_.dealloc_array(yglob_host_);
    exec_space_host_.dealloc_array(ya_host_);
  } else {
    exec_space_.dealloc_array(yglob_host_);
    exec_space_.dealloc_array(ya_host_);
  }
  exec_space_.dealloc_array(data_dev_);

  data_host_ = nullptr;
  data_dev_ = nullptr;
  yglob_host_ = nullptr;
  ya_host_ = nullptr;

  if(buff_mxnlocal_host_ != nullptr) {
    exec_space_host_.dealloc_array(buff_mxnlocal_host_);
    buff_mxnlocal_host_ = nullptr;
  }
}

/**
 * @brief Matrix copy constructor
 *
 */
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::hiopMatrixDenseRaja(const hiopMatrixDenseRaja& dm)
{
  n_local_ = dm.n_local_;
  m_local_ = dm.m_local_;
  n_global_ = dm.n_global_;
  glob_jl_ = dm.glob_jl_;
  glob_ju_ = dm.glob_ju_;
  comm_ = dm.comm_;
  myrank_ = dm.myrank_;
  max_rows_ = dm.max_rows_;
  mem_space_ = dm.mem_space_;

  data_dev_ = exec_space_.template alloc_array<double>(n_local_ * max_rows_);
  if(exec_space_.mem_backend().is_device()) {
    // If memory space is on device, create a host mirror
    data_host_ = exec_space_host_.template alloc_array<double>(n_local_ * max_rows_);
    yglob_host_ = exec_space_host_.template alloc_array<double>(m_local_);
    ya_host_ = exec_space_host_.template alloc_array<double>(m_local_);
    buff_mxnlocal_host_ = nullptr;
    if(dm.buff_mxnlocal_host_ != nullptr) {
      buff_mxnlocal_host_ = exec_space_host_.template alloc_array<double>(max_rows_ * n_local_);
    }
  } else {
    data_host_ = data_dev_;
    // If memory space is not on device, these buffers are allocated in memory space
    yglob_host_ = exec_space_.template alloc_array<double>(m_local_);
    ya_host_ = exec_space_.template alloc_array<double>(m_local_);
    buff_mxnlocal_host_ = nullptr;
    if(dm.buff_mxnlocal_host_ != nullptr) {
      buff_mxnlocal_host_ = exec_space_.template alloc_array<double>(max_rows_ * n_local_);
    }
  }
}

/**
 * @brief Appends the contents of an input vector to the row past the end in this matrix.
 *
 * @pre The length of the vector must equal this->n_local_
 * @pre m_local < max_rows_
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::appendRow(const hiopVector& rowvec)
{
  const auto& row = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(rowvec);
#ifdef HIOP_DEEPCHECKS
  assert(row.get_local_size() == n_local_);
  assert(m_local_ < max_rows_ && "no more space to append rows ... should have preallocated more rows.");
#endif
  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  exec_space_.copy(&Mview(m_local_, 0), row.local_data_const(), n_local_, row.exec_space());
  m_local_++;
}

/**
 * @brief Copies the elements of the input matrix to this matrix.
 *
 * @param[in] dm - Matrix whose elements will be copied.
 *
 * @pre The input and this matrix must have the same size and partitioning.
 * @post Matrix `dm` is unchanged.
 * @post Elements of `this` are overwritten
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyFrom(const hiopMatrixDense& dmmat)
{
  const auto& dm = dynamic_cast<const hiopMatrixDenseRaja&>(dmmat);
  // Verify sizes and partitions
  assert(n_local_ == dm.n_local_);
  assert(m_local_ == dm.m_local_);
  assert(n_global_ == dm.n_global_);
  assert(glob_jl_ == dm.glob_jl_);
  assert(glob_ju_ == dm.glob_ju_);

  exec_space_.copy(data_dev_, dm.data_dev_, n_local_ * max_rows_, dm.exec_space_);
}

/**
 * @brief Copies the elements of `this` matrix to output buffer.
 *
 * @pre The input buffer is big enough to hold the entire matrix.
 * @pre The memory pointed at by the input is in the same execution space as `this`.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copy_to(double* dest)
{
  if(nullptr != dest) {
    exec_space_.copy(dest, data_dev_, n_local_ * max_rows_);
  }
}

/**
 * @brief Copies the elements of the input buffer to this matrix.
 *
 * @param[in] buffer - The beginning of a matrix
 *
 * @pre The input matrix is a pointer to the beginning of a row-major 2D
 * data block with the same dimensions as this matrix.
 * @pre The memory pointed at by the input is in the same execution space as `this`.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyFrom(const double* src)
{
  if(nullptr != src) {
    exec_space_.copy(data_dev_, src, m_local_ * n_local_);
  }
}

/**
 * @brief Copies rows from a source matrix to this matrix.
 *
 * @param[in] src - Matrix whose rows will be copied.
 * @param[in] num_rows - Number of rows to copy.
 * @param[in] row_dest - Starting row in this matrix to copy to.
 *
 * @pre this->n_global_ == src.n_global_ && this->n_local_ == src.n_local_
 * @pre row_dest + num_rows <= this->m_local_
 * @pre num_rows <= src.m_local_
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyRowsFrom(const hiopMatrixDense& srcmat, int num_rows, int row_dest)
{
  if(num_rows == 0) {
    return;
  }

  const auto& src = dynamic_cast<const hiopMatrixDenseRaja&>(srcmat);
#ifdef HIOP_DEEPCHECKS
  assert(row_dest >= 0);
  assert(n_global_ == src.n_global_);
  assert(n_local_ == src.n_local_);
  assert(row_dest + num_rows <= m_local_);
  assert(num_rows <= src.m_local_);
#endif
  // copies num_rows rows of length n_local_ from beginning of src
  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  exec_space_.copy(&Mview(row_dest, 0), src.data_dev_, n_local_ * num_rows, src.exec_space_);
}

/**
 * @brief Copies rows from a source matrix to this matrix.
 *
 * @param[in] src_mat - source matrix
 * @param[in] rows_idxs - indices of rows in src_mat to be copied; a device pointer to an array
 * of n_rows indeces
 * @param[in] n_rows - number of rows to be copied
 *
 * @pre `src_mat` and `this` have same number of columns and partitioning.
 * @pre m_local_ == n_rows <= src_mat.m_local_
 *
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyRowsFrom(const hiopMatrix& src_mat,
                                                                const index_type* rows_idxs,
                                                                size_type n_rows)
{
  const hiopMatrixDenseRaja& src = dynamic_cast<const hiopMatrixDenseRaja&>(src_mat);
  assert(n_global_ == src.n_global_);
  assert(n_local_ == src.n_local_);
  assert(n_rows <= src.m_local_);
  assert(n_rows == m_local_);

  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  RAJA::View<const double, RAJA::Layout<2>> Sview(src.data_dev_, src.m_local_, src.n_local_);

  for(int i = 0; i < n_rows; ++i) {
    // rm.copy(&Mview(i, 0), const_cast<double*>(&Sview(rows_idxs[i], 0)), n_local_ * sizeof(double));
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type j) { Mview(i, j) = Sview(rows_idxs[i], j); });
  }
}

/**
 * @brief Copies the content of `src` into a location in `this` matrix.
 *
 * @param[in] i_start Starting row of this matrix to copy to.
 * @param[in] j_start Starting column of this matrix to copy to.
 * @param[in] src     Source matrix to copy into this one.
 *
 * @pre The size of _src_ plus the starting location must be within the bounds
 * of this matrix.
 * @pre This method should only be used with non-distributed matrices.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyBlockFromMatrix(const index_type i_start,
                                                                       const index_type j_start,
                                                                       const hiopMatrixDense& srcmat)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRaja&>(srcmat);
  assert(n_local_ == n_global_ && "this method should be used only in 'serial' mode");
  assert(src.n_local_ == src.n_global_ && "this method should be used only in 'serial' mode");
  assert(m_local_ >= i_start + src.m_local_ && "the matrix does not fit as a sublock in 'this' at specified coordinates");
  assert(n_local_ >= j_start + src.n_local_ && "the matrix does not fit as a sublock in 'this' at specified coordinates");

  // quick returns for empty source matrices
  if(src.n() == 0) return;
  if(src.m() == 0) return;
#ifdef HIOP_DEEPCHECKS
  assert(i_start < m_local_ || !m_local_);
  assert(j_start < n_local_ || !n_local_);
  assert(i_start >= 0);
  assert(j_start >= 0);
#endif
  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Sview(src.data_dev_, src.m_local_, src.n_local_);
  for(index_type ii = 0; ii < src.m_local_; ii++)
    exec_space_.copy(&Mview(ii + i_start, j_start), &Sview(ii, 0), src.n_local_, src.exec_space_);
}

/**
 * @brief Copies a block of elements from an input matrix to this matrix.
 *
 * @param[in] src     Matrix to copy elements from.
 * @param[in] i_block Starting row to copy from _src_
 * @param[in] j_block Starting column to copy from _src_
 *
 * @pre This method should only be used with non-distributed matrices.
 * @pre The selected block in _src_ must fill the entirety of this matrix.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyFromMatrixBlock(const hiopMatrixDense& srcmat,
                                                                       const int i_block,
                                                                       const int j_block)
{
  const auto& src = dynamic_cast<const hiopMatrixDenseRaja&>(srcmat);
  assert(n_local_ == n_global_ && "this method should be used only in 'serial' mode");
  assert(src.n_local_ == src.n_global_ && "this method should be used only in 'serial' mode");
  assert(m_local_ + i_block <= src.m_local_ && "the source does not enough rows to fill 'this'");
  assert(n_local_ + j_block <= src.n_local_ && "the source does not enough cols to fill 'this'");

  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Sview(src.data_dev_, src.m_local_, src.n_local_);

  if(n_local_ == src.n_local_) {
    // j_block=0
    exec_space_.copy(&Mview(0, 0), &Sview(i_block, 0), n_local_ * m_local_, src.exec_space_);
  } else {
    for(int i = 0; i < m_local_; i++) {
      exec_space_.copy(&Mview(i, 0), &Sview(i + i_block, j_block), n_local_, src.exec_space_);
    }
  }
}

/**
 * @brief Shifts the rows of this matrix up or down by a specified amount.
 *
 * @todo Document this better.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::shiftRows(size_type shift)
{
  if(shift == 0) return;
  if(abs(shift) == m_local_) return;  // nothing to shift
  if(m_local_ <= 1) return;           // nothing to shift

  assert(abs(shift) < m_local_);

  // at this point m_local_ should be >=2
  assert(m_local_ >= 2);
  // and
  assert(m_local_ - abs(shift) >= 1);
#if defined(HIOP_DEEPCHECKS) && !defined(NDEBUG)
  copyFromDev();
  double test1 = 8.3, test2 = -98.3;
  if(n_local_ > 0) {
    // not sure if memcpy is copying sequentially on all systems. we check this.
    // let's at least check it
    //! test1=shift<0 ? M_host_[-shift][0] : M_host_[m_local_-shift-1][0];
    test1 = shift < 0 ? data_host_[-shift * n_local_] : data_host_[(m_local_ - shift - 1) * n_local_];
    //! test2=shift<0 ? M_host_[-shift][n_local_-1] : M_host_[m_local_-shift-1][n_local_-1];
    test2 = shift < 0 ? data_host_[-shift * n_local_ + n_local_ - 1]
                      : data_host_[(m_local_ - shift - 1) * n_local_ + n_local_ - 1];
  }
#endif

  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  if(shift < 0) {
    for(int row = 0; row < m_local_ + shift; row++) {
      exec_space_.copy(&Mview(row, 0), &Mview(row - shift, 0), n_local_);
    }
  } else {
    for(int row = m_local_ - 1; row >= shift; row--) {
      exec_space_.copy(&Mview(row, 0), &Mview(row - shift, 0), n_local_);
    }
  }

#if defined(HIOP_DEEPCHECKS) && !defined(NDEBUG)
  copyFromDev();
  if(n_local_ > 0) {
    assert(test1 == data_host_[n_local_ * (shift < 0 ? 0 : m_local_ - 1)] &&
           "a different copy technique than memcpy is needed on this system");
    assert(test2 == data_host_[n_local_ * (shift < 0 ? 0 : m_local_ - 1) + n_local_ - 1] &&
           "a different copy technique than memcpy is needed on this system");
  }
#endif
}

/// Replaces a row in this matrix with the elements of a vector
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::replaceRow(index_type row, const hiopVector& vec)
{
  const auto& rvec = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(vec);
  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  assert(row >= 0);
  assert(row < m_local_);
  size_type vec_size = rvec.get_local_size();
  exec_space_.copy(&Mview(row, 0), rvec.local_data_const(), (vec_size >= n_local_ ? n_local_ : vec_size));
}

/// Overwrites the values in row_vec with those from a specified row in this matrix
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::getRow(index_type irow, hiopVector& row_vec)
{
  RAJA::View<double, RAJA::Layout<2>> Mview(this->data_dev_, m_local_, n_local_);
  assert(irow >= 0);
  assert(irow < m_local_);
  auto& vec = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(row_vec);
  assert(n_local_ == vec.get_local_size());
  exec_space_.copy(vec.local_data(), &Mview(irow, 0), n_local_, vec.exec_space());
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::set_Hess_FR(const hiopMatrixDense& Hess, const hiopVector& add_diag_de)
{
  double one{1.0};
  copyFrom(Hess);
  addDiagonal(one, add_diag_de);
}

#ifdef HIOP_DEEPCHECKS
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::overwriteUpperTriangleWithLower()
{
  assert(n_local_ == n_global_ && "Use only with local, non-distributed matrices");
  int n_local = n_local_;
  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = i + 1; j < n_local; j++) Mview(i, j) = Mview(j, i);
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::overwriteLowerTriangleWithUpper()
{
  assert(n_local_ == n_global_ && "Use only with local, non-distributed matrices");
  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(1, m_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = 0; j < i; j++) Mview(i, j) = Mview(j, i);
      });
}
#endif

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixDense* hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::alloc_clone() const
{
  hiopMatrixDense* c = new hiopMatrixDenseRaja(*this);
  return c;
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixDense* hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::new_copy() const
{
  hiopMatrixDense* c = new hiopMatrixDenseRaja(*this);
  c->copyFrom(*this);
  return c;
}

/**
 * @brief Sets all the elements of the source matrix to zero.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::setToZero()
{
  this->setToConstant(0.0);
}

/**
 * @brief Sets all the elements of this matrix to a constant.
 * @todo consider GPU BLAS for this (DCOPY)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::setToConstant(double c)
{
  double* dd = this->data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_ * m_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] = c; });
}

/**
 * @brief Checks if all the elements of the source matrix are finite.
 * @return True if all source elements are finite, false otherwise.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::isfinite() const
{
  double* dd = this->data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_ * m_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(!std::isfinite(dd[i])) any += 1;
      });

  return any.get() == 0;
}

/**
 * @brief Print matrix to a file
 *
 * @note This is I/O function and takes place on the host. Need to move
 * matrix data to the host mirror memory.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::print(FILE* f,
                                                         const char* msg /*=NULL*/,
                                                         int maxRows /*=-1*/,
                                                         int maxCols /*=-1*/,
                                                         int rank /*=-1*/) const
{
  if(myrank_ == rank || rank == -1) {
    if(NULL == f) f = stdout;
    if(maxRows > m_local_) maxRows = m_local_;
    if(maxCols > n_local_) maxCols = n_local_;

    if(msg) {
      fprintf(f, "%s (local_dims=[%d,%d])\n", msg, m_local_, n_local_);
    } else {
      fprintf(f,
              "hiopMatrixDenseRaja::printing max=[%d,%d] (local_dims=[%d,%d], on rank=%d)\n",
              maxRows,
              maxCols,
              m_local_,
              n_local_,
              myrank_);
    }
    maxRows = maxRows >= 0 ? maxRows : m_local_;
    maxCols = maxCols >= 0 ? maxCols : n_local_;
    fprintf(f, "[");
    for(int i = 0; i < maxRows; i++) {
      if(i > 0) fprintf(f, " ");
      for(int j = 0; j < maxCols; j++) fprintf(f, "%20.12e ", data_host_[i * n_local_ + j]);
      if(i < maxRows - 1)
        fprintf(f, "; ...\n");
      else
        fprintf(f, "];\n");
    }
  }  // if(myrank_==rank || rank==-1)
}

#include <unistd.h>

/**
 * @brief Multiplies this matrix by a vector and
 * stores the result in another vector.
 *
 * The full operation performed is:
 * _y_ = _beta_ * _y_ + _alpha_ * _this_ * _x_
 *
 * @param[in] beta Amount to scale _y_ by
 * @param[out] y Vector to store result in
 * @param[in] alpha Amount to scale result of this * _x_ by
 * @param[in] x Vector to multiply this matrix with
 *
 * @pre The length of _x_ equals the number of column
 *   of this matrix (n_local_).
 * @pre The length of _y_ equals the number of rows
 *   of this matrix (m_local_).
 * @pre _y_ is non-distributed.
 * @pre _x_ and _y_ contain only finite values.
 * @post _y_ will have its elements overwritten with the result of the
 * calculation.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta,
                                                            hiopVector& y_,
                                                            double alpha,
                                                            const hiopVector& x_) const
{
  auto& y = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(y_);
  const auto& x = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(y.get_local_size() == m_local_);
  assert(y.get_size() == m_local_);  // y should not be distributed
  assert(x.get_local_size() == n_local_);
  assert(x.get_size() == n_global_);

  if(beta != 0) {
    assert(y.isfinite_local() && "pre timesvec");
  }
  assert(x.isfinite_local());
#endif

  timesVec(beta, y.local_data(), alpha, x.local_data_const());

#ifdef HIOP_DEEPCHECKS
  assert(y.isfinite_local() && "post timesVec");
#endif
}

/**
 * @brief Multiplies this matrix by a vector and
 * stores the result in another vector.
 *
 * @pre The input and output vectors are pointers to memory allocated in the same
 * execution/memory space.
 * @todo GPU BLAS (<D>GEMV, <D>SCAL) for this
 *
 * see timesVec for more detail
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta, double* ya, double alpha, const double* xa) const
{
#ifdef HIOP_USE_MPI
  // only add beta*y on one processor (rank 0)
  if(myrank_ != 0) beta = 0.0;
#endif
  double* data = data_dev_;
  int m_local = m_local_;
  int n_local = n_local_;
  //  y = beta * y + alpha * this * x
  RAJA::View<const double, RAJA::Layout<2>> Mview(data, m_local, n_local);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m_local),
      RAJA_LAMBDA(RAJA::Index_type i) {
        double dot = 0;
        for(int j = 0; j < n_local; j++) dot += Mview(i, j) * xa[j];
        ya[i] = beta * ya[i] + alpha * dot;
      });

#ifdef HIOP_USE_MPI
  // here m_local_ is > 0
  exec_space_host_.copy(ya_host_, ya, m_local_, exec_space_);
  int ierr = MPI_Allreduce(ya_host_, yglob_host_, m_local_, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  exec_space_.copy(ya, yglob_host_, m_local_, exec_space_host_);
#endif
}

/**
 * @brief Multiplies the transpose of this matrix by a vector and
 * stores the result in another vector:
 * y = beta * y + alpha * transpose(this) * x
 *
 * @todo GPU BLAS (DGEMV, DSCAL) for this
 *
 * see timesVec for more detail
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::transTimesVec(double beta,
                                                                 hiopVector& y_,
                                                                 double alpha,
                                                                 const hiopVector& x_) const
{
  auto& y = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(y_);
  const auto& x = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(x_);
#ifdef HIOP_DEEPCHECKS
  assert(x.get_local_size() == m_local_);
  assert(x.get_size() == m_local_);  // x should not be distributed
  assert(y.get_local_size() == n_local_);
  assert(y.get_size() == n_global_);
  assert(y.isfinite_local());
  assert(x.isfinite_local());
#endif
  transTimesVec(beta, y.local_data(), alpha, x.local_data_const());
}

/**
 * @brief Multiplies the transpose of this matrix by a vector and
 * stores the result in another vector.
 *
 * @pre The input and output vectors are pointers to the same
 * execution/memory space.
 * @todo GPU BLAS (DGEMV, DSCAL) for this
 *
 * see timesVec for more detail
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::transTimesVec(double beta,
                                                                 double* ya,
                                                                 double alpha,
                                                                 const double* xa) const
{
  double* data = data_dev_;
  int m_local = m_local_;
  int n_local = n_local_;
  // this loop is inefficient if m_local_ is large
  // low n_local_ values effectively serialize this kernel
  // TODO: consider performance benefits of using nested RAJA loop

  RAJA::View<const double, RAJA::Layout<2>> Mview(data, m_local, n_local);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local),
      RAJA_LAMBDA(RAJA::Index_type j) {
        double dot = 0;
        for(int i = 0; i < m_local; i++) dot += Mview(i, j) * xa[i];
        ya[j] = beta * ya[j] + alpha * dot;
      });
}

/**
 * @brief Multiplies this matrix by a matrix and
 * stores the result in another matrix.
 *
 * @param[in] beta Amount to scale _W_ by
 * @param[out] W Matrix to store result in
 * @param[in] alpha Amount to scale result of this * _X_ by
 * @param[in] X Matrix to multiply this matrix with
 *
 * @pre this is an mxn matrix
 * @pre _X_ is an nxk matrix
 * @pre _W_ is an mxk matrix
 * @pre W, this, and X need to be local matrices (not distributed).
 * @pre _X_ and _W_ contain only finite values.
 * @post _W_ will have its elements modified to reflect
 *   the below calculation.
 * @todo GPU BLAS (DGEMV, DSCAL) for this
 *
 * The operation that is performed is:
 * W = beta * W + alpha * this * X
 * @warning MPI parallel computations are _not_ supported.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesMat(double beta,
                                                            hiopMatrix& Wmat,
                                                            double alpha,
                                                            const hiopMatrix& Xmat) const
{
#ifndef HIOP_USE_MPI
  timesMat_local(beta, Wmat, alpha, Xmat);
#else
  auto& W = dynamic_cast<hiopMatrixDenseRaja&>(Wmat);
  // double* WM = W.local_data_host();
  const auto& X = dynamic_cast<const hiopMatrixDenseRaja&>(Xmat);

  assert(W.m() == this->m());
  assert(X.m() == this->n());
  assert(W.n() == X.n());

  if(W.m() == 0 || X.m() == 0 || W.n() == 0) return;
#ifdef HIOP_DEEPCHECKS
  assert(W.isfinite());
  assert(X.isfinite());
#endif

  if(X.n_local_ != X.n_global_ || this->n_local_ != this->n_global_) {
    assert(false && "'timesMat' involving distributed matrices is not needed/supported" &&
           "also, it cannot be performed efficiently with the data distribution used by this class");
    W.setToConstant(beta);
    return;
  }
  timesMat_local(beta, Wmat, alpha, Xmat);
  // if(0==myrank_) timesMat_local(beta,W_,alpha,X_);
  // else          timesMat_local(0.,  W_,alpha,X_);

  // int n2Red=W.m()*W.n();
  // double* Wglob = new_mxnlocal_buff(); //[n2Red];
  // int ierr = MPI_Allreduce(WM[0], Wglob, n2Red, MPI_DOUBLE, MPI_SUM,comm); assert(ierr==MPI_SUCCESS);
  // memcpy(WM[0], Wglob, n2Red*sizeof(double));
#endif  // HIOP_USE_MPI
}

/**
 * @brief Multiplies this matrix by a matrix and stores the result in another matrix.
 *
 * @todo GPU BLAS call for this (DGEMM)
 *
 * see timesMat for more detail
 * MPI NOT SUPPORTED
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesMat_local(double beta,
                                                                  hiopMatrix& W_,
                                                                  double alpha,
                                                                  const hiopMatrix& X_) const
{
  const hiopMatrixDenseRaja& X = dynamic_cast<const hiopMatrixDenseRaja&>(X_);
  hiopMatrixDenseRaja& W = dynamic_cast<hiopMatrixDenseRaja&>(W_);
#ifdef HIOP_DEEPCHECKS
  assert(W.m() == this->m());
  assert(X.m() == this->n());
  assert(W.n() == X.n());
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  assert(W.n_local_ == W.n_global_ && "requested multiplication is not supported, see timesMat");

  double* data = data_dev_;
  double* xdata = X.data_dev_;
  double* wdata = W.data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Xview(xdata, X.m_local_, X.n_local_);
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.m_local_, W.n_local_);
  RAJA::RangeSegment row_range(0, W.m_local_);
  RAJA::RangeSegment col_range(0, W.n_local_);

  auto n_local = n_local_;
  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(int col, int row) {
        double dot = 0;
        for(int k = 0; k < n_local; k++) dot += Mview(row, k) * Xview(k, col);
        Wview(row, col) = beta * Wview(row, col) + alpha * dot;
      });
}

/**
 * @brief Multiplies the transpose of this matrix by a matrix, storing the result
 * in an output matrix.
 *
 * @pre Size of `this` is mxn, X is mxk, W is nxk
 *
 * See timesMat for more detail.
 * Operation performed is: W = beta * W + alpha * this^T * X
 *
 * @warning This method is not MPI distributed!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::transTimesMat(double beta,
                                                                 hiopMatrix& W_,
                                                                 double alpha,
                                                                 const hiopMatrix& X_) const
{
  const hiopMatrixDenseRaja& X = dynamic_cast<const hiopMatrixDenseRaja&>(X_);
  hiopMatrixDenseRaja& W = dynamic_cast<hiopMatrixDenseRaja&>(W_);

  assert(W.m() == n_local_);
  assert(X.m() == m_local_);
  assert(W.n() == X.n());
#ifdef HIOP_DEEPCHECKS
  assert(W.isfinite());
  assert(X.isfinite());
#endif
  if(W.m() == 0) return;

  assert(this->n_global_ == this->n_local_ && "requested parallel multiplication is not supported");

  double* data = data_dev_;
  double* xdata = X.data_dev_;
  double* wdata = W.data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Xview(xdata, X.m_local_, X.n_local_);
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.m_local_, W.n_local_);
  RAJA::RangeSegment row_range(0, W.m_local_);
  RAJA::RangeSegment col_range(0, W.n_local_);

  auto Mm = m_local_;
  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(int col, int row) {
        double dot = 0;
        for(int k = 0; k < Mm; k++) dot += Mview(k, row) * Xview(k, col);
        Wview(row, col) = beta * Wview(row, col) + alpha * dot;
      });
}

/**
 * @brief Multiplies this matrix by the transpose of a matrix and
 * stores the result in another matrix.
 *
 * @todo GPU BLAS call for this (<D>GEMM)
 * @todo Fix the distributed version of this method.
 * (Something to think about: how to find the dot product of two rows
 * of distributed matrices? This is what we're solving here.)
 *
 * see timesMat for more detail
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesMatTrans_local(double beta,
                                                                       hiopMatrix& W_,
                                                                       double alpha,
                                                                       const hiopMatrix& X_) const
{
  const auto& X = dynamic_cast<const hiopMatrixDenseRaja&>(X_);
  auto& W = dynamic_cast<hiopMatrixDenseRaja&>(W_);
#ifdef HIOP_DEEPCHECKS
  assert(W.m() == m_local_);
  assert(X.n_local_ == n_local_);
  assert(W.n() == X.m());
#endif
  assert(W.n_local_ == W.n_global_ && "not intended for the case when the result matrix is distributed.");
  if(W.m() == 0) return;
  if(W.n() == 0) return;

  double* data = data_dev_;
  double* xdata = X.data_dev_;
  double* wdata = W.data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Xview(xdata, X.m_local_, X.n_local_);
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.m_local_, W.n_local_);
  RAJA::RangeSegment row_range(0, W.m_local_);
  RAJA::RangeSegment col_range(0, W.n_local_);

  auto Mn = n_local_;
  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(int col, int row) {
        double dot = 0;
        for(int k = 0; k < Mn; k++) dot += Mview(row, k) * Xview(col, k);  // X^T
        Wview(row, col) = beta * Wview(row, col) + alpha * dot;
      });
}

/**
 * @brief Multiplies this matrix by the transpose of a matrix and
 * stores the result in another matrix: `W = beta*W + alpha*this*X^T`
 *
 * @todo Fix the distributed version of this method.
 *
 * see timesMat for more detail
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::timesMatTrans(double beta,
                                                                 hiopMatrix& Wmat,
                                                                 double alpha,
                                                                 const hiopMatrix& Xmat) const
{
  auto& W = dynamic_cast<hiopMatrixDenseRaja&>(Wmat);
  assert(W.n_local_ == W.n_global_ && "not intended for the case when the result matrix is distributed.");
#ifdef HIOP_DEEPCHECKS
  const auto& X = dynamic_cast<const hiopMatrixDenseRaja&>(Xmat);
  assert(W.isfinite());
  assert(X.isfinite());
  assert(this->n() == X.n());
  assert(this->m() == W.m());
  assert(X.m() == W.n());
#endif

  if(W.m() == 0) return;
  if(W.n() == 0) return;

  // only apply W * beta on one rank
  if(myrank_ == 0)
    timesMatTrans_local(beta, Wmat, alpha, Xmat);
  else
    timesMatTrans_local(0.0, Wmat, alpha, Xmat);

#ifdef HIOP_USE_MPI
  // printf("W.m: %d, W.n: %d\n", W.m(), W.n());
  int n2Red = W.m() * W.n();
  double* Wdata_host = W.data_host_;
  W.copyFromDev();
  double* Wglob = W.new_mxnlocal_host_buff();
  int ierr = MPI_Allreduce(Wdata_host, Wglob, n2Red, MPI_DOUBLE, MPI_SUM, comm_);
  assert(ierr == MPI_SUCCESS);
  memcpy(Wdata_host, Wglob, n2Red * sizeof(double));
  W.copyToDev();
#endif
}

/**
 * @brief Adds the values of a vector to the diagonal of this matrix.
 *
 * @param[in] alpha - Amount to scale values of _d_ by.
 * @param[in] dvec  - Vector to add to this matrix's diagonal.
 *
 * @pre This matrix is square.
 * @pre The length of _dvec_ equals the length of the diagonal of this matrix.
 *
 * @warning This method is not MPI distributed
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addDiagonal(const double& alpha, const hiopVector& dvec)
{
  const auto& d = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(dvec);

#ifdef HIOP_DEEPCHECKS
  assert(d.get_size() == n());
  assert(d.get_size() == m());
  assert(d.get_local_size() == m_local_);
  assert(d.get_local_size() == n_local_);
#endif

  // the min() is symbolic as n/m_local_ should be equal
  int diag = std::min(get_local_size_m(), get_local_size_n());
  double* data = data_dev_;
  const double* dd = d.local_data_const();
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());  // matrix
  RAJA::View<const double, RAJA::Layout<1>> Dview(dd, d.get_size());                        // vector
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, diag),
      RAJA_LAMBDA(RAJA::Index_type i) { Mview(i, i) += Dview(i) * alpha; });
}

/**
 * @brief Adds a constant to the diagonal of this matrix.
 *
 * @param[in] value Adding constant.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addDiagonal(const double& value)
{
  int diag = std::min(get_local_size_m(), get_local_size_n());
  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, diag), RAJA_LAMBDA(RAJA::Index_type i) { Mview(i, i) += value; });
}

/**
 * @brief Adds the values of a vector to the diagonal of this matrix,
 * starting at an offset in the diagonal.
 *
 * @param[in] alpha Amount to scale values of _d_ by.
 * @param[in] start Offset from beginning of this matrix's diagonal.
 * @param[in] d Vector whose elements will be added to the diagonal.
 *
 * @post The elements written will be in the range [start, start + _d_.len)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addSubDiagonal(const double& alpha,
                                                                  index_type start,
                                                                  const hiopVector& dvec)
{
  const auto& d = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(dvec);
  size_type dlen = d.get_size();
#ifdef HIOP_DEEPCHECKS
  assert(start >= 0);
  assert(start + dlen <= n_local_);
#endif

  double* data = data_dev_;
  const double* dd = d.local_data_const();
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());  // matrix
  RAJA::View<const double, RAJA::Layout<1>> Dview(dd, dlen);                                // vector
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(start, start + dlen),
      RAJA_LAMBDA(RAJA::Index_type i) { Mview(i, i) += Dview(i - start) * alpha; });
}

/**
 * @brief Adds the values of vector _dvec_ to the diagonal of this matrix,
 * starting at an offset in both destination and source.
 *
 * @param[in] start_on_dest_diag - Offset on `this` matrix's diagonal.
 * @param[in] alpha - Amount to scale values of _d_ by.
 * @param[in] dvec - Vector whose elements will be added to the diagonal.
 * @param[in] start_on_src_vec - Offset in vector.
 * @param[in] num_elems - Number of elements to add if >= 0, otherwise
 * the remaining elements on _dvec_ starting at _start_on_src_vec_
 *
 * @pre This matrix must be non-distributed and symmetric/square
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addSubDiagonal(int start_on_dest_diag,
                                                                  const double& alpha,
                                                                  const hiopVector& dvec,
                                                                  int start_on_src_vec,
                                                                  int num_elems /* = -1 */)
{
  const auto& d = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(dvec);
  if(num_elems < 0) num_elems = d.get_size() - start_on_src_vec;
  assert(num_elems <= d.get_size());
  assert(n_local_ == n_global_ && "method supported only for non-distributed matrices");
  assert(n_local_ == m_local_ && "method supported only for symmetric matrices");

  assert(start_on_dest_diag >= 0 && start_on_dest_diag < m_local_);
  num_elems = std::min(num_elems, m_local_ - start_on_dest_diag);

  double* data = data_dev_;
  const double* dd = d.local_data_const();
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());
  RAJA::View<const double, RAJA::Layout<1>> Dview(dd, d.get_size());  // vector
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, num_elems),
      RAJA_LAMBDA(RAJA::Index_type i) {
        const int loc = i + start_on_dest_diag;
        Mview(loc, loc) += Dview(i + start_on_src_vec) * alpha;
      });
}

/**
 * @brief Adds a value to a subdiagonal of this matrix, starting
 * at an offset.
 *
 * @param[in] start_on_dest_diag Start on this matrix's diagonal.
 * @param[in] num_elems Number of elements to add.
 * @param[in] c Constant to add.
 *
 * @pre _num_elems_ must be >= 0
 * @pre This matrix must be non-distributed and symmetric/square
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c)
{
  assert(num_elems >= 0);
  assert(start_on_dest_diag >= 0 && start_on_dest_diag + num_elems <= n_local_);
  assert(n_local_ == n_global_ && "method supported only for non-distributed matrices");
  assert(n_local_ == m_local_ && "method supported only for symmetric matrices");

  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, get_local_size_m(), get_local_size_n());
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, num_elems),
      RAJA_LAMBDA(RAJA::Index_type i) {
        const int loc = i + start_on_dest_diag;
        Mview(loc, loc) += c;
      });
}

/**
 * @brief Adds the elements of a matrix to the elements of this matrix.
 *
 * @param[in] alpha Value to scale the elements of _X_ by.
 * @param[in] X_    Matrix to add to this one.
 *
 * @pre _X_ and this matrix must have matching dimensions.
 *
 * @todo update with GPU BLAS call (AXPY)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addMatrix(double alpha, const hiopMatrix& X_)
{
  const hiopMatrixDenseRaja& X = dynamic_cast<const hiopMatrixDenseRaja&>(X_);
#ifdef HIOP_DEEPCHECKS
  assert(m_local_ == X.m_local_);
  assert(n_local_ == X.n_local_);
#endif

  double* dd = this->data_dev_;
  double* dd2 = X.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_ * m_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { dd[i] += dd2[i] * alpha; });
}

/**
 * @brief Adds the values of this matrix to a corresponding section in
 * the upper triangle of the input matrix. (W += alpha*this @ offset)
 *
 * @param[in] row_start Row offset into target matrix.
 * @param[in] col_start Column offset into target matrix.
 * @param[in] alpha     Amount to scale values of this matrix by when adding.
 * @param[out] W        Matrix to be added to.
 *
 * @pre _row_start_ and _col_start_ must be >= 0
 * @pre _row_start_ + this->m() <= W.m()
 * @pre _col_start_ + this->n() <= W.n()
 * @pre _W_ must be a square matrix
 *
 * @post The modified section of _W_ will be a square in the upper triangle
 * starting at (_row_start_, _col_start_) with the dimensions of this matrix.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addToSymDenseMatrixUpperTriangle(int row_start,
                                                                                    int col_start,
                                                                                    double alpha,
                                                                                    hiopMatrixDense& Wmat) const
{
  hiopMatrixDenseRaja& W = dynamic_cast<hiopMatrixDenseRaja&>(Wmat);
  double* wdata = W.data_dev_;
  double* data = data_dev_;

  assert(row_start >= 0 && m() + row_start <= W.m());
  assert(col_start >= 0 && n() + col_start <= W.n());
  assert(W.n() == W.m());

  RAJA::View<double, RAJA::Layout<2>> Mview(data, m_local_, n_local_);
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.get_local_size_m(), W.get_local_size_n());
  RAJA::RangeSegment row_range(0, m_local_);
  RAJA::RangeSegment col_range(0, n_local_);

// Assert that we are only modifying entries in the upper triangle
#if defined(HIOP_DEEPCHECKS) && !defined(NDEBUG)
  int iWmax = row_start + m_local_;
  int jWmin = col_start;
  assert(iWmax <= jWmin && "source entries need to map inside the upper triangular part of destination");
#endif

  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(int jcol, int irow) {
        const int iW = irow + row_start;
        const int jW = jcol + col_start;
        Wview(iW, jW) += alpha * Mview(irow, jcol);
      });
}

/**
 * @brief Adds the values of the transpose of this matrix to a
 * corresponding section in the upper triangle of the input matrix.
 *  W += alpha*this' @ offset
 *
 * see addToSymDenseMatrixUpperTriangle for more information.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::transAddToSymDenseMatrixUpperTriangle(int row_start,
                                                                                         int col_start,
                                                                                         double alpha,
                                                                                         hiopMatrixDense& Wmat) const
{
  hiopMatrixDenseRaja& W = dynamic_cast<hiopMatrixDenseRaja&>(Wmat);
  double* wdata = W.data_dev_;
  double* data = data_dev_;

  assert(row_start >= 0 && n() + row_start <= W.m());
  assert(col_start >= 0 && m() + col_start <= W.n());
  assert(W.n() == W.m());

  RAJA::View<double, RAJA::Layout<2>> Mview(data, this->get_local_size_m(), this->get_local_size_n());
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.get_local_size_m(), W.get_local_size_n());
  RAJA::RangeSegment row_range(0, m_local_);
  RAJA::RangeSegment col_range(0, n_local_);

  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(int jcol, int irow) {
        const int jW = irow + col_start;
        const int iW = jcol + row_start;

#ifdef HIOP_DEEPCHECKS
        assert(iW <= jW && "source entries need to map inside the upper triangular part of destination");
#endif
        Wview(iW, jW) += alpha * Mview(irow, jcol);
      });
}

/**
 * @brief diagonal block of W += alpha*this with 'diag_start' indicating the
 * diagonal entry of W where 'this' should start to contribute.
 *
 * @param[in] diag_start
 * For efficiency, only upper triangle of W is updated since this will be
 * eventually sent to LAPACK and only the upper triangle of 'this' is accessed
 *
 * @pre this->n()==this->m()
 * @pre W.n() == W.m()
 *
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start,
                                                                                                 double alpha,
                                                                                                 hiopMatrixDense& Wmat) const
{
  hiopMatrixDenseRaja& W = dynamic_cast<hiopMatrixDenseRaja&>(Wmat);
  double* wdata = W.data_dev_;
  double* data = data_dev_;

  RAJA::View<double, RAJA::Layout<2>> Mview(data, this->get_local_size_m(), this->get_local_size_n());
  RAJA::View<double, RAJA::Layout<2>> Wview(wdata, W.get_local_size_m(), W.get_local_size_n());
  RAJA::RangeSegment row_range(0, m_local_);
  RAJA::RangeSegment col_range(0, n_local_);

  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(col_range, row_range),
      RAJA_LAMBDA(RAJA::Index_type j, RAJA::Index_type i) {
        if(j < i) return;
        auto iw = i + diag_start;
        auto jw = j + diag_start;
        Wview(iw, jw) += alpha * Mview(i, j);
      });
}

/**
 * @brief Returns the value of the element with the maximum absolute value.
 *
 * @todo Consider using BLAS call (<D>LANGE)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
double hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::max_abs_value()
{
  double* dd = this->data_dev_;
  RAJA::ReduceMax<hiop_raja_reduce, double> norm(0.0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_ * m_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { norm.max(fabs(dd[i])); });
  double maxv = static_cast<double>(norm.get());

#ifdef HIOP_USE_MPI
  double maxvg;
  int ierr = MPI_Allreduce(&maxv, &maxvg, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(ierr == MPI_SUCCESS);
  return maxvg;
#endif
  return maxv;
}

/**
 * @brief Returns the value of the element with the maximum absolute value.
 *
 * @todo Consider using BLAS call (<D>LANGE)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::row_max_abs_value(hiopVector& ret_vec)
{
  assert(ret_vec.get_size() == m());
  ret_vec.setToZero();
  if(0 == m_local_) {
    return;
  }

  auto& vec = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(ret_vec);
  double* vd = vec.local_data();

  double* data = data_dev_;
  int m_local = m_local_;
  int n_local = n_local_;

  RAJA::View<const double, RAJA::Layout<2>> Mview(data, m_local, n_local);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m_local),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = 0; j < n_local; j++) {
          double abs_val = fabs(Mview(i, j));
          vd[i] = (vd[i] > abs_val) ? vd[i] : abs_val;
        }
      });

#ifdef HIOP_USE_MPI
  hiopVectorPar maxvg(m_local_);
  hiopVectorPar vec_host(m_local_);
  vec.copy_to_vectorpar(vec_host);

  int ierr = MPI_Allreduce(vec_host.local_data(), maxvg.local_data(), m_local_, MPI_DOUBLE, MPI_MAX, comm_);
  assert(ierr == MPI_SUCCESS);
  vec.copy_from_vectorpar(maxvg);
#endif
}

/// Scale each row of matrix, according to the scale factor in `ret_vec`
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::scale_row(hiopVector& vec_scal, const bool inv_scale)
{
  double* data = data_dev_;
  auto& vec = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(vec_scal);
  double* vd = vec.local_data();

  int m_local = m_local_;
  int n_local = n_local_;

  RAJA::View<double, RAJA::Layout<2>> Mview(data, m_local, n_local);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m_local),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = 0; j < n_local; j++) {
          Mview(i, j) *= vd[i];
        }
      });
}

#ifdef HIOP_DEEPCHECKS
template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::assertSymmetry(double tol) const
{
  if(n_local_ != n_global_) {
    assert(false && "should be used only for local matrices");
    return false;
  }
  // must be square
  if(m_local_ != n_global_) {
    assert(false);
    return false;
  }

  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, n_local_, n_local_);
  RAJA::RangeSegment range(0, n_local_);

  // symmetry
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(range, range),
      RAJA_LAMBDA(int j, int i) {
        double ij = Mview(i, j);
        double ji = Mview(j, i);
        double relerr = fabs(ij - ji) / (1 + fabs(ij));
        assert(relerr < tol);
        if(relerr >= tol) any += 1;
      });
  return any.get() == 0;
}
#endif

template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::symmetrize()
{
  if(n_local_ != n_global_) {
    assert(false && "should be used only for local matrices");
    return false;
  }
  // must be square
  if(m_local_ != n_global_) {
    assert(false);
    return false;
  }

  double* data = data_dev_;
  RAJA::View<double, RAJA::Layout<2>> Mview(data, n_local_, n_local_);
  RAJA::RangeSegment range(0, n_local_);

  // symmetrize --- copy the upper triangular part to lower tirangular part
  RAJA::kernel<matrix_exec>(
      RAJA::make_tuple(range, range),
      RAJA_LAMBDA(int j, int i) {
        double ij = Mview(i, j);
        if(i < j) {
          Mview(j, i) = ij;
        }
      });
  return true;
}

/// Copy local host mirror data to the memory space
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyToDev()
{
  if(data_dev_ != data_host_) {
    exec_space_.copy(data_dev_, data_host_, n_local_ * max_rows_, exec_space_host_);
  }
}

/// Copy local data from the memory space to host mirror
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::copyFromDev()
{
  if(data_dev_ != data_host_) {
    exec_space_host_.copy(data_host_, data_dev_, n_local_ * max_rows_, exec_space_);
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
double* hiopMatrixDenseRaja<MEMBACKEND, RAJAEXECPOL>::new_mxnlocal_host_buff() const
{
  if(buff_mxnlocal_host_ == nullptr) {
    buff_mxnlocal_host_ = exec_space_host_.template alloc_array<double>(max_rows_ * n_local_);
  }
  return buff_mxnlocal_host_;
}

}  // namespace hiop
