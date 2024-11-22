// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
 * @file hiopVectorRajaImpl.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Cameron Rutherford <cameron.rutherford@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include "ExecSpace.hpp"

#include "LinAlgFactory.hpp"
#include "hiopVectorRaja.hpp"

#include "hiopVectorIntRaja.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <sstream>
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include <limits>
#include <cstddef>

#include <RAJA/RAJA.hpp>

#include "hiopVectorPar.hpp"

namespace hiop
{
// Define type aliases
using real_type = double;
using local_index_type = index_type;
using global_index_type = index_type;

// Define constants
static constexpr real_type zero = 0.0;
static constexpr real_type one = 1.0;

template<class MEMBACKEND, class RAJAEXECPOL>
hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>::hiopVectorRaja(const size_type& glob_n,
                                                        std::string mem_space /* = "HOST" */,
                                                        index_type* col_part /* = NULL */,
                                                        MPI_Comm comm /* = MPI_COMM_NULL */)
    : hiopVector(),
      exec_space_(ExecSpace<MEMBACKEND, RAJAEXECPOL>(MEMBACKEND(mem_space))),
      exec_space_host_(ExecSpace<MEMBACKENDHOST, EXECPOLICYHOST>(MEMBACKENDHOST::new_backend_host())),
      mem_space_(mem_space),
      comm_(comm),
      idx_cumsum_{nullptr}
{
  n_ = glob_n;

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_ == MPI_COMM_NULL) comm_ = MPI_COMM_SELF;
#endif

  int P = 0;
  if(col_part) {
#ifdef HIOP_USE_MPI
    int ierr = MPI_Comm_rank(comm_, &P);
    assert(ierr == MPI_SUCCESS);
#endif
    glob_il_ = col_part[P];
    glob_iu_ = col_part[P + 1];
  } else {
    glob_il_ = 0;
    glob_iu_ = n_;
  }
  n_local_ = glob_iu_ - glob_il_;

#ifndef HIOP_USE_GPU
  assert(mem_space_ == "HOST");
#endif

  data_dev_ = exec_space_.template alloc_array<double>(n_local_);
  if(exec_space_.mem_backend().is_device()) {
    // Create host mirror if the memory space is on device
    data_host_ = exec_space_host_.template alloc_array<double>(n_local_);
  } else {
    data_host_ = data_dev_;
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>::hiopVectorRaja(const hiopVectorRaja& v)
    : hiopVector(),
      exec_space_(v.exec_space_),
      exec_space_host_(v.exec_space_host_),
      idx_cumsum_{nullptr}
{
  n_local_ = v.n_local_;
  n_ = v.n_;
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;
  mem_space_ = v.mem_space_;

#ifndef HIOP_USE_GPU
  assert(mem_space_ == "HOST");
#endif

  data_dev_ = exec_space_.template alloc_array<double>(n_local_);
  if(exec_space_.mem_backend().is_device()) {
    // Create host mirror if the memory space is on device
    data_host_ = exec_space_host_.template alloc_array<double>(n_local_);
  } else {
    data_host_ = data_dev_;
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>::~hiopVectorRaja()
{
  if(data_dev_ != data_host_) {
    exec_space_host_.dealloc_array(data_host_);
  }
  exec_space_.dealloc_array(data_dev_);
  data_dev_ = nullptr;
  data_host_ = nullptr;
  delete idx_cumsum_;
}

template<class MEM, class POL>
hiopVector* hiopVectorRaja<MEM, POL>::alloc_clone() const
{
  hiopVector* v = new hiopVectorRaja<MEM, POL>(*this);
  assert(v);
  return v;
}
template<class MEM, class POL>
hiopVector* hiopVectorRaja<MEM, POL>::new_copy() const
{
  hiopVector* v = new hiopVectorRaja<MEM, POL>(*this);
  assert(v);
  v->copyFrom(*this);
  return v;
}

//
// Compute kernels
//

/// Set all vector elements to zero
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::setToZero()
{
  setToConstant(0.0);
}

/// Set all vector elements to constant c
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::setToConstant(double c)
{
  double* data = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { data[i] = c; });
}

/// Set selected elements to constant, zero otherwise
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const auto& s = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
  const double* pattern = s.local_data_const();
  double* data = data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { data[i] = pattern[i] * c; });
}

/**
 * @brief Copy data from `vec` to this vector
 *
 * @param[in] vec - Vector from which to copy into `this`
 *
 * @pre `vec` and `this` must have same partitioning.
 * @post Elements of `this` are overwritten with elements of `vec`
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyFrom(const hiopVector& vec)
{
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);
  assert(glob_il_ == v.glob_il_);
  assert(glob_iu_ == v.glob_iu_);

  exec_space_.copy(data_dev_, v.data_dev_, n_local_, v.exec_space_);
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_vectorpar(const hiopVectorPar& v)
{
  assert(n_local_ == v.get_local_size());
  exec_space_.copy(data_dev_, v.local_data_const(), n_local_, v.exec_space());
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_to_vectorpar(hiopVectorPar& v) const
{
  assert(n_local_ == v.get_local_size());
  v.exec_space().copy(v.local_data(), data_dev_, n_local_, exec_space_);
}

/**
 * @brief Copy data from local_array to this vector
 *
 * @param[in] local_array - A raw array from which to copy into `this`
 *
 * @pre `local_array` is allocated by same memory backend and in the same
 * memory space used by `this`.
 * @pre `local_array` must be of same size as the data block of `this`.
 * @post Elements of `this` are overwritten with elements of `local_array`.
 *
 * @warning Method has no way to check for the size of `local_array`. May
 * read past the end of the array.
 *
 * @warning Method casts away const from the `local_array`.
 *
 * @warning Not tested - not part of the hiopVector interface.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyFrom(const double* local_array)
{
  if(local_array) {
    exec_space_.copy(data_dev_, local_array, n_local_);
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src.
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_w_pattern(const hiopVector& vec, const hiopVector& select)
{
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  const auto& ix = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);

  assert(n_local_ == ix.n_local_);

  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(id[i] == one) {
          dd[i] = vd[i];
        }
      });
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_indexes(const hiopVector& vv, const hiopVectorInt& index_in_src)
{
  const auto& indexes = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(index_in_src);
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vv);

  assert(indexes.get_local_size() == n_local_);

  index_type* id = const_cast<index_type*>(indexes.local_data_const());
  double* dd = data_dev_;
  double* vd = v.data_dev_;

  size_type nv = v.get_local_size();

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] < nv);
        dd[i] = vd[id[i]];
      });
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_indexes(const double* vv, const hiopVectorInt& index_in_src)
{
  if(nullptr == vv) {
    return;
  }

  const auto& indexes = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(index_in_src);
  assert(indexes.get_local_size() == n_local_);
  index_type* id = const_cast<index_type*>(indexes.local_data_const());
  double* dd = data_dev_;

  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] = vv[id[i]]; });
}

/**
 * @brief Copy `nv` elements from array `v` to this vector starting from `start_index_in_this`
 *
 * @param[in] start_index_in_this - position in this where to copy
 * @param[in] v  - a raw array from which to copy into `this`
 * @param[in] nv - how many elements of `v` to copy
 *
 * @pre Size of `v` must be >= nv.
 * @pre start_index_in_this+nv <= n_local_
 * @pre `this` is not distributed
 * @pre `v` should be allocated in the memory space/backend of `this`
 *
 * @warning Method casts away const from the `v`.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this + nv <= n_local_);

  // If nothing to copy, return.
  if(nv == 0) return;

  // TODO: data_dev_+start_index_in_this   -> is not portable, may not work on the device. RAJA loop should be used
  exec_space_.copy(data_dev_ + start_index_in_this, v, nv);
}

/**
 * @brief Copy `vec` to this vector starting from `start_index` in `this`.
 *
 * @param[in] start_index - position in `this` where to copy
 * @param[in] src - the source vector from which to copy into `this`
 *
 * @pre Size of `src` must be >= nv.
 * @pre start_index + src.n_local_ <= n_local_
 * @pre `this` is not distributed
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyFromStarting(int start_index, const hiopVector& src)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(src);
  assert(start_index + v.n_local_ <= n_local_);

  // If there is nothing to copy, return.
  if(v.n_local_ == 0) return;

  // TODO: data_dev_+start_index   -> is not portable, may not work on the device. RAJA loop should be used
  exec_space_.copy(data_dev_ + start_index, v.data_dev_, v.n_local_, v.exec_space_);
}

/**
 * @brief Copy `nv` elements from `start_index_in_v` at array `v` to this vector
 *
 * @param[in] start_index_in_v - position in v
 * @param[in] v  - a raw array from which to copy into `this`
 * @param[in] nv - how many elements of `v` to copy
 *
 * @pre Size of `v` must be >= nv.
 * @pre start_index_in_v+nv <= size of 'v'
 * @pre `this` is not distributed
 * @pre `v` should be allocated in the memory space/backend of `this`
 *
 * @warning Method casts away const from the `v`.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_starting_at(const double* v, int start_index_in_v, int nv)
{
  // If nothing to copy, return.
  if(nv == 0) return;

  // TODO: v+start_index_in_v   -> is not portable, may not work on the device. RAJA loop should be used
  exec_space_.copy(data_dev_, v + start_index_in_v, nv);
}

/**
 * @brief Copy from `vec_src` starting at `start_idx_src` into
 * `this` vector starting at `start_idx_dest`.
 *
 * @pre `vec_src` and `this` are not distributed.
 * @pre `start_idx_dest` + `howManyToCopySrc` <= `n_local_`
 * @pre `start_idx_src` + `howManyToCopySrc` <= `vec_src.n_local_`
 * @post Elements of `vec_src` are unchanged.
 * @post All elements of `this` starting from `start_idx_dest` are overwritten
 *
 * @todo Implentation differs from CPU - check with upstream what is correct!
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& vec_src, int start_idx_src)
{
  size_type howManyToCopyDest = this->n_local_ - start_idx_dest;

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif

  assert((start_idx_dest >= 0 && start_idx_dest < this->n_local_) || this->n_local_ == 0);
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec_src);
  assert((start_idx_src >= 0 && start_idx_src < v.n_local_) || v.n_local_ == 0 || v.n_local_ == start_idx_src);
  const size_type howManyToCopySrc = v.n_local_ - start_idx_src;

  if(howManyToCopyDest == 0 || howManyToCopySrc == 0) {
    return;
  }

  assert(howManyToCopyDest <= howManyToCopySrc);

  // TODO: this also looks like is not portable
  exec_space_.copy(data_dev_ + start_idx_dest, v.data_dev_ + start_idx_src, howManyToCopyDest, v.exec_space_);
}

/**
 * @brief Copy to `vec` elements of `this` vector starting from `start_index`.
 *
 * @param[in] start_index - position in `this` from where to copy
 * @param[out] dst - the destination vector where to copy elements of `this`
 *
 * @pre start_index + dst.n_local_ <= n_local_
 * @pre `this` and `dst` are not distributed
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyToStarting(int start_index, hiopVector& dst) const
{
  auto& v = dynamic_cast<hiopVectorRaja<MEM, POL>&>(dst);

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert(start_index + v.n_local_ <= n_local_);

  // If nowhere to copy, return.
  if(v.n_local_ == 0) return;

  // TODO: pointer arithmetic on host should be avoided
  v.exec_space_.copy(v.data_dev_, this->data_dev_ + start_index, v.n_local_, exec_space_);
}

/**
 * @brief Copy elements of `this` vector to `vec` starting at `start_index_in_dest`.
 *
 * @param[out] vec - a vector where to copy elements of `this`
 * @param[in] start_index_in_dest - position in `vec` where to copy
 *
 * @pre start_index_in_dest + vec.n_local_ <= n_local_
 * @pre `this` and `vec` are not distributed
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyToStarting(hiopVector& vec, int start_index_in_dest) const
{
  auto& v = dynamic_cast<hiopVectorRaja<MEM, POL>&>(vec);
  assert(start_index_in_dest + n_local_ <= v.n_local_);

  // If there is nothing to copy, return.
  if(n_local_ == 0) return;

  // TODO: pointer arithmetic on host should be avoided
  v.exec_space_.copy(v.data_dev_ + start_index_in_dest, data_dev_, n_local_, exec_space_);
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyToStartingAt_w_pattern(hiopVector& vec,
                                                          int start_index_in_dest,
                                                          const hiopVector& select) const
{
  if(n_local_ == 0) {
    return;
  }

  hiopVectorRaja& v = dynamic_cast<hiopVectorRaja<MEM, POL>&>(vec);
  const hiopVectorRaja& selected = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
  assert(n_local_ == selected.n_local_);

  double* dd = data_dev_;
  double* vd = v.data_dev_;
  const double* pattern = selected.local_data_const();

  if(nullptr == idx_cumsum_) {
    idx_cumsum_ = LinearAlgebraFactory::create_vector_int(mem_space_, n_local_ + 1);
    index_type* nnz_in_row = idx_cumsum_->local_data();

    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_local_ + 1),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if(i == 0) {
            nnz_in_row[i] = 0;
          } else {
            // from i=1..n
            if(pattern[i - 1] != 0.0) {
              nnz_in_row[i] = 1;
            } else {
              nnz_in_row[i] = 0;
            }
          }
        });
    RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(nnz_in_row, n_local_ + 1),
                                                 RAJA::operators::plus<index_type>());
  }

  index_type* nnz_cumsum = idx_cumsum_->local_data();
  index_type v_n_local = v.n_local_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(1, n_local_ + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(nnz_cumsum[i] != nnz_cumsum[i - 1]) {
          assert(nnz_cumsum[i] == nnz_cumsum[i - 1] + 1);
          index_type idx_dest = nnz_cumsum[i - 1] + start_index_in_dest;
          assert(idx_dest < v_n_local);
          vd[idx_dest] = dd[i - 1];
        }
      });
}

/* copy 'c' and `d` into `this`, according to the map 'c_map` and `d_map`, respectively.
 *  e.g., this[c_map[i]] = c[i];
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                           const hiopVectorInt& c_map,
                                                           const hiopVector& d,
                                                           const hiopVectorInt& d_map)
{
  const auto& v1 = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(c);
  const auto& v2 = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(d);
  const auto& ix1 = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(c_map);
  const auto& ix2 = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(d_map);

  size_type n1_local = v1.n_local_;
  size_type n2_local = v2.n_local_;

#ifdef HIOP_DEEPCHECKS
  assert(n1_local + n2_local == n_local_);
  assert(n_local_ == ix1.get_local_size() + ix2.get_local_size());
#endif
  double* dd = data_dev_;
  double* vd1 = v1.data_dev_;
  double* vd2 = v2.data_dev_;

  const index_type* id1 = ix1.local_data_const();
  const index_type* id2 = ix2.local_data_const();

  int n1_local_int = (int)n1_local;
  int n2_local_int = (int)n2_local;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n1_local_int),
      RAJA_LAMBDA(RAJA::Index_type i) {
        int idx = id1[i];
        dd[idx] = vd1[i];
      });

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n2_local_int),
      RAJA_LAMBDA(RAJA::Index_type i) {
        int idx = id2[i];
        dd[idx] = vd2[i];
      });
}

/* split `this` to `c` and `d`, according to the map 'c_map` and `d_map`, respectively.
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copy_to_two_vec_w_pattern(hiopVector& c,
                                                         const hiopVectorInt& c_map,
                                                         hiopVector& d,
                                                         const hiopVectorInt& d_map) const
{
  const auto& v1 = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(c);
  const auto& v2 = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(d);
  const auto& ix1 = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(c_map);
  const auto& ix2 = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(d_map);

  size_type n1_local = v1.n_local_;
  size_type n2_local = v2.n_local_;

#ifdef HIOP_DEEPCHECKS
  assert(n1_local + n2_local == n_local_);
  assert(n_local_ == ix1.get_local_size() + ix2.get_local_size());
#endif
  double* dd = data_dev_;
  double* vd1 = v1.data_dev_;
  double* vd2 = v2.data_dev_;
  const index_type* id1 = ix1.local_data_const();
  const index_type* id2 = ix2.local_data_const();

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, (int)n1_local),
      RAJA_LAMBDA(RAJA::Index_type i) {
        int idx = id1[i];
        vd1[i] = dd[idx];
      });

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, (int)n2_local),
      RAJA_LAMBDA(RAJA::Index_type i) {
        int idx = id2[i];
        vd2[i] = dd[idx];
      });
}

/**
 * @brief Copy elements of `this` vector to `dest` with offsets.
 *
 * Copy `this` (source) starting at `start_idx_in_src` to `dest`
 * starting at index 'int start_idx_dest'. If num_elems>=0, 'num_elems' will be copied;
 *
 * @param[in] start_idx_in_src - position in `this` from where to copy
 * @param[out] dest - destination vector to where to copy vector data
 * @param[in] start_idx_dest - position in `dest` to where to copy
 * @param[in] num_elems - number of elements to copy
 *
 * @pre start_idx_in_src <= n_local_
 * @pre start_idx_dest   <= dest.n_local_
 * @pre `this` and `dest` are not distributed
 * @post If num_elems >= 0, `num_elems` will be copied
 * @post If num_elems < 0, elements will be copied till the end of
 * either source (`this`) or `dest` is reached
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::startingAtCopyToStartingAt(index_type start_idx_in_src,
                                                          hiopVector& dest,
                                                          index_type start_idx_dest,
                                                          size_type num_elems /* = -1 */) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
#endif

  hiopVectorRaja& dest_raja = dynamic_cast<hiopVectorRaja<MEM, POL>&>(dest);

  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
  assert(start_idx_dest >= 0 && start_idx_dest <= dest_raja.n_local_);

#ifndef NDEBUG
  if(start_idx_dest == dest_raja.n_local_ || start_idx_in_src == this->n_local_) assert((num_elems == -1 || num_elems == 0));
#endif

  if(num_elems < 0) {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest_raja.n_local_ - start_idx_dest);
  } else {
    assert(num_elems + start_idx_in_src <= this->n_local_);
    assert(num_elems + start_idx_dest <= dest_raja.n_local_);
    // make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_ - start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest_raja.n_local_ - start_idx_dest);
  }

  if(num_elems == 0) return;

  // rm.copy(dest.data_dev_ + start_idx_dest, this->data_dev_ + start_idx_in_src, num_elems*sizeof(double));
  // TODO: fix pointer arithmetic on host
  dest_raja.exec_space_.copy(dest_raja.data_dev_ + start_idx_dest, data_dev_ + start_idx_in_src, num_elems, exec_space_);
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                                    hiopVector& destination,
                                                                    index_type start_idx_dest,
                                                                    const hiopVector& selec_dest,
                                                                    size_type num_elems /*=-1*/) const
{
#if 0  
  hiopVectorRaja& dest = dynamic_cast<hiopVectorRaja<MEM, POL>&>(destination);
  const hiopVectorRaja& ix = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(selec_dest);
    
  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
  assert(start_idx_dest   >= 0 && start_idx_dest   <= dest.n_local_);
    
  if(num_elems<0)
  {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  }
  else
  {
    assert(num_elems+start_idx_in_src <= this->n_local_);
    assert(num_elems+start_idx_dest   <= dest.n_local_);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_-start_idx_dest);
  }
      
  int find_nnz = 0;
  double* dd = data_dev_;
  double* vd = dest.data_dev_;
  double* id = ix.data_dev_;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    [&](RAJA::Index_type i)
    {
      assert(id[i] == zero || id[i] == one);
      if(id[i] == one && find_nnz<num_elems)
        vd[start_idx_dest+find_nnz] = dd[ start_idx_in_src + (find_nnz++)];
    });
#else
  assert(false && "not needed / implemented");
#endif
}

/**
 * @brief Copy `this` vector local data to `dest` buffer.
 *
 * @param[out] dest - destination buffer where to copy vector data
 *
 * @pre Size of `dest` must be >= n_local_
 * @pre `dest` should be on the same memory space/backend as `this`
 *
 * @post `this` is not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyTo(double* dest) const
{
  auto* this_nonconst = const_cast<hiopVectorRaja*>(this);
  assert(nullptr != this_nonconst);
  this_nonconst->exec_space_.copy(dest, data_dev_, n_local_);
}

/**
 * @brief L2 vector norm.
 *
 * @post `this` is not modified
 *
 * @todo Consider implementing with BLAS call (<D>NRM2).
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::twonorm() const
{
  double* self_dev = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { sum += self_dev[i] * self_dev[i]; });
  double nrm = sum.get();

#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return std::sqrt(nrm_global);
#endif
  return std::sqrt(nrm);
}

/**
 * @brief scalar (dot) product.
 *
 * @param[in] vec - vector which is scalar-multiplied to `this`.
 *
 * @pre `vec` has same size and partitioning as `this`.
 * @post `this` and `vec` are not modified.
 *
 * @todo Consider implementing with BLAS call (<D>DOT).
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::dotProductWith(const hiopVector& vec) const
{
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);

  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> dot(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dot += dd[i] * vd[i]; });
  double dotprod = dot.get();

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  dotprod = dotprodG;
#endif

  return dotprod;
}

/**
 * @brief L-infinity (max) vector norm.
 *
 * @post `this` is not modified
 *
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::infnorm() const
{
  double nrm = infnorm_local();
#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(MPI_SUCCESS == ierr);
  return nrm_global;
#endif

  return nrm;
}

/**
 * @brief Local L-infinity (max) vector norm.
 *
 * @pre  `this` is not empty vector
 * @post `this` is not modified
 *
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::infnorm_local() const
{
  assert(n_local_ >= 0);
  double* data = data_dev_;
  RAJA::ReduceMax<hiop_raja_reduce, double> norm(0.0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { norm.max(fabs(data[i])); });
  return norm.get();
}

/**
 * @brief 1-norm of `this` vector.
 *
 * @post `this` is not modified
 *
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::onenorm() const
{
  double norm1 = onenorm_local();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&norm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return nrm1_global;
#endif
  return norm1;
}

/**
 * @brief Local 1-norm of `this` vector.
 *
 * @pre  `this` is not empty vector
 * @post `this` is not modified
 *
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::onenorm_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { sum += fabs(data[i]); });
  return sum.get();
}

/**
 * @brief Multiply `this` by `vec` elementwise and store result in `this`.
 *
 * @pre  `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::componentMult(const hiopVector& vec)
{
  const hiopVectorRaja& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] *= vd[i]; });
}

/**
 * @brief Divide `this` vector elemenwise in-place by `vec`.
 *
 * @pre `this` and `vec` have same partitioning.
 * @pre vec[i] != 0 forall i
 * @post `vec` is not modified
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::componentDiv(const hiopVector& vec)
{
  const hiopVectorRaja& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] /= vd[i]; });
}

/**
 * @brief Divide `this` vector elemenwise in-place by `vec`
 * with pattern selection.
 *
 * @pre `this`, `select` and `vec` have same partitioning.
 * @pre vec[i] != 0 when select[i] = 1
 * @post `vec` and `select` are not modified
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::componentDiv_w_selectPattern(const hiopVector& vec, const hiopVector& select)
{
  const hiopVectorRaja& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  const hiopVectorRaja& ix = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(v.n_local_ == n_local_);
  assert(n_local_ == ix.n_local_);
#endif
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] == zero || id[i] == one);
        if(id[i] == zero)
          dd[i] = zero;
        else
          dd[i] /= vd[i];
      });
}

/**
 * @brief Set `this` vector elemenwise to the minimum of itself and the given `constant`
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_min(const double constant)
{
  double* dd = data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(dd[i] > constant) {
          dd[i] = constant;
        }
      });
}

/**
 * @brief Set `this` vector elemenwise to the minimum of itself and the corresponding component of 'vec'.
 *
 * @pre `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_min(const hiopVector& vec)
{
  const auto& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(dd[i] > vd[i]) {
          dd[i] = vd[i];
        }
      });
}

/**
 * @brief Set `this` vector elemenwise to the maximum of itself and the given `constant`
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_max(const double constant)
{
  double* dd = data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(dd[i] < constant) {
          dd[i] = constant;
        }
      });
}

/**
 * @brief Set `this` vector elemenwise to the maximum of itself and the corresponding component of 'vec'.
 *
 * @pre `this` and `vec` have same partitioning.
 * @post `vec` is not modified
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_max(const hiopVector& vec)
{
  const hiopVectorRaja& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(dd[i] < vd[i]) {
          dd[i] = vd[i];
        }
      });
}

/**
 * @brief Set each component to its absolute value
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_abs()
{
  double* dd = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] = fabs(dd[i]); });
}

/**
 * @brief Apply sign function to each component
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_sgn()
{
  double* dd = data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        int sign = (0.0 < dd[i]) - (dd[i] < 0.0);
        dd[i] = static_cast<double>(sign);
      });
}

/**
 * @brief compute square root of each element
 * @pre all the elements are non-negative
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::component_sqrt()
{
  double* dd = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] = sqrt(dd[i]); });
}

/**
 * @brief Scale `this` vector by `c`
 *
 * @note Consider implementing with BLAS call (<D>SCAL)
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::scale(double c)
{
  if(1.0 == c) return;

  double* data = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { data[i] *= c; });
}

/**
 * @brief Implementation of AXPY kernel
 *
 * @pre `this` and `xvec` have same partitioning.
 * @post `xvec` is not modified
 *
 * @note Consider implementing with BLAS call (<D>AXPY)
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axpy(double alpha, const hiopVector& xvec)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);

  double* yd = data_dev_;
  double* xd = x.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        // y := a * x + y
        yd[i] = alpha * xd[i] + yd[i];
      });
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i)
{
  const auto& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const auto& idxs = dynamic_cast<const hiopVectorIntRaja<MEM, POL>&>(i);

  assert(x.get_size() == i.get_local_size());
  assert(x.get_local_size() == i.get_local_size());
  assert(i.get_local_size() <= n_local_);

  double* dd = data_dev_;
  double* xd = const_cast<double*>(x.data_dev_);
  index_type* id = const_cast<index_type*>(idxs.local_data_const());
  auto tmp_n_local = n_local_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] < tmp_n_local);
        // y := a * x + y
        dd[id[i]] = alpha * xd[i] + dd[id[i]];
      });
}

/// @brief Performs axpy, this += alpha*x, for selected entries
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == sel.n_local_);
  assert(n_local_ == sel.n_local_);
#endif
  double* dd = data_dev_;
  const double* xd = x.local_data_const();
  const double* id = sel.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { dd[i] += alpha * xd[i] * id[i]; });
}

/**
 * @brief this[i] += alpha*x[i]*z[i] forall i
 *
 * @pre `this`, `xvec` and `zvec` have same partitioning.
 * @post `xvec` and `zvec` are not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& z = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(n_local_ == z.n_local_);
#endif
  double* dd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { dd[i] += alpha * xd[i] * zd[i]; });
}

/**
 * @brief this[i] += alpha*x[i]/z[i] forall i
 *
 * @pre `this`, `xvec` and `zvec` have same partitioning.
 * @pre zvec[i] != 0 forall i
 * @post `xvec` and `zvec` are not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& z = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(n_local_ == z.n_local_);
#endif
  double* yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { yd[i] += alpha * xd[i] / zd[i]; });
}

/**
 * @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection
 *
 * @pre `this`, `xvec`, `zvec` and `select` have same partitioning.
 * @pre zvec[i] != 0 when select[i] = 1
 * @post `xvec`, `zvec` and `select` are not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::axdzpy_w_pattern(double alpha,
                                                const hiopVector& xvec,
                                                const hiopVector& zvec,
                                                const hiopVector& select)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& z = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(zvec);
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(n_local_ == z.n_local_);
#endif
  double* yd = data_dev_;
  const double* xd = x.local_data_const();
  const double* zd = z.local_data_const();
  const double* id = sel.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] == one || id[i] == zero);
        if(id[i] == one) yd[i] += alpha * xd[i] / zd[i];
      });
}

/**
 * @brief this[i] += c forall i
 *
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::addConstant(double c)
{
  double* yd = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { yd[i] += c; });
}

/**
 * @brief this[i] += c forall i with pattern selection
 *
 * @pre `this` and `select` have same partitioning.
 * @post `select` is not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::addConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
  assert(this->n_local_ == sel.n_local_);
  double* data = data_dev_;
  const double* id = sel.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] == one || id[i] == zero);
        data[i] += id[i] * c;
      });
}

/// Find minimum vector element
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::min() const
{
  double* data = data_dev_;
  RAJA::ReduceMin<hiop_raja_reduce, double> minimum(std::numeric_limits<double>::max());
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { minimum.min(data[i]); });
  double ret_val = minimum.get();

#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr = MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

/// Find minimum vector element for `select` pattern
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::min_w_pattern(const hiopVector& select) const
{
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
  assert(this->n_local_ == sel.n_local_);
  double* data = data_dev_;
  const double* id = sel.local_data_const();

  RAJA::ReduceMin<hiop_raja_reduce, double> minimum(std::numeric_limits<double>::max());
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(id[i] == one) {
          minimum.min(data[i]);
        }
      });
  double ret_val = minimum.get();

#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr = MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

/// Find minimum vector element
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::min(double& /* m */, int& /* index */) const
{
  assert(false && "not implemented");
}

/**
 * @brief Negate all vector elements
 *
 * @note Consider implementing with BLAS call (<D>SCAL)
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::negate()
{
  double* data = data_dev_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { data[i] *= -1; });
}

/**
 * @brief Invert vector elements
 *
 * @pre this[i] != 0 forall i
 * @post `this` is overwritten
 *
 * @todo Consider having HiOp-wide `small_real` constant defined.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::invert()
{
#ifdef HIOP_DEEPCHECKS
#ifndef NDEBUG
  const double small_real = 1e-35;
#endif
#endif
  double* data = data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
#ifdef HIOP_DEEPCHECKS
        assert(fabs(data[i]) > small_real);
#endif
        data[i] = one / data[i];
      });
}

/**
 * @brief Sum all selected log(this[i])
 *
 * @pre `this` and `select` have same partitioning.
 * @pre Selected elements of `this` are > 0.
 * @post `this` and `select` are not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::logBarrier_local(const hiopVector& select) const
{
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
  assert(this->n_local_ == sel.n_local_);

  double* data = data_dev_;
  const double* id = sel.local_data_const();
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
#ifdef HIOP_DEEPCHECKS
        assert(id[i] == one || id[i] == zero);
#endif
        if(id[i] == one) sum += std::log(data[i]);
      });

  return sum.get();
}

/**
 * @brief Sum all elements
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::sum_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { sum += data[i]; });

  return sum.get();
}

/**
 * @brief adds the gradient of the log barrier, namely this[i]=this[i]+alpha*1/select(x[i])
 *
 * @pre `this`, `xvec` and `select` have same partitioning.
 * @pre xvec[i] != 0 forall i
 * @post `xvec` and `select` are not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::addLogBarrierGrad(double alpha, const hiopVector& xvec, const hiopVector& select)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& sel = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == x.n_local_);
  assert(n_local_ == sel.n_local_);
#endif
  double* data = data_dev_;
  const double* xd = x.local_data_const();
  const double* id = sel.local_data_const();
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(id[i] == 1.0) data[i] += alpha / xd[i];
      });
}

/**
 * @brief Linear damping term
 *
 * @pre `this`, `ixleft` and `ixright` have same partitioning.
 * @pre `ixleft` and `ixright` elements are 0 or 1 only.
 * @post `this`, `ixleft` and `ixright` are not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::linearDampingTerm_local(const hiopVector& ixleft,
                                                         const hiopVector& ixright,
                                                         const double& mu,
                                                         const double& kappa_d) const
{
  const hiopVectorRaja& ixl = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixleft);
  const hiopVectorRaja& ixr = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixright);
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == ixl.n_local_);
  assert(n_local_ == ixr.n_local_);
#endif
  const double* ld = ixl.local_data_const();
  const double* rd = ixr.local_data_const();
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(zero);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(ld[i] == one && rd[i] == zero) sum += data[i];
      });
  double term = sum.get();
  term *= mu;
  term *= kappa_d;
  return term;
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::addLinearDampingTerm(const hiopVector& ixleft,
                                                    const hiopVector& ixright,
                                                    const double& alpha,
                                                    const double& ct)
{
  assert((dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixleft)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixright)).n_local_ == n_local_);

  const double* ixl = (dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixleft)).local_data_const();
  const double* ixr = (dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixright)).local_data_const();

  double* data = data_dev_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        // y := a * x + ...
        data[i] = alpha * data[i] + ct * (ixl[i] - ixr[i]);
      });
}

/**
 * @brief Check if all elements of the vector are positive
 *
 * @post `this` is not modified
 */
template<class MEM, class POL>
int hiopVectorRaja<MEM, POL>::allPositive()
{
  double* data = data_dev_;
  RAJA::ReduceMin<hiop_raja_reduce, double> minimum(one);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { minimum.min(data[i]); });
  int allPos = minimum.get() > zero ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

/**
 * @brief Project solution into bounds
 *
 * @pre `this`, `xlo`, `ixl`, `xup` and `ixu` have same partitioning.
 * @pre `ixl` and `ixu` elements are 0 or 1 only.
 * @post `xlo`, `ixl`, `xup` and `ixu` are not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::projectIntoBounds_local(const hiopVector& xlo,
                                                       const hiopVector& ixl,
                                                       const hiopVector& xup,
                                                       const hiopVector& ixu,
                                                       double kappa1,
                                                       double kappa2)
{
  const hiopVectorRaja& xl = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xlo);
  const hiopVectorRaja& il = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixl);
  const hiopVectorRaja& xu = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xup);
  const hiopVectorRaja& iu = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixu);

#ifdef HIOP_DEEPCHECKS
  assert(xl.n_local_ == n_local_);
  assert(il.n_local_ == n_local_);
  assert(xu.n_local_ == n_local_);
  assert(iu.n_local_ == n_local_);
#endif

  const double* xld = xl.local_data_const();
  const double* ild = il.local_data_const();
  const double* xud = xu.local_data_const();
  const double* iud = iu.local_data_const();
  double* xd = data_dev_;
  // Perform preliminary check to see of all upper value
  RAJA::ReduceMin<hiop_raja_reduce, double> minimum(one);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { minimum.min(xud[i] - xld[i]); });
  if(minimum.get() < zero) return false;

  const double small_real = std::numeric_limits<double>::min() * 100;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        double aux = zero;
        double aux2 = zero;
        if(ild[i] != zero && iud[i] != zero) {
          aux = kappa2 * (xud[i] - xld[i]) - small_real;
          aux2 = xld[i] + fmin(kappa1 * fmax(one, fabs(xld[i])), aux);
          if(xd[i] < aux2) {
            xd[i] = aux2;
          } else {
            aux2 = xud[i] - fmin(kappa1 * fmax(one, fabs(xud[i])), aux);
            if(xd[i] > aux2) {
              xd[i] = aux2;
            }
          }
#ifdef HIOP_DEEPCHECKS
          assert(xd[i] > xld[i] && xd[i] < xud[i] && "this should not happen -> HiOp bug");
#endif
        } else {
          if(ild[i] != zero)
            xd[i] = fmax(xd[i], xld[i] + kappa1 * fmax(one, fabs(xld[i])) - small_real);
          else if(iud[i] != zero)
            xd[i] = fmin(xd[i], xud[i] - kappa1 * fmax(one, fabs(xud[i])) - small_real);
          else { /*nothing for free vars  */
          }
        }
      });
  return true;
}

/**
 * @brief max{a\in(0,1]| x+ad >=(1-tau)x}
 *
 * @pre `this` and `dvec` have same partitioning.
 * @post `this` and `dvec` are not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const
{
  const hiopVectorRaja& d = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(dvec);
#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1);  // TODO: per documentation above it should be tau <= 1 (?).
#endif

  const double* dd = d.local_data_const();
  const double* xd = data_dev_;

  RAJA::ReduceMin<hiop_raja_reduce, double> minimum(one);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(dd[i] >= zero) return;
#ifdef HIOP_DEEPCHECKS
        assert(xd[i] > zero);
#endif
        minimum.min(-tau * xd[i] / dd[i]);
      });
  return minimum.get();
}

/**
 * @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select
 *
 * @pre `this`, `select` and `dvec` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `this`, `select` and `dvec` are not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
double hiopVectorRaja<MEM, POL>::fractionToTheBdry_w_pattern_local(const hiopVector& dvec,
                                                                   const double& tau,
                                                                   const hiopVector& select) const
{
  const hiopVectorRaja& d = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(dvec);
  const hiopVectorRaja& s = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);

#ifdef HIOP_DEEPCHECKS
  assert(d.n_local_ == n_local_);
  assert(s.n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1);
#endif
  const double* dd = d.local_data_const();
  const double* xd = data_dev_;
  const double* id = s.local_data_const();

  RAJA::ReduceMin<hiop_raja_reduce, double> aux(one);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(id[i] == one || id[i] == zero);
        if(dd[i] < 0 && id[i] == one) {
#ifdef HIOP_DEEPCHECKS
          assert(xd[i] > 0);
#endif
          aux.min(-tau * xd[i] / dd[i]);
        }
      });
  return aux.get();
}

/**
 * @brief Set elements of `this` to zero based on `select`.
 *
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::selectPattern(const hiopVector& select)
{
  const hiopVectorRaja& s = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(s.n_local_ == n_local_);
#endif

  double* data = data_dev_;
  double* sd = s.data_dev_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(sd[i] == zero) data[i] = zero;
      });
}

/**
 * @brief Checks if `this` matches nonzero pattern of `select`.
 *
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::matchesPattern(const hiopVector& pattern)
{
  const hiopVectorRaja& p = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(pattern);

#ifdef HIOP_DEEPCHECKS
  assert(p.n_local_ == n_local_);
#endif

  double* data = data_dev_;
  double* pd = p.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { sum += (data[i] != 0.0 && pd[i] == 0.0); });
  int mismatch = sum.get();

#ifdef HIOP_USE_MPI
  int mismatch_glob = mismatch;
  int ierr = MPI_Allreduce(&mismatch, &mismatch_glob, 1, MPI_INT, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return (mismatch_glob == 0);
#endif
  return (mismatch == 0);
}

/**
 * @brief Checks if selected elements of `this` are positive.
 *
 * @pre `this` and `select` have same partitioning.
 * @pre Elements of `select` are either 0 or 1.
 * @post `select` is not modified
 */
template<class MEM, class POL>
int hiopVectorRaja<MEM, POL>::allPositive_w_patternSelect(const hiopVector& select)
{
  const hiopVectorRaja& w = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(select);

#ifdef HIOP_DEEPCHECKS
  assert(w.n_local_ == n_local_);
#endif

  const double* wd = w.local_data_const();
  const double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(wd[i] != zero && data[i] <= zero) sum += 1;
      });
  int allPos = (sum.get() == 0);

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

/**
 * @brief Adjusts duals.
 *
 * @pre `this`, `xvec` and `ixvec` have same partitioning.
 * @pre Elements of `ixvec` are either 0 or 1.
 * @post `xvec` and `ixvec` are not modified
 *
 * @note Implementation probably inefficient.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::adjustDuals_plh(const hiopVector& xvec,
                                               const hiopVector& ixvec,
                                               const double& mu,
                                               const double& kappa)
{
  const hiopVectorRaja& x = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(xvec);
  const hiopVectorRaja& ix = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(ixvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == n_local_);
  assert(ix.n_local_ == n_local_);
#endif
  const double* xd = x.local_data_const();
  const double* id = ix.local_data_const();
  double* z = data_dev_;  // the dual

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        double a, b;
        // preemptive loop to reduce number of iterations?
        if(id[i] == 1.) {
          // precompute a and b in another loop?
          a = mu / xd[i];
          b = a / kappa;
          a = a * kappa;
          // Necessary conditionals
          if(z[i] < b)
            z[i] = b;
          else  // z[i]>=b
            if(a <= b)
              z[i] = b;
            else  // a>b
              if(a < z[i]) z[i] = a;
          // - - - -
          // else a>=z[i] then *z=*z (z[i] does not need adjustment)
        }
      });
}

/**
 * @brief Check if all elements of the vector are zero
 *
 * @post `this` is not modified
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::is_zero() const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(data[i] != 0.0) {
          sum += 1;
        }
      });
  int all_zero = (sum.get() == 0) ? 1 : 0;

#ifdef HIOP_USE_MPI
  int all_zero_G;
  int ierr = MPI_Allreduce(&all_zero, &all_zero_G, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return all_zero_G;
#endif
  return all_zero;
}

/**
 * @brief Returns true if any element of `this` is NaN.
 *
 * @post `this` is not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::isnan_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(std::isnan(data[i])) any += 1;
      });
  return any.get();
}

/**
 * @brief Returns true if any element of `this` is Inf.
 *
 * @post `this` is not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::isinf_local() const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(std::isinf(data[i])) any += 1;
      });
  return any.get();
}

/**
 * @brief Returns true if all elements of `this` are finite.
 *
 * @post `this` is not modified
 *
 * @warning This is local method only!
 */
template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::isfinite_local() const
{
  double* data = data_dev_;
  RAJA::ReduceMin<hiop_raja_reduce, int> smallest(1);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(!std::isfinite(data[i])) smallest.min(0);
      });
  return smallest.get();
}

/**
 * @brief Prints vector data to a file in Matlab format.
 *
 * @pre Vector data was moved from the memory space to the host mirror.
 */
template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::print(FILE* file, const char* msg /*=NULL*/, int max_elems /*=-1*/, int rank /*=-1*/) const
{
  int myrank = 0, numranks = 1;
#ifdef HIOP_USE_MPI
  if(rank >= 0) {
    int err = MPI_Comm_rank(comm_, &myrank);
    assert(err == MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &numranks);
    assert(err == MPI_SUCCESS);
  }
#endif

  if(nullptr == file) {
    file = stdout;
  }

  if(myrank == rank || rank == -1) {
    if(max_elems > n_local_) max_elems = n_local_;

    if(NULL == msg) {
      std::stringstream ss;
      ss << "vector of size " << n_ << ", printing " << max_elems << " elems ";
      if(numranks > 1) {
        ss << "(on rank=" << myrank << ")";
      } else {
        ss << "(serial)";
      }
      ss << "\n";
      fprintf(file, "%s", ss.str().c_str());
    } else {
      fprintf(file, "%s ", msg);
    }
    fprintf(file, "=[");
    max_elems = max_elems >= 0 ? max_elems : n_local_;
    for(int it = 0; it < max_elems; it++) fprintf(file, "%22.16e ; ", data_host_[it]);
    fprintf(file, "];\n");
  }
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::print() const
{
  auto* inst = const_cast<hiopVectorRaja<MEM, POL>*>(this);
  assert(nullptr != inst);
  inst->copyFromDev();
  for(index_type it = 0; it < n_local_; it++) {
    printf("vec [%d] = %1.16e\n", it + 1, data_host_[it]);
  }
  printf("\n");
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyToDev()
{
  if(data_dev_ == data_host_) return;
  assert(exec_space_.mem_backend().is_device() && "should have data_dev_==data_host_");
  exec_space_.copy(data_dev_, data_host_, n_local_, exec_space_host_);
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::copyFromDev()
{
  if(data_dev_ == data_host_) return;
  exec_space_host_.copy(data_host_, data_dev_, n_local_, exec_space_);
}

template<class MEM, class POL>
size_type hiopVectorRaja<MEM, POL>::numOfElemsLessThan(const double& val) const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_), RAJA_LAMBDA(RAJA::Index_type i) { sum += (data[i] < val); });

  size_type nrm = sum.get();

#ifdef HIOP_USE_MPI
  size_type nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = nrm_global;
#endif

  return nrm;
}

template<class MEM, class POL>
size_type hiopVectorRaja<MEM, POL>::numOfElemsAbsLessThan(const double& val) const
{
  double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) { sum += static_cast<size_type>(fabs(data[i]) < val); });

  size_type nrm = sum.get();

#ifdef HIOP_USE_MPI
  size_type nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = nrm_global;
#endif

  return nrm;
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                                 const int start,
                                                 const int end,
                                                 const hiopInterfaceBase::NonlinearityType* arr_src,
                                                 const int start_src) const
{
  assert(end <= n_local_ && start <= end && start >= 0 && start_src >= 0);

  // If there is nothing to copy, return.
  if(end - start == 0) return;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, end - start),
      RAJA_LAMBDA(RAJA::Index_type i) { arr[start + i] = arr_src[start_src + i]; });
}

template<class MEM, class POL>
void hiopVectorRaja<MEM, POL>::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                                 const int start,
                                                 const int end,
                                                 const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(end <= n_local_ && start <= end && start >= 0);

  // If there is nothing to copy, return.
  if(end - start == 0) return;

  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(start, end), RAJA_LAMBDA(RAJA::Index_type i) { arr[i] = arr_src; });
}

template<class MEM, class POL>
bool hiopVectorRaja<MEM, POL>::is_equal(const hiopVector& vec) const
{
#ifdef HIOP_DEEPCHECKS
  const hiopVectorRaja& v = dynamic_cast<const hiopVectorRaja<MEM, POL>&>(vec);
  assert(v.n_local_ == n_local_);
#endif

  const double* data_v = vec.local_data_const();
  const double* data = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(data[i] != data_v[i]) {
          sum += 1;
        }
      });
  int all_equal = (sum.get() == 0);

#ifdef HIOP_USE_MPI
  int all_equalG;
  int ierr = MPI_Allreduce(&all_equal, &all_equalG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return all_equalG;
#endif
  return all_equal;
}

}  // namespace hiop
