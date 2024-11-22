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

#include "hiopVectorPar.hpp"
#include "MemBackendCppImpl.hpp"
#include "hiopVectorIntSeq.hpp"
#include "hiopCppStdUtils.hpp"

#include <cmath>
#include <cstring>  //for memcpy
#include <algorithm>
#include <cassert>
#include <iostream>

#include "hiop_blasdefs.hpp"
#include "MathKernelsHost.hpp"

#include <limits>
#include <cstddef>

namespace hiop
{

hiopVectorPar::hiopVectorPar(const size_type& glob_n, index_type* col_part /*=NULL*/, MPI_Comm comm /*=MPI_COMM_NULL*/)
    : comm_(comm)
{
  n_ = glob_n;
  assert(n_ >= 0);
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

  data_ = exec_space_.template alloc_array<double>(n_local_);
}

/// internal use only: allocates data_
hiopVectorPar::hiopVectorPar(const hiopVectorPar& v)
{
  n_local_ = v.n_local_;
  n_ = v.n_;
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;
  data_ = exec_space_.template alloc_array<double>(n_local_);
}

hiopVectorPar::~hiopVectorPar()
{
  exec_space_.dealloc_array(data_);
  data_ = nullptr;
}

hiopVector* hiopVectorPar::alloc_clone() const
{
  hiopVector* v = new hiopVectorPar(*this);
  assert(v);
  return v;
}
hiopVector* hiopVectorPar::new_copy() const
{
  hiopVector* v = new hiopVectorPar(*this);
  assert(v);
  v->copyFrom(*this);
  return v;
}

void hiopVectorPar::setToZero()
{
  for(int i = 0; i < n_local_; i++) {
    data_[i] = 0.0;
  }
}
void hiopVectorPar::setToConstant(double c)
{
  for(int i = 0; i < n_local_; i++) {
    data_[i] = c;
  }
}

void hiopVectorPar::set_to_random_uniform(double minv, double maxv)
{
  hiop::host::array_random_uniform_kernel(n_local_, data_, minv, maxv);
}

void hiopVectorPar::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorPar& s = dynamic_cast<const hiopVectorPar&>(select);
  const double* svec = s.data_;
  for(int i = 0; i < n_local_; i++) {
    if(svec[i] == 1.) {
      data_[i] = c;
    } else {
      data_[i] = 0.;
    }
  }
}
void hiopVectorPar::copyFrom(const hiopVector& v_in)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_in);
  copy_from_vectorpar(v);
}

void hiopVectorPar::copy_from_vectorpar(const hiopVectorPar& v)
{
  assert(n_local_ == v.n_local_);
  assert(glob_il_ == v.glob_il_);
  assert(glob_iu_ == v.glob_iu_);
  exec_space_.copy(this->data_, v.data_, n_local_, v.exec_space_);
}

void hiopVectorPar::copy_to_vectorpar(hiopVectorPar& vdest) const { vdest.copy_from_vectorpar(*this); }

void hiopVectorPar::copyFrom(const double* v_local_data)
{
  if(v_local_data) {
    exec_space_.copy(this->data_, v_local_data, n_local_);
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorPar::copy_from_w_pattern(const hiopVector& vv, const hiopVector& select)
{
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(select);
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(vv);

  assert(n_local_ == ix.n_local_);
  const double* ix_vec = ix.data_;
  const double* v_vec = v.data_;

  for(index_type i = 0; i < n_local_; i++) {
    if(ix_vec[i] == 1.) {
      data_[i] = v_vec[i];
    }
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorPar::copy_from_indexes(const hiopVector& vv, const hiopVectorInt& index_in_src)
{
  const hiopVectorIntSeq& indexes = dynamic_cast<const hiopVectorIntSeq&>(index_in_src);
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(vv);

  assert(indexes.get_local_size() == n_local_);

  const index_type* index_arr = indexes.local_data_const();
  size_type nv = v.get_local_size();
  for(index_type i = 0; i < n_local_; i++) {
    assert(index_arr[i] < nv);
    this->data_[i] = v.data_[index_arr[i]];
  }
}

/// @brief Copy from vec the elements specified by the indices in index_in_src
void hiopVectorPar::copy_from_indexes(const double* vv, const hiopVectorInt& index_in_src)
{
  if(vv) {
    const hiopVectorIntSeq& indexes = dynamic_cast<const hiopVectorIntSeq&>(index_in_src);
    const index_type* index_arr = indexes.local_data_const();
    assert(indexes.get_local_size() == n_local_);
    for(int i = 0; i < n_local_; i++) {
      this->data_[i] = vv[index_arr[i]];
    }
  }
}

void hiopVectorPar::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this + nv <= n_local_);
  exec_space_.copy(data_ + start_index_in_this, v, nv);
}

void hiopVectorPar::copyFromStarting(int start_index /*_in_src*/, const hiopVector& v_)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
#endif
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(start_index + v.n_local_ <= n_local_);
  exec_space_.copy(data_ + start_index, v.data_, v.n_local_, v.exec_space_);
}

void hiopVectorPar::copy_from_starting_at(const double* v, int start_index_in_global_v, int global_nv)
{
  //  exec_space_.copy(data_, v+start_index_in_global_v, global_nv);
  //  return;

  if(n_local_ > 0) {
    if(global_nv == n_local_ && n_local_ == n_) {
      exec_space_.copy(data_, v + start_index_in_global_v, global_nv);
    } else {
      if(glob_il_ <= start_index_in_global_v + global_nv && glob_iu_ >= start_index_in_global_v) {
        int local_start_idx = (glob_il_ >= start_index_in_global_v) ? 0 : start_index_in_global_v - glob_il_;
        int local_end_idx =
            (glob_iu_ <= start_index_in_global_v + global_nv) ? glob_iu_ - glob_il_ : start_index_in_global_v - glob_il_;
        exec_space_.copy(data_, v + local_start_idx, local_end_idx - local_start_idx);
      }
    }
  }
}

void hiopVectorPar::startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v_in, int start_idx_src)
{
  size_type howManyToCopyDest = this->n_local_ - start_idx_dest;
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_in);

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
  assert(v.n_local_ == v.n_ && "only for local/non-distributed vectors");
#endif
  assert((start_idx_dest >= 0 && start_idx_dest < this->n_local_) || this->n_local_ == 0);
  assert((start_idx_src >= 0 && start_idx_src < v.n_local_) || v.n_local_ == 0 || v.n_local_ == start_idx_src);
  size_type howManyToCopySrc = v.n_local_ - start_idx_src;

  if(howManyToCopyDest == 0 || howManyToCopySrc == 0) {
    return;
  }

  assert(howManyToCopyDest <= howManyToCopySrc);

  // just to be safe when not NDEBUG
  if(howManyToCopyDest > howManyToCopySrc) howManyToCopyDest = howManyToCopySrc;

  assert(howManyToCopyDest >= 0);
  exec_space_.copy(data_ + start_idx_dest, v.data_ + start_idx_src, howManyToCopyDest, v.exec_space_);
}

void hiopVectorPar::copyToStarting(int start_index, hiopVector& v_) const
{
  hiopVectorPar& v = dynamic_cast<hiopVectorPar&>(v_);
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert(start_index + v.n_local_ <= n_local_);
  v.exec_space_.copy(v.data_, data_ + start_index, v.n_local_, exec_space_);
}
/* Copy 'this' to dest starting at start_index in 'dest'. */
void hiopVectorPar::copyToStarting(hiopVector& dest, int start_global_index_in_dest) const
{
  hiopVectorPar& v = dynamic_cast<hiopVectorPar&>(dest);
  //  v.exec_space_.copy(v.data_+start_global_index_in_dest, data_, n_local_, exec_space_);
  //  return;
  assert(start_global_index_in_dest + this->get_size() <= v.get_size());

  if(n_local_ > 0) {
    size_type start_local_index_in_dest = 0;
    if(v.glob_il_ <= start_global_index_in_dest && start_global_index_in_dest < v.glob_iu_) {
      start_local_index_in_dest = start_global_index_in_dest - v.glob_il_;
    } else if(start_global_index_in_dest >= v.glob_iu_) {
      start_local_index_in_dest = -1;
    }
    if(start_local_index_in_dest >= 0) {
      assert(n_local_ + start_local_index_in_dest <= v.n_local_);
      v.exec_space_.copy(v.data_ + start_local_index_in_dest, data_, n_local_, exec_space_);
    }
  }
}

void hiopVectorPar::copyToStartingAt_w_pattern(hiopVector& v_,
                                               index_type start_index /*_in_dest*/,
                                               const hiopVector& select) const
{
  hiopVectorPar& v = dynamic_cast<hiopVectorPar&>(v_);
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(select);
  assert(n_local_ == ix.n_local_);
  const double* ix_vec = ix.data_;
  int find_nnz = 0;

  for(index_type i = 0; i < n_local_; i++)
    if(ix_vec[i] == 1.) {
      v.data_[start_index + (find_nnz++)] = data_[i];
    }
  assert(find_nnz + start_index <= v.n_local_);
}

/* copy 'c' and `d` into `this`, according to the map 'c_map` and `d_map`, respectively.
 *  e.g., this[c_map[i]] = c[i];
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
void hiopVectorPar::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                const hiopVectorInt& c_map,
                                                const hiopVector& d,
                                                const hiopVectorInt& d_map)
{
  const double* c_arr = c.local_data_const();
  const double* d_arr = d.local_data_const();
  double* arr = local_data();
  const int c_size = c.get_size();
  assert(c_size == c_map.get_local_size());
  const int d_size = d.get_size();
  assert(d_size == d_map.get_local_size());
  assert(c_size + d_size == n_local_);

  // concatanate multipliers -> copy into whole lambda array
  for(int i = 0; i < c_size; ++i) {
    arr[c_map.local_data_host_const()[i]] = c_arr[i];
  }
  for(int i = 0; i < d_size; ++i) {
    arr[d_map.local_data_host_const()[i]] = d_arr[i];
  }
}

/* split `this` to `c` and `d`, according to the map 'c_map` and `d_map`, respectively.
 *
 *  @pre the size of `this` = the size of `c` + the size of `d`.
 *  @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
 */
void hiopVectorPar::copy_to_two_vec_w_pattern(hiopVector& c,
                                              const hiopVectorInt& c_map,
                                              hiopVector& d,
                                              const hiopVectorInt& d_map) const
{
  double* c_arr = c.local_data();
  double* d_arr = d.local_data();
  const double* arr = local_data_const();
  const int c_size = c.get_size();
  assert(c_size == c_map.get_local_size());
  const int d_size = d.get_size();
  assert(d_size == d_map.get_local_size());
  assert(c_size + d_size == n_local_);

  // concatanate multipliers -> copy into whole lambda array
  for(int i = 0; i < c_size; ++i) {
    c_arr[i] = arr[c_map.local_data_host_const()[i]];
  }
  for(int i = 0; i < d_size; ++i) {
    d_arr[i] = arr[d_map.local_data_host_const()[i]];
  }
}

/* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest'
 * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
 * either source ('this') or destination ('dest') is reached */
void hiopVectorPar::startingAtCopyToStartingAt(index_type start_idx_in_src,
                                               hiopVector& dest_,
                                               index_type start_idx_dest,
                                               size_type num_elems /*=-1*/) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
#endif
  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
#ifndef NDEBUG
  if(start_idx_in_src == this->n_local_) assert((num_elems == -1 || num_elems == 0));
#endif
  hiopVectorPar& dest = dynamic_cast<hiopVectorPar&>(dest_);
  assert(start_idx_dest >= 0 && start_idx_dest <= dest.n_local_);
#ifndef NDEBUG
  if(start_idx_dest == dest.n_local_) assert((num_elems == -1 || num_elems == 0));
#endif
  if(num_elems < 0) {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  } else {
    assert(num_elems + start_idx_in_src <= this->n_local_);
    assert(num_elems + start_idx_dest <= dest.n_local_);
    // make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_ - start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_ - start_idx_dest);
  }

  dest.exec_space_.copy(dest.data_ + start_idx_dest, this->data_ + start_idx_in_src, num_elems, this->exec_space_);
}

void hiopVectorPar::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                         hiopVector& dest_,
                                                         index_type start_idx_dest,
                                                         const hiopVector& selec_dest,
                                                         size_type num_elems /*=-1*/) const
{
  assert(start_idx_in_src >= 0 && start_idx_in_src <= n_local_);
  hiopVectorPar& dest = dynamic_cast<hiopVectorPar&>(dest_);
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(selec_dest);
  assert(start_idx_dest >= 0 && start_idx_dest <= dest.n_local_);
  if(num_elems < 0) {
    num_elems = std::min(n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  } else {
    assert(num_elems + start_idx_in_src <= this->n_local_);
    assert(num_elems + start_idx_dest <= dest.n_local_);
    // make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_ - start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_ - start_idx_dest);
  }
  int find_nnz = 0;
  const double* ix_vec = ix.data_;
  for(int i = start_idx_dest; i < dest.n_local_ && find_nnz < num_elems; i++)
    if(ix_vec[i] == 1.) {
      dest.data_[i] = data_[start_idx_in_src + find_nnz];
      find_nnz++;
    }
}

void hiopVectorPar::copyTo(double* dest) const
{
  ExecSpace<MemBackendCpp, ExecPolicySeq> exec_space_dest;
  exec_space_dest.copy(dest, this->data_, n_local_, exec_space_);
}

double hiopVectorPar::twonorm() const
{
  int one = 1;
  int n = n_local_;
  double nrm = 0.;
  if(n > 0) {
    nrm = DNRM2(&n, data_, &one);
  }

#ifdef HIOP_USE_MPI
  nrm *= nrm;
  double nrmG;
  int ierr = MPI_Allreduce(&nrm, &nrmG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = sqrt(nrmG);
#endif
  return nrm;
}

double hiopVectorPar::dotProductWith(const hiopVector& v_) const
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  int one = 1;
  int n = n_local_;
  assert(this->n_local_ == v.n_local_);

  double dotprod;
  if(n > 0) {
    dotprod = DDOT(&n, this->data_, &one, v.data_, &one);
  } else {
    dotprod = 0.;
  }
#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  dotprod = dotprodG;
#endif

  return dotprod;
}

double hiopVectorPar::infnorm() const
{
  assert(n_local_ >= 0);
  double nrm = 0.;
  if(n_local_ != 0) {
    nrm = fabs(data_[0]);
    double aux;

    for(int i = 1; i < n_local_; i++) {
      aux = fabs(data_[i]);
      if(aux > nrm) nrm = aux;
    }
  }
#ifdef HIOP_USE_MPI
  double nrm_glob;
  int ierr = MPI_Allreduce(&nrm, &nrm_glob, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(MPI_SUCCESS == ierr);
  return nrm_glob;
#endif

  return nrm;
}

double hiopVectorPar::infnorm_local() const
{
  assert(n_local_ >= 0);
  double nrm = 0.;
  if(n_local_ > 0) {
    nrm = fabs(data_[0]);
    double aux;

    for(int i = 1; i < n_local_; i++) {
      aux = fabs(data_[i]);
      if(aux > nrm) nrm = aux;
    }
  }
  return nrm;
}

double hiopVectorPar::onenorm() const
{
  double nrm1 = 0.;
  for(int i = 0; i < n_local_; i++) {
    nrm1 += fabs(data_[i]);
  }
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return nrm1_global;
#endif
  return nrm1;
}

double hiopVectorPar::onenorm_local() const
{
  double nrm1 = 0.;
  for(int i = 0; i < n_local_; i++) {
    nrm1 += fabs(data_[i]);
  }
  return nrm1;
}

void hiopVectorPar::componentMult(const hiopVector& v_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local_ == v.n_local_);
  for(int i = 0; i < n_local_; ++i) {
    data_[i] *= v.data_[i];
  }
}

void hiopVectorPar::componentDiv(const hiopVector& v_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local_ == v.n_local_);
  for(int i = 0; i < n_local_; i++) data_[i] /= v.data_[i];
}

void hiopVectorPar::componentDiv_w_selectPattern(const hiopVector& v_, const hiopVector& ix_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(ix_);
#ifdef HIOP_DEEPCHECKS
  assert(v.n_local_ == n_local_);
  assert(n_local_ == ix.n_local_);
#endif
  double *s = this->data_, *x = v.data_, *pattern = ix.data_;
  for(int i = 0; i < n_local_; i++)
    if(pattern[i] == 0.0)
      s[i] = 0.0;
    else
      s[i] /= x[i];
}

void hiopVectorPar::component_min(const double constant)
{
  for(int i = 0; i < n_local_; i++) {
    if(data_[i] > constant) {
      data_[i] = constant;
    }
  }
}

void hiopVectorPar::component_min(const hiopVector& v_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local_ == v.n_local_);
  for(int i = 0; i < n_local_; i++) {
    if(data_[i] > v.data_[i]) {
      data_[i] = v.data_[i];
    }
  }
}

void hiopVectorPar::component_max(const double constant)
{
  for(int i = 0; i < n_local_; i++) {
    if(data_[i] < constant) {
      data_[i] = constant;
    }
  }
}

void hiopVectorPar::component_max(const hiopVector& v_)
{
  const hiopVectorPar& v = dynamic_cast<const hiopVectorPar&>(v_);
  assert(n_local_ == v.n_local_);
  for(int i = 0; i < n_local_; i++) {
    if(data_[i] < v.data_[i]) {
      data_[i] = v.data_[i];
    }
  }
}

void hiopVectorPar::component_abs()
{
  for(int i = 0; i < n_local_; i++) {
    data_[i] = fabs(data_[i]);
  }
}

void hiopVectorPar::component_sgn()
{
  for(int i = 0; i < n_local_; i++) {
    int sign = (0.0 < data_[i]) - (data_[i] < 0.0);
    data_[i] = static_cast<double>(sign);
  }
}

void hiopVectorPar::component_sqrt()
{
  for(int i = 0; i < n_local_; i++) {
    assert(data_[i] >= 0);
    data_[i] = std::sqrt(data_[i]);
  }
}

void hiopVectorPar::scale(double num)
{
  if(1.0 == num) return;
  int one = 1;
  int n = n_local_;
  DSCAL(&n, &num, data_, &one);
}

void hiopVectorPar::axpy(double alpha, const hiopVector& x_in)
{
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_in);
  int one = 1;
  int n = n_local_;
  DAXPY(&n, &alpha, x.data_, &one, data_, &one);
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void hiopVectorPar::axpy(double alpha, const hiopVector& x, const hiopVectorInt& i)
{
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);
  const hiopVectorIntSeq& idxs = dynamic_cast<const hiopVectorIntSeq&>(i);

  assert(x.get_local_size() == idxs.get_local_size());
  assert(n_local_ >= idxs.get_local_size());

  const double* xd = xx.local_data_const();
  const index_type* id = idxs.local_data_const();

  for(index_type j = 0; j < idxs.get_local_size(); ++j) {
    assert(id[j] < n_local_);
    data_[id[j]] += alpha * xd[j];
  }
}

/// @brief Performs axpy, this += alpha*x, for selected entries
void hiopVectorPar::axpy_w_pattern(double alpha, const hiopVector& x, const hiopVector& select)
{
  const hiopVectorPar& xx = dynamic_cast<const hiopVectorPar&>(x);
  const hiopVectorPar& idxs = dynamic_cast<const hiopVectorPar&>(select);

  assert(x.get_local_size() == idxs.get_local_size());
  assert(n_local_ == idxs.get_local_size());

  const double* xd = xx.local_data_const();
  const double* z = idxs.local_data_const();

  for(index_type j = 0; j < n_local_; ++j) {
    if(z[j] == 1.0) {
      data_[j] += alpha * xd[j];
    }
  }
}

void hiopVectorPar::axzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_ == vz.n_local_);
  assert(n_local_ == vz.n_local_);
#endif
  // this += alpha * x * z   (data+=alpha*x*z)
  const double *x = vx.local_data_const(), *z = vz.local_data_const();

  if(alpha == 1.0) {
    for(int i = 0; i < n_local_; ++i) {
      data_[i] += x[i] * z[i];
    }
  } else if(alpha == -1.0) {
    for(int i = 0; i < n_local_; ++i) {
      data_[i] -= x[i] * z[i];
    }
  } else if(alpha != 0.) {  // alpha is not 1.0 nor -1.0 nor 0.0
    for(int i = 0; i < n_local_; ++i) {
      data_[i] += alpha * x[i] * z[i];
    }
  }
}

void hiopVectorPar::axdzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  if(alpha == 0.) return;
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_ == vz.n_local_);
  assert(n_local_ == vz.n_local_);
#endif
  // this += alpha * x / z
  const double *x = vx.local_data_const(), *z = vz.local_data_const();

  if(alpha == 1.0) {
    for(int i = 0; i < n_local_; ++i) {
      data_[i] += x[i] / z[i];
    }

  } else if(alpha == -1.0) {
    for(int i = 0; i < n_local_; ++i) {
      data_[i] -= x[i] / z[i];
    }

  } else {  // alpha is neither 1.0 nor -1.0
    for(int i = 0; i < n_local_; ++i) {
      data_[i] += x[i] / z[i] * alpha;
    }
  }
}

void hiopVectorPar::axdzpy_w_pattern(double alpha, const hiopVector& x_, const hiopVector& z_, const hiopVector& select)
{
  const hiopVectorPar& vx = dynamic_cast<const hiopVectorPar&>(x_);
  const hiopVectorPar& vz = dynamic_cast<const hiopVectorPar&>(z_);
  const hiopVectorPar& sel = dynamic_cast<const hiopVectorPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_ == vz.n_local_);
  assert(n_local_ == vz.n_local_);
#endif
  // this += alpha * x / z   (y+=alpha*x/z)
  double* y = data_;
  const double *x = vx.local_data_const(), *z = vz.local_data_const(), *s = sel.local_data_const();
  int it;
  if(alpha == 1.0) {
    for(it = 0; it < n_local_; it++)
      if(s[it] == 1.0) y[it] += x[it] / z[it];
  } else if(alpha == -1.0) {
    for(it = 0; it < n_local_; it++)
      if(s[it] == 1.0) y[it] -= x[it] / z[it];
  } else
    for(it = 0; it < n_local_; it++)
      if(s[it] == 1.0) y[it] += alpha * x[it] / z[it];
}

void hiopVectorPar::addConstant(double c)
{
  for(size_type i = 0; i < n_local_; i++) data_[i] += c;
}

void hiopVectorPar::addConstant_w_patternSelect(double c, const hiopVector& ix_)
{
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(ix_);
  assert(this->n_local_ == ix.n_local_);
  const double* ix_vec = ix.data_;
  for(int i = 0; i < n_local_; i++)
    if(ix_vec[i] == 1.) data_[i] += c;
}

double hiopVectorPar::min() const
{
  double ret_val = std::numeric_limits<double>::max();
  for(int i = 0; i < n_local_; i++) {
    ret_val = (ret_val < data_[i]) ? ret_val : data_[i];
  }
#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr = MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

double hiopVectorPar::min_w_pattern(const hiopVector& select) const
{
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(select);
  assert(this->n_local_ == ix.n_local_);

  double ret_val = std::numeric_limits<double>::max();
  const double* ix_vec = ix.data_;
  for(int i = 0; i < n_local_; i++) {
    if(ix_vec[i] == 1.) {
      ret_val = (ret_val < data_[i]) ? ret_val : data_[i];
    }
  }
#ifdef HIOP_USE_MPI
  double ret_val_g;
  int ierr = MPI_Allreduce(&ret_val, &ret_val_g, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_val = ret_val_g;
#endif
  return ret_val;
}

void hiopVectorPar::min(double& m, int& index) const { assert(false && "not implemented"); }

void hiopVectorPar::negate()
{
  double minusOne = -1.0;
  int one = 1, n = n_local_;
  DSCAL(&n, &minusOne, data_, &one);
}

void hiopVectorPar::invert()
{
  for(int i = 0; i < n_local_; i++) {
#ifdef HIOP_DEEPCHECKS
    if(fabs(data_[i]) < 1e-35) assert(false);
#endif
    data_[i] = 1. / data_[i];
  }
}

// uses Kahan's summation algorithm to reduce numerical error
double hiopVectorPar::logBarrier_local(const hiopVector& select) const
{
  double sum = 0.0;
  double comp = 0.0;
  const hiopVectorPar& ix = dynamic_cast<const hiopVectorPar&>(select);
  assert(this->n_local_ == ix.n_local_);
  const double* ix_vec = ix.data_;
  for(int i = 0; i < n_local_; i++) {
    if(ix_vec[i] == 1.) {
      double y = log(data_[i]) - comp;
      double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
  }
  return sum;
}

double hiopVectorPar::sum_local() const
{
  double sum = 0.0;
  for(int i = 0; i < n_local_; i++) {
    sum += data_[i];
  }
  return sum;
}

/* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
void hiopVectorPar::addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& ix)
{
#ifdef HIOP_DEEPCHECKS
  assert(this->n_local_ == dynamic_cast<const hiopVectorPar&>(ix).n_local_);
  assert(this->n_local_ == dynamic_cast<const hiopVectorPar&>(x).n_local_);
#endif
  const double* ix_vec = dynamic_cast<const hiopVectorPar&>(ix).data_;
  const double* x_vec = dynamic_cast<const hiopVectorPar&>(x).data_;

  for(int i = 0; i < n_local_; i++)
    if(ix_vec[i] == 1.) data_[i] += alpha / x_vec[i];
}

double hiopVectorPar::linearDampingTerm_local(const hiopVector& ixleft,
                                              const hiopVector& ixright,
                                              const double& mu,
                                              const double& kappa_d) const
{
  const double* ixl = (dynamic_cast<const hiopVectorPar&>(ixleft)).local_data_const();
  const double* ixr = (dynamic_cast<const hiopVectorPar&>(ixright)).local_data_const();
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == (dynamic_cast<const hiopVectorPar&>(ixleft)).n_local_);
  assert(n_local_ == (dynamic_cast<const hiopVectorPar&>(ixright)).n_local_);
#endif
  double term = 0.0;
  for(size_type i = 0; i < n_local_; i++) {
    if(ixl[i] == 1. && ixr[i] == 0.) term += data_[i];
  }
  term *= mu;
  term *= kappa_d;
  return term;
}

void hiopVectorPar::addLinearDampingTerm(const hiopVector& ixleft,
                                         const hiopVector& ixright,
                                         const double& alpha,
                                         const double& ct)
{
  const double* ixl = (dynamic_cast<const hiopVectorPar&>(ixleft)).local_data_const();
  const double* ixr = (dynamic_cast<const hiopVectorPar&>(ixright)).local_data_const();
  double* v = this->local_data();

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == (dynamic_cast<const hiopVectorPar&>(ixleft)).n_local_);
  assert(n_local_ == (dynamic_cast<const hiopVectorPar&>(ixright)).n_local_);
#endif

  for(index_type i = 0; i < n_local_; i++) {
    v[i] = alpha * v[i] + (ixl[i] - ixr[i]) * ct;
  }
}

int hiopVectorPar::allPositive()
{
  int allPos = true;
  int i = 0;
  while(i < n_local_ && allPos) {
    if(data_[i++] <= 0) {
      allPos = false;
    }
  }

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

bool hiopVectorPar::projectIntoBounds_local(const hiopVector& xl_,
                                            const hiopVector& ixl_,
                                            const hiopVector& xu_,
                                            const hiopVector& ixu_,
                                            double kappa1,
                                            double kappa2)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(xl_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorPar&>(ixl_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorPar&>(xu_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorPar&>(ixu_)).n_local_ == n_local_);
#endif
  const double* xl = (dynamic_cast<const hiopVectorPar&>(xl_)).local_data_const();
  const double* ixl = (dynamic_cast<const hiopVectorPar&>(ixl_)).local_data_const();
  const double* xu = (dynamic_cast<const hiopVectorPar&>(xu_)).local_data_const();
  const double* ixu = (dynamic_cast<const hiopVectorPar&>(ixu_)).local_data_const();
  double* x0 = data_;

  const double small_double = std::numeric_limits<double>::min() * 100;

  double aux, aux2;
  for(size_type i = 0; i < n_local_; i++) {
    if(ixl[i] != 0 && ixu[i] != 0) {
      if(xl[i] > xu[i]) return false;
      aux = kappa2 * (xu[i] - xl[i]) - small_double;
      aux2 = xl[i] + fmin(kappa1 * fmax(1., fabs(xl[i])), aux);
      if(x0[i] < aux2) {
        x0[i] = aux2;
      } else {
        aux2 = xu[i] - fmin(kappa1 * fmax(1., fabs(xu[i])), aux);
        if(x0[i] > aux2) {
          x0[i] = aux2;
        }
      }
#ifdef HIOP_DEEPCHECKS
      // if(x0[i]>xl[i] && x0[i]<xu[i]) {
      // } else {
      // printf("i=%d  x0=%g xl=%g xu=%g\n", i, x0[i], xl[i], xu[i]);
      // }
      assert(x0[i] > xl[i] && x0[i] < xu[i] && "this should not happen -> HiOp bug");

#endif
    } else {
      if(ixl[i] != 0.)
        x0[i] = fmax(x0[i], xl[i] + kappa1 * fmax(1, fabs(xl[i])) - small_double);
      else if(ixu[i] != 0)
        x0[i] = fmin(x0[i], xu[i] - kappa1 * fmax(1, fabs(xu[i])) - small_double);
      else { /*nothing for free vars  */
      }
    }
  }
  return true;
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorPar::fractionToTheBdry_local(const hiopVector& dx, const double& tau) const
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(dx)).n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1);
#endif
  double alpha = 1.0, aux;
  const double* d = (dynamic_cast<const hiopVectorPar&>(dx)).local_data_const();
  const double* x = data_;
  for(int i = 0; i < n_local_; i++) {
#ifdef HIOP_DEEPCHECKS
    assert(x[i] > 0);
#endif
    if(d[i] >= 0) continue;
    aux = -tau * x[i] / d[i];
    if(aux < alpha) alpha = aux;
  }
  return alpha;
}
/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorPar::fractionToTheBdry_w_pattern_local(const hiopVector& dx, const double& tau, const hiopVector& ix) const
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(dx)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorPar&>(ix)).n_local_ == n_local_);
  assert(tau > 0);
  assert(tau < 1);
#endif
  double alpha = 1.0, aux;
  const double* d = (dynamic_cast<const hiopVectorPar&>(dx)).local_data_const();
  const double* x = data_;
  const double* pat = (dynamic_cast<const hiopVectorPar&>(ix)).local_data_const();
  for(int i = 0; i < n_local_; i++) {
    if(d[i] >= 0) continue;
    if(pat[i] == 0) continue;
#ifdef HIOP_DEEPCHECKS
    assert(x[i] > 0);
#endif
    aux = -tau * x[i] / d[i];
    if(aux < alpha) alpha = aux;
  }
  return alpha;
}

void hiopVectorPar::selectPattern(const hiopVector& ix_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(ix_)).n_local_ == n_local_);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_)).local_data_const();
  double* x = data_;
  for(int i = 0; i < n_local_; i++)
    if(ix[i] == 0.0) x[i] = 0.0;
}

bool hiopVectorPar::matchesPattern(const hiopVector& ix_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(ix_)).n_local_ == n_local_);
#endif
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_)).local_data_const();
  int bmatches = true;
  double* x = data_;
  for(int i = 0; (i < n_local_) && bmatches; i++) {
    if(ix[i] == 0.0 && x[i] != 0.0) {
      bmatches = false;
    }
  }
#ifdef HIOP_USE_MPI
  int bmatches_glob = bmatches;
  int ierr = MPI_Allreduce(&bmatches, &bmatches_glob, 1, MPI_INT, MPI_LAND, comm_);
  assert(MPI_SUCCESS == ierr);
  return bmatches_glob;
#endif
  return bmatches;
}

int hiopVectorPar::allPositive_w_patternSelect(const hiopVector& w_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(w_)).n_local_ == n_local_);
#endif
  const double* w = (dynamic_cast<const hiopVectorPar&>(w_)).local_data_const();
  const double* x = data_;
  int allPos = 1;
  for(int i = 0; i < n_local_ && allPos; i++) {
    if(w[i] != 0.0 && x[i] <= 0.) {
      allPos = 0;
    }
  }
#ifdef HIOP_USE_MPI
  int allPosG = allPos;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

void hiopVectorPar::adjustDuals_plh(const hiopVector& x_, const hiopVector& ix_, const double& mu, const double& kappa)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorPar&>(x_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorPar&>(ix_)).n_local_ == n_local_);
#endif
  const double* x = (dynamic_cast<const hiopVectorPar&>(x_)).local_data_const();
  const double* ix = (dynamic_cast<const hiopVectorPar&>(ix_)).local_data_const();
  double* z = data_;  // the dual
  double a, b;
  for(size_type i = 0; i < n_local_; i++) {
    if(ix[i] == 1.) {
      a = mu / x[i];
      b = a / kappa;
      a = a * kappa;
      if(*z < b) {
        *z = b;
      } else {  // z[i]>=b
        if(a <= b) {
          *z = b;
        } else {  // a>b
          if(a < *z) {
            *z = a;
          }
          // else a>=z[i] then *z=*z (z[i] does not need adjustment)
        }
      }
    }
    z++;
  }
}

bool hiopVectorPar::is_zero() const
{
  int all_zero = true;
  int i{0};
  while(i < n_local_ && all_zero) {
    if(data_[i++] != 0.0) {
      all_zero = false;
    }
  }
#ifdef HIOP_USE_MPI
  int all_zero_G;
  int ierr = MPI_Allreduce(&all_zero, &all_zero_G, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return all_zero_G;
#endif
  return all_zero;
}

bool hiopVectorPar::isnan_local() const
{
  for(size_type i = 0; i < n_local_; i++)
    if(std::isnan(data_[i])) return true;
  return false;
}

bool hiopVectorPar::isinf_local() const
{
  for(size_type i = 0; i < n_local_; i++)
    if(std::isinf(data_[i])) return true;
  return false;
}

bool hiopVectorPar::isfinite_local() const
{
  for(size_type i = 0; i < n_local_; i++)
    if(0 == std::isfinite(data_[i])) return false;
  return true;
}

void hiopVectorPar::print(FILE* file /*=nullptr*/,
                          const char* msg /*=nullptr*/,
                          int max_elems /*=-1*/,
                          int rank /*=-1*/) const
{
  int myrank_ = 0, numranks = 1;

  if(nullptr == file) {
    file = stdout;
  }

#ifdef HIOP_USE_MPI
  if(rank >= 0) {
    int err = MPI_Comm_rank(comm_, &myrank_);
    assert(err == MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &numranks);
    assert(err == MPI_SUCCESS);
  }
#endif
  if(myrank_ == rank || rank == -1) {
    if(max_elems > n_local_) max_elems = n_local_;

    if(nullptr == msg) {
      if(numranks > 1) {
        fprintf(file, "vector of size %d, printing %d elems (on rank=%d)\n", n_, max_elems, myrank_);
      } else {
        fprintf(file, "vector of size %d, printing %d elems (serial)\n", n_, max_elems);
      }
    } else {
      fprintf(file, "%s ", msg);
    }
    fprintf(file, "=[");
    max_elems = max_elems >= 0 ? max_elems : n_local_;
    for(int it = 0; it < max_elems; it++) {
      fprintf(file, "%24.18e ; ", data_[it]);
    }
    fprintf(file, "];\n");
  }
}

size_type hiopVectorPar::numOfElemsLessThan(const double& val) const
{
  size_type ret_num = 0;
  for(size_type i = 0; i < n_local_; i++) {
    if(data_[i] < val) {
      ret_num++;
    }
  }

#ifdef HIOP_USE_MPI
  size_type ret_num_global;
  int ierr = MPI_Allreduce(&ret_num, &ret_num_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_num = ret_num_global;
#endif

  return ret_num;
}

size_type hiopVectorPar::numOfElemsAbsLessThan(const double& val) const
{
  size_type ret_num = 0;
  for(size_type i = 0; i < n_local_; i++) {
    if(fabs(data_[i]) < val) {
      ret_num++;
    }
  }

#ifdef HIOP_USE_MPI
  size_type ret_num_global;
  int ierr = MPI_Allreduce(&ret_num, &ret_num_global, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  ret_num = ret_num_global;
#endif

  return ret_num;
}

void hiopVectorPar::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                      const int start,
                                      const int end,
                                      const hiopInterfaceBase::NonlinearityType* arr_src,
                                      const int start_src) const
{
  assert(end <= n_local_ && start <= end && start >= 0 && start_src >= 0);

  const index_type how_many = end - start;
  if(how_many > 0) {
    ExecSpace<MemBackendCpp, ExecPolicySeq> exec_space_dest;
    exec_space_dest.copy(arr + start, arr_src + start_src, how_many, exec_space_);
  }
}

void hiopVectorPar::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                      const int start,
                                      const int end,
                                      const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(end <= n_local_ && start <= end);

  for(int i = start; i < end; i++) {
    arr[i] = arr_src;
  }
}

bool hiopVectorPar::is_equal(const hiopVector& vec) const
{
  if(n_local_ != vec.get_local_size()) {
    return false;
  }
  int all_equal = true;
  const double* data_v = vec.local_data_const();
  for(auto i = 0; i < n_local_; ++i) {
    if(data_[i] != data_v[i]) {
      all_equal = false;
      break;
    }
  }

#ifdef HIOP_USE_MPI
  int all_equalG;
  int ierr = MPI_Allreduce(&all_equal, &all_equalG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return all_equalG;
#endif
  return all_equal;
}

};  // namespace hiop
