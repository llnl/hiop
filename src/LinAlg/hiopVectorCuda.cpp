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
 * @file hiopVectorCuda.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include "hiopVectorCuda.hpp"
#include "MemBackendCudaImpl.hpp"
#include "MemBackendCppImpl.hpp"
#include "hiopVectorIntCuda.hpp"
#include "VectorCudaKernels.hpp"
#include "MathKernelsCuda.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "LinAlgFactory.hpp"
#include "hiopVectorPar.hpp"

#include <cmath>
#include <limits>

namespace hiop
{

hiopVectorCuda::hiopVectorCuda(const size_type& glob_n, index_type* col_part, MPI_Comm comm)
    : hiopVector(),
      comm_(comm),
      idx_cumsum_{nullptr}
{
  n_ = glob_n;

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_ == MPI_COMM_NULL) {
    comm_ = MPI_COMM_SELF;
  }
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

  data_ = exec_space_.alloc_array<double>(n_local_);
  // Create host mirror if the memory space is on device
  data_host_mirror_ = exec_space_host_.alloc_array<double>(n_local_);

  // handles
  cublasCreate(&handle_cublas_);
}

hiopVectorCuda::hiopVectorCuda(const hiopVectorCuda& v)
    : hiopVector(),
      idx_cumsum_{nullptr}
{
  n_local_ = v.get_local_size();
  n_ = v.get_size();
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;

  data_ = exec_space_.alloc_array<double>(n_local_);
  // Create host mirror if the memory space is on device
  data_host_mirror_ = exec_space_host_.alloc_array<double>(n_local_);

  // handles
  cublasCreate(&handle_cublas_);
}

hiopVectorCuda::~hiopVectorCuda()
{
  exec_space_host_.dealloc_array(data_host_mirror_);
  exec_space_.dealloc_array(data_);
  data_ = nullptr;
  data_host_mirror_ = nullptr;

  // Delete workspaces and handles
  cublasDestroy(handle_cublas_);

  delete idx_cumsum_;
}

/// @brief Set all elements to zero.
void hiopVectorCuda::setToZero() { hiop::cuda::thrust_fill_kernel(n_local_, data_, 0.0); }

/// @brief Set all elements to c
void hiopVectorCuda::setToConstant(double c) { hiop::cuda::thrust_fill_kernel(n_local_, data_, c); }

/// @brief Set all elements to random values uniformly distributed between `minv` and `maxv`.
void hiopVectorCuda::set_to_random_uniform(double minv, double maxv)
{
  double* data = data_;
  hiop::cuda::array_random_uniform_kernel(n_local_, data, minv, maxv);
}  // namespace hiop

/// @brief Set all elements that are not zero in ix to  c, and the rest to 0
void hiopVectorCuda::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  // TODO: add one cu function to perform the following two functions
  setToConstant(c);
  componentMult(select);
}

/// @brief Copy the elements of v
void hiopVectorCuda::copyFrom(const hiopVector& v_in)
{
  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(v_in);
  assert(v.n_local_ == n_local_);

  auto b = exec_space_.copy(data_, v.data_, n_local_, v.exec_space());
  assert(b);
}

/// @brief Copy the elements of v
void hiopVectorCuda::copyFrom(const double* v_local_data)
{
  if(v_local_data) {
    auto b = exec_space_.copy(data_, v_local_data, n_local_);
    assert(b);
  }
}

void hiopVectorCuda::copy_from_vectorpar(const hiopVectorPar& vsrc)
{
  assert(n_local_ == vsrc.get_local_size());
  exec_space_.copy(data_, vsrc.local_data_const(), n_local_, vsrc.exec_space());
}

void hiopVectorCuda::copy_to_vectorpar(hiopVectorPar& vdest) const
{
  assert(n_local_ == vdest.get_local_size());
  vdest.exec_space().copy(vdest.local_data(), data_, n_local_, exec_space_);
}

/// @brief Copy from vec the elements specified by the indices in select
void hiopVectorCuda::copy_from_w_pattern(const hiopVector& vv, const hiopVector& select)
{
  copyFrom(vv);
  componentMult(select);
}

/// @brief Copy the `n` elements of v starting at `start_index_in_dest` in `this`
void hiopVectorCuda::copyFromStarting(int start_index_in_dest, const double* v, int nv)
{
  assert(start_index_in_dest + nv <= n_local_);
  auto b = exec_space_.copy(data_ + start_index_in_dest, v, nv);
  assert(b);
}

/// @brief Copy v_src into `this` starting at start_index_in_dest in `this`. */
void hiopVectorCuda::copyFromStarting(int start_index_in_dest, const hiopVector& v_src)
{
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
  assert(start_index_in_dest + v_src.get_local_size() <= n_local_);

  const hiopVectorCuda& v = dynamic_cast<const hiopVectorCuda&>(v_src);
  auto b = exec_space_.copy(data_ + start_index_in_dest, v.data_, v.n_local_, v.exec_space());
  assert(b);
}

/// @brief Copy the `n` elements of v starting at `start_index_in_v` into `this`
void hiopVectorCuda::copy_from_starting_at(const double* v, int start_index_in_v, int nv)
{
  auto b = exec_space_.copy(data_, v + start_index_in_v, nv);
  assert(b);
}

/// @brief Copy from src the elements specified by the indices in index_in_src.
void hiopVectorCuda::copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src)
{
  assert(index_in_src.get_local_size() == n_local_);

  int* id = const_cast<int*>(index_in_src.local_data_const());
  double* dd = data_;
  const double* vd = src.local_data_const();

  hiop::cuda::copy_from_index_kernel(n_local_, dd, vd, id);
}

/// @brief Copy from src the elements specified by the indices in index_in_src.
void hiopVectorCuda::copy_from_indexes(const double* src, const hiopVectorInt& index_in_src)
{
  assert(index_in_src.get_local_size() == n_local_);

  hiop::cuda::copy_from_index_kernel(n_local_, data_, src, index_in_src.local_data_const());
}

///  @brief Copy from `v` starting at `start_idx_src` to `this` starting at `start_idx_dest`
void hiopVectorCuda::startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& vec_src, int start_idx_src)
{
  size_type howManyToCopyDest = this->n_local_ - start_idx_dest;

#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  int v_size = vec_src.get_local_size();
  assert((start_idx_dest >= 0 && start_idx_dest < this->n_local_) || this->n_local_ == 0);
  assert((start_idx_src >= 0 && start_idx_src < v_size) || v_size == 0 || v_size == start_idx_src);
  const size_type howManyToCopySrc = v_size - start_idx_src;

  if(howManyToCopyDest == 0 || howManyToCopySrc == 0) {
    return;
  }

  assert(howManyToCopyDest <= howManyToCopySrc);

  auto& v_src = dynamic_cast<const hiopVectorCuda&>(vec_src);
  exec_space_.copy(data_ + start_idx_dest, v_src.data_ + start_idx_src, howManyToCopyDest, v_src.exec_space());
}

/// @brief Copy `this` to double array, which is assumed to be at least of `n_local_` size.
void hiopVectorCuda::copyTo(double* dest) const
{
  auto* this_nonconst = const_cast<hiopVectorCuda*>(this);
  assert(nullptr != this_nonconst);
  auto b = this_nonconst->exec_space_.copy(dest, data_, n_local_);
  assert(b);
}

/// @brief Copy `this` to dst starting at `start_index` in `this`.
void hiopVectorCuda::copyToStarting(int start_index, hiopVector& dst) const
{
  int v_size = dst.get_local_size();
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "are you sure you want to call this?");
#endif
  assert(start_index + v_size <= n_local_);

  // If nothing to copy, return.
  if(v_size == 0) return;

  auto& dst_cu = dynamic_cast<hiopVectorCuda&>(dst);
  dst_cu.exec_space().copy(dst_cu.data_, data_ + start_index, v_size, exec_space_);
}

/// @brief Copy `this` to `dst` starting at `start_index` in `dst`.
void hiopVectorCuda::copyToStarting(hiopVector& dst, int start_index) const
{
  int v_size = dst.get_local_size();
  assert(start_index + n_local_ <= v_size);

  // If there is nothing to copy, return.
  if(n_local_ == 0) return;

  auto& dst_cu = dynamic_cast<hiopVectorCuda&>(dst);
  dst_cu.exec_space().copy(dst_cu.data_ + start_index, data_, n_local_, exec_space_);
}

/// @brief Copy the entries in `this` where corresponding `ix` is nonzero, to v starting at start_index in `v`.
void hiopVectorCuda::copyToStartingAt_w_pattern(hiopVector& vec, int start_index_in_dest, const hiopVector& select) const
{
  if(n_local_ == 0) {
    return;
  }

  double* dd = data_;
  double* vd = vec.local_data();
  const double* pattern = select.local_data_const();

  if(nullptr == idx_cumsum_) {
    idx_cumsum_ = LinearAlgebraFactory::create_vector_int("CUDA", n_local_ + 1);
    index_type* nnz_in_row = idx_cumsum_->local_data();

    hiop::cuda::compute_cusum_kernel(n_local_ + 1, nnz_in_row, pattern);
  }

  index_type* nnz_cumsum = idx_cumsum_->local_data();
  index_type v_n_local = vec.get_local_size();

  hiop::cuda::copyToStartingAt_w_pattern_kernel(n_local_, v_n_local, start_index_in_dest, nnz_cumsum, vd, dd);
}

/// @brief Copy the entries in `c` and `d` to `this`, according to the mapping in `c_map` and `d_map`
void hiopVectorCuda::copy_from_two_vec_w_pattern(const hiopVector& c,
                                                 const hiopVectorInt& c_map,
                                                 const hiopVector& d,
                                                 const hiopVectorInt& d_map)
{
  const int c_size = c.get_size();
  const int d_size = d.get_size();

  assert(c_size == c_map.get_local_size());
  assert(d_size == d_map.get_local_size());
  assert(c_size + d_size == n_local_);

  hiop::cuda::copy_src_to_mapped_dest_kernel(c_size, c.local_data_const(), local_data(), c_map.local_data_const());
  hiop::cuda::copy_src_to_mapped_dest_kernel(d_size, d.local_data_const(), local_data(), d_map.local_data_const());
}

/// @brief Copy the entries in `this` to `c` and `d`, according to the mapping `c_map` and `d_map`
void hiopVectorCuda::copy_to_two_vec_w_pattern(hiopVector& c,
                                               const hiopVectorInt& c_map,
                                               hiopVector& d,
                                               const hiopVectorInt& d_map) const
{
  const int c_size = c.get_size();
  const int d_size = d.get_size();

  assert(c_size == c_map.get_local_size());
  assert(d_size == d_map.get_local_size());
  assert(c_size + d_size == n_local_);

  hiop::cuda::copy_mapped_src_to_dest_kernel(c_size, local_data_const(), c.local_data(), c_map.local_data_const());
  hiop::cuda::copy_mapped_src_to_dest_kernel(d_size, local_data_const(), d.local_data(), d_map.local_data_const());
}

/// @brief Copy `this` (source) starting at `start_idx_in_src` to `dest` starting at index `int start_idx_dest`
void hiopVectorCuda::startingAtCopyToStartingAt(index_type start_idx_in_src,
                                                hiopVector& dest,
                                                index_type start_idx_dest,
                                                size_type num_elems /* = -1 */) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n_ && "only for local/non-distributed vectors");
#endif

  assert(start_idx_in_src >= 0 && start_idx_in_src <= this->n_local_);
  assert(start_idx_dest >= 0 && start_idx_dest <= dest.get_local_size());

  const int dest_size = dest.get_local_size();
#ifndef NDEBUG
  if(start_idx_dest == dest_size || start_idx_in_src == this->n_local_) assert((num_elems == -1 || num_elems == 0));
#endif

  if(num_elems < 0) {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest_size - start_idx_dest);
  } else {
    assert(num_elems + start_idx_in_src <= this->n_local_);
    assert(num_elems + start_idx_dest <= dest_size);
    // make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)(this->n_local_ - start_idx_in_src));
    num_elems = std::min(num_elems, (int)(dest_size - start_idx_dest));
  }

  if(num_elems == 0) return;

  auto& dest_cu = dynamic_cast<hiopVectorCuda&>(dest);
  dest_cu.exec_space().copy(dest_cu.data_ + start_idx_dest, data_ + start_idx_in_src, num_elems, exec_space_);
}

/**
 * @brief Copy `this` (source) starting at `start_idx_in_src` to `dest` starting at index `int start_idx_dest`
 * The values are copy to `dest` where the corresponding entry in `selec_dest` is nonzero
 */
void hiopVectorCuda::startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                          hiopVector& destination,
                                                          index_type start_idx_dest,
                                                          const hiopVector& selec_dest,
                                                          size_type num_elems /*=-1*/) const
{
  assert(false && "TODO --- only used in the full linear system");
}

/** @brief Return the two norm */
double hiopVectorCuda::twonorm() const
{
  int one = 1;
  double nrm = 0.;
  if(n_local_ > 0) {
    cublasStatus_t ret_cublas = cublasDnrm2(handle_cublas_, n_local_, data_, one, &nrm);
    assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
  }

#ifdef HIOP_USE_MPI
  nrm *= nrm;
  double nrmG;
  int ierr = MPI_Allreduce(&nrm, &nrmG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  nrm = std::sqrt(nrmG);
#endif
  return nrm;
}

/** @brief Return the infinity norm */
double hiopVectorCuda::infnorm() const
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

/** @brief inf norm on single rank */
double hiopVectorCuda::infnorm_local() const { return hiop::cuda::infnorm_local_kernel(n_local_, data_); }

/** @brief Return the one norm */
double hiopVectorCuda::onenorm() const
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

/** @brief L1 norm on single rank */
double hiopVectorCuda::onenorm_local() const { return hiop::cuda::onenorm_local_kernel(n_local_, data_); }

/** @brief Multiply the components of this by the components of v. */
void hiopVectorCuda::componentMult(const hiopVector& vec)
{
  assert(n_local_ == vec.get_local_size());
  hiop::cuda::thrust_component_mult_kernel(n_local_, data_, vec.local_data_const());
}

/** @brief Divide the components of this hiopVector by the components of v. */
void hiopVectorCuda::componentDiv(const hiopVector& vec)
{
  assert(n_local_ == vec.get_local_size());
  hiop::cuda::thrust_component_div_kernel(n_local_, data_, vec.local_data_const());
}

/**
 * @brief Elements of this that corespond to nonzeros in ix are divided by elements of v.
 * The rest of elements of this are set to zero.
 */
void hiopVectorCuda::componentDiv_w_selectPattern(const hiopVector& vec, const hiopVector& select)
{
  assert(n_local_ == vec.get_local_size());
  hiop::cuda::component_div_w_pattern_kernel(n_local_, data_, vec.local_data_const(), select.local_data_const());
}

/** @brief Set each component of this hiopVector to the minimum of itself and the given constant. */
void hiopVectorCuda::component_min(const double constant) { hiop::cuda::component_min_kernel(n_local_, data_, constant); }

/** @brief Set each component of this hiopVector to the minimum of itself and the corresponding component of `v`. */
void hiopVectorCuda::component_min(const hiopVector& vec)
{
  assert(vec.get_local_size() == n_local_);
  const double* vd = vec.local_data_const();
  hiop::cuda::component_min_kernel(n_local_, data_, vd);
}

/** @brief Set each component of this hiopVector to the maximum of itself and the given constant. */
void hiopVectorCuda::component_max(const double constant) { hiop::cuda::component_max_kernel(n_local_, data_, constant); }

/** @brief Set each component of this hiopVector to the maximum of itself and the corresponding component of `v`. */
void hiopVectorCuda::component_max(const hiopVector& vec)
{
  assert(vec.get_local_size() == n_local_);

  const double* vd = vec.local_data_const();

  hiop::cuda::component_max_kernel(n_local_, data_, vd);
}

/** @brief Set each component to its absolute value */
void hiopVectorCuda::component_abs() { hiop::cuda::thrust_component_abs_kernel(n_local_, data_); }

/** @brief Apply sign function to each component */
void hiopVectorCuda::component_sgn() { hiop::cuda::thrust_component_sgn_kernel(n_local_, data_); }

/** @brief compute sqrt of each component */
void hiopVectorCuda::component_sqrt() { hiop::cuda::thrust_component_sqrt_kernel(n_local_, data_); }

/// @brief Scale each element of this  by the constant alpha
void hiopVectorCuda::scale(double alpha)
{
  int one = 1;
  cublasStatus_t ret_cublas = cublasDscal(handle_cublas_, n_local_, &alpha, data_, one);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
}

/// @brief this += alpha * x
void hiopVectorCuda::axpy(double alpha, const hiopVector& xvec)
{
  int one = 1;
  cublasStatus_t ret_cublas = cublasDaxpy(handle_cublas_, n_local_, &alpha, xvec.local_data_const(), one, data_, one);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);
}

/// @brief this += alpha * x, for the entries in `this` where corresponding `select` is nonzero.
void hiopVectorCuda::axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select)
{
  axpy(alpha, xvec);
  componentMult(select);
}

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void hiopVectorCuda::axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i)
{
  assert(xvec.get_size() == i.get_local_size());
  assert(xvec.get_local_size() == i.get_local_size());
  assert(i.get_local_size() <= n_local_);

  double* yd = data_;
  const double* xd = const_cast<const double*>(xvec.local_data_const());
  int* id = const_cast<int*>(i.local_data_const());

  hiop::cuda::axpy_w_map_kernel(n_local_, yd, xd, id, alpha);
}

/** @brief this[i] += alpha*x[i]*z[i] forall i */
void hiopVectorCuda::axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
#ifdef HIOP_DEEPCHECKS
  assert(xvec.get_local_size() == zvec.get_local_size());
  assert(n_local_ == zvec.get_local_size());
#endif
  double* dd = data_;
  const double* xd = xvec.local_data_const();
  const double* zd = zvec.local_data_const();

  hiop::cuda::axzpy_kernel(n_local_, dd, xd, zd, alpha);
}

/** @brief this[i] += alpha*x[i]/z[i] forall i */
void hiopVectorCuda::axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
#ifdef HIOP_DEEPCHECKS
  assert(xvec.get_local_size() == zvec.get_local_size());
  assert(n_local_ == zvec.get_local_size());
#endif
  double* yd = data_;
  const double* xd = xvec.local_data_const();
  const double* zd = zvec.local_data_const();

  hiop::cuda::axdzpy_kernel(n_local_, yd, xd, zd, alpha);
}

/** @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection */
void hiopVectorCuda::axdzpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& zvec, const hiopVector& select)
{
#ifdef HIOP_DEEPCHECKS
  assert(xvec.get_local_size() == zvec.get_local_size());
  assert(n_local_ == zvec.get_local_size());
#endif
  double* yd = data_;
  const double* xd = xvec.local_data_const();
  const double* zd = zvec.local_data_const();
  const double* id = select.local_data_const();

  hiop::cuda::axdzpy_w_pattern_kernel(n_local_, yd, xd, zd, id, alpha);
}

/** @brief this[i] += c forall i */
void hiopVectorCuda::addConstant(double c) { hiop::cuda::add_constant_kernel(n_local_, data_, c); }

/** @brief this[i] += c forall i with pattern selection */
void hiopVectorCuda::addConstant_w_patternSelect(double c, const hiopVector& select)
{
  assert(this->n_local_ == select.get_local_size());
  const double* id = select.local_data_const();

  hiop::cuda::add_constant_w_pattern_kernel(n_local_, data_, id, c);
}

/** @brief Return the dot product of this hiopVector with v */
double hiopVectorCuda::dotProductWith(const hiopVector& v) const
{
  int one = 1;
  double retval;
  cublasStatus_t ret_cublas = cublasDdot(handle_cublas_, n_local_, v.local_data_const(), one, data_, one, &retval);
  assert(ret_cublas == CUBLAS_STATUS_SUCCESS);

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&retval, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  retval = dotprodG;
#endif

  return retval;
}

/// @brief Negate all the elements of this
void hiopVectorCuda::negate() { hiop::cuda::thrust_negate_kernel(n_local_, data_); }

/// @brief Invert (1/x) the elements of this
void hiopVectorCuda::invert() { hiop::cuda::invert_kernel(n_local_, data_); }

/** @brief Sum all selected log(this[i]) */
double hiopVectorCuda::logBarrier_local(const hiopVector& select) const
{
  assert(n_local_ == select.get_local_size());
  const double* id = select.local_data_const();

  return hiop::cuda::log_barr_obj_kernel(n_local_, data_, id);
}

/* @brief adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
void hiopVectorCuda::addLogBarrierGrad(double alpha, const hiopVector& xvec, const hiopVector& select)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == xvec.get_local_size());
  assert(n_local_ == select.get_local_size());
#endif
  double* yd = data_;
  const double* xd = xvec.local_data_const();
  const double* id = select.local_data_const();

  hiop::cuda::adxpy_w_pattern_kernel(n_local_, yd, xd, id, alpha);
}

/** @brief Sum all elements */
double hiopVectorCuda::sum_local() const { return hiop::cuda::thrust_sum_kernel(n_local_, data_); }

/** @brief Linear damping term */
double hiopVectorCuda::linearDampingTerm_local(const hiopVector& ixleft,
                                               const hiopVector& ixright,
                                               const double& mu,
                                               const double& kappa_d) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == ixleft.get_local_size());
  assert(n_local_ == ixright.get_local_size());
#endif
  const double* ld = ixleft.local_data_const();
  const double* rd = ixright.local_data_const();
  const double* vd = data_;

  return hiop::cuda::linear_damping_term_kernel(n_local_, vd, ld, rd, mu, kappa_d);
}

/**
 * @brief Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when exactly one of
 * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
 */
void hiopVectorCuda::addLinearDampingTerm(const hiopVector& ixleft,
                                          const hiopVector& ixright,
                                          const double& alpha,
                                          const double& ct)
{
  assert(ixleft.get_local_size() == n_local_);
  assert(ixright.get_local_size() == n_local_);

  const double* ixl = ixleft.local_data_const();
  const double* ixr = ixright.local_data_const();

  double* data = data_;

  // compute linear damping term
  hiop::cuda::add_linear_damping_term_kernel(n_local_, data, ixl, ixr, alpha, ct);
}

/** @brief Check if all elements of the vector are positive */
int hiopVectorCuda::allPositive()
{
  double min_val = hiop::cuda::min_local_kernel(n_local_, data_);

  int allPos = (min_val > 0.0) ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

/** @brief Checks if selected elements of `this` are positive */
int hiopVectorCuda::allPositive_w_patternSelect(const hiopVector& wvec)
{
#ifdef HIOP_DEEPCHECKS
  assert(wvec.get_local_size() == n_local_);
#endif

  const double* id = wvec.local_data_const();
  const double* data = data_;

  int allPos = hiop::cuda::all_positive_w_pattern_kernel(n_local_, data, id);

  allPos = (allPos == n_local_) ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return allPosG;
#endif
  return allPos;
}

/// @brief Return the minimum value in this vector
double hiopVectorCuda::min() const
{
  double result = hiop::cuda::min_local_kernel(n_local_, data_);

#ifdef HIOP_USE_MPI
  double resultG;
  double ierr = MPI_Allreduce(&result, &resultG, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return resultG;
#endif
  return result;
}

/// @brief Find minimum vector element for `select` pattern
double hiopVectorCuda::min_w_pattern(const hiopVector& select) const
{
  assert(this->n_local_ == select.get_local_size());
  const double* data = data_;
  const double* id = select.local_data_const();

  double max_val = std::numeric_limits<double>::max();
  double result = hiop::cuda::min_w_pattern_kernel(n_local_, data, id, max_val);

#ifdef HIOP_USE_MPI
  double resultG;
  double ierr = MPI_Allreduce(&result, &resultG, 1, MPI_DOUBLE, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return resultG;
#endif
  return result;
}

/// @brief Return the minimum value in this vector, and the index at which it occurs. TODO
void hiopVectorCuda::min(double& /* m */, int& /* index */) const { assert(false && "not implemented"); }

/** @brief Project solution into bounds  */
bool hiopVectorCuda::projectIntoBounds_local(const hiopVector& xlo,
                                             const hiopVector& ixl,
                                             const hiopVector& xup,
                                             const hiopVector& ixu,
                                             double kappa1,
                                             double kappa2)
{
#ifdef HIOP_DEEPCHECKS
  assert(xlo.get_local_size() == n_local_);
  assert(ixl.get_local_size() == n_local_);
  assert(xup.get_local_size() == n_local_);
  assert(ixu.get_local_size() == n_local_);
#endif

  const double* xld = xlo.local_data_const();
  const double* ild = ixl.local_data_const();
  const double* xud = xup.local_data_const();
  const double* iud = ixu.local_data_const();
  double* xd = data_;

  // Perform preliminary check to see of all upper value < lower value
  bool bval = hiop::cuda::check_bounds_kernel(n_local_, xld, xud);

  if(false == bval) return false;

  const double small_real = std::numeric_limits<double>::min() * 100;

  hiop::cuda::project_into_bounds_kernel(n_local_, xd, xld, ild, xud, iud, kappa1, kappa2, small_real);

  return true;
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorCuda::fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const
{
#ifdef HIOP_DEEPCHECKS
  assert(dvec.get_local_size() == n_local_);
  assert(tau > 0);
  assert(tau < 1);  // TODO: per documentation above it should be tau <= 1 (?).
#endif

  const double* dd = dvec.local_data_const();
  const double* xd = data_;

  double alpha = hiop::cuda::min_frac_to_bds_kernel(n_local_, xd, dd, tau);

  return alpha;
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select */
double hiopVectorCuda::fractionToTheBdry_w_pattern_local(const hiopVector& dvec,
                                                         const double& tau,
                                                         const hiopVector& select) const
{
#ifdef HIOP_DEEPCHECKS
  assert(dvec.get_local_size() == n_local_);
  assert(select.get_local_size() == n_local_);
  assert(tau > 0);
  assert(tau < 1);
#endif
  const double* dd = dvec.local_data_const();
  const double* xd = data_;
  const double* id = select.local_data_const();

  double alpha = hiop::cuda::min_frac_to_bds_w_pattern_kernel(n_local_, xd, dd, id, tau);

  return alpha;
}

/** @brief Set elements of `this` to zero based on `select`.*/
void hiopVectorCuda::selectPattern(const hiopVector& select)
{
#ifdef HIOP_DEEPCHECKS
  assert(select.get_local_size() == n_local_);
#endif

  double* data = data_;
  const double* id = select.local_data_const();

  // set value with pattern
  hiop::cuda::select_pattern_kernel(n_local_, data, id);
}

/** @brief Checks if `this` matches nonzero pattern of `select`. */
bool hiopVectorCuda::matchesPattern(const hiopVector& pattern)
{
#ifdef HIOP_DEEPCHECKS
  assert(pattern.get_local_size() == n_local_);
#endif

  double* xd = data_;
  const double* id = pattern.local_data_const();

  int bret = hiop::cuda::match_pattern_kernel(n_local_, xd, id);

#ifdef HIOP_USE_MPI
  int mismatch_glob = bret;
  int ierr = MPI_Allreduce(&bret, &mismatch_glob, 1, MPI_INT, MPI_MIN, comm_);
  assert(MPI_SUCCESS == ierr);
  return (mismatch_glob != 0);
#endif
  return bret;
}

/** @brief Adjusts duals. */
void hiopVectorCuda::adjustDuals_plh(const hiopVector& xvec, const hiopVector& ixvec, const double& mu, const double& kappa)
{
#ifdef HIOP_DEEPCHECKS
  assert(xvec.get_local_size() == n_local_);
  assert(ixvec.get_local_size() == n_local_);
#endif
  const double* xd = xvec.local_data_const();
  const double* id = ixvec.local_data_const();
  double* zd = data_;  // the dual

  hiop::cuda::adjustDuals_plh_kernel(n_local_, zd, xd, id, mu, kappa);
}

/** @brief Check if all elements of the vector are zero */
bool hiopVectorCuda::is_zero() const { return hiop::cuda::is_zero_kernel(n_local_, data_); }

/** @brief Returns true if any element of `this` is NaN. */
bool hiopVectorCuda::isnan_local() const { return hiop::cuda::isnan_kernel(n_local_, data_); }

/**
 * @brief Returns true if any element of `this` is Inf.
 *
 * @post `this` is not modified
 *
 * @warning This is local method only!
 */
bool hiopVectorCuda::isinf_local() const { return hiop::cuda::isinf_kernel(n_local_, data_); }

/** @brief Returns true if all elements of `this` are finite. */
bool hiopVectorCuda::isfinite_local() const { return hiop::cuda::isfinite_kernel(n_local_, data_); }

/** @brief Prints vector data to a file in Matlab format. */
void hiopVectorCuda::print(FILE* file /*=nullptr*/,
                           const char* msg /*=nullptr*/,
                           int max_elems /*=-1*/,
                           int rank /*=-1*/) const
{
  // TODO. no fprintf. use printf to print everything on screen?
  // Alternative: create a hiopVectorPar copy and use hiopVectorPar::print
  assert(false && "Not implemented in CUDA device");
}

/// @brief allocates a vector that mirrors this, but doesn't copy the values
hiopVector* hiopVectorCuda::alloc_clone() const
{
  hiopVector* v = new hiopVectorCuda(*this);
  assert(v);
  return v;
}

/// @brief allocates a vector that mirrors this, and copies the values
hiopVector* hiopVectorCuda::new_copy() const
{
  hiopVector* v = new hiopVectorCuda(*this);
  assert(v);
  v->copyFrom(*this);
  return v;
}

/// @brief copy data from host mirror to device
void hiopVectorCuda::copyToDev() { exec_space_.copy(data_, data_host_mirror_, n_local_, exec_space_host_); }

/// @brief copy data from device to host mirror
void hiopVectorCuda::copyFromDev() { exec_space_host_.copy(data_host_mirror_, data_, n_local_, exec_space_); }

/// @brief copy data from host mirror to device
void hiopVectorCuda::copyToDev() const
{
  auto* this_nonconst = const_cast<hiopVectorCuda*>(this);
  assert(nullptr != this_nonconst);

  this_nonconst->copyToDev();
}

/// @brief copy data from device to host mirror
void hiopVectorCuda::copyFromDev() const
{
  auto* this_nonconst = const_cast<hiopVectorCuda*>(this);
  assert(nullptr != this_nonconst);

  this_nonconst->copyFromDev();
}

/// @brief get number of values that are less than the given value `val`. TODO: add unit test
size_type hiopVectorCuda::numOfElemsLessThan(const double& val) const
{
  return hiop::cuda::num_of_elem_less_than_kernel(n_local_, data_, val);
}

/// @brief get number of values whose absolute value are less than the given value `val`. TODO: add unit test
size_type hiopVectorCuda::numOfElemsAbsLessThan(const double& val) const
{
  return hiop::cuda::num_of_elem_absless_than_kernel(n_local_, data_, val);
}

/// @brief set int array `arr`, starting at `start` and ending at `end`, to the values in `arr_src` from `start_src`
void hiopVectorCuda::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                       const int start,
                                       const int end,
                                       const hiopInterfaceBase::NonlinearityType* arr_src,
                                       const int start_src) const
{
  assert(arr && arr_src);
  assert(end <= n_local_ && start <= end && start >= 0);
  // If there is nothing to copy, return.
  const int length = end - start;
  if(length == 0) {
    return;
  }

  hiop::cuda::set_array_from_to_kernel(n_local_, arr, start, length, arr_src, start_src);
}

/// @brief set int array `arr`, starting at `start` and ending at `end`, to the values in `arr_src` from `start_src`
void hiopVectorCuda::set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                       const int start,
                                       const int end,
                                       const hiopInterfaceBase::NonlinearityType arr_src) const
{
  assert(arr && arr_src);
  assert(end <= n_local_ && start <= end && start >= 0);
  // If there is nothing to copy, return.
  int length = end - start;
  if(length == 0) {
    return;
  }

  hiop::cuda::set_array_from_to_kernel(n_local_, arr, start, length, arr_src);
}

/// @brief check if `this` vector is identical to `vec`
bool hiopVectorCuda::is_equal(const hiopVector& vec) const { assert(false && "NOT needed. Remove this func. TODO"); }

}  // namespace hiop
