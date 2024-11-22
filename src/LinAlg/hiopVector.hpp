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

#include <cstdio>
#include <cassert>
#include "hiopInterface.hpp"
#include "hiopVectorInt.hpp"

namespace hiop
{

//"forward" defs
class hiopVectorPar;

class hiopVector
{
public:
  hiopVector()
      : n_(0)
  {}
  virtual ~hiopVector() {};

  /**
   * @brief Set all elements to zero.
   */
  virtual void setToZero() = 0;

  /**
   * @brief Set all elements to zero.
   *
   * @param[in] c - constant
   */
  virtual void setToConstant(double c) = 0;

  /**
   * @brief Set all elements to random values uniformly distributed between `minv` and `maxv`.
   *
   * @param[in] minv - the minimun value
   * @param[in] maxv - the maximun value
   */
  virtual void set_to_random_uniform(double minv, double maxv) = 0;

  /**
   * @brief Set selected elements to `c`, and the rest to 0
   *
   * @param[in] c - constant
   * @param[in] select - pattern selection
   */
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select) = 0;

  /**
   * @brief Copy data from `vec` to this vector
   *
   * @param[in] vec - Vector from which to copy into `this`
   *
   * @pre `vec` and `this` must have same partitioning.
   * @post Elements of `this` are overwritten with elements of `vec`
   */
  virtual void copyFrom(const hiopVector& vec) = 0;

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
  virtual void copyFrom(const double* local_array) = 0;

  /// @
  /**
   * @brief Copy data from `vec` to `this` vector with pattern selection
   *
   * @param[in] vec - Vector from which to copy into `this`
   * @param[in] select - indices of the elements to copy
   *
   * @pre `vec`, `select` and `this` must have same partitioning.
   * @post Elements of `this` are overwritten with elements of `vec`
   */
  virtual void copy_from_w_pattern(const hiopVector& src, const hiopVector& select) = 0;

  /**
   * @brief Copy `nv` elements from array `v` to this vector starting from `start_index_in_this`
   *
   * @param[in] start_index_in_this - position in this where to copy
   * @param[in] v - a raw array from which to copy into `this`
   * @param[in] nv - how many elements of `v` to copy
   *
   * @pre Size of `v` must be >= nv.
   * @pre start_index_in_this+nv <= n_local_
   * @pre `this` is not distributed
   * @pre `v` should be allocated in the memory space/backend of `this`
   *
   * @warning Method casts away const from the `v`.
   */
  virtual void copyFromStarting(int start_index_in_this, const double* v, int nv) = 0;

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
  virtual void copyFromStarting(int start_index, const hiopVector& src) = 0;

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
  virtual void copy_from_starting_at(const double* v, int start_index_in_v, int n) = 0;

  /** Copy to `this` the array content of the hiopVectorPar vector passed as argument.
   *
   * Host-device memory transfer will occur for device implementations.
   *
   * @param[in] vsrc - the source vector from which to copy into `this`
   *
   * @pre `this` and source vector should have the same size.
   * @pre `this` and source vector should have the same MPI distributions (and,
   * hence, same number of local elements) when applicable.
   */
  virtual void copy_from_vectorpar(const hiopVectorPar& vsrc) = 0;

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src.
   *
   * @param[in] src - the source vector from which to copy into `this`
   * @param[in] index_in_src - position in the source vector
   *
   * @pre All vectors must reside in the same memory space.
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   */
  virtual void copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src) = 0;

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src.
   *
   * @param[in] src - the raw array from which to copy into `this`
   * @param[in] index_in_src - position in the source vector
   *
   * @pre All vectors and arrays must reside in the same memory space.
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */
  virtual void copy_from_indexes(const double* src, const hiopVectorInt& index_in_src) = 0;

  /*
   * @brief Copy from 'v' starting at 'start_idx_src' to 'this' starting at 'start_idx_dest'
   *
   * @param[in] start_idx_dest - position in `this` to where to copy
   * @param[in] v - the source vector
   * @param[in] start_idx_src - position in `v` from where to copy
   *
   * @pre Elements are copied into 'this' till the end of the 'this' is reached, more exactly a number
   * of lenght(this) - start_idx_dest elements.
   * @pre The method expects that in 'v' there are at least as many elements starting
   * 'start_idx_src' as 'this' has starting at start_idx_dest, or in other words,
   * length(this) - start_idx_dest <= length(v) - start_idx_src
   */
  virtual void startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v, int start_idx_src) = 0;

  /**
   * @brief Copy `this` vector local data to `dest` buffer.
   *
   * @param[out] dest - destination buffer where to copy vector data to
   *
   * @pre Size of `dest` must be >= n_local_
   * @pre `dest` should be on the same memory space/backend as `this`
   *
   * @post `this` is not modified
   */
  virtual void copyTo(double* dest) const = 0;

  /** Copy the array content `this` in the hiopVectorPar passed as argument
   *
   * @param[out] vdest - destination vector where to copy vector data to
   *
   * Host-device memory transfer will occur for device implementations.
   *
   * @pre `this` and destination vector should have the same size.
   * @pre `this` and destination vector should have the same MPI distributions (and,
   * hence, same number of local elements) when applicable.
   */
  virtual void copy_to_vectorpar(hiopVectorPar& vdest) const = 0;

  /**
   * @brief Copy to `vec` elements of `this` vector starting from `start_index`.
   *
   * @param[in] start_index - position in `this` from where to copy
   * @param[out] dst - the destination vector where to copy elements of `this`
   *
   * @pre start_index + dst.n_local_ <= n_local_
   * @pre `this` and `dst` are not distributed
   */
  virtual void copyToStarting(int start_index, hiopVector& dst) const = 0;

  /**
   * @brief Copy elements of `this` vector to `vec` starting at `start_index_in_dest`.
   *
   * @param[out] vec - a vector where to copy elements of `this`
   * @param[in] start_index_in_dest - position in `vec` where to copy
   *
   * @pre start_index_in_dest + vec.n_local_ <= n_local_
   * @pre `this` and `vec` are not distributed
   */
  virtual void copyToStarting(hiopVector& vec, int start_index_in_dest) const = 0;

  /**
   * @brief Copy specified elements of `this` vector to `vec` starting at `start_index_in_dest`
   *
   * @param[out] vec - a vector where to copy elements of `this`
   * @param[in] start_index_in_dest - position in `vec` where to copy
   * @param[in] ix - a nonzero pattern that implies which elements of `this` to copy
   *
   * @pre start_index_in_dest + vec.n_local_ <= n_local_
   * @pre `this` and `vec` are not distributed
   * @pre `ix` and `this` must have same partitioning.
   */
  /// @brief Copy the entries in 'this' where corresponding 'ix' is nonzero, to v starting at start_index in 'v'.
  virtual void copyToStartingAt_w_pattern(hiopVector& vec, index_type start_index_in_dest, const hiopVector& ix) const = 0;

  /**
   * @brief copy 'c' and `d` into `this`, according to the map 'c_map` and `d_map`, respectively.
   * e.g., this[c_map[i]] = c[i];
   *
   * @param[in] c - the 1st source vector
   * @param[in] d - the 2nd source vector
   * @param[in] c_map - the element mapping bewteen `this` and `c`
   * @param[in] d_map - the element mapping bewteen `this` and `d`
   *
   * @pre the size of `this` = the size of `c` + the size of `d`.
   * @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
   */
  virtual void copy_from_two_vec_w_pattern(const hiopVector& c,
                                           const hiopVectorInt& c_map,
                                           const hiopVector& d,
                                           const hiopVectorInt& d_map) = 0;

  /**
   * @brief split `this` to `c` and `d`, according to the map 'c_map` and `d_map`, respectively.
   *
   * @param[out] c - the 1st destination vector
   * @param[out] d - the 2nd destination vector
   * @param[in] c_map - the element mapping bewteen `this` and `c`
   * @param[in] d_map - the element mapping bewteen `this` and `d`
   *
   * @pre the size of `this` = the size of `c` + the size of `d`.
   * @pre `c_map` \Union `d_map` = {0, ..., size_of_this_vec-1}
   */
  virtual void copy_to_two_vec_w_pattern(hiopVector& c,
                                         const hiopVectorInt& c_map,
                                         hiopVector& d,
                                         const hiopVectorInt& d_map) const = 0;

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
  virtual void startingAtCopyToStartingAt(index_type start_idx_in_src,
                                          hiopVector& dest,
                                          index_type start_idx_dest,
                                          size_type num_elems = -1) const = 0;

  /**
   * @brief Copy elements of `this` vector to `dest` with offsets.
   *
   * Copy `this` (source) starting at `start_idx_in_src` to `dest`
   * starting at index 'int start_idx_dest'. If num_elems>=0, 'num_elems' will be copied;
   *
   * @param[in] start_idx_in_src - position in `this` from where to copy
   * @param[out] dest - destination vector to where to copy vector data
   * @param[in] start_idx_dest - position in `dest` to where to copy
   * @param[in] selec_dest - a nonzero pattern that implies which elements of `dest` should be overwritten
   * @param[in] num_elems - number of elements to copy
   *
   * @pre start_idx_in_src <= n_local_
   * @pre start_idx_dest   <= dest.n_local_
   * @pre `this` and `dest` are not distributed
   * @post If num_elems >= 0, `num_elems` will be copied
   * @post If num_elems < 0, elements will be copied till the end of
   * either source (`this`) or `dest` is reached
   */
  virtual void startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                    hiopVector& dest,
                                                    index_type start_idx_dest,
                                                    const hiopVector& selec_dest,
                                                    size_type num_elems = -1) const = 0;

  /**
   * @brief L2 vector norm.
   *
   * @post `this` is not modified
   */
  virtual double twonorm() const = 0;

  /**
   * @brief L-infinity (max) vector norm.
   *
   * @post `this` is not modified
   */
  virtual double infnorm() const = 0;

  /**
   * @brief Local L-infinity (max) vector norm.
   *
   * @pre  `this` is not empty vector
   * @post `this` is not modified
   */
  virtual double infnorm_local() const = 0;

  /**
   * @brief 1-norm of `this` vector.
   *
   * @post `this` is not modified
   */
  virtual double onenorm() const = 0;

  /**
   * @brief Local 1-norm of `this` vector.
   *
   * @pre  `this` is not empty vector
   * @post `this` is not modified
   */
  virtual double onenorm_local() const = 0;

  /**
   * @brief Multiply `this` by `vec` elementwise and store result in `this`.
   *
   * @param[in] vec - input vector
   *
   * @pre  `this` and `vec` have same partitioning.
   * @post `vec` is not modified
   */
  virtual void componentMult(const hiopVector& vec) = 0;

  /**
   * @brief Divide `this` vector elemenwise in-place by `vec`.
   *
   * @param[in] vec - input vector
   *
   * @pre `this` and `vec` have same partitioning.
   * @pre vec[i] != 0 forall i
   * @post `vec` is not modified
   */
  virtual void componentDiv(const hiopVector& vec) = 0;

  /**
   * @brief Divide `this` vector elemenwise in-place by `vec`
   * with pattern selection.
   *
   * @param[in] vec - input vector
   * @param[in] select - pattern selection
   *
   * @pre `this`, `select` and `vec` have same partitioning.
   * @pre vec[i] != 0 when select[i] = 1
   * @post `vec` and `select` are not modified
   */
  virtual void componentDiv_w_selectPattern(const hiopVector& vec, const hiopVector& select) = 0;

  /**
   * @brief Set `this` vector elemenwise to the minimum of itself and the given `constant`
   *
   * @param[in] constant - input constant
   */
  virtual void component_min(const double constant) = 0;

  /**
   * @brief Set `this` vector elemenwise to the minimum of itself and the corresponding component of 'vec'.
   *
   * @param[in] vec - input vector
   *
   * @pre `this` and `vec` have same partitioning.
   * @post `vec` is not modified
   */
  virtual void component_min(const hiopVector& vec) = 0;

  /**
   * @brief Set `this` vector elemenwise to the maximum of itself and the given `constant`
   *
   * @param[in] constant - input constant
   */
  virtual void component_max(const double constant) = 0;

  /**
   * @brief Set `this` vector elemenwise to the maximum of itself and the corresponding component of 'vec'.
   *
   * @param[in] vec - input vector
   *
   * @pre `this` and `vec` have same partitioning.
   * @post `vec` is not modified
   */
  virtual void component_max(const hiopVector& v) = 0;

  /**
   * @brief Set each component to its absolute value
   */
  virtual void component_abs() = 0;

  /**
   * @brief Apply sign function to each component
   */
  virtual void component_sgn() = 0;

  /**
   * @brief compute square root of each element
   * @pre all the elements are non-negative
   */
  virtual void component_sqrt() = 0;

  /**
   * @brief Scale `this` vector by `c`
   *
   * @param[in] c - scaling factor
   */
  virtual void scale(double c) = 0;

  /**
   * @brief Implementation of AXPY kernel. this += alpha * x
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axpy-ed to this (size equal to size of this)
   *
   * @pre `this` and `xvec` have same partitioning.
   * @post `xvec` is not modified
   *
   * @note Consider implementing with BLAS call (<D>AXPY)
   */
  virtual void axpy(double alpha, const hiopVector& xvec) = 0;

  /**
   * @brief Implementation of AXPY kernel, for selected entries.
   * this[i] += alpha * x[i] for all i where select[i] == 1.0;
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axpy-ed to this (size equal to size of this)
   * @param[in] select - pattern selection
   *
   * @pre `this`, `select` and `xvec` have same partitioning.
   * @post `xvec` and `select` is not modified
   *
   * @note Consider implementing with BLAS call (<D>AXPY)
   */
  virtual void axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select) = 0;

  /**
   * @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axpy-ed to this
   *  (size equal to size of i and less than or equal to size of this)
   * @param[in] i - vector of indexes in this to which the axpy operation is performed
   *  (size equal to size of x and less than or equal to size of this)
   *
   * @pre The entries of i must be valid (zero-based) indexes in this
   */
  virtual void axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i) = 0;

  /**
   * @brief this[i] += alpha*x[i]*z[i] forall i
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axzpy-ed to this
   * @param[in] zvec - vector of doubles to be axzpy-ed to this
   *
   * @pre `this`, `xvec` and `zvec` have same partitioning.
   * @post `xvec` and `zvec` are not modified
   */
  virtual void axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec) = 0;

  /**
   * @brief this[i] += alpha*x[i]/z[i] forall i
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axdzpy-ed to this
   * @param[in] zvec - vector of doubles to be axdzpy-ed to this
   *
   * @pre `this`, `xvec` and `zvec` have same partitioning.
   * @pre zvec[i] != 0 forall i
   * @post `xvec` and `zvec` are not modified
   */
  virtual void axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec) = 0;

  /**
   * @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of doubles to be axdzpy-ed to this
   * @param[in] zvec - vector of doubles to be axdzpy-ed to this
   * @param[in] select - pattern selection
   *
   * @pre `this`, `xvec`, `zvec` and `select` have same partitioning.
   * @pre zvec[i] != 0 when select[i] = 1
   * @post `xvec`, `zvec` and `select` are not modified
   */
  virtual void axdzpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& zvec, const hiopVector& select) = 0;

  /**
   * @brief this[i] += c forall i
   *
   * @param[in] c - a constant that is added to each element of `this`
   */
  virtual void addConstant(double c) = 0;

  /**
   * @brief this[i] += c forall i with pattern selection
   *
   * @param[in] c - a constant that is added to each element of `this`
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @post `select` is not modified
   */
  virtual void addConstant_w_patternSelect(double c, const hiopVector& select) = 0;

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
  virtual double dotProductWith(const hiopVector& vec) const = 0;

  /**
   * @brief Negate all vector elements
   *
   * @note Consider implementing with BLAS call (<D>SCAL)
   */
  virtual void negate() = 0;

  /**
   * @brief Invert vector elements
   *
   * @pre this[i] != 0 forall i
   * @post `this` is overwritten
   *
   * @todo Consider having HiOp-wide `small_real` constant defined.
   */
  virtual void invert() = 0;

  /**
   * @brief Sum all selected log(this[i])
   *
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @pre Selected elements of `this` are > 0.
   * @post `this` and `select` are not modified
   *
   * @warning This is local method only!
   */
  virtual double logBarrier_local(const hiopVector& select) const = 0;

  /**
   * @brief adds the gradient of the log barrier, namely this[i]=this[i]+alpha*1/select(x[i])
   *
   * @param[in] alpha - scaling factor
   * @param[in] xvec - vector of the gradient of the log barrier
   * @param[in] select - pattern selection
   *
   * @pre `this`, `xvec` and `select` have same partitioning.
   * @pre xvec[i] != 0 forall i
   * @post `xvec` and `select` are not modified
   */
  virtual void addLogBarrierGrad(double alpha, const hiopVector& xvec, const hiopVector& select) = 0;

  /**
   * @brief Sum all elements
   */
  virtual double sum_local() const = 0;

  /**
   * @brief Linear damping term
   * Computes the log barrier's linear damping term of the Filter-IPM method of
   * WaectherBiegler (see paper, section 3.7).
   * Essentially compute  kappa_d*mu* \sum { this[i] | ixleft[i]==1 and ixright[i]==0 }
   *
   * @param[in] ixleft - pattern selection 1
   * @param[in] ixright - pattern selection 2
   * @param[in] mu - constant used in the above equation
   * @param[in] kappa_d - constant used in the above equation
   *
   * @pre `this`, `ixleft` and `ixright` have same partitioning.
   * @pre `ixleft` and `ixright` elements are 0 or 1 only.
   * @post `this`, `ixleft` and `ixright` are not modified
   *
   * @warning This is local method only!
   */
  virtual double linearDampingTerm_local(const hiopVector& ixleft,
                                         const hiopVector& ixright,
                                         const double& mu,
                                         const double& kappa_d) const = 0;

  /**
   * @brief add linear damping term
   * Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of
   * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
   *
   * Supports distributed/MPI vectors, but performs only elementwise operations and do not
   * require communication.
   *
   * This method is used to add gradient contributions from the (linear) damping term used
   * to handle unbounded problems. The damping terms are used for variables that are
   * bounded on one side only.
   *
   * @param[in] ixleft - pattern selection 1
   * @param[in] ixright - pattern selection 2
   * @param[in] alpha - constant used in the above equation
   * @param[in] ct - constant used in the above equation
   */
  virtual void addLinearDampingTerm(const hiopVector& ixleft,
                                    const hiopVector& ixright,
                                    const double& alpha,
                                    const double& ct) = 0;

  /**
   * @brief Check if all elements of the vector are positive
   *
   * @post `this` is not modified
   */
  virtual int allPositive() = 0;

  /**
   * @brief Checks if selected elements of `this` are positive.
   *
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @pre Elements of `select` are either 0 or 1.
   * @post `select` is not modified
   */
  virtual int allPositive_w_patternSelect(const hiopVector& select) = 0;

  /**
   * @brief Return the minimum value in `this` vector
   */
  virtual double min() const = 0;

  /**
   * @brief Return the minimum value in selected elements of `this` vector
   *
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @pre Elements of `select` are either 0 or 1.
   * @post `select` is not modified
   */
  virtual double min_w_pattern(const hiopVector& select) const = 0;

  /**
   * @brief Return the minimum value in this vector, and the index at which it occurs
   *
   * @param[out] minval - minimum value in `this` vector
   * @param[out] index - the index at which the minimum occurs.
   */
  virtual void min(double& minval, int& index) const = 0;

  /**
   * @brief Project solution (`this`) into bounds
   * This method is used to shift the initial point into the lower/upper bounds
   *
   * @param[in] xlo - lower bounds
   * @param[in] ixl - indices for the lower bounds
   * @param[in] xup - upper bounds
   * @param[in] ixu - indices for the upper bounds
   * @param[in] kappa1 - user parameter which is used to control the projection
   * @param[in] kappa2 - user parameter which is used to control the projection
   *
   * @pre `this`, `xlo`, `ixl`, `xup` and `ixu` have same partitioning.
   * @pre `ixl` and `ixu` elements are 0 or 1 only.
   * @post `xlo`, `ixl`, `xup` and `ixu` are not modified
   *
   * @warning This is local method only!
   */
  virtual bool projectIntoBounds_local(const hiopVector& xlo,
                                       const hiopVector& ixl,
                                       const hiopVector& xup,
                                       const hiopVector& ixu,
                                       double kappa1,
                                       double kappa2) = 0;

  /**
   * @brief max{a\in(0,1]| x+dvec >=(1-tau)x}, where x is `this`
   *
   * @param[in] dvec - vector used in the above equation
   * @param[in] tau - user parameter
   *
   * @pre `this` and `dvec` have same partitioning.
   * @post `this` and `dvec` are not modified
   *
   * @warning This is local method only!
   */
  virtual double fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const = 0;

  /**
   * @brief max{a\in(0,1]| x+dvec >=(1-tau)x} with pattern select
   *
   * @param[in] dvec - vector used in the above equation
   * @param[in] tau - user parameter
   * @param[in] select - pattern selection
   *
   * @pre `this`, `select` and `dvec` have same partitioning.
   * @pre Elements of `select` are either 0 or 1.
   * @post `this`, `select` and `dvec` are not modified
   *
   * @warning This is local method only!
   */
  virtual double fractionToTheBdry_w_pattern_local(const hiopVector& dvec,
                                                   const double& tau,
                                                   const hiopVector& select) const = 0;

  /**
   * @brief Set elements of `this` to zero based on `select`.
   *
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @pre Elements of `select` are either 0 or 1.
   * @post `select` is not modified
   */
  virtual void selectPattern(const hiopVector& select) = 0;
  /// @brief checks whether entries in this matches pattern in ix

  /**
   * @brief Checks if `this` matches nonzero pattern of `select`.
   *
   * @param[in] select - pattern selection
   *
   * @pre `this` and `select` have same partitioning.
   * @pre Elements of `select` are either 0 or 1.
   * @post `select` is not modified
   */
  virtual bool matchesPattern(const hiopVector& select) = 0;

  /**
   * @brief Adjusts duals.
   *
   * @param[in] xvec - vector used in the computation
   * @param[in] ixvec - indices used in the computation
   * @param[in] mu - constant
   * @param[in] kappa - user parameter
   *
   * @pre `this`, `xvec` and `ixvec` have same partitioning.
   * @pre Elements of `ixvec` are either 0 or 1.
   * @post `xvec` and `ixvec` are not modified
   *
   * @note Implementation probably inefficient.
   */
  virtual void adjustDuals_plh(const hiopVector& xvec, const hiopVector& ixvec, const double& mu, const double& kappa) = 0;

  /**
   * @brief Check if all elements of the vector are zero
   *
   * @post `this` is not modified
   * @todo: add unit test, or should we remove this function?
   */
  virtual bool is_zero() const = 0;

  /**
   * @brief Returns true if any element of `this` is NaN.
   *
   * @post `this` is not modified
   *
   * @warning This is local method only!
   */
  virtual bool isnan_local() const = 0;

  /**
   * @brief Returns true if any element of `this` is Inf.
   *
   * @post `this` is not modified
   *
   * @warning This is local method only!
   */
  /// @brief check for infs in the local vector
  virtual bool isinf_local() const = 0;

  /**
   * @brief Returns true if all elements of `this` are finite.
   *
   * @post `this` is not modified
   *
   * @warning This is local method only!
   */
  virtual bool isfinite_local() const = 0;

  /**
   * @brief Prints vector data to a file in Matlab format.
   *
   * @pre Vector data was moved from the memory space to the host mirror.
   */
  virtual void print(FILE* file = nullptr, const char* message = nullptr, int max_elems = -1, int rank = -1) const = 0;

  /**
   * @brief allocates a vector that mirrors this, but doesn't copy the values
   */
  virtual hiopVector* alloc_clone() const = 0;

  /**
   * @brief allocates a vector that mirrors this, and copies the values
   */
  virtual hiopVector* new_copy() const = 0;

  /**
   * @brief return the global size of `this` vector
   */
  virtual size_type get_size() const { return n_; }

  /**
   * @brief return the size of the local part of `this` vector
   */
  virtual size_type get_local_size() const = 0;

  /**
   * @brief accessor to the local data of `this` vector
   */
  virtual double* local_data() = 0;

  /**
   * @brief accessor to the local data of `this` vector
   */
  virtual const double* local_data_const() const = 0;

  /**
   * @brief accessor to the local data of `this` vector
   */
  virtual double* local_data_host() = 0;

  /**
   * @brief accessor to the local data of `this` vector
   */
  virtual const double* local_data_host_const() const = 0;

  /**
   * @brief get the number of values that are less than the given tolerance 'val'.
   *
   * @param[in] val - tolerance
   *
   * @post `val` is not modified
   * @todo: add unit test
   */
  virtual size_type numOfElemsLessThan(const double& val) const = 0;

  /**
   * @brief get the number of values whose absolute value are less than the given tolerance 'val'.
   *
   * @param[in] val - tolerance
   *
   * @post `val` is not modified
   * @todo: add unit test
   */
  virtual size_type numOfElemsAbsLessThan(const double& val) const = 0;

  /**
   * @brief set enum-type array 'arr', starting at `start` and ending at `end`,
   * to the values in array `arr_src` from 'start_src`
   *
   * @param[out] arr - array of used to define hiopInterfaceBase::NonlinearityType
   * @param[in] start - the first position to update `arr`
   * @param[in] end - the last position to update `arr`
   * @param[in] arr_src - the source array of type hiopInterfaceBase::NonlinearityType
   * @param[in] start_src - the first position of `arr_src` to be copied from
   *
   * @pre the size of `arr_src` >= start - end + start_src
   * @pre the size of `arr` >= start - end
   * @post `arr` is modified
   * @post `arr_src` is not modified
   * @todo: add unit test
   */
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType* arr_src,
                                 const int start_src) const = 0;

  /**
   * @brief set enum-type array 'arr', starting at `start` and ending at `end`,
   * to the values in array `arr_src` from 'start_src`
   *
   * @param[out] arr - array of used to define hiopInterfaceBase::NonlinearityType
   * @param[in] start - the first position to update `arr`
   * @param[in] end - the last position to update `arr`
   * @param[in] arr_src - constant of type hiopInterfaceBase::NonlinearityType
   *
   * @pre the size of `arr` >= start - end
   * @post `arr` is modified
   * @post `arr_src` is not modified
   * @todo: add unit test
   */
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType arr_src) const = 0;

  /**
   * @brief check if `this` vector is identical to `vec`
   *
   * @param[in] vec - vector used to be compared with `this`
   * @todo: add unit test, or should we remove this function?
   */
  virtual bool is_equal(const hiopVector& vec) const = 0;

protected:
  size_type n_;  // we assume sequential data
protected:
  /**
   * @brief for internal use only; derived classes may use copy constructor and always allocate data_
   */
  hiopVector(const hiopVector& v)
      : n_(v.n_) {};
};

}  // namespace hiop
