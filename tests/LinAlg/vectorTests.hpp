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
 * @file vectorTests.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 *
 */
#pragma once

#include <iostream>
#include <cmath>
#include <cfloat>
#include <assert.h>
#include <limits>
#include <functional>

#include <hiopVector.hpp>
#include <hiopVectorInt.hpp>
#include <LinAlgFactory.hpp>
#include "testBase.hpp"

namespace hiop
{
namespace tests
{

/**
 * @brief Collection of tests for abstract hiopVector implementations.
 *
 * This class contains implementation of all vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 * @pre All input vectors should be allocated to the same size and have
 * the same partitioning.
 *
 * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
 *
 * @post All tests return `true` on all ranks if the test fails on any rank
 * and return `false` otherwise.
 *
 */
class VectorTests : public TestBase
{
public:
  VectorTests() {}
  virtual ~VectorTests() {}

  /*
   * this[i] = 0
   */
  bool vectorSetToZero(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(one);

    v.setToZero();

    int fail = verifyAnswer(&v, zero);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /// Test get_size() method of hiop vector implementation
  bool vectorGetSize(hiop::hiopVector& x, global_ordinal_type answer, const int rank)
  {
    bool fail = (x.get_size() != answer);
    printMessage(fail, __func__, rank);
    return fail;
  }

  /// Test setToConstant method of hiop vector implementation
  bool vectorSetToConstant(hiop::hiopVector& x, const int rank)
  {
    int fail = 0;
    local_ordinal_type N = getLocalSize(&x);

    for(local_ordinal_type i = 0; i < N; ++i) {
      setLocalElement(&x, i, zero);
    }

    x.setToConstant(one);

    fail = verifyAnswer(&x, one);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &x);
  }

  /// Test set_to_random_uniform method of hiop vector implementation
  bool vector_set_to_random_uniform(hiop::hiopVector& x, const int rank)
  {
    int fail = 0;
    local_ordinal_type N = getLocalSize(&x);

    for(local_ordinal_type i = 0; i < N; ++i) {
      setLocalElement(&x, i, zero);
    }

    x.set_to_random_uniform(one, two);

    fail = verifyAnswer(&x, one, two);
    if(fail > 0) {
      std::cout << "num fails = " << fail << std::endl;
    }

    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test method:
   * forall n in n_local if (pattern[n] != 0.0) this[n] = x_val
   */
  bool vectorSetToConstant_w_patternSelect(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    static const real_type x_val = two;

    x.setToConstant(zero);
    pattern.setToConstant(one);

    // Ensure that a single element (globally) is
    // set to zero in the pattern
    if(rank == 0) setLocalElement(&pattern, N - 1, zero);

    x.setToConstant_w_patternSelect(x_val, pattern);

    // Check that the last element of rank zero's vector is
    // zero, and that x_val was added to all other elements
    const int fail =
        verifyAnswer(&x, [=](local_ordinal_type i) -> real_type { return (rank == 0 && i == N - 1) ? zero : x_val; });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /*
   * Test for function that copies data from x to this.
   */
  bool vectorCopyFrom(hiop::hiopVector& v, hiop::hiopVector& from, const int rank)
  {
    local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == from.get_size());
    assert(N == getLocalSize(&from));

    from.setToConstant(one);
    v.setToConstant(two);
    v.copyFrom(from);
    int fail = verifyAnswer(&v, one);

    real_type* from_buffer = createLocalBuffer(N, three);
    v.copyFrom(from_buffer);
    fail += verifyAnswer(&v, three);

    deleteLocalBuffer(from_buffer);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /*
   * Test for function that copies to v the entries of from specified by idxs.
   */
  bool vector_copy_from_indexes(hiop::hiopVector& v, hiop::hiopVector& from, hiop::hiopVectorInt& idxs)
  {
    local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == idxs.get_local_size());
    assert(N == idxs.get_local_size());
    assert(N <= getLocalSize(&from));

    // copy indexes 0, 1, 2, ..., N (copy first N entries of from)
    idxs.linspace(0, 1);

    v.setToConstant(three);

    from.setToConstant(one);
    setLocalElement(&from, N - 1, two);

    v.copy_from_indexes(from, idxs);

    int fail = verifyAnswer(&v, [=](local_ordinal_type i) -> real_type { return (i == N - 1) ? two : one; });

    real_type* from_buffer = from.local_data();
    v.copy_from_indexes(from_buffer, idxs);
    fail += verifyAnswer(&v, [=](local_ordinal_type i) -> real_type { return (i == N - 1) ? two : one; });

    printMessage(fail, __func__);
    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test vector method for copying data from another vector
   * or data buffer.
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vectorCopyFromStarting(hiop::hiopVector& x, hiop::hiopVector& from, const int rank = 0)
  {
    int fail = 0;
    const local_ordinal_type Nx = getLocalSize(&x);
    const local_ordinal_type Nfrom = getLocalSize(&from);
    assert(Nx == x.get_size() && "This test cannot be ran with distributed vectors");
    assert(Nx > Nfrom);
    x.setToConstant(two);

    real_type* from_buffer = createLocalBuffer(Nx, one);

    x.copyFromStarting(1, from_buffer, Nx - 1);
    fail += verifyAnswer(&x, [=](local_ordinal_type i) -> real_type { return (i == 0) ? two : one; });
    deleteLocalBuffer(from_buffer);

    x.setToConstant(two);
    from.setToConstant(one);
    x.copyFromStarting(Nx - Nfrom, from);
    fail += verifyAnswer(&x, [=](local_ordinal_type i) -> real_type { return (i < (Nx - Nfrom)) ? two : one; });

    x.setToConstant(two);
    from.setToConstant(one);
    x.copyFromStarting(1, from);
    fail += verifyAnswer(&x, [=](local_ordinal_type i) -> real_type { return (i < 1 || i > (Nfrom)) ? two : one; });

    // Testing copying from a zero size vector
    hiop::hiopVector* zero = hiop::LinearAlgebraFactory::create_vector(mem_space_, 0);
    zero->setToConstant(one);
    x.setToConstant(two);
    x.copyFromStarting(0, *zero);
    fail += verifyAnswer(&x, two);

    // Testing copying from a zero size array
    real_type* zero_buffer = createLocalBuffer(0, one);
    x.setToConstant(two);
    x.copyFromStarting(0, zero_buffer, 0);
    fail += verifyAnswer(&x, two);
    deleteLocalBuffer(zero_buffer);

    delete zero;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Tests function that copies from one vector to another, specifying
   * both the start index in the source and the destination.
   *
   * @pre `src` and `dest` are allocated to nonzero sizes and
   * size of `src` > size of `dest`.
   */
  bool vectorStartingAtCopyFromStartingAt(hiop::hiopVector& dest, hiop::hiopVector& src, const int rank = 0)
  {
    const local_ordinal_type Ndest = getLocalSize(&dest);
    const local_ordinal_type Nsrc = getLocalSize(&src);
    assert(Ndest == dest.get_size() && "This test cannot be run with distributed vectors");
    assert(Ndest < Nsrc && "This test assumes source is bigger than destination vector");

    const real_type dest_val = one;
    const real_type src_val = two;
    dest.setToConstant(dest_val);
    src.setToConstant(src_val);

    // Copy one element from `src` to `dest`
    local_ordinal_type start_dest = Ndest - 1;
    local_ordinal_type start_src = Nsrc / 2;
    dest.startingAtCopyFromStartingAt(start_dest, src, start_src);
    int fail = verifyAnswer(&dest, [=](local_ordinal_type i) -> real_type { return i == start_dest ? src_val : dest_val; });
    // Restore destination values
    dest.setToConstant(dest_val);

    // Overwrite all `dest` elements with last Ndest elements of `src`
    start_dest = 0;
    start_src = Nsrc - Ndest;
    dest.startingAtCopyFromStartingAt(start_dest, src, start_src);
    fail += verifyAnswer(&dest, src_val);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dest);
  }

  /**
   * Test for function that copies data from `this` to a data buffer.
   *
   * @note This test calls `local_data` vector method. Here this is OK,
   * because for as long copies between vectors and bufers are implemented
   * as public methods, `local_data` will be a public method, as well.
   */
  bool vectorCopyTo(hiop::hiopVector& v, hiop::hiopVector& to, const int rank)
  {
    assert(v.get_size() == to.get_size());
    assert(getLocalSize(&v) == getLocalSize(&to));

    to.setToConstant(one);
    v.setToConstant(two);

    real_type* todata = to.local_data();
    v.copyTo(todata);

    int fail = verifyAnswer(&to, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test vector method for copying data to another vector
   * starting from prescribed index in destination vector.
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vectorCopyToStarting(hiop::hiopVector& to, hiop::hiopVector& from, const int rank = 0)
  {
    const local_ordinal_type dest_size = getLocalSize(&to);
    const local_ordinal_type src_size = getLocalSize(&from);
    assert(dest_size == to.get_size() && "This test cannot be ran with distributed vectors");
    assert(dest_size > src_size && "Must pass in a destination vector larger than source vector");

    const int start_idx = dest_size - src_size;
    const real_type from_val = one;
    const real_type to_val = two;

    from.setToConstant(from_val);
    to.setToConstant(to_val);

    from.copyToStarting(to, start_idx);

    /*
     * Test that values at indices less than the start
     * index remain unchanged, and that values at indices
     * greater than or equal to the start idx are set
     * to the source value
     */
    int fail = verifyAnswer(&to, [=](local_ordinal_type i) -> real_type { return i < start_idx ? to_val : from_val; });

    // Testing copying from a zero size vector
    hiop::hiopVector* zero = hiop::LinearAlgebraFactory::create_vector(mem_space_, 0);
    zero->setToConstant(one);
    to.setToConstant(to_val);
    zero->copyToStarting(to, 0);
    fail += verifyAnswer(&to, to_val);

    delete zero;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &from);
  }

  /**
   * @brief Test vector method for copying data to another vector
   * starting from prescribed index in destination vector.
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vectorCopyToStartingAt_w_pattern(hiop::hiopVector& from,
                                        hiop::hiopVector& to,
                                        hiop::hiopVector& pattern,
                                        const int rank = 0)
  {
    const local_ordinal_type to_size = getLocalSize(&to);
    const local_ordinal_type from_size = getLocalSize(&from);
    assert(to_size == to.get_size() && to_size > from_size + 2 && "This test cannot be ran with distributed vectors");
    assert(getLocalSize(&pattern) == from_size && from_size > 1 && "pattern_size must be equal to the source size");
    const int start_idx = to_size - from_size;
    const real_type from_val = one;
    const real_type from_val_st_ed = three;
    const real_type to_val = two;

    from.setToConstant(from_val);
    to.setToConstant(to_val);
    pattern.setToConstant(zero);
    if(rank == 0) {
      setLocalElement(&from, 0, from_val_st_ed);
      setLocalElement(&from, from_size - 1, from_val_st_ed);
      setLocalElement(&pattern, 0, one);
      setLocalElement(&pattern, from_size - 1, one);
    }

    from.copyToStartingAt_w_pattern(to, start_idx, pattern);

    // Check that the start and end values of `from' vector are copied to the `to' vector
    const int fail = verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      return (rank == 0 && (i == start_idx || i == start_idx + 1) ? from_val_st_ed : to_val);
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &from);
  }

  /**
   * @brief Test vector method for copying data from another two vectors
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vector_copy_from_two_vec(hiop::hiopVector& cd,
                                hiop::hiopVector& c,
                                hiop::hiopVectorInt& c_map,
                                hiop::hiopVector& d,
                                hiop::hiopVectorInt& d_map,
                                const int rank = 0)
  {
    const local_ordinal_type cd_size = getLocalSize(&cd);
    const local_ordinal_type c_size = getLocalSize(&c);
    const local_ordinal_type d_size = getLocalSize(&d);
    const local_ordinal_type c_map_size = c_map.get_local_size();
    const local_ordinal_type d_map_size = d_map.get_local_size();

    assert(c_size == c_map_size && "size doesn't match");
    assert(d_size == d_map_size && "size doesn't match");
    assert(c_size + d_size == cd_size && "size doesn't match");

    const real_type c_val = one;
    const real_type d_val = two;

    c.setToConstant(c_val);
    d.setToConstant(d_val);

    c_map.linspace(0, 1);
    d_map.linspace(c_size, 1);

    cd.copy_from_two_vec_w_pattern(c, c_map, d, d_map);

    int fail = verifyAnswer(&cd, [=](local_ordinal_type i) -> real_type { return i < c_size ? c_val : d_val; });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &cd);
  }

  /**
   * @brief Test vector method for copying data to another two vectors
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vector_copy_to_two_vec(hiop::hiopVector& cd,
                              hiop::hiopVector& c,
                              hiop::hiopVectorInt& c_map,
                              hiop::hiopVector& d,
                              hiop::hiopVectorInt& d_map,
                              const int rank = 0)
  {
    const local_ordinal_type cd_size = getLocalSize(&cd);
    const local_ordinal_type c_size = getLocalSize(&c);
    const local_ordinal_type d_size = getLocalSize(&d);

    const local_ordinal_type c_map_size = c_map.get_local_size();
    const local_ordinal_type d_map_size = d_map.get_local_size();
    assert(c_size == c_map_size && "size doesn't match");
    assert(d_size == d_map_size && "size doesn't match");
    assert(c_size + d_size == cd_size && "size doesn't match");

    const real_type cd_val = two;

    c_map.linspace(0, 1);
    d_map.linspace(c_size, 1);

    cd.setToConstant(cd_val);
    cd.copy_to_two_vec_w_pattern(c, c_map, d, d_map);

    int fail = verifyAnswer(&c, cd_val);
    fail += verifyAnswer(&d, cd_val);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &cd);
  }

  /**
   * @brief Test vector method for copying data to another vector
   * starting from prescribed indices in source and destination
   * vectors.
   *
   * @pre Vectors are not distributed.
   * @pre Memory space for hiop::LinearAlgebraFactory is set appropriately
   */
  bool vectorStartingAtCopyToStartingAt(hiop::hiopVector& to, hiop::hiopVector& from, const int rank = 0)
  {
    const local_ordinal_type dest_size = getLocalSize(&to);
    const local_ordinal_type src_size = getLocalSize(&from);
    assert(dest_size == to.get_size() && "This test cannot be ran with distributed vectors");
    assert(dest_size > src_size && "Must pass in a destination vector larger than source vector");

    const real_type from_val = one;
    const real_type to_val = two;
    int start_idx_src = 0;
    int start_idx_dst = 0;
    int fail = 0;
    int num_elements_to_copy = -1;

    // Iteratively checking various edge cases for calls to the function

    hiop::hiopVector* zero = hiop::LinearAlgebraFactory::create_vector(mem_space_, 0);

    // Copying from a size 0 vector
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    zero->startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, to_val);

    // Copying to a size 0 vector
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, *zero, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(zero, 0);

    // Copying 0 elements
    num_elements_to_copy = 0;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, to_val);

    // Copying from start of from to start of to
    num_elements_to_copy = src_size;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      int tmp;
      if(num_elements_to_copy == -1) {
        tmp = src_size;
      } else {
        tmp = num_elements_to_copy;
      }
      const bool isValueCopied = (i >= start_idx_dst && i < start_idx_dst + tmp);
      return isValueCopied ? from_val : to_val;
    });

    // Copying from start of from to end of to
    start_idx_dst = dest_size - src_size;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      int tmp;
      if(num_elements_to_copy == -1) {
        tmp = src_size;
      } else {
        tmp = num_elements_to_copy;
      }
      const bool isValueCopied = (i >= start_idx_dst && i < start_idx_dst + tmp);
      return isValueCopied ? from_val : to_val;
    });

    // Not copying all elemtents
    num_elements_to_copy = num_elements_to_copy / 2;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      int tmp;
      if(num_elements_to_copy == -1) {
        tmp = src_size;
      } else {
        tmp = num_elements_to_copy;
      }
      const bool isValueCopied = (i >= start_idx_dst && i < start_idx_dst + tmp);
      return isValueCopied ? from_val : to_val;
    });

    // Passing -1 as the number of elements
    num_elements_to_copy = -1;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      int tmp;
      if(num_elements_to_copy == -1) {
        tmp = src_size;
      } else {
        tmp = num_elements_to_copy;
      }
      const bool isValueCopied = (i >= start_idx_dst && i < start_idx_dst + tmp);
      return isValueCopied ? from_val : to_val;
    });

    // Passing starting indices equal to the sizes
    start_idx_dst = src_size;
    start_idx_dst = dest_size;
    from.setToConstant(from_val);
    to.setToConstant(to_val);
    from.startingAtCopyToStartingAt(start_idx_src, to, start_idx_dst, num_elements_to_copy);

    fail += verifyAnswer(&to, [=](local_ordinal_type i) -> real_type {
      int tmp;
      if(num_elements_to_copy == -1) {
        tmp = src_size;
      } else {
        tmp = num_elements_to_copy;
      }
      const bool isValueCopied = (i >= start_idx_dst && i < start_idx_dst + tmp);
      return isValueCopied ? from_val : to_val;
    });

    delete zero;
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &to);
  }

  /**
   * @brief Test:
   * this[i] = (pattern[i] == 0 ? 0 : this[i])
   */
  bool vectorSelectPattern(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(x.get_size() == pattern.get_size());
    assert(N == getLocalSize(&pattern));
    const real_type x_val = two;

    x.setToConstant(x_val);
    pattern.setToConstant(one);
    if(rank == 0) setLocalElement(&pattern, N - 1, zero);

    x.selectPattern(pattern);

    const int fail = verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? zero : x_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test: this[i] *= alpha
   */
  bool vectorScale(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(half);
    v.scale(half);

    int fail = verifyAnswer(&v, quarter);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] *= x[i]
   */
  bool vectorComponentMult(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(half);
    x.setToConstant(half);

    v.componentMult(x);

    int fail = verifyAnswer(&v, quarter);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] /= x[i]
   */
  bool vectorComponentDiv(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(one);
    x.setToConstant(two);

    v.componentDiv(x);

    int fail = verifyAnswer(&v, half);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = (pattern[i] == 0 ? 0 : this[i]/x[i])
   */
  bool vectorComponentDiv_p_selectPattern(hiop::hiopVector& v,
                                          hiop::hiopVector& x,
                                          hiop::hiopVector& pattern,
                                          const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(v.get_size() == pattern.get_size());
    assert(N == getLocalSize(&x));
    assert(N == getLocalSize(&pattern));
    const real_type x_val = one;
    const real_type v_val = half;

    x.setToConstant(x_val);
    v.setToConstant(v_val);
    pattern.setToConstant(one);
    if(rank == 0) {
      setLocalElement(&pattern, N - 1, zero);
      setLocalElement(&x, N - 1, zero);
    }

    v.componentDiv_w_selectPattern(x, pattern);

    const int fail = verifyAnswer(&v, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? zero : v_val / x_val;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = min(this[i], constant)
   */
  bool vector_component_min(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(one);

    v.component_min(half);

    int fail = verifyAnswer(&v, half);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = min(this[i], x[i])
   */
  bool vector_component_min(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(one);
    x.setToConstant(half);

    v.component_min(x);

    int fail = verifyAnswer(&v, half);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = max(this[i], constant)
   */
  bool vector_component_max(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(one);

    v.component_max(two);

    int fail = verifyAnswer(&v, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = max(this[i], x[i])
   */
  bool vector_component_max(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    assert(v.get_size() == x.get_size());
    assert(getLocalSize(&v) == getLocalSize(&x));

    v.setToConstant(one);
    x.setToConstant(two);

    v.component_max(x);

    int fail = verifyAnswer(&v, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test: this[i] = abs(this[i])
   */
  bool vector_component_abs(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    static const real_type x_val = -quarter;

    x.setToConstant(x_val);

    const real_type expected = half;
    if(rank == 0) {
      setLocalElement(&x, N - 1, expected);
    }

    x.component_abs();

    const int fail = verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? fabs(expected) : fabs(x_val);
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test: this[i] = sqrt(this[i])
   */
  bool vector_component_sqrt(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    static const real_type x_val = quarter;

    x.setToConstant(x_val);

    const real_type expected = two * two;
    if(rank == 0) {
      setLocalElement(&x, N - 1, expected);
    }

    x.component_sqrt();

    const int fail = verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? fabs(two) : fabs(half);
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test: this[i] = sgn(this[i])
   */
  bool vector_component_sgn(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    static const real_type x_val = -quarter;

    x.setToConstant(x_val);

    if(rank == 0) {
      setLocalElement(&x, N - 1, half);
    }

    x.component_sgn();

    const int fail = verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? one : -one;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test computing 1-norm ||v||  of vector v
   *                                   1
   */
  bool vectorOnenorm(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(-one);
    real_type actual = v.onenorm();
    real_type expected = static_cast<real_type>(v.get_size());

    int fail = (actual != expected);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test computing 2-norm ||v||  of vector v
   *                                   2
   */
  bool vectorTwonorm(hiop::hiopVector& v, const int rank)
  {
    v.setToConstant(-one);
    real_type actual = v.twonorm();
    const real_type expected = sqrt(static_cast<real_type>(v.get_size()));

    int fail = !isEqual(expected, actual);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * infinity-norm = max(abs(this[i]))
   *                  i
   */
  bool vectorInfnorm(hiop::hiopVector& v, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    const real_type expected = two;

    v.setToConstant(one);
    if(rank == 0) setLocalElement(&v, N - 1, -two);
    real_type actual = v.infnorm();

    int fail = (expected != actual);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * this[i] += alpha * x[i]
   */
  bool vectorAxpy(hiop::hiopVector& v, hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = half;
    const real_type x_val = two;
    const real_type v_val = two;

    x.setToConstant(x_val);
    v.setToConstant(v_val);

    v.axpy(alpha, x);

    const real_type expected = v_val + alpha * x_val;
    int fail = verifyAnswer(&v, expected);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * this[i] += alpha * x[i] * z[i]
   */
  bool vectorAxzpy(hiop::hiopVector& v, hiop::hiopVector& x, hiop::hiopVector& z, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = half;
    const real_type x_val = two;
    const real_type v_val = two;
    const real_type z_val = two;

    x.setToConstant(x_val);
    z.setToConstant(z_val);
    v.setToConstant(v_val);

    v.axzpy(alpha, x, z);

    const real_type expected = v_val + (alpha * x_val * z_val);
    const int fail = verifyAnswer(&v, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * this[i] += alpha * x[i] / z[i]
   */
  bool vectorAxdzpy(hiop::hiopVector& v, hiop::hiopVector& x, hiop::hiopVector& z, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = three;
    const real_type x_val = half;
    const real_type v_val = two;
    const real_type z_val = half;

    x.setToConstant(x_val);
    z.setToConstant(z_val);
    v.setToConstant(v_val);

    v.axdzpy(alpha, x, z);

    const real_type expected = v_val + (alpha * x_val / z_val);
    const int fail = verifyAnswer(&v, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * this[i] += alpha * x[i] / z[i]
   */
  bool vectorAxdzpy_w_patternSelect(hiop::hiopVector& v,
                                    hiop::hiopVector& x,
                                    hiop::hiopVector& z,
                                    hiop::hiopVector& pattern,
                                    const int rank)
  {
    const local_ordinal_type N = getLocalSize(&v);
    assert(v.get_size() == x.get_size());
    assert(N == getLocalSize(&x));

    const real_type alpha = three;
    const real_type x_val = half;
    const real_type v_val = two;
    const real_type z_val = half;

    x.setToConstant(x_val);
    z.setToConstant(z_val);
    v.setToConstant(v_val);
    pattern.setToConstant(one);
    if(rank == 0) {
      setLocalElement(&pattern, N - 1, zero);
      setLocalElement(&z, N - 1, zero);
    }

    const real_type expected = v_val + (alpha * x_val / z_val);
    v.axdzpy_w_pattern(alpha, x, z, pattern);

    const int fail = verifyAnswer(&v, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? v_val : expected;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &v);
  }

  /**
   * @brief Test:
   * this[i] += C forall i
   */
  bool vectorAddConstant(hiop::hiopVector& x, const int rank)
  {
    int fail = 0;

    x.setToConstant(zero);
    x.addConstant(two);

    fail = verifyAnswer(&x, two);
    printMessage(fail, __func__, rank);

    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * if (pattern[i] > 0.0) this[i] += C forall i
   */
  bool vectorAddConstant_w_patternSelect(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(pattern.get_size() == x.get_size());
    assert(N == getLocalSize(&pattern));
    const real_type x_val = half;

    pattern.setToConstant(one);
    if(rank == 0) setLocalElement(&pattern, N - 1, zero);

    x.setToConstant(zero);
    x.addConstant_w_patternSelect(x_val, pattern);

    const int fail =
        verifyAnswer(&x, [=](local_ordinal_type i) -> real_type { return (rank == 0 && i == N - 1) ? zero : x_val; });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * Dot product == \sum{this[i] * other[i]}
   */
  bool vectorDotProductWith(hiop::hiopVector& x, hiop::hiopVector& y, const int rank)
  {
    // Must use global size, as every rank will get global
    const global_ordinal_type N = x.get_size();
    assert(getLocalSize(&x) == getLocalSize(&y));

    x.setToConstant(one);
    y.setToConstant(two);

    const real_type expected = two * (real_type)N;
    const real_type actual = x.dotProductWith(y);
    const bool fail = !isEqual(actual, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * this[i] *= -1 forall i
   */
  bool vectorNegate(hiop::hiopVector& x, const int rank)
  {
    x.setToConstant(one);
    x.negate();
    const bool fail = verifyAnswer(&x, -one);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * this[i]^-1 forall i
   */
  bool vectorInvert(hiop::hiopVector& x, const int rank)
  {
    x.setToConstant(two);
    x.invert();
    const bool fail = verifyAnswer(&x, half);
    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * sum{ln(x[i]): pattern[i] = 1}
   */
  bool vectorLogBarrier(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));

    // Ensure that only N-1 elements of x are
    // used in the log calculation
    pattern.setToConstant(one);
    setLocalElement(&pattern, N - 1, zero);

    const real_type x_val = one;
    x.setToConstant(x_val);
    // Make sure pattern eliminates the correct element
    setLocalElement(&x, N - 1, 1000 * three);

    real_type expected = (N - 1) * std::log(x_val);
    real_type result = x.logBarrier_local(pattern);

    int fail = !isEqual(result, expected);

    // Make sure pattern eliminates the correct elements
    pattern.setToConstant(zero);
    setLocalElement(&pattern, N - 1, one);
    x.setToConstant(zero);
    setLocalElement(&x, N - 1, x_val);

    expected = std::log(x_val);
    result = x.logBarrier_local(pattern);
    fail += !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * sum{x[i]}
   */
  bool vector_sum_local(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);

    // Ensure that only N-1 elements of x are
    // used in the log calculation
    x.setToConstant(half);
    setLocalElement(&x, N - 1, two);

    real_type expected = (N - 1) * half + two;
    real_type result = x.sum_local();

    int fail = !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * if(pattern[i] == 1) this[i] += alpha /x[i] forall i
   */
  bool vectorAddLogBarrierGrad(hiop::hiopVector& x, hiop::hiopVector& y, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    assert(N == getLocalSize(&y));
    static const real_type alpha = half;
    static const real_type x_val = two;
    static const real_type y_val = two;

    pattern.setToConstant(one);
    x.setToConstant(x_val);
    y.setToConstant(y_val);

    if(rank == 0) setLocalElement(&pattern, N - 1, zero);

    x.addLogBarrierGrad(alpha, y, pattern);

    const real_type logBarrierGradVal = x_val + (alpha / y_val);
    const int fail = verifyAnswer(&x, [=](local_ordinal_type i) -> real_type {
      const bool isLastElementOnRank0 = (i == N - 1 && rank == 0);
      return isLastElementOnRank0 ? x_val : logBarrierGradVal;
    });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * @verbatim
   * term := 0.0
   * \forall n \in n_local
   *     if left[n] == 1.0 \land right[n] == 0.0
   *         term += this[n]
   * term *= mu * kappa
   * return term
   * @endverbatim
   */
  bool vectorLinearDampingTerm(hiop::hiopVector& x, hiop::hiopVector& left, hiop::hiopVector& right, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&left));
    assert(N == getLocalSize(&right));
    static const real_type mu = two;
    static const real_type kappa_d = two;

    x.setToConstant(one);
    left.setToConstant(one);
    right.setToConstant(zero);

    if(rank == 0) {
      setLocalElement(&left, N - 1, two);
      setLocalElement(&right, N - 1, two);
    }

    real_type expected = zero;
    for(local_ordinal_type i = 0; i < N - 1; ++i) {
      expected += one;
    }
    if(rank != 0) expected += one;
    expected *= mu;
    expected *= kappa_d;

    const real_type term = x.linearDampingTerm_local(left, right, mu, kappa_d);

    const int fail = !isEqual(term, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test: For addLinearDampingTerm method, which performs a "signed" axpy:
   * `x[i] = alpha*x[i] + sign*ct` where sign=1 when exactly one of left[i] and right[i]
   * is 1. and sign=0 otherwise.
   */
  bool vectorAddLinearDampingTerm(hiop::hiopVector& x, hiop::hiopVector& left, hiop::hiopVector& right, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&left));
    assert(N == getLocalSize(&right));
    static const real_type ct = two;
    static const real_type alpha = quarter;

    x.setToConstant(one);
    left.setToConstant(one);
    right.setToConstant(zero);

    // idx 0: left=0, right=1
    if(N >= 1) {
      setLocalElement(&left, 0, zero);
      setLocalElement(&right, 0, one);
    }

    // idx 1: left=0, right=0
    if(N >= 2) {
      setLocalElement(&left, 1, zero);
      setLocalElement(&right, 1, zero);
    }

    // idx 2: left=1 right=1
    if(N >= 3) {
      setLocalElement(&left, 2, one);
      setLocalElement(&right, 2, one);
    }

    // idx 3: left=1 right=0
    if(N >= 4) {
      setLocalElement(&left, 3, one);
      setLocalElement(&right, 3, zero);
    }

    real_type expected[4];

    // expected for idx 0
    expected[0] = getLocalElement(&x, 0) * alpha - ct;

    // expected for idx 1
    if(N >= 2) {
      expected[1] = getLocalElement(&x, 1) * alpha;
    }

    // expected for idx 2
    if(N >= 3) {
      expected[2] = getLocalElement(&x, 2) * alpha;
    }

    // expected for idx 3
    if(N >= 4) {
      expected[3] = getLocalElement(&x, 3) * alpha + ct;
    }

    //
    // the call
    //
    x.addLinearDampingTerm(left, right, alpha, ct);

    //
    // compare with actual values
    //
    bool fail = false;
    for(local_ordinal_type test = 0; test < std::min(N, 4) && !fail; ++test) {
      fail = !isEqual(expected[test], getLocalElement(&x, test));
    }

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * this[i] > 0
   */
  bool vectorAllPositive(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(one);
    if(!x.allPositive()) fail++;

    x.setToConstant(one);
    if(rank == 0) setLocalElement(&x, N - 1, -one);
    if(x.allPositive()) fail++;

    printMessage(fail, __func__, rank);
    return fail;
  }

  /**
   * @brief Test:
   * this[i] > 0 \lor pattern[i] != 1.0
   */
  bool vectorAllPositive_w_patternSelect(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));

    int fail = 0;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if(!x.allPositive_w_patternSelect(pattern)) fail++;

    x.setToConstant(-one);
    if(x.allPositive_w_patternSelect(pattern)) fail++;

    x.setToConstant(one);
    if(rank == 0) setLocalElement(&x, N - 1, -one);
    if(x.allPositive_w_patternSelect(pattern)) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test: min value in a vector
   */
  bool vectorMin(hiop::hiopVector& x, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;

    x.setToConstant(two);
    if(rank == 0) setLocalElement(&x, N - 1, -one);

    fail += (x.min() != -one);

    x.setToConstant(one);
    if(rank == 0) setLocalElement(&x, N - 1, two);

    fail += (x.min() != one);

    printMessage(fail, __func__, rank);
    return 0;
  }

  /**
   * @brief Test: min value in a vector
   */
  bool vectorMin_w_pattern(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));

    int fail = 0;

    x.setToConstant(one);
    if(rank == 0) setLocalElement(&x, N - 1, -one);
    pattern.setToConstant(one);
    fail += (x.min_w_pattern(pattern) != -one);

    if(rank == 0) setLocalElement(&pattern, N - 1, zero);
    fail += (x.min_w_pattern(pattern) != one);

    printMessage(fail, __func__, rank);
    return 0;
  }

  /**
   * @brief Test: Project vector into bounds
   */
  bool vectorProjectIntoBounds(hiop::hiopVector& x,
                               hiop::hiopVector& lower,
                               hiop::hiopVector& upper,
                               hiop::hiopVector& lower_pattern,
                               hiop::hiopVector& upper_pattern,
                               const int rank)
  {
    // setup constants and make assertions
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&lower));
    assert(N == getLocalSize(&upper));
    assert(N == getLocalSize(&lower_pattern));
    assert(N == getLocalSize(&upper_pattern));
    static const real_type kappa1 = half;
    static const real_type kappa2 = half;
    int fail = 0;

    // Check that lower > upper returns false
    x.setToConstant(one);
    lower.setToConstant(one);
    upper.setToConstant(-one);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);
    if(x.projectIntoBounds_local(lower, lower_pattern, upper, upper_pattern, kappa1, kappa2)) fail++;

    // check that patterns are correctly applied and
    // x[0] is left at 1
    x.setToConstant(one);
    lower.setToConstant(-one);
    upper.setToConstant(one);
    lower_pattern.setToConstant(one);
    setLocalElement(&lower_pattern, 0, zero);
    upper_pattern.setToConstant(one);
    setLocalElement(&upper_pattern, 0, zero);

    // Call should return true
    fail += !x.projectIntoBounds_local(lower, lower_pattern, upper, upper_pattern, kappa1, kappa2);

    // First element should be one
    fail += !isEqual(getLocalElement(&x, 0), one);

    // Testing when x is on a boundary:
    // Check that projection of 1 into (-1, 1)
    // returns `true' and x == half
    x.setToConstant(one);
    lower.setToConstant(-one);
    upper.setToConstant(one);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);
    x.projectIntoBounds_local(lower, lower_pattern, upper, upper_pattern, kappa1, kappa2);

    // x[i] == 1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, half);

    // testing when x is below boundaries
    // check that projection of -2 into (0, 2)
    // returns `true' and x == half
    x.setToConstant(-two);
    lower.setToConstant(zero);
    upper.setToConstant(two);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);

    // Call should return true
    fail += !x.projectIntoBounds_local(lower, lower_pattern, upper, upper_pattern, kappa1, kappa2);

    // x[i] == 1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, half);

    // testing when x is above boundaries
    // check that projection of -2 into (0, 2)
    // returns `true' and x == half
    x.setToConstant(two);
    lower.setToConstant(-two);
    upper.setToConstant(zero);
    lower_pattern.setToConstant(one);
    upper_pattern.setToConstant(one);

    // Call should return true
    fail += !x.projectIntoBounds_local(lower, lower_pattern, upper, upper_pattern, kappa1, kappa2);

    // x[i] == -1/2 \forall i \in [1, N)
    fail += verifyAnswer(&x, -half);

    printMessage(fail, __func__, rank);
    return 0;
  }

  /**
   * @brief Test
   * @verbatim
   * \forall dxi \in dx, dxi >= 0 \implies
   *     return 1.0
   *
   * \exists dxi \in dx s.t. dxi < 0 \implies
   *     return_value := 1.0
   *     auxilary := 0.0
   *     \forall n \in n_local
   *         auxilary = compute_step_to_boundary(x[n], dx[n])
   *         if auxilary < return_value
   *             return_value = auxilary
   *     return auxilary
   * @endverbatim
   */
  bool vectorFractionToTheBdry(hiop::hiopVector& x, hiop::hiopVector& dx, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&dx));
    static const real_type tau = half;
    int fail = 0;

    x.setToConstant(one);

    // Test correct default value is returned for dx >= 0
    dx.setToConstant(two);
    real_type result = x.fractionToTheBdry_local(dx, tau);

    real_type expected = one;
    fail += !isEqual(result, expected);

    // Test minumum finding for dx < 0
    dx.setToConstant(-one);
    setLocalElement(&dx, N - 1, -two);

    result = x.fractionToTheBdry_local(dx, tau);
    expected = quarter;  // -0.5*1/(-2)
    fail += !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * Same as fractionToTheBdry, except that
   * no x[i] where pattern[i]==0 will be calculated
   */
  bool vectorFractionToTheBdry_w_pattern(hiop::hiopVector& x,
                                         hiop::hiopVector& dx,
                                         hiop::hiopVector& pattern,
                                         const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&dx));
    assert(N == getLocalSize(&pattern));
    static const real_type tau = half;
    int fail = 0;

    // Fraction to boundary is const, so no need to reset x after each test
    x.setToConstant(one);

    // Pattern all ones, X all ones, result should be
    // default (alpha == one)
    pattern.setToConstant(one);
    dx.setToConstant(one);
    real_type result = x.fractionToTheBdry_w_pattern_local(dx, tau, pattern);
    real_type expected = one;  // default value if dx >= 0
    fail += !isEqual(result, expected);

    // Pattern all ones except for one value, should still be default
    // value of one
    pattern.setToConstant(one);
    dx.setToConstant(one);
    setLocalElement(&pattern, N - 1, zero);
    setLocalElement(&dx, N - 1, -half);

    result = x.fractionToTheBdry_w_pattern_local(dx, tau, pattern);
    expected = one;  // default value if dx >= 0
    fail += !isEqual(result, expected);

    // Pattern all ones, dx will be <0
    pattern.setToConstant(one);
    dx.setToConstant(-one);
    setLocalElement(&dx, N - 1, -two);

    result = x.fractionToTheBdry_w_pattern_local(dx, tau, pattern);
    expected = quarter;  // -0.5*1/(-2)
    fail += !isEqual(result, expected);

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   *  pattern != 0 \lor this == 0
   */
  bool vectorMatchesPattern(hiop::hiopVector& x, hiop::hiopVector& pattern, const int rank)
  {
    const local_ordinal_type N = getLocalSize(&x);
    assert(N == getLocalSize(&pattern));
    int fail = 0;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if(!x.matchesPattern(pattern)) fail++;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if(rank == 0) setLocalElement(&pattern, N - 1, 0);
    if(x.matchesPattern(pattern)) fail++;

    x.setToConstant(one);
    pattern.setToConstant(one);
    if(rank == 0) setLocalElement(&x, N - 1, 0);
    if(!x.matchesPattern(pattern)) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test that hiop correctly adjusts based on the
   * hessian of the duals function
   */
  bool vectorAdjustDuals_plh(hiop::hiopVector& z1,
                             hiop::hiopVector& z2,
                             hiop::hiopVector& x,
                             hiop::hiopVector& pattern,
                             const int rank)
  {
    const local_ordinal_type N = getLocalSize(&z1);
    assert(N == getLocalSize(&z2));
    assert(N == getLocalSize(&x));
    assert(N == getLocalSize(&pattern));

    // z1 will adjust duals with it's method
    z1.setToConstant(one);

    // z2's duals will be adjusted by hand
    z2.setToConstant(one);

    x.setToConstant(two);
    pattern.setToConstant(one);

    static const real_type mu = half;
    static const real_type kappa = half;
    z1.adjustDuals_plh(x, pattern, mu, kappa);

    real_type a, b;
    for(local_ordinal_type i = 0; i < N; i++) {
      a = mu / getLocalElement(&x, i);
      b = a / kappa;
      a *= kappa;
      if(getLocalElement(&x, i) < b)
        setLocalElement(&z2, i, b);
      else if(a <= b)
        setLocalElement(&z2, i, b);
      else if(a < getLocalElement(&x, i))
        setLocalElement(&z2, i, a);
    }

    // the method's adjustDuals_plh should yield
    // the same result as computing by hand
    int fail = 0;
    for(local_ordinal_type i = 0; i < N; i++) {
      fail += !isEqual(getLocalElement(&z1, i),   // expected
                       getLocalElement(&z2, i));  // actual
    }

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * \exists e \in this s.t. isnan(e)
   *
   * @note This is local method only
   */
  bool vectorIsnan(hiop::hiopVector& x, const int rank = 0)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if(x.isnan_local()) fail++;

    x.setToConstant(one / zero);
    if(x.isnan_local()) fail++;

    x.setToConstant(zero / zero);
    if(!x.isnan_local()) fail++;

    x.setToConstant(one);
    setLocalElement(&x, N - 1, zero / zero);
    if(!x.isnan_local()) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * \exists e \in this s.t. isinf(e)
   *
   * @note This is local method only
   */
  bool vectorIsinf(hiop::hiopVector& x, const int rank = 0)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if(x.isinf_local()) fail++;

    x.setToConstant(zero / zero);
    if(x.isinf_local()) fail++;

    x.setToConstant(one / zero);
    if(!x.isinf_local()) fail++;

    x.setToConstant(one);
    setLocalElement(&x, N - 1, one / zero);
    if(!x.isinf_local()) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * \forall e \in this, isfinite(e)
   *
   * @note This is local method only
   */
  bool vectorIsfinite(hiop::hiopVector& x, const int rank = 0)
  {
    const local_ordinal_type N = getLocalSize(&x);
    int fail = 0;
    x.setToConstant(zero);
    if(!x.isfinite_local()) fail++;

    x.setToConstant(zero / zero);
    if(x.isfinite_local()) fail++;

    x.setToConstant(one);
    setLocalElement(&x, N - 1, one / zero);
    if(x.isfinite_local()) fail++;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /**
   * @brief Test:
   * x[i] == y[i]
   */
  bool vector_is_equal(hiop::hiopVector& x, hiop::hiopVector& y, const int rank)
  {
    int fail = 0;
    x.setToConstant(one);
    y.setToConstant(one);
    fail += (int)(!(x.is_equal(y)));  // x.is_equal(y) should returns true (1).

    y.setToConstant(two);
    fail += (int)((x.is_equal(y)));  // x.is_equal(y) should returns false (0).

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &x);
  }

  /// Returns element _i_ of vector _x_.
  real_type getLocalElement(hiop::hiopVector* x, local_ordinal_type i) { return getLocalDataConst(x)[i]; }

  /// Returns pointer to local vector data
  local_ordinal_type getLocalSize(const hiop::hiopVector* x) { return static_cast<local_ordinal_type>(x->get_local_size()); }

  /// Checks if _local_ vector elements are set to `answer`.
  int verifyAnswer(hiop::hiopVector* x, real_type answer)
  {
    const local_ordinal_type N = getLocalSize(x);
    const real_type* xdata = getLocalDataConst(x);

    int local_fail = 0;

    for(local_ordinal_type i = 0; i < N; ++i) {
      if(!isEqual(xdata[i], answer)) {
        ++local_fail;
      }
    }

    return local_fail;
  }

  /// Checks if _local_ vector elements are set to `answer`.
  int verifyAnswer(hiop::hiopVector* x, real_type min_val, real_type max_val)
  {
    const local_ordinal_type N = getLocalSize(x);
    const real_type* xdata = getLocalDataConst(x);

    int local_fail = 0;

    for(local_ordinal_type i = 0; i < N; ++i) {
      if(xdata[i] > max_val || xdata[i] < min_val) {
        std::cout << "minv: " << min_val << ", maxv: " << max_val << ", x:[" << i << "]:" << xdata[i] << std::endl;
        ++local_fail;
      }
    }

    return local_fail;
  }

  /**
   * @brief Verifies:
   * \forall x in _local_ vector data at index i,
   *    x == expect(i)
   */
  int verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect)
  {
    const local_ordinal_type N = getLocalSize(x);
    const real_type* xdata = getLocalDataConst(x);

    int local_fail = 0;

    for(local_ordinal_type i = 0; i < N; ++i) {
      if(!isEqual(xdata[i], expect(i))) {
        ++local_fail;
      }
    }

    return local_fail;
  }

protected:
  // Interface to methods specific to vector implementation
  virtual const real_type* getLocalDataConst(hiop::hiopVector* x) = 0;
  virtual void setLocalElement(hiop::hiopVector* x, local_ordinal_type i, real_type val) = 0;
  virtual real_type* createLocalBuffer(local_ordinal_type N, real_type val) = 0;
  virtual local_ordinal_type* createIdxBuffer(local_ordinal_type N, local_ordinal_type val) = 0;
  virtual void deleteLocalBuffer(real_type* buffer) = 0;
  virtual bool reduceReturn(int failures, hiop::hiopVector* x) = 0;
};

}  // namespace tests
}  // namespace hiop
