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
 * @file hiopVectorHip.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 */

#ifndef HIOP_VECTOR_HIP
#define HIOP_VECTOR_HIP

#include <cstdio>
#include <string>
#include <cassert>
#include <cstring>

#include "ExecSpace.hpp"

#include <hiopMPI.hpp>

#include "hiopVector.hpp"

#include <hipblas.h>

namespace hiop
{
// Forward declarations
class hiopVectorPar;
// Forward declarations of tester classes that needs to be friends with this vector
namespace tests
{
class VectorTestsHip;
}

/// Implementation of abstract class hiopVector using HIP API
class hiopVectorHip : public hiopVector
{
public:
  hiopVectorHip(const size_type& glob_n, index_type* col_part = nullptr, MPI_Comm comm = MPI_COMM_SELF);
  virtual ~hiopVectorHip();

  /// @brief Set all elements to zero.
  virtual void setToZero();
  /// @brief Set all elements to c
  virtual void setToConstant(double c);
  /// @brief Set all elements to random values uniformly distributed between `minv` and `maxv`.
  virtual void set_to_random_uniform(double minv, double maxv);
  /// @brief Set all elements that are not zero in ix to  c, and the rest to 0
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select);

  /// @brief Copy the elements of v
  virtual void copyFrom(const hiopVector& vec);
  virtual void copyFrom(const double* local_array);
  /// @brief Copy from vec the elements specified by the indices in select
  virtual void copy_from_w_pattern(const hiopVector& src, const hiopVector& select);
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_dest' in 'this'
  virtual void copyFromStarting(int start_index_in_dest, const double* v, int n);
  /// @brief Copy v_src into 'this' starting at start_index_in_dest in 'this'. */
  virtual void copyFromStarting(int start_index_in_dest, const hiopVector& v_src);
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_v' into 'this'
  virtual void copy_from_starting_at(const double* v, int start_index_in_v, int n);

  /// @brief Copy from src the elements specified by the indices in index_in_src.
  virtual void copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src);
  /// @brief Copy from src the elements specified by the indices in index_in_src.
  virtual void copy_from_indexes(const double* src, const hiopVectorInt& index_in_src);

  /// @brief Copy entries from a hiopVectorPar, see method documentation in the parent class.
  void copy_from_vectorpar(const hiopVectorPar& vsrc);
  /// @brief Copy entries to a hiopVectorPar, see method documentation in the parent class.
  void copy_to_vectorpar(hiopVectorPar& vdest) const;

  ///  @brief Copy from 'v' starting at 'start_idx_src' to 'this' starting at 'start_idx_dest'
  virtual void startingAtCopyFromStartingAt(int start_idx_dest, const hiopVector& v, int start_idx_src);

  /// @brief Copy 'this' to double array, which is assumed to be at least of 'n_local_' size.
  virtual void copyTo(double* dest) const;
  /// @brief Copy 'this' to dst starting at start_index in 'this'.
  virtual void copyToStarting(int start_index, hiopVector& dst) const;
  /// @brief Copy 'this' to dst starting at start_index in 'dst'.
  virtual void copyToStarting(hiopVector& dst, int start_index_in_dest) const;
  /// @brief Copy the entries in 'this' where corresponding 'ix' is nonzero, to v starting at start_index in 'v'.
  virtual void copyToStartingAt_w_pattern(hiopVector& vec, int start_index_in_dest, const hiopVector& ix) const;

  /// @brief Copy the entries in `c` and `d` to `this`, according to the mapping in `c_map` and `d_map`
  virtual void copy_from_two_vec_w_pattern(const hiopVector& c,
                                           const hiopVectorInt& c_map,
                                           const hiopVector& d,
                                           const hiopVectorInt& d_map);

  /// @brief Copy the entries in `this` to `c` and `d`, according to the mapping `c_map` and `d_map`
  virtual void copy_to_two_vec_w_pattern(hiopVector& c,
                                         const hiopVectorInt& c_map,
                                         hiopVector& d,
                                         const hiopVectorInt& d_map) const;

  /**
   * copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest'
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached
   */
  virtual void startingAtCopyToStartingAt(int start_idx_in_src,
                                          hiopVector& dest,
                                          int start_idx_dest,
                                          int num_elems = -1) const;
  /**
   * @brief Copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest'
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached
   * The values are copy to 'dest' where the corresponding entry in 'selec_dest' is nonzero
   */
  virtual void startingAtCopyToStartingAt_w_pattern(int start_idx_in_src,
                                                    hiopVector& dest,
                                                    int start_idx_dest,
                                                    const hiopVector& selec_dest,
                                                    int num_elems = -1) const;

  /** @brief Return the two norm */
  virtual double twonorm() const;
  /** @brief Return the infinity norm */
  virtual double infnorm() const;
  /** @brief Linf norm on single rank */
  virtual double infnorm_local() const;
  /** @brief Return the one norm */
  virtual double onenorm() const;
  /** @brief L1 norm on single rank */
  virtual double onenorm_local() const;
  /** @brief Multiply the components of this by the components of v. */
  virtual void componentMult(const hiopVector& v);
  /** @brief Divide the components of this hiopVector by the components of v. */
  virtual void componentDiv(const hiopVector& v);

  /**
   * @brief Elements of this that corespond to nonzeros in ix are divided by elements of v.
   * The rest of elements of this are set to zero.
   */
  virtual void componentDiv_w_selectPattern(const hiopVector& v, const hiopVector& ix);

  /** @brief Set each component of this hiopVector to the minimum of itself and the given constant. */
  virtual void component_min(const double constant);
  /** @brief Set each component of this hiopVector to the minimum of itself and the corresponding component of 'v'. */
  virtual void component_min(const hiopVector& vec);
  /** @brief Set each component of this hiopVector to the maximum of itself and the given constant. */
  virtual void component_max(const double constant);
  /** @brief Set each component of this hiopVector to the maximum of itself and the corresponding component of 'v'. */
  virtual void component_max(const hiopVector& v);
  /** @brief Set each component to its absolute value */
  virtual void component_abs();
  /** @brief Apply sign function to each component */
  virtual void component_sgn();
  /** @brief compute sqrt of each component */
  virtual void component_sqrt();

  /// @brief Scale each element of this  by the constant alpha
  virtual void scale(double alpha);
  /// @brief this += alpha * x
  virtual void axpy(double alpha, const hiopVector& xvec);
  /// @brief this += alpha * x, for the entries in 'this' where corresponding 'select' is nonzero.
  virtual void axpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& select);

  /**
   * @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
   *
   * @param alpha scaling factor
   * @param x vector of doubles to be axpy-ed to this (size equal to size of i and less than or equal to size of this)
   * @param i vector of indexes in this to which the axpy operation is performed (size equal to size of x and less than
   * or equal to size of this)
   *
   * @pre The entries of i must be valid (zero-based) indexes in this
   *
   */
  virtual void axpy(double alpha, const hiopVector& xvec, const hiopVectorInt& i);

  /// @brief this += alpha * x * z
  virtual void axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec);
  /// @brief this += alpha * x / z
  virtual void axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec);
  /// @brief this += alpha * x / z on entries 'i' for which select[i]==1.
  virtual void axdzpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& zvec, const hiopVector& select);
  /// @brief Add c to the elements of this
  virtual void addConstant(double c);
  virtual void addConstant_w_patternSelect(double c, const hiopVector& select);
  /// @brief Return the dot product of this hiopVector with v
  virtual double dotProductWith(const hiopVector& vec) const;
  /// @brief Negate all the elements of this
  virtual void negate();
  /// @brief Invert (1/x) the elements of this
  virtual void invert();
  /// @brief compute log barrier term, that is sum{ln(x_i):i=1,..,n}
  virtual double logBarrier_local(const hiopVector& select) const;
  /// @brief adds the gradient of the log barrier, namely this=this+alpha*1/select(x)
  virtual void addLogBarrierGrad(double alpha, const hiopVector& xvec, const hiopVector& select);
  /// @brief compute sum{(x_i):i=1,..,n}
  virtual double sum_local() const;

  /**
   * @brief Computes the log barrier's linear damping term of the Filter-IPM method of
   * WaectherBiegler (see paper, section 3.7).
   * Essentially compute  kappa_d*mu* \sum { this[i] | ixleft[i]==1 and ixright[i]==0 }
   */
  virtual double linearDampingTerm_local(const hiopVector& ixleft,
                                         const hiopVector& ixright,
                                         const double& mu,
                                         const double& kappa_d) const;

  /**
   * Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of
   * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
   *
   * Supports distributed/MPI vectors, but performs only elementwise operations and do not
   * require communication.
   *
   * This method is used to add gradient contributions from the (linear) damping term used
   * to handle unbounded problems. The damping terms are used for variables that are
   * bounded on one side only.
   */
  virtual void addLinearDampingTerm(const hiopVector& ixleft,
                                    const hiopVector& ixright,
                                    const double& alpha,
                                    const double& ct);

  /// @brief True if all elements of this are positive.
  virtual int allPositive();
  /// @brief True if elements corresponding to nonzeros in w are all positive
  virtual int allPositive_w_patternSelect(const hiopVector& select);
  /// @brief Return the minimum value in this vector
  virtual double min() const;
  virtual double min_w_pattern(const hiopVector& select) const;
  /// @brief Return the minimum value in this vector, and the index at which it occurs.
  virtual void min(double& minval, int& index) const;
  /// @brief Project the vector into the bounds, used for shifting the ini pt in the bounds
  virtual bool projectIntoBounds_local(const hiopVector& xl,
                                       const hiopVector& ixl,
                                       const hiopVector& xu,
                                       const hiopVector& ixu,
                                       double kappa1,
                                       double kappa2);
  /// @brief max{a\in(0,1]| x+ad >=(1-tau)x}
  virtual double fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const;
  virtual double fractionToTheBdry_w_pattern_local(const hiopVector& dvec,
                                                   const double& tau,
                                                   const hiopVector& select) const;
  /// @brief Entries corresponding to zeros in ix are set to zero
  virtual void selectPattern(const hiopVector& select);
  /// @brief checks whether entries in this matches pattern in ix
  virtual bool matchesPattern(const hiopVector& select);
  /// @brief dual adjustment -> see hiopIterate::adjustDuals_primalLogHessian
  virtual void adjustDuals_plh(const hiopVector& xvec, const hiopVector& ixvec, const double& mu, const double& kappa);

  /// @brief True if all elements of this are zero. TODO: add unit test
  virtual bool is_zero() const;
  /// @brief check for nans in the local vector
  virtual bool isnan_local() const;
  /// @brief check for infs in the local vector
  virtual bool isinf_local() const;
  /// @brief check if all values are finite /well-defined floats. Returns false if nan or infs are present.
  virtual bool isfinite_local() const;

  /// @brief prints up to max_elems (by default all), on rank 'rank' (by default on all)
  virtual void print(FILE* file = nullptr, const char* message = nullptr, int max_elems = -1, int rank = -1) const;
  /// @brief allocates a vector that mirrors this, but doesn't copy the values
  virtual hiopVector* alloc_clone() const;
  /// @brief allocates a vector that mirrors this, and copies the values
  virtual hiopVector* new_copy() const;

  /* more accessors */
  inline size_type get_local_size() const { return n_local_; }
  inline double* local_data() { return data_; }
  inline const double* local_data_const() const { return data_; }
  inline double* local_data_host() { return data_host_mirror_; }
  inline const double* local_data_host_const() const { return data_host_mirror_; }

private:
  virtual void copyToDev();
  virtual void copyFromDev();
  virtual void copyToDev() const;
  virtual void copyFromDev() const;
  friend class tests::VectorTestsHip;

public:
  /// @brief get number of values that are less than the given value 'val'. TODO: add unit test
  virtual size_type numOfElemsLessThan(const double& val) const;
  /// @brief get number of values whose absolute value are less than the given value 'val'. TODO: add unit test
  virtual size_type numOfElemsAbsLessThan(const double& val) const;

  /// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
  /// TODO: add unit test
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType* arr_src,
                                 const int start_src) const;
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType arr_src) const;

  /// @brief check if `this` vector is identical to `vec`
  virtual bool is_equal(const hiopVector& vec) const;

  /* functions for this class */
  inline MPI_Comm get_mpi_comm() const { return comm_; }

  ExecSpace<MemBackendHip, ExecPolicyHip>& exec_space() { return exec_space_; }
  const ExecSpace<MemBackendHip, ExecPolicyHip>& exec_space() const { return exec_space_; }

private:
  ExecSpace<MemBackendHip, ExecPolicyHip> exec_space_;
  ExecSpace<MemBackendCpp, ExecPolicySeq> exec_space_host_;

  MPI_Comm comm_;
  double* data_host_mirror_;
  double* data_;

  size_type glob_il_, glob_iu_;
  size_type n_local_;
  mutable hiopVectorInt* idx_cumsum_;

  /** needed for cuda **/
  hipblasHandle_t handle_hipblas_;

  /** copy constructor, for internal/private use only (it doesn't copy the elements.) */
  hiopVectorHip(const hiopVectorHip&);
};

}  // namespace hiop
#endif
