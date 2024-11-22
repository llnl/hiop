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
 * @file hiopVectorRaja.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#ifndef HIOP_VECTOR_RAJA
#define HIOP_VECTOR_RAJA

#include <cstdio>
#include <string>
#include <cassert>
#include <cstring>
#include <random>

#include <hiopMPI.hpp>
#include "hiopVector.hpp"
#include "hiopVectorInt.hpp"

#include "ExecSpace.hpp"

namespace hiop
{

// forward declarations of the test classes that are friends to this class
namespace tests
{
class VectorTestsRajaPar;
class MatrixTestsRajaDense;
class MatrixTestsRajaSparseTriplet;
class MatrixTestsRajaSymSparseTriplet;
}  // namespace tests

template<class MEMBACKEND, class EXECPOLICYRAJA>
class hiopVectorRaja : public hiopVector
{
public:
  hiopVectorRaja(const size_type& glob_n, std::string mem_space, index_type* col_part = NULL, MPI_Comm comm = MPI_COMM_SELF);
  hiopVectorRaja() = delete;
  virtual ~hiopVectorRaja();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void set_to_random_uniform(double minv, double maxv);
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select);
  virtual void copyFrom(const hiopVector& vec);
  virtual void copyFrom(const double* local_array);  // v should be of length at least n_local
  virtual void copy_from_w_pattern(const hiopVector& src, const hiopVector& select);

  /// @brief Copy entries from a hiopVectorPar, see method documentation in the parent class.
  void copy_from_vectorpar(const hiopVectorPar& vsrc);
  /// @brief Copy entries to a hiopVectorPar, see method documentation in the parent class.
  void copy_to_vectorpar(hiopVectorPar& vdest) const;

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src.
   *
   * @pre All vectors must reside in the same memory space.
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */
  virtual void copy_from_indexes(const hiopVector& src, const hiopVectorInt& index_in_src);

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src.
   *
   * @pre All vectors must reside in the same memory space.
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */

  /**
   * @brief Copy from src the elements specified by the indices in index_in_src.
   *
   * @pre All vectors and arrays must reside in the same memory space.
   * @pre Size of src must be greater or equal than size of this
   * @pre Size of index_in_src must be equal to size of this
   * @pre Elements of index_in_src must be valid (zero-based) indexes in src
   *
   */
  virtual void copy_from_indexes(const double* src, const hiopVectorInt& index_in_src);

  /** Copy the 'n' elements of v starting at 'start_index_in_this' in 'this' */
  virtual void copyFromStarting(int start_index_in_this, const double* v, int nv);
  virtual void copyFromStarting(int start_index, const hiopVector& src);
  /// @brief Copy the 'n' elements of v starting at 'start_index_in_v' into 'this'
  virtual void copy_from_starting_at(const double* v, int start_index_in_v, int n);
  /* copy 'dest' starting at 'start_idx_dest' to 'this' starting at 'start_idx_src' */
  virtual void startingAtCopyFromStartingAt(int start_idx_src, const hiopVector& v, int start_idx_dest);

  virtual void copyTo(double* dest) const;
  virtual void copyToStarting(int start_index, hiopVector& dst) const;
  /* Copy 'this' to v starting at start_index in 'v'. */
  virtual void copyToStarting(hiopVector& vec, int start_index_in_dest) const;
  virtual void copyToStartingAt_w_pattern(hiopVector& vec, index_type start_index_in_dest, const hiopVector& ix) const;

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

  /* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest'
   * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
   * either source ('this') or destination ('dest') is reached
   * if 'selec_dest' is given, the values are copy to 'dest' where the corresponding entry in 'selec_dest' is nonzero */
  virtual void startingAtCopyToStartingAt(index_type start_idx_in_src,
                                          hiopVector& dest,
                                          index_type start_idx_dest,
                                          size_type num_elems = -1) const;
  virtual void startingAtCopyToStartingAt_w_pattern(index_type start_idx_in_src,
                                                    hiopVector& dest,
                                                    index_type start_idx_dest,
                                                    const hiopVector& selec_dest,
                                                    size_type num_elems = -1) const;

  virtual double twonorm() const;
  virtual double dotProductWith(const hiopVector& vec) const;
  virtual double infnorm() const;
  virtual double infnorm_local() const;
  virtual double onenorm() const;
  virtual double onenorm_local() const;
  virtual void componentMult(const hiopVector& v);
  virtual void componentDiv(const hiopVector& v);
  virtual void componentDiv_w_selectPattern(const hiopVector& v, const hiopVector& ix);
  virtual void component_min(const double constant);
  virtual void component_min(const hiopVector& vec);
  virtual void component_max(const double constant);
  virtual void component_max(const hiopVector& v);
  virtual void component_abs();
  virtual void component_sgn();
  virtual void component_sqrt();
  virtual void scale(double alpha);
  /** this += alpha * x */
  virtual void axpy(double alpha, const hiopVector& x);
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

  /** this += alpha * x * z */
  virtual void axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec);
  /** this += alpha * x / z */
  virtual void axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec);
  virtual void axdzpy_w_pattern(double alpha, const hiopVector& xvec, const hiopVector& zvec, const hiopVector& select);
  /** Add c to the elements of this */
  virtual void addConstant(double c);
  virtual void addConstant_w_patternSelect(double c, const hiopVector& select);
  virtual double min() const;
  virtual void min(double& minval, int& index) const;
  virtual double min_w_pattern(const hiopVector& select) const;
  virtual void negate();
  virtual void invert();
  virtual double logBarrier_local(const hiopVector& select) const;
  virtual double sum_local() const;
  virtual void addLogBarrierGrad(double alpha, const hiopVector& xvec, const hiopVector& select);

  virtual double linearDampingTerm_local(const hiopVector& ixl_select,
                                         const hiopVector& ixu_select,
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

  virtual int allPositive();
  virtual int allPositive_w_patternSelect(const hiopVector& select);
  virtual bool projectIntoBounds_local(const hiopVector& xlo,
                                       const hiopVector& ixl,
                                       const hiopVector& xup,
                                       const hiopVector& ixu,
                                       double kappa1,
                                       double kappa2);
  virtual double fractionToTheBdry_local(const hiopVector& dvec, const double& tau) const;
  virtual double fractionToTheBdry_w_pattern_local(const hiopVector& dvec, const double& tau, const hiopVector& ix) const;
  virtual void selectPattern(const hiopVector& select);
  virtual bool matchesPattern(const hiopVector& select);

  virtual hiopVector* alloc_clone() const;
  virtual hiopVector* new_copy() const;

  virtual void adjustDuals_plh(const hiopVector& xvec, const hiopVector& ixvec, const double& mu, const double& kappa);

  virtual bool is_zero() const;
  virtual bool isnan_local() const;
  virtual bool isinf_local() const;
  virtual bool isfinite_local() const;

  virtual void print(FILE*, const char* withMessage = NULL, int max_elems = -1, int rank = -1) const;
  virtual void print() const;

  /* more accessors */
  inline size_type get_local_size() const { return n_local_; }
  inline double* local_data_host() { return data_host_; }
  inline const double* local_data_host_const() const { return data_host_; }
  inline double* local_data() { return data_dev_; }
  inline const double* local_data_const() const { return data_dev_; }
  inline MPI_Comm get_mpi_comm() const { return comm_; }

private:
  void copyToDev();
  void copyFromDev();
  friend class tests::VectorTestsRajaPar;
  friend class tests::MatrixTestsRajaDense;
  friend class tests::MatrixTestsRajaSparseTriplet;
  friend class tests::MatrixTestsRajaSymSparseTriplet;

public:
  virtual size_type numOfElemsLessThan(const double& val) const;
  virtual size_type numOfElemsAbsLessThan(const double& val) const;

  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType* arr_src,
                                 const int start_src) const;
  virtual void set_array_from_to(hiopInterfaceBase::NonlinearityType* arr,
                                 const int start,
                                 const int end,
                                 const hiopInterfaceBase::NonlinearityType arr_src) const;

  virtual bool is_equal(const hiopVector& vec) const;

  const ExecSpace<MEMBACKEND, EXECPOLICYRAJA>& exec_space() const { return exec_space_; }

private:
  ExecSpace<MEMBACKEND, EXECPOLICYRAJA> exec_space_;
  using MEMBACKENDHOST = typename MEMBACKEND::MemBackendHost;

  // EXECPOLICYRAJA is used internally as a execution policy. EXECPOLICYHOST is not used internally
  // in this class. EXECPOLICYHOST can be any host policy as long as memory allocations and
  // and transfers within and from `exec_space_host_` work with EXECPOLICYHOST (currently all such
  // combinations work).
  using EXECPOLICYHOST = hiop::ExecPolicySeq;
  ExecSpace<MEMBACKENDHOST, EXECPOLICYHOST> exec_space_host_;

  std::string mem_space_;
  MPI_Comm comm_;
  double* data_host_;
  double* data_dev_;
  size_type glob_il_, glob_iu_;
  size_type n_local_;
  mutable hiopVectorInt* idx_cumsum_;
  /// Copy constructor, for internal/private use only (it doesn't copy the elements.) */
  hiopVectorRaja(const hiopVectorRaja&);
};

}  // namespace hiop

#endif  // HIOP_VECTOR_RAJA
