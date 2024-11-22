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
 * @file hiopMatrixRajaSparseTriplet.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#include "hiopMatrixRajaSparseTriplet.hpp"
#include "hiopVectorRaja.hpp"

#include "hiop_blasdefs.hpp"

#include <algorithm>  //for std::min
#include <cmath>      //for std::isfinite
#include <cstring>
#include <sstream>

#include <cassert>

namespace hiop
{

/// @brief Constructs a hiopMatrixRajaSparseTriplet with the given dimensions and memory space
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::hiopMatrixRajaSparseTriplet(int rows,
                                                                                  int cols,
                                                                                  int _nnz,
                                                                                  std::string memspace)
    : hiopMatrixSparse(rows, cols, _nnz),
      mem_space_(memspace),
      row_starts_(nullptr)
{
  if(rows == 0 || cols == 0) {
    assert(nnz_ == 0 && "number of nonzeros must be zero when any of the dimensions are 0");
    nnz_ = 0;
  }

#ifndef HIOP_USE_GPU
  mem_space_ = "HOST";
#endif

  // printf("Memory space: %s\n", mem_space_.c_str());

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devAlloc = resmgr.getAllocator(mem_space_);
  umpire::Allocator hostAlloc = resmgr.getAllocator("HOST");

  iRow_ = static_cast<int*>(devAlloc.allocate(nnz_ * sizeof(int)));
  jCol_ = static_cast<int*>(devAlloc.allocate(nnz_ * sizeof(int)));
  values_ = static_cast<double*>(devAlloc.allocate(nnz_ * sizeof(double)));

  // create host mirror if memory space is on the device
  if(mem_space_ == "DEVICE") {
    iRow_host_ = static_cast<int*>(hostAlloc.allocate(nnz_ * sizeof(int)));
    jCol_host_ = static_cast<int*>(hostAlloc.allocate(nnz_ * sizeof(int)));
    values_host_ = static_cast<double*>(hostAlloc.allocate(nnz_ * sizeof(double)));
  } else {
    iRow_host_ = iRow_;
    jCol_host_ = jCol_;
    values_host_ = values_;
  }
}

/// @brief Destructor for hiopMatrixRajaSparseTriplet
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::~hiopMatrixRajaSparseTriplet()
{
  delete row_starts_;
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devAlloc = resmgr.getAllocator(mem_space_);
  umpire::Allocator hostAlloc = resmgr.getAllocator("HOST");

  devAlloc.deallocate(iRow_);
  devAlloc.deallocate(jCol_);
  devAlloc.deallocate(values_);

  // deallocate host mirror if memory space is on device
  if(mem_space_ == "DEVICE") {
    hostAlloc.deallocate(iRow_host_);
    hostAlloc.deallocate(jCol_host_);
    hostAlloc.deallocate(values_host_);
  }
}

/**
 * @brief Sets all the values of this matrix to zero.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::setToZero()
{
  setToConstant(0.0);
}

/**
 * @brief Sets all the values of this matrix to some constant.
 *
 * @param c A real number.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::setToConstant(double c)
{
  double* dd = this->values_;
  auto nz = nnz_;
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nz), RAJA_LAMBDA(RAJA::Index_type i) { dd[i] = c; });
}

/**
 * @brief Multiplies this matrix by a vector and stores it in an output vector.
 *
 * @param beta Amount to scale the output vector by before adding to it.
 * @param y The output vector.
 * @param alpha The amount to scale this matrix by before multiplying.
 * @param x The vector by which to multiply this matrix.
 *
 * @pre _x_'s length must equal the number of columns in this matrix.
 * @pre _y_'s length must equal the number of rows in this matrix.
 * @post _y_ will contain the output of the following equation:
 *
 * The full operation performed is:
 * _y_ = _beta_ * _y_ + _alpha_ * this * _x_
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta,
                                                                    hiopVector& y,
                                                                    double alpha,
                                                                    const hiopVector& x) const
{
  assert(x.get_size() == ncols_);
  assert(y.get_size() == nrows_);

  auto& yy = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(y);
  const auto& xx = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}

/**
 * @brief Multiplies this matrix by a vector and stores it in an output vector.
 *
 * @see above timesVec function for more detail. This overload takes raw data
 * pointers rather than hiop constructs.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta,
                                                                    double* y,
                                                                    double alpha,
                                                                    const double* x) const
{
  // y = beta * y
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nrows_), RAJA_LAMBDA(RAJA::Index_type i) { y[i] *= beta; });

  // nrs and ncs are used in assert statements only
#ifndef NDEBUG
  auto nrs = nrows_;
  auto ncs = ncols_;
#endif

  auto irw = iRow_;
  auto jcl = jCol_;
  auto vls = values_;
  // atomic is needed to prevent data race from ocurring;
  // y[jCol_[i]] can be referenced by multiple threads concurrently
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(irw[i] < nrs);
        assert(jcl[i] < ncs);
        RAJA::AtomicRef<double, hiop_raja_atomic> yy(&y[irw[i]]);
        yy += alpha * x[jcl[i]] * vls[i];
      });
}

/**
 * @brief Multiplies the transpose of this matrix by a vector and stores it
 * in an output vector.
 *
 * @see above timesVec function for more detail. This function implicitly transposes
 * this matrix for the multiplication.
 *
 * The full operation performed is:
 * y = beta * y + alpha * this^T * x
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::transTimesVec(double beta,
                                                                         hiopVector& y,
                                                                         double alpha,
                                                                         const hiopVector& x) const
{
  assert(x.get_size() == nrows_);
  assert(y.get_size() == ncols_);

  auto& yy = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(y);
  const auto& xx = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  transTimesVec(beta, y_data, alpha, x_data);
}

/**
 * @brief Multiplies the transpose of this matrix by a vector and stores it
 * in an output vector.
 *
 * @see above transTimesVec function for more detail. This overload takes raw data
 * pointers rather than hiop constructs.
 *
 * The full operation performed is:
 * y = beta * y + alpha * this^T * x
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::transTimesVec(double beta,
                                                                         double* y,
                                                                         double alpha,
                                                                         const double* x) const
{
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, ncols_), RAJA_LAMBDA(RAJA::Index_type i) { y[i] *= beta; });

  // num_rows and num_columns are used in assert statements only
#ifndef NDEBUG
  int num_rows = nrows_;
  int num_cols = ncols_;
#endif

  int* iRow = iRow_;
  int* jCol = jCol_;
  double* values = values_;
  // atomic is needed to prevent data race from ocurring;
  // y[jCol_[i]] can be referenced by multiple threads concurrently
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(iRow[i] < num_rows);
        assert(jCol[i] < num_cols);
        RAJA::AtomicRef<double, hiop_raja_atomic> yy(&y[jCol[i]]);
        yy += alpha * x[iRow[i]] * values[i];
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesMat(double beta,
                                                                    hiopMatrix& W,
                                                                    double alpha,
                                                                    const hiopMatrix& X) const
{
  assert(false && "not needed");
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::transTimesMat(double beta,
                                                                         hiopMatrix& W,
                                                                         double alpha,
                                                                         const hiopMatrix& X) const
{
  assert(false && "not needed");
}

/**
 * @brief W = beta*W + alpha*this*X^T
 * Sizes: M1(this) is (m1 x nx) and M2 is (m2, nx).
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesMatTrans(double beta,
                                                                         hiopMatrix& Wmat,
                                                                         double alpha,
                                                                         const hiopMatrix& M2mat) const
{
  auto& W = dynamic_cast<hiopMatrixDense&>(Wmat);
  const auto& M2 = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(M2mat);

  const int m1 = nrows_;
  const int m2 = M2.nrows_;
  assert(ncols_ == M2.ncols_);

  assert(m1 == W.m());
  assert(m2 == W.n());

  // double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());

  // TODO: allocAndBuildRowStarts -> should create row_starts_ internally (name='prepareRowStarts' ?)
  if(this->row_starts_ == nullptr) this->row_starts_ = this->allocAndBuildRowStarts();
  assert(this->row_starts_);

  if(M2.row_starts_ == NULL) M2.row_starts_ = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_);

  // M1nnz and M2nnz are used in assert statements only
#ifndef NDEBUG
  int M1nnz = this->nnz_;
  int M2nnz = M2.nnz_;
#endif

  index_type* M1_idx_start = this->row_starts_->idx_start_;
  index_type* M2_idx_start = M2.row_starts_->idx_start_;

  int* M1jCol = this->jCol_;
  int* M2jCol = M2.jCol_;
  double* M1values = this->values_;
  double* M2values = M2.values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = 0; j < m2; j++) {
          // dest[i,j] = weigthed_dotprod(M1_row_i,M2_row_j)
          double acc = 0.;
          index_type ki = M1_idx_start[i];
          index_type kj = M2_idx_start[j];

          while(ki < M1_idx_start[i + 1] && kj < M2_idx_start[j + 1]) {
            assert(ki < M1nnz);
            assert(kj < M2nnz);

            if(M1jCol[ki] == M2jCol[kj]) {
              acc += M1values[ki] * M2values[kj];
              ki++;
              kj++;
            } else if(M1jCol[ki] < M2jCol[kj]) {
              ki++;
            } else {
              kj++;
            }
          }  // end of while(ki... && kj...)
          WM(i, j) = beta * WM(i, j) + alpha * acc;
        }  // end j
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addDiagonal(const double& alpha, const hiopVector& d_)
{
  assert(false && "not needed");
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addDiagonal(const double& value)
{
  assert(false && "not needed");
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addSubDiagonal(const double& alpha,
                                                                          index_type start,
                                                                          const hiopVector& d_)
{
  assert(false && "not needed");
}

/// @brief: set a subdiagonal block, whose diagonal values come from the input vector `vec_d`
/// @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copySubDiagonalFrom(const index_type& start_on_dest_diag,
                                                                               const size_type& num_elems,
                                                                               const hiopVector& vec_d,
                                                                               const index_type& start_on_nnz_idx,
                                                                               double scal)
{
  const auto& vd = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(vec_d);
  assert(num_elems <= vd.get_size());
  assert(start_on_dest_diag >= 0 && start_on_dest_diag + num_elems <= this->nrows_);
  const double* v = vd.local_data_const();

  // local copy for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, num_elems),
      RAJA_LAMBDA(RAJA::Index_type row_src) {
        const index_type row_dest = row_src + start_on_dest_diag;
        const index_type nnz_dest = row_src + start_on_nnz_idx;
        iRow[nnz_dest] = jCol[nnz_dest] = row_dest;
        values[nnz_dest] = scal * v[row_src];
      });
}

/// @brief: set a subdiagonal block, whose diagonal values are set to `c`
/// @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!!
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::setSubDiagonalTo(const index_type& start_on_dest_diag,
                                                                            const size_type& num_elems,
                                                                            const double& c,
                                                                            const index_type& start_on_nnz_idx)
{
  assert(start_on_dest_diag >= 0 && start_on_dest_diag + num_elems <= this->nrows_);

  // local copy for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, num_elems),
      RAJA_LAMBDA(RAJA::Index_type row_src) {
        const index_type row_dest = row_src + start_on_dest_diag;
        const index_type nnz_dest = row_src + start_on_nnz_idx;
        iRow[nnz_dest] = row_dest;
        jCol[nnz_dest] = row_dest;
        values[nnz_dest] = c;
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addMatrix(double alpha, const hiopMatrix& X)
{
  assert(false && "not needed");
}

/**
 * @brief Adds the transpose of this matrix to a block within a dense matrix.
 *
 * @todo Test this function
 * @todo Better document this function
 *
 * block of W += alpha*transpose(this)
 * Note W; contains only the upper triangular entries
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::transAddToSymDenseMatrixUpperTriangle(int row_start,
                                                                                                 int col_start,
                                                                                                 double alpha,
                                                                                                 hiopMatrixDense& W) const
{
  auto Wm = W.m();
  auto Wn = W.n();

  assert(row_start >= 0 && row_start + ncols_ <= Wm);
  assert(col_start >= 0 && col_start + nrows_ <= Wn);
  assert(Wn == Wm);

  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  int* iRow = iRow_;
  int* jCol = jCol_;
  double* values = values_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type it) {
        const int i = jCol[it] + row_start;
        const int j = iRow[it] + col_start;
#ifdef HIOP_DEEPCHECKS
        assert(i < Wm && j < Wn);
        assert(i >= 0 && j >= 0);
        assert(i <= j && "source entries need to map inside the upper triangular part of destination");
#endif
        WM(i, j) += alpha * values[it];
      });
}

/**
 * @brief Finds the maximum absolute value of the values in this matrix.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
double hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::max_abs_value()
{
  double* values = values_;
  RAJA::ReduceMax<hiop_raja_reduce, double> norm(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, nnz_), RAJA_LAMBDA(RAJA::Index_type i) { norm.max(fabs(values[i])); });
  double maxv = static_cast<double>(norm.get());
  return maxv;
}

/**
 * @brief Find the maximum absolute value in each row of `this` matrix, and return them in `ret_vec`
 *
 * @pre 'ret_vec' has exactly same number of rows as `this` matrix
 * @pre row indices must be sorted
 * @pre col indices must be sorted
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::row_max_abs_value(hiopVector& ret_vec)
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  assert(ret_vec.get_size() == nrows_);
  ret_vec.setToZero();
  if(0 == nrows_) {
    return;
  }

  auto& vec = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(ret_vec);
  double* vd = vec.local_data();

  if(row_starts_ == NULL) {
    row_starts_ = allocAndBuildRowStarts();
  }
  assert(row_starts_);

  int num_rows = this->nrows_;
  index_type* idx_start = row_starts_->idx_start_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, num_rows),
      RAJA_LAMBDA(RAJA::Index_type row_id) {
        for(index_type itnz = idx_start[row_id]; itnz < idx_start[row_id + 1]; itnz++) {
          double abs_val = fabs(values[itnz]);
          vd[row_id] = (vd[row_id] > abs_val) ? vd[row_id] : abs_val;
        }
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::scale_row(hiopVector& vec_scal, const bool inv_scale)
{
  assert(vec_scal.get_size() == nrows_);

  auto& vec = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(vec_scal);
  double* vd = vec.local_data();

  auto iRow = this->iRow_;
  auto values = this->values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type itnz) {
        double scal;
        if(inv_scale) {
          scal = 1. / vd[iRow[itnz]];
        } else {
          scal = vd[iRow[itnz]];
        }
        values[itnz] *= scal;
      });
}

/**
 * @brief Returns whether all the values of this matrix are finite or not.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::isfinite() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  double* values = values_;
  RAJA::ReduceSum<hiop_raja_reduce, int> any(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(!std::isfinite(values[i])) any += 1;
      });

  return any.get() == 0;
}

/**
 * @brief Allocates a new hiopMatrixRajaSparseTriplet with the same dimensions
 * and size as this one.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixSparse* hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::alloc_clone() const
{
  return new hiopMatrixRajaSparseTriplet(nrows_, ncols_, nnz_, mem_space_);
}

/**
 * @brief Creates a deep copy of this matrix.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixSparse* hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::new_copy() const
{
#ifdef HIOP_DEEPCHECKS
  assert(this->checkIndexesAreOrdered());
#endif
  hiopMatrixRajaSparseTriplet* copy = new hiopMatrixRajaSparseTriplet(nrows_, ncols_, nnz_, mem_space_);
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(copy->iRow_, iRow_);
  resmgr.copy(copy->jCol_, jCol_);
  resmgr.copy(copy->values_, values_);
  resmgr.copy(copy->iRow_host_, iRow_host_);
  resmgr.copy(copy->jCol_host_, jCol_host_);
  resmgr.copy(copy->values_host_, values_host_);
  return copy;
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyFrom(const hiopMatrixSparse& dm)
{
  assert(false && "this is to be implemented - method def too vague for now");
}

/// @brief copy to 3 arrays.
/// @pre these 3 arrays are not nullptr
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copy_to(int* irow, int* jcol, double* val)
{
  assert(irow && jcol && val);
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(irow, iRow_);
  resmgr.copy(jcol, jCol_);
  resmgr.copy(val, values_);
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copy_to(hiopMatrixDense& W)
{
  assert(W.m() == nrows_);
  assert(W.n() == ncols_);
  W.setToZero();

  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());

  size_type nnz = this->nnz_;
  size_type nrows = this->nrows_;
  size_type ncols = this->ncols_;
  index_type* jCol = jCol_;
  index_type* iRow = iRow_;
  double* values = values_;

  // atomic is needed to prevent data race from ocurring;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(iRow[i] < nrows);
        assert(jCol[i] < ncols);

        RAJA::AtomicRef<double, hiop_raja_atomic> yy(&WM(iRow[i], jCol[i]));
        yy += values[i];
      });
}

#ifdef HIOP_DEEPCHECKS
/// @brief Ensures the rows and column triplet entries are ordered.
template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::checkIndexesAreOrdered() const
{
  copyFromDev();
  if(nnz_ == 0) return true;
  for(int i = 1; i < nnz_; i++) {
    if(iRow_host_[i] < iRow_host_[i - 1]) return false;
    /* else */
    if(iRow_host_[i] == iRow_host_[i - 1])
      if(jCol_host_[i] < jCol_host_[i - 1]) return false;
  }
  return true;
}
#endif

/**
 * @brief This function cannot be described briefly. See below for more detail.
 *
 * @param rowAndCol_dest_start Starting row & col within _W_ to be added to
 * in the operation.
 * @param alpha Amount to scale this matrix's values by in the operation.
 * @param D The inverse of this vector's values will be multiplied by with this
 * matrix's values in the operation.
 * @param W The output matrix, a block of which's values will be added to in
 * the operation.
 *
 * @pre rowAndCol_dest_start >= 0
 * @pre rowAndCol_dest_start + this->nrows_ <= W.m()
 * @pre rowAndCol_dest_start + this->nrows_ <= W.n()
 * @pre D.get_size() == this->ncols_
 *
 * @post A this->nrows_^2 block will be written to in _W_, containing the output
 * of the operation.
 *
 * The full operation performed is:
 * diag block of _W_ += _alpha_ * this * _D_^{-1} * transpose(this)
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowAndCol_dest_start,
                                                                                                   const double& alpha,
                                                                                                   const hiopVector& D,
                                                                                                   hiopMatrixDense& W) const
{
  const int row_dest_start = rowAndCol_dest_start, col_dest_start = rowAndCol_dest_start;

  // nnz is used in assert statements only
#ifndef NDEBUG
  int nnz = this->nnz_;
#endif

  assert(row_dest_start >= 0 && row_dest_start + nrows_ <= W.m());
  assert(col_dest_start >= 0 && col_dest_start + nrows_ <= W.n());
  assert(D.get_size() == ncols_);
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  const double* DM = D.local_data_const();

  if(row_starts_ == NULL) row_starts_ = allocAndBuildRowStarts();
  assert(row_starts_);

  int nrows = this->nrows_;
  index_type* idx_start = row_starts_->idx_start_;
  int* jCol = jCol_;
  double* values = values_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nrows),
      RAJA_LAMBDA(RAJA::Index_type i) {
        // j==i
        double acc = 0.;
        for(index_type k = idx_start[i]; k < idx_start[i + 1]; k++) {
          acc += values[k] / DM[jCol[k]] * values[k];
        }
        WM(i + row_dest_start, i + col_dest_start) += alpha * acc;

        // j>i
        for(int j = i + 1; j < nrows; j++) {
          // dest[i,j] = weigthed_dotprod(this_row_i,this_row_j)
          acc = 0.;

          index_type ki = idx_start[i];
          index_type kj = idx_start[j];
          while(ki < idx_start[i + 1] && kj < idx_start[j + 1]) {
            assert(ki < nnz);
            assert(kj < nnz);
            if(jCol[ki] == jCol[kj]) {
              acc += values[ki] / DM[jCol[ki]] * values[kj];
              ki++;
              kj++;
            } else {
              if(jCol[ki] < jCol[kj])
                ki++;
              else
                kj++;
            }
          }  // end of loop over ki and kj

          WM(i + row_dest_start, j + col_dest_start) += alpha * acc;
        }  // end j
      });
}

/**
 * @brief This function cannot be described briefly. See below for more detail.
 *
 * @param row_dest_start Starting row in destination block.
 * @param col_dest_start Starting col in destination block.
 * @param alpha Amount to scale this matrix by during the operation.
 * @param D The inverse of this vector's values will be multiplied by with this
 * matrix's values in the operation.
 * @param M2mat Another sparse matrix, the transpose of which will be multiplied in
 * the following operation.
 * @param W A dense matrix, a block in which will be used to store the result of
 * the operation.
 *
 * @pre this->ncols_ == M2mat.ncols_
 * @pre D.get_size() == this->ncols_
 * @pre row_dest_start >= 0
 * @pre row_dest_start + this->nrows_ <= W.m()
 * @pre col_dest_start >= 0
 * @pre col_dest_start + M2mat.nrows_ <= W.n()
 *
 * The full operation performed is:
 * block of _W_ += _alpha_ * this * _D_^{-1} * transpose(_M2mat_)
 * Sizes: M1 is (m1 x nx);  D is vector of len nx, M2 is  (m2, nx).
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addMDinvNtransToSymDeMatUTri(int row_dest_start,
                                                                                        int col_dest_start,
                                                                                        const double& alpha,
                                                                                        const hiopVector& D,
                                                                                        const hiopMatrixSparse& M2mat,
                                                                                        hiopMatrixDense& W) const
{
  const auto& M2 = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(M2mat);

  const int m1 = nrows_;
  const int m2 = M2.nrows_;
  assert(ncols_ == M2.ncols_);
  assert(D.get_size() == ncols_);

  // does it fit in W ?
  assert(row_dest_start >= 0 && row_dest_start + m1 <= W.m());
  assert(col_dest_start >= 0 && col_dest_start + m2 <= W.n());

  // double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.m(), W.n());
  const double* DM = D.local_data_const();

  // TODO: allocAndBuildRowStarts -> should create row_starts_ internally (name='prepareRowStarts' ?)
  if(this->row_starts_ == NULL) this->row_starts_ = this->allocAndBuildRowStarts();
  assert(this->row_starts_);

  if(M2.row_starts_ == NULL) M2.row_starts_ = M2.allocAndBuildRowStarts();
  assert(M2.row_starts_);

  index_type* M1_idx_start = this->row_starts_->idx_start_;
  index_type* M2_idx_start = M2.row_starts_->idx_start_;

  // M1nnz and M2nnz are used in assert statements only
#ifndef NDEBUG
  int M1nnz = this->nnz_;
  int M2nnz = M2.nnz_;
#endif

  int* M1jCol = this->jCol_;
  int* M2jCol = M2.jCol_;
  double* M1values = this->values_;
  double* M2values = M2.values_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, m1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        for(int j = 0; j < m2; j++) {
          // dest[i,j] = weigthed_dotprod(M1_row_i,M2_row_j)
          double acc = 0.;
          index_type ki = M1_idx_start[i];
          index_type kj = M2_idx_start[j];

          while(ki < M1_idx_start[i + 1] && kj < M2_idx_start[j + 1]) {
            assert(ki < M1nnz);
            assert(kj < M2nnz);

            if(M1jCol[ki] == M2jCol[kj]) {
              acc += M1values[ki] / DM[M1jCol[ki]] * M2values[kj];
              ki++;
              kj++;
            } else {
              if(M1jCol[ki] < M2jCol[kj])
                ki++;
              else
                kj++;
            }
          }  // end of loop over ki and kj

#ifdef HIOP_DEEPCHECKS
          if(i + row_dest_start > j + col_dest_start)
            printf("[warning] lower triangular element updated in addMDinvNtransToSymDeMatUTri\n");
          assert(i + row_dest_start <= j + col_dest_start);
#endif
          WM(i + row_dest_start, j + col_dest_start) += alpha * acc;
        }  // end j
      });
}

/**
 * @brief Generates a pointer to a single RowStartsInfo struct containing
 * the number of rows and indices at which row data starts from this matrix.
 *
 * Assumes triplets are ordered.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
typename hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::RowStartsInfo*
hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::allocAndBuildRowStarts() const
{
  assert(nrows_ >= 0);

  RowStartsInfo* rsi = new RowStartsInfo(nrows_, mem_space_);
  assert(rsi);
  if(nrows_ <= 0) {
    return rsi;
  }

  this->copyFromDev();

  // build rsi on the host, then copy it to the device for simplicity
  int it_triplet = 0;
  rsi->idx_start_host_[0] = 0;

  for(int i = 1; i <= this->nrows_; i++) {
    rsi->idx_start_host_[i] = rsi->idx_start_host_[i - 1];

    while(it_triplet < this->nnz_ && this->iRow_host_[it_triplet] == i - 1) {
#ifdef HIOP_DEEPCHECKS
      if(it_triplet >= 1) {
        assert(iRow_host_[it_triplet - 1] <= iRow_host_[it_triplet] && "row indices are not sorted");
        // assert(iCol[it_triplet-1]<=iCol[it_triplet]);
        if(iRow_host_[it_triplet - 1] == iRow_host_[it_triplet])
          assert(jCol_host_[it_triplet - 1] < jCol_host_[it_triplet] && "col indices are not sorted");
      }
#endif
      rsi->idx_start_host_[i]++;
      it_triplet++;
    }
    assert(rsi->idx_start_host_[i] == it_triplet);
  }
  assert(it_triplet == this->nnz_);

  rsi->copy_to_dev();

  return rsi;
}

/**
 * @brief Copies rows from another sparse matrix into this one.
 *
 * @pre 'src' is sorted
 * @pre 'this' has exactly 'n_rows' rows
 * @pre 'src' and 'this' must have same number of columns
 * @pre number of rows in 'src' must be at least the number of rows in 'this'
 * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyRowsFrom(const hiopMatrix& src_gen,
                                                                        const index_type* rows_idxs,
                                                                        size_type n_rows)
{
  const hiopMatrixRajaSparseTriplet& src = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(src_gen);
  assert(this->m() == n_rows);
  assert(this->numberOfNonzeros() <= src.numberOfNonzeros());
  assert(this->n() == src.n());
  assert(n_rows <= src.m());

  const int* jCol_src = src.j_col();
  const double* values_src = src.M();

  if(src.row_starts_ == nullptr) {
    src.row_starts_ = src.allocAndBuildRowStarts();
  }
  assert(src.row_starts_);

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  // this function only set up sparsity in the first run. Sparsity won't change after the first run.
  if(row_starts_ == nullptr) {
    row_starts_ = new RowStartsInfo(nrows_, mem_space_);
    assert(row_starts_);

    //
    // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
    // It is replaced by RAJA::inclusive_scan after that commit
    //
    index_type* src_row_st = src.row_starts_->idx_start_;
    index_type* dst_row_st_init = row_starts_->idx_start_;

    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_rows + 1),
        RAJA_LAMBDA(RAJA::Index_type i) { dst_row_st_init[i] = 0; });

    // comput nnz in each row from source
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_rows),
        RAJA_LAMBDA(RAJA::Index_type row_dst) {
          const index_type row_src = rows_idxs[row_dst];
          dst_row_st_init[row_dst + 1] = src_row_st[row_src + 1] - src_row_st[row_src];
        });
    RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(dst_row_st_init, n_rows + 1),
                                                 RAJA::operators::plus<index_type>());
  }

  index_type* dst_row_st = row_starts_->idx_start_;
  index_type* src_row_st = src.row_starts_->idx_start_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_rows),
      RAJA_LAMBDA(RAJA::Index_type row_dst) {
        const index_type row_src = rows_idxs[row_dst];
        index_type k_dst = dst_row_st[row_dst];
        index_type k_src = src_row_st[row_src];

        // copy from src
        while(k_src < src_row_st[row_src + 1]) {
          iRow[k_dst] = row_dst;
          jCol[k_dst] = jCol_src[k_src];
          values[k_dst] = values_src[k_src];
          k_dst++;
          k_src++;
        }
      });
}

/**
 * @brief Copy 'n_rows' rows started from 'rows_src_idx_st' (array of size 'n_rows') from 'src' to the destination,
 * which starts from the 'rows_dst_idx_st'th row in 'this'
 *
 * @pre 'this' must have exactly, or more than 'n_rows' rows
 * @pre 'this' must have exactly, or more cols than 'src'
 * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyRowsBlockFrom(const hiopMatrix& src_gen,
                                                                             const index_type& rows_src_idx_st,
                                                                             const size_type& n_rows,
                                                                             const index_type& rows_dst_idx_st,
                                                                             const size_type& dest_nnz_st)
{
  if(n_rows <= 0) {
    return;
  }

  const hiopMatrixRajaSparseTriplet& src = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(src_gen);
  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(this->n() >= src.n());
  assert(n_rows + rows_src_idx_st <= src.m());
  assert(n_rows + rows_dst_idx_st <= this->m());

  const index_type* jCol_src = src.j_col();
  const double* values_src = src.M();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;
  size_type n_rows_dst = this->m();

  if(src.row_starts_ == nullptr) {
    src.row_starts_ = src.allocAndBuildRowStarts();
  }
  assert(src.row_starts_);

  index_type* src_row_st = src.row_starts_->idx_start_;

  //
  // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
  // It is replaced by RAJA::inclusive_scan after that commit
  //
  if(row_starts_ == nullptr) {
    row_starts_ = new RowStartsInfo(n_rows_dst, mem_space_);
    assert(row_starts_);
    index_type* dst_row_st_init = row_starts_->idx_start_;

    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_rows_dst + 1),
        RAJA_LAMBDA(RAJA::Index_type i) { dst_row_st_init[i] = 0; });
  }
  index_type* dst_row_st_dev = row_starts_->idx_start_;
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator hostalloc = rm.getAllocator("HOST");

  int* next_row_nnz = static_cast<size_type*>(hostalloc.allocate(sizeof(size_type)));
  index_type register_row_st = row_starts_->register_row_st_;

  rm.copy(next_row_nnz, dst_row_st_dev + 1 + rows_dst_idx_st, 1 * sizeof(size_type));

  if(next_row_nnz[0] == 0) {
    assert(rows_dst_idx_st >= register_row_st);
    // comput nnz in each row from source
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, n_rows),
        RAJA_LAMBDA(RAJA::Index_type row_add) {
          const index_type row_src = rows_src_idx_st + row_add;
          const index_type row_dst = rows_dst_idx_st + row_add;
          dst_row_st_dev[row_dst + 1] = src_row_st[row_src + 1] - src_row_st[row_src];
        });
    RAJA::inclusive_scan_inplace<hiop_raja_exec>(
        RAJA::make_span(dst_row_st_dev + register_row_st, n_rows + 1 + rows_dst_idx_st - register_row_st),
        RAJA::operators::plus<index_type>());
    row_starts_->register_row_st_ = n_rows + rows_dst_idx_st;
  }

  index_type* dst_row_st = row_starts_->idx_start_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n_rows),
      RAJA_LAMBDA(RAJA::Index_type row_add) {
        const index_type row_src = rows_src_idx_st + row_add;
        const index_type row_dst = rows_dst_idx_st + row_add;
        index_type k_src = src_row_st[row_src];
        index_type k_dst = dst_row_st[row_dst] - dst_row_st[rows_dst_idx_st] + dest_nnz_st;

        // copy from src
        while(k_src < src_row_st[row_src + 1]) {
          iRow[k_dst] = row_dst;
          jCol[k_dst] = jCol_src[k_src];
          values[k_dst] = values_src[k_src];
          k_dst++;
          k_src++;
        }
      });
  //  delete [] next_row_nnz;
}

/// @brief Prints the contents of this function to a file.
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::print(FILE* file,
                                                                 const char* msg /*=NULL*/,
                                                                 int maxRows /*=-1*/,
                                                                 int maxCols /*=-1*/,
                                                                 int rank /*=-1*/) const
{
  int myrank_ = 0, numranks = 1;  // this is a local object => always print
  copyFromDev();

  if(file == NULL) file = stdout;

  int max_elems = maxRows >= 0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);

  if(myrank_ == rank || rank == -1) {
    if(NULL == msg) {
      std::stringstream ss;
      ss << "matrix of size " << m() << " " << n() << " and nonzeros " << numberOfNonzeros() << ", printing " << max_elems
         << " elems";
      if(numranks > 1) {
        ss << " (on rank=" << myrank_ << ")";
      }
      ss << "\n";
      fprintf(file, "%s", ss.str().c_str());
    } else {
      fprintf(file, "%s ", msg);
    }

    // using matlab indices
    fprintf(file, "iRow_host_=[");
    for(int it = 0; it < max_elems; it++) fprintf(file, "%d; ", iRow_host_[it] + 1);
    fprintf(file, "];\n");

    fprintf(file, "jCol_host_=[");
    for(int it = 0; it < max_elems; it++) fprintf(file, "%d; ", jCol_host_[it] + 1);
    fprintf(file, "];\n");

    fprintf(file, "v=[");
    for(int it = 0; it < max_elems; it++) fprintf(file, "%22.16e; ", values_host_[it]);
    fprintf(file, "];\n");
  }
}

/// @brief Copies the data stored in the host mirror to the device.
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyToDev()
{
  if(mem_space_ == "DEVICE") {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(iRow_, iRow_host_);
    resmgr.copy(jCol_, jCol_host_);
    resmgr.copy(values_, values_host_);
  }
}

/// @brief Copies the data stored on the device to the host mirror.
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyFromDev() const
{
  if(mem_space_ == "DEVICE") {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(iRow_host_, iRow_);
    resmgr.copy(jCol_host_, jCol_);
    resmgr.copy(values_host_, values_);
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::RowStartsInfo::RowStartsInfo(size_type n_rows, std::string memspace)
    : register_row_st_{0},
      num_rows_(n_rows),
      mem_space_(memspace)
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = resmgr.getAllocator(mem_space_);
  idx_start_ = static_cast<index_type*>(alloc.allocate((num_rows_ + 1) * sizeof(index_type)));
  if(mem_space_ == "DEVICE") {
    umpire::Allocator hostalloc = resmgr.getAllocator("HOST");
    idx_start_host_ = static_cast<index_type*>(hostalloc.allocate((num_rows_ + 1) * sizeof(index_type)));
  } else {
    idx_start_host_ = idx_start_;
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::RowStartsInfo::~RowStartsInfo()
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc = resmgr.getAllocator(mem_space_);
  devalloc.deallocate(idx_start_);
  if(mem_space_ == "DEVICE") {
    umpire::Allocator hostalloc = resmgr.getAllocator("HOST");
    hostalloc.deallocate(idx_start_host_);
  }
  idx_start_host_ = nullptr;
  idx_start_ = nullptr;
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::RowStartsInfo::copy_from_dev()
{
  if(idx_start_ != idx_start_host_) {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(idx_start_host_, idx_start_);
  }
}

template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::RowStartsInfo::copy_to_dev()
{
  if(idx_start_ != idx_start_host_) {
    auto& resmgr = umpire::ResourceManager::getInstance();
    resmgr.copy(idx_start_, idx_start_host_);
  }
}

/*
 *  extend original Jac to [Jac -I I]
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::set_Jac_FR(const hiopMatrixSparse& Jac_c,
                                                                      const hiopMatrixSparse& Jac_d,
                                                                      int* iJacS,
                                                                      int* jJacS,
                                                                      double* MJacS)
{
  const auto& J_c = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(Jac_c);
  const auto& J_d = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(Jac_d);

  // shortcut to the original Jac
  const int* jcol_c = J_c.jCol_;
  const int* jcol_d = J_d.jCol_;

  // assuming original Jac is sorted!
  int nnz_Jac_c = J_c.numberOfNonzeros();
  int nnz_Jac_d = J_d.numberOfNonzeros();
  int m_c = J_c.nrows_;
  int m_d = J_d.nrows_;
  int n_c = J_c.ncols_;
  int n_d = J_d.ncols_;
  assert(n_c == n_d);
  assert(ncols_ == n_c + 2 * m_c + 2 * m_d);

  int nnz_Jac_c_new = nnz_Jac_c + 2 * m_c;

  assert(nnz_ == nnz_Jac_c_new + nnz_Jac_d + 2 * m_d);

  if(J_c.row_starts_ == nullptr) {
    J_c.row_starts_ = J_c.allocAndBuildRowStarts();
  }
  assert(J_c.row_starts_);
  index_type* Jc_row_st = J_c.row_starts_->idx_start_;

  if(J_d.row_starts_ == nullptr) {
    J_d.row_starts_ = J_d.allocAndBuildRowStarts();
  }
  assert(J_d.row_starts_);
  index_type* Jd_row_st = J_d.row_starts_->idx_start_;

  // extend Jac to the p and n parts --- sparsity
  if(iJacS != nullptr && jJacS != nullptr) {
    // local copy for RAJA access
    int* iRow = iRow_;
    int* jCol = jCol_;

    // Jac for c(x) - p + n
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, m_c),
        RAJA_LAMBDA(RAJA::Index_type i) {
          index_type k_base = Jc_row_st[i];
          index_type k = k_base + 2 * i;  // append 2 nnz in each row

          // copy from base Jac_c
          while(k_base < Jc_row_st[i + 1]) {
            iRow[k] = iJacS[k] = i;
            jCol[k] = jJacS[k] = jcol_c[k_base];
            k++;
            k_base++;
          }

          // extra parts for p and n
          iRow[k] = iJacS[k] = i;
          jCol[k] = jJacS[k] = n_c + i;
          k++;

          iRow[k] = iJacS[k] = i;
          jCol[k] = jJacS[k] = n_c + m_c + i;
          k++;
        });

    // Jac for d(x) - p + n
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, m_d),
        RAJA_LAMBDA(RAJA::Index_type i) {
          index_type k_base = Jd_row_st[i];
          index_type k = nnz_Jac_c_new + k_base + 2 * i;  // append 2 nnz in each row

          // copy from base Jac_c
          while(k_base < Jd_row_st[i + 1]) {
            iRow[k] = iJacS[k] = m_c + i;
            jCol[k] = jJacS[k] = jcol_d[k_base];
            k++;
            k_base++;
          }

          // extra parts for p and n
          iRow[k] = iJacS[k] = m_c + i;
          jCol[k] = jJacS[k] = n_d + 2 * m_c + i;
          k++;

          iRow[k] = iJacS[k] = m_c + i;
          jCol[k] = jJacS[k] = n_d + 2 * m_c + m_d + i;
          k++;
        });
  }

  // extend Jac to the p and n parts --- element
  if(MJacS != nullptr) {
    // local copy for RAJA access
    double* values = values_;

    const double* J_c_val = J_c.values_;
    const double* J_d_val = J_d.values_;

    // Jac for c(x) - p + n
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, m_c),
        RAJA_LAMBDA(RAJA::Index_type i) {
          index_type k_base = Jc_row_st[i];
          index_type k = k_base + 2 * i;  // append 2 nnz in each row

          // copy from base Jac_c
          while(k_base < Jc_row_st[i + 1]) {
            values[k] = MJacS[k] = J_c_val[k_base];
            k++;
            k_base++;
          }

          // extra parts for p and n
          values[k] = MJacS[k] = -1.0;
          k++;

          values[k] = MJacS[k] = 1.0;
          k++;
        });

    // Jac for d(x) - p + n
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, m_d),
        RAJA_LAMBDA(RAJA::Index_type i) {
          index_type k_base = Jd_row_st[i];
          index_type k = nnz_Jac_c_new + k_base + 2 * i;  // append 2 nnz in each row

          // copy from base Jac_c
          while(k_base < Jd_row_st[i + 1]) {
            values[k] = MJacS[k] = J_d_val[k_base];
            k++;
            k_base++;
          }

          // extra parts for p and n
          values[k] = MJacS[k] = -1.0;
          k++;

          values[k] = MJacS[k] = 1.0;
          k++;
        });
  }
  copyFromDev();
}

/// @brief copy a submatrix from another matrix.
/// @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copySubmatrixFrom(const hiopMatrix& src_gen,
                                                                             const index_type& dest_row_st,
                                                                             const index_type& dest_col_st,
                                                                             const size_type& dest_nnz_st,
                                                                             const bool offdiag_only)
{
  const hiopMatrixRajaSparseTriplet& src = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(src_gen);
  auto m_rows = src.m();
  auto n_cols = src.n();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n());
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st + src.numberOfNonzeros() <= this->numberOfNonzeros());

  const index_type* src_iRow = src.i_row();
  const index_type* src_jCol = src.j_col();
  const double* src_val = src.M();
  size_type src_nnz = src.numberOfNonzeros();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, src_nnz),
      RAJA_LAMBDA(RAJA::Index_type src_k) {
        if(!offdiag_only || src_iRow[src_k] != src_jCol[src_k]) {
          index_type dest_k = dest_nnz_st + src_k;
          iRow[dest_k] = dest_row_st + src_iRow[src_k];
          jCol[dest_k] = dest_col_st + src_jCol[src_k];
          values[dest_k] = src_val[src_k];
        }
      });
}

/// @brief copy a submatrix from a transpose of another matrix.
/// @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copySubmatrixFromTrans(const hiopMatrix& src_gen,
                                                                                  const index_type& dest_row_st,
                                                                                  const index_type& dest_col_st,
                                                                                  const size_type& dest_nnz_st,
                                                                                  const bool offdiag_only)
{
  const hiopMatrixRajaSparseTriplet& src = dynamic_cast<const hiopMatrixRajaSparseTriplet&>(src_gen);
  auto m_rows = src.m();
  auto n_cols = src.n();

  assert(this->numberOfNonzeros() >= src.numberOfNonzeros());
  assert(n_cols + dest_col_st <= this->n());
  assert(m_rows + dest_row_st <= this->m());
  assert(dest_nnz_st + src.numberOfNonzeros() <= this->numberOfNonzeros());

  const index_type* src_iRow = src.j_col();
  const index_type* src_jCol = src.i_row();
  const double* src_val = src.M();
  size_type src_nnz = src.numberOfNonzeros();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, src_nnz),
      RAJA_LAMBDA(RAJA::Index_type src_k) {
        if(!offdiag_only || src_iRow[src_k] != src_jCol[src_k]) {
          index_type dest_k = dest_nnz_st + src_k;
          iRow[dest_k] = dest_row_st + src_iRow[src_k];
          jCol[dest_k] = dest_col_st + src_jCol[src_k];
          values[dest_k] = src_val[src_k];
        }
      });
}

/**
 * @brief Copy selected cols of a diagonal matrix (a constant 'scalar' times identity),
 * into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
 * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
 *
 * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
 * @pre this function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::setSubmatrixToConstantDiag_w_colpattern(
    const double& scalar,
    const index_type& dest_row_st,
    const index_type& dest_col_st,
    const size_type& dest_nnz_st,
    const size_type& nnz_to_copy,
    const hiopVector& ix)
{
  assert(ix.get_local_size() + dest_col_st <= this->n());
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  const auto& selected = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(ix);

  size_type n = ix.get_local_size();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;
  const double* pattern = selected.local_data_const();

#ifdef HIOP_DEEPCHECKS
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(pattern[i] != 0.0) {
          sum += 1;
        }
      });
  size_type nrm = sum.get();
  assert(nrm == nnz_to_copy);
#endif

  //
  // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
  // It is replaced by RAJA::inclusive_scan after that commit
  //
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc = resmgr.getAllocator(mem_space_);
  index_type* row_start_dev = static_cast<index_type*>(devalloc.allocate((n + 1) * sizeof(index_type)));

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(i == 0) {
          row_start_dev[i] = 0;
        } else {
          // from i=1..n
          if(pattern[i - 1] != 0.0) {
            row_start_dev[i] = 1;
          } else {
            row_start_dev[i] = 0;
          }
        }
      });
  RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(row_start_dev, n + 1), RAJA::operators::plus<index_type>());

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(1, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(row_start_dev[i] != row_start_dev[i - 1]) {
          index_type ele_add = row_start_dev[i] - 1;
          assert(ele_add >= 0 && ele_add < nnz_to_copy);
          index_type itnz_dest = dest_nnz_st + ele_add;
          iRow[itnz_dest] = dest_row_st + i - 1;
          jCol[itnz_dest] = dest_col_st + ele_add;
          values[itnz_dest] = scalar;
        }
      });

  devalloc.deallocate(row_start_dev);
}

/**
 * @brief Copy selected rows of a diagonal matrix (a constant 'scalar' times identity),
 * into 'this' as a submatrix from corner 'dest_row_st' and 'dest_col_st'
 * The non-zero elements start from 'dest_nnz_st' will be replaced by the new elements.
 *
 * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
 * @pre this function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::setSubmatrixToConstantDiag_w_rowpattern(
    const double& scalar,
    const index_type& dest_row_st,
    const index_type& dest_col_st,
    const size_type& dest_nnz_st,
    const size_type& nnz_to_copy,
    const hiopVector& ix)
{
  assert(ix.get_local_size() + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());
  assert(dest_nnz_st + nnz_to_copy <= this->numberOfNonzeros());

  const auto& selected = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(ix);

  size_type n = ix.get_local_size();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;
  const double* pattern = selected.local_data_const();

#ifdef HIOP_DEEPCHECKS
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(pattern[i] != 0.0) {
          sum += 1;
        }
      });
  size_type nrm = sum.get();
  assert(nrm == nnz_to_copy);
#endif

  //
  // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
  // It is replaced by RAJA::inclusive_scan after that commit
  //
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc = resmgr.getAllocator(mem_space_);
  index_type* row_start_dev = static_cast<index_type*>(devalloc.allocate((n + 1) * sizeof(index_type)));

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(i == 0) {
          row_start_dev[i] = 0;
        } else {
          // from i=1..n
          if(pattern[i - 1] != 0.0) {
            row_start_dev[i] = 1;
          } else {
            row_start_dev[i] = 0;
          }
        }
      });
  RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(row_start_dev, n + 1), RAJA::operators::plus<index_type>());

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(1, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(row_start_dev[i] != row_start_dev[i - 1]) {
          index_type ele_add = row_start_dev[i] - 1;
          assert(ele_add >= 0 && ele_add < nnz_to_copy);
          index_type itnz_dest = dest_nnz_st + ele_add;
          iRow[itnz_dest] = dest_row_st + ele_add;
          jCol[itnz_dest] = dest_col_st + i - 1;
          values[itnz_dest] = scalar;
        }
      });

  devalloc.deallocate(row_start_dev);
}

/**
 * @brief Copy a diagonal matrix to destination.
 * This diagonal matrix is 'src_val'*identity matrix with size 'nnz_to_copy'x'nnz_to_copy'.
 * The destination is updated from the start row 'row_dest_st' and start column 'col_dest_st'.
 * At the destination, 'nnz_to_copy` nonzeros starting from index `dest_nnz_st` will be replaced.
 * @pre The diagonal entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
 * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyDiagMatrixToSubblock(const double& src_val,
                                                                                    const index_type& dest_row_st,
                                                                                    const index_type& col_dest_st,
                                                                                    const size_type& dest_nnz_st,
                                                                                    const size_type& nnz_to_copy)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy + dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + col_dest_st <= this->n());

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_to_copy),
      RAJA_LAMBDA(RAJA::Index_type ele_add) {
        index_type itnz_dest = dest_nnz_st + ele_add;
        iRow[itnz_dest] = dest_row_st + ele_add;
        jCol[itnz_dest] = col_dest_st + ele_add;
        values[itnz_dest] = src_val;
      });
}

/**
 * @brief same as @copyDiagMatrixToSubblock, but copies only diagonal entries specified by `pattern`.
 * At the destination, 'nnz_to_copy` nonzeros starting from index `dest_nnz_st` will be replaced.
 * @pre The added entries in the destination need to be contiguous in the sparse triplet arrays of the destinations.
 * @pre This function does NOT preserve the sorted row/col indices. USE WITH CAUTION!
 * @pre 'pattern' has same size as `dx`
 * @pre 'pattern` has exactly `nnz_to_copy` nonzeros
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::copyDiagMatrixToSubblock_w_pattern(const hiopVector& dx,
                                                                                              const index_type& dest_row_st,
                                                                                              const index_type& dest_col_st,
                                                                                              const size_type& dest_nnz_st,
                                                                                              const size_type& nnz_to_copy,
                                                                                              const hiopVector& pattern)
{
  assert(this->numberOfNonzeros() >= nnz_to_copy + dest_nnz_st);
  assert(this->n() >= nnz_to_copy);
  assert(nnz_to_copy + dest_row_st <= this->m());
  assert(nnz_to_copy + dest_col_st <= this->n());
  assert(pattern.get_local_size() == dx.get_local_size());

  const auto& selected = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(pattern);
  const auto& xx = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(dx);
  const double* x = xx.local_data_const();

  size_type n = pattern.get_local_size();

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;
  double* values = values_;
  const double* pattern_dev = selected.local_data_const();

#ifdef HIOP_DEEPCHECKS
  RAJA::ReduceSum<hiop_raja_reduce, size_type> sum(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(pattern_dev[i] != 0.0) {
          sum += 1;
        }
      });
  size_type nrm = sum.get();
  assert(nrm == nnz_to_copy);
#endif

  //
  // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
  // It is replaced by RAJA::inclusive_scan after that commit
  //
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator devalloc = resmgr.getAllocator(mem_space_);
  index_type* row_start_dev = static_cast<index_type*>(devalloc.allocate((n + 1) * sizeof(index_type)));

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(i == 0) {
          row_start_dev[i] = 0;
        } else {
          // from i=1..n
          if(pattern_dev[i - 1] != 0.0) {
            row_start_dev[i] = 1;
          } else {
            row_start_dev[i] = 0;
          }
        }
      });
  RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(row_start_dev, n + 1), RAJA::operators::plus<index_type>());

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(1, n + 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(row_start_dev[i] != row_start_dev[i - 1]) {
          index_type ele_add = row_start_dev[i] - 1;
          assert(ele_add >= 0 && ele_add < nnz_to_copy);
          index_type itnz_dest = dest_nnz_st + ele_add;
          iRow[itnz_dest] = dest_row_st + ele_add;
          jCol[itnz_dest] = dest_col_st + ele_add;
          values[itnz_dest] = x[i - 1];
        }
      });

  devalloc.deallocate(row_start_dev);
}

template<class MEMBACKEND, class RAJAEXECPOL>
bool hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::is_diagonal() const
{
  bool bret{false};

  if(ncols_ != nrows_) {
    bret = false;
    return bret;
  }

  // local copy of member variable/function, for RAJA access
  index_type* iRow = iRow_;
  index_type* jCol = jCol_;

  RAJA::ReduceSum<hiop_raja_reduce, int> sum_no_diag(0);
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, nnz_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        if(iRow[i] != jCol[i]) {
          sum_no_diag += 1;
        }
      });
  bret = ((sum_no_diag.get()) == 0);

  return bret;
}

/**********************************************************************************
 * Sparse symmetric matrix in triplet format. Only the UPPER triangle is stored
 **********************************************************************************/
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta,
                                                                       hiopVector& y,
                                                                       double alpha,
                                                                       const hiopVector& x) const
{
  assert(this->ncols_ == this->nrows_);
  assert(x.get_size() == this->ncols_);
  assert(y.get_size() == this->nrows_);

  auto& yy = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(y);
  const auto& xx = dynamic_cast<const hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(x);

  double* y_data = yy.local_data();
  const double* x_data = xx.local_data_const();

  timesVec(beta, y_data, alpha, x_data);
}

/** y = beta * y + alpha * this * x */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::timesVec(double beta,
                                                                       double* y,
                                                                       double alpha,
                                                                       const double* x) const
{
  assert(this->ncols_ == this->nrows_);

  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, this->nrows_), RAJA_LAMBDA(RAJA::Index_type i) { y[i] *= beta; });

  // addition to y[iRow[i]] must be atomic
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;

  // nrows and ncols are used in assert statements only
#ifndef NDEBUG
  auto nrows = this->nrows_;
  auto ncols = this->ncols_;
#endif

  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, this->nnz_),
      RAJA_LAMBDA(RAJA::Index_type i) {
        assert(iRow[i] < nrows);
        assert(jCol[i] < ncols);
        RAJA::AtomicRef<double, hiop_raja_atomic> yy1(&y[iRow[i]]);
        yy1 += alpha * x[jCol[i]] * values[i];
        if(iRow[i] != jCol[i]) {
          RAJA::AtomicRef<double, hiop_raja_atomic> yy2(&y[jCol[i]]);
          yy2 += alpha * x[iRow[i]] * values[i];
        }
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixSparse* hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::alloc_clone() const
{
  assert(this->nrows_ == this->ncols_);
  return new hiopMatrixRajaSymSparseTriplet(this->nrows_, this->nnz_, this->mem_space_);
}

template<class MEMBACKEND, class RAJAEXECPOL>
hiopMatrixSparse* hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::new_copy() const
{
  assert(this->nrows_ == this->ncols_);
  auto* copy = new hiopMatrixRajaSymSparseTriplet(this->nrows_, this->nnz_, this->mem_space_);
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(copy->iRow_, this->iRow_);
  resmgr.copy(copy->jCol_, this->jCol_);
  resmgr.copy(copy->values_, this->values_);
  resmgr.copy(copy->iRow_host_, this->iRow_host_);
  resmgr.copy(copy->jCol_host_, this->jCol_host_);
  resmgr.copy(copy->values_host_, this->values_host_);
  return copy;
}

/**
 * @brief block of W += alpha*this
 * @note W contains only the upper triangular entries
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::addUpperTriangleToSymDenseMatrixUpperTriangle(
    int diag_start,
    double alpha,
    hiopMatrixDense& W) const
{
  assert(diag_start >= 0 && diag_start + this->nrows_ <= W.m());
  assert(diag_start + this->ncols_ <= W.n());
  assert(W.n() == W.m());

  // double** WM = W.get_M();
  RAJA::View<double, RAJA::Layout<2>> WM(W.local_data(), W.get_local_size_m(), W.get_local_size_n());
  auto Wm = W.m();
  auto Wn = W.n();
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, this->nnz_),
      RAJA_LAMBDA(RAJA::Index_type it) {
        assert(iRow[it] <= jCol[it] && "sparse symmetric matrices should contain only upper triangular entries");
        const int i = iRow[it] + diag_start;
        const int j = jCol[it] + diag_start;
        assert(i < Wm && j < Wn);
        assert(i >= 0 && j >= 0);
        assert(i <= j &&
               "symMatrices not aligned; source entries need to map inside the upper triangular part of destination");
        WM(i, j) += alpha * values[it];
      });
}

/**
 * @brief block of W += alpha*(this)^T
 * @note W contains only the upper triangular entries
 *
 * @warning This method should not be called directly.
 * Use addUpperTriangleToSymDenseMatrixUpperTriangle instead.
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::transAddToSymDenseMatrixUpperTriangle(int row_start,
                                                                                                    int col_start,
                                                                                                    double alpha,
                                                                                                    hiopMatrixDense& W) const
{
  assert(0 && "This method should not be called for symmetric matrices.");
}

/* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
 * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
 * are available in 'vec_dest' starting at 'vec_start'
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::startingAtAddSubDiagonalToStartingAt(
    int diag_src_start,
    const double& alpha,
    hiopVector& vec_dest,
    int vec_start,
    int num_elems /*=-1*/) const
{
  auto& vd = dynamic_cast<hiopVectorRaja<MEMBACKEND, RAJAEXECPOL>&>(vec_dest);
  if(num_elems < 0) num_elems = vd.get_size();
  assert(num_elems <= vd.get_size());

  assert(diag_src_start >= 0 && diag_src_start + num_elems <= this->nrows_);
  double* v = vd.local_data();

  auto vds = vd.get_size();
  auto iRow = this->iRow_;
  auto jCol = this->jCol_;
  auto values = this->values_;
  RAJA::forall<hiop_raja_exec>(
      RAJA::RangeSegment(0, this->nnz_),
      RAJA_LAMBDA(RAJA::Index_type itnz) {
        const int row = iRow[itnz];
        if(row == jCol[itnz]) {
          if(row >= diag_src_start && row < diag_src_start + num_elems) {
            assert(row + vec_start < vds);
            v[vec_start + row] += alpha * values[itnz];
          }
        }
      });
}

template<class MEMBACKEND, class RAJAEXECPOL>
size_type hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::numberOfOffDiagNonzeros() const
{
  if(-1 == nnz_offdiag_) {
    this->nnz_offdiag_ = this->nnz_;
    int* irow = this->iRow_;
    int* jcol = this->jCol_;
    RAJA::ReduceSum<hiop_raja_reduce, int> sum(0);
    RAJA::forall<hiop_raja_exec>(
        RAJA::RangeSegment(0, this->nnz_),
        RAJA_LAMBDA(RAJA::Index_type i) {
          if(irow[i] == jcol[i]) {
            sum += 1;
          }
        });
    this->nnz_offdiag_ -= static_cast<int>(sum.get());
  }

  return this->nnz_offdiag_;
}

/*
 *  extend original Hess to [Hess+diag_term]
 */
template<class MEMBACKEND, class RAJAEXECPOL>
void hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>::set_Hess_FR(const hiopMatrixSparse& Hess,
                                                                          int* iHSS,
                                                                          int* jHSS,
                                                                          double* MHSS,
                                                                          const hiopVector& add_diag)
{
  if(this->nnz_ == 0) {
    return;
  }

  hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>& M1 = *this;
  const auto& M2 = dynamic_cast<const hiopMatrixRajaSymSparseTriplet<MEMBACKEND, RAJAEXECPOL>&>(Hess);

  // assuming original Hess is sorted, and in upper-triangle format
  const int m1 = M1.m();
  const int n1 = M1.n();
  const int m2 = M2.m();
  const int n2 = M2.n();
  int m_row = add_diag.get_size();

  assert(n1 == m1);
  assert(n2 == m2);
  assert(m2 <= m1);

  // note that nnz2 can be zero, i.e., original hess is empty.
  // Hence we use add_diag.get_size() to detect the length of x in the base problem
  assert(m_row == m2 || m2 == 0);

  int nnz1 = m_row + M2.numberOfOffDiagNonzeros();

  assert(this->nnz_ == nnz1);

  if(M2.row_starts_ == NULL) {
    M2.row_starts_ = M2.allocAndBuildRowStarts();
  }
  assert(M2.row_starts_);

  index_type* M1_row_start{nullptr};
  index_type* M2_row_start = M2.row_starts_->idx_start_;
  const int* M2iRow = M2.i_row();
  const int* M2jCol = M2.j_col();

  // extend Hess to the p and n parts --- sparsity
  // sparsity may change due to te new obj term zeta*DR^2.*(x-x_ref)
  if(iHSS != nullptr && jHSS != nullptr) {
    int* M1iRow = M1.i_row();
    int* M1jCol = M1.j_col();

    if(m2 > 0) {
      if(M1.row_starts_ == nullptr) {
        M1.row_starts_ = nullptr;
        M1.row_starts_ = hiopMatrixRajaSparseTriplet<MEMBACKEND, RAJAEXECPOL>::allocRowStarts(m1, this->mem_space_);
        M1_row_start = M1.row_starts_->idx_start_;

        //
        // The latest CPU code can be found in 342eb99ec16d45f57a492be1bf1e39cce73995a5
        // It is replaced by RAJA::inclusive_scan after that commit
        //
        RAJA::forall<hiop_raja_exec>(
            RAJA::RangeSegment(0, m1 + 1),
            RAJA_LAMBDA(RAJA::Index_type i) {
              if(i > 0) {
                M1_row_start[i] = 1;
              } else {
                M1_row_start[i] = 0;
              }
            });

        RAJA::forall<hiop_raja_exec>(
            RAJA::RangeSegment(0, m2),
            RAJA_LAMBDA(RAJA::Index_type i) {
              index_type k_base = M2_row_start[i];
              index_type nnz_in_row = M2_row_start[i + 1] - k_base;

              if(nnz_in_row > 0 && M2iRow[k_base] == M2jCol[k_base]) {
                // first nonzero in this row is a diagonal term
                // skip it since we will defined the diagonal nonezero
                M1_row_start[i + 1] += nnz_in_row - 1;
              } else {
                M1_row_start[i + 1] += nnz_in_row;
              }
            });

        RAJA::inclusive_scan_inplace<hiop_raja_exec>(RAJA::make_span(M1_row_start, m1 + 1),
                                                     RAJA::operators::plus<index_type>());
      }

      RAJA::forall<hiop_raja_exec>(
          RAJA::RangeSegment(0, m2),
          RAJA_LAMBDA(RAJA::Index_type i) {
            index_type k = M1_row_start[i];
            index_type k_base = M2_row_start[i];
            size_type nnz_in_row = M2_row_start[i + 1] - k_base;

            // insert diagonal entry due to the new obj term
            M1iRow[k] = iHSS[k] = i;
            M1jCol[k] = jHSS[k] = i;
            k++;

            if(nnz_in_row > 0 && M2iRow[k_base] == M2jCol[k_base]) {
              // first nonzero in this row is a diagonal term
              // skip it since we have defined the diagonal nonezero
              k_base++;
            }

            // copy from base Hess
            while(k_base < M2_row_start[i + 1]) {
              M1iRow[k] = iHSS[k] = i;
              M1jCol[k] = jHSS[k] = M2jCol[k_base];
              k++;
              k_base++;
            }
          });

    } else {
      // hess in the base problem is empty. just insert the new diag elements
      RAJA::forall<hiop_raja_exec>(
          RAJA::RangeSegment(0, m_row),
          RAJA_LAMBDA(RAJA::Index_type i) {
            M1iRow[i] = iHSS[i] = i;
            M1jCol[i] = jHSS[i] = i;
          });
    }
  }

  // extend Hess to the p and n parts --- element
  if(MHSS != nullptr) {
    assert(M1.row_starts_);
    index_type* M1_row_start = M1.row_starts_->idx_start_;

    double* M1values = M1.M();
    const double* M2values = M2.M();

    const double* diag_data = add_diag.local_data_const();

    if(m2 > 0) {
      RAJA::forall<hiop_raja_exec>(
          RAJA::RangeSegment(0, m2),
          RAJA_LAMBDA(RAJA::Index_type i) {
            index_type k = M1_row_start[i];
            index_type k_base = M2_row_start[i];
            size_type nnz_in_row_base = M2_row_start[i + 1] - k_base;

            // insert diagonal entry due to the new obj term
            M1values[k] = MHSS[k] = diag_data[i];

            if(nnz_in_row_base > 0 && M2iRow[k_base] == M2jCol[k_base]) {
              // first nonzero in this row is a diagonal term
              // add it since we will defined the diagonal nonezero
              M1values[k] += M2values[k_base];
              MHSS[k] = M1values[k];
              k_base++;
            }
            k++;

            // copy from base Hess
            while(k_base < M2_row_start[i + 1]) {
              M1values[k] = MHSS[k] = M2values[k_base];
              k++;
              k_base++;
            }
          });
    } else {
      // hess in the base problem is empty. just insert the new diag elements
      RAJA::forall<hiop_raja_exec>(
          RAJA::RangeSegment(0, m_row),
          RAJA_LAMBDA(RAJA::Index_type i) { M1values[i] = MHSS[i] = diag_data[i]; });
    }
  }
  this->copyFromDev();
}

}  // namespace hiop
