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
 * @file VectorHipKernels.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include "VectorHipKernels.hpp"
#include <hip/hip_runtime.h>
#include <hipblas.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

// #include <cmath>
// #include <limits>

/// @brief compute abs(b-a)
template<typename T>
struct thrust_abs_diff : public thrust::binary_function<T, T, T>
{
  __host__ __device__ T operator()(const T& a, const T& b) { return fabs(b - a); }
};

/// @brief compute abs(a)
template<typename T>
struct thrust_abs : public thrust::unary_function<T, T>
{
  __host__ __device__ T operator()(const T& a) { return fabs(a); }
};

/// @brief return true if abs(a) < tol_
struct thrust_abs_less
{
  const double tol_;
  thrust_abs_less(double tol)
      : tol_(tol)
  {}

  __host__ __device__ int operator()(const double& a) { return (fabs(a) < tol_); }
};

/// @brief return true if a < tol_
struct thrust_less
{
  const double tol_;
  thrust_less(double tol)
      : tol_(tol)
  {}

  __host__ __device__ int operator()(const double& a) { return (a < tol_); }
};

/// @brief return true if (0.0 < a) - (a < 0.0)
template<typename T>
struct thrust_sig : public thrust::unary_function<T, T>
{
  __host__ __device__ T operator()(const T& a) { return static_cast<double>((0.0 < a) - (a < 0.0)); }
};

/// @brief compute sqrt(a)
template<typename T>
struct thrust_sqrt : public thrust::unary_function<T, T>
{
  __host__ __device__ T operator()(const T& a) { return sqrt(a); }
};

/// @brief compute log(a) if a > 0, otherwise returns 0
template<typename T>
struct thrust_log_select : public thrust::unary_function<T, double>
{
  __host__ __device__ double operator()(const T& a)
  {
    if(a > 0) {
      return log(a);
    }
    return 0.;
  }
};

/// @brief compute isinf(a)
template<typename T>
struct thrust_isinf : public thrust::unary_function<T, bool>
{
  __host__ __device__ bool operator()(const T& a) { return isinf(a); }
};

/// @brief compute isfinite(a)
template<typename T>
struct thrust_isfinite : public thrust::unary_function<T, bool>
{
  __host__ __device__ bool operator()(const T& a) { return isfinite(a); }
};

/// @brief compute a==0.0
template<typename T>
struct thrust_iszero : public thrust::unary_function<T, bool>
{
  __host__ __device__ bool operator()(const T& a) { return a == (T)(0.0); }
};

/// @brief compute isnan(a)
template<typename T>
struct thrust_isnan : public thrust::unary_function<T, bool>
{
  __host__ __device__ bool operator()(const T& a) { return isnan(a); }
};

/// @brief compute (bool) (a)
struct thrust_istrue : public thrust::unary_function<int, bool>
{
  __host__ __device__ bool operator()(const int& a) { return a != 0; }
};

/** @brief Set y[i] = min(y[i],c), for i=[0,n_local-1] */
__global__ void component_min_hip(int n, double* y, const double c)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    y[i] = (y[i] < c) ? y[i] : c;
  }
}

/** @brief Set y[i] = min(y[i],x[i]), for i=[0,n_local-1] */
__global__ void component_min_hip(int n, double* y, const double* x)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    y[i] = (y[i] < x[i]) ? y[i] : x[i];
  }
}

/** @brief Set y[i] = max(y[i],c), for i=[0,n_local-1] */
__global__ void component_max_hip(int n, double* y, const double c)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    y[i] = (y[i] > c) ? y[i] : c;
  }
}

/** @brief Set y[i] = max(y[i],x[i]), for i=[0,n_local-1] */
__global__ void component_max_hip(int n, double* y, const double* x)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    y[i] = (y[i] > x[i]) ? y[i] : x[i];
  }
}

/// @brief Copy from src the elements specified by the indices in id.
__global__ void copy_from_index_hip(int n, double* vec, const double* val, const int* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    vec[i] = val[id[i]];
  }
}

/// @brief Performs axpy, y += alpha*x, on the indexes in this specified by id.
__global__ void axpy_w_map_hip(int n, double* yd, const double* xd, const int* id, double alpha)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    assert(id[i] < n);
    yd[id[i]] = alpha * xd[i] + yd[id[i]];
  }
}

/** @brief this[i] += alpha*x[i]*z[i] forall i */
__global__ void axzpy_hip(int n, double* yd, const double* xd, const double* zd, double alpha)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    yd[i] = alpha * xd[i] * zd[i] + yd[i];
  }
}

/** @brief this[i] += alpha*x[i]/z[i] forall i */
__global__ void axdzpy_hip(int n, double* yd, const double* xd, const double* zd, double alpha)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    yd[i] = alpha * xd[i] / zd[i] + yd[i];
  }
}

/** @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection */
__global__ void axdzpy_w_pattern_hip(int n, double* yd, const double* xd, const double* zd, const double* id, double alpha)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 1.0) {
      yd[i] = alpha * xd[i] / zd[i] + yd[i];
    }
  }
}

/** @brief y[i] += alpha*1/x[i] + y[i] forall i with pattern selection */
__global__ void adxpy_w_pattern_hip(int n, double* yd, const double* xd, const double* id, double alpha)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 1.0) {
      yd[i] = alpha / xd[i] + yd[i];
    }
  }
}

/**  @brief  elements of this that corespond to nonzeros in ix are divided by elements of v.
     The rest of elements of this are set to zero.*/
__global__ void component_div_w_pattern_hip(int n, double* yd, const double* xd, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 1.0) {
      yd[i] = yd[i] / xd[i];
    } else {
      yd[i] = 0.0;
    }
  }
}

/** @brief y[i] += c forall i */
__global__ void add_constant_hip(int n, double* yd, double c)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    yd[i] = yd[i] + c;
  }
}

/** @brief y[i] += c forall i with pattern selection */
__global__ void add_constant_w_pattern_hip(int n, double* yd, double c, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    yd[i] = yd[i] + c * id[i];
  }
}

/// @brief Invert (1/x) the elements of this
__global__ void invert_hip(int n, double* yd)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    yd[i] = 1. / yd[i];
  }
}

/** @brief Linear damping term */
__global__ void set_linear_damping_term_hip(int n, double* yd, const double* vd, const double* ld, const double* rd)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(ld[i] == 1.0 && rd[i] == 0.0) {
      yd[i] = vd[i];
    } else {
      yd[i] = 0.0;
    }
  }
}

/**
 * @brief Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of
 * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
 */
__global__ void add_linear_damping_term_hip(int n,
                                            double* data,
                                            const double* ixl,
                                            const double* ixr,
                                            double alpha,
                                            double ct)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    data[i] = alpha * data[i] + ct * (ixl[i] - ixr[i]);
  }
}

/** @brief y[i] = 1.0 if x[i] is positive and id[i] = 1.0, otherwise y[i] = 0 */
__global__ void is_posive_w_pattern_hip(int n, double* data, const double* vd, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    data[i] = (id[i] == 1.0 && vd[i] > 0.0) ? 1 : 0;
  }
}

/** @brief y[i] = x[i] if id[i] = 1.0, otherwise y[i] = val_else */
__global__ void set_val_w_pattern_hip(int n, double* data, const double* vd, const double* id, double val_else)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    data[i] = (id[i] == 1.0) ? vd[i] : val_else;
  }
}

/** @brief data[i] = 0 if id[i]==0.0 */
__global__ void select_pattern_hip(int n, double* data, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 0.0) {
      data[i] = 0.0;
    }
  }
}

__global__ void match_pattern_hip(int n, double* data, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 0.0) {
      data[i] = 0.0;
    }
  }
}

/** @brief Project solution into bounds  */
__global__ void project_into_bounds_hip(int n,
                                        double* xd,
                                        const double* xld,
                                        const double* ild,
                                        const double* xud,
                                        const double* iud,
                                        double kappa1,
                                        double kappa2,
                                        double small_real)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = tid; i < n; i += num_threads) {
    double aux = 0.0;
    double aux2 = 0.0;
    if(ild[i] != 0.0 && iud[i] != 0.0) {
      aux = kappa2 * (xud[i] - xld[i]) - small_real;
      aux2 = xld[i] + fmin(kappa1 * fmax(1.0, fabs(xld[i])), aux);
      if(xd[i] < aux2) {
        xd[i] = aux2;
      } else {
        aux2 = xud[i] - fmin(kappa1 * fmax(1.0, fabs(xud[i])), aux);
        if(xd[i] > aux2) {
          xd[i] = aux2;
        }
      }
#ifdef HIOP_DEEPCHECKS
      assert(xd[i] > xld[i] && xd[i] < xud[i] && "this should not happen -> HiOp bug");
#endif
    } else {
      if(ild[i] != 0.0) {
        xd[i] = fmax(xd[i], xld[i] + kappa1 * fmax(1.0, fabs(xld[i])) - small_real);
      }
      if(iud[i] != 0.0) {
        xd[i] = fmin(xd[i], xud[i] - kappa1 * fmax(1.0, fabs(xud[i])) - small_real);
      } else {
        /*nothing for free vars  */
      }
    }
  }
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} */
__global__ void fraction_to_the_boundry_hip(int n, double* yd, const double* xd, const double* dd, double tau)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(dd[i] >= 0) {
      yd[i] = 1.0;
    } else {
      yd[i] = -tau * xd[i] / dd[i];
    }
  }
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select */
__global__ void fraction_to_the_boundry_w_pattern_hip(int n,
                                                      double* yd,
                                                      const double* xd,
                                                      const double* dd,
                                                      const double* id,
                                                      double tau)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(dd[i] >= 0 || id[i] == 0) {
      yd[i] = 1.0;
    } else {
      yd[i] = -tau * xd[i] / dd[i];
    }
  }
}

/** @brief y[i] = 0 if id[i]==0.0 && xd[i]!=0.0, otherwise y[i] = 1*/
__global__ void set_match_pattern_hip(int n, int* yd, const double* xd, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(id[i] == 0.0 && xd[i] != 0.0) {
      yd[i] = 0;
    } else {
      yd[i] = 1;
    }
  }
}

/** @brief Adjusts duals. */
__global__ void adjust_duals_hip(int n, double* zd, const double* xd, const double* id, double mu, double kappa)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double a, b;
  for(int i = tid; i < n; i += num_threads) {
    // preemptive loop to reduce number of iterations?
    if(id[i] == 1.) {
      // precompute a and b in another loop?
      a = mu / xd[i];
      b = a / kappa;
      a = a * kappa;
      // Necessary conditionals
      if(zd[i] < b) {
        zd[i] = b;
      } else {
        // zd[i]>=b
        if(a <= b) {
          zd[i] = b;
        } else {
          // a>b
          if(a < zd[i]) {
            zd[i] = a;
          }
        }
      }
      // - - - -
      // else a>=z[i] then *z=*z (z[i] does not need adjustment)
    }
  }
}

/// set nonlinear type
__global__ void set_nonlinear_type_hip(const int n,
                                       const int length,
                                       hiop::hiopInterfaceBase::NonlinearityType* arr,
                                       const int start,
                                       const hiop::hiopInterfaceBase::NonlinearityType* arr_src,
                                       const int start_src)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n && i < length; i += num_threads) {
    arr[start + i] = arr_src[start_src + i];
  }
}

/// set nonlinear type
__global__ void set_nonlinear_type_hip(const int n,
                                       const int length,
                                       hiop::hiopInterfaceBase::NonlinearityType* arr,
                                       const int start,
                                       const hiop::hiopInterfaceBase::NonlinearityType arr_src)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n && i < length; i += num_threads) {
    arr[start + i] = arr_src;
  }
}

/// for hiopVectorIntHip
/**
 * @brief Set the vector entries to be a linear space of starting at i0 containing evenly
 * incremented integers up to i0+(n-1)di, when n is the length of this vector
 */
__global__ void set_to_linspace_hip(int n, int* vec, int i0, int di)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    vec[i] = i0 + i * di;
  }
}

/** @brief compute cusum from the given pattern*/
__global__ void compute_cusum_hip(int n, int* vec, const double* id)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += num_threads) {
    if(i == 0) {
      vec[i] = 0;
    } else {
      // from i=1..n
      if(id[i - 1] != 0.0) {
        vec[i] = 1;
      } else {
        vec[i] = 0;
      }
    }
  }
}

/// @brief Copy the entries in 'dd' where corresponding 'ix' is nonzero, to vd starting at start_index_in_dest.
__global__ void copyToStartingAt_w_pattern_hip(int n_src,
                                               int n_dest,
                                               int start_index_in_dest,
                                               int* nnz_cumsum,
                                               double* vd,
                                               const double* dd)
{
  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid + 1; i < n_src + 1; i += num_threads) {
    if(nnz_cumsum[i] != nnz_cumsum[i - 1]) {
      int idx_dest = nnz_cumsum[i - 1] + start_index_in_dest;
      vd[idx_dest] = dd[i - 1];
    }
  }
}

namespace hiop
{
namespace hip
{

constexpr int block_size = 256;

/// @brief Copy from src the elements specified by the indices in id.
void copy_from_index_kernel(int n_local, double* yd, const double* src, const int* id)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  copy_from_index_hip<<<num_blocks, block_size>>>(n_local, yd, src, id);
}

/** @brief Set y[i] = min(y[i],c), for i=[0,n_local-1] */
void component_min_kernel(int n_local, double* yd, double c)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  component_min_hip<<<num_blocks, block_size>>>(n_local, yd, c);
}

/** @brief Set y[i] = min(y[i],x[i], for i=[0,n_local-1] */
void component_min_kernel(int n_local, double* yd, const double* xd)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  component_min_hip<<<num_blocks, block_size>>>(n_local, yd, xd);
}

/** @brief Set y[i] = max(y[i],c), for i=[0,n_local-1] */
void component_max_kernel(int n_local, double* yd, double c)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  component_max_hip<<<num_blocks, block_size>>>(n_local, yd, c);
}

/** @brief Set y[i] = max(y[i],x[i]), for i=[0,n_local-1] */
void component_max_kernel(int n_local, double* yd, const double* xd)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  component_max_hip<<<num_blocks, block_size>>>(n_local, yd, xd);
}

/// @brief Performs axpy, y += alpha*x, on the indexes in this specified by id.
void axpy_w_map_kernel(int n_local, double* yd, const double* xd, const int* id, double alpha)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  axpy_w_map_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id, alpha);
}

/** @brief y[i] += alpha*x[i]*z[i] forall i */
void axzpy_kernel(int n_local, double* yd, const double* xd, const double* zd, double alpha)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  axzpy_hip<<<num_blocks, block_size>>>(n_local, yd, xd, zd, alpha);
}

/** @brief y[i] += alpha*x[i]/z[i] forall i */
void axdzpy_kernel(int n_local, double* yd, const double* xd, const double* zd, double alpha)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  axdzpy_hip<<<num_blocks, block_size>>>(n_local, yd, xd, zd, alpha);
}

/** @brief y[i] += alpha*x[i]/z[i] forall i with pattern selection */
void axdzpy_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* zd, const double* id, double alpha)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  axdzpy_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, zd, id, alpha);
}

/** @brief y[i] += c forall i */
void add_constant_kernel(int n_local, double* yd, double c)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  add_constant_hip<<<num_blocks, block_size>>>(n_local, yd, c);
}

/** @brief y[i] += c forall i with pattern selection */
void add_constant_w_pattern_kernel(int n_local, double* yd, const double* id, double c)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  add_constant_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, c, id);
}

/// @brief Invert (1/x) the elements of this
void invert_kernel(int n_local, double* yd)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  invert_hip<<<num_blocks, block_size>>>(n_local, yd);
}

/** @brief y[i] += alpha*1/x[i] + y[i] forall i with pattern selection */
void adxpy_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id, double alpha)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  adxpy_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id, alpha);
}

/**  @brief  elements of this that corespond to nonzeros in ix are divided by elements of v.
     The rest of elements of this are set to zero.*/
void component_div_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  component_div_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id);
}

/** @brief Linear damping term */
void set_linear_damping_term_kernel(int n_local, double* yd, const double* vd, const double* ld, const double* rd)
{
  // compute linear damping term
  int num_blocks = (n_local + block_size - 1) / block_size;
  set_linear_damping_term_hip<<<num_blocks, block_size>>>(n_local, yd, vd, ld, rd);
}

/**
 * @brief Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of
 * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
 */
void add_linear_damping_term_kernel(int n_local, double* yd, const double* ixl, const double* ixr, double alpha, double ct)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  add_linear_damping_term_hip<<<num_blocks, block_size>>>(n_local, yd, ixl, ixr, alpha, ct);
}

/** @brief Checks if selected elements of `this` are positive */
void is_posive_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  is_posive_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id);
}

/// set value with pattern
void set_val_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id, double max_val)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  set_val_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id, max_val);
}

/** @brief Project solution into bounds  */
void project_into_bounds_kernel(int n_local,
                                double* xd,
                                const double* xld,
                                const double* ild,
                                const double* xud,
                                const double* iud,
                                double kappa1,
                                double kappa2,
                                double small_real)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  project_into_bounds_hip<<<num_blocks, block_size>>>(n_local, xd, xld, ild, xud, iud, kappa1, kappa2, small_real);
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} */
void fraction_to_the_boundry_kernel(int n_local, double* yd, const double* xd, const double* dd, double tau)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  fraction_to_the_boundry_hip<<<num_blocks, block_size>>>(n_local, yd, xd, dd, tau);
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select */
void fraction_to_the_boundry_w_pattern_kernel(int n_local,
                                              double* yd,
                                              const double* xd,
                                              const double* dd,
                                              const double* id,
                                              double tau)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  fraction_to_the_boundry_w_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, dd, id, tau);
}

/** @brief Set elements of `this` to zero based on `select`.*/
void select_pattern_kernel(int n_local, double* yd, const double* id)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  select_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, id);
}

/** @brief y[i] = 0 if id[i]==0.0 && xd[i]!=0.0, otherwise y[i] = 1*/
void component_match_pattern_kernel(int n_local, int* yd, const double* xd, const double* id)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  set_match_pattern_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id);
}

/** @brief Adjusts duals. */
void adjustDuals_plh_kernel(int n_local, double* yd, const double* xd, const double* id, double mu, double kappa)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  adjust_duals_hip<<<num_blocks, block_size>>>(n_local, yd, xd, id, mu, kappa);
}

/// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
void set_array_from_to_kernel(int n_local,
                              hiop::hiopInterfaceBase::NonlinearityType* arr,
                              int start,
                              int length,
                              const hiop::hiopInterfaceBase::NonlinearityType* arr_src,
                              int start_src)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  set_nonlinear_type_hip<<<num_blocks, block_size>>>(n_local, length, arr, start, arr_src, start_src);
}

/// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
void set_array_from_to_kernel(int n_local,
                              hiop::hiopInterfaceBase::NonlinearityType* arr,
                              int start,
                              int length,
                              hiop::hiopInterfaceBase::NonlinearityType arr_src)
{
  int num_blocks = (n_local + block_size - 1) / block_size;
  set_nonlinear_type_hip<<<num_blocks, block_size>>>(n_local, length, arr, start, arr_src);
}

/// @brief Set all elements to c.
void thrust_fill_kernel(int n, double* ptr, double c)
{
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(ptr);
  thrust::fill(thrust::device, dev_ptr, dev_ptr + n, c);
}

/** @brief inf norm on single rank */
double infnorm_local_kernel(int n, double* data_dev)
{
  thrust_abs<double> abs_op;
  thrust::maximum<double> max_op;
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(data_dev);

  // compute one norm
  double norm = thrust::transform_reduce(thrust::device, data_dev, data_dev + n, abs_op, 0.0, max_op);

  return norm;
}

/** @brief Return the one norm */
double onenorm_local_kernel(int n, double* data_dev)
{
  thrust_abs<double> abs_op;
  thrust::plus<double> plus_op;
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(data_dev);
  // thrust::device_ptr<double> dev_ptr(data_dev);

  // compute one norm
  double norm = thrust::transform_reduce(thrust::device, data_dev, data_dev + n, abs_op, 0.0, plus_op);

  return norm;
}

/** @brief d1[i] = d1[i] * d2[i] forall i */
void thrust_component_mult_kernel(int n, double* d1, const double* d2)
{
  // wrap raw pointer with a device_ptr
  thrust::multiplies<double> mult_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);
  thrust::device_ptr<const double> dev_v2 = thrust::device_pointer_cast(d2);

  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v2, dev_v1, mult_op);
}

/** @brief d1[i] = d1[i] / d2[i] forall i */
void thrust_component_div_kernel(int n, double* d1, const double* d2)
{
  // wrap raw pointer with a device_ptr
  thrust::divides<double> div_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);
  thrust::device_ptr<const double> dev_v2 = thrust::device_pointer_cast(d2);

  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v2, dev_v1, div_op);
}

/** @brief d1[i] = abs(d1[i]) forall i */
void thrust_component_abs_kernel(int n, double* d1)
{
  // wrap raw pointer with a device_ptr
  thrust_abs<double> abs_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);

  // compute abs
  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v1, abs_op);
}

/** @brief d1[i] = sign(d1[i]) forall i */
void thrust_component_sgn_kernel(int n, double* d1)
{
  // wrap raw pointer with a device_ptr
  thrust_sig<double> sig_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);

  // compute sign
  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v1, sig_op);
}

/** @brief d1[i] = sqrt(d1[i]) forall i */
void thrust_component_sqrt_kernel(int n, double* d1)
{
  // wrap raw pointer with a device_ptr
  thrust_sqrt<double> sqrt_op;
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);

  // compute sqrt
  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v1, sqrt_op);
}

/** @brief d1[i] = -(d1[i]) forall i */
void thrust_negate_kernel(int n, double* d1)
{
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);
  thrust::transform(thrust::device, dev_v1, dev_v1 + n, dev_v1, thrust::negate<double>());
}

/** @brief compute sum(log(d1[i])) forall i where id[i]=1*/
double log_barr_obj_kernel(int n, double* d1, const double* id)
{
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(d1);
  thrust::device_ptr<const double> id_v = thrust::device_pointer_cast(id);

  // wrap raw pointer with a device_ptr
  thrust_log_select<double> log_select_op;
  thrust::plus<double> plus_op;
  thrust::multiplies<double> mult_op;

  // TODO: how to avoid this temp vec?
  thrust::device_ptr<double> v_temp = thrust::device_malloc(n * sizeof(double));

  // compute x*id
  thrust::transform(thrust::device, dev_v, dev_v + n, id_v, v_temp, mult_op);
  // compute log(y) for y > 0
  double sum = thrust::transform_reduce(thrust::device, v_temp, v_temp + n, log_select_op, 0.0, plus_op);

  thrust::device_free(v_temp);

  return sum;
}

/** @brief compute sum(d1[i]) */
double thrust_sum_kernel(int n, double* d1)
{
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);
  // compute sum
  return thrust::reduce(thrust::device, dev_v1, dev_v1 + n, 0.0, thrust::plus<double>());
}

/** @brief Linear damping term */
double linear_damping_term_kernel(int n, const double* vd, const double* ld, const double* rd, double mu, double kappa_d)
{
  // TODO: how to avoid this temp vec?
  thrust::device_vector<double> v_temp(n);
  double* dv_ptr = thrust::raw_pointer_cast(v_temp.data());

  // compute linear damping term
  hiop::hip::set_linear_damping_term_kernel(n, dv_ptr, vd, ld, rd);

  double term = thrust::reduce(thrust::device, v_temp.begin(), v_temp.end(), 0.0, thrust::plus<double>());

  term *= mu;
  term *= kappa_d;
  return term;
}

/** @brief compute min(d1) */
double min_local_kernel(int n, double* d1)
{
  thrust::device_ptr<double> dev_v1 = thrust::device_pointer_cast(d1);
  thrust::device_ptr<double> ret_dev_ptr = thrust::min_element(thrust::device, dev_v1, dev_v1 + n);

  double* ret_ptr = thrust::raw_pointer_cast(ret_dev_ptr);
  double* ret_host = new double[1];
  hipError_t cuerr = hipMemcpy(ret_host, ret_ptr, (1) * sizeof(double), hipMemcpyDeviceToHost);

  double rv = ret_host[0];
  delete[] ret_host;

  return rv;
}

/** @brief Checks if selected elements of `this` are positive */
int all_positive_w_pattern_kernel(int n, const double* d1, const double* id)
{
  // TODO: how to avoid this temp vec?
  thrust::device_vector<double> v_temp(n);
  double* dv_ptr = thrust::raw_pointer_cast(v_temp.data());

  hiop::hip::is_posive_w_pattern_kernel(n, dv_ptr, d1, id);

  return thrust::reduce(thrust::device, v_temp.begin(), v_temp.end(), (int)0, thrust::plus<int>());
}

/** @brief compute min(d1) for selected elements*/
double min_w_pattern_kernel(int n, const double* d1, const double* id, double max_val)
{
  // TODO: how to avoid this temp vec?
  thrust::device_ptr<double> dv_ptr = thrust::device_malloc(n * sizeof(double));
  double* d_ptr = thrust::raw_pointer_cast(dv_ptr);

  // set value with pattern
  hiop::hip::set_val_w_pattern_kernel(n, d_ptr, d1, id, max_val);

  thrust::device_ptr<double> ret_dev_ptr = thrust::min_element(thrust::device, dv_ptr, dv_ptr + n);

  // TODO: how to return double from device to host?
  double* ret_host = new double[1];
  double* ret_ptr = thrust::raw_pointer_cast(ret_dev_ptr);
  hipError_t cuerr = hipMemcpy(ret_host, ret_ptr, (1) * sizeof(double), hipMemcpyDeviceToHost);

  double ret_v = ret_host[0];
  delete[] ret_host;

  thrust::device_free(dv_ptr);

  return ret_v;
}

/** @brief check if xld[i] < xud[i] forall i */
bool check_bounds_kernel(int n, const double* xld, const double* xud)
{
  // Perform preliminary check to see of all upper value
  thrust::minus<double> minus_op;
  thrust::device_ptr<double> dev_xud = thrust::device_pointer_cast(const_cast<double*>(xud));
  thrust::device_ptr<double> dev_xld = thrust::device_pointer_cast(const_cast<double*>(xld));

  // TODO: how to avoid this temp vec?
  thrust::device_ptr<double> dv_ptr = thrust::device_malloc(n * sizeof(double));

  thrust::transform(thrust::device, dev_xud, dev_xud + n, dev_xld, dv_ptr, minus_op);

  int res_offset = thrust::min_element(thrust::device, dv_ptr, dv_ptr + n) - dv_ptr;
  double ret_v = *(dv_ptr + res_offset);

  bool bval = (ret_v > 0.0) ? 1 : 0;

  thrust::device_free(dv_ptr);

  if(false == bval) return false;

  return true;
}

/** @brief compute max{a\in(0,1]| x+ad >=(1-tau)x} */
double min_frac_to_bds_kernel(int n, const double* xd, const double* dd, double tau)
{
  thrust::device_ptr<double> dv_ptr = thrust::device_malloc(n * sizeof(double));
  double* d_ptr = thrust::raw_pointer_cast(dv_ptr);

  // set values
  hiop::hip::fraction_to_the_boundry_kernel(n, d_ptr, xd, dd, tau);
  int res_offset = thrust::min_element(thrust::device, dv_ptr, dv_ptr + n) - dv_ptr;
  double alpha = *(dv_ptr + res_offset);

  thrust::device_free(dv_ptr);

  return alpha;
}

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern id */
double min_frac_to_bds_w_pattern_kernel(int n, const double* xd, const double* dd, const double* id, double tau)
{
  // TODO: how to avoid this temp vec?
  thrust::device_vector<double> v_temp(n);
  double* dv_ptr = thrust::raw_pointer_cast(v_temp.data());

  // set value with pattern
  hiop::hip::fraction_to_the_boundry_w_pattern_kernel(n, dv_ptr, xd, dd, id, tau);
  double alpha = *(thrust::min_element(thrust::device, v_temp.begin(), v_temp.end()));

  return alpha;
}

/** @brief Checks if `xd` matches nonzero pattern of `id`. */
bool match_pattern_kernel(int n, const double* xd, const double* id)
{
  // TODO: how to avoid this temp vec?
  thrust::device_vector<int> v_temp(n);
  int* dv_ptr = thrust::raw_pointer_cast(v_temp.data());

  // check if xd matches the pattern given by id
  hiop::hip::component_match_pattern_kernel(n, dv_ptr, xd, id);

  thrust_istrue istrue_op;

  return thrust::all_of(thrust::device, v_temp.begin(), v_temp.end(), istrue_op);
}

/** @brief Checks if all x[i] = 0 */
bool is_zero_kernel(int n, double* xd)
{
  // wrap raw pointer with a device_ptr
  thrust_iszero<double> iszero_op;
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);

  return thrust::all_of(thrust::device, dev_v, dev_v + n, iszero_op);
}

/** @brief Checks if any x[i] = nan */
bool isnan_kernel(int n, double* xd)
{
  // wrap raw pointer with a device_ptr
  thrust_isnan<double> isnan_op;
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);

  return thrust::any_of(thrust::device, dev_v, dev_v + n, isnan_op);
}

/** @brief Checks if any x[i] = inf */
bool isinf_kernel(int n, double* xd)
{
  // wrap raw pointer with a device_ptr
  thrust_isinf<double> isinf_op;
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);

  return thrust::any_of(thrust::device, dev_v, dev_v + n, isinf_op);
}

/** @brief Checks if all x[i] != inf */
bool isfinite_kernel(int n, double* xd)
{
  // wrap raw pointer with a device_ptr
  thrust_isfinite<double> isfinite_op;
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);

  return thrust::all_of(thrust::device, dev_v, dev_v + n, isfinite_op);
}

/// @brief get number of values that are less than the given value 'val'.
int num_of_elem_less_than_kernel(int n, double* xd, double val)
{
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);
  int rval = thrust::transform_reduce(thrust::device, dev_v, dev_v + n, thrust_less(val), (int)0, thrust::plus<int>());
  return rval;
}

/// @brief get number of values whose absolute value are less than the given value 'val'.
int num_of_elem_absless_than_kernel(int n, double* xd, double val)
{
  thrust::device_ptr<double> dev_v = thrust::device_pointer_cast(xd);
  int rval = thrust::transform_reduce(thrust::device, dev_v, dev_v + n, thrust_abs_less(val), (int)0, thrust::plus<int>());
  return rval;
}

/// @brief Copy the entries in 'dd' where corresponding 'ix' is nonzero, to vd starting at start_index_in_dest.
void copyToStartingAt_w_pattern_kernel(int n_src,
                                       int n_dest,
                                       int start_index_in_dest,
                                       int* nnz_cumsum,
                                       double* vd,
                                       const double* dd)
{
  int num_blocks = (n_src + block_size - 1) / block_size;
  copyToStartingAt_w_pattern_hip<<<num_blocks, block_size>>>(n_src, n_dest, start_index_in_dest, nnz_cumsum, vd, dd);
}

/// for hiopVectorIntHip
/**
 * @brief Set the vector entries to be a linear space of starting at i0 containing evenly
 * incremented integers up to i0+(n-1)di, when n is the length of this vector
 */
void set_to_linspace_kernel(int sz, int* buf, int i0, int di)
{
  int num_blocks = (sz + block_size - 1) / block_size;
  set_to_linspace_hip<<<num_blocks, block_size>>>(sz, buf, i0, di);
}

/** @brief compute cusum from the given pattern*/
void compute_cusum_kernel(int sz, int* buf, const double* id)
{
  int num_blocks = (sz + block_size - 1) / block_size;
  compute_cusum_hip<<<num_blocks, block_size>>>(sz, buf, id);

  thrust::device_ptr<int> dev_v = thrust::device_pointer_cast(buf);
  thrust::inclusive_scan(dev_v, dev_v + sz, dev_v);  // in-place scan
}

}  // namespace hip

}  // namespace hiop
