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
 * @file VectorCudaKernels.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#ifndef HIOP_VECTOR_CUDA_KER
#define HIOP_VECTOR_CUDA_KER

#include <thrust/functional.h>
#include "hiopInterface.hpp"

namespace hiop
{
namespace cuda
{

/// @brief Copy from src the elements specified by the indices in id.
void copy_from_index_kernel(int n_local, double* yd, const double* src, const int* id);

/** @brief Set y[i] = min(y[i],c), for i=[0,n_local-1] */
void component_min_kernel(int n_local, double* yd, double c);

/** @brief Set y[i] = min(y[i],x[i]), for i=[0,n_local-1] */
void component_min_kernel(int n_local, double* yd, const double* xd);

/** @brief Set y[i] = max(y[i],c), for i=[0,n_local-1] */
void component_max_kernel(int n_local, double* yd, double c);

/** @brief Set y[i] = max(y[i],x[i]), for i=[0,n_local-1] */
void component_max_kernel(int n_local, double* yd, const double* xd);

/// @brief Performs axpy, this += alpha*x, on the indexes in this specified by i.
void axpy_w_map_kernel(int n_local, double* yd, const double* xd, const int* id, double alpha);

/** @brief this[i] += alpha*x[i]*z[i] forall i */
void axzpy_kernel(int n_local, double* yd, const double* xd, const double* zd, double alpha);

/** @brief this[i] += alpha*x[i]/z[i] forall i */
void axdzpy_kernel(int n_local, double* yd, const double* xd, const double* zd, double alpha);

/** @brief this[i] += alpha*x[i]/z[i] forall i with pattern selection */
void axdzpy_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* zd, const double* id, double alpha);

/** @brief this[i] += c forall i */
void add_constant_kernel(int n_local, double* yd, double c);

/** @brief this[i] += c forall i with pattern selection */
void add_constant_w_pattern_kernel(int n_local, double* yd, const double* id, double c);

/// @brief Invert (1/x) the elements of this
void invert_kernel(int n_local, double* yd);

/** @brief y[i] += alpha*1/x[i] + y[i] forall i with pattern selection */
void adxpy_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* ld, double alpha);

/** @brief y[i] = y[i]/x[i] c forall i with pattern selection */
void component_div_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id);

/** @brief Linear damping term */
void set_linear_damping_term_kernel(int n_local, double* yd, const double* vd, const double* ld, const double* rd);

/**
 * @brief Performs `this[i] = alpha*this[i] + sign*ct` where sign=1 when EXACTLY one of
 * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
 */
void add_linear_damping_term_kernel(int n_local, double* yd, const double* ixl, const double* ixr, double alpha, double ct);

/** @brief y[i] = 1.0 if x[i] is positive and id[i] = 1.0, otherwise y[i] = 0 */
void is_posive_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id);

/** @brief y[i] = x[i] if id[i] = 1.0, otherwise y[i] = val_else */
void set_val_w_pattern_kernel(int n_local, double* yd, const double* xd, const double* id, double val_else);

/** @brief Project solution into bounds  */
void project_into_bounds_kernel(int n_local,
                                double* xd,
                                const double* xld,
                                const double* ild,
                                const double* xud,
                                const double* iud,
                                double kappa1,
                                double kappa2,
                                double small_real);

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} */
void fraction_to_the_boundry_kernel(int n_local, double* yd, const double* xd, const double* dd, double tau);

/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern select */
void fraction_to_the_boundry_w_pattern_kernel(int n_local,
                                              double* yd,
                                              const double* xd,
                                              const double* dd,
                                              const double* id,
                                              double tau);

/** @brief Set elements of `this` to zero based on `select`.*/
void select_pattern_kernel(int n_local, double* yd, const double* id);

/** @brief Checks if each component in `this` matches nonzero pattern of `select`.  */
void component_match_pattern_kernel(int n_local, int* yd, const double* xd, const double* id);

/** @brief Adjusts duals. */
void adjustDuals_plh_kernel(int n_local, double* yd, const double* xd, const double* id, double mu, double kappa);

/// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
void set_array_from_to_kernel(int n_local,
                              hiop::hiopInterfaceBase::NonlinearityType* arr,
                              int start,
                              int length,
                              const hiop::hiopInterfaceBase::NonlinearityType* arr_src,
                              int start_src);

/// @brief set int array 'arr', starting at `start` and ending at `end`, to the values in `arr_src` from 'start_src`
void set_array_from_to_kernel(int n_local,
                              hiop::hiopInterfaceBase::NonlinearityType* arr,
                              int start,
                              int length,
                              hiop::hiopInterfaceBase::NonlinearityType arr_src);

/// @brief Set all elements to c.
void thrust_fill_kernel(int n, double* ptr, double c);

/** @brief inf norm on single rank */
double infnorm_local_kernel(int n, double* data_dev);
/** @brief Return the one norm */
double onenorm_local_kernel(int n, double* data_dev);

/** @brief d1[i] = d1[i] * d2[i] forall i */
void thrust_component_mult_kernel(int n, double* d1, const double* d2);
/** @brief d1[i] = d1[i] / d2[i] forall i */
void thrust_component_div_kernel(int n, double* d1, const double* d2);
/** @brief d1[i] = abs(d1[i]) forall i */
void thrust_component_abs_kernel(int n, double* d1);
/** @brief d1[i] = sign(d1[i]) forall i */
void thrust_component_sgn_kernel(int n, double* d1);
/** @brief d1[i] = sqrt(d1[i]) forall i */
void thrust_component_sqrt_kernel(int n, double* d1);
/** @brief d1[i] = -(d1[i]) forall i */
void thrust_negate_kernel(int n, double* d1);
/** @brief compute sum(log(d1[i])) forall i where id[i]=1*/
double log_barr_obj_kernel(int n, double* d1, const double* id);
/** @brief compute sum(d1[i]) */
double thrust_sum_kernel(int n, double* d1);
/** @brief Linear damping term */
double linear_damping_term_kernel(int n, const double* vd, const double* ld, const double* rd, double mu, double kappa_d);
/** @brief compute min(d1) */
double min_local_kernel(int n, double* d1);
/** @brief Checks if selected elements of `d1` are positive */
int all_positive_w_pattern_kernel(int n, const double* d1, const double* id);
/** @brief compute min(d1) for selected elements*/
double min_w_pattern_kernel(int n, const double* d1, const double* id, double max_val);
/** @brief check if xld[i] < xud[i] forall i */
bool check_bounds_kernel(int n, const double* xld, const double* xud);
/** @brief compute max{a\in(0,1]| x+ad >=(1-tau)x} */
double min_frac_to_bds_kernel(int n, const double* xd, const double* dd, double tau);
/** @brief max{a\in(0,1]| x+ad >=(1-tau)x} with pattern id */
double min_frac_to_bds_w_pattern_kernel(int n, const double* xd, const double* dd, const double* id, double tau);
/** @brief Checks if `xd` matches nonzero pattern of `id`. */
bool match_pattern_kernel(int n, const double* xd, const double* id);
/** @brief Checks if all x[i] = 0 */
bool is_zero_kernel(int n, double* xd);
/** @brief Checks if any x[i] = nan */
bool isnan_kernel(int n, double* xd);
/** @brief Checks if any x[i] = inf */
bool isinf_kernel(int n, double* xd);
/** @brief Checks if all x[i] != inf */
bool isfinite_kernel(int n, double* xd);
/// @brief get number of values that are less than the given value 'val'.
int num_of_elem_less_than_kernel(int n, double* xd, double val);
/// @brief get number of values whose absolute value are less than the given value 'val'.
int num_of_elem_absless_than_kernel(int n, double* xd, double val);

/// @brief Copy the entries in 'dd' where corresponding 'ix' is nonzero, to vd starting at start_index_in_dest.
void copyToStartingAt_w_pattern_kernel(int n_src,
                                       int n_dest,
                                       int start_index_in_dest,
                                       int* nnz_cumsum,
                                       double* vd,
                                       const double* dd);

/// for hiopVectorIntCuda
/**
 * @brief Set the vector entries to be a linear space of starting at i0 containing evenly
 * incremented integers up to i0+(n-1)di, when n is the length of this vector
 *
 */
void set_to_linspace_kernel(int sz, int* buf, int i0, int di);
/** @brief compute cusum from the given pattern*/
void compute_cusum_kernel(int sz, int* buf, const double* id);

}  // namespace cuda
}  // namespace hiop
#endif
