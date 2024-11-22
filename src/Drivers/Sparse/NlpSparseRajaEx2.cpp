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
 * @file NlpSparseRajaEx2.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 *
 */

#include "NlpSparseRajaEx2.hpp"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>

// #include <hiopMatrixRajaSparseTriplet.hpp>

#include <cmath>
#include <cstring>  //for memcpy
#include <cstdio>

// TODO: A good idea to not use the internal HiOp Raja policies here and, instead, give self-containing
//  definitions of the policies here so that the user gets a better grasp of the concept and does not
//  rely on the internals of HiOp. For example:
//  #define RAJA_LAMBDA [=] __device__
//  using ex2_raja_exec = RAJA::cuda_exec<128>;
//  more defs here

#if defined(HIOP_USE_CUDA)
#include "ExecPoliciesRajaCudaImpl.hpp"
using ex2_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_exec;
using ex2_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaCuda>::hiop_raja_reduce;
// using hiopMatrixRajaSparse = hiop::hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using ex2_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_exec;
using ex2_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaHip>::hiop_raja_reduce;
// using hiopMatrixRajaSparse = hiop::hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using ex2_raja_exec = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaOmp>::hiop_raja_exec;
using ex2_raja_reduce = hiop::ExecRajaPoliciesBackend<hiop::ExecPolicyRajaOmp>::hiop_raja_reduce;
// using hiopMatrixRajaSparse = hiop::hiopMatrixRajaSparseTriplet<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
#endif

/** Nonlinear *highly nonconvex* and *rank deficient* problem test for the Filter IPM
 * Newton of HiOp. It uses a Sparse NLP formulation. The problem is based on SparseEx1.
 *
 *  min   (2*convex_obj-1)*scal*sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + 0.5x^Tx
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.0 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]                  (rnkdef-con1)
 *        4*x_1 + 2*x_2 == 10                                (rnkdef-con2)
 *
 *  other parameters are:
 *  convex_obj: set to 1 to have a convex problem, otherwise set it to 0.
 *  scale_quartic_obj_term: scaling factor for the quartic term in the objective (1.0 by default).
 *
 */
SparseRajaEx2::SparseRajaEx2(std::string mem_space,
                             int n,
                             bool convex_obj,
                             bool rankdefic_Jac_eq,
                             bool rankdefic_Jac_ineq,
                             double scal_neg_obj)
    : mem_space_{mem_space},
      convex_obj_{convex_obj},
      rankdefic_eq_{rankdefic_Jac_eq},
      rankdefic_ineq_{rankdefic_Jac_ineq},
      n_vars_{n},
      scal_neg_obj_{scal_neg_obj},
      n_cons_{2}
{
  // Make sure mem_space_ is uppercase
  transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);

  assert(n >= 3 && "number of variables should be greater than 3 for this example");
  if(n > 3) {
    n_cons_ += n - 3;
  }
  n_cons_ += rankdefic_eq_ + rankdefic_ineq_;
}

SparseRajaEx2::~SparseRajaEx2() {}

bool SparseRajaEx2::get_prob_sizes(size_type& n, size_type& m)
{
  n = n_vars_;
  m = n_cons_;
  return true;
}

bool SparseRajaEx2::get_vars_info(const size_type& n, double* xlow, double* xupp, NonlinearityType* type)
{
  assert(n == n_vars_);

  RAJA::forall<ex2_raja_exec>(
      RAJA::RangeSegment(0, 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        xlow[0] = -1e20;
        xupp[0] = 1e20;
        //      type[0] = hiopNonlinear;
        xlow[1] = 0.0;
        xupp[1] = 1e20;
        //      type[1] = hiopNonlinear;
        xlow[2] = 1.0;
        xupp[2] = 10.0;
        //      type[2] = hiopNonlinear;
      });

  if(n > 3) {
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(3, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          xlow[i] = 0.5;
          xupp[i] = 1e20;
          //        type[i] = hiopNonlinear;
        });
  }

  // Use a sequential policy for host computations for now
  RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, n), [=](RAJA::Index_type i) { type[i] = hiopNonlinear; });

  return true;
}

bool SparseRajaEx2::get_cons_info(const size_type& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m == n_cons_);
  size_type n = n_vars_;
  assert(m - 1 == n - 1 + rankdefic_ineq_);

  // RAJA doesn't like member objects
  bool rankdefic_eq = rankdefic_eq_;
  bool rankdefic_ineq = rankdefic_ineq_;

  // serial part
  RAJA::forall<ex2_raja_exec>(
      RAJA::RangeSegment(0, 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        clow[0] = 10.0;
        cupp[0] = 10.0;
        //      type[0] = hiopInterfaceBase::hiopNonlinear;
        clow[1] = 5.0;
        cupp[1] = 1e20;
        //      type[1] = hiopInterfaceBase::hiopNonlinear;

        if(rankdefic_ineq) {
          // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
          clow[n - 1] = -1e+20;
          cupp[n - 1] = 19.;
          //        type[n-1] = hiopInterfaceBase::hiopNonlinear;
        }

        if(rankdefic_eq) {
          //  4*x_1 + 2*x_2 == 10
          clow[m - 1] = 10;
          cupp[m - 1] = 10;
          //          type[m-1] = hiopInterfaceBase::hiopNonlinear;
        }
      });

  if(n > 3) {
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(2, n - 1),
        RAJA_LAMBDA(RAJA::Index_type conidx) {
          clow[conidx] = 1.0;
          cupp[conidx] = 2 * n;
          //        type[conidx] = hiopInterfaceBase::hiopNonlinear;
        });
  }

  // Must be a sequential host policy for now
  RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, m), [=](RAJA::Index_type i) { type[i] = hiopNonlinear; });

  return true;
}

bool SparseRajaEx2::get_sparse_blocks_info(int& nx,
                                           int& nnz_sparse_Jaceq,
                                           int& nnz_sparse_Jacineq,
                                           int& nnz_sparse_Hess_Lagr)
{
  nx = n_vars_;
  ;
  nnz_sparse_Jaceq = 2 + 2 * rankdefic_eq_;
  nnz_sparse_Jacineq = 2 + 2 * (n_vars_ - 3) + 2 * rankdefic_ineq_;
  nnz_sparse_Hess_Lagr = n_vars_;
  return true;
}

bool SparseRajaEx2::eval_f(const size_type& n, const double* x, bool new_x, double& obj_value)
{
  assert(n == n_vars_);
  obj_value = 0.;
  {
    int convex_obj = (int)convex_obj_;
    double scal_neg_obj = scal_neg_obj_;

    RAJA::ReduceSum<ex2_raja_reduce, double> aux(0);
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          aux += (2 * convex_obj - 1) * scal_neg_obj * 0.25 * std::pow(x[i] - 1., 4) + 0.5 * std::pow(x[i], 2);
        });
    obj_value += aux.get();
  }
  return true;
}

bool SparseRajaEx2::eval_grad_f(const size_type& n, const double* x, bool new_x, double* gradf)
{
  assert(n == n_vars_);
  {
    int convex_obj = (int)convex_obj_;
    double scal_neg_obj = scal_neg_obj_;

    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) { gradf[i] = (2 * convex_obj - 1) * scal_neg_obj * std::pow(x[i] - 1., 3) + x[i]; });
  }
  return true;
}

bool SparseRajaEx2::eval_cons(const size_type& n, const size_type& m, const double* x, bool new_x, double* cons)
{
  assert(n == n_vars_);
  assert(m == n_cons_);
  assert(n_cons_ == 2 + n - 3 + rankdefic_eq_ + rankdefic_ineq_);

  // RAJA doesn't like member objects
  bool rankdefic_eq = rankdefic_eq_;
  bool rankdefic_ineq = rankdefic_ineq_;

  // serial part
  RAJA::forall<ex2_raja_exec>(
      RAJA::RangeSegment(0, 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
        cons[0] = 4 * x[0] + 2 * x[1];
        // --- constraint 2 body ---> 2*x_1 + x_3
        cons[1] = 2 * x[0] + 1 * x[2];

        if(rankdefic_ineq) {
          // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
          cons[n - 1] = 4 * x[0] + 2 * x[2];
        }

        if(rankdefic_eq) {
          //  4*x_1 + 2*x_2 == 10
          cons[m - 1] = 4 * x[0] + 2 * x[1];
        }
      });

  RAJA::forall<ex2_raja_exec>(
      RAJA::RangeSegment(2, n - 1),
      RAJA_LAMBDA(RAJA::Index_type i) {
        // --- constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
        cons[i] = 2 * x[0] + 0.5 * x[i + 1];
      });

  return true;
}

bool SparseRajaEx2::eval_Jac_cons(const size_type& n,
                                  const size_type& m,
                                  const double* x,
                                  bool new_x,
                                  const int& nnzJacS,
                                  index_type* iJacS,
                                  index_type* jJacS,
                                  double* MJacS)
{
  assert(n == n_vars_);
  assert(m == n_cons_);
  assert(n >= 3);

  assert(nnzJacS == 4 + 2 * (n - 3) + 2 * rankdefic_eq_ + 2 * rankdefic_ineq_);

  // RAJA doesn't like member objects
  bool rankdefic_eq = rankdefic_eq_;
  bool rankdefic_ineq = rankdefic_ineq_;

  if(iJacS != nullptr && jJacS != nullptr) {
    // serial part
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, 1),
        RAJA_LAMBDA(RAJA::Index_type itrow) {
          // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
          iJacS[0] = 0;
          jJacS[0] = 0;
          iJacS[1] = 0;
          jJacS[1] = 1;
          // --- constraint 2 body ---> 2*x_1 + x_3
          iJacS[2] = 1;
          jJacS[2] = 0;
          iJacS[3] = 1;
          jJacS[3] = 2;

          if(rankdefic_ineq) {
            // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
            iJacS[2 * n - 2] = n - 1;
            jJacS[2 * n - 2] = 0;
            iJacS[2 * n - 1] = n - 1;
            jJacS[2 * n - 1] = 2;
          }

          if(rankdefic_eq) {
            //  4*x_1 + 2*x_2 == 10
            iJacS[2 * m - 2] = m - 1;
            jJacS[2 * m - 2] = 0;
            iJacS[2 * m - 1] = m - 1;
            jJacS[2 * m - 1] = 1;
          }
        });

    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(2, n - 1),
        RAJA_LAMBDA(RAJA::Index_type itrow) {
          // --- constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
          iJacS[2 * itrow] = itrow;
          jJacS[2 * itrow] = 0;
          iJacS[2 * itrow + 1] = itrow;
          jJacS[2 * itrow + 1] = itrow + 1;
        });
  }

  if(MJacS != nullptr) {
    // serial part
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, 1),
        RAJA_LAMBDA(RAJA::Index_type itrow) {
          // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
          MJacS[0] = 4.0;
          MJacS[1] = 2.0;
          // --- constraint 2 body ---> 2*x_1 + x_3
          MJacS[2] = 2.0;
          MJacS[3] = 1.0;

          if(rankdefic_ineq) {
            // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
            MJacS[2 * n - 2] = 4.0;
            MJacS[2 * n - 1] = 2.0;
          }

          if(rankdefic_eq) {
            //  4*x_1 + 2*x_2 == 10
            MJacS[2 * m - 2] = 4.0;
            MJacS[2 * m - 1] = 2.0;
          }
        });

    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(2, n - 1),
        RAJA_LAMBDA(RAJA::Index_type itrow) {
          // --- constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
          MJacS[2 * itrow] = 2.0;
          MJacS[2 * itrow + 1] = 0.5;
        });
  }

  return true;
}

bool SparseRajaEx2::eval_Hess_Lagr(const size_type& n,
                                   const size_type& m,
                                   const double* x,
                                   bool new_x,
                                   const double& obj_factor,
                                   const double* lambda,
                                   bool new_lambda,
                                   const size_type& nnzHSS,
                                   index_type* iHSS,
                                   index_type* jHSS,
                                   double* MHSS)
{
  // Note: lambda is not used since all the constraints are linear and, therefore, do
  // not contribute to the Hessian of the Lagrangian
  assert(nnzHSS == n);

  if(iHSS != nullptr && jHSS != nullptr) {
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          iHSS[i] = i;
          jHSS[i] = i;
        });
  }

  int convex_obj = (int)convex_obj_;
  double scal_neg_obj = scal_neg_obj_;
  if(MHSS != nullptr) {
    RAJA::forall<ex2_raja_exec>(
        RAJA::RangeSegment(0, n),
        RAJA_LAMBDA(RAJA::Index_type i) {
          MHSS[i] = obj_factor * ((2 * convex_obj - 1) * scal_neg_obj * 3 * std::pow(x[i] - 1., 2) + 1);
        });
  }
  return true;
}

bool SparseRajaEx2::get_starting_point(const size_type& n, double* x0)
{
  assert(n == n_vars_);
  RAJA::forall<ex2_raja_exec>(RAJA::RangeSegment(0, n), RAJA_LAMBDA(RAJA::Index_type i) { x0[i] = 0.0; });
  return true;
}
