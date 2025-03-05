// Copyright (c) 2025, Lawrence Livermore National Security, LLC.
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
 * @file InnerProduct.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_NLP_INNERPROD
#define HIOP_NLP_INNERPROD

#include "hiopVector.hpp"

namespace hiop
{

// some forward decls
class hiopNlpFormulation;

/** 
 * Provides functionality required for using (weighted) inner products within the IPM algorithm(s). 
 *
 * These weighted inner products appear when optimizing over (discretization of) function spaces,
 * such as PDE-constrained optimization. It wraps around user-provided methods for computing the 
 * mass matrix M and the weight matrix H generally associated with L^2 or H^1 finite element 
 * discretizations and corresponding weighted inner products: <u_h,v_h> = u^T H v. For L^2, 
 * H is the mass matrix, while for H^1 is the mass plus stiffness. These user methods are called
 * to perform various operations associated with Hilbert spaces, such as inner products and norms. 
 *
 * Additional info: C. G. Petra et. al., On the implementation of a quasi-Newton 
 * interior-point method for PDE-constrained optimization using finite element 
 * discretizations, Optimiz. Meth. and Software, Vol. 38, 2023.
 *
 * This class also covers Euclidean (i.e., non-weighted) inner products, for which M=H=I.
 */
class InnerProduct
{
public:
  InnerProduct(hiopNlpFormulation* nlp)
    : nlp_(nlp)
  {
    if(nlp->useWeightedInnerProd()) {
      vec_n_ = nlp_->alloc_primal_vec();      
    } else {
      vec_n_ = nullptr;
    }
  }
  
  virtual ~InnerProduct()
  {
    delete vec_n_;
  }

  // Computes ||x||_M
  double norm_M(const hiopVector& x)
  {
    if(nlp_->useWeightedInnerProd()) {      
      nlp_->eval_M(x, *vec_n_);
      auto dp = x.dotProductWith(*vec_n_);
      return ::std::sqrt(dp);
    } else {
      return x.twonorm();
    }
  }
  // Computes H primal norm
  double norm_H_primal(const hiopVector& x)
  {
    if(nlp_->useWeightedInnerProd()) {      
      nlp_->eval_H(x, *vec_n_);
      auto dp = x.dotProductWith(*vec_n_);
      return ::std::sqrt(dp);
    } else {
      return x.twonorm();
    }
  }
  // Computes H dual norm
  double norm_H_dual(const hiopVector& x)
  {
    if(nlp_->useWeightedInnerProd()) {      
      nlp_->eval_H_inv(x, *vec_n_);
      auto dp = x.dotProductWith(*vec_n_);
      return ::std::sqrt(dp);
    } else {
      return x.twonorm();
    }
  }

private:
  // Pointer to "client" NLP
  hiopNlpFormulation* nlp_;

  // Working vector in the size n of the variables, allocated only when for the weighted case
  hiopVector* vec_n_;
};

} //end namespace
#endif
