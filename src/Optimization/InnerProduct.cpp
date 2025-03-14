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
 *
 */


#include "InnerProduct.hpp"
#include "hiopNlpFormulation.hpp"

namespace hiop
{

InnerProduct::InnerProduct(hiopNlpFormulation* nlp)
  : nlp_(nlp)
{
  printf("InnerProduct::InnerProduct begin\n"); fflush(stdout);
  assert(nlp);
  M_lump_ = nullptr;
  if(nlp->useWeightedInnerProd()) {
    vec_n_ = nlp_->alloc_primal_vec();
    vec_n2_ = nlp_->alloc_primal_vec();
  } else {
    vec_n_ = nullptr;
    vec_n2_ = nullptr;
  }

  printf("InnerProduct::InnerProduct end\n"); fflush(stdout);
}
  
InnerProduct::~InnerProduct()
{
  delete vec_n2_;
  delete vec_n_;
  delete M_lump_;
}

bool InnerProduct::apply_M(const hiopVector& x, hiopVector& y) const
{
  if(nlp_->useWeightedInnerProd()) {
    return nlp_->eval_M(x, y);
  } else {
    y.copyFrom(x);
    return true;
  }
}
  
// Computes ||x||_M
double InnerProduct::norm_M(const hiopVector& x) const
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
double InnerProduct::norm_H_primal(const hiopVector& x) const
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
double InnerProduct::norm_H_dual(const hiopVector& x) const
{
  if(nlp_->useWeightedInnerProd()) {      
    nlp_->eval_H_inv(x, *vec_n_);
    auto dp = x.dotProductWith(*vec_n_);
    return ::std::sqrt(dp);
  } else {
    return x.twonorm();
  }
}

double InnerProduct::norm_stationarity(const hiopVector& x) const
{
  if(nlp_->useWeightedInnerProd()) {
    nlp_->eval_H_inv(x, *vec_n_);
    auto dp = x.dotProductWith(*vec_n_);
    return ::std::sqrt(dp);
  } else {
    return x.infnorm();
  }
}

// Compute norm one weighted by M, i.e., 1^T*M*|x|
double InnerProduct::norm_M_one(const hiopVector&x) const
{
  if(nlp_->useWeightedInnerProd()) {
    //opt! pre-compute M*1
    vec_n_->copyFrom(x);
    vec_n_->component_abs();
    nlp_->eval_M(*vec_n_, *vec_n2_);
    vec_n_->setToConstant(1.);
    return vec_n_->dotProductWith(*vec_n2_);
  } else {
    return x.onenorm();
  }
}

double InnerProduct::norm_complementarity(const hiopVector& x) const
{
  if(nlp_->useWeightedInnerProd()) {
    // //opt! pre-compute M*1
    // vec_n_->copyFrom(x);
    // vec_n_->component_abs();
    // nlp_->eval_M(*vec_n_, *vec_n2_);
    // vec_n_->setToConstant(1.);
    // return vec_n_->dotProductWith(*vec_n2_);
    return x.infnorm();
  } else {
    return x.infnorm();
  }
}

// Computes the "volume" of the space, 1^T M*1 
double InnerProduct::volume() const
{
  if(nlp_->useWeightedInnerProd()) {
    double vol_total = nlp_->m_ineq_low() + nlp_->m_ineq_upp();
    if(nlp_->n_low() > 0 || nlp_->n_upp() > 0) {
      //compute ||1||_M
      //vec_n_->setToConstant(1.);      
      const double vol_mult_bnds = M_lumped()->onenorm();
      if(nlp_->n_low() > 0) {
        //For weighted Hilbert spaces we assume that if lower bounds are present, they are for all vars
        vol_total += vol_mult_bnds;
      }
      if(nlp_->n_upp() > 0) {
        //For weighted Hilbert spaces we assume that if lower bounds are present, they are for all vars
        vol_total += vol_mult_bnds;
      }      
    }
    return vol_total;
  } else {
    return nlp_->n_complem();
  }
}

// Return vector containing the diagonals of the lumped mass matrix, possibly creating the internal object
const hiopVector* InnerProduct::M_lumped() const
{
  if(M_lump_ == nullptr) {
    M_lump_ = nlp_->alloc_primal_vec();
    if(nlp_->useWeightedInnerProd()) {    
      vec_n_->setToConstant(1.);
      apply_M(*vec_n_, *M_lump_);
    } else {
      M_lump_->setToConstant(1.);
    }
  }
  return M_lump_;
}

void InnerProduct::
add_linear_damping_term(const hiopVector& ixl, const hiopVector& ixu, const double& ct, hiopVector& x) const
{
  if(nlp_->useWeightedInnerProd()) {
    vec_n_->copyFrom(ixl);
    vec_n_->axpy(-1.0, ixu);
    vec_n_->componentMult(*M_lumped());
    x.axpy(ct, *vec_n_);
  } else {
    x.addLinearDampingTerm(ixl, ixu, 1.0, ct);
  }
}

  
} // end namespace
