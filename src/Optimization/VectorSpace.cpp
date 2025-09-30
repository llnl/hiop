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
 * @file VectorSpace.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#include "VectorSpace.hpp"
#include "hiopNlpFormulation.hpp"

namespace hiop
{

VectorSpace::VectorSpace(hiopNlpFormulation* nlp)
  : nlp_(nlp)
{
  assert(nlp);
  M_lump_ = nullptr;
  if(nlp->get_weighted_space_type()) {
    vec_n_ = nlp_->alloc_primal_vec();
    vec_n2_ = nlp_->alloc_primal_vec();
  } else {
    vec_n_ = nullptr;
    vec_n2_ = nullptr;
  }
}
  
VectorSpace::~VectorSpace()
{
  delete vec_n2_;
  delete vec_n_;
  delete M_lump_;
}

// Applies lumped mass matrix;
bool VectorSpace::apply_M_lumped(const hiopVector& x, hiopVector& y) const
{
  y.copyFrom(x);
  y.componentMult(*M_lumped());
  return true;
}

// Computes H primal norm
double VectorSpace::norm_H_primal(const hiopVector& x) const
{
  if(hiopInterfaceBase::Hilbert==nlp_->get_weighted_space_type()) {
    nlp_->eval_H(x, *vec_n_);
    auto dp = x.dotProductWith(*vec_n_);
    return ::std::sqrt(dp);
  } if(hiopInterfaceBase::HilbertLumped==nlp_->get_weighted_space_type()) {
    apply_M_lumped(x, *vec_n_);
    auto dp = x.dotProductWith(*vec_n_);
    return ::std::sqrt(dp);
  } else {
    assert(hiopInterfaceBase::Euclidean==nlp_->get_weighted_space_type());
    return x.twonorm();
  }
}
// Computes H dual norm
double VectorSpace::norm_H_dual(const hiopVector& x) const
{
  if(hiopInterfaceBase::Hilbert==nlp_->get_weighted_space_type()) {      
    nlp_->eval_H_inv(x, *vec_n_);
    auto dp = x.dotProductWith(*vec_n_);
    return ::std::sqrt(dp);
  } else {
    if(hiopInterfaceBase::HilbertLumped==nlp_->get_weighted_space_type()) {
      vec_n_->copyFrom(x);
      vec_n_->componentDiv(*M_lumped());
      auto dp = x.dotProductWith(*vec_n_);                                                                                               return ::std::sqrt(dp); 
    } else {
      assert(hiopInterfaceBase::Euclidean==nlp_->get_weighted_space_type());
      return x.twonorm();
    }
  }
}

double VectorSpace::norm_stationarity(const hiopVector& x) const
{
  if(nlp_->get_weighted_space_type()) {
    vec_n_->copyFrom(x);
    vec_n_->componentDiv(*M_lumped());
    return vec_n_->infnorm();
  } else {
    return x.infnorm();
  }
}

// Compute norm one weighted by M, i.e., 1^T*M*|x|
double VectorSpace::norm_M_one(const hiopVector& x) const
{
  if(nlp_->get_weighted_space_type()) {
    //use vec_n2_ since vec_n_ may be changed in M_lumped_
    vec_n2_->copyFrom(x);
    vec_n2_->component_abs();
    //M_lumped_ is already M*1
    return M_lumped()->dotProductWith(*vec_n2_);
  } else {
    return x.onenorm();
  }
}

double VectorSpace::norm_complementarity(const hiopVector& x) const
{
  if(nlp_->get_weighted_space_type()) {
    // since both x (slacks) and the bound duals are in the same (primal) space, inf norm "is
    // mesh independent".
    return x.infnorm();
  } else {
    return x.infnorm();
  }
}

// Computes the "volume" of the space, 1^T M*1 
double VectorSpace::volume() const
{
  if(nlp_->get_weighted_space_type()) {
    double vol_total = nlp_->m_ineq_low() + nlp_->m_ineq_upp();
    if(nlp_->n_low() > 0 || nlp_->n_upp() > 0) {
      //compute ||1||_M
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

// Return vector with the (diagonals of the) lumped mass matrix, possibly creating the internal object
const hiopVector* VectorSpace::M_lumped() const
{
  if(M_lump_ == nullptr) {
    M_lump_ = nlp_->alloc_primal_vec();
    if(nlp_->get_weighted_space_type()) {    
      vec_n_->setToConstant(1.);
      nlp_->eval_M(*vec_n_, *M_lump_);
    } else {
      M_lump_->setToConstant(1.);
    }
  }
  return M_lump_;
}

void VectorSpace::
add_linear_damping_term(const hiopVector& ixl, const hiopVector& ixu, const double& ct, hiopVector& x) const
{
  if(nlp_->get_weighted_space_type()) {
    vec_n_->copyFrom(ixl);
    vec_n_->axpy(-1.0, ixu);
    vec_n_->componentMult(*M_lumped());
    x.axpy(ct, *vec_n_);
  } else {
    x.addLinearDampingTerm(ixl, ixu, 1.0, ct);
  }
}

double VectorSpace::log_barrier_eval_local(const hiopVector& x, const hiopVector& ix) const
{
  if(nlp_->get_weighted_space_type()) {
    return x.logBarrierWeighted_local(ix, *M_lumped());
  } else {
    return x.logBarrier_local(ix);
  }
}

// Adds (to `gradx`) the gradient of the weighted log, namely gradx = gradx - mu * M_lumped * ix/s
void VectorSpace::log_barrier_grad_add(const double& mu, const hiopVector& s, const hiopVector& ix, hiopVector& gradx) const
{
  if(nlp_->get_weighted_space_type()) {
    vec_n_->copyFrom(ix);
    vec_n_->componentDiv(s);
    vec_n_->componentMult(*M_lumped());
    gradx.axpy(mu, *vec_n_);
  } else {
    gradx.addLogBarrierGrad(mu, s, ix);
  }
}

double VectorSpace::linear_damping_term_local(const hiopVector& s,
                                              const hiopVector& ixl,
                                              const hiopVector& ixr,
                                              const double& mu,
                                              const double& kappa_d) const
{
  if(nlp_->get_weighted_space_type()) {
    vec_n_->copyFrom(s);
    vec_n_->componentMult(*M_lumped());
    return vec_n_->linearDampingTerm_local(ixl, ixr, mu, kappa_d);
  } else {
    return s.linearDampingTerm_local(ixl, ixr, mu, kappa_d);
  }
  
}

} // end namespace
