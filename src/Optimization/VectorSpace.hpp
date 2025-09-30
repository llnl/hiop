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
 * @file VectorSpace.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_NLP_VECSPACE
#define HIOP_NLP_VECSPACE

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
 * Currently, this class uses lumps the H matrix in a diagonal matrix and uses as the weight for
 * norms and inner products over the underlying (primal) vector space. For the dual space, it uses
 * the inverse of this diagonal matrix. Using the H and matrices directly complicates the linear 
 * algebra of the quasi-Newton IPM solver, since it uses direct solves while M and H comes as
 * mat-vec applies. 
 *
 * This class also covers Euclidean (i.e., non-weighted) inner products, for which M=H=I.
 */
class VectorSpace
{
public:
  VectorSpace(hiopNlpFormulation* nlp);
  
  virtual ~VectorSpace();

  // Apply lumped mass matrix, y=M_lumped*x
  bool apply_M_lumped(const hiopVector& x, hiopVector& y) const;
  
  // Computes H primal norm, ||x||_H
  double norm_H_primal(const hiopVector& x) const;
  
  // Computes H dual norm, ||y||_{H^{-1}}
  double norm_H_dual(const hiopVector& y) const;

  // Computes norm of stationarity residual, using inf-norm for Euclidean spaces, otherwise uses inf-norm
  // of rescaled dual representer
  double norm_stationarity(const hiopVector& x) const;

  // Computes norm of complementarity, using inf-norm 
  double norm_complementarity(const hiopVector& x) const;
  
  // Computes 1M norm, i.e., ||x|| =  1^T*M*|x|
  double norm_M_one(const hiopVector& x) const;

  // Computes the "volume" of the space, norm of 1 function
  double volume() const;

  // Return vector containing the diagonals of the lumped mass matrix, possibly creating the internal object
  const hiopVector* M_lumped() const;

  /**
   * Compute linear damping terms required by the weighted log barrier subproblem 
   * 
   * Essentially computes kappa_d*mu* \sum { this[i] | ixl[i]==1 and ixr[i]==0 }
   */
  double linear_damping_term_local(const hiopVector& s,
                                   const hiopVector& ixl,
                                   const hiopVector& ixr,
                                   const double& mu,
                                   const double& kappa_d) const;
  /**
   * @brief Add linear damping term
   * Performs `this[i] = alpha*this[i] + sign[i]*ct*M_lumped[i]` where sign[i]=1 when EXACTLY one of
   * ixleft[i] and ixright[i] is 1.0 and sign=0 otherwise.
   */
  void add_linear_damping_term(const hiopVector& ixl, const hiopVector& ixu, const double& ct, hiopVector& x) const;

  // Computes weighted (by the lumped matrix) sum of log of selected (by `ix`) entries of `x`
  double log_barrier_eval_local(const hiopVector& x, const hiopVector& ix) const;

  // Adds (to `gradx`) the gradient of the weighted log, namely gradx = gradx - mu * M_lumped * ix/s
  void log_barrier_grad_add(const double& mu, const hiopVector& s, const hiopVector& ix, hiopVector& gradx) const;
private:
  // Pointer to "client" NLP
  hiopNlpFormulation* nlp_;

  // Working vector in the size n of the variables, allocated only when for the weighted case
  mutable hiopVector* vec_n_;
  // Working vector in the size n of the variables, allocated only when for the weighted case
  mutable hiopVector* vec_n2_;
  // Lumped mass matrix
  mutable hiopVector* M_lump_;
};

} //end namespace
#endif
