// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read ~SAdditional BSD Notice~T below.
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
//

/**
 * @file hiopFactAcceptor.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 * @author Cosmin G. Petra <petra@lnnl.gov>, LLNL
 *
 */

#ifndef HIOP_FACT_ACCEPTOR
#define HIOP_FACT_ACCEPTOR

#include "hiopNlpFormulation.hpp"
#include "hiopPDPerturbation.hpp"

namespace hiop
{

class hiopFactAcceptor
{
public:
  /**
   * Default constructor
   * Determine if a factorization is acceptable or not
   */
  hiopFactAcceptor(hiopPDPerturbation* p)
      : perturb_calc_{p}
  {}

  virtual ~hiopFactAcceptor() {}

  /**
   * @brief method to check if current factorization is acceptable or/and if
   * a re-factorization is reqired by increasing 'delta_wx'-'delta_cd'.
   *
   * Returns '1' if current factorization is rejected
   * Returns '0' if current factorization is ok
   * Returns '-1' if current factorization failed due to singularity
   */
  virtual int requireReFactorization(const hiopNlpFormulation& nlp, const int& n_neg_eig, const bool force_reg = false) = 0;

protected:
  hiopPDPerturbation* perturb_calc_;
};

class hiopFactAcceptorIC : public hiopFactAcceptor
{
public:
  /**
   * Default constructor
   * Check inertia condition to determine if a factorization is acceptable or not
   */
  hiopFactAcceptorIC(hiopPDPerturbation* p, const size_type n_required_neg_eig)
      : hiopFactAcceptor(p),
        n_required_neg_eig_(n_required_neg_eig)
  {}

  virtual ~hiopFactAcceptorIC() {}

  virtual int requireReFactorization(const hiopNlpFormulation& nlp, const int& n_neg_eig, const bool force_reg = false);

protected:
  int n_required_neg_eig_;
};

class hiopFactAcceptorInertiaFreeDWD : public hiopFactAcceptor
{
public:
  /**
   * Default constructor
   * Check inertia condition to determine if a factorization is acceptable or not
   */
  hiopFactAcceptorInertiaFreeDWD(hiopPDPerturbation* p, const size_type n_required_neg_eig)
      : hiopFactAcceptor(p),
        n_required_neg_eig_(n_required_neg_eig)
  {}

  virtual ~hiopFactAcceptorInertiaFreeDWD() {}

  virtual int requireReFactorization(const hiopNlpFormulation& nlp, const int& n_neg_eig, const bool force_reg = false);

protected:
  int n_required_neg_eig_;
};

}  // namespace hiop
#endif
