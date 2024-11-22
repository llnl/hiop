// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read “Additional BSD Notice” below.
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

#ifndef HIOP_RESIDUAL
#define HIOP_RESIDUAL

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"
#include "hiopIterate.hpp"

#include "hiopLogBarProblem.hpp"

namespace hiop
{

class hiopResidual
{
public:
  hiopResidual(hiopNlpFormulation* nlp);
  virtual ~hiopResidual();

  virtual int update(const hiopIterate& it,
                     const double& f,
                     const hiopVector& c,
                     const hiopVector& d,
                     const hiopVector& gradf,
                     const hiopMatrix& jac_c,
                     const hiopMatrix& jac_d,
                     const hiopLogBarProblem& logbar);

  ///
  /// @pre
  /**
   * @brief Evaluate the infeasibilities at the new second-order-correction iterate.
   *
   * Evaluate the the infeasibilities at the new second-order-correction iterate. Save
   * The method modifies 'this', in particular ryd,ryc, rxl,rxu, rdl, rdu in an attempt
   * to reuse storage/buffers, and it updates the cached `nrmXXX` members.
   *
   * @pre unlike method `update`, here vector `c_soc` and `d_soc` is the constraint
   *      violation including the constraint right-hand-side/slack, i.e.,
   *      c_soc = c_rhs - c_body and d_soc = d - d_body
   *
   * computed in c_eval and d_eval, respectively.
   * The method modifies 'this', in particular ryd,ryc, rxl,rxu, rdl, rdu in an attemp
   */
  virtual void update_soc(const hiopIterate& it,
                          const hiopVector& c_soc,
                          const hiopVector& d_soc,
                          const hiopVector& grad,
                          const hiopMatrix& jac_c,
                          const hiopMatrix& jac_d,
                          const hiopLogBarProblem& logprob);

  /* Return the Nlp and Log-bar errors computed at the previous update call. */
  inline void getNlpErrors(double& optim, double& feas, double& comple, double& cons_violation) const
  {
    optim = nrmInf_nlp_optim;
    feas = nrmInf_nlp_feasib;
    comple = nrmInf_nlp_complem;
    cons_violation = nrmInf_cons_violation;
  };
  inline void getBarrierErrors(double& optim, double& feas, double& comple) const
  {
    optim = nrmInf_bar_optim;
    feas = nrmInf_bar_feasib;
    comple = nrmInf_bar_complem;
  };
  /* get the previously computed Infeasibility */
  inline double getInfeasInfNorm() const { return nrmInf_nlp_feasib; }
  /* get the previously computed Infeasibility */
  inline double get_theta() const { return nrmOne_nlp_feasib; }
  /* evaluate the Infeasibility at the new iterate, which has eq and ineq functions
   * computed in c_eval and d_eval, respectively.
   * The method modifies 'this', in particular ryd,ryc, rxl,rxu, rdl, rdu in an attempt
   * to reuse storage/buffers, but does not update the cached nrmInf_XXX members.
   * It computes and returns the one norm of [ryc ryd] */
  double compute_nlp_infeasib_onenorm(const hiopIterate& iter, const hiopVector& c_eval, const hiopVector& d_eval);

  /* residual printing function - calls hiopVector::print
   * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* msg = NULL, int max_elems = -1, int rank = -1) const;

  /* accessors */
  inline hiopVector* get_rx() const { return rx; }
  inline hiopVector* get_rd() const { return rd; }
  inline hiopVector* get_rxl() const { return rxl; }
  inline hiopVector* get_rxu() const { return rxu; }
  inline hiopVector* get_rdl() const { return rdl; }
  inline hiopVector* get_rdu() const { return rdu; }
  inline hiopVector* get_ryc() const { return ryc; }
  inline hiopVector* get_ryd() const { return ryd; }
  inline hiopVector* get_rszl() const { return rszl; }
  inline hiopVector* get_rszu() const { return rszu; }
  inline hiopVector* get_rsvl() const { return rsvl; }
  inline hiopVector* get_rsvu() const { return rsvu; }

  inline double get_nrmInf_nlp_optim() const { return nrmInf_nlp_optim; }
  inline double get_nrmInf_nlp_feasib() const { return nrmInf_nlp_feasib; }
  inline double get_nrmInf_nlp_complem() const { return nrmInf_nlp_complem; }
  inline double get_nrmInf_bar_optim() const { return nrmInf_bar_optim; }
  inline double get_nrmInf_bar_feasib() const { return nrmInf_bar_feasib; }
  inline double get_nrmInf_bar_complem() const { return nrmInf_bar_complem; }
  inline double get_nrmOne_nlp_feasib() const { return nrmOne_nlp_feasib; }
  inline double get_nrmOne_bar_feasib() const { return nrmOne_bar_feasib; }
  inline double get_nrmOne_nlp_optim() const { return nrmOne_nlp_optim; }
  inline double get_nrmOne_bar_optim() const { return nrmOne_bar_optim; }
  inline hiopNlpFormulation* get_nlp() const { return nlp; }

  void copyFrom(const hiopResidual& resid_src);

private:
  hiopVector* rx;         // -\grad f - J_c^t y_c - J_d^t y_d + z_l - z_u
  hiopVector* rd;         //  y_d + v_l - v_u
  hiopVector *rxl, *rxu;  //  x - sxl-xl, -x-sxu+xu
  hiopVector *rdl, *rdu;  //  as above but for d

  hiopVector* ryc;  // -c(x)   (c(x)=0!//!)
  hiopVector* ryd;  // for d- d(x)

  hiopVector *rszl, *rszu;  // \mu e-sxl zl, \mu e - sxu zu
  hiopVector *rsvl, *rsvu;  // \mu e-sdl vl, \mu e - sdu vu

  /** storage for the norm of [rx,rd], [rxl,...,rdu,ryc,ryd], and [rszl,...,rsvu]
   *  for the nlp (\mu=0)
   */
  double nrmInf_nlp_optim, nrmInf_nlp_feasib, nrmInf_nlp_complem;
  /** storage for the norm of [rx,rd], [rxl,...,rdu,ryc,ryd], and [rszl,...,rsvu]
   *  for the barrier subproblem
   */
  double nrmInf_bar_optim, nrmInf_bar_feasib, nrmInf_bar_complem;

  /** storage for the one norm of [ryc,ryd]. This is the one norm of constraint violations.
   */
  double nrmOne_nlp_feasib;
  double nrmOne_bar_feasib;
  double nrmOne_nlp_optim;
  double nrmOne_bar_optim;

  /** inf norm of constraint violation */
  double nrmInf_cons_violation;

  // and associated info from problem formulation
  hiopNlpFormulation* nlp;

private:
  hiopResidual() {};
  hiopResidual(const hiopResidual&) {};
  hiopResidual& operator=(const hiopResidual& o) { return *this; };
  friend class hiopKKTLinSysFull;
  friend class hiopKKTLinSysCompressed;
  friend class hiopKKTLinSysCompressedXYcYd;
  friend class hiopKKTLinSysCompressedXDYcYd;
  friend class hiopKKTLinSysNormalEquation;
  friend class hiopKKTLinSysLowRank;
  friend class hiopKKTLinSys;
  friend class hiopVectorCompoundPD;
};

}  // namespace hiop
#endif
