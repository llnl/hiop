// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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

/**
 * @file HessianDiagPlusRowRank.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#include "HessianDiagPlusRowRank.hpp"
#include "LinAlgFactory.hpp"
#include "hiopVectorPar.hpp"

#include "hiop_blasdefs.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#endif

#include <cassert>
#include <cstring>
#include <cmath>

#include <vector>
#include <algorithm>
using namespace std;

#define SIGMA_STRATEGY1 1
#define SIGMA_STRATEGY2 2
#define SIGMA_STRATEGY3 3
#define SIGMA_STRATEGY4 4
#define SIGMA_CONSTANT  5

namespace hiop
{

HessianDiagPlusRowRank::HessianDiagPlusRowRank(hiopNlpDenseConstraints* nlp_in, int max_mem_len)
  : l_max_(max_mem_len),
    l_curr_(-1),
    sigma_(1.),
    sigma0_(1.),
    nlp_(nlp_in),
    matrix_changed_(false)
{
  DhInv_ = nlp_->alloc_primal_vec();
  St_ = nlp_->alloc_multivector_primal(0, l_max_);
  Yt_ = St_->alloc_clone(); //faster than nlp_->alloc_multivector_primal(...);
  //these are local
  L_  = LinearAlgebraFactory::create_matrix_dense("DEFAULT", 0, 0);
  D_  = LinearAlgebraFactory::create_vector("DEFAULT", 0);
  V_  = LinearAlgebraFactory::create_matrix_dense("DEFAULT", 0, 0);

  //the previous iteration
  it_prev_ = new hiopIterate(nlp_);
  grad_f_prev_ = nlp_->alloc_primal_vec();
  Jac_c_prev_ = nlp_->alloc_Jac_c();
  Jac_d_prev_ = nlp_->alloc_Jac_d();

  //internal buffers for memory pool (none of them should be in n)
#ifdef HIOP_USE_MPI
  buff_kxk_    = new double[nlp_->m() * nlp_->m()];
  buff_2lxk_   = new double[nlp_->m() * 2*l_max_];
  buff1_lxlx3_ = new double[3*l_max_*l_max_];
  buff2_lxlx3_ = new double[3*l_max_*l_max_];
#else
   //not needed in non-MPI mode
  buff_kxk_  = nullptr;
  buff_2lxk_ = nullptr;
  buff1_lxlx3_ = nullptr;
  buff2_lxlx3_ = nullptr;
#endif

  //auxiliary objects/buffers
  S1_ = nullptr;
  Y1_ = nullptr;
  lxl_mat1_ = nullptr;
  kxl_mat1_ = nullptr;
  kx2l_mat1_ = nullptr;
  l_vec1_ = nullptr;
  l_vec2_ = nullptr;
  twol_vec1_ = nullptr;
  n_vec1_ = DhInv_->alloc_clone();
  n_vec2_ = DhInv_->alloc_clone();

  V_work_vec_ = LinearAlgebraFactory::create_vector("DEFAULT", 0);
  V_ipiv_vec_ = nullptr;
  V_ipiv_size_ = -1;
  
  sigma0_ = nlp_->options->GetNumeric("sigma0");
  sigma_ = sigma0_;

  string sigma_strategy = nlp_->options->GetString("sigma_update_strategy");
  transform(sigma_strategy.begin(), sigma_strategy.end(), sigma_strategy.begin(), ::tolower);
  sigma_update_strategy_ = SIGMA_STRATEGY3;
  if(sigma_strategy=="sty") {
    sigma_update_strategy_=SIGMA_STRATEGY1;
  } else if(sigma_strategy=="sty_inv") {
    sigma_update_strategy_=SIGMA_STRATEGY2;
  } else if(sigma_strategy=="snrm_ynrm") {
    sigma_update_strategy_=SIGMA_STRATEGY3;
  } else if(sigma_strategy=="sty_srnm_ynrm") {
    sigma_update_strategy_=SIGMA_STRATEGY4;
  } else if(sigma_strategy=="sigma0") {
    sigma_update_strategy_=SIGMA_CONSTANT;
  } else {
    assert(false && "sigma_update_strategy option not recognized");
  }

  sigma_safe_min_ = 1e-8;
  sigma_safe_max_ = 1e+8;
  nlp_->log->printf(hovScalars, "Hessian Low Rank: initial sigma is %g\n", sigma_);
  nlp_->log->printf(hovScalars,
                    "Hessian Low Rank: sigma update strategy is %d [%s]\n",
                    sigma_update_strategy_,
                    sigma_strategy.c_str());

  Dx_   = DhInv_->alloc_clone();
#ifdef HIOP_DEEPCHECKS
  Vmat_ = V_->alloc_clone();
#endif

  yk = nlp_->alloc_primal_vec();
  sk = nlp_->alloc_primal_vec();

}  

HessianDiagPlusRowRank::~HessianDiagPlusRowRank()
{
  delete DhInv_;
  delete Dx_;

  delete St_;
  delete Yt_;
  delete L_;
  delete D_;
  delete V_;
  delete yk;
  delete sk;
#ifdef HIOP_DEEPCHECKS
  delete Vmat_;
#endif


  delete it_prev_;
  delete grad_f_prev_;
  delete Jac_c_prev_;
  delete Jac_d_prev_;

  delete[] buff_kxk_;
  delete[] buff_2lxk_;
  delete[] buff1_lxlx3_;
  delete[] buff2_lxlx3_;

  delete S1_;
  delete Y1_;
  delete lxl_mat1_;
  delete kxl_mat1_; 
  delete kx2l_mat1_;

  delete l_vec1_;
  delete l_vec2_;
  delete n_vec1_;
  delete n_vec2_;
  delete twol_vec1_;
  delete[] V_ipiv_vec_;
  delete V_work_vec_;

  for(auto* it: a) {
    delete it;
  }

  for(auto* it: b) {
    delete it;
  }
}

void HessianDiagPlusRowRank::alloc_for_limited_mem(const size_type& mem_length)
{
  //note: St_ and Yt_ always have l_curr_ rows
  if(l_curr_ == mem_length) {
    assert(D_->get_size() == l_curr_);
    return;
  }
  delete D_;
  delete L_;
  delete Yt_;
  delete St_;
  St_ = nlp_->alloc_multivector_primal(mem_length, l_max_);
  Yt_ = St_->alloc_clone();

  //these are local
  L_  = LinearAlgebraFactory::create_matrix_dense("DEFAULT", mem_length, mem_length);
  D_  = LinearAlgebraFactory::create_vector("DEFAULT", mem_length);
}

bool HessianDiagPlusRowRank::update_logbar_diag(const hiopVector& Dx)
{
  DhInv_->setToConstant(sigma_);
  DhInv_->axpy(1.0,Dx);
  Dx_->copyFrom(Dx);
#ifdef HIOP_DEEPCHECKS
  assert(DhInv_->allPositive());
#endif
  DhInv_->invert();
  nlp_->log->write("HessianDiagPlusRowRank: inverse diag DhInv:", *DhInv_, hovMatrices);
  matrix_changed_ = true;
  return true;
}

#ifdef HIOP_DEEPCHECKS
void HessianDiagPlusRowRank::print(FILE* f, hiopOutVerbosity v, const char* msg) const
{
  fprintf(f, "%s\n", msg);
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("Dx", *Dx_, v);
#else
  fprintf(f, "Dx is not stored in this class, but it can be computed from Dx=DhInv^(1)-sigma");
#endif
  nlp_->log->printf(v, "sigma=%22.16f;\n", sigma_);
  nlp_->log->write("DhInv", *DhInv_, v);
  nlp_->log->write("S_trans", *St_, v);
  nlp_->log->write("Y_trans", *Yt_, v);

  fprintf(f, " [[Internal representation]]\n");
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("V", *Vmat_, v);
#else
  fprintf(f,
          "V matrix is available at this point (only its LAPACK factorization). Print it in "
          "updateInternalBFGSRepresentation() instead, before factorizeV()\n");
#endif
  nlp_->log->write("L", *L_, v);
  nlp_->log->write("D", *D_, v);
}
#endif

#include <limits>

bool HessianDiagPlusRowRank::update(const hiopIterate& it_curr,
                                    const hiopVector& grad_f_curr,
                                    const hiopMatrix& Jac_c_curr_in,
                                    const hiopMatrix& Jac_d_curr_in)
{
  nlp_->runStats.tmSolverInternal.start();

  const hiopMatrixDense& Jac_c_curr = dynamic_cast<const hiopMatrixDense&>(Jac_c_curr_in);
  const hiopMatrixDense& Jac_d_curr = dynamic_cast<const hiopMatrixDense&>(Jac_d_curr_in);

#ifdef HIOP_DEEPCHECKS
  assert(it_curr.zl->matchesPattern(nlp_->get_ixl()));
  assert(it_curr.zu->matchesPattern(nlp_->get_ixu()));
  assert(it_curr.sxl->matchesPattern(nlp_->get_ixl()));
  assert(it_curr.sxu->matchesPattern(nlp_->get_ixu()));
#endif
  //on first call l_curr_=-1
  if(l_curr_>=0) {
    size_type n = grad_f_curr.get_size();
    //compute s_new = x_curr-x_prev
    hiopVector& s_new = new_n_vec1(n);
    s_new.copyFrom(*it_curr.x);
    s_new.axpy(-1.,*it_prev_->x);
    double s_infnorm=s_new.infnorm();
    if(s_infnorm>=100*std::numeric_limits<double>::epsilon()) { //norm of s not too small

      //compute y_new = \grad J(x_curr,\lambda_curr) - \grad J(x_prev, \lambda_curr) (yes, J(x_prev, \lambda_curr))
      //              = graf_f_curr-grad_f_prev + (Jac_c_curr-Jac_c_prev)yc_curr+ (Jac_d_curr-Jac_c_prev)yd_curr - zl_curr*s_new + zu_curr*s_new
      hiopVector& y_new = new_n_vec2(n);
      y_new.copyFrom(grad_f_curr); 
      y_new.axpy(-1., *grad_f_prev_);
      Jac_c_curr.transTimesVec  (1.0, y_new, 1.0, *it_curr.yc);
      //!opt if nlp_->Jac_c_isLinear no need for the multiplications
      Jac_c_prev_->transTimesVec(1.0, y_new,-1.0, *it_curr.yc);
      //!opt same here
      Jac_d_curr.transTimesVec  (1.0, y_new, 1.0, *it_curr.yd); 
      Jac_d_prev_->transTimesVec(1.0, y_new,-1.0, *it_curr.yd);
      
      double sTy = s_new.dotProductWith(y_new), s_nrm2=s_new.twonorm(), y_nrm2=y_new.twonorm();

#ifdef HIOP_DEEPCHECKS
      nlp_->log->printf(hovLinAlgScalarsVerb,
                        "HessianDiagPlusRowRank: s^T*y=%20.14e ||s||=%20.14e ||y||=%20.14e\n",
                        sTy,
                        s_nrm2,
                        y_nrm2);
      nlp_->log->write("HessianDiagPlusRowRank s_new",s_new, hovIteration);
      nlp_->log->write("HessianDiagPlusRowRank y_new",y_new, hovIteration);
#endif
      if(sTy>s_nrm2*y_nrm2*sqrt(std::numeric_limits<double>::epsilon())) { //sTy far away from zero

        if(l_max_>0) {
          //compute the new row in L, update S and Y (either augment them or shift cols and add s_new and y_new)
          hiopVector& YTs = new_l_vec1(l_curr_);
          Yt_->timesVec(0.0, YTs, 1.0, s_new);
          //update representation
          if(l_curr_<l_max_) {
            //just grow/augment the matrices
            St_->appendRow(s_new);
            Yt_->appendRow(y_new);
            growL(l_curr_, l_max_, YTs);
            growD(l_curr_, l_max_, sTy);
            l_curr_++;
          } else {
            //shift
            St_->shiftRows(-1);
            Yt_->shiftRows(-1);
            St_->replaceRow(l_max_-1, s_new);
            Yt_->replaceRow(l_max_-1, y_new);
            updateL(YTs,sTy);
            updateD(sTy);
            l_curr_ = l_max_;
          }
        } //end of l_max_>0
#ifdef HIOP_DEEPCHECKS
        nlp_->log->printf(hovMatrices, "\nHessianDiagPlusRowRank: these are L and D from the BFGS compact representation\n");
        nlp_->log->write("L", *L_, hovMatrices);
        nlp_->log->write("D", *D_, hovMatrices);
        nlp_->log->printf(hovMatrices, "\n");
#endif
        //update B0 (i.e., sigma)
        switch (sigma_update_strategy_ ) {
        case SIGMA_STRATEGY1:
          sigma_ = sTy/(s_nrm2*s_nrm2);
          break;
        case SIGMA_STRATEGY2:
          sigma_ = y_nrm2*y_nrm2/sTy;
          break;
        case SIGMA_STRATEGY3:
          sigma_ = sqrt(s_nrm2*s_nrm2 / y_nrm2 / y_nrm2);
          break;
        case SIGMA_STRATEGY4:
          sigma_ = 0.5*(sTy/(s_nrm2*s_nrm2)+y_nrm2*y_nrm2/sTy);
          break;
        case SIGMA_CONSTANT:
          sigma_ = sigma0_;
          break;
        default:
          assert(false && "Option value for sigma_update_strategy was not recognized.");
          break;
        } // else of the switch
        //safe guard it
        sigma_ = fmax(fmin(sigma_safe_max_, sigma_), sigma_safe_min_);
        nlp_->log->printf(hovLinAlgScalars, "HessianDiagPlusRowRank: sigma was updated to %22.16e\n", sigma_);
      } else { //sTy is too small or negative -> skip
        nlp_->log->printf(hovLinAlgScalars,
                          "HessianDiagPlusRowRank: s^T*y=%12.6e not positive enough... skipping the Hessian update\n",
                          sTy);
      }
    } else {// norm of s_new is too small -> skip
      nlp_->log->printf(hovLinAlgScalars,
                        "HessianDiagPlusRowRank: ||s_new||=%12.6e too small... skipping the Hessian update\n",
                        s_infnorm);
    }
    //save this stuff for next update
    it_prev_->copyFrom(it_curr);
    grad_f_prev_->copyFrom(grad_f_curr);
    Jac_c_prev_->copyFrom(Jac_c_curr);
    Jac_d_prev_->copyFrom(Jac_d_curr);
    nlp_->log->printf(hovLinAlgScalarsVerb, "HessianDiagPlusRowRank: storing the iteration info as 'previous'\n", s_infnorm);
  } else {
    //this is the first optimization iterate, just save the iterate and exit
    it_prev_->copyFrom(it_curr);
    grad_f_prev_->copyFrom(grad_f_curr);
    Jac_c_prev_->copyFrom(Jac_c_curr);
    Jac_d_prev_->copyFrom(Jac_d_curr);

    nlp_->log->printf(hovLinAlgScalarsVerb, "HessianLowRank on first update, just saving iteration\n");

    l_curr_++;
  }
  nlp_->runStats.tmSolverInternal.stop();
  return true;
}

/* 
 * The dirty work to bring this^{-1} to the form
 * M = DhInv - DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 * Namely it computes V, a symmetric 2lx2l given by
 *  V =  [S'*B0*(DhInv*B0-I)*S    -L+S'*B0*DhInv*Y ]
 *       [-L'+Y'*Dhinv*B0*S       +D+Y'*Dhinv*Y    ]
 * In this function V is factorized and it will hold the factors at the end of the function
 * Note that L, D, S, and Y are from the BFGS secant representation and are updated/computed in 'update'
 */
void HessianDiagPlusRowRank::updateInternalBFGSRepresentation()
{
  size_type n=St_->n();
  size_type l=St_->m();

  //grow L,D, andV if needed
  if(L_->m()!=l) {
    delete L_;
    L_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", l, l);
  }
  if(D_->get_size()!=l) {
    delete D_;
    D_ = LinearAlgebraFactory::create_vector("DEFAULT", l);
  }
  if(V_->m()!=2*l) {
    delete V_;
    V_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", 2*l, 2*l);
  }

  //-- block (2,2)
  hiopMatrixDense& DpYtDhInvY = new_lxl_mat1(l);
  sym_mat_times_diag_times_mattrans_local(0.0, DpYtDhInvY, 1.0,*Yt_,*DhInv_);
#ifdef HIOP_USE_MPI
  const size_t buffsize=l*l*sizeof(double);
  memcpy(buff1_lxlx3_, DpYtDhInvY.local_data(), buffsize);
#else
  DpYtDhInvY.addDiagonal(1., *D_);
  V_->copyBlockFromMatrix(l,l,DpYtDhInvY);
#endif

  //-- block (1,2)
  hiopMatrixDense& StB0DhInvYmL = DpYtDhInvY; //just a rename
  hiopVector& B0DhInv = new_n_vec1(n);
  B0DhInv.copyFrom(*DhInv_);
  B0DhInv.scale(sigma_);
  mat_times_diag_times_mattrans_local(StB0DhInvYmL, *St_, B0DhInv, *Yt_);
#ifdef HIOP_USE_MPI
  memcpy(buff1_lxlx3_+l*l, StB0DhInvYmL.local_data(), buffsize);
#else
  //substract L
  StB0DhInvYmL.addMatrix(-1.0, *L_);
  // (1,2) block in V
  V_->copyBlockFromMatrix(0,l,StB0DhInvYmL);
#endif

  //-- block (2,2)
  hiopVector& theDiag = B0DhInv; //just a rename, also reuses values
  theDiag.addConstant(-1.0); //at this point theDiag=DhInv*B0-I
  theDiag.scale(sigma_);
  hiopMatrixDense& StDS = DpYtDhInvY; //a rename
  sym_mat_times_diag_times_mattrans_local(0.0, StDS, 1.0, *St_, theDiag);
#ifdef HIOP_USE_MPI
  memcpy(buff1_lxlx3_+2*l*l, DpYtDhInvY.local_data(), buffsize);
#else
  V_->copyBlockFromMatrix(0,0,StDS);
#endif


#ifdef HIOP_USE_MPI
  int ierr;
  ierr = MPI_Allreduce(buff1_lxlx3_, buff2_lxlx3_, 3*l*l, MPI_DOUBLE, MPI_SUM, nlp_->get_comm());
  assert(ierr==MPI_SUCCESS);

  // - block (2,2)
  DpYtDhInvY.copyFrom(buff2_lxlx3_);
  DpYtDhInvY.addDiagonal(1., *D_);
  V_->copyBlockFromMatrix(l, l, DpYtDhInvY);

  // - block (1,2)
  StB0DhInvYmL.copyFrom(buff2_lxlx3_+l*l);
  StB0DhInvYmL.addMatrix(-1.0, *L_);
  V_->copyBlockFromMatrix(0, l, StB0DhInvYmL);

  // - block (1,1)
  StDS.copyFrom(buff2_lxlx3_ + 2*l*l);
  V_->copyBlockFromMatrix(0, 0, StDS);
#endif
#ifdef HIOP_DEEPCHECKS
  delete Vmat_;
  Vmat_ = V_->new_copy();
  Vmat_->overwriteLowerTriangleWithUpper();
#endif

  //finally, factorize V
  factorizeV();

  matrix_changed_ = false;
}

/* Solves this*x = res as x = this^{-1}*res
 * where 'this^{-1}' is
 * M = DhInv - DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 *
 * M is is nxn, S,Y are nxl, V is upper triangular 2lx2l, and x is nx1
 * Remember we store Yt=Y^T and St=S^T
 */  
void HessianDiagPlusRowRank::solve(const hiopVector& rhsx, hiopVector& x)
{
  if(matrix_changed_) {
    updateInternalBFGSRepresentation();
  }

  size_type n=St_->n(), l=St_->m();
#ifdef HIOP_DEEPCHECKS
  assert(rhsx.get_size()==n);
  assert(x.get_size()==n);
  assert(DhInv_->get_size()==n);
  assert(DhInv_->isfinite_local() && "inf or nan entry detected");
  assert(rhsx.isfinite_local() && "inf or nan entry detected in rhs");
#endif

  //1. x = DhInv*res
  x.copyFrom(rhsx);
  x.componentMult(*DhInv_);

  //2. stx= S^T*B0*DhInv*res and ytx=Y^T*DhInv*res
  hiopVector& stx = new_l_vec1(l);
  hiopVector& ytx = new_l_vec2(l);
  stx.setToZero();
  ytx.setToZero();
  Yt_->timesVec(0.0, ytx, 1.0, x);

  hiopVector& B0DhInvx = new_n_vec1(n);
  B0DhInvx.copyFrom(x); //it contains DhInv*res
  B0DhInvx.scale(sigma_); //B0*(DhInv*res) 
  St_->timesVec(0.0, stx, 1.0, B0DhInvx);

  //3. solve with V
  hiopVector& spart=stx; hiopVector& ypart=ytx;
  solve_with_V(spart,ypart);

  //4. multiply with  DhInv*[B0*S Y], namely
  // result = DhInv*(B0*S*spart + Y*ypart)
  hiopVector&  result = new_n_vec1(n);
  St_->transTimesVec(0.0, result, 1.0, spart);
  result.scale(sigma_);
  Yt_->transTimesVec(1.0, result, 1.0, ypart);
  result.componentMult(*DhInv_);

  //5. x = first term - second term = x_computed_in_1 - result 
  x.axpy(-1.0,result);
#ifdef HIOP_DEEPCHECKS
  assert(x.isfinite_local() && "inf or nan entry detected in computed solution");
#endif
  
}

/* W = beta*W + alpha*X*inverse(this)*X^T (a more efficient version of solve)
 * where 'this^{-1}' is
 * M = DhInv + DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 * W is kxk, S,Y are nxl, DhInv,B0 are n, V is 2lx2l
 * X is kxn
 */ 
void HessianDiagPlusRowRank::sym_mat_times_inverse_times_mattrans(double beta,
                                                                  hiopMatrixDense& W, 
                                                                  double alpha,
                                                                  const hiopMatrixDense& X)
{
  if(matrix_changed_) {
    updateInternalBFGSRepresentation();
  }

  size_type n=St_->n(), l=St_->m();
  size_type k=W.m(); 
  assert(X.m()==k);
  assert(X.n()==n);

#ifdef HIOP_DEEPCHECKS
   nlp_->log->write("sym_mat_times_inverse_times_mattrans: X is: ", X, hovMatrices);
#endif 

  //1. compute W=beta*W + alpha*X*DhInv*X'
#ifdef HIOP_USE_MPI
  if(0==nlp_->get_rank()) {
    sym_mat_times_diag_times_mattrans_local(beta,W,alpha,X,*DhInv_);
  } else {
    sym_mat_times_diag_times_mattrans_local(0.0, W,alpha,X,*DhInv_);
  }
  //W will be MPI_All_reduced later
#else
  sym_mat_times_diag_times_mattrans_local(beta,W,alpha,X,*DhInv_);
#endif
  //2. compute S1=X*DhInv*B0*S and Y1=X*DhInv*Y
  auto& S1 = new_S1(X, *St_);
  auto& Y1 = new_Y1(X, *Yt_); //both are kxl
  hiopVector& B0DhInv = new_n_vec1(n);
  B0DhInv.copyFrom(*DhInv_);
  B0DhInv.scale(sigma_);
  mat_times_diag_times_mattrans_local(S1, X, B0DhInv, *St_);
  mat_times_diag_times_mattrans_local(Y1, X, *DhInv_,  *Yt_);

  //3. reduce W, S1, and Y1 (dimensions: kxk, kxl, kxl)
  hiopMatrixDense& S2Y2 = new_kx2l_mat1(k,l);  //Initialy S2Y2 = [Y1 S1]
  S2Y2.copyBlockFromMatrix(0,0,S1);
  S2Y2.copyBlockFromMatrix(0,l,Y1);
#ifdef HIOP_USE_MPI
  int ierr;
  ierr = MPI_Allreduce(S2Y2.local_data(), buff_2lxk_, 2*l*k, MPI_DOUBLE, MPI_SUM, nlp_->get_comm());
  assert(ierr==MPI_SUCCESS);
  ierr = MPI_Allreduce(W.local_data(),    buff_kxk_,  k*k,   MPI_DOUBLE, MPI_SUM, nlp_->get_comm());
  assert(ierr==MPI_SUCCESS);
  S2Y2.copyFrom(buff_2lxk_);
  W.copyFrom(buff_kxk_);
  //also copy S1 and Y1
  S1.copyFromMatrixBlock(S2Y2, 0,0);
  Y1.copyFromMatrixBlock(S2Y2, 0,l);
#endif
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("sym_mat_times_inverse_times_mattrans: W first term is: ", W, hovMatrices);
#endif 
  //4. [S2] = V \ [S1^T]
  //   [Y2]       [Y1^T]
  //S2Y2 is exactly [S1^T] when Fortran Lapack looks at it
  //                [Y1^T]
  hiopMatrixDense& RHS_fortran = S2Y2; 
  solve_with_V(RHS_fortran);

  //5. W = W-alpha*[S1 Y1]*[S2^T] 
  //                       [Y2^T]
  S2Y2 = RHS_fortran;
  alpha = 0-alpha;
  hiopMatrixDense& S2=new_kxl_mat1(k,l);
  S2.copyFromMatrixBlock(S2Y2, 0, 0);
  S1.timesMatTrans_local(1.0, W, alpha, S2);

  hiopMatrixDense& Y2=S2;
  Y2.copyFromMatrixBlock(S2Y2, 0, l);
  Y1.timesMatTrans_local(1.0, W, alpha, Y2);

  //nlp_->log->write("sym_mat_times_inverse_times_mattrans: Y1 is : ", Y1, hovMatrices);
  //nlp_->log->write("sym_mat_times_inverse_times_mattrans: Y2 is : ", Y2, hovMatrices);
  //nlp_->log->write("sym_mat_times_inverse_times_mattrans: W is : ", W, hovMatrices);
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("sym_mat_times_inverse_times_mattrans: final matrix is : ", W, hovMatrices);
#endif 
}

void HessianDiagPlusRowRank::factorizeV()
{
  int N = V_->n();
  int lda = N;
  int info;
  if(N==0) {
    return;
  }

#ifdef HIOP_DEEPCHECKS
    nlp_->log->write("factorizeV:  V is ", *V_, hovMatrices);
#endif

  char uplo='L'; //V is upper in C++ so it's lower in fortran

  if(V_ipiv_vec_==nullptr) {
    V_ipiv_vec_ = new int[N];
  }
  else {
    if(V_ipiv_size_!=N) {
      delete[] V_ipiv_vec_;
      V_ipiv_vec_ = new int[N];
      V_ipiv_size_ = N;
    }
  }

  int lwork=-1;//inquire sizes
  double Vwork_tmp;
  DSYTRF(&uplo, &N, V_->local_data(), &lda, V_ipiv_vec_, &Vwork_tmp, &lwork, &info);
  assert(info==0);

  lwork=(int)Vwork_tmp;
  if(lwork != V_work_vec_->get_size()) {
    if(V_work_vec_!=nullptr) {
      delete V_work_vec_;
    }
    V_work_vec_ = LinearAlgebraFactory::create_vector("DEFAULT", lwork);
  } else assert(V_work_vec_);

  DSYTRF(&uplo, &N, V_->local_data(), &lda, V_ipiv_vec_, V_work_vec_->local_data(), &lwork, &info);
  
  if(info<0) {
    nlp_->log->printf(hovError, "HessianDiagPlusRowRank::factorizeV error: %d arg to dsytrf has an illegal value\n", -info);
  } else if(info>0) {
    nlp_->log->printf(hovError,
                     "HessianDiagPlusRowRank::factorizeV error: %d entry in the factorization's diagonal is exactly zero. "
                     "Division by zero will occur if a solve is attempted.\n",
                     info);
  }
  assert(info==0);
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("factorizeV:  factors of V: ", *V_, hovMatrices);
#endif

}

void HessianDiagPlusRowRank::solve_with_V(hiopVector& rhs_s, hiopVector& rhs_y)
{
  int N = V_->n();
  if(N==0) {
    return;
  }

  int l = rhs_s.get_size();

#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("HessianDiagPlusRowRank::solve_with_V: RHS IN 's' part: ", rhs_s, hovMatrices);
  nlp_->log->write("HessianDiagPlusRowRank::solve_with_V: RHS IN 'y' part: ", rhs_y, hovMatrices);
  hiopVector* rhs_saved= LinearAlgebraFactory::create_vector("DEFAULT", rhs_s.get_size()+rhs_y.get_size());
  rhs_saved->copyFromStarting(0, rhs_s);
  rhs_saved->copyFromStarting(l, rhs_y);
#endif

  int lda=N, one=1, info;
  char uplo='L'; 
#ifdef HIOP_DEEPCHECKS
  assert(N==rhs_s.get_size()+rhs_y.get_size());
#endif
  hiopVector& rhs=new_2l_vec1(l);
  rhs.copyFromStarting(0, rhs_s);
  rhs.copyFromStarting(l, rhs_y);

  DSYTRS(&uplo, &N, &one, V_->local_data(), &lda, V_ipiv_vec_, rhs.local_data(), &N, &info);

  if(info<0) {
    nlp_->log->printf(hovError, "HessianDiagPlusRowRank::solve_with_V error: %d arg to dsytrf has an illegal value\n", -info);
  }
  assert(info==0);

  //copy back the solution
  rhs.copyToStarting(0,rhs_s);
  rhs.copyToStarting(l,rhs_y);

#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("solve_with_V: SOL OUT 's' part: ", rhs_s, hovMatrices);
  nlp_->log->write("solve_with_V: SOL OUT 'y' part: ", rhs_y, hovMatrices);

  //residual calculation
  double nrmrhs=rhs_saved->infnorm();
  Vmat_->timesVec(1.0, *rhs_saved, -1.0, rhs);
  double nrmres=rhs_saved->infnorm();
  //nlp_->log->printf(hovLinAlgScalars, "HessianDiagPlusRowRank::solve_with_V 1rhs: rel resid norm=%g\n", nrmres/(1+nrmrhs));
  nlp_->log->printf(hovScalars, "HessianDiagPlusRowRank::solve_with_V 1rhs: rel resid norm=%g\n", nrmres/(1+nrmrhs));
  if(nrmres>1e-8) {
    nlp_->log->printf(hovWarning, "HessianDiagPlusRowRank::solve_with_V large residual=%g\n", nrmres);
  }
  delete rhs_saved;
#endif

}

void HessianDiagPlusRowRank::solve_with_V(hiopMatrixDense& rhs)
{
  int N = V_->n();
  if(0==N) {
    return;
  }

#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("solve_with_V: RHS IN: ", rhs, hovMatrices);
  hiopMatrixDense* rhs_saved = rhs.new_copy();
#endif

  //rhs is transpose in C++

  char uplo='L'; 
  int lda=N, ldb=N, nrhs=rhs.m(), info;
#ifdef HIOP_DEEPCHECKS
  assert(N==rhs.n()); 
#endif
  DSYTRS(&uplo, &N, &nrhs, V_->local_data(), &lda, V_ipiv_vec_, rhs.local_data(), &ldb, &info);

  if(info<0) {
    nlp_->log->printf(hovError, "HessianDiagPlusRowRank::solve_with_V error: %d arg to dsytrf has an illegal value\n", -info);
  }
  assert(info==0);
#ifdef HIOP_DEEPCHECKS
  nlp_->log->write("solve_with_V: SOL OUT: ", rhs, hovMatrices);
  
  hiopMatrixDense& sol = rhs; //matrix of solutions
  /// TODO: get rid of these uses of specific hiopVector implementation
  hiopVector* x = LinearAlgebraFactory::create_vector("DEFAULT", rhs.n()); //again, keep in mind rhs is transposed
  hiopVector* r = LinearAlgebraFactory::create_vector("DEFAULT", rhs.n());

  double resnorm=0.0;
  for(int k=0; k<rhs.m(); k++) {
    rhs_saved->getRow(k, *r);
    sol.getRow(k,*x);
    double nrmrhs = r->infnorm();//nrmrhs=.0;
    Vmat_->timesVec(1.0, *r, -1.0, *x);
    double nrmres = r->infnorm();
    if(nrmres>1e-8) {
      nlp_->log->printf(hovWarning,
                        "HessianDiagPlusRowRank::solve_with_V mult-rhs: rhs number %d has large resid norm=%g\n",
                        k,
                        nrmres);
    }
    if(nrmres/(nrmrhs+1)>resnorm) {
      resnorm=nrmres/(nrmrhs+1);
    }
  }
  nlp_->log->printf(hovLinAlgScalars, "HessianDiagPlusRowRank::solve_with_V mult-rhs: rel resid norm=%g\n", resnorm);
  delete x;
  delete r;
  delete rhs_saved;
#endif

}

void HessianDiagPlusRowRank::growL(const int& lmem_curr, const int& lmem_max, const hiopVector& YTs)
{
  int l = L_->m();
#ifdef HIOP_DEEPCHECKS
  assert(l==L_->n());
  assert(lmem_curr==l);
  assert(lmem_max>=l);
#endif
  //newL = [   L     0]
  //       [ Y^T*s   0]
  hiopMatrixDense* newL = LinearAlgebraFactory::create_matrix_dense("DEFAULT", l+1, l+1);
  assert(newL);
  //copy from L to newL
  newL->copyBlockFromMatrix(0,0, *L_);

  double* newL_mat = newL->local_data(); //doing the rest here
  const double* YTs_vec = YTs.local_data_const();
  //for(int j=0; j<l; j++) newL_mat[l][j] = YTs_vec[j];
  for(int j=0; j<l; j++) {
    newL_mat[l*(l+1)+j] = YTs_vec[j];
  }

  //and the zero entries of the last column
  //for(int i=0; i<l+1; i++) newL_mat[i][l] = 0.0;
  for(int i=0; i<l+1; i++) {
    newL_mat[i*(l+1)+l] = 0.0;
  }

  //swap the pointers
  delete L_;
  L_ = newL;
}

void HessianDiagPlusRowRank::growD(const int& lmem_curr, const int& lmem_max, const double& sTy)
{
  int l = D_->get_size();
  assert(l==lmem_curr);
  assert(lmem_max>=l);

  hiopVector* Dnew = LinearAlgebraFactory::create_vector("DEFAULT", l+1);
  double* Dnew_vec = Dnew->local_data();
  memcpy(Dnew_vec, D_->local_data_const(), l*sizeof(double));
  Dnew_vec[l] = sTy;

  delete D_;
  D_ = Dnew;
}

/* L_{ij} = s_{i-1}^T y_{j-1}, if i>j, otherwise zero. Here i,j = 0,1,...,l_curr-1
 * L_new = lift and shift L to the left; replace last row with [Yts;0]
 */
void HessianDiagPlusRowRank::updateL(const hiopVector& YTs, const double& sTy)
{
  int l=YTs.get_size();
  assert(l==L_->m());
  assert(l==L_->n());
#ifdef HIOP_DEEPCHECKS
  assert(l_curr_==l);
  assert(l_curr_==l_max_);
#endif
  const int lm1=l-1;
  double* L_mat=L_->local_data();
  const double* yts_vec=YTs.local_data_const();
  for(int i=1; i<lm1; i++) {
    for(int j=0; j<i; j++) {
      //L_mat[i][j] = L_mat[i+1][j+1];
      L_mat[i*l+j] = L_mat[(i+1)*l+j+1];
    }
  }
      

  //is this really needed?
  //for(int i=0; i<lm1; i++)
  //  L_mat[i][lm1]=0.0;

  //first entry in YTs corresponds to y_to_be_discarded_since_it_is_the_oldest'* s_new and is discarded
  for(int j=0; j<lm1; j++) {
    //L_mat[lm1][j]=yts_vec[j+1];
    L_mat[lm1*l+j] = yts_vec[j+1];
    
  }

  //L_mat[lm1][lm1]=0.0;
  L_mat[lm1*l+lm1] = 0.0;
}
void HessianDiagPlusRowRank::updateD(const double& sTy)
{
  int l=D_->get_size();
  double* D_vec = D_->local_data();
  for(int i=0; i<l-1; i++) {
    D_vec[i] = D_vec[i+1];
  }
  D_vec[l-1] = sTy;
}

hiopVector&  HessianDiagPlusRowRank::new_l_vec1(int l)
{
  if(l_vec1_!=nullptr && l_vec1_->get_size()==l) {
    return *l_vec1_;
  }
  if(l_vec1_!=nullptr) {
    delete l_vec1_;
  }
  l_vec1_= LinearAlgebraFactory::create_vector("DEFAULT", l);
  return *l_vec1_;
}

hiopVector&  HessianDiagPlusRowRank::new_l_vec2(int l)
{
  if(l_vec2_!=nullptr && l_vec2_->get_size()==l) {
    return *l_vec2_;
  }
  if(l_vec2_!=nullptr) {
    delete l_vec2_;
  }
  l_vec2_= LinearAlgebraFactory::create_vector("DEFAULT", l);
  return *l_vec2_;
}

hiopMatrixDense& HessianDiagPlusRowRank::new_lxl_mat1(int l)
{
  if(lxl_mat1_!=nullptr) {
    if(l==lxl_mat1_->m()) {
      return *lxl_mat1_;
    } else {
      delete lxl_mat1_; 
      lxl_mat1_=nullptr;
    }
  }
  lxl_mat1_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", l, l);
  return *lxl_mat1_;
}

hiopMatrixDense& HessianDiagPlusRowRank::new_kx2l_mat1(int k, int l)
{
  const int twol=2*l;
  if(nullptr!=kx2l_mat1_) {
    assert(kx2l_mat1_->m()==k);
    if(twol==kx2l_mat1_->n()) {
      return *kx2l_mat1_;
    } else {
      delete kx2l_mat1_; 
      kx2l_mat1_=nullptr;
    }
  }
  kx2l_mat1_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", k, twol);
  return *kx2l_mat1_;
}

hiopMatrixDense& HessianDiagPlusRowRank::new_kxl_mat1(int k, int l)
{
  if(kxl_mat1_!=nullptr) {
    assert(kxl_mat1_->m()==k);
    if( l==kxl_mat1_->n() ) {
      return *kxl_mat1_;
    } else {
      delete kxl_mat1_; 
      kxl_mat1_=nullptr;
    }
  }
  kxl_mat1_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", k, l);
  return *kxl_mat1_;
}

hiopMatrixDense& HessianDiagPlusRowRank::new_S1(const hiopMatrixDense& X, const hiopMatrixDense& St)
{
  //S1 is X*some_diag*S  (kxl). Here St=S^T is lxn and X is kxn (l BFGS memory size, k number of constraints)
  size_type k = X.m();
  size_type l = St.m();
#ifdef HIOP_DEEPCHECKS
  assert(St.n()==X.n());
  if(S1_!=nullptr) { 
    assert(S1_->m()==k);
  }
#endif
  if(nullptr!=S1_ && S1_->n()!=l) {
    delete S1_;
    S1_=nullptr;
  }
  if(nullptr==S1_) {
    S1_=LinearAlgebraFactory::create_matrix_dense("DEFAULT", k, l);
  }
  return *S1_;
}

hiopMatrixDense& HessianDiagPlusRowRank::new_Y1(const hiopMatrixDense& X, const hiopMatrixDense& Yt)
{
  //Y1 is X*somediag*Y (kxl). Here Yt=Y^T is lxn,  X is kxn
  size_type k = X.m();
  size_type l = Yt.m();
#ifdef HIOP_DEEPCHECKS
  assert(X.n()==Yt.n());
  if(Y1_!=nullptr) {
    assert(Y1_->m()==k);
  }
#endif
  if(nullptr!=Y1_ && Y1_->n()!=l) {
    delete Y1_;
    Y1_ = nullptr;
  }
  if(nullptr==Y1_) {
    Y1_ = LinearAlgebraFactory::create_matrix_dense("DEFAULT", k, l);
  }
  return *Y1_;
}
#ifdef HIOP_DEEPCHECKS

void HessianDiagPlusRowRank::times_vec_no_logbar_term(double beta, hiopVector& y, double alpha, const hiopVector&x)
{
  this->times_vec_common(beta, y, alpha, x, false);
}

#endif //HIOP_DEEPCHECKS


void HessianDiagPlusRowRank::
times_vec_common(double beta, hiopVector& y, double alpha, const hiopVector& x, bool addLogTerm) const
{
  size_type n=St_->n();
  assert(l_curr_==St_->m());
  assert(y.get_size()==n);
  assert(St_->get_local_size_n() == Yt_->get_local_size_n());

  //we have B+=B-B*s*B*s'/(s'*B*s)+yy'/(y'*s)
  //B0 is sigma*I. There is an additional diagonal log-barrier term Dx_

  bool print=false;
  if(print) {
    nlp_->log->printf(hovMatrices, "---HessianDiagPlusRowRank::times_vec \n");
    nlp_->log->write("S=", *St_, hovMatrices);
    nlp_->log->write("Y=", *Yt_, hovMatrices);
    nlp_->log->write("DhInv=", *DhInv_, hovMatrices);
    nlp_->log->printf(hovMatrices, "sigma=%22.16e;  addLogTerm=%d;\n", sigma_, addLogTerm);
    if(addLogTerm) {
      nlp_->log->write("Dx=", *Dx_, hovMatrices);
    }
    nlp_->log->printf(hovMatrices, "y=beta*y + alpha*this*x : beta=%g alpha=%g\n", beta, alpha);
    nlp_->log->write("x_in=", x, hovMatrices);
    nlp_->log->write("y_in=", y, hovMatrices);
  }

  //allocate and compute a_k and b_k
  //! make sure the pointers within these std::vectors are deallocated
  a.resize(l_curr_, nullptr);
  b.resize(l_curr_, nullptr);
  int n_local = Yt_->get_local_size_n();
  for(int k=0; k<l_curr_; k++) {
    //bk=yk/sqrt(yk'*sk)
    yk->copyFrom(Yt_->local_data() + k*n_local);
    sk->copyFrom(St_->local_data() + k*n_local);
    double skTyk=yk->dotProductWith(*sk);
    
    if(skTyk < std::numeric_limits<double>::epsilon()) {
      nlp_->log->printf(hovLinAlgScalars,
                        "HessianDiagPlusRowRank: ||s_k^T*y_k||=%12.6e too small and was set it to mach eps = %12.6e \n",
                        skTyk,
                        std::numeric_limits<double>::epsilon());
      skTyk = std::numeric_limits<double>::epsilon();
    }

    if(a[k] == nullptr && b[k] == nullptr) {
      b[k] = nlp_->alloc_primal_vec();
      a[k] = nlp_->alloc_primal_vec();
    }
    
    b[k]->copyFrom(*yk);
    b[k]->scale(1/sqrt(skTyk));

    //compute ak by an inner loop
    a[k]->copyFrom(*sk);
    a[k]->scale(sigma_);

    for(int i=0; i<k; i++) {
      double biTsk = b[i]->dotProductWith(*sk);
      a[k]->axpy(+biTsk, *b[i]);
      double aiTsk = a[i]->dotProductWith(*sk);
      a[k]->axpy(-aiTsk, *a[i]);
    }
    double skTak = a[k]->dotProductWith(*sk);
    a[k]->scale(1/sqrt(skTak));
  }

  //now we have B= B_0 + sum{ bk bk' - ak ak' : k=0,1,...,l_curr-1} 
  //compute the product with x
  //y = beta*y+alpha*(B0+Dx)*x + alpha* sum { bk'x bk - ak'x ak : k=0,1,...,l_curr-1}
  y.scale(beta);
  if(addLogTerm) 
    y.axzpy(alpha, x, *Dx_);

  y.axpy(alpha*sigma_, x); 

  for(int k=0; k<l_curr_; k++) {
    double bkTx = b[k]->dotProductWith(x);
    double akTx = a[k]->dotProductWith(x);
    
    y.axpy( alpha*bkTx, *b[k]);
    y.axpy(-alpha*akTx, *a[k]);
  }

  if(print) {
    nlp_->log->write("y_out=", y, hovMatrices);
  }

}

void HessianDiagPlusRowRank::times_vec(double beta, hiopVector& y, double alpha, const hiopVector&x)
{
  this->times_vec_common(beta, y, alpha, x);
}

void HessianDiagPlusRowRank::timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x) const
{
  this->times_vec_common(beta, y, alpha, x);
}

/**************************************************************************
 * Internal helpers
 *************************************************************************/

/* symmetric multiplication W = beta*W + alpha*X*Diag*X^T 
 * W is kxk local, X is kxn distributed and Diag is n, distributed
 * The ops are perform locally. The reduce is done separately/externally to decrease comm
 */
void HessianDiagPlusRowRank::sym_mat_times_diag_times_mattrans_local(double beta,
                                                                     hiopMatrixDense& W,
                                                                     double alpha,
                                                                     const hiopMatrixDense& X,
                                                                     const hiopVector& d)
{
  size_type k=W.m();
  size_type n_local=X.get_local_size_n();

  assert(X.m()==k);
    
#ifdef HIOP_DEEPCHECKS
  assert(W.n()==k);
  assert(d.get_size()==X.n());
  assert(d.get_local_size()==n_local);
#endif
  
  //#define chunk 512; //!opt
  const double *xi, *xj;
  double acc;
  double *Wdata=W.local_data();
  const double *Xdata=X.local_data_const();
  const double* dd=d.local_data_const();
  for(int i=0; i<k; i++) {
    //xi=Xdata[i];
    xi=Xdata+i*n_local;
    for(int j=i; j<k; j++) {
      //xj=Xdata[j];
      xj=Xdata+j*n_local;
      //compute W[i,j] = sum {X[i,p]*d[p]*X[j,p] : p=1,...,n_local}
      acc=0.0;
      for(size_type p=0; p<n_local; p++)
	acc += xi[p]*dd[p]*xj[p];

      //Wdata[i][j]=Wdata[j][i]=beta*Wdata[i][j]+alpha*acc;
      Wdata[i*k+j] = Wdata[j*k+i] = beta*Wdata[i*k+j]+alpha*acc;
    }
  }
}

/* W=S*D*X^T, where S is lxn, D is diag nxn, and X is kxn */
void HessianDiagPlusRowRank::mat_times_diag_times_mattrans_local(hiopMatrixDense& W,
                                                                 const hiopMatrixDense& S,
                                                                 const hiopVector& d,
                                                                 const hiopMatrixDense& X)
{
#ifdef HIOP_DEEPCHECKS
  assert(S.n()==d.get_size());
  assert(S.n()==X.n());
#endif  
  int l=S.m(), n=d.get_local_size(), k=X.m();
  assert(X.get_local_size_n() == d.get_local_size());
  
  const double* Sdi;
  double* Wdi;
  const double* Xdj;
  double acc;
  double* Wd=W.local_data();
  const double* Sd=S.local_data_const();
  const double* Xd=X.local_data_const();
  const double *diag=d.local_data_const();
  //!opt
  for(int i=0;i<l; i++) {
    //Sdi=Sd[i]; Wdi=Wd[i];
    Sdi = Sd+i*n;
    Wdi = Wd+i*W.get_local_size_n();
    
    for(int j=0; j<k; j++) {
      //Xdj=Xd[j];
      Xdj = Xd+j*n;
      acc=0.;
      for(int p=0; p<n; p++) 
	//acc += Sdi[p]*diag[p]*Xdj[p];
        acc += Sdi[p]*diag[p]*Xdj[p];
      
      Wdi[j]=acc;
    }
  }
}
};
