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

/**
 * @file HessianDiagPlusRowRank.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_HESSIANLOWRANK
#define HIOP_HESSIANLOWRANK

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"

#include <cassert>

namespace hiop
{

/** 
 * Encapsultes storage and solves with the Hessian being identity plus low-rank term. 
 *
 * Stores the Hessian (w.r.t. x) approximation as Hk=Dk+Bk, where 
 *  - Dk is the log barrier diagonal
 *  - Bk = B0 - M*N^{-1}*M^T is the secant approximation for the Hessian of the Lagrangian
 * in the limited-memory compact representation of  Byrd, Nocedal, and Schnabel (1994)
 * Reference: Byrd, Nocedal, and Schnabel, "Representations of quasi-Newton matrices and
 * and there use in limited memory methods", Math. Programming 63 (1994), p. 129-156.
 * 
 * M=[B0*Sk Yk] is nx2l, and N is 2lx2l, where n=dim(x) and l is the length of the memory of secant 
 * approximation. This class is for when n>>k and l=O(10).
 * 
 * This class provides functionality to KKT linear system class for updating the secant approximation 
 * and solving with Hk=Dk+Bk.
 * 
 * Solving with Hk is performed by 
 *  1. computing the inverse as
 * Hk{-1} = (Dk+B0)^{-1} - (Dk+B0)^{-1}*[B0*Sk Yk]*( -N + [Sk'*B0]*(Dk+B0)^{-1}*[B0*Sk Yk] )^{-1} *[Sk'*B0]*(Dk+B0)^{-1}
 *                                                 (      [Yk'   ]                         )       [Yk'   ]
 *  2. multiplying with the above expression. The inner 2lx2l inverse matrix is not explicitly computed; instead 
 * V=(-N + [Sk'*B0]*(Dk+B0)^{-1}*[B0*Sk Yk] ) is stored, factorized, and solved with.
 *   (     [Yk'   ]              [Yk'     ] ) 
 *
 * Notation used in the implementation provided by this class
 *  - DhInv:=(Dk+B0)^{-1}, thus Hk{-1}=DhInv-DhInv*U*V^{-1}*U'*DhInv with
 *  - U:=[B0*St' Yt'] and  V is defined above
 *  - 
 *  
 * Parallel computations: Dk, B0 are distributed vectors, M is distributed 
 * column-wise, and N is local (stored on all processors).
 */
class HessianDiagPlusRowRank : public hiopMatrix
{
public:
  HessianDiagPlusRowRank(hiopNlpDenseConstraints* nlp_in, int max_memory_length);
  virtual ~HessianDiagPlusRowRank();

  /// Updates Hessian if hereditary positive definitness is maintained and returns true, otherwise false.
  virtual bool update(const hiopIterate& x_curr,
                      const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_curr,
                      const hiopMatrix& Jac_d_curr);

  /* updates the logBar diagonal term from the representation */
  virtual bool update_logbar_diag(const hiopVector& Dx);

  /* solves this*x=res */
  virtual void solve(const hiopVector& rhs, hiopVector& x);
  
  /* W = beta*W + alpha*X*inverse(this)*X^T (a more efficient version of solve)
   * This is performed as W = beta*W + alpha*X*(this\X^T)
   */ 
  virtual void sym_mat_times_inverse_times_mattrans(double beta, hiopMatrixDense& W, double alpha, const hiopMatrixDense& X);
#ifdef HIOP_DEEPCHECKS
  /* same as above but without the Dx term in H */
  virtual void times_vec_no_logbar_term(double beta, hiopVector& y, double alpha, const hiopVector&x);
  virtual void print(FILE* f, hiopOutVerbosity v, const char* msg) const;
#endif

  /* Computes the product of the Hessian with a vector: y=beta*y+alpha*H*x.
   * The function is supposed to use the underlying ***recursive*** definition of the 
   * quasi-Newton Hessian and is used for checking/testing/error calculation.
   */
  virtual void times_vec(double beta, hiopVector& y, double alpha, const hiopVector&x);

  /* code shared by the above two methods*/
  virtual void times_vec_common(double beta, hiopVector& y, double alpha, const hiopVector&x, bool add_logbar = false) const;

protected:
  friend class hiopAlgFilterIPMQuasiNewton;  
  int l_max_; //max memory size
  int l_curr_; //number of pairs currently stored
  double sigma_; //initial scaling factor of identity
  double sigma0_; //default scaling factor of identity

  //Integer for the sigma update strategy
  int sigma_update_strategy_;
  //Min safety thresholds for sigma
  double sigma_safe_min_;
  //Max safety thresholds for sigma
  double sigma_safe_max_;
  //Pointer to the NLP formulation
  hiopNlpDenseConstraints* nlp_;
  
  mutable std::vector<hiopVector*> a;
  mutable std::vector<hiopVector*> b;
  hiopVector* yk;
  hiopVector* sk;
private:
  // Vector for (B0+Dk)^{-1}
  hiopVector* DhInv_; 
  // Dx_ is needed in times_vec (for residual checking in solveCompressed). Can be recomputed from DhInv, but I decided to
  //store it instead to avoid round-off errors
  hiopVector* Dx_;
  
  bool matrix_changed_;

  //These are matrices from the compact representation; they are updated at each iteration.
  //More exactly Bk=B0-[B0*St' Yt']*[St*B0*St'  L]*[St*B0]
  //                                [  L'      -D] [Yt   ]
  //Transpose of S and T are store to easily access columns
  hiopMatrixDense* St_;
  hiopMatrixDense* Yt_;

  /// Lower triangular matrix from the compact representation
  hiopMatrixDense* L_;
  /// Diagonal matrix from the compact representation
  hiopVector* D_; 
  // Matrix V from the representation of the inverse
  hiopMatrixDense* V_;    
#ifdef HIOP_DEEPCHECKS
  //copy of the V matrix - needed to check the residual
   hiopMatrixDense* Vmat_; 
#endif
  void growL(const int& lmem_curr, const int& lmem_max, const hiopVector& YTs);
  void growD(const int& l_curr, const int& l_max, const double& sTy);
  void updateL(const hiopVector& STy, const double& sTy);
  void updateD(const double& sTy);
  //also stored are the iterate, gradient obj, and Jacobians at the previous optimization iteration
  hiopIterate *it_prev_;
  hiopVector *grad_f_prev_;
  hiopMatrixDense *Jac_c_prev_;
  hiopMatrixDense *Jac_d_prev_;

  //internal helpers
  void updateInternalBFGSRepresentation();

  //internals buffers, mostly for MPIAll_reduce
  double* buff_kxk_; // size = num_constraints^2 
  double* buff_2lxk_; // size = 2 x q-Newton mem size x num_constraints
  double* buff1_lxlx3_;
  double* buff2_lxlx3_;
  
  // auxiliary objects preallocated and used in internally in various computation blocks

  /// See new_S1
  hiopMatrixDense* S1_;
  /// See new_Y1
  hiopMatrixDense* Y1_;
  
  hiopMatrixDense* lxl_mat1_;
  hiopMatrixDense* kx2l_mat1_;
  hiopMatrixDense* kxl_mat1_;
  
  /**
   * (Re)Allocates S1_ of size kxl to store is X*D*S, where D is a diagonal matrix. S comes in 
   * as St=S^T (lxn) and X comes in as kxn, where l is the BFGS memory size and k number of 
   * constraints. S1_ is allocated only if not already allocated or realocated only if it does 
   * not have the right dimesions to store X*D*S.
   */
  hiopMatrixDense& new_S1(const hiopMatrixDense& X, const hiopMatrixDense& St);

  /**
   * (Re)Allocates Y1_ of size kxl to store is X*D*Y, where D is a diagonal matrix. Y comes in 
   * as Yt=Y^T (lxn) and X comes in as kxn, where l is the BFGS memory size and k number of 
   * constraints. Y1_ is allocated only if not already allocated or reallocated only if it does 
   * not have the right dimesions to store X*D*Y.
   */
  hiopMatrixDense& new_Y1(const hiopMatrixDense& X, const hiopMatrixDense& Yt);
  
  hiopMatrixDense& new_lxl_mat1 (int l);
  hiopMatrixDense& new_kxl_mat1 (int k, int l);
  hiopMatrixDense& new_kx2l_mat1(int k, int l);
  
  hiopVector* l_vec1_;
  hiopVector* l_vec2_;
  hiopVector* n_vec1_;
  hiopVector* n_vec2_;
  hiopVector* twol_vec1_;
  hiopVector& new_l_vec1(int l);
  hiopVector& new_l_vec2(int l);
  inline hiopVector& new_n_vec1(size_type n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(n_vec1_!=nullptr);
    assert(n_vec1_->get_size()==n);
#endif
    return *n_vec1_;
  }
  inline hiopVector& new_n_vec2(size_type n)
  {
#ifdef HIOP_DEEPCHECKS
    assert(n_vec2_!=nullptr);
    assert(n_vec2_->get_size()==n);
#endif
    return *n_vec2_;
  }
  inline hiopVector& new_2l_vec1(int l)
  {
    if(twol_vec1_!=nullptr && twol_vec1_->get_size()==2*l) {
      return *twol_vec1_;
    }
    if(twol_vec1_!=nullptr)
    {
      delete twol_vec1_;
    }
    twol_vec1_=LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), 2*l);
    return *twol_vec1_;
  }
private:
  //utilities
  
  /// @brief Ensures the internal containers are ready to work with "limited memory" mem_length  
  void alloc_for_limited_mem(const size_type& mem_length);

  /* symmetric multiplication W = beta*W + alpha*X*Diag*X^T */
  static void sym_mat_times_diag_times_mattrans_local(double beta,
                                                      hiopMatrixDense& W_,
                                                      double alpha,
                                                      const hiopMatrixDense& X_,
                                                      const hiopVector& d);
  /* W=S*Diag*X^T */
  static void mat_times_diag_times_mattrans_local(hiopMatrixDense& W,
                                                  const hiopMatrixDense& S, 
                                                  const hiopVector& d,
                                                  const hiopMatrixDense& X);
  /* members and utilities related to V matrix: factorization and solve */
  hiopVector* V_work_vec_;
  int V_ipiv_size_;
  int* V_ipiv_vec_;
  
  void factorizeV();
  void solve_with_V(hiopVector& rhs_s, hiopVector& rhs_y);
  void solve_with_V(hiopMatrixDense& rhs);
private:
  HessianDiagPlusRowRank() {};
  HessianDiagPlusRowRank(const HessianDiagPlusRowRank&) {};
  HessianDiagPlusRowRank& operator=(const HessianDiagPlusRowRank&) {return *this;};

  /* methods that need to be implemented as the class inherits from hiopMatrix*/
public:
  virtual hiopMatrix* alloc_clone() const
  {
    assert(false && "not provided because it is not needed");
    return nullptr;
  }
  virtual hiopMatrix* new_copy() const
  {
    assert(false && "not provided because it is not needed");
    return nullptr;
  }

  virtual void setToZero()
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void setToConstant(double c)
  {
    assert(false && "not provided because it is not needed");
  }

  void timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x) const;

  /** y = beta * y + alpha * this^T * x */
  virtual void transTimesVec(double beta, hiopVector& y, double alpha,  const hiopVector& x ) const
  {
    assert(false && "not provided because it is not needed");
  }

  /* W = beta*W + alpha*this*X */  
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  /* W = beta*W + alpha*this^T*X */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  /* W = beta*W + alpha*this*X^T */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addDiagonal(const double& alpha, const hiopVector& d_)
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addDiagonal(const double& value)
  {
    assert(false && "not provided because it is not needed");
  }
  virtual void addSubDiagonal(const double& alpha, index_type start, const hiopVector& d_)
  {
    assert(false && "not provided because it is not needed");
  }
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag,
                              const double& alpha, 
			      const hiopVector& d_,
                              int start_on_src_vec,
                              int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c) 
  {
    assert(false && "not needed / implemented");
  }
  
  /* this += alpha*X */
  virtual void addMatrix(double alpah, const hiopMatrix& X)
  {
    assert(false && "not provided because it is not needed");
  }

  void addToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  void transAddToSymDenseMatrixUpperTriangle(int row_start, int col_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const
  {
    assert(false && "not needed; should not be used");
  }
  virtual double max_abs_value()
  {
    assert(false && "not provided because it is not needed");
    return 0.;
  }

  virtual void row_max_abs_value(hiopVector &ret_vec)
  {
    assert(false && "not provided because it is not needed");
  }
  
  virtual void scale_row(hiopVector &vec_scal, const bool inv_scale)
  {
    assert(false && "not provided because it is not needed");
  }

  void copyRowsFrom(const hiopMatrix& src_in, const index_type* rows_idxs, size_type n_rows)
  {
    assert(false && "not needed / should not be used");
  }
  
  /* return false is any of the entry is a nan, inf, or denormalized */
  virtual bool isfinite() const
  {
    assert(false && "not provided because it is not needed");
    return false;
  }
  
  /* call with -1 to print all rows, all columns, or on all ranks; otherwise will
  *  will print the first rows and/or columns on the specified rank.
  * 
  * If the underlying matrix is sparse, maxCols is ignored and a max number elements 
  * given by the value of 'maxRows' will be printed. If this value is negative, all
  * elements will be printed.
  */
  virtual void print(FILE* f=nullptr, const char* msg=nullptr, int maxRows=-1, int maxCols=-1, int rank=-1) const
  {
    assert(false && "not provided because it is not needed");
  }

  /* number of rows */
  virtual size_type m() const
  {
    assert(false && "method is not provided because it is not needed");
    return 0;
  }
  /* number of columns */
  virtual size_type n() const
  {
    assert(false && "method is not provided because it is not needed");
    return 0;
  }
#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  virtual bool assertSymmetry(double tol=1e-16) const { return true; }
#endif
};

} //~namespace
#endif
