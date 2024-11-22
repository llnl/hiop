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

/* implements the Krylov iterative solver
 * @file hiopKrylovSolver.hpp
 * @ingroup LinearSolvers
 * @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LLNL
 * @author Cosmin G. Petra <petra1@lnnl.gov>, LLNL
 */

#ifndef HIOP_KrylovSolver
#define HIOP_KrylovSolver

#include "hiopMatrix.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopVector.hpp"
#include "hiopVectorCompoundPD.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"

#include "hiop_blasdefs.hpp"

#include "hiopCppStdUtils.hpp"

#include "hiopLinearOperator.hpp"
#include "hiopNlpFormulation.hpp"

namespace hiop
{

/**
 * Base class for Krylov Solvers used by HiOp
 */

class hiopKrylovSolver
{
public:
  hiopKrylovSolver(int n,
                   hiopLinearOperator* A_opr,
                   hiopLinearOperator* Mleft_opr = nullptr,
                   hiopLinearOperator* Mright_opr = nullptr,
                   const hiopVector* x0 = nullptr);
  virtual ~hiopKrylovSolver();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).
   */
  virtual bool solve(hiopIterate* xsol, const hiopResidual* bresid) = 0;

  /// Set the iterate to a constant value
  virtual void set_x0(double xval);

  /// Set the maximun number of iteration
  inline virtual void set_max_num_iter(int num_iter) { maxit_ = num_iter; }

  /**
   * Set Krylov solver tolerance relative to norm of the right-hand side, meaning
   * that the solver will stop when two-norm of the residual is less than the tolerance
   * times two-norm of the right-hand side.
   */
  inline virtual void set_tol(double tol) { tol_ = tol; }

  /// Return the absolute residual at the end of Krylov solver
  inline virtual double get_sol_abs_resid() { return abs_resid_; }

  /// Return the relative residual at the end of Krylov solver
  inline virtual double get_sol_rel_resid() { return rel_resid_; }

  /// Return the number of iterations at the end of Krylov solver
  inline virtual double get_sol_num_iter() { return iter_; }

  /// Return the message about the convergence
  inline virtual std::string get_convergence_info() { return ss_info_.str(); }

  /**
   * Convergence flag: 0 for success, the other codes depending on the Krylov method
   * used. Concrete message about the convergence can be obtained from
   * get_convergence_info.
   */
  inline virtual int get_convergence_flag() { return flag_; }

protected:
  double tol_;                 // convergence tolerence
  size_type maxit_;            // maximun number of iteratiions
  double iter_;                // number of iterations at convergence
  int flag_;                   // convergence flag
  double abs_resid_;           // absolute residual
  double rel_resid_;           // relative residual
  const size_type n_;          // size of the rhs
  std::stringstream ss_info_;  // message about the convergence

  /// Memory space
  std::string mem_space_;

  /// Linear operator to apply the linear system matrix to a residual/vector
  hiopLinearOperator* A_opr_;

  /// Left preconditioner
  hiopLinearOperator* ML_opr_;

  /// Right preconditioners
  hiopLinearOperator* MR_opr_;

  /// Vector used to save the initial value
  hiopVector* x0_;
  hiopVectorCompoundPD* b_;
};

/**
 * a Krylov solver class implementing the PCG framework
 */
class hiopPCGSolver : public hiopKrylovSolver
{
public:
  /** initialization constructor */
  hiopPCGSolver(int n,
                hiopLinearOperator* A_opr,
                hiopLinearOperator* Mleft_opr = nullptr,
                hiopLinearOperator* Mright_opr = nullptr,
                const hiopVector* x0 = nullptr);
  virtual ~hiopPCGSolver();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).
   */
  virtual bool solve(hiopIterate* xsol, const hiopResidual* bresid);
  virtual bool solve(hiopVector* b);

protected:
  hiopVector* xmin_;
  hiopVector* res_;
  hiopVector* yk_;
  hiopVector* zk_;
  hiopVector* pk_;
  hiopVector* qk_;
};

/**
 * a Krylov solver class implementing the BiCGStab framework
 */
class hiopBiCGStabSolver : public hiopKrylovSolver
{
public:
  /** initialization constructor */
  hiopBiCGStabSolver(int n,
                     hiopLinearOperator* A_opr,
                     hiopLinearOperator* Mleft_opr = nullptr,
                     hiopLinearOperator* Mright_opr = nullptr,
                     const hiopVector* x0 = nullptr);
  virtual ~hiopBiCGStabSolver();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).
   */
  virtual bool solve(hiopIterate* xsol, const hiopResidual* bresid);
  virtual bool solve(hiopVector* b);

protected:
  hiopVector* xmin_;
  hiopVector* res_;
  hiopVector* pk_;
  hiopVector* ph_;
  hiopVector* v_;
  hiopVector* sk_;
  hiopVector* t_;
  hiopVector* rt_;
};

}  // namespace hiop

#endif
