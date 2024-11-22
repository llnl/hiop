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
 * @file hiopLinSolverSymDenseMagma.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_MAGMASOLVER
#define HIOP_MAGMASOLVER

#include "hiopNlpFormulation.hpp"
#include "hiopLinSolver.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"

/** Wrapper classes around MAGMA solver for symmetric indefinite matrices
 *
 */

// these FLOPS counter are from MAGMA
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)) * (n_) - (1. / 6.)))
#define FADDS_POTRI(n_) ((n_) * ((1. / 6.) + (n_) * ((1. / 3.) * (n_) - 0.5)))
#define FADDS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1))
#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FMULS_POTRI(n_) ((n_) * ((2. / 3.) + (n_) * ((1. / 3.) * (n_) + 1.)))
#define FMULS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) + 1))
#define FLOPS_DPOTRF(n_) (FMULS_POTRF((double)(n_)) + FADDS_POTRF((double)(n_)))
#define FLOPS_DPOTRI(n_) (FMULS_POTRI((double)(n_)) + FADDS_POTRI((double)(n_)))
#define FLOPS_DPOTRS(n_, nrhs_) (FMULS_POTRS((double)(n_), (double)(nrhs_)) + FADDS_POTRS((double)(n_), (double)(nrhs_)))

namespace hiop
{

/**
 * Solver based on Magma's GPU interface for 'dsytrf' using Bunch-Kaufmann. This is
 * a numerically stable factorization with decent peak performance on large matrices.
 *
 * @note: The option "compute_mode" decides whether this class is instantiated (values
 * "gpu" and "hybrid"); otherwise, for "cpu", the LAPACK-based counterpart linear solver
 * class is used. However, this class works with all the values for "mem_space" option.
 * Regardless of the value of the "mem_space" option (and when instructed by the
 * "compute_mode" option mentioned above) this class will always perform the factorization
 * and the triangular solves on the device. The host-device communication will be minimal
 * (only scalars) for gpu compute mode; under hybrid compute mode, the class will offload
 * the matrix and rhs to the device.
 *
 */

class hiopLinSolverSymDenseMagmaBuKa : public hiopLinSolverSymDense
{
public:
  hiopLinSolverSymDenseMagmaBuKa(int n, hiopNlpFormulation* nlp_);
  virtual ~hiopLinSolverSymDenseMagmaBuKa();

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).
   */
  bool solve(hiopVector& x_in);

  inline hiopMatrixDense& sysMatrix() { return *M_; }

protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers
   * consisting of the counts of positive, negative, and null eigenvalues
   *
   * @pre The system matrix is factorized and, as a result, `ipiv` has been
   * also updated properly.
   */
  virtual bool compute_inertia(int n, int* ipiv_in, int& posEig, int& negEig, int& zeroEig);

protected:
  int* ipiv_;
  /// array storing the inertia (on the device pointer)
  int* dinert_;
  magma_queue_t magma_device_queue_;
  magmaDouble_ptr device_M_, device_rhs_;
  magma_int_t ldda_, lddb_;

private:
  hiopLinSolverSymDenseMagmaBuKa() { assert(false); }
};

/**
 * Solver class for MAGMA symmetric indefinite GPU factorization "_nopiv". This
 * is as much as twice faster but numerically less stable than the above
 * factorization.
 *
 */

class hiopLinSolverSymDenseMagmaNopiv : public hiopLinSolverSymDense
{
public:
  hiopLinSolverSymDenseMagmaNopiv(int n, hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverSymDenseMagmaNopiv();

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged();

  /**
   * Solves a linear system with the right-hand side `x`. This is also an out
   * parameter and on exit it contains the solution.
   */
  bool solve(hiopVector& x);

  inline hiopMatrixDense& sysMatrix() { return *M_; }

protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers
   * of positive, negative, and null eigenvalues. This method runs on device and
   * accesses the device pointer(s). All the parameters reside on device.
   *
   * @pre The system matrix is factorized and is present on the device.
   *
   */
  bool compute_inertia(int n, int& posEigvals, int& negEigvals, int& zeroEigvals);

protected:
  magma_queue_t magma_device_queue_;
  magmaDouble_ptr device_M_, device_rhs_;
  magma_int_t ldda_, lddb_;

private:
  hiopLinSolverSymDenseMagmaNopiv() { assert(false); }
};

#if 0

/** 
 * Solver based on Magma's CPU interface for GPU implementation of 'dsytrf' using 
 * Bunch-Kaufmann. This is a numerically stable factorization with decent peak 
 * performance on large matrices.
 *
 * Superceeded by BuKa solver - code is disabled
 */
class hiopLinSolverSymDenseMagmaBuKa_old2 : public hiopLinSolverSymDense
{
public:
  hiopLinSolverSymDenseMagmaBuKa_old2(int n, hiopNlpFormulation* nlp_);
  virtual ~hiopLinSolverSymDenseMagmaBuKa_old2();

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  
   */
  bool solve(hiopVector& x_in);

  inline hiopMatrixDense& sysMatrix() 
  { 
    return *M_; 
  }
protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers 
   * consisting of the counts of positive, negative, and null eigenvalues
   *
   * @pre The system matrix is factorized and, as a result, `ipiv` has been 
   * also updated properly.
   */
  virtual bool compute_inertia(int n, int *ipiv_in, int& posEig, int& negEig, int& zeroEig); 

protected:
  int* ipiv_;

  //allocated on demand; for example, it may only be required by Magma SIDI.
  double* work_;

protected:

private:
  hiopLinSolverSymDenseMagmaBuKa_old2() { assert(false); }
};

#endif  // 0

}  // end namespace hiop

#endif  // of HIOP_USE_MAGMA
#endif
