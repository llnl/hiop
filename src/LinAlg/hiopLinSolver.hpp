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
 * @file hiopLinSolver.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_LINSOLVER
#define HIOP_LINSOLVER

#include "hiopNlpFormulation.hpp"
#include "hiopMatrix.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopVector.hpp"

#include "hiop_blasdefs.hpp"

#include "hiopCppStdUtils.hpp"

namespace hiop
{

/**
 * Abstract class specifying the linear solver interface needed by interior-point
 * methods of HiOp. Implementations of this abstract class should be wrappers
 * of existing CPU and GPU libraries for solving linear systems.
 */

class hiopLinSolver
{
public:
  hiopLinSolver();
  virtual ~hiopLinSolver();

  /** Triggers a refactorization of the matrix, if necessary.
   * Returns number of negative eigenvalues or -1 if null eigenvalues
   * are encountered.
   */
  virtual int matrixChanged() = 0;

  /**
   * Method to solve the linear system once the factorization phase has been
   * completed by matrixChanged().
   *
   * @param x is on entry the right-hand side of the system to be solved. On exit
   * it contains the solution.
   */
  virtual bool solve(hiopVector& x) = 0;

  /**
   * Method to solve the linear system with multiple right-hand sides once the
   * factorization phase has been completed by matrixChanged().
   *
   * @param x contains on entry the right-hand side(s) of the system to be solved
   * and storesthe solutions on exit.
   */
  virtual bool solve(hiopMatrix& x)
  {
    assert(false && "not yet supported");
    return false;
  }

public:
  hiopNlpFormulation* nlp_;
  bool perf_report_;
};

/** Base class for Indefinite Dense Solvers */
class hiopLinSolverSymDense : public hiopLinSolver
{
public:
  hiopLinSolverSymDense(int n, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverSymDense();

  hiopMatrixDense& sysMatrix();

protected:
  hiopMatrixDense* M_;

protected:
  hiopLinSolverSymDense();
};

/**
 * Base class for symmetric and non-symmetric sparse linear systems/solvers
 */
class hiopLinSolverSparseBase : public hiopLinSolver
{
public:
  hiopLinSolverSparseBase()
      : M_(nullptr),
        sys_mat_owned_(true)
  {}

  virtual ~hiopLinSolverSparseBase()
  {
    if(sys_mat_owned_) {
      delete M_;
    }
  }

  inline hiopMatrixSparse* sys_matrix() { return M_; }

  inline void set_sys_matrix(hiopMatrixSparse* M)
  {
    if(sys_mat_owned_) {
      assert(false && "system matrix should not have been allocated internally when calling set_sys_matrix");
      delete M_;
    }
    sys_mat_owned_ = false;
    M_ = M;
  }

protected:
  hiopMatrixSparse* M_;
  bool sys_mat_owned_;
};

/**
 * Base class for symmetric (indefinite or positive definite) sparse solvers
 */
class hiopLinSolverSymSparse : public hiopLinSolverSparseBase
{
public:
  /// Constructor that creates and owns internal system matrix of size `n` and with `nnz` nonzeros
  hiopLinSolverSymSparse(size_type n, size_type nnz, hiopNlpFormulation* nlp);

  /**
   * Constructor that uses the matrix passed as argument as internal system matrix. The system matrix will NOT be
   * managed by this class
   *
   * @note This constructor should set `sys_mat_owned_` to `false`.
   */
  hiopLinSolverSymSparse(hiopMatrixSparse* M, hiopNlpFormulation* nlp);

  /**
   * Barebone constructor that does not create or set internal system matrix. It should be used for cases when
   * the system matrix is not available upon instantiation of this class. This system matrix should be subsequently
   * set by the calling code by invoking `set_sys_matrix`.
   *
   * @note This constructor should set `sys_mat_owned_` to false.
   */
  hiopLinSolverSymSparse(hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverSymSparse() {}

protected:
  hiopLinSolverSymSparse() {}
};

/**
 * Base class for non-symmetric sparse solvers
 */
class hiopLinSolverNonSymSparse : public hiopLinSolverSparseBase
{
public:
  hiopLinSolverNonSymSparse(size_type n, size_type nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverNonSymSparse() {}

protected:
  hiopLinSolverNonSymSparse() = delete;
};

}  // namespace hiop

#endif
