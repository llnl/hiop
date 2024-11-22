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
 * @file MatrixTestsRajaSparseTriplet.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */

#pragma once

// #include <hiopMatrixRajaSparseTriplet.hpp>
#include "matrixTestsSparse.hpp"

namespace hiop
{
namespace tests
{

/**
 * @brief Tests are re-implemented here if necessary for SparseTriplet Matrices,
 * as the data layout is significantly different compares to dense matrices.
 * Any tests that would modify the sparsity pattern are not implemented.
 * Any tests that would make calls to non-implemented/needed functions are not implemented.
 *
 */
class MatrixTestsRajaSparseTriplet : public MatrixTestsSparse
{
public:
  MatrixTestsRajaSparseTriplet() {}
  virtual ~MatrixTestsRajaSparseTriplet() {}

private:
  virtual void setLocalElement(hiop::hiopVector* _x, const local_ordinal_type i, const real_type val) override;
  virtual real_type getLocalElement(const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) override;
  virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i) override;
  virtual real_type* getMatrixData(hiop::hiopMatrixSparse* a) override;
  virtual real_type getMatrixData(hiop::hiopMatrixSparse* a, local_ordinal_type i, local_ordinal_type j) override
  {
    assert(false);
    return 0;
  };
  virtual const local_ordinal_type* getRowIndices(const hiop::hiopMatrixSparse* a) override;
  virtual const local_ordinal_type* getColumnIndices(const hiop::hiopMatrixSparse* a) override;
  virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) override;
  virtual int verifyAnswer(hiop::hiopMatrixSparse* A, real_type answer) override;
  virtual int verifyAnswer(hiop::hiopMatrix* A,
                           local_ordinal_type nnz_st,
                           local_ordinal_type nnz_ed,
                           const double answer) override;
  virtual int verifyAnswer(hiop::hiopMatrixDense* A,
                           std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) override;
  virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) override;
  virtual int verifyAnswer(hiop::hiopVector* x, std::function<real_type(local_ordinal_type)> expect) override;
  virtual local_ordinal_type* numNonzerosPerRow(hiop::hiopMatrixSparse* mat) override;
  virtual local_ordinal_type* numNonzerosPerCol(hiop::hiopMatrixSparse* mat) override;
  virtual void maybeCopyToDev(hiop::hiopMatrixSparse*) override;
  virtual void maybeCopyFromDev(hiop::hiopMatrixSparse*) override;

  virtual int getLocalElement(hiop::hiopVectorInt*, int) const override;
  virtual void setLocalElement(hiop::hiopVectorInt*, int, int) const override;

public:
  virtual void initializeMatrix(hiop::hiopMatrixSparse* mat, local_ordinal_type entries_per_row) override;
};

}  // namespace tests
}  // namespace hiop
