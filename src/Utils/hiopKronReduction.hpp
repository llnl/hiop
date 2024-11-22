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

#ifndef HIOP_KRONRED
#define HIOP_KRONRED

#include "hiop_blasdefs.hpp"
#include "hiopMatrixComplexSparseTriplet.hpp"
#include "hiopMatrixComplexDense.hpp"

#include <string>
#include <vector>
#include <map>

namespace hiop
{
// forward defs
class hiopLinSolverUMFPACKZ;
class hiopMatrixComplexDense;

/* Utility to perform the Kron reduction of the Ybus matrix (sparse symmetric complex)
 * into the reduced Ybus (dense symmetric complex) matrix
 */
class hiopKronReduction
{
public:
  hiopKronReduction();
  virtual ~hiopKronReduction();

  /* Performs the Kron reduction (computes Schur complement)
   * In parameters
   *  - idx_nonaux_buses, idx_aux_buses: indexes of the auxiliary and non-auxiliary
   * buses (in the rows/cols of Ybus)
   *  - Ybus
   * Out parameters
   *  - Ybus_red: reduced Ybus of size (nonaux,nonaux) = Yaa - Yab'*(Ybb\Yba)
   *
   * The function factorizes Ybb and stores the factorization for later use, for example for use
   * in @axpy_nonaux_to_aux
   */
  bool go(const std::vector<int>& idx_nonaux_buses,
          const std::vector<int>& idx_aux_buses,
          const hiopMatrixComplexSparseTriplet& Ybus,
          hiopMatrixComplexDense& Ybus_red);

  /**
   * Performs v_aux_out = (Ybb\Yba)* v_nonaux_in
   */
  bool apply_nonaux_to_aux(const std::vector<std::complex<double> >& v_nonaux_in,
                           std::vector<std::complex<double> >& v_aux_out);

  const hiopMatrixComplexDense& map_nonaux_to_aux() const { return *map_nonaux_to_aux_; }

private:
  hiopLinSolverUMFPACKZ* linsolver_;
  hiopMatrixComplexDense* map_nonaux_to_aux_;
};

}  // namespace hiop
#endif
