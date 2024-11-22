// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file MatrixCsr.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#include "hiop_blasdefs.hpp"
#include "MatrixCsr.hpp"

#include "cusparse_v2.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#define checkCudaErrors(val) resolveCheckCudaError((val), __FILE__, __LINE__)

namespace ReSolve
{

MatrixCsr::MatrixCsr() {}

MatrixCsr::~MatrixCsr()
{
  if(n_ == 0) return;

  clear_data();
}

void MatrixCsr::allocate_size(int n)
{
  n_ = n;
  checkCudaErrors(cudaMalloc(&irows_, (n_ + 1) * sizeof(int)));
  irows_host_ = new int[n_ + 1]{0};
}

void MatrixCsr::allocate_nnz(int nnz)
{
  nnz_ = nnz;
  checkCudaErrors(cudaMalloc(&jcols_, nnz_ * sizeof(int)));
  checkCudaErrors(cudaMalloc(&vals_, nnz_ * sizeof(double)));
  jcols_host_ = new int[nnz_]{0};
  vals_host_ = new double[nnz_]{0};
}

void MatrixCsr::clear_data()
{
  checkCudaErrors(cudaFree(irows_));
  checkCudaErrors(cudaFree(jcols_));
  checkCudaErrors(cudaFree(vals_));

  irows_ = nullptr;
  jcols_ = nullptr;
  vals_ = nullptr;

  delete[] irows_host_;
  delete[] jcols_host_;
  delete[] vals_host_;

  irows_host_ = nullptr;
  jcols_host_ = nullptr;
  vals_host_ = nullptr;

  n_ = 0;
  nnz_ = 0;
}

void MatrixCsr::update_from_host_mirror()
{
  checkCudaErrors(cudaMemcpy(irows_, irows_host_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(jcols_, jcols_host_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(vals_, vals_host_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
}

void MatrixCsr::copy_to_host_mirror()
{
  checkCudaErrors(cudaMemcpy(irows_host_, irows_, sizeof(int) * (n_ + 1), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(jcols_host_, jcols_, sizeof(int) * nnz_, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(vals_host_, vals_, sizeof(double) * nnz_, cudaMemcpyDeviceToHost));
}

// Error checking utility for CUDA
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void MatrixCsr::resolveCheckCudaError(T result, const char* const file, int const line)
{
  if(result) {
    std::cout << "CUDA error at " << file << ":" << line << " error# " << result << "\n";
    assert(false);
  }
}

}  // namespace ReSolve
