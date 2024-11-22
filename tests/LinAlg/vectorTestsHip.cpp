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
 * @file vectorTestsHip.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include <hiopVectorHip.hpp>
#include "vectorTestsHip.hpp"

namespace hiop
{
namespace tests
{

/// Returns const pointer to local vector data
const real_type* VectorTestsHip::getLocalDataConst(hiop::hiopVector* x_in)
{
  if(auto* x = dynamic_cast<hiop::hiopVectorHip*>(x_in)) {
    x->copyFromDev();
    return x->local_data_host_const();
  } else {
    assert(false && "Wrong type of vector passed into `VectorTestsRajaPar::getLocalDataConst`!");
    THROW_NULL_DEREF;
  }
}

/// Method to set vector _x_ element _i_ to _value_.
void VectorTestsHip::setLocalElement(hiop::hiopVector* x_in, local_ordinal_type i, real_type val)
{
  if(auto* x = dynamic_cast<hiop::hiopVectorHip*>(x_in)) {
    x->copyFromDev();
    real_type* xdat = x->local_data_host();
    xdat[i] = val;
    x->copyToDev();
  } else {
    assert(false && "Wrong type of vector passed into `vectorTestsHip::setLocalElement`!");
    THROW_NULL_DEREF;
  }
}

/// Get communicator
MPI_Comm VectorTestsHip::getMPIComm(hiop::hiopVector* x)
{
  if(auto* xvec = dynamic_cast<const hiop::hiopVectorHip*>(x)) {
    return xvec->get_mpi_comm();
  } else {
    assert(false && "Wrong type of vector passed into `vectorTestsHip::getMPIComm`!");
    THROW_NULL_DEREF;
  }
}

/// Wrap new command
real_type* VectorTestsHip::createLocalBuffer(local_ordinal_type N, real_type val)
{
  real_type* buffer = new real_type[N];
  real_type* dev_buffer = nullptr;

  // Set buffer elements to the initial value
  for(local_ordinal_type i = 0; i < N; ++i) buffer[i] = val;

#ifdef HIOP_USE_GPU
  // Allocate memory on GPU
  hipError_t cuerr = hipMalloc((void**)&dev_buffer, N * sizeof(real_type));
  assert(hipSuccess == cuerr);
  cuerr = hipMemcpy(dev_buffer, buffer, N * sizeof(real_type), hipMemcpyHostToDevice);
  assert(cuerr == hipSuccess);

  delete[] buffer;
  return dev_buffer;
#endif

  return buffer;
}

local_ordinal_type* VectorTestsHip::createIdxBuffer(local_ordinal_type N, local_ordinal_type val)
{
  local_ordinal_type* buffer = new local_ordinal_type[N];
  // Set buffer elements to the initial value
  for(local_ordinal_type i = 0; i < N; ++i) buffer[i] = val;
  buffer[N - 1] = 0;

#ifdef HIOP_USE_GPU
  // Allocate memory on GPU
  local_ordinal_type* dev_buffer = nullptr;
  hipError_t cuerr = hipMalloc((void**)&dev_buffer, N * sizeof(local_ordinal_type));
  assert(hipSuccess == cuerr);
  cuerr = hipMemcpy(dev_buffer, buffer, N * sizeof(local_ordinal_type), hipMemcpyHostToDevice);
  assert(cuerr == hipSuccess);

  delete[] buffer;
  return dev_buffer;
#endif

  return buffer;
}

/// Wrap delete command
void VectorTestsHip::deleteLocalBuffer(real_type* buffer)
{
#ifdef HIOP_USE_GPU
  hipFree(buffer);
  return;
#endif
  delete[] buffer;
}

/// If test fails on any rank set fail flag on all ranks
bool VectorTestsHip::reduceReturn(int failures, hiop::hiopVector* x)
{
  int fail = 0;

#ifdef HIOP_USE_MPI
  MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(x));
#else
  fail = failures;
#endif

  return (fail != 0);
}

}  // namespace tests
}  // namespace hiop
