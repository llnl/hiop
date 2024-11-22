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
 * @file vectorTestsRajaPar.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 * @author Cameron Rutherford <robert.rutherford@pnnl.gov>, PNNL
 *
 */

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

#include <hiopVectorRaja.hpp>
#include "vectorTestsRajaPar.hpp"

// TODO: this is a quick hack. Will need to modify this class to be aware of the instantiated
//  vector template. Likely a better approach would be to revise the `runTests` in testVector.cpp
//  to test multiple configurations of the memory backend and execution policies for RAJA vector.
#if defined(HIOP_USE_CUDA)
#include <ExecPoliciesRajaCudaImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaCuda>;
#elif defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaHipImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaHip>;
#else
// #if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
#include <ExecPoliciesRajaOmpImpl.hpp>
using hiopVectorRajaT = hiop::hiopVectorRaja<hiop::MemBackendUmpire, hiop::ExecPolicyRajaOmp>;
#endif

namespace hiop
{
namespace tests
{

/// Returns const pointer to local vector data
const real_type* VectorTestsRajaPar::getLocalDataConst(hiop::hiopVector* x_in)
{
  if(auto* x = dynamic_cast<hiopVectorRajaT*>(x_in)) {
    x->copyFromDev();
    return x->local_data_host_const();
  } else {
    assert(false && "Wrong type of vector passed into `VectorTestsRajaPar::getLocalDataConst`!");
    THROW_NULL_DEREF;
  }
}

/// Method to set vector _x_ element _i_ to _value_.
void VectorTestsRajaPar::setLocalElement(hiop::hiopVector* x_in, local_ordinal_type i, real_type val)
{
  if(auto* x = dynamic_cast<hiopVectorRajaT*>(x_in)) {
    x->copyFromDev();
    real_type* xdat = x->local_data_host();
    xdat[i] = val;
    x->copyToDev();
  } else {
    assert(false && "Wrong type of vector passed into `VectorTestsRajaPar::setLocalElement`!");
    THROW_NULL_DEREF;
  }
}

/// Get communicator
MPI_Comm VectorTestsRajaPar::getMPIComm(hiop::hiopVector* x)
{
  if(auto* xvec = dynamic_cast<const hiopVectorRajaT*>(x)) {
    return xvec->get_mpi_comm();
  } else {
    assert(false && "Wrong type of vector passed into `VectorTestsRajaPar::getMPIComm`!");
    THROW_NULL_DEREF;
  }
}

/// Wrap new command
real_type* VectorTestsRajaPar::createLocalBuffer(local_ordinal_type N, real_type val)
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator hal = resmgr.getAllocator("HOST");
  real_type* buffer = static_cast<real_type*>(hal.allocate(N * sizeof(real_type)));

  // Set buffer elements to the initial value
  for(local_ordinal_type i = 0; i < N; ++i) buffer[i] = val;

#ifdef HIOP_USE_GPU
  umpire::Allocator dal = resmgr.getAllocator("DEVICE");
  real_type* dev_buffer = static_cast<real_type*>(dal.allocate(N * sizeof(real_type)));
  resmgr.copy(dev_buffer, buffer, N * sizeof(real_type));
  hal.deallocate(buffer);
  return dev_buffer;
#endif

  return buffer;
}

local_ordinal_type* VectorTestsRajaPar::createIdxBuffer(local_ordinal_type N, local_ordinal_type val)
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator hal = resmgr.getAllocator("HOST");
  local_ordinal_type* buffer = static_cast<local_ordinal_type*>(hal.allocate(N * sizeof(local_ordinal_type)));

  // Set buffer elements to the initial value
  for(local_ordinal_type i = 0; i < N; ++i) buffer[i] = val;
  buffer[N - 1] = 0;
#ifdef HIOP_USE_GPU
  umpire::Allocator dal = resmgr.getAllocator("DEVICE");
  local_ordinal_type* dev_buffer = static_cast<local_ordinal_type*>(dal.allocate(N * sizeof(local_ordinal_type)));
  resmgr.copy(dev_buffer, buffer, N * sizeof(local_ordinal_type));
  hal.deallocate(buffer);
  return dev_buffer;
#endif

  return buffer;
}

/// Wrap delete command
void VectorTestsRajaPar::deleteLocalBuffer(real_type* buffer)
{
#ifdef HIOP_USE_GPU
  const std::string hiop_umpire_dev = "DEVICE";
#else
  const std::string hiop_umpire_dev = "HOST";
#endif

  auto& resmgr = umpire::ResourceManager::getInstance();
  umpire::Allocator al = resmgr.getAllocator(hiop_umpire_dev);
  al.deallocate(buffer);
}

/// If test fails on any rank set fail flag on all ranks
bool VectorTestsRajaPar::reduceReturn(int failures, hiop::hiopVector* x)
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
