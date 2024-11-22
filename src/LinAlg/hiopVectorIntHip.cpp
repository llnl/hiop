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
 * @file hiopVectorIntHip.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include "hiopVectorIntHip.hpp"
#include "hiopVectorIntSeq.hpp"

#include "MemBackendHipImpl.hpp"
#include "MemBackendCppImpl.hpp"
#include "VectorHipKernels.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace hiop
{

hiopVectorIntHip::hiopVectorIntHip(size_type sz, std::string mem_space)
    : hiopVectorInt(sz)
{
  buf_ = exec_space_.alloc_array<index_type>(sz);
  // Create host mirror if the memory space is on device
  buf_host_ = exec_space_host_.alloc_array<index_type>(sz);
}

hiopVectorIntHip::~hiopVectorIntHip()
{
  exec_space_host_.dealloc_array(buf_host_);
  exec_space_.dealloc_array(buf_);
  buf_ = nullptr;
  buf_host_ = nullptr;
}

void hiopVectorIntHip::copy_from(const index_type* v_local)
{
  if(v_local) {
    exec_space_.copy(buf_, v_local, sz_);
  }
}

void hiopVectorIntHip::copy_from_vectorseq(const hiopVectorIntSeq& src)
{
  assert(src.get_local_size() == sz_);
  auto b = exec_space_.copy(buf_, src.local_data_const(), sz_, src.exec_space());
  assert(b);
}

void hiopVectorIntHip::copy_to_vectorseq(hiopVectorIntSeq& dest) const
{
  assert(dest.get_local_size() == sz_);
  auto b = dest.exec_space().copy(dest.local_data(), buf_, sz_, exec_space_);
  assert(b);
}

void hiopVectorIntHip::set_to_zero()
{
  hipError_t cuerr = hipMemset(buf_, 0, sz_);
  assert(cuerr == hipSuccess);
}

/// Set all vector elements to constant c
void hiopVectorIntHip::set_to_constant(const index_type c)
{
  hipError_t cuerr = hipMemset(buf_, c, sz_);
  assert(cuerr == hipSuccess);
}

/**
 * @brief Set the vector entries to be a linear space of starting at i0 containing evenly
 * incremented integers up to i0+(n-1)di, when n is the length of this vector
 *
 * @pre The elements of the linear space should not overflow the index_type type
 *
 * @param i0 the starting element in the linear space (entry 0 in vector)
 * @param di the increment for subsequent entries in the vector
 *
 */
void hiopVectorIntHip::linspace(const index_type& i0, const index_type& di)
{
  hiop::hip::set_to_linspace_kernel(sz_, buf_, i0, di);
}

}  // namespace hiop
