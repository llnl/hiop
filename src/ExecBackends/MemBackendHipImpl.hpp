// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
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
 * @file MemBackendHipImpl.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

/**
 * This file contains HIP implementation of memory backend.
 */

#ifndef HIOP_MEM_SPACE_HIP
#define HIOP_MEM_SPACE_HIP

#include <ExecSpace.hpp>

#include <hip/hip_runtime.h>
#include <cassert>

namespace hiop
{
//
// Memory allocator and deallocator
//
template<typename T, typename I>
struct AllocImpl<MemBackendHip, T, I>
{
  inline static T* alloc(MemBackendHip& mb, const I& n)
  {
    T* p = nullptr;
    auto err = hipMalloc((void**)&p, n * sizeof(T));
    assert(hipSuccess == err);
    return p;
  }
};

template<typename T>
struct DeAllocImpl<MemBackendHip, T>
{
  inline static void dealloc(MemBackendHip& mb, T* p)
  {
    auto err = hipFree((void*)p);
    assert(hipSuccess == err);
  }
};

//
// Transfers
//
template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendHip, EXECPOLDEST, MemBackendHip, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendHip, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendHip, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyDeviceToDevice);
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendHip, EXECPOLDEST, MemBackendCpp, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendHip, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyHostToDevice);
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendCpp, EXECPOLDEST, MemBackendHip, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendHip, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyDeviceToHost);
  }
};

}  // end namespace hiop
#endif  // HIOP_MEM_SPACE_HIP
