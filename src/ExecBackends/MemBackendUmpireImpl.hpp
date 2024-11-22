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
 * @file MemBackendUmpireImpl.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

/**
 * This file contains Umpire implementation memory backend.
 */

#ifndef HIOP_MEM_BCK_UMPIRE
#define HIOP_MEM_BCK_UMPIRE

#include <ExecSpace.hpp>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

namespace hiop
{

//
// Memory allocator and deallocator
//
template<typename T, typename I>
struct AllocImpl<MemBackendUmpire, T, I>
{
  inline static T* alloc(MemBackendUmpire& mb, const I& n)
  {
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc = resmgr.getAllocator(mb.mem_space());
    return static_cast<T*>(devalloc.allocate(n * sizeof(T)));
  }
};

template<typename T>
struct DeAllocImpl<MemBackendUmpire, T>
{
  inline static void dealloc(MemBackendUmpire& mb, T* p)
  {
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc = resmgr.getAllocator(mb.mem_space());
    devalloc.deallocate(p);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Transfers
//////////////////////////////////////////////////////////////////////////////////////////

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(n > 0) {
      auto& rm = umpire::ResourceManager::getInstance();
      T* src = const_cast<T*>(p_src);
      rm.copy(p_dest, src, n * sizeof(T));
    }
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from Host C++ memory
////////////////////////////////////////////////////////////////////////////////////////
template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendCpp, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_src.mem_backend().is_host()) {
      std::memcpy(p_dest, p_src, n * sizeof(T));
    } else {
      if(n > 0) {
        auto& rm = umpire::ResourceManager::getInstance();
        T* src = const_cast<T*>(p_src);
        assert(src);

        // This is a hack to go around the fact that Umpire cannot copy to a pointer (on host
        // in this case) that it does not manage.
        //
        // The solution is to have Umpire allocate and transfer to a pointer on host, followed
        // by a std::memcpy from the new pointer to the desired (also host) destination.
        //
        // This is temporary and should be removed when hiopNlpFormulation::process_bounds
        // will be ported to use vectors other than hiopVectorPar. TODO
        umpire::Allocator host_alloc = rm.getAllocator("HOST");
        T* src_host = static_cast<T*>(host_alloc.allocate(n * sizeof(T)));
        rm.copy(src_host, src, n * sizeof(T));

        std::memcpy(p_dest, src_host, n * sizeof(T));

        host_alloc.deallocate(src_host);
      }
    }
    return true;
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendCpp, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_dest.mem_backend().is_host()) {
      std::memcpy(p_dest, p_src, n * sizeof(T));
    } else {
      if(n > 0) {
        // TODO: Note: see note above in the sister TransferImpl
        auto& rm = umpire::ResourceManager::getInstance();
        umpire::Allocator host_alloc = rm.getAllocator("HOST");
        T* dest_host = static_cast<T*>(host_alloc.allocate(n * sizeof(T)));

        std::memcpy(dest_host, p_src, n * sizeof(T));

        rm.copy(p_dest, dest_host, n * sizeof(T));
        host_alloc.deallocate(dest_host);
      }
    }
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from CUDA memory
////////////////////////////////////////////////////////////////////////////////////////
#ifdef HIOP_USE_CUDA
template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendCuda, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCuda, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_src.mem_backend().is_device()) {
      return cudaSuccess == cudaMemcpy(p_dest, p_src, n * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      if(hwb_src.mem_backend().is_host()) {
        return cudaSuccess == cudaMemcpy(p_dest, p_src, n * sizeof(T), cudaMemcpyHostToDevice);
      } else {
        assert(false && "Transfer BACKENDS(TO:Cuda,FROM:umpire) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendCuda, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCuda, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_dest.mem_backend().is_device()) {
      return cudaSuccess == cudaMemcpy(p_dest, p_src, n * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      if(hwb_dest.mem_backend().is_host()) {
        return cudaSuccess == cudaMemcpy(p_dest, p_src, n * sizeof(T), cudaMemcpyDeviceToHost);
      } else {
        assert(false && "Transfer BACKENDS(TO:Umpire,FROM:Cuda) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};
#endif  // HIOP_USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from HIP memory
////////////////////////////////////////////////////////////////////////////////////////
#ifdef HIOP_USE_HIP
template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendHip, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendHip, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_src.mem_backend().is_device()) {
      return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyDeviceToDevice);
    } else {
      if(hwb_src.mem_backend().is_host()) {
        return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyHostToDevice);
      } else {
        assert(false && "Transfer BACKENDS(TO:Hip,FROM:umpire) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T, typename I>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendHip, EXECPOLSRC, T, I>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendHip, EXECPOLSRC>& hwb_src,
                           const I& n)
  {
    if(hwb_dest.mem_backend().is_device()) {
      return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyDeviceToDevice);
    } else {
      if(hwb_dest.mem_backend().is_host()) {
        return hipSuccess == hipMemcpy(p_dest, p_src, n * sizeof(T), hipMemcpyDeviceToHost);
      } else {
        assert(false && "Transfer BACKENDS(TO:Umpire,FROM:Hip) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};

#endif
}  // end namespace hiop
#endif  // HIOP_MEM_BCK_UMPIRE
