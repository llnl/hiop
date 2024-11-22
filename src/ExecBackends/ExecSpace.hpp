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
 * @file ExecSpace.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_EXEC_SPACE
#define HIOP_EXEC_SPACE

#include <hiop_defs.hpp>
#include <string>
#include <cassert>

#include "hiopCppStdUtils.hpp"

///////////////////////////////////////////////////////////////////////////////////////
// This header defines the generic execution space class and its various generic types.
///////////////////////////////////////////////////////////////////////////////////////

namespace hiop
{

/**
 * Runtime information about the execution space, namely memory space, memory backend, and execution policies.
 * Closely related to HiOp's option 'mem_space', 'compute_mode', 'mem_backend', and 'exec_policies'.
 */
struct ExecSpaceInfo
{
  ExecSpaceInfo(const std::string mem_space_in)
  {
    mem_space_ = toupper(mem_space_in);
    if(mem_space_ == "DEFAULT") {
      mem_backend_ = "STDCPP";
      mem_backend_host_ = "STDCPP";
      exec_backend_ = "HOST";
    } else if(mem_space_ == "CUDA") {
      mem_backend_ = "CUDA";
      mem_backend_host_ = "STDCPP";
      exec_backend_ = "CUDA";
    } else if(mem_space_ == "HIP") {
      mem_backend_ = "HIP";
      mem_backend_host_ = "STDCPP";
      exec_backend_ = "HIP";
    } else {
      assert(mem_space_ == "DEVICE" || mem_space_ == "UM" || mem_space_ == "HOST");
      mem_backend_ = "UMPIRE";
      mem_backend_host_ = "UMPIRE";
      exec_backend_ = "RAJA";
    }
  }
  ExecSpaceInfo(const char* mem_space_in)
      : ExecSpaceInfo(std::string(mem_space_in))
  {}

  std::string mem_space_;
  std::string mem_backend_;
  std::string mem_backend_host_;
  std::string exec_backend_;
};

///////////////////////////////////////////////////////////////////////////////////////
// Memory backends
///////////////////////////////////////////////////////////////////////////////////////

/// Standard C++ memory backend on host
struct MemBackendCpp
{
  /** Constructor that makes this class compatible to use as a memory backend with RAJA
   * linear algebra objects (i.e., it is exchangeable with Umpire "HOST" memory backend.
   *
   * @pre: input string should be always be "HOST"
   */
  MemBackendCpp(std::string mem_space = "HOST") { assert(mem_space == "HOST"); }

  /// Always on host memory space
  static bool is_host() { return true; }

  /// No host memory space is supported.
  static bool is_device() { return false; }

  // for when the class is used as memory backend with RAJA
  using MemBackendHost = MemBackendCpp;
  /// Returns a backend set up for host memory space
  static MemBackendCpp new_backend_host() { return MemBackendCpp(); };
};

#ifdef HIOP_USE_RAJA  // HIOP_USE_UMPIRE would be better since Hiop RAJA code can now work without Umpire
/**
 * Umpire-based memory backend that supports "HOST", "UM" (unified memory), and "DEVICE"
 * memory spaces.
 */
struct MemBackendUmpire
{
  MemBackendUmpire(const std::string& l)
      : mem_space_(l)
  {}
  MemBackendUmpire()  // todo = delete;
  {
    mem_space_ = "HOST";
  }

  std::string mem_space() const { return mem_space_; }

  inline bool is_host() const { return mem_space_ == "HOST"; }
  inline bool is_device() const { return mem_space_ == "DEVICE"; }

  using MemBackendHost = MemBackendUmpire;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host() { return MemBackendHost("HOST"); };

private:
  std::string mem_space_;
};
#endif  // HIOP_USE_RAJA //HIOP_USE_UMPIRE

#ifdef HIOP_USE_CUDA
/// Cuda memory backend for device memory space that is implemented using Cuda API
struct MemBackendCuda
{
  /**
   * Constructor taking a memory space as input; provided for exchangeability with
   * other memory backends.
   */
  MemBackendCuda(std::string mem_space = "DEVICE") { assert(mem_space == "DEVICE"); }

  /// For now does not support host memory space (but can/will be implemented).
  inline static bool is_host() { return false; }

  inline static bool is_device() { return true; }

  using MemBackendHost = MemBackendCpp;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host() { return MemBackendHost(); };
};
#endif  // HIOP_USE_CUDA

#ifdef HIOP_USE_HIP
/// Cuda memory backend for device memory space that is implemented using Hip API
struct MemBackendHip
{
  /**
   * Constructor taking a memory space as input; provided for exchangeability with
   * other memory backends.
   */
  MemBackendHip(std::string mem_space = "DEVICE") { assert(mem_space == "DEVICE"); }

  /// For now does not support host memory space (but can/will be implemented).
  inline static bool is_host() { return false; }

  inline static bool is_device() { return true; }

  using MemBackendHost = MemBackendCpp;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host() { return MemBackendHost(); };
};
#endif  // HIOP_USE_HIP

///////////////////////////////////////////////////////////////////////////////////////
// Execution policies
///////////////////////////////////////////////////////////////////////////////////////

/// Standard C++ sequential execution
struct ExecPolicySeq
{};

#ifdef HIOP_USE_CUDA
struct ExecPolicyCuda
{
  ExecPolicyCuda()
      : bl_sz_binary_search(16),
        bl_sz_vector_loop(256)
  {}
  /** Block size for kernels performing binary search (e.g., updating or getting diagonal
   *  in CSR CUDA matrices. Default value 16.
   */
  unsigned short int bl_sz_binary_search;
  /// Block size for kernels performing element-wise ops on vectors. Default 256.
  unsigned short int bl_sz_vector_loop;
};
#endif

#ifdef HIOP_USE_HIP
struct ExecPolicyHip
{
  unsigned short int num_blocks_vector;
  unsigned short int num_blocks_search;
};
#endif

///////////////////////////
// RAJA execution policies
//////////////////////////
#ifdef HIOP_USE_RAJA

#ifdef HIOP_USE_CUDA
struct ExecPolicyRajaCuda
{
  // empty since no runtime info is stored
};
#endif

#ifdef HIOP_USE_HIP
struct ExecPolicyRajaHip
{
  // empty since no runtime info is stored
};
#endif

// RAJA OMP execution policies backend
#if !defined(HIOP_USE_CUDA) && !defined(HIOP_USE_HIP)
struct ExecPolicyRajaOmp
{
  // empty since no runtime info is stored
};
#endif

/**
 * The backend RAJA policies that needs to be provided for each one of the ExecPolicyRajaCuda,
 * ExecPolicyRajaHip, and/or ExecPolicyRajaOmp. The class is specialized in HiOp's vendor-specific
 * Raja execution policies source files. Namely, the class' inner types are specialized to
 * vendor-specific RAJA policies types. The inner type below are just for reference and this
 * generic templated struct is/should not be used.
 */
template<class RAJAEXECPOLICIES>
struct ExecRajaPoliciesBackend
{
  using hiop_raja_exec = void;
  using hiop_raja_reduce = void;
  using hiop_raja_atomic = void;

  // The following are primarily for _matrix_exec_
  using hiop_block_x_loop = void;
  using hiop_thread_x_loop = void;
  template<typename T>
  using hiop_kernel = void;
};
#endif  // HIOP_USE_RAJA

///////////////////////////////////////////////////////////////////////////////////////
// The generic/template execution backend class
///////////////////////////////////////////////////////////////////////////////////////

//
// Forward declarations of implementation internals of class ExecSpace
//
template<class MEMBACKEND, typename T, typename I>
struct AllocImpl;

template<class MEMBACKEND, typename T, typename I = void>
struct DeAllocImpl;

template<class MEMBACKENDDEST, class EXEPOLDEST, class MEMBACKENDSRC, class EXEPOLSRC, typename T, typename I>
struct TransferImpl;

/**
 * Hardware backend wrapping a concrete memory backend and a concrete set of execution policies.
 */
template<class MEMBACKEND, class EXECPOLICIES>
class ExecSpace
{
public:
  ExecSpace() = default;
  ExecSpace(const ExecSpace&) = default;

  ExecSpace(const MEMBACKEND& mb)
      : mb_(mb),
        ep_()
  {}

  const MEMBACKEND& mem_backend() const { return mb_; }

  const EXECPOLICIES& exec_policies() const { return ep_; }

  template<typename T, typename I>
  inline T* alloc_array(const I& n)
  {
    return AllocImpl<MEMBACKEND, T, I>::alloc(mb_, n);
  }

  template<typename T>
  inline void dealloc_array(T* p)
  {
    DeAllocImpl<MEMBACKEND, T>::dealloc(mb_, p);
  }

  /**
   * Copy `n` elements of the array `p_src` to the `p_dest` array.
   *
   * @pre `p_src` and `p_dest` should be allocated so that they can hold at least
   * `n` elements.
   * @pre `p_dest` should be managed by the memory backend of `this`.
   * @pre `p_src` should be managed by the memory backend of `ms`.
   */
  template<class MEMSRC, class EXEPOLSRC, typename T, typename I>
  inline bool copy(T* p_dest, const T* p_src, const I& n, const ExecSpace<MEMSRC, EXEPOLSRC>& ms)
  {
    return TransferImpl<MEMBACKEND, EXECPOLICIES, MEMSRC, EXEPOLSRC, T, I>::do_it(p_dest, *this, p_src, ms, n);
  }

  /**
   * Copy `n` elements of the array `p_src` to the `p_dest` array.
   *
   * @pre `p_src` and `p_dest` should be allocated so that they can hold at least
   * `n` elements.
   * @pre Both `p_dest` and `p_src` should be managed by the memory backend of `this`.
   */
  template<typename T, typename I>
  inline bool copy(T* p_dest, const T* p_src, const I& n)
  {
    return TransferImpl<MEMBACKEND, EXECPOLICIES, MEMBACKEND, EXECPOLICIES, T, I>::do_it(p_dest, *this, p_src, *this, n);
  }

private:
  MEMBACKEND mb_;
  EXECPOLICIES ep_;
};

//
// Internals start here
//

/**
 * Memory allocations should be provided via `AllocImpl` for concrete memory backends.
 */
template<class MEMBACKEND, typename T, typename I>
struct AllocImpl
{
  inline static T* alloc(MEMBACKEND& mb, const I& n)
  {
    assert(false && "Specialization for template parameters needs to be provided.");
    return nullptr;
  }
};

/**
 * Memory deallocations should be provided via `DeAllocImpl` for concrete memory backends.
 * The size type `I` is not needed by current implementation and defaulted to `void`.
 */
template<class MEMBACKEND, typename T, typename I /*=void*/>
struct DeAllocImpl
{
  inline static void dealloc(MEMBACKEND& mb, T* p)
  {
    assert(false && "Specialization for template parameters needs to be provided.");
  }
};

/**
 * Transfers between memory backends and memory spaces should be provided by specializations
 * of `TransferImpl` class.
 */
template<class MEMDEST, class EXEPOLDEST, class MEMSRC, class EXEPOLSRC, typename T, typename I>
struct TransferImpl
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MEMDEST, EXEPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MEMSRC, EXEPOLSRC>& hwb_src,
                           const I& n)
  {
    return false;
  }
};

}  // namespace hiop

#endif
