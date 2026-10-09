#[[

Default CMake cache for building in CI on AMDGPU platforms

#]]

if(DEFINED ROCM_PATH AND NOT "${ROCM_PATH}" STREQUAL "")
  set(_hiop_rocm_path "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  set(_hiop_rocm_path "$ENV{ROCM_PATH}")
else()
  set(_hiop_rocm_path "/opt/rocm")
endif()
set(ROCM_PATH "${_hiop_rocm_path}" CACHE PATH "ROCm installation prefix")

# HIP compile options propagated by RAJA and Umpire require ROCm Clang. Select
# it before project() unless the user chose compilers through CMake or the
# standard CC/CXX environment variables.
if(NOT DEFINED CMAKE_C_COMPILER AND "$ENV{CC}" STREQUAL "")
  set(_hiop_amdclang "${_hiop_rocm_path}/bin/amdclang")
  if(NOT EXISTS "${_hiop_amdclang}")
    message(FATAL_ERROR
      "amdclang was not found at ${_hiop_amdclang}. Set ROCM_PATH to the ROCm installation prefix.")
  endif()
  set(CMAKE_C_COMPILER "${_hiop_amdclang}" CACHE FILEPATH "C compiler")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
  set(_hiop_amdclangxx "${_hiop_rocm_path}/bin/amdclang++")
  if(NOT EXISTS "${_hiop_amdclangxx}")
    message(FATAL_ERROR
      "amdclang++ was not found at ${_hiop_amdclangxx}. Set ROCM_PATH to the ROCm installation prefix.")
  endif()
  set(CMAKE_CXX_COMPILER "${_hiop_amdclangxx}" CACHE FILEPATH "C++ compiler")
endif()

unset(_hiop_amdclang)
unset(_hiop_amdclangxx)
unset(_hiop_rocm_path)

set(HIOP_BUILD_SHARED OFF CACHE BOOL "")
set(HIOP_BUILD_STATIC ON CACHE BOOL "")
set(HIOP_USE_MPI ON CACHE BOOL "")
set(HIOP_USE_RAJA ON CACHE BOOL "")
set(HIOP_USE_CUDA OFF CACHE BOOL "")
set(HIOP_USE_HIP ON CACHE BOOL "")
set(HIOP_SPARSE OFF CACHE BOOL "")
set(HIOP_DEEPCHECKS ON CACHE BOOL "")
set(AMDGPU_TARGETS "gfx908" CACHE STRING "")
set(GPU_TARGETS "gfx908" CACHE STRING "")
set(CMAKE_HIP_ARCHITECTURES "gfx908" CACHE STRING "")
