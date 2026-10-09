#[[

Default CMake cache for building in CI without MPI and GPU

#]]
message(STATUS "Enabling both shared and static libraries")
set(HIOP_BUILD_SHARED ON CACHE BOOL "")
set(HIOP_BUILD_STATIC ON CACHE BOOL "")

set(ENABLE_TESTS ON CACHE BOOL "")

message(STATUS "Disabling MPI, RAJA, Umpire, Kron Reduction, and CUDA")
set(HIOP_USE_MPI OFF CACHE BOOL "")
set(HIOP_USE_RAJA OFF CACHE BOOL "")
set(HIOP_USE_UMPIRE OFF CACHE BOOL "")
set(HIOP_WITH_KRON_REDUCTION OFF CACHE BOOL "")
set(HIOP_USE_CUDA OFF CACHE BOOL "")
set(HIOP_USE_GINKGO OFF CACHE BOOL "")

message(STATUS "Enabling HiOp's Sparse Interface")
set(HIOP_SPARSE ON CACHE BOOL "")

message(STATUS "Enabling HiOp's deepchecking")
set(HIOP_DEEPCHECKS ON CACHE BOOL "")

message(STATUS "Setting default cuda architecture to 60")
set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "")

message(STATUS "Done preloading CMake cache with values for continuous integration")
