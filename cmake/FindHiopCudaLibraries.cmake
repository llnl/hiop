
#[[

Exports target `hiop_cuda` which finds all cuda libraries needed by hiop.

]]

add_library(hiop_cuda INTERFACE)

find_package(CUDAToolkit REQUIRED)

if(HIOP_BUILD_SHARED)
  target_link_libraries(hiop_cuda INTERFACE
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cudart
    CUDA::curand
    )
endif()
if(HIOP_BUILD_STATIC)
  target_link_libraries(hiop_cuda INTERFACE
    CUDA::cusolver_static
    CUDA::cusparse_static
    CUDA::cublas_static
    CUDA::cublasLt_static
    CUDA::cudart_static
    CUDA::curand_static
    CUDA::culibos
    )
endif()

install(TARGETS hiop_cuda EXPORT hiop-targets)

message(STATUS "HiOp support for GPUs is on")
message(STATUS "CUDA include directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
get_target_property(cuda_libraries hiop_cuda INTERFACE_LINK_LIBRARIES)
message(STATUS "CUDA linked libraries: ${cuda_libraries}")

