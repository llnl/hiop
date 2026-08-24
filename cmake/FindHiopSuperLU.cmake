
#[[

Exports target `SUPERLU`

Users may set the following variables:

- HIOP_SUPERLU_DIR or SUPERLU_DIST_ROOT

]]

find_path(SUPERLU_INCLUDE_DIR
  NAMES
  superlu_ddefs.h
  superlu_defs.h
  PATHS
  ${SUPERLU_DIR} $ENV{SUPERLU_DIR}
  ${SUPERLU_DIST_ROOT} $ENV{SUPERLU_DIST_ROOT}
  ${HIOP_SUPERLU_DIR}
  PATH_SUFFIXES
  include)

find_library(SUPERLU_LIBRARY
  NAMES
  superlu_dist
  PATHS
  ${SUPERLU_DIR} $ENV{SUPERLU_DIR}
  ${SUPERLU_DIST_ROOT} $ENV{SUPERLU_DIST_ROOT}
  ${HIOP_SUPERLU_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

# Find ParMETIS (required by SuperLU_DIST)
find_library(PARMETIS_LIBRARY
  NAMES
  parmetis
  PATHS
  ${PARMETIS_DIR} $ENV{PARMETIS_DIR}
  ${PARMETIS_ROOT} $ENV{PARMETIS_ROOT}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(SUPERLU_LIBRARY AND SUPERLU_INCLUDE_DIR)
  get_filename_component(SUPERLU_LIBRARY_DIR ${SUPERLU_LIBRARY} DIRECTORY)
  message(STATUS "Found SuperLU_DIST library: ${SUPERLU_LIBRARY}")
  message(STATUS "Found SuperLU_DIST include: ${SUPERLU_INCLUDE_DIR}")

  add_library(SUPERLU INTERFACE)
  target_link_libraries(SUPERLU INTERFACE ${SUPERLU_LIBRARY})
  target_include_directories(SUPERLU INTERFACE ${SUPERLU_INCLUDE_DIR})

  # SuperLU_DIST requires ParMETIS - link it explicitly
  if(PARMETIS_LIBRARY)
    target_link_libraries(SUPERLU INTERFACE ${PARMETIS_LIBRARY})
    message(STATUS "Linking SuperLU with ParMETIS: ${PARMETIS_LIBRARY}")
  else()
    message(WARNING "ParMETIS not found - SuperLU_DIST may have linking issues")
  endif()

  # SuperLU_DIST requires METIS - will be linked via hiop_tpl
  # SuperLU_DIST requires BLAS/LAPACK - will be linked via hiop_tpl

  # If SuperLU_DIST was built with symmetric matching support (SUITOR/SUMAC/MC80),
  # we need to link the matching libraries
  get_filename_component(SUPERLU_ROOT ${SUPERLU_INCLUDE_DIR} DIRECTORY)

  # Look for SUITOR library (CPU-based symmetric matching)
  # The matching libraries are in the SuperLU source tree, not the install tree
  find_library(SUITOR_LIBRARY
    NAMES suitor
    HINTS
      ${SUPERLU_ROOT}/../matching/lib/matching/lib
      ${SUPERLU_ROOT}/../../matching/lib/matching/lib
      ${SUPERLU_DIR}/../matching/lib/matching/lib
      ${SUPERLU_DIST_ROOT}/../matching/lib/matching/lib
      $ENV{SUPERLU_DIST_ROOT}/../matching/lib/matching/lib
    NO_DEFAULT_PATH)

  if(SUITOR_LIBRARY)
    target_link_libraries(SUPERLU INTERFACE ${SUITOR_LIBRARY})
    message(STATUS "Found SUITOR matching library: ${SUITOR_LIBRARY}")

    # Look for SUMAC library (GPU-accelerated matching)
    if(HIOP_USE_CUDA)
      find_library(SUMAC_LIBRARY
        NAMES sumac
        HINTS
          ${SUPERLU_ROOT}/../matching/lib/sumac
          ${SUPERLU_ROOT}/../../matching/lib/sumac
          ${SUPERLU_DIR}/../matching/lib/sumac
          ${SUPERLU_DIST_ROOT}/../matching/lib/sumac
          $ENV{SUPERLU_DIST_ROOT}/../matching/lib/sumac
        NO_DEFAULT_PATH)

      if(SUMAC_LIBRARY)
        target_link_libraries(SUPERLU INTERFACE ${SUMAC_LIBRARY})
        message(STATUS "Found SUMAC matching library: ${SUMAC_LIBRARY}")

        # SUMAC requires NCCL for GPU communication
        if(DEFINED ENV{NCCL_HOME})
          set(NCCL_LIB "$ENV{NCCL_HOME}/lib/libnccl.so")
          if(EXISTS "${NCCL_LIB}")
            target_link_libraries(SUPERLU INTERFACE ${NCCL_LIB})
            message(STATUS "Linking SUMAC with NCCL: ${NCCL_LIB}")
          endif()
        endif()
      endif()

      # Look for MC80 library (HSL algorithm)
      find_library(MC80_LIBRARY
        NAMES hsl_mc80
        HINTS
          ${SUPERLU_ROOT}/../matching/lib/hsl_mc80-1.1.4/build/lib
          ${SUPERLU_ROOT}/../../matching/lib/hsl_mc80-1.1.4/build/lib
          ${SUPERLU_DIR}/../matching/lib/hsl_mc80-1.1.4/build/lib
          ${SUPERLU_DIST_ROOT}/../matching/lib/hsl_mc80-1.1.4/build/lib
          $ENV{SUPERLU_DIST_ROOT}/../matching/lib/hsl_mc80-1.1.4/build/lib
        NO_DEFAULT_PATH)

      if(MC80_LIBRARY)
        target_link_libraries(SUPERLU INTERFACE ${MC80_LIBRARY})
        message(STATUS "Found MC80 matching library: ${MC80_LIBRARY}")
      endif()
    endif()
  else()
    message(STATUS "SuperLU matching libraries not found - basic functionality only")
  endif()

  # If SuperLU_DIST was built with CUDA support, it needs CUDA libraries
  if(HIOP_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Link shared CUDA libraries for SuperLU (works with both static and shared HiOp builds)
    target_link_libraries(SUPERLU INTERFACE
      CUDA::cublas
      CUDA::cusolver
      CUDA::cusparse)
  endif()

  install(TARGETS SUPERLU EXPORT hiop-targets)
else()
  message(STATUS "SuperLU_DIST was not found.")
endif()
