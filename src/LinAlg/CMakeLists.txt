
# Set headers to be installed as part of the hiop interface
set(hiopLinAlg_INTERFACE_HEADERS
  hiop_blasdefs.hpp
  LinAlgFactory.hpp
  hiopLinSolver.hpp
  hiopLinSolverSymDenseLapack.hpp
  hiopLinSolverSymDenseMagma.hpp
  hiopLinSolverSymSparseMA57.hpp
  hiopLinSolverMA86Z.hpp
  hiopLinSolverSparseGinkgo.hpp
  hiopLinSolverSparseSTRUMPACK.hpp
  hiopLinSolverSparsePARDISO.hpp
  hiopLinSolverSparseSuperLU.hpp
  hiopLinSolverSparseReSolve.hpp
  hiopLinSolverUMFPACKZ.hpp
  hiopLinSolverCholCuSparse.hpp
  hiopMatrix.hpp
  hiopMatrixComplexDense.hpp
  hiopMatrixComplexSparseTriplet.hpp
  hiopMatrixDense.hpp
  hiopMatrixDenseRowMajor.hpp
  hiopMatrixMDS.hpp
  hiopMatrixDenseRaja.hpp
  hiopMatrixRajaSparseTriplet.hpp
  hiopMatrixSparse.hpp
  hiopMatrixSparseTriplet.hpp
  hiopMatrixSparseTripletStorage.hpp
  hiopMatrixSparseCSR.hpp
  hiopMatrixSparseCSRSeq.hpp
  hiopMatrixSparseCsrCuda.hpp
  hiopVector.hpp
  hiopVectorInt.hpp
  hiopVectorIntCuda.hpp
  hiopVectorIntHip.hpp
  hiopVectorIntRaja.hpp
  hiopVectorIntSeq.hpp
  hiopVectorCuda.hpp
  hiopVectorHip.hpp
  hiopVectorPar.hpp
  hiopLinearOperator.hpp
  hiopKrylovSolver.hpp
  hiopVectorCompoundPD.hpp
  hiopVectorIntCompoundPD.hpp
  )

# Set linear algebra common source files
set(hiopLinAlg_SRC
  hiopVectorPar.cpp
  hiopVectorIntSeq.cpp
  hiopMatrixDenseRowMajor.cpp
  hiopLinSolver.cpp
  LinAlgFactory.cpp
  hiopMatrixMDS.cpp
  hiopMatrixComplexDense.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp
  hiopMatrixComplexSparseTriplet.cpp
  hiopMatrixSparseCSRSeq.cpp
  hiopLinearOperator.cpp
  hiopKrylovSolver.cpp
  hiopVectorCompoundPD.cpp
  hiopVectorIntCompoundPD.cpp
)

if(HIOP_USE_CUDA)
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorRajaCuda.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorIntRajaCuda.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixDenseRajaCuda.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixRajaSparseTripletCuda.cpp)
elseif(HIOP_USE_HIP)
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorRajaHip.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorIntRajaHip.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixDenseRajaHip.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixRajaSparseTripletHip.cpp)
else()
  #raja - omp
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorRajaOmp.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopVectorIntRajaOmp.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixDenseRajaOmp.cpp)
  list(APPEND hiopLinAlg_RAJA_SRC hiopMatrixRajaSparseTripletOmp.cpp)
endif()


set(hiopLinAlg_MAGMA_SRC
  hiopLinSolverSymDenseMagma.cpp
)

set(hiopLinAlg_MA57_SRC
  hiopLinSolverSymSparseMA57.cpp
)

set(hiopLinAlg_STRUMPACK_SRC
  hiopLinSolverSparseSTRUMPACK.cpp
)

set(hiopLinAlg_PARDISO_SRC
  hiopLinSolverSparsePARDISO.cpp
)

set(hiopLinAlg_SUPERLU_SRC
  hiopLinSolverSparseSuperLU.cpp
)

set(hiopLinAlg_CUSOLVER_LU_SRC
  hiopLinSolverSparseReSolve.cpp
)

set(hiopLinAlg_CUSOLVER_CHOL_SRC
  hiopLinSolverCholCuSparse.cpp
)

set(hiopLinAlg_CUDASPARSEKERNELS_SRC
  MatrixSparseCsrCudaKernels.cu
)

set(hiopLinAlg_CUDASPARSE_SRC
  hiopMatrixSparseCsrCuda.cpp
)

set(hiopLinAlg_CUDAVECTOR_SRC
  hiopVectorCuda.cpp
  hiopVectorIntCuda.cpp
)
set(hiopLinAlg_CUDAVECTORKERNELS_SRC
  VectorCudaKernels.cu
)

set(hiopLinAlg_HIPVECTOR_SRC
  hiopVectorHip.cpp
  hiopVectorIntHip.cpp
)
set(hiopLinAlg_HIPVECTORKERNELS_SRC
  VectorHipKernels.cpp
)

set(hiopLinAlg_KRON_REDUCTION_SRC
  hiopLinSolverUMFPACKZ.cpp
)

set(hiopLinAlg_Ginkgo_SRC
  hiopLinSolverSparseGinkgo.cpp
)

# Add interfaces for sparse linear solvers when enabled
if(HIOP_SPARSE)
    if(HIOP_USE_COINHSL)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_MA57_SRC})
    endif(HIOP_USE_COINHSL)
    if(HIOP_USE_STRUMPACK)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_STRUMPACK_SRC})
    endif(HIOP_USE_STRUMPACK)
    if(HIOP_USE_PARDISO)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_PARDISO_SRC})
    endif(HIOP_USE_PARDISO)
    if(HIOP_USE_SUPERLU)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_SUPERLU_SRC})
      # SuperLU headers declare BLAS functions which conflict with HiOp's BLAS declarations
      # Use fpermissive to allow the conflicting declarations
      set_source_files_properties(${hiopLinAlg_SUPERLU_SRC}
        PROPERTIES COMPILE_FLAGS "-Wno-error -fpermissive")
    endif(HIOP_USE_SUPERLU)
    if(HIOP_USE_RESOLVE)
      add_subdirectory(ReSolve) 
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUSOLVER_LU_SRC})
      set_source_files_properties(${hiopLinAlg_CUSOLVER_LU_SRC} PROPERTIES LANGUAGE CUDA)
    endif(HIOP_USE_RESOLVE)
    if(HIOP_USE_CUDA)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUSOLVER_CHOL_SRC})
      set_source_files_properties(${hiopLinAlg_CUSOLVER_CHOL_SRC} PROPERTIES LANGUAGE CUDA)
    endif(HIOP_USE_CUDA)
    
    if(HIOP_USE_GINKGO)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_Ginkgo_SRC})
  endif(HIOP_USE_GINKGO)
endif()

# Add sparse CUDA matrix code (even when HIOP_SPARSE is not defined since it can be used in MDS)
if(HIOP_USE_CUDA)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUDASPARSE_SRC} ${hiopLinAlg_CUDAVECTOR_SRC})
  set_source_files_properties(${hiopLinAlg_CUDASPARSEKERNELS_SRC} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${hiopLinAlg_CUDAVECTORKERNELS_SRC} PROPERTIES LANGUAGE CUDA)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUDASPARSEKERNELS_SRC} ${hiopLinAlg_CUDAVECTORKERNELS_SRC})
endif(HIOP_USE_CUDA)

# Add Hip vector code (even when HIOP_SPARSE is not defined since it can be used in MDS)
if(HIOP_USE_HIP)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_HIPVECTOR_SRC})

  # Mark appropriate source files as HIP. I guess in the future just a
  # LANGUAGE HIP property will suffice. For now do it via compile flags
  # Reference: https://www.olcf.ornl.gov/wp-content/uploads/2021/01/2021UM-Day-3-Joo-Developing-for-Frontier-using-HIP-on-Spock.pdf
#  set_source_files_properties(${hiopLinAlg_HIPVECTORKERNELS_SRC} PROPERTIES LANGUAGE HIP)

  set_source_files_properties( ${hiopLinAlg_HIPVECTORKERNELS_SRC} PROPERTIES LANGUAGE CXX)
  set_source_files_properties( ${hiopLinAlg_HIPVECTORKERNELS_SRC} PROPERTIES COMPILE_FLAGS "-x hip")

  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_HIPVECTORKERNELS_SRC})
endif(HIOP_USE_HIP)

# Add RAJA/Umpire sources when enabled
if(HIOP_USE_RAJA)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_RAJA_SRC})
endif()

# If GPU support is enabled add Magma interface (CUDA version)
# Treat RAJA sources as CUDA (temporary, need more flexible solutions)
if(HIOP_USE_GPU)
  if(HIOP_USE_MAGMA)
    list(APPEND hiopLinAlg_SRC ${hiopLinAlg_MAGMA_SRC})
  endif()
  if(HIOP_USE_CUDA)
    set_source_files_properties(${hiopLinAlg_RAJA_SRC} ${hiopLinAlg_CUSOLVER_SRC} PROPERTIES LANGUAGE CUDA)
  endif()
endif()

# Add interface to UMFPACK when Kron reduction is enabled
if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_KRON_REDUCTION_SRC})
endif()

# Build Kron reduction app
if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE HiOp::HiOp)
endif(HIOP_WITH_KRON_REDUCTION)

install(
  FILES ${hiopLinAlg_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
if(HIOP_USE_RESOLVE)
   target_link_libraries(hiop_tpl INTERFACE ReSolve)
   install(TARGETS ReSolve EXPORT hiop-targets)
endif()
target_link_libraries(hiopLinAlg PRIVATE hiop_tpl)
