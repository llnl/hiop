# Set sources for vector tests
set(testVector_SRC testVector.cpp LinAlg/vectorTestsPar.cpp LinAlg/vectorTestsIntSeq.cpp)

# Set sources for dense matrix tests
set(testMatrix_SRC testMatrixDense.cpp LinAlg/matrixTestsDenseRowMajor.cpp)

# Set sources for sparse matrix tests
set(testMatrixSparse_SRC testMatrixSparse.cpp LinAlg/matrixTestsSparseTriplet.cpp)

# Set sources for symmetric sparse matrix tests
set(testMatrixSymSparse_SRC testMatrixSymSparse.cpp LinAlg/matrixTestsSymSparseTriplet.cpp)

# Set sources for BiCGStab
set(testBiCGStab_SRC test_bicgstab.cpp)

# Set sources for symmetric sparse matrix tests
set(testPCG_SRC test_pcg.cpp)

if(HIOP_SPARSE)
  add_executable(testMPSInterface testMPSInterface.cpp)
  target_link_libraries(testMPSInterface PRIVATE HiOp::HiOp)
  add_executable(testMPSSolve testMPSSolve.cpp)
  target_link_libraries(testMPSSolve PRIVATE HiOp::HiOp)
endif()

# Check if using RAJA and Umpire and add RAJA sources
if(HIOP_USE_RAJA)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsRajaPar.cpp LinAlg/vectorTestsIntRaja.cpp)
  set(testMatrix_SRC ${testMatrix_SRC} LinAlg/matrixTestsRajaDense.cpp)
  set(testMatrixSparse_SRC ${testMatrixSparse_SRC} LinAlg/matrixTestsRajaSparseTriplet.cpp)
  set(testMatrixSymSparse_SRC ${testMatrixSymSparse_SRC} LinAlg/matrixTestsRajaSymSparseTriplet.cpp)
endif()

if(HIOP_USE_CUDA)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsCuda.cpp)
  set(cuda_sources ${testVector_SRC} ${testMatrix_SRC} ${testMatrixSparse_SRC} ${testMatrixSymSparse_SRC} ${testPCG_SRC} ${testBiCGStab_SRC})
  set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)
elseif(HIOP_USE_HIP)
  #set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsHip.cpp)
  set(hip_sources ${testVector_SRC} ${testMatrix_SRC} ${testMatrixSparse_SRC} ${testMatrixSymSparse_SRC} ${testPCG_SRC} ${testBiCGStab_SRC})
  #set_source_files_properties(${hip_sources} PROPERTIES LANGUAGE HIP)
else()
  # raja-omp or non-raja builds do nothing
endif()

if(HIOP_USE_GPU AND HIOP_USE_HIP)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsHip.cpp )
  set(hip_sources ${testVector_SRC} ${testMatrix_SRC} ${testMatrixSparse_SRC} ${testMatrixSymSparse_SRC})

  # Mark appropriate source files as HIP. I guess in the future just a
  # LANGUAGE HIP property will suffice. For now do it via compile flags
  # Reference: https://www.olcf.ornl.gov/wp-content/uploads/2021/01/2021UM-Day-3-Joo-Developing-for-Frontier-using-HIP-on-Spock.pdf
#  set_source_files_properties(${hip_sources} PROPERTIES LANGUAGE HIP)

  # HIP kernels
  set_source_files_properties( ${hip_sources} PROPERTIES LANGUAGE CXX)
  set_source_files_properties( ${hip_sources} PROPERTIES COMPILE_FLAGS "-x hip")
endif()

# Build vector test
add_executable(testVector ${testVector_SRC})
target_link_libraries(testVector PRIVATE HiOp::HiOp)

# Build matrix test
add_executable(testMatrixDense ${testMatrix_SRC})
target_link_libraries(testMatrixDense PRIVATE HiOp::HiOp)

# Build sparse matrix test
add_executable(testMatrixSparse ${testMatrixSparse_SRC})
target_link_libraries(testMatrixSparse PRIVATE HiOp::HiOp)

# Build symmetric sparse matrix test
add_executable(testMatrixSymSparse ${testMatrixSymSparse_SRC})
target_link_libraries(testMatrixSymSparse PRIVATE HiOp::HiOp)

add_executable(test_pcg ${testPCG_SRC})
target_link_libraries(test_pcg PRIVATE HiOp::HiOp)

add_executable(test_bicgstab ${testBiCGStab_SRC})
target_link_libraries(test_bicgstab PRIVATE HiOp::HiOp)
