set(hiopSparseEx_INTERFACE_HEADERS
  NlpSparseEx1.hpp
  NlpSparseEx2.hpp
  NlpSparseEx3.hpp
  NlpSparseEx4.hpp
  )

add_executable(NlpSparseEx1.exe NlpSparseEx1.cpp NlpSparseEx1Driver.cpp)
target_link_libraries(NlpSparseEx1.exe HiOp::HiOp)

add_executable(NlpSparseEx2.exe NlpSparseEx2.cpp NlpSparseEx2Driver.cpp)
target_link_libraries(NlpSparseEx2.exe HiOp::HiOp)

add_executable(NlpSparseEx3.exe NlpSparseEx3.cpp NlpSparseEx3Driver.cpp)
target_link_libraries(NlpSparseEx3.exe HiOp::HiOp)

add_executable(NlpSparseEx4.exe NlpSparseEx4.cpp NlpSparseEx4Driver.cpp)
target_link_libraries(NlpSparseEx4.exe HiOp::HiOp)

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND HIOP_USE_CUDA AND HIOP_USE_RESOLVE)
    set_source_files_properties(
      NlpSparseRajaEx2.cpp 
      NlpSparseRajaEx2Driver.cpp 
      PROPERTIES LANGUAGE CUDA
    )
  
    add_executable(NlpSparseRajaEx2.exe  NlpSparseRajaEx2Driver.cpp  NlpSparseRajaEx2.cpp)
    target_link_libraries(NlpSparseRajaEx2.exe HiOp::HiOp)
    install(TARGETS NlpSparseRajaEx2.exe DESTINATION bin)
    list(APPEND hiopSparseEx_INTERFACE_HEADERS NlpSparseRajaEx2.hpp)
  endif()
endif()

if(HIOP_BUILD_SHARED)
  add_executable(NlpSparseCEx1.exe NlpSparseCEx1.c)
  target_link_libraries(NlpSparseCEx1.exe HiOp::HiOp)
  if(HIOP_BUILD_FORTRAN_EXAMPLE)
    add_executable(NlpSparseFEx1.exe NlpSparseFEx1.F90)
    target_link_libraries(NlpSparseFEx1.exe HiOp::HiOp)
    add_executable(NlpSparseHS071.exe NlpSparseExHS071.f)
    target_link_libraries(NlpSparseHS071.exe HiOp::HiOp)
  endif()
endif()

install(
  FILES ${hiopSparseEx_INTERFACE_HEADERS}
  DESTINATION include
  )

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpSparse1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse1_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-fr" "-selfcheck")
if(HIOP_USE_CUDA)
  add_test(NAME NlpSparse1_3 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-cusolver" "-selfcheck")
endif(HIOP_USE_CUDA)
if(HIOP_USE_PARDISO)
  add_test(NAME NlpSparse1_4 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-pardiso" "-selfcheck")
endif(HIOP_USE_PARDISO)
if(HIOP_USE_GINKGO)
  add_test(NAME NlpSparse1_5 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo" "-selfcheck")
  if(HIOP_USE_CUDA)
      add_test(NAME NlpSparse1_6 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo_cuda" "-selfcheck")
  endif(HIOP_USE_CUDA)
  if(HIOP_USE_HIP)
      add_test(NAME NlpSparse1_7 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-ginkgo_hip" "-selfcheck")
  endif(HIOP_USE_HIP)
endif(HIOP_USE_GINKGO)
add_test(NAME NlpSparse2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse2_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-inertiafree" "-selfcheck")
if(HIOP_USE_CUDA)
  add_test(NAME NlpSparse2_3 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-cusolver" "-inertiafree" "-selfcheck")
endif(HIOP_USE_CUDA)
if(HIOP_USE_GINKGO)
  add_test(NAME NlpSparse2_4 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo" "-inertiafree" "-selfcheck")
  if(HIOP_USE_CUDA)
      add_test(NAME NlpSparse2_5 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo_cuda" "-inertiafree" "-selfcheck")
  endif(HIOP_USE_CUDA)
  if(HIOP_USE_HIP)
      add_test(NAME NlpSparse2_6 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-ginkgo_hip" "-inertiafree" "-selfcheck")
  endif(HIOP_USE_HIP)
endif(HIOP_USE_GINKGO)

if(HIOP_USE_RAJA AND HIOP_USE_GPU AND HIOP_USE_CUDA AND HIOP_USE_RESOLVE)
  add_test(NAME NlpSparseRaja2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseRajaEx2.exe>" "500" "-inertiafree" "-selfcheck" "-resolve_cuda_glu")
  add_test(NAME NlpSparseRaja2_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseRajaEx2.exe>" "500" "-inertiafree" "-selfcheck" "-resolve_cuda_rf")
endif()

add_test(NAME NlpSparse3_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx3.exe>" "500" "-selfcheck")
if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU )
  add_test(NAME NlpSparseCinterface COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseCEx1.exe>")
  if(HIOP_BUILD_FORTRAN_EXAMPLE)
    add_test(NAME NlpSparseFinterface COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseFEx1.exe>")
  endif()
endif()

