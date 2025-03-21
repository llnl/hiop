set(hiopDenseEx_INTERFACE_HEADERS
  NlpDenseConsEx1.hpp
  NlpDenseConsEx2.hpp
  NlpDenseConsEx3.hpp
  NlpDenseConsEx4.hpp
  )

add_executable(NlpDenseConsEx1.exe NlpDenseConsEx1.cpp NlpDenseConsEx1Driver.cpp)
target_link_libraries(NlpDenseConsEx1.exe HiOp::HiOp)

add_executable(NlpDenseConsEx2.exe NlpDenseConsEx2.cpp NlpDenseConsEx2Driver.cpp)
target_link_libraries(NlpDenseConsEx2.exe HiOp::HiOp)

add_executable(NlpDenseConsEx3.exe NlpDenseConsEx3Driver.cpp)
target_link_libraries(NlpDenseConsEx3.exe HiOp::HiOp)

add_executable(NlpDenseConsEx4.exe NlpDenseConsEx4.cpp NlpDenseConsEx4Driver.cpp)
target_link_libraries(NlpDenseConsEx4.exe HiOp::HiOp)

if(HIOP_BUILD_FORTRAN_EXAMPLE)
  add_executable(NlpDenseConsFEx1.exe NlpDenseConsFEx1.F90)
  target_link_libraries(NlpDenseConsFEx1.exe HiOp::HiOp)
endif()

install(
  FILES ${hiopDenseEx_INTERFACE_HEADERS}
  DESTINATION include
  )

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpDenseCons1_5H  COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>"  "500" "1.0" "-selfcheck")
add_test(NAME NlpDenseCons1_5K  COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>" "5000" "1.0" "-selfcheck")
add_test(NAME NlpDenseCons1_25K_InfDim COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>" "25000" "1.0" "-selfcheck" "-use_L2")
add_test(NAME NlpDenseCons1_50K COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>" "50000" "1.0" "-selfcheck")
if(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons1_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpDenseConsEx1.exe>" "50000" "1.0" "-selfcheck")
endif(HIOP_USE_MPI)
add_test(NAME NlpDenseCons2_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"   "500" "-selfcheck")
add_test(NAME NlpDenseCons2_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"  "5000" "-selfcheck")
add_test(NAME NlpDenseCons2_UN_5K COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"  "5000" "-unconstrained" "-selfcheck")
add_test(NAME NlpDenseCons3_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>"   "500" "-selfcheck")
add_test(NAME NlpDenseCons3_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>"  "5000" "-selfcheck")
add_test(NAME NlpDenseCons3_50K   COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>" "50000" "-selfcheck")
if(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons3_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpDenseConsEx3.exe>" "50000" "-selfcheck")
endif(HIOP_USE_MPI)

add_test(NAME NlpDenseCons4       COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx4.exe>" "-selfcheck")
if(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons4_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpDenseConsEx4.exe>" "-selfcheck")
endif(HIOP_USE_MPI)

if(HIOP_WITH_VALGRIND_TESTS)
  string(REPLACE ";" " " runcmd_str "${RUNCMD}")
  add_test(
    NAME NlpDenseCons1_5H_Valgrind
    COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:NlpDenseConsEx1.exe> 500 1.0 -selfcheck"
    )
endif()

if(HIOP_BUILD_FORTRAN_EXAMPLE)
  add_test(NAME NlpDenseFinterface COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsFEx1.exe>")
endif()

