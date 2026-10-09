add_subdirectory(Interface)
add_subdirectory(ExecBackends)
add_subdirectory(Optimization)
add_subdirectory(LinAlg)
add_subdirectory(Utils)

# Interface headers not already picked up by another subdirectory
set(hiop_INTERFACE_HEADERS
  ${PROJECT_BINARY_DIR}/hiop_defs.hpp
  ${PROJECT_BINARY_DIR}/FortranCInterface.hpp
  )

install(
  FILES ${hiop_INTERFACE_HEADERS}
  DESTINATION include
  )

# Create shared and static libraries from sources defined in subdirectories
foreach(libtype shared static)
  string(TOUPPER ${libtype} libtype_upper)
  if(${HIOP_BUILD_${libtype_upper}})
    add_library(hiop_${libtype}
      ${libtype_upper}
      $<TARGET_OBJECTS:hiopLinAlg>
      $<TARGET_OBJECTS:hiopOptimization>
      $<TARGET_OBJECTS:hiopUtils>
      $<TARGET_OBJECTS:hiopInterface>
      $<$<BOOL:${HIOP_BUILD_SHARED}>:$<TARGET_OBJECTS:hiopInterface>>
      )
    target_link_libraries(hiop_${libtype} PUBLIC hiop_tpl)
    target_include_directories(hiop_${libtype} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Interface>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
    set_target_properties(hiop_${libtype} PROPERTIES OUTPUT_NAME hiop)
    install(TARGETS hiop_${libtype} DESTINATION lib EXPORT hiop-targets)
  endif()
endforeach()

# Create default library pointed to by HiOp::HiOp
add_library(HiOp::HiOp ALIAS ${hiop_default_library_name})

if(HIOP_SPARSE)
  add_executable(hiop-solve-mps Drivers/Sparse/hiopSolveMPS.cpp)
  target_link_libraries(hiop-solve-mps PRIVATE HiOp::HiOp)
  set_target_properties(hiop-solve-mps PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Drivers/Sparse"
    )
  install(TARGETS hiop-solve-mps DESTINATION bin)
endif()
