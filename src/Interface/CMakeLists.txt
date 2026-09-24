if(HIOP_BUILD_SHARED)
 set(hiopInterface_SRC
    hiopInterfacePrimalDecomp.cpp	
    chiopInterface.cpp
    hiopFortranInterface.c
  )
else()
 set(hiopInterface_SRC
    hiopInterfacePrimalDecomp.cpp	
  )
endif()

if(HIOP_SPARSE)
  list(APPEND hiopInterface_SRC hiopInterfaceMPS.cpp)
  if(HIOP_USE_RAJA AND HIOP_USE_GPU AND HIOP_USE_CUDA AND HIOP_USE_RESOLVE)
    set_source_files_properties(hiopInterfaceMPS.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

set(hiopInterface_INTERFACE_HEADERS
  chiopInterface.hpp
  hiopInterface.h
  hiopInterface.hpp
  hiopInterfacePrimalDecomp.hpp
  hiopVersion.hpp
  hiop_types.h
  hiopFortranInterface.h
  )

if(HIOP_SPARSE)
  list(APPEND hiopInterface_INTERFACE_HEADERS hiopInterfaceMPS.hpp)
endif()

install(
  FILES ${hiopInterface_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopInterface OBJECT ${hiopInterface_SRC})
