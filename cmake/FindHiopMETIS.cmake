
#[[

Exports target `METIS`

Users may set the following variables:

- METIS_DIR: METIS installation prefix

]]

set(METIS_DIR "" CACHE PATH "METIS installation prefix")

find_library(METIS_LIBRARY
  NAMES
  metis
  PATHS
  ${METIS_DIR} $ENV{METIS_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(METIS_LIBRARY)
  get_filename_component(METIS_LIBRARY_DIR ${METIS_LIBRARY} DIRECTORY)
endif()

find_path(METIS_INCLUDE_DIR
  NAMES
  metis.h
  PATHS
  ${METIS_DIR} $ENV{METIS_DIR} ${METIS_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include)

if(METIS_LIBRARY)
  message(STATUS "Found metis include: ${METIS_INCLUDE_DIR}")
  message(STATUS "Found metis library: ${METIS_LIBRARY}")
  add_library(METIS INTERFACE)
  target_link_libraries(METIS INTERFACE ${METIS_LIBRARY})
  target_include_directories(METIS INTERFACE ${METIS_INCLUDE_DIR})
  message(STATUS "Found METIS library: ${METIS_LIBRARY}")
  install(TARGETS METIS EXPORT hiop-targets)
else()
  message(STATUS "METIS was not found.")
endif()

set(METIS_INCLUDE_DIR CACHE PATH "Path to metis.h")
set(METIS_LIBRARY CACHE PATH "Path to metis library")

