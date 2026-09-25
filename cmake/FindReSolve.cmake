#[[

Locates an external ReSolve installation and links target `ReSolve::ReSolve`
into `hiop_tpl`.

Users may set the following variables:

- ReSolve_DIR  Install prefix of ReSolve, or the directory containing
             ReSolveConfig.cmake.

Configuration stops with an error asking for ReSolve_DIR if ReSolve is not
found or if its version is older than HIOP_RESOLVE_MIN_VERSION.

]]

# ReSolve_DIR is the standard CMake variable consulted by find_package in
# CONFIG mode. Declaring it as a cache PATH makes it show up (and editable)
# in ccmake / cmake-gui, and lets users pass -DReSolve_DIR=<path>.
set(
  ReSolve_DIR ""
  CACHE PATH
  "Directory containing ReSolveConfig.cmake (or the ReSolve install prefix)"
)

set(HIOP_RESOLVE_MIN_VERSION "0.99.2")

# Remember user selected path: a failed CONFIG-mode find_package
# overwrites ReSolve_DIR in the cache with "ReSolve_DIR-NOTFOUND".
set(_hiop_resolve_user_dir "${ReSolve_DIR}")

# Prefer a user-specified location, then fall back to the usual search paths.
if(_hiop_resolve_user_dir)
  find_package(ReSolve ${HIOP_RESOLVE_MIN_VERSION} CONFIG QUIET
    HINTS ${_hiop_resolve_user_dir} ${_hiop_resolve_user_dir}/share/resolve/cmake
    NO_DEFAULT_PATH)
endif()
# Only fall back to the default search paths if nothing was found in the
# user-specified location. If an installation was found there but rejected
# (wrong version), report that rather than silently picking another one.
if(NOT ReSolve_FOUND AND NOT ReSolve_CONSIDERED_VERSIONS)
  find_package(ReSolve ${HIOP_RESOLVE_MIN_VERSION} CONFIG QUIET)
endif()

if(NOT ReSolve_FOUND)
  # Prompt the user for the location and stop so they can re-run configure.
  if(ReSolve_CONSIDERED_VERSIONS)
    # A ReSolve installation was found, but its version is too old.
    list(GET ReSolve_CONSIDERED_CONFIGS 0 _hiop_resolve_cfg)
    list(GET ReSolve_CONSIDERED_VERSIONS 0 _hiop_resolve_ver)
    string(CONCAT _hiop_resolve_msg
      "ReSolve version ${_hiop_resolve_ver} found at ${_hiop_resolve_cfg} "
      "is too old; HiOp requires ReSolve >= ${HIOP_RESOLVE_MIN_VERSION}.")
    unset(_hiop_resolve_cfg)
    unset(_hiop_resolve_ver)
  elseif(_hiop_resolve_user_dir)
    set(_hiop_resolve_msg
      "ReSolve was not found in ReSolve_DIR=\"${_hiop_resolve_user_dir}\".")
  else()
    set(_hiop_resolve_msg
      "ReSolve was not found in the default search paths and ReSolve_DIR is not set.")
  endif()
  message(FATAL_ERROR
    "${_hiop_resolve_msg}\n"
    "Please enter the directory of a ReSolve >= ${HIOP_RESOLVE_MIN_VERSION} "
    "installation in ReSolve_DIR, e.g.\n"
    "  cmake -DReSolve_DIR=/path/to/resolve/install ${CMAKE_SOURCE_DIR}\n"
    "or edit the ReSolve_DIR entry in ccmake/cmake-gui. ReSolve_DIR may be "
    "the install prefix or the directory containing ReSolveConfig.cmake. "
    "To build without ReSolve, configure with -DHIOP_USE_RESOLVE=OFF."
  )
elseif(NOT TARGET ReSolve::ReSolve)
  message(FATAL_ERROR
    "ReSolve was found at ${ReSolve_DIR}, but its configuration does not "
    "define the expected target ReSolve::ReSolve. Check that ReSolve_DIR "
    "points to a complete ReSolve installation."
  )
endif()

message(STATUS "Found ReSolve ${ReSolve_VERSION}: ${ReSolve_DIR}")
target_link_libraries(hiop_tpl INTERFACE ReSolve::ReSolve)
unset(_hiop_resolve_user_dir)
unset(_hiop_resolve_msg)
