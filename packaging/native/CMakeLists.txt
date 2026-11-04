cmake_minimum_required(VERSION 3.26)
project(muse_glimmer_package LANGUAGES C CXX)

if(NOT EXECUTORCH_SOURCE_DIR)
  message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR must name the ExecuTorch checkout")
endif()

# Consume upstream targets directly, without building/installing the full SDK.
set(CMAKE_SKIP_INSTALL_RULES ON)
add_subdirectory("${EXECUTORCH_SOURCE_DIR}" executorch EXCLUDE_FROM_ALL)

# The standalone model looks up installed packages. CMake's package redirects
# let it consume the targets above without installing an SDK or editing upstream.
foreach(package IN ITEMS executorch gflags)
  file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${package}-config.cmake"
       "if(NOT TARGET ${package})\n  message(FATAL_ERROR \"Missing ${package} target\")\nendif()\n")
endforeach()
add_subdirectory(
  "${EXECUTORCH_SOURCE_DIR}/examples/models/muse-glimmer"
  examples/models/muse-glimmer-batching EXCLUDE_FROM_ALL
)
