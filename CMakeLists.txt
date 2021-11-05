cmake_minimum_required(VERSION 2.8.12)
project(rcg-mpi)

## Determine build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) ## Build Release by default
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -g -Wall -Wextra -Wpedantic")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ffast-math -funroll-loops")
  endif()
endif()

## Check dependencies
find_package(OpenMP)
if (OPENMP_FOUND)
  set(RCGMPI_OPENMP_SUPPORT 1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
else()
  set(RCGMPI_OPENMP_SUPPORT 0)
endif()

find_package(MPI)
include_directories(MPI_C_INCLUDE_DIRS)

## Set C++11 support depending on cmake version
if (${CMAKE_MAJOR_VERSION} GREATER 2 AND ${CMAKE_MINOR_VERSION} GREATER 0)
  set (CMAKE_CXX_STANDARD 11)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  add_compile_options(-std=c++11)
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## Get version number from git tags
find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags
    WORKING_DIRECTORY "${local_dir}"
    OUTPUT_VARIABLE _BUILD_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message( STATUS "rcg-mpi version: ${_BUILD_VERSION}")
else()
  message(STATUS "GIT not found, version numbering disabled")
  set(_BUILD_VERSION "unknown")
endif()
string(TIMESTAMP _BUILD_TIMESTAMP)

## Generate a version.h file containing build version and timestamp
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/version.h.in ${CMAKE_BINARY_DIR}/include/version.h @ONLY)
## Configure OpenMP if it supported on the system.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/openmp_config.hpp.in ${CMAKE_BINARY_DIR}/include/openmp_config.hpp @ONLY)

## Includes
include_directories(${CMAKE_SOURCE_DIR}/include
${CMAKE_BINARY_DIR}/include)

## Libraries
add_library(rcgomp
  ${CMAKE_SOURCE_DIR}/src/rcg.cpp
  ${CMAKE_SOURCE_DIR}/src/rcg_omp.cpp
  ${CMAKE_SOURCE_DIR}/src/Matrix_omp.cpp)

## Executables
add_executable(rcg-omp-test
  ${CMAKE_SOURCE_DIR}/src/test_util.cpp
  ${CMAKE_SOURCE_DIR}/src/test_omp.cpp)
add_executable(rcg-mpi-test
  ${CMAKE_SOURCE_DIR}/src/test_util.cpp
  ${CMAKE_SOURCE_DIR}/src/test_mpi.cpp)

target_link_libraries(rcg-omp-test rcgomp)
target_link_libraries(rcg-mpi-test rcgomp)

# Link libraries
if (OPENMP_FOUND)
  target_link_libraries(rcgomp OpenMP::OpenMP_CXX)
  target_link_libraries(rcg-omp-test OpenMP::OpenMP_CXX)
endif()
