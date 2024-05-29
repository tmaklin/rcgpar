cmake_minimum_required(VERSION 2.8.12)
project(rcgpar)

include(FetchContent)

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

## Do an out-of-source build
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## Check dependencies
### OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set(RCGPAR_OPENMP_SUPPORT 1)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -D_GLIBCXX_PARALLEL")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
else()
  set(RCGPAR_OPENMP_SUPPORT 0)
endif()

## seamat
if (DEFINED CMAKE_SEAMAT_HEADERS)
  message(STATUS "seamat headers provided in: ${CMAKE_SEAMAT_HEADERS}")
else()
  message(STATUS "Downloading seamat headers...")
  FetchContent_Declare(seamat
    GIT_REPOSITORY    https://github.com/tmaklin/seamat.git
    GIT_TAG           v0.2.1
    PREFIX            "external"
    SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/external/seamat"
    BUILD_IN_SOURCE   0
    CMAKE_ARGS        -D CMAKE_BUILD_TESTS=0
    BUILD_COMMAND     ""
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND   ""
    )
  FetchContent_MakeAvailable(seamat)
  set(CMAKE_SEAMAT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/external/seamat/include)
endif()
include_directories(${CMAKE_SEAMAT_HEADERS})

## Set C++17 support depending on cmake version
if (${CMAKE_MAJOR_VERSION} GREATER 2 AND ${CMAKE_MINOR_VERSION} GREATER 0)
  set (CMAKE_CXX_STANDARD 17)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
else()
  add_compile_options(-std=c++17)
endif()

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
  message( STATUS "rcgpar version: ${_BUILD_VERSION}")
else()
  message(STATUS "GIT not found, version numbering disabled")
  set(_BUILD_VERSION "unknown")
endif()
string(TIMESTAMP _BUILD_TIMESTAMP)

## Download googletest if building tests
if(CMAKE_BUILD_TESTS)
  if (DEFINED CMAKE_GOOGLETEST_HEADERS)
    message(STATUS "googletest headers provided in: ${CMAKE_GOOGLETEST_HEADERS}")
  else()
    FetchContent_Declare(googletest
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           release-1.11.0
      PREFIX            "external"
      SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/external/googletest"
      BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/external/googletest"
      BUILD_IN_SOURCE   0
      CMAKE_ARGS      -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
		      -D "CMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
		      -D "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
		      -D "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
		      -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    INSTALL_COMMAND   ""
    )
  enable_testing()
  add_executable(runUnitTests
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/rcg_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/rcgpar_omp_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/util_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/src/unittest_common_inputs.cpp)
  add_test(runUnitTests runUnitTests)
  endif()
endif()

## BitMagic
if (DEFINED CMAKE_BITMAGIC_HEADERS)
  message(STATUS "BitMagic headers provided in: ${CMAKE_BITMAGIC_HEADERS}")
else()
  message(FATAL_ERROR "Provide BitMagic headers with -DCMAKE_BITMAGIC_HEADERS")
endif()
include_directories(${CMAKE_BITMAGIC_HEADERS})

## Generate a version.h file containing build version and timestamp
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/rcgpar_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/rcgpar_version.h @ONLY)
## Configure OpenMP if it's supported on the system.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/rcgpar_openmp_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/include/rcgpar_openmp_config.hpp @ONLY)

## Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/test/include)

## Libraries and executables
## utils
add_library(rcgutils ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp)

### OpenMP parallellized rcg
add_library(rcgomp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rcg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rcgpar.cpp)

## Link unit tests with rcgomp
if(CMAKE_BUILD_TESTS)
  target_link_libraries(runUnitTests rcgomp rcgutils gtest gtest_main)
endif()

# Link libraries
if (OPENMP_FOUND)
  if (MPI_FOUND)
    target_link_libraries(rcgmpi OpenMP::OpenMP_CXX ${LIBRARY_OUTPUT_PATH}/libbigmpi.a) ## Hybrid parallellization
  endif()
  if (BUILD_TESTS)
    target_link_libraries(rcgomp rcgutils runUnitTests OpenMP::OpenMP_CXX)
  else()
    target_link_libraries(rcgomp rcgutils OpenMP::OpenMP_CXX)
  endif()
endif()
