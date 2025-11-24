cmake_minimum_required(VERSION 3.12)

set(CARGO_OPTIONS --features cxxbridge --crate-type=staticlib --lib --verbose)
set(RUST_FLAGS "")

# Quotes mess up providing flags via CMAKE_CARGO_OPTIONS and CMAKE_RUST_FLAGS
if(CMAKE_CARGO_OPTIONS)
    string(REPLACE " " ";" CARGO_OPTIONS_LIST "${CMAKE_CARGO_OPTIONS}")
    list(APPEND CARGO_OPTIONS ${CARGO_OPTIONS_LIST})
endif()
if(CMAKE_RUST_FLAGS)
    string(REPLACE " " ";" RUST_FLAGS_LIST "${CMAKE_RUST_FLAGS}")
    list(APPEND RUST_FLAGS ${RUST_FLAGS_LIST})
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TARGET_DIR "debug")
else ()
    set(CARGO_OPTIONS ${CARGO_OPTIONS} "--release")
    set(TARGET_DIR "release")
endif ()
set(CARGO_CMD cargo rustc ${CARGO_OPTIONS})

# If the user is cross compiling to a different target, update the location of the target directory
# *Note*: Make sure to install the target with rustup, as well as setting the proper environment variables, such as CC, HOST_CC, CARGO_TARGET_<triple>_LINKER, ...
# See https://doc.rust-lang.org/cargo/reference/environment-variables.html#configuration-environment-variables
if (DEFINED ENV{CARGO_BUILD_TARGET})
    set(TARGET_DIR "$ENV{CARGO_BUILD_TARGET}/${TARGET_DIR}")
endif()

if(ENABLE_LTO)
    set(RUST_FLAGS ${RUST_FLAGS} "-Clinker-plugin-lto" "-Clinker=clang-17" "-Clink-arg=-fuse-ld=lld-17")
endif()

set(MIXT_LIB "${CMAKE_CURRENT_BINARY_DIR}/libmixt.a")
set(MIXT_CXX "${CMAKE_CURRENT_BINARY_DIR}/mixt_cxx.cpp")
set(MIXT_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/mixt_cxx.h")

add_library(mixt_cxx STATIC ${MIXT_CXX})

add_custom_command(
    OUTPUT ${MIXT_LIB}
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}" ${CARGO_CMD}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libmixt.a ${MIXT_LIB}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(
    OUTPUT ${MIXT_CXX}
    DEPENDS ${MIXT_LIB}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/mixt/src/cxx_api/mod.rs.cc ${MIXT_CXX}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/mixt/src/cxx_api/mod.rs.h ${MIXT_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(mixt_cxx ${MIXT_LIB} pthread dl)
