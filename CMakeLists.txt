cmake_minimum_required(VERSION 3.20)

project(bolt
    VERSION     0.1.0
    DESCRIPTION "Zero-dependency columnar execution core"
    LANGUAGES   CXX
)

# ---------------------------------------------------------------------------
# Top-level detection + options
# ---------------------------------------------------------------------------
if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(BOLT_IS_TOPLEVEL ON)
else()
    set(BOLT_IS_TOPLEVEL OFF)
endif()

option(BOLT_BUILD_TESTS        "Build bolt unit tests"     ${BOLT_IS_TOPLEVEL})
option(BOLT_BUILD_BENCHMARKS   "Build bolt benchmarks"     ${BOLT_IS_TOPLEVEL})
option(BOLT_WARNINGS_AS_ERRORS "Treat warnings as errors"  OFF)
option(BOLT_ENABLE_NATIVE      "Compile benches with -march=native / /arch:AVX2" OFF)

# SIMD tier for bolt::core (propagates to every consumer).
#   NATIVE   — MSVC /arch:AVX2, GCC/Clang -march=native (best on build host)
#   AVX2     — MSVC /arch:AVX2, GCC/Clang -mavx2 -msse4.2 (portable modern x86)
#   SSE42    — GCC/Clang -msse4.2; MSVC no flag (baseline = SSE2, scalar path)
#   PORTABLE — no SIMD flag; reproducible cross-machine builds (scalar on MSVC)
# `BOLT_ENABLE_NATIVE` remains for back-compat and only affects benchmark
# tuning (see bolt_apply_bench_tuning). `BOLT_SIMD_TIER` is the authoritative
# knob for core ISA selection.
set(BOLT_SIMD_TIER "NATIVE" CACHE STRING
    "SIMD ISA tier for bolt::core: NATIVE|AVX512|AVX2|SSE42|PORTABLE")
set_property(CACHE BOLT_SIMD_TIER PROPERTY STRINGS NATIVE AVX512 AVX2 SSE42 PORTABLE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(BoltCompileOptions)

set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# Modules — each is a header-only INTERFACE library, namespaced under bolt::
# ---------------------------------------------------------------------------
# Core: types, arena, channel, column, branchless, scheduler, port.
add_library(bolt_core INTERFACE)
add_library(bolt::core ALIAS bolt_core)
target_include_directories(bolt_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(bolt_core INTERFACE cxx_std_20)
bolt_apply_hardening(bolt_core)
bolt_apply_simd(bolt_core ${BOLT_SIMD_TIER})

# Placeholder sibling modules. Populated as phases land.
foreach(mod IN ITEMS kernels join wire arrow)
    add_library(bolt_${mod} INTERFACE)
    add_library(bolt::${mod} ALIAS bolt_${mod})
    target_link_libraries(bolt_${mod} INTERFACE bolt::core)
endforeach()

# Umbrella — pulls core + every sibling. Most consumers link this.
add_library(bolt INTERFACE)
add_library(bolt::bolt ALIAS bolt)
target_link_libraries(bolt INTERFACE
    bolt::core bolt::kernels bolt::join bolt::wire bolt::arrow
)

# ---------------------------------------------------------------------------
# bolt::parse — compiled static library (ASCII numeric parsers).
# Header-only rules don't apply here: we want the magic-multiplier code
# in a single TU so it links once, not per consumer.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_PARSE "Build bolt::parse compiled library" ${BOLT_IS_TOPLEVEL})

if(BOLT_BUILD_PARSE)
    add_library(bolt_parse STATIC src/parse/bolt_ascii.cpp)
    add_library(bolt::parse ALIAS bolt_parse)
    target_link_libraries(bolt_parse PUBLIC bolt::core)
    target_include_directories(bolt_parse PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_parse)
    bolt_apply_simd(bolt_parse ${BOLT_SIMD_TIER})
    target_link_libraries(bolt INTERFACE bolt::parse)
endif()

# ---------------------------------------------------------------------------
# bolt::ingest — compiled static library (CSV loader).
# Delegates numeric parsing to bolt::parse; ships as a TU so the parser
# tables link once per binary instead of per consumer.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_INGEST "Build bolt::ingest compiled library" ${BOLT_IS_TOPLEVEL})

if(BOLT_BUILD_INGEST)
    if(NOT BOLT_BUILD_PARSE)
        message(FATAL_ERROR "BOLT_BUILD_INGEST requires BOLT_BUILD_PARSE=ON")
    endif()
    add_library(bolt_ingest STATIC src/ingest/bolt_csv.cpp)
    add_library(bolt::ingest ALIAS bolt_ingest)
    target_link_libraries(bolt_ingest PUBLIC bolt::core bolt::parse)
    target_include_directories(bolt_ingest PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_ingest)
    bolt_apply_simd(bolt_ingest ${BOLT_SIMD_TIER})
    target_link_libraries(bolt INTERFACE bolt::ingest)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BOLT_BUILD_TESTS)
    enable_testing()
    include(BoltGTest)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(BOLT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Install + export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/bolt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

set(_bolt_install_targets bolt_core bolt_kernels bolt_join bolt_wire bolt_arrow bolt)
if(BOLT_BUILD_PARSE)
    list(APPEND _bolt_install_targets bolt_parse)
endif()
if(BOLT_BUILD_INGEST)
    list(APPEND _bolt_install_targets bolt_ingest)
endif()
install(TARGETS ${_bolt_install_targets}
    EXPORT bolt-targets
)

install(EXPORT bolt-targets
    FILE      bolt-targets.cmake
    NAMESPACE bolt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bolt
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-config-version.cmake
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bolt-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bolt
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/bolt-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bolt
)
