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

# ---------------------------------------------------------------------------
# Perf-punchlist feature toggles (docs/BOLT_PERF_PUNCHLIST.md).
# Each is a compile-time switch per the multiple-impls-one-default rule:
# one default implementation ships active, alternatives live behind these
# toggles with zero runtime branch cost.
# ---------------------------------------------------------------------------

# Huge pages (2 MB) for arenas ≥16 MB.  See D1.
#   Linux: default ON — `mmap(MAP_HUGETLB)` + `madvise(MADV_HUGEPAGE)`.
#   Windows: default OFF — `VirtualAlloc(MEM_LARGE_PAGES)` needs
#            `SeLockMemoryPrivilege` which isn't granted by default.
if(WIN32)
    option(BOLT_ENABLE_HUGE_PAGES "Opt-in 2MB huge pages for large arenas" OFF)
else()
    option(BOLT_ENABLE_HUGE_PAGES "Opt-in 2MB huge pages for large arenas" ON)
endif()

# NUMA-aware per-socket channels + worker affinity.  See E1.
option(BOLT_ENABLE_NUMA
    "Enable NUMA-aware per-socket channel pools and worker affinity" OFF)

# SwissTable layout alternative.  See C3.
#   FLAT         — single flat 16-slot SIMD groups (default; current code)
#   INTERLEAVED  — row-major groups+keys; wins on >L3 tables (TLB-friendly)
set(BOLT_SWISS_LAYOUT "FLAT" CACHE STRING
    "SwissTable physical layout: FLAT|INTERLEAVED")
set_property(CACHE BOLT_SWISS_LAYOUT PROPERTY STRINGS FLAT INTERLEAVED)

# Hash mixer tier.  See A3 + docs/research/hash-functions.md.
# All three variants are always compiled into the binary as named
# functions in bolt_hash.h (keep-code-paths-for-JIT-later policy); this
# switch only selects which one the default `swiss_mix` symbol resolves
# to.
#   WYHASH3 — wyhash-family 3-op mix (xor-shift / multiply / xor-shift).
#             Current ship-default; wins on 1BRC-class workloads.
#   MURMUR3 — Murmur3 finalizer (6 ops). Slower but stronger avalanche.
#   XXH3    — xxh3-64 finalizer shape. Stronger distribution than
#             WYHASH3 at similar op count.
set(BOLT_HASH_TIER "WYHASH3" CACHE STRING
    "Hash mixer for SwissTable/hashjoin: WYHASH3|XXH3|MURMUR3")
set_property(CACHE BOLT_HASH_TIER PROPERTY STRINGS WYHASH3 XXH3 MURMUR3)

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
bolt_apply_feature_toggles(bolt_core)

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
# bolt::io — compiled static library (I/O-adjacent pure-compute primitives:
# CRC32C software fallback, varints). No actual I/O — keeps Bolt's no-IO
# invariant intact. The .cpp hosts the 8 KB slicing-by-8 CRC32C tables so
# they link once per binary.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_IO "Build bolt::io compiled library" ${BOLT_IS_TOPLEVEL})

if(BOLT_BUILD_IO)
    add_library(bolt_io STATIC src/io/bolt_crc32c.cpp)
    add_library(bolt::io ALIAS bolt_io)
    target_link_libraries(bolt_io PUBLIC bolt::core)
    target_include_directories(bolt_io PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_io)
    bolt_apply_simd(bolt_io ${BOLT_SIMD_TIER})
    target_link_libraries(bolt INTERFACE bolt::io)
endif()

# ---------------------------------------------------------------------------
# bolt::ribbon — Wave 20D.4 banded GF(2) Ribbon filter (Dillinger 2021).
# Pure-compute (no I/O); we promote the body to a single TU so callers
# don't have to redefine the construction kernel inline at every consumer.
# Header `bolt/bolt_ribbon.h` declares the API; this static lib carries
# `build` + `might_contain`.
# ---------------------------------------------------------------------------
add_library(bolt_ribbon STATIC src/bolt_ribbon.cpp)
add_library(bolt::ribbon ALIAS bolt_ribbon)
target_link_libraries(bolt_ribbon PUBLIC bolt::core)
target_include_directories(bolt_ribbon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
bolt_apply_hardening(bolt_ribbon)
bolt_apply_simd(bolt_ribbon ${BOLT_SIMD_TIER})
target_link_libraries(bolt INTERFACE bolt::ribbon)

# ---------------------------------------------------------------------------
# bolt::parse — compiled static library (ASCII numeric parsers).
# Header-only rules don't apply here: we want the magic-multiplier code
# in a single TU so it links once, not per consumer.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_PARSE "Build bolt::parse compiled library" ${BOLT_IS_TOPLEVEL})

if(BOLT_BUILD_PARSE)
    add_library(bolt_parse STATIC src/parse/bolt_ascii.cpp src/parse/bolt_json.cpp)
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
    add_library(bolt_ingest STATIC
        src/ingest/bolt_csv.cpp
        src/ingest/bolt_parquet_meta.cpp)
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

set(_bolt_install_targets bolt_core bolt_kernels bolt_join bolt_wire bolt_arrow bolt bolt_ribbon)
if(BOLT_BUILD_IO)
    list(APPEND _bolt_install_targets bolt_io)
endif()
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
