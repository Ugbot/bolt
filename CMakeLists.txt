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
# Default ON (not ${BOLT_IS_TOPLEVEL}): this target was unconditional before the
# switch existed, so ON keeps every consumer building exactly as it did. The
# switch is here so a consumer that only wants bolt::core can decline the cost.
option(BOLT_BUILD_RIBBON "Build bolt::ribbon banded GF(2) Ribbon filter" ON)

if(BOLT_BUILD_RIBBON)
    add_library(bolt_ribbon STATIC src/bolt_ribbon.cpp)
    add_library(bolt::ribbon ALIAS bolt_ribbon)
    target_link_libraries(bolt_ribbon PUBLIC bolt::core)
    target_include_directories(bolt_ribbon PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_ribbon)
    bolt_apply_simd(bolt_ribbon ${BOLT_SIMD_TIER})
    target_link_libraries(bolt INTERFACE bolt::ribbon)
endif()

# ---------------------------------------------------------------------------
# bolt::parse — compiled static library (ASCII numeric parsers).
# Header-only rules don't apply here: we want the magic-multiplier code
# in a single TU so it links once, not per consumer.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_PARSE "Build bolt::parse compiled library" ${BOLT_IS_TOPLEVEL})

if(BOLT_BUILD_PARSE)
    add_library(bolt_parse STATIC src/parse/bolt_ascii.cpp src/parse/bolt_json.cpp src/parse/bolt_jinja.cpp src/parse/bolt_jinja_ast.cpp src/parse/bolt_jinja_stmt.cpp src/parse/bolt_jinja_value.cpp src/parse/bolt_jinja_eval.cpp)
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
# bolt::vulkan — device/queue/memory scaffolding behind bolt::compute's
# reserved Vulkan device slot (BLLM-46). Compiled (not header-only) for the
# same reason as bolt::parse: Vulkan's C API + loader linkage doesn't suit
# a header-only inline function. Gated on CMake actually finding a Vulkan
# SDK (via find_package's standard FindVulkan module, which consults the
# VULKAN_SDK env var the LunarG installer sets) -- if none is found, this
# target is simply not built, and bolt::compute's Vulkan slot stays a
# hard-failing stub exactly as it is today. Never required for a build to
# succeed; every other consumer keeps working CPU-only regardless.
# ---------------------------------------------------------------------------
find_package(Vulkan QUIET)

# Default follows the same rule as every other optional compiled target
# (bolt::io / bolt::parse / bolt::ingest): ON only when bolt is the TOP-LEVEL
# project -- i.e. someone is building/testing bolt itself -- and the SDK is
# actually present. Availability is NOT need: a consumer that links bolt::bolt
# for bolt::core must never silently compile bolt::vulkan + a SPIR-V shader
# pipeline just because the dev box happens to have the LunarG SDK installed.
# Consumers that DO want it opt in explicitly with -DBOLT_BUILD_VULKAN=ON.
if(BOLT_IS_TOPLEVEL AND Vulkan_FOUND)
    set(BOLT_VULKAN_DEFAULT ON)
else()
    set(BOLT_VULKAN_DEFAULT OFF)
endif()
option(BOLT_BUILD_VULKAN "Build bolt::vulkan device/queue/memory scaffolding"
       ${BOLT_VULKAN_DEFAULT})

if(BOLT_BUILD_VULKAN AND NOT Vulkan_FOUND)
    message(WARNING "BOLT_BUILD_VULKAN requested but no Vulkan SDK found (set VULKAN_SDK) -- skipping bolt::vulkan")
    set(BOLT_BUILD_VULKAN OFF)
endif()

if(BOLT_BUILD_VULKAN)
    add_library(bolt_vulkan STATIC src/bolt_vulkan.cpp)
    add_library(bolt::vulkan ALIAS bolt_vulkan)
    target_link_libraries(bolt_vulkan PUBLIC bolt::core Vulkan::Vulkan)
    target_include_directories(bolt_vulkan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_vulkan)
    target_link_libraries(bolt INTERFACE bolt::vulkan)

    # -------------------------------------------------------------------
    # BLLM-47: matmul+dequant compute shaders (F32/Int8/Int4/Int2), written
    # from scratch for OUR exact quant layout (low-nibble-first Int4,
    # separate per-row scale array) after reading llama.cpp's real
    # ggml-vulkan mul_mat_vec.comp design for transferable technique facts
    # (MIT-licensed; see the repo-root NOTICE file for the full
    # attribution) -- see each .comp file's own header comment for the
    # byte-packing details. Compiled to SPIR-V via glslc
    # at build time, then embedded as a C++ uint32_t array (glslc
    # -mfmt=c) so bolt::vulkan needs no runtime filesystem access to load
    # a shader -- CompileShaderToHeader.cmake does the wrapping.
    # -------------------------------------------------------------------
    find_program(Vulkan_GLSLC_EXECUTABLE glslc
                 HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(WARNING "BOLT_BUILD_VULKAN is ON but glslc was not found -- BLLM-47 shaders will not be embedded")
    else()
        set(_bolt_vulkan_shader_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/bolt/vulkan_shaders")
        file(MAKE_DIRECTORY "${_bolt_vulkan_shader_gen_dir}")
        set(_bolt_vulkan_shader_names matmul_dequant_f32 matmul_dequant_int8 matmul_dequant_int4 matmul_dequant_int2
                                       matmul_dequant_int8act_int8 matmul_dequant_int8act_int4
                                       matmul_dequant_int8act_int8_batched matmul_dequant_int8act_int4_batched)
        set(_bolt_vulkan_shader_headers "")
        foreach(_shader_name ${_bolt_vulkan_shader_names})
            set(_src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_shader_name}.comp")
            set(_out "${_bolt_vulkan_shader_gen_dir}/${_shader_name}.h")
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND ${CMAKE_COMMAND}
                        -DGLSLC=${Vulkan_GLSLC_EXECUTABLE}
                        -DSRC=${_src}
                        -DOUT=${_out}
                        -DVAR_NAME=k${_shader_name}_spv
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaderToHeader.cmake"
                DEPENDS "${_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaderToHeader.cmake"
                COMMENT "Compiling ${_shader_name}.comp -> embedded SPIR-V header"
                VERBATIM
            )
            list(APPEND _bolt_vulkan_shader_headers "${_out}")
        endforeach()
        add_custom_target(bolt_vulkan_shaders DEPENDS ${_bolt_vulkan_shader_headers})
        add_dependencies(bolt_vulkan bolt_vulkan_shaders)
        target_include_directories(bolt_vulkan PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
    endif()
endif()

# ---------------------------------------------------------------------------
# bolt::rocm -- ROCm/HIP matmul+dequant kernel core behind bolt::compute's
# reserved ROCm device slot (BLLM-78).
#
# Checked 2026-07-16: no HIP_PATH, no hipcc, no ROCm install dir found on
# this dev box (AMD Ryzen AI Max+ 395 / Radeon 8060S, gfx1151) -- BLLM-90
# then found AMD's Windows HIP SDK officially lists gfx1151 as supported
# (reversing that finding); HIP SDK 7.1 installed 2026-07-17. bolt::rocm's
# kernel WAS validated for real against this box's actual GPU
# (F32/Int8/Int4/Int2, including the multi-row misaligned-stride
# regression case -- 50/50 checks pass) by invoking hipcc directly.
#
# UNLIKE bolt::vulkan above, this option DOES NOT auto-enable just because
# hipcc/clang++ are found (BLLM-92 finding): CMake's enable_language(HIP)
# does not currently link cleanly on Windows against this HIP SDK -- the
# HIP-language link step inherits an MSVC-style linker flag (/machine:x64)
# that plain clang++ (CMAKE_HIP_COMPILER) rejects, even with
# CMAKE_HIP_ARCHITECTURES/CMAKE_HIP_COMPILER_ROCM_ROOT/--rocm-path all
# supplied correctly (the COMPILE step succeeds; only the final link
# fails). enable_language(HIP) has no graceful-failure mode -- it hard-
# errors the ENTIRE configure, not just this target -- so auto-enabling
# here would break bolt's default `cmake -S . -B build` for anyone on a
# machine with HIP_PATH set (which the AMD installer sets as a PERSISTENT
# SYSTEM environment variable, confirmed on this exact box: every future
# shell has it, not just the one that ran the installer). Requires
# EXPLICIT `-DBOLT_BUILD_ROCM=ON` until the Windows CMake+HIP linking gap
# is resolved (a newer CMake/ROCm release, or a working
# CMAKE_HIP_LINK_EXECUTABLE override) -- passing it today will still fail
# the configure with the /machine:x64 error, this flag is a placeholder
# for whoever picks up that fix, not a currently-working opt-in.
# ---------------------------------------------------------------------------
find_program(BOLT_HIPCC_EXECUTABLE NAMES hipcc
             HINTS "$ENV{HIP_PATH}/bin" "$ENV{ROCM_PATH}/bin" /opt/rocm/bin)
# CMake's HIP language support rejects the hipcc WRAPPER script as
# CMAKE_HIP_COMPILER ("Use Clang directly, or let CMake pick a default" --
# found 2026-07-17 getting this working for real against this box's newly
# installed HIP SDK 7.1). The real compiler is clang++ in that same bin/
# directory (ROCm's HIP toolchain bundles its own Clang) -- hipcc itself is
# just a driver script around it. Probe for hipcc (a reliable "is a HIP SDK
# here at all" signal, matching how a human would check) but point CMake at
# clang++ directly for the actual compiler.
find_program(BOLT_HIP_CLANGXX_EXECUTABLE NAMES clang++
             HINTS "$ENV{HIP_PATH}/bin" "$ENV{ROCM_PATH}/bin" /opt/rocm/bin)

# Deliberately ALWAYS OFF by default, even when BOLT_IS_TOPLEVEL and both
# hipcc/clang++ are found (stronger than BOLT_BUILD_VULKAN's
# "top-level AND SDK found" rule above): confirmed 2026-07-17 on this exact
# box that enable_language(HIP) does not currently link cleanly on Windows
# against HIP SDK 7.1 even for a top-level bolt build with a real hipcc +
# clang++ present -- the HIP-language link step inherits an MSVC-style
# linker flag (/machine:x64) that plain clang++ rejects (the COMPILE step
# succeeds; only the final link fails; see the header comment above for the
# full diagnosis). enable_language(HIP) has no graceful-failure mode -- it
# hard-errors the ENTIRE configure, not just this target -- so any
# auto-enable rule here (even a narrower "top-level only" one) risks
# breaking bolt's own default `cmake -S . -B build` for whoever next builds
# it standalone on a box with a HIP SDK installed (which the AMD installer
# sets up via a PERSISTENT SYSTEM HIP_PATH env var, so this isn't a
# one-time-shell concern). Requires explicit `-DBOLT_BUILD_ROCM=ON` --
# which today will still fail the configure with the /machine:x64 error;
# this option is a placeholder for whoever fixes that Windows CMake+HIP
# linking gap (a newer CMake/ROCm release, or a working
# CMAKE_HIP_LINK_EXECUTABLE override), not a currently-working opt-in.
# Also note: a bare `set(BOLT_BUILD_ROCM ON)` after option() would silently
# override an explicit -DBOLT_BUILD_ROCM=OFF (set() beats the cache) --
# never do that; this block only ever narrows via the warning-guard below,
# never widens.
option(BOLT_BUILD_ROCM "Build bolt::rocm ROCm/HIP matmul+dequant kernel core (see the known Windows linking gap above before enabling)" OFF)

if(BOLT_BUILD_ROCM AND NOT (BOLT_HIPCC_EXECUTABLE AND BOLT_HIP_CLANGXX_EXECUTABLE))
    message(WARNING "BOLT_BUILD_ROCM requested but no hipcc/clang++ found (set HIP_PATH/ROCM_PATH) -- skipping bolt::rocm")
    set(BOLT_BUILD_ROCM OFF)
endif()

if(BOLT_BUILD_ROCM)
    # BLLM-99 FIX (2026-07-17): bypass CMake's native enable_language(HIP)
    # entirely, rather than fight it further -- confirmed with a minimal
    # isolated repro that CMake's own HIP-language integration is what's
    # broken here, not cross-toolchain linking itself:
    #   - Ninja generator: CMake's Windows-Clang-HIP.cmake platform file
    #     hard-refuses outright ("The current configuration mixes Clang and
    #     MSVC ... This is not supported") the moment CXX=MSVC and HIP=Clang
    #     coexist in one project -- a deliberate CMake-side same-toolchain-
    #     family check, not a bug to work around via flags.
    #   - VS generator: gets further (compile succeeds) before the HIP-
    #     language LINK step inherits an MSVC-style /machine:x64 flag plain
    #     clang++ rejects (the original symptom this ticket was filed for).
    #   - Proven separately (isolated repro, this box, 2026-07-17): a HIP
    #     kernel compiled DIRECTLY via clang++ (no enable_language(HIP) at
    #     all) produces a plain COFF .obj that MSVC's own link.exe links
    #     cleanly against MSVC-compiled host code -- ran for real on this
    #     box's GPU (hipMalloc/hipMemcpy/kernel dispatch/hipDeviceSynchronize,
    #     correct result). The object files and toolchains are NOT
    #     incompatible; only CMake's own HIP-language machinery's assumptions
    #     are.
    #
    # Fix: an add_custom_command invokes clang++ directly (same flags
    # BLLM-92's proven-working hipcc-direct validation used), producing a
    # plain .obj, archived into a .lib via MSVC's own lib.exe (already on
    # PATH -- bolt's build already requires it for cl.exe/link.exe), exposed
    # as an IMPORTED STATIC library the rest of the (MSVC-built) project
    # links against normally. No enable_language(HIP) call anywhere.
    if(DEFINED ENV{ROCM_PATH})
        set(_bolt_rocm_root "$ENV{ROCM_PATH}")
    else()
        set(_bolt_rocm_root "$ENV{HIP_PATH}")
    endif()
    # clang++ mishandles a space-containing --rocm-path on Windows
    # (confirmed BLLM-90/92) -- resolve the 8.3 short path so
    # "Program Files" never appears in the flag's value.
    if(WIN32)
        execute_process(
            COMMAND powershell -NoProfile -Command
                "(New-Object -ComObject Scripting.FileSystemObject).GetFolder('${_bolt_rocm_root}').ShortPath"
            OUTPUT_VARIABLE _bolt_rocm_root_short
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _bolt_rocm_short_result)
        if(_bolt_rocm_short_result EQUAL 0 AND _bolt_rocm_root_short)
            set(_bolt_rocm_root "${_bolt_rocm_root_short}")
        endif()
    endif()

    set(_bolt_rocm_obj "${CMAKE_CURRENT_BINARY_DIR}/bolt_rocm_hip.obj")
    set(_bolt_rocm_lib "${CMAKE_CURRENT_BINARY_DIR}/bolt_rocm_hip.lib")

    find_program(BOLT_LIB_EXE_EXECUTABLE NAMES lib lib.exe)
    if(NOT BOLT_LIB_EXE_EXECUTABLE)
        message(WARNING "BOLT_BUILD_ROCM: lib.exe not found on PATH (needed to archive the "
                         "clang++-compiled bolt_rocm.cpp into a linkable .lib) -- skipping bolt::rocm. "
                         "Run from a Developer/VS environment (vcvarsall.bat) so lib.exe is on PATH.")
        set(BOLT_BUILD_ROCM OFF)
    endif()
endif()

if(BOLT_BUILD_ROCM)
    add_custom_command(
        OUTPUT "${_bolt_rocm_obj}"
        COMMAND "${BOLT_HIP_CLANGXX_EXECUTABLE}" -x hip -c
                "${CMAKE_CURRENT_SOURCE_DIR}/src/bolt_rocm.cpp"
                -o "${_bolt_rocm_obj}"
                --offload-arch=gfx1151
                "--rocm-path=${_bolt_rocm_root}"
                -O2 -DNDEBUG -std=c++20
                -fms-runtime-lib=dll
                -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/bolt_rocm.cpp"
        COMMENT "BLLM-99: compiling bolt_rocm.cpp via clang++ HIP directly (bypassing enable_language(HIP))"
        VERBATIM)
    add_custom_command(
        OUTPUT "${_bolt_rocm_lib}"
        COMMAND "${BOLT_LIB_EXE_EXECUTABLE}" /nologo "/OUT:${_bolt_rocm_lib}" "${_bolt_rocm_obj}"
        DEPENDS "${_bolt_rocm_obj}"
        COMMENT "BLLM-99: archiving bolt_rocm_hip.obj into bolt_rocm_hip.lib"
        VERBATIM)
    add_custom_target(bolt_rocm_hip_build DEPENDS "${_bolt_rocm_lib}")

    add_library(bolt_rocm STATIC IMPORTED GLOBAL)
    set_target_properties(bolt_rocm PROPERTIES IMPORTED_LOCATION "${_bolt_rocm_lib}")
    add_dependencies(bolt_rocm bolt_rocm_hip_build)
    add_library(bolt::rocm ALIAS bolt_rocm)

    target_include_directories(bolt_rocm INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    # amdhip64.lib -- the HIP runtime import library every bolt::rocm
    # consumer needs (hipMalloc/hipMemcpy/kernel-launch symbols).
    target_link_libraries(bolt_rocm INTERFACE "$ENV{HIP_PATH}/lib/amdhip64.lib")
    target_link_libraries(bolt INTERFACE bolt::rocm)
endif()

# ---------------------------------------------------------------------------
# bolt::ingest — compiled static library (CSV loader).
# Delegates numeric parsing to bolt::parse; ships as a TU so the parser
# tables link once per binary instead of per consumer.
# ---------------------------------------------------------------------------
option(BOLT_BUILD_INGEST "Build bolt::ingest compiled library" ${BOLT_IS_TOPLEVEL})

# ---------------------------------------------------------------------------
# Optional codec libraries (lakehouse W1). Each is opt-in via find_package;
# default ON when found, OFF otherwise. When unavailable the codec .cpp still
# compiles (stubbed) and tests GTEST_SKIP. BOLT_HAS_* compile defs gate the
# library include + calls inside the .cpp.
# ---------------------------------------------------------------------------
find_package(zstd QUIET)
if(NOT zstd_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZSTD_PC QUIET libzstd)
    endif()
endif()
if(zstd_FOUND OR ZSTD_PC_FOUND)
    option(BOLT_WITH_ZSTD "Build bolt codecs with libzstd" ON)
else()
    option(BOLT_WITH_ZSTD "Build bolt codecs with libzstd" OFF)
endif()

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    option(BOLT_WITH_ZLIB "Build bolt codecs with zlib" ON)
else()
    option(BOLT_WITH_ZLIB "Build bolt codecs with zlib" OFF)
endif()

find_package(lz4 QUIET)
if(NOT lz4_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LZ4_PC QUIET liblz4)
    endif()
endif()
if(lz4_FOUND OR LZ4_PC_FOUND)
    option(BOLT_WITH_LZ4 "Build bolt codecs with liblz4" ON)
else()
    option(BOLT_WITH_LZ4 "Build bolt codecs with liblz4" OFF)
endif()

if(BOLT_BUILD_INGEST)
    if(NOT BOLT_BUILD_PARSE)
        message(FATAL_ERROR "BOLT_BUILD_INGEST requires BOLT_BUILD_PARSE=ON")
    endif()
    add_library(bolt_ingest STATIC
        src/ingest/bolt_csv.cpp
        src/ingest/bolt_parquet_meta.cpp
        src/ingest/bolt_parquet_pageindex.cpp
        src/ingest/bolt_parquet_bloom.cpp
        src/ingest/bolt_parquet_read.cpp
        src/ingest/bolt_parquet_write.cpp
        src/ingest/bolt_avro.cpp
        src/ingest/bolt_roaring.cpp
        src/ingest/bolt_zstd.cpp
        src/ingest/bolt_gzip.cpp
        src/ingest/bolt_lz4.cpp)
    add_library(bolt::ingest ALIAS bolt_ingest)
    target_link_libraries(bolt_ingest PUBLIC bolt::core bolt::parse)
    target_include_directories(bolt_ingest PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_ingest)
    bolt_apply_simd(bolt_ingest ${BOLT_SIMD_TIER})

    if(BOLT_WITH_ZSTD)
        target_compile_definitions(bolt_ingest PRIVATE BOLT_HAS_ZSTD=1)
        if(TARGET zstd::libzstd_shared)
            target_link_libraries(bolt_ingest PRIVATE zstd::libzstd_shared)
        elseif(TARGET zstd::libzstd_static)
            target_link_libraries(bolt_ingest PRIVATE zstd::libzstd_static)
        elseif(TARGET zstd::libzstd)
            target_link_libraries(bolt_ingest PRIVATE zstd::libzstd)
        elseif(ZSTD_PC_FOUND)
            target_include_directories(bolt_ingest PRIVATE ${ZSTD_PC_INCLUDE_DIRS})
            target_link_libraries(bolt_ingest PRIVATE ${ZSTD_PC_LIBRARIES})
        endif()
    endif()
    if(BOLT_WITH_ZLIB)
        target_compile_definitions(bolt_ingest PRIVATE BOLT_HAS_ZLIB=1)
        target_link_libraries(bolt_ingest PRIVATE ZLIB::ZLIB)
    endif()
    if(BOLT_WITH_LZ4)
        target_compile_definitions(bolt_ingest PRIVATE BOLT_HAS_LZ4=1)
        if(TARGET lz4::lz4)
            target_link_libraries(bolt_ingest PRIVATE lz4::lz4)
        elseif(TARGET LZ4::lz4_shared)
            target_link_libraries(bolt_ingest PRIVATE LZ4::lz4_shared)
        elseif(LZ4_PC_FOUND)
            target_include_directories(bolt_ingest PRIVATE ${LZ4_PC_INCLUDE_DIRS})
            target_link_libraries(bolt_ingest PRIVATE ${LZ4_PC_LIBRARIES})
        endif()
    endif()

    target_link_libraries(bolt INTERFACE bolt::ingest)
endif()

# ---------------------------------------------------------------------------
# bolt::crypto — compiled static library (lakehouse W1). Self-contained
# SHA-256 + HMAC-SHA256 + AWS SigV4 signing (zero external dependency), plus
# the Noise XX handshake + ChaCha20-Poly1305 AEAD (noise.cpp), whose X25519 /
# AEAD math is OpenSSL 3.x EVP — gated on OpenSSL being found below.
# ---------------------------------------------------------------------------
# All three default ON: they were unconditional before these switches existed,
# so ON preserves every consumer's build exactly. See the consistency check
# below -- bolt::lakehouse genuinely cannot link without crypto/net/parse.
option(BOLT_BUILD_CRYPTO    "Build bolt::crypto (SigV4 + Noise XX + Ed25519)" ON)
option(BOLT_BUILD_NET       "Build bolt::net (TLS + HTTP client)"             ON)
option(BOLT_BUILD_LAKEHOUSE "Build bolt::lakehouse (Delta + Iceberg + catalogs + object stores)" ON)

# bolt::lakehouse links bolt::crypto (SigV4), bolt::net (HTTP/TLS for the REST
# catalogs) and bolt::parse (fionn JSON). Fail loudly on an impossible combo
# rather than silently flipping a switch the caller explicitly set -- an
# unbuildable request is a programmer error, and a confusing "target bolt::net
# not found" 300 lines later is worse than saying so here.
if(BOLT_BUILD_LAKEHOUSE)
    set(_bolt_lh_missing "")
    if(NOT BOLT_BUILD_CRYPTO)
        list(APPEND _bolt_lh_missing "BOLT_BUILD_CRYPTO")
    endif()
    if(NOT BOLT_BUILD_NET)
        list(APPEND _bolt_lh_missing "BOLT_BUILD_NET")
    endif()
    if(NOT BOLT_BUILD_PARSE)
        list(APPEND _bolt_lh_missing "BOLT_BUILD_PARSE")
    endif()
    if(_bolt_lh_missing)
        string(REPLACE ";" ", " _bolt_lh_missing_str "${_bolt_lh_missing}")
        message(FATAL_ERROR
            "BOLT_BUILD_LAKEHOUSE=ON requires: ${_bolt_lh_missing_str}. "
            "Either enable them, or pass -DBOLT_BUILD_LAKEHOUSE=OFF if you "
            "don't need Delta/Iceberg.")
    endif()
endif()

if(BOLT_BUILD_CRYPTO)
    add_library(bolt_crypto STATIC src/crypto/sigv4.cpp src/crypto/noise.cpp
                src/crypto/ed25519.cpp)
    add_library(bolt::crypto ALIAS bolt_crypto)
    target_link_libraries(bolt_crypto PUBLIC bolt::core)
    target_include_directories(bolt_crypto PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    bolt_apply_hardening(bolt_crypto)
    target_link_libraries(bolt INTERFACE bolt::crypto)
endif()

# ---------------------------------------------------------------------------
# bolt::lakehouse — compiled static library (lakehouse W1 foundations):
# ObjectStore (filesystem + S3-stub) + Catalog (filesystem). Depends on
# bolt::crypto for SigV4. Delta/Iceberg read/write land in later waves.
# ---------------------------------------------------------------------------
# bolt::lakehouse declaration moved below bolt::net (W6 catalogs link bolt::net).

# ---------------------------------------------------------------------------
# bolt::net — TLS client (OpenSSL 3.x, TLS 1.3 only).
# Static lib so the OpenSSL link lives in one TU; consumers (boltapi,
# MQTT, S3) link bolt::net to get bolt::net::TlsSocket. Toggle off if
# OpenSSL is unavailable; the header still compiles via stubs.
# ---------------------------------------------------------------------------
find_package(OpenSSL 3.0 QUIET)
if(OpenSSL_FOUND)
    option(BOLT_WITH_TLS "Build bolt::net with OpenSSL 3.x TLS client" ON)
else()
    option(BOLT_WITH_TLS "Build bolt::net with OpenSSL 3.x TLS client" OFF)
endif()

# bolt::crypto's Noise XX (X25519 + ChaCha20-Poly1305) needs OpenSSL EVP.
# Without it, noise.cpp compiles to explicit-failure stubs (returns false)
# so the link still succeeds; SigV4 stays dependency-free either way.
# Guarded on BOLT_BUILD_CRYPTO: the target only exists when it's enabled.
if(BOLT_BUILD_CRYPTO AND BOLT_WITH_TLS AND OpenSSL_FOUND)
    target_compile_definitions(bolt_crypto PUBLIC BOLT_WITH_TLS=1)
    target_link_libraries(bolt_crypto PUBLIC OpenSSL::Crypto)
endif()

if(BOLT_BUILD_NET)
add_library(bolt_net STATIC src/net/bolt_tls.cpp src/net/bolt_http_client.cpp src/net/bolt_udp.cpp)
add_library(bolt::net ALIAS bolt_net)
target_link_libraries(bolt_net PUBLIC bolt::core)
target_include_directories(bolt_net PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
bolt_apply_hardening(bolt_net)
if(BOLT_WITH_TLS AND OpenSSL_FOUND)
    target_compile_definitions(bolt_net PUBLIC BOLT_WITH_TLS=1)
    target_link_libraries(bolt_net PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()
if(WIN32)
    target_link_libraries(bolt_net PUBLIC ws2_32)
endif()
target_link_libraries(bolt INTERFACE bolt::net)
endif()  # BOLT_BUILD_NET

# ---------------------------------------------------------------------------
# bolt::lakehouse (W1 foundations + W6 catalogs). Depends on bolt::net for
# the HTTP client + TLS used by the Iceberg REST / Unity / Glue / Polaris /
# Nessie / Gravitino catalogs, and bolt::parse for fionn JSON.
# ---------------------------------------------------------------------------
if(BOLT_BUILD_LAKEHOUSE)
add_library(bolt_lakehouse STATIC
    src/lakehouse/object_store_filesystem.cpp
    src/lakehouse/object_store_s3.cpp
    src/lakehouse/object_store_azure.cpp
    src/lakehouse/object_store_gcs.cpp
    src/lakehouse/object_store_compat.cpp
    src/lakehouse/object_store_prefetch.cpp
    src/lakehouse/catalog_filesystem.cpp
    src/lakehouse/catalog_iceberg_rest.cpp
    src/lakehouse/catalog_hive_metastore.cpp
    src/lakehouse/catalog_unity.cpp
    src/lakehouse/catalog_unity_rest.cpp
    src/lakehouse/catalog_glue.cpp
    src/lakehouse/catalog_glue_ext.cpp
    src/lakehouse/catalog_polaris.cpp
    src/lakehouse/catalog_nessie.cpp
    src/lakehouse/catalog_gravitino.cpp
    src/lakehouse/iceberg_transform.cpp
    src/lakehouse/iceberg_metadata.cpp
    src/lakehouse/iceberg_snapshot.cpp
    src/lakehouse/iceberg_manifest.cpp
    src/lakehouse/iceberg_partition.cpp
    src/lakehouse/iceberg_statistics.cpp
    src/lakehouse/iceberg_delete_file.cpp
    src/lakehouse/iceberg_scan.cpp
    src/lakehouse/iceberg_writer.cpp
    src/lakehouse/iceberg_metadata_writer.cpp
    src/lakehouse/iceberg_manifest_writer.cpp
    src/lakehouse/iceberg_data_file_writer.cpp
    src/lakehouse/iceberg_compaction.cpp
    src/lakehouse/iceberg_expire.cpp
    src/lakehouse/iceberg_branches_tags.cpp
    src/lakehouse/iceberg_view.cpp
    src/lakehouse/iceberg_delete_writer.cpp
    src/lakehouse/delta_log.cpp
    src/lakehouse/delta_snapshot.cpp
    src/lakehouse/delta_deletion_vector.cpp
    src/lakehouse/delta_column_mapping.cpp
    src/lakehouse/delta_generated_column.cpp
    src/lakehouse/delta_cdf.cpp
    src/lakehouse/delta_checkpoint.cpp
    src/lakehouse/delta_scan.cpp
    src/lakehouse/delta_writer.cpp
    src/lakehouse/delta_update.cpp
    src/lakehouse/delta_delete.cpp
    src/lakehouse/delta_merge.cpp
    src/lakehouse/delta_restore.cpp
    src/lakehouse/delta_optimize.cpp
    src/lakehouse/delta_vacuum.cpp
    src/lakehouse/delta_checkpoint_write.cpp
    src/lakehouse/scan_optimizer.cpp
    src/lakehouse/column_pruning.cpp
    src/lakehouse/predicate_pushdown.cpp
    src/lakehouse/dv_inline_filter.cpp
    src/lakehouse/simd_scan.cpp
    src/lakehouse/parallel_scan.cpp)
add_library(bolt::lakehouse ALIAS bolt_lakehouse)
target_link_libraries(bolt_lakehouse PUBLIC bolt::core bolt::crypto bolt::net)
if(TARGET bolt::parse)
    target_link_libraries(bolt_lakehouse PUBLIC bolt::parse)
endif()
if(TARGET bolt::ingest)
    target_link_libraries(bolt_lakehouse PUBLIC bolt::ingest)
endif()
target_include_directories(bolt_lakehouse PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
bolt_apply_hardening(bolt_lakehouse)
# W7: GCS RS256 signing uses OpenSSL EVP.
if(OpenSSL_FOUND)
    target_compile_definitions(bolt_lakehouse PRIVATE BOLT_WITH_TLS=1)
    target_link_libraries(bolt_lakehouse PRIVATE OpenSSL::Crypto)
endif()
target_link_libraries(bolt INTERFACE bolt::lakehouse)
endif()  # BOLT_BUILD_LAKEHOUSE

# ---------------------------------------------------------------------------
# bolt::ybolt — Y CRDT runtime adapter (ycpp + bolt::Arena).
# Header-only; builds only when ycpp::headers is present. Provides
# `bolt::ybolt::BoltArenaAllocator` (satisfies ycpp::Allocator), type
# aliases (DeleteSet / StructStore / DecodedUpdate over the bolt arena),
# and a `make_doc(arena, client_id)` helper once ycpp W3 ships. The seam
# is policy-based templates — every alloc inlines through bolt::Arena.
# ---------------------------------------------------------------------------
if(TARGET ycpp::headers OR TARGET ycpp::ycpp)
    add_subdirectory(ybolt)
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

set(_bolt_install_targets bolt_core bolt_kernels bolt_join bolt_wire bolt_arrow bolt bolt_ribbon bolt_net bolt_crypto bolt_lakehouse)
if(BOLT_BUILD_IO)
    list(APPEND _bolt_install_targets bolt_io)
endif()
if(BOLT_BUILD_PARSE)
    list(APPEND _bolt_install_targets bolt_parse)
endif()
if(BOLT_BUILD_VULKAN)
    list(APPEND _bolt_install_targets bolt_vulkan)
endif()
## BLLM-99: bolt_rocm is an IMPORTED target (built via a direct clang++
## invocation, not enable_language(HIP) -- see that block's own comment) --
## install(TARGETS) does not support imported targets ("given target
## bolt_rocm which does not exist"), so it is deliberately NOT added to
## the install list. A consumer wanting to install bolt::rocm's own
## artifact would need to install "${_bolt_rocm_lib}" as a FILE, not a
## TARGET -- not needed for this repo's own build+test use, so not added.
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

# ---------------------------------------------------------------------------
# Feature summary. bolt::core + kernels/join/wire/arrow are header-only and
# always available; everything below is a compiled target you can decline with
# -DBOLT_BUILD_<X>=OFF. Printed so a consumer can SEE what they're paying for
# rather than discovering it in a link line.
# ---------------------------------------------------------------------------
message(STATUS "bolt: compiled targets -- "
               "io=${BOLT_BUILD_IO} ribbon=${BOLT_BUILD_RIBBON} "
               "parse=${BOLT_BUILD_PARSE} ingest=${BOLT_BUILD_INGEST} "
               "crypto=${BOLT_BUILD_CRYPTO} net=${BOLT_BUILD_NET} "
               "lakehouse=${BOLT_BUILD_LAKEHOUSE} "
               "vulkan=${BOLT_BUILD_VULKAN} rocm=${BOLT_BUILD_ROCM}")

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
