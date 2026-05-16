# BoltCompileOptions.cmake — reusable compile option helpers.
#
# Every bolt::* target calls bolt_apply_hardening() to enforce the
# no-exceptions / no-rtti / warnings policy, portably across MSVC/clang/gcc.
#
# SIMD policy:
#   bolt_apply_simd(tgt mode)  — pin an ISA tier on `tgt`.
#       mode ∈ {NATIVE, AVX512, AVX2, SSE42, PORTABLE}
#   bolt_apply_bench_tuning(tgt) — benchmark-only -O3/-march=native/arch:AVX2
#       when BOLT_ENABLE_NATIVE=ON.
#
# Precedence (documented): the top-level CMakeLists applies
# `bolt_apply_simd(bolt_core ${BOLT_SIMD_TIER})` so every consumer of
# `bolt::core` inherits the selected ISA flag. `BOLT_ENABLE_NATIVE` is kept
# for back-compat and only affects `bolt_apply_bench_tuning` (benchmarks).
# If both are set, `BOLT_SIMD_TIER` wins for core; bench tuning layers on top.

include_guard(GLOBAL)

function(bolt_apply_hardening tgt)
    get_target_property(type ${tgt} TYPE)
    if(type STREQUAL "INTERFACE_LIBRARY")
        set(scope INTERFACE)
    else()
        set(scope PRIVATE)
    endif()

    if(MSVC)
        # Note: MSVC stdlib headers (<chrono>, <mutex>, ...) require /EHsc, so
        # we keep exceptions *enabled at the ABI level*. Our own code is still
        # exception-free — enforced by `noexcept` on every function, not by
        # the compiler flag.
        #
        # `/GR-` (no-RTTI) is applied PRIVATE only — Bolt's own code never
        # uses dynamic_cast. Consumers may legitimately need RTTI (e.g.
        # llm-station's tool-registry dynamic_cast); propagating `/GR-` via
        # INTERFACE caused silent crashes in consumer code on Windows where
        # consumer dynamic_cast sites compiled to UB.
        target_compile_options(${tgt} ${scope}
            /W4
            /permissive-
            /Zc:__cplusplus
            /Zc:preprocessor
            /EHsc
            /utf-8
            /wd4324   # structure padded due to alignas — deliberate
            /wd4201   # nameless struct/union — used in BoltColumn
            /DNOMINMAX
            /D_CRT_SECURE_NO_WARNINGS
        )
        if(NOT scope STREQUAL "INTERFACE")
            # PRIVATE for compiled libraries — keeps Bolt's own TUs RTTI-free.
            target_compile_options(${tgt} PRIVATE /GR-)
        endif()
        if(BOLT_WARNINGS_AS_ERRORS)
            target_compile_options(${tgt} ${scope} /WX)
        endif()
    else()
        target_compile_options(${tgt} ${scope}
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wno-unused-parameter
        )
        # -fno-exceptions / -fno-rtti: Bolt's own code is exception- and
        # RTTI-free, but propagating these via INTERFACE breaks consumers
        # that legitimately use them (cppzmq throws on socket errors;
        # llm-station's ToolExecutorRegistry uses dynamic_cast). Apply
        # PRIVATE only — same precedent the MSVC branch uses for /GR-.
        if(NOT scope STREQUAL "INTERFACE")
            target_compile_options(${tgt} PRIVATE -fno-exceptions -fno-rtti)
        endif()
        if(BOLT_WARNINGS_AS_ERRORS)
            target_compile_options(${tgt} ${scope} -Werror)
        endif()
    endif()
endfunction()

# bolt_apply_simd(tgt mode)
#   Pin a SIMD ISA tier for `tgt`.  `mode` is one of:
#     NATIVE   — MSVC /arch:AVX2, GCC/Clang -march=native
#     AVX512   — MSVC /arch:AVX512, GCC/Clang
#                -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx2 -msse4.2
#                (stub: bolt_port.h sets BOLT_SIMD_AVX512=1 and falls through
#                 to AVX2 wrappers; native 512-bit specialisations land later)
#     AVX2     — MSVC /arch:AVX2, GCC/Clang -mavx2 -msse4.2
#     SSE42    — MSVC x64 baseline (no flag; MSVC has no /arch:SSE4.2 —
#                SSE2 is baseline and SSE4.2 intrinsics compile without
#                a flag), GCC/Clang -msse4.2
#     PORTABLE — no SIMD flag (compiler default baseline; on x86 this
#                usually means SSE2 for MSVC x64, or whatever the target
#                triple implies for GCC/Clang).
#
# Applied with INTERFACE scope for INTERFACE libs (propagates to consumers),
# PRIVATE otherwise — same pattern as bolt_apply_hardening.
function(bolt_apply_simd tgt mode)
    get_target_property(type ${tgt} TYPE)
    if(type STREQUAL "INTERFACE_LIBRARY")
        set(scope INTERFACE)
    else()
        set(scope PRIVATE)
    endif()

    string(TOUPPER "${mode}" mode_up)
    set(_bolt_simd_effective "${mode_up}")

    if(mode_up STREQUAL "NATIVE")
        if(MSVC)
            target_compile_options(${tgt} ${scope} /arch:AVX2)
        else()
            target_compile_options(${tgt} ${scope} -march=native)
        endif()
    elseif(mode_up STREQUAL "AVX512")
        if(MSVC)
            target_compile_options(${tgt} ${scope} /arch:AVX512)
        else()
            target_compile_options(${tgt} ${scope}
                -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx2 -msse4.2)
        endif()
    elseif(mode_up STREQUAL "AVX2")
        if(MSVC)
            target_compile_options(${tgt} ${scope} /arch:AVX2)
        else()
            target_compile_options(${tgt} ${scope} -mavx2 -msse4.2)
        endif()
    elseif(mode_up STREQUAL "SSE42")
        if(MSVC)
            # MSVC x64 has no /arch:SSE4.2 flag — SSE2 is baseline, and
            # SSE4.2 intrinsics compile without any flag. __SSE4_2__ is
            # NOT auto-defined, so bolt_port.h will fall through to the
            # scalar path unless /arch:AVX2 is used.
        else()
            target_compile_options(${tgt} ${scope} -msse4.2)
        endif()
    elseif(mode_up STREQUAL "PORTABLE")
        # Intentionally no flag — compiler baseline only.
    else()
        message(FATAL_ERROR
            "bolt_apply_simd: unknown mode '${mode}' "
            "(expected NATIVE|AVX512|AVX2|SSE42|PORTABLE)")
    endif()

    # Report the effective SIMD tier once, the first time this function
    # runs for a given mode. Subsequent calls are silent.
    get_property(_reported GLOBAL PROPERTY _bolt_simd_reported)
    if(NOT _reported)
        if(MSVC)
            set(_tc "MSVC")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(_tc "Clang")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(_tc "GCC")
        else()
            set(_tc "${CMAKE_CXX_COMPILER_ID}")
        endif()
        message(STATUS "bolt: SIMD tier = ${_bolt_simd_effective} (toolchain: ${_tc})")
        set_property(GLOBAL PROPERTY _bolt_simd_reported ON)
    endif()
endfunction()

# bolt_apply_feature_toggles(tgt)
#   Propagate the perf-punchlist compile-time toggles as preprocessor
#   defines on `tgt` (INTERFACE scope for INTERFACE libs, PRIVATE otherwise).
#   Toggles: BOLT_ENABLE_HUGE_PAGES, BOLT_ENABLE_NUMA, BOLT_SWISS_LAYOUT,
#   BOLT_HASH_TIER.  See docs/BOLT_PERF_PUNCHLIST.md.
function(bolt_apply_feature_toggles tgt)
    get_target_property(type ${tgt} TYPE)
    if(type STREQUAL "INTERFACE_LIBRARY")
        set(scope INTERFACE)
    else()
        set(scope PRIVATE)
    endif()

    if(BOLT_ENABLE_HUGE_PAGES)
        target_compile_definitions(${tgt} ${scope} BOLT_ENABLE_HUGE_PAGES=1)
    else()
        target_compile_definitions(${tgt} ${scope} BOLT_ENABLE_HUGE_PAGES=0)
    endif()

    if(BOLT_ENABLE_NUMA)
        target_compile_definitions(${tgt} ${scope} BOLT_ENABLE_NUMA=1)
    else()
        target_compile_definitions(${tgt} ${scope} BOLT_ENABLE_NUMA=0)
    endif()

    string(TOUPPER "${BOLT_SWISS_LAYOUT}" _swiss_up)
    if(_swiss_up STREQUAL "FLAT")
        target_compile_definitions(${tgt} ${scope} BOLT_SWISS_LAYOUT_FLAT=1)
    elseif(_swiss_up STREQUAL "INTERLEAVED")
        target_compile_definitions(${tgt} ${scope} BOLT_SWISS_LAYOUT_INTERLEAVED=1)
    else()
        message(FATAL_ERROR
            "bolt_apply_feature_toggles: BOLT_SWISS_LAYOUT='${BOLT_SWISS_LAYOUT}' "
            "(expected FLAT|INTERLEAVED)")
    endif()

    string(TOUPPER "${BOLT_HASH_TIER}" _hash_up)
    if(_hash_up STREQUAL "WYHASH3")
        target_compile_definitions(${tgt} ${scope} BOLT_HASH_TIER_WYHASH3=1)
    elseif(_hash_up STREQUAL "XXH3")
        target_compile_definitions(${tgt} ${scope} BOLT_HASH_TIER_XXH3=1)
    elseif(_hash_up STREQUAL "MURMUR3")
        target_compile_definitions(${tgt} ${scope} BOLT_HASH_TIER_MURMUR3=1)
    else()
        message(FATAL_ERROR
            "bolt_apply_feature_toggles: BOLT_HASH_TIER='${BOLT_HASH_TIER}' "
            "(expected WYHASH3|XXH3|MURMUR3)")
    endif()
endfunction()

function(bolt_apply_bench_tuning tgt)
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /O2)
        if(BOLT_ENABLE_NATIVE)
            # MSVC has no -march=native; AVX2 is the closest uniform
            # "modern x86" baseline that unlocks __AVX2__ in bolt_port.h.
            target_compile_options(${tgt} PRIVATE /arch:AVX2)
        endif()
    else()
        target_compile_options(${tgt} PRIVATE -O3)
        if(BOLT_ENABLE_NATIVE)
            target_compile_options(${tgt} PRIVATE -march=native)
        endif()
    endif()
endfunction()
