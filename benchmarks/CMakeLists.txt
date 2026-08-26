add_executable(bench_bolt bench_bolt.cpp)

target_link_libraries(bench_bolt PRIVATE bolt::bolt)

if(NOT MSVC)
    find_package(Threads REQUIRED)
    target_link_libraries(bench_bolt PRIVATE Threads::Threads)
endif()

bolt_apply_hardening(bench_bolt)
bolt_apply_bench_tuning(bench_bolt)

# -----------------------------------------------------------------------
# TPC-H-lite micro-query benchmark (Phase 6)
# -----------------------------------------------------------------------
add_executable(bench_tpch_lite bench_tpch_lite.cpp)

target_link_libraries(bench_tpch_lite PRIVATE bolt::bolt)

if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_tpch_lite PRIVATE Threads::Threads)
endif()

bolt_apply_hardening(bench_tpch_lite)
bolt_apply_bench_tuning(bench_tpch_lite)

# -----------------------------------------------------------------------
# Kernel-level microbenchmarks — isolate SIMD kernels from query glue
# -----------------------------------------------------------------------
add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_kernels PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_kernels)
bolt_apply_bench_tuning(bench_kernels)

# -----------------------------------------------------------------------
# 1BRC-shape benchmark — per-station min/max/sum over 100M rows
# -----------------------------------------------------------------------
add_executable(bench_1brc bench_1brc.cpp)
target_link_libraries(bench_1brc PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_1brc PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_1brc)
bolt_apply_bench_tuning(bench_1brc)

# -----------------------------------------------------------------------
# Two-input sorted-merge set ops — UNION/INTERSECT/EXCEPT, DISTINCT+ALL.
# -----------------------------------------------------------------------
add_executable(bench_setop bench_setop.cpp)
target_link_libraries(bench_setop PRIVATE bolt::bolt)
bolt_apply_hardening(bench_setop)
bolt_apply_bench_tuning(bench_setop)

# UTF-8 native operator kernel microbenchmarks
add_executable(bench_utf8 bench_utf8.cpp)
target_link_libraries(bench_utf8 PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_utf8 PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_utf8)
bolt_apply_bench_tuning(bench_utf8)

# -----------------------------------------------------------------------
# K-JOIN-MM: chained hash-join multi-match + composite-key microbench.
# -----------------------------------------------------------------------
add_executable(bench_hashjoin_multi_match bench_hashjoin_multi_match.cpp)
target_link_libraries(bench_hashjoin_multi_match PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_hashjoin_multi_match PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_hashjoin_multi_match)
bolt_apply_bench_tuning(bench_hashjoin_multi_match)

# -----------------------------------------------------------------------
# Card A2: typed all-shapes SwissTable join kernel microbench
# (int64 fast path + composite + spilled-Utf8 key).
# -----------------------------------------------------------------------
add_executable(bench_joinkernel bench_joinkernel.cpp)
target_link_libraries(bench_joinkernel PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_joinkernel PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_joinkernel)
bolt_apply_bench_tuning(bench_joinkernel)

# -----------------------------------------------------------------------
# K-AGG-A: typed multi-key GROUP BY kernel microbench.
# -----------------------------------------------------------------------
add_executable(bench_groupby bench_groupby.cpp)
target_link_libraries(bench_groupby PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_groupby PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_groupby)
bolt_apply_bench_tuning(bench_groupby)

# -----------------------------------------------------------------------
# INTERVAL arithmetic microbenchmark — date32 + (months, days) paths.
# -----------------------------------------------------------------------
add_executable(bench_interval bench_interval.cpp)
target_link_libraries(bench_interval PRIVATE bolt::bolt)
bolt_apply_hardening(bench_interval)
bolt_apply_bench_tuning(bench_interval)

# -----------------------------------------------------------------------
# LSD radix sort vs merge argsort microbench — pure-int64-key fast path.
# -----------------------------------------------------------------------
add_executable(bench_radixsort bench_radixsort.cpp)
target_link_libraries(bench_radixsort PRIVATE bolt::bolt)
bolt_apply_hardening(bench_radixsort)
bolt_apply_bench_tuning(bench_radixsort)

# -----------------------------------------------------------------------
# JSON parse throughput — tape / SAX / NDJSON (fionn-parity wave).
# -----------------------------------------------------------------------
add_executable(bench_json bench_json.cpp)
target_link_libraries(bench_json PRIVATE bolt::bolt)
bolt_apply_hardening(bench_json)
bolt_apply_bench_tuning(bench_json)

# -----------------------------------------------------------------------
# Indirect multi-key sort + Top-K microbench (K-SORT-A)
# Removed: bench_indirect_sort.cpp not present on this branch.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Perf Wave 2: col-literal arithmetic (numeric + Decimal128), scale-aware
# d128 compare-filter, f64 mixed-compare filter + coercion helpers.
# -----------------------------------------------------------------------
add_executable(bench_decimal bench_decimal.cpp)
target_link_libraries(bench_decimal PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_decimal PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_decimal)
bolt_apply_bench_tuning(bench_decimal)

# -----------------------------------------------------------------------
# G2FEAT-89/362: salt-in-entry SwissTable probe go/no-go microbenchmark.
# -----------------------------------------------------------------------
add_executable(bench_salt_probe bench_salt_probe.cpp)
target_link_libraries(bench_salt_probe PRIVATE bolt::bolt)
if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_salt_probe PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_salt_probe)
bolt_apply_bench_tuning(bench_salt_probe)

# -----------------------------------------------------------------------
# BLLM-14: int8-activation IDOT GEMV perf tripwire. Times the dispatched
# (CPUID-selected) kernel vs the scalar reference and asserts a GFLOPS-ratio
# floor when a SIMD tier is present -- the only gate that catches a silent
# SIMD->scalar regression (the equality tests can't). Registered as a ctest
# so it runs on x86 CI lanes; no-op PASS on scalar-only hardware. Both kernels
# come from the baseline/per-source-flagged bolt::compute_idot lib, so the
# benchmark TU's own compile flags don't advantage either side.
# -----------------------------------------------------------------------
if(BOLT_BUILD_COMPUTE_IDOT)
    add_executable(bench_compute_idot bench_compute_idot.cpp)
    target_link_libraries(bench_compute_idot PRIVATE bolt::bolt bolt::compute_idot)
    bolt_apply_hardening(bench_compute_idot)
    bolt_apply_bench_tuning(bench_compute_idot)
    add_test(NAME bench_compute_idot COMMAND bench_compute_idot)
endif()

# -----------------------------------------------------------------------
# Snappy + parquet decode. Runs standalone with NO data file (synthesises a
# corpus and compresses it with a reference compressor in the benchmark
# itself), or against a real parquet file when given a path. Both modes verify
# correctness before reporting a time.
# -----------------------------------------------------------------------
add_executable(bench_parquet_vs_arrow bench_parquet_vs_arrow.cpp)
target_link_libraries(bench_parquet_vs_arrow PRIVATE bolt::bolt)
bolt_apply_hardening(bench_parquet_vs_arrow)
bolt_apply_bench_tuning(bench_parquet_vs_arrow)

add_executable(bench_parquet_write bench_parquet_write.cpp)
target_link_libraries(bench_parquet_write PRIVATE bolt::bolt)
if(Threads_FOUND)
    target_link_libraries(bench_parquet_write PRIVATE Threads::Threads)
endif()
bolt_apply_hardening(bench_parquet_write)
bolt_apply_bench_tuning(bench_parquet_write)

add_executable(bench_parquet_decode bench_parquet_decode.cpp)

target_link_libraries(bench_parquet_decode PRIVATE bolt::bolt)

if(NOT MSVC)
    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()
    target_link_libraries(bench_parquet_decode PRIVATE Threads::Threads)
endif()

bolt_apply_hardening(bench_parquet_decode)
bolt_apply_bench_tuning(bench_parquet_decode)
