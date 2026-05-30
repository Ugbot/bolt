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
# Indirect multi-key sort + Top-K microbench (K-SORT-A)
# Removed: bench_indirect_sort.cpp not present on this branch.
# -----------------------------------------------------------------------
