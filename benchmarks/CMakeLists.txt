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
