add_executable(test_bolt_primitives test_bolt_primitives.cpp)

target_link_libraries(test_bolt_primitives
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_primitives)

add_test(NAME test_bolt_primitives COMMAND test_bolt_primitives)

# ZoneMap primitive — MarbleDB Wave 10.1 consolidation.
add_executable(test_bolt_zonemap test_bolt_zonemap.cpp)
target_link_libraries(test_bolt_zonemap
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_zonemap)
add_test(NAME test_bolt_zonemap COMMAND test_bolt_zonemap)

# Vector distance kernels — L2/IP/cosine pair + vertical accumulators.
# Math leaf lifted from src/ext/pdx_kernels.cpp during the
# inherited-noodle-agent consolidation.
add_executable(test_bolt_vector_distance test_bolt_vector_distance.cpp)
target_link_libraries(test_bolt_vector_distance
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_vector_distance)
add_test(NAME test_bolt_vector_distance COMMAND test_bolt_vector_distance)

# Wave 9.4 ζ.3 — half-precision (f16) vector distance kernels.
# Cross-checks NEON FP16 / AVX2 cvtph / scalar fallback against the
# well-tested f32 path, modulo f16 quantisation error.
add_executable(test_bolt_vector_distance_f16 test_bolt_vector_distance_f16.cpp)
target_link_libraries(test_bolt_vector_distance_f16
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_vector_distance_f16)
add_test(NAME test_bolt_vector_distance_f16 COMMAND test_bolt_vector_distance_f16)

# Wave 9.4 ζ.4 — 8-bit scalar-quantised vector distance kernels.
# Cross-checks NEON DOTPROD / AVX2 / NEON baseline / scalar against
# a hand-rolled integer reference. Exact-equality assertions (no
# rounding tolerance — u8 L2² is integer arithmetic end-to-end).
add_executable(test_bolt_vector_distance_u8 test_bolt_vector_distance_u8.cpp)
target_link_libraries(test_bolt_vector_distance_u8
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_vector_distance_u8)
add_test(NAME test_bolt_vector_distance_u8 COMMAND test_bolt_vector_distance_u8)

# Rotation kernels — Householder QR + DCT-II.
add_executable(test_bolt_rotation test_bolt_rotation.cpp)
target_link_libraries(test_bolt_rotation
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_rotation)
add_test(NAME test_bolt_rotation COMMAND test_bolt_rotation)

# Column stats f32 kernel — Welford mean/var/min/max.
add_executable(test_bolt_column_stats test_bolt_column_stats.cpp)
target_link_libraries(test_bolt_column_stats
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_column_stats)
add_test(NAME test_bolt_column_stats COMMAND test_bolt_column_stats)

# Bitmap primitive — MarbleDB inherited-noodle agent consolidation.
add_executable(test_bolt_bitmap test_bolt_bitmap.cpp)
target_link_libraries(test_bolt_bitmap
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_bitmap)
add_test(NAME test_bolt_bitmap COMMAND test_bolt_bitmap)

# TopK kernel — argmin + max-heap top-k.
add_executable(test_bolt_topk test_bolt_topk.cpp)
target_link_libraries(test_bolt_topk
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_topk)
add_test(NAME test_bolt_topk COMMAND test_bolt_topk)

# HyperLogLog++ sketch — approximate COUNT(DISTINCT) / NDV estimation.
add_executable(test_bolt_hll test_bolt_hll.cpp)
target_link_libraries(test_bolt_hll
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_hll)
add_test(NAME test_bolt_hll COMMAND test_bolt_hll)

# K-means kernel — assign / update / convergence delta. Wave 19 Plan D.
add_executable(test_bolt_kmeans test_bolt_kmeans.cpp)
target_link_libraries(test_bolt_kmeans
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_kmeans)
add_test(NAME test_bolt_kmeans COMMAND test_bolt_kmeans)

# Wave 9.4 — vector column primitive + centroid kernels (row-major).
# Covers BoltType::Embedding factories, VectorStats sidecar, wire
# round-trip with fixed_size, k-means train, BOND zone ranking, σ-bound.
add_executable(test_bolt_vector_column test_bolt_vector_column.cpp)
target_link_libraries(test_bolt_vector_column
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_vector_column)
add_test(NAME test_bolt_vector_column COMMAND test_bolt_vector_column)

# Wave 9.4 ζ.5b — PDX-BOND probabilistic Chebyshev LOWER bound kernel.
# Tightens the box-LB block-skip in the canonical PDX-64 walker.
add_executable(test_bolt_vector_bond test_bolt_vector_bond.cpp)
target_link_libraries(test_bolt_vector_bond
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_vector_bond)
add_test(NAME test_bolt_vector_bond COMMAND test_bolt_vector_bond)

# Quantized distance kernels — SQ8 / RaBitQ / F16, scalar-first
# (Wave 19 plan B; AVX2 specialisations deferred).
add_executable(test_bolt_quantized_distance test_bolt_quantized_distance.cpp)
target_link_libraries(test_bolt_quantized_distance
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_quantized_distance)
add_test(NAME test_bolt_quantized_distance COMMAND test_bolt_quantized_distance)

add_executable(test_bolt_scheduler test_bolt_scheduler.cpp)

target_link_libraries(test_bolt_scheduler
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_scheduler)

add_test(NAME test_bolt_scheduler COMMAND test_bolt_scheduler)

add_executable(test_bolt_strtemp test_bolt_strtemp.cpp)

target_link_libraries(test_bolt_strtemp
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_strtemp)

add_test(NAME test_bolt_strtemp COMMAND test_bolt_strtemp)

add_executable(test_bolt_join test_bolt_join.cpp)

target_link_libraries(test_bolt_join
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_join)

add_test(NAME test_bolt_join COMMAND test_bolt_join)

add_executable(test_bolt_mergejoin test_bolt_mergejoin.cpp)

target_link_libraries(test_bolt_mergejoin
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_mergejoin)

add_test(NAME test_bolt_mergejoin COMMAND test_bolt_mergejoin)

add_executable(test_bolt_fintech_phase1 test_bolt_fintech_phase1.cpp)

target_link_libraries(test_bolt_fintech_phase1
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase1)

add_test(NAME test_bolt_fintech_phase1 COMMAND test_bolt_fintech_phase1)

add_executable(test_bolt_fintech_phase2 test_bolt_fintech_phase2.cpp)

target_link_libraries(test_bolt_fintech_phase2
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase2)

add_test(NAME test_bolt_fintech_phase2 COMMAND test_bolt_fintech_phase2)

add_executable(test_bolt_fintech_phase4a test_bolt_fintech_phase4a.cpp)

target_link_libraries(test_bolt_fintech_phase4a
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase4a)

add_test(NAME test_bolt_fintech_phase4a COMMAND test_bolt_fintech_phase4a)

add_executable(test_bolt_fintech_phase4b test_bolt_fintech_phase4b.cpp)

target_link_libraries(test_bolt_fintech_phase4b
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase4b)

add_test(NAME test_bolt_fintech_phase4b COMMAND test_bolt_fintech_phase4b)

add_executable(test_bolt_fintech_phase5r test_bolt_fintech_phase5r.cpp)

target_link_libraries(test_bolt_fintech_phase5r
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5r)

add_test(NAME test_bolt_fintech_phase5r COMMAND test_bolt_fintech_phase5r)

add_executable(test_bolt_fintech_phase3 test_bolt_fintech_phase3.cpp)

target_link_libraries(test_bolt_fintech_phase3
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase3)

add_test(NAME test_bolt_fintech_phase3 COMMAND test_bolt_fintech_phase3)

add_executable(test_bolt_fintech_phase5e test_bolt_fintech_phase5e.cpp)

target_link_libraries(test_bolt_fintech_phase5e
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5e)

add_test(NAME test_bolt_fintech_phase5e COMMAND test_bolt_fintech_phase5e)

add_executable(test_bolt_fintech_phase5w test_bolt_fintech_phase5w.cpp)

target_link_libraries(test_bolt_fintech_phase5w
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5w)

add_test(NAME test_bolt_fintech_phase5w COMMAND test_bolt_fintech_phase5w)

add_executable(test_bolt_fintech_phase5s_sorted test_bolt_fintech_phase5s_sorted.cpp)

target_link_libraries(test_bolt_fintech_phase5s_sorted
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_sorted)

add_test(NAME test_bolt_fintech_phase5s_sorted COMMAND test_bolt_fintech_phase5s_sorted)

add_executable(test_bolt_fintech_phase5s_core test_bolt_fintech_phase5s_core.cpp)

target_link_libraries(test_bolt_fintech_phase5s_core
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_core)

add_test(NAME test_bolt_fintech_phase5s_core COMMAND test_bolt_fintech_phase5s_core)

add_executable(test_bolt_fintech_phase5s_kalman test_bolt_fintech_phase5s_kalman.cpp)

target_link_libraries(test_bolt_fintech_phase5s_kalman
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_fintech_phase5s_kalman)

add_test(NAME test_bolt_fintech_phase5s_kalman COMMAND test_bolt_fintech_phase5s_kalman)

add_executable(test_bolt_wave2 test_bolt_wave2.cpp)

target_link_libraries(test_bolt_wave2
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wave2)

add_test(NAME test_bolt_wave2 COMMAND test_bolt_wave2)

add_executable(test_bolt_wire test_bolt_wire.cpp)

target_link_libraries(test_bolt_wire
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wire)

add_test(NAME test_bolt_wire COMMAND test_bolt_wire)

add_executable(test_bolt_wire_stream test_bolt_wire_stream.cpp)

target_link_libraries(test_bolt_wire_stream
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_wire_stream)

add_test(NAME test_bolt_wire_stream COMMAND test_bolt_wire_stream)

# Two-input sorted-merge set operations (UNION/INTERSECT/EXCEPT,
# both DISTINCT and ALL). Companion to bolt_kmerge for k = 2.
add_executable(test_bolt_setop test_bolt_setop.cpp)

target_link_libraries(test_bolt_setop
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_setop)

add_test(NAME test_bolt_setop COMMAND test_bolt_setop)

add_executable(test_bolt_kmerge test_bolt_kmerge.cpp)

target_link_libraries(test_bolt_kmerge
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_kmerge)

add_test(NAME test_bolt_kmerge COMMAND test_bolt_kmerge)

add_executable(test_bolt_kernels test_bolt_kernels.cpp)

target_link_libraries(test_bolt_kernels
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_kernels)

add_test(NAME test_bolt_kernels COMMAND test_bolt_kernels)

add_executable(test_bolt_simd test_bolt_simd.cpp)

target_link_libraries(test_bolt_simd
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_simd)

add_test(NAME test_bolt_simd COMMAND test_bolt_simd)

add_executable(test_bolt_topology test_bolt_topology.cpp)

target_link_libraries(test_bolt_topology
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_topology)

add_test(NAME test_bolt_topology COMMAND test_bolt_topology)

add_executable(test_bolt_parallel test_bolt_parallel.cpp)

target_link_libraries(test_bolt_parallel
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_parallel)

add_test(NAME test_bolt_parallel COMMAND test_bolt_parallel)

add_executable(test_bolt_config test_bolt_config.cpp)

target_link_libraries(test_bolt_config
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_config)

add_test(NAME test_bolt_config COMMAND test_bolt_config)

add_executable(test_bolt_arrow test_bolt_arrow.cpp)

target_link_libraries(test_bolt_arrow
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_arrow)

add_test(NAME test_bolt_arrow COMMAND test_bolt_arrow)

add_executable(test_bolt_stream test_bolt_stream.cpp)

target_link_libraries(test_bolt_stream
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_stream)

add_test(NAME test_bolt_stream COMMAND test_bolt_stream)

add_executable(test_bolt_branchless test_bolt_branchless.cpp)

target_link_libraries(test_bolt_branchless
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_branchless)

add_test(NAME test_bolt_branchless COMMAND test_bolt_branchless)

add_executable(test_bolt_ebr test_bolt_ebr.cpp)

target_link_libraries(test_bolt_ebr
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_ebr)

add_test(NAME test_bolt_ebr COMMAND test_bolt_ebr)

if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_ascii test_bolt_ascii.cpp)
    target_link_libraries(test_bolt_ascii
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_ascii)
    add_test(NAME test_bolt_ascii COMMAND test_bolt_ascii)
endif()

if(BOLT_BUILD_INGEST)
    add_executable(test_bolt_csv test_bolt_csv.cpp)
    target_link_libraries(test_bolt_csv
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_csv)
    add_test(NAME test_bolt_csv COMMAND test_bolt_csv)
endif()

if(BOLT_BUILD_IO)
    add_executable(test_bolt_binsearch_crc32c test_bolt_binsearch_crc32c.cpp)
    target_link_libraries(test_bolt_binsearch_crc32c
        PRIVATE bolt::bolt bolt::io GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_binsearch_crc32c)
    add_test(NAME test_bolt_binsearch_crc32c COMMAND test_bolt_binsearch_crc32c)
endif()

# Variable-width payload column — Layer 1.1 of the parse / JSON / doc work.
add_executable(test_bolt_var_column test_bolt_var_column.cpp)
target_link_libraries(test_bolt_var_column
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_var_column)
add_test(NAME test_bolt_var_column COMMAND test_bolt_var_column)

# VarBinary string kernels + str8 zone-map — Layer 1.2.
add_executable(test_bolt_var_kernels test_bolt_var_kernels.cpp)
target_link_libraries(test_bolt_var_kernels
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_var_kernels)
add_test(NAME test_bolt_var_kernels COMMAND test_bolt_var_kernels)

# Variant (sum-type) column primitive — Layer 1.2b.
add_executable(test_bolt_variant_column test_bolt_variant_column.cpp)
target_link_libraries(test_bolt_variant_column
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_variant_column)
add_test(NAME test_bolt_variant_column COMMAND test_bolt_variant_column)

# Skip-aware JSON parser — Layer 1.3.
if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_parse_json test_bolt_parse_json.cpp)
    target_link_libraries(test_bolt_parse_json
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_parse_json)
    add_test(NAME test_bolt_parse_json COMMAND test_bolt_parse_json)
endif()

# Atomic-bitmap primitives — soft-delete / live-mask substrate.
add_executable(test_bolt_atomic_bitmap test_bolt_atomic_bitmap.cpp)
target_link_libraries(test_bolt_atomic_bitmap
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_atomic_bitmap)
add_test(NAME test_bolt_atomic_bitmap COMMAND test_bolt_atomic_bitmap)

# JSONB sorted-keys binary container — Layer 1.4.
add_executable(test_bolt_jsonb test_bolt_jsonb.cpp)
target_link_libraries(test_bolt_jsonb
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_jsonb)
add_test(NAME test_bolt_jsonb COMMAND test_bolt_jsonb)

# Small-array sort kernels — fixed-cap insertion sorts for hot top-k drains.
add_executable(test_bolt_smallsort test_bolt_smallsort.cpp)
target_link_libraries(test_bolt_smallsort
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_smallsort)
add_test(NAME test_bolt_smallsort COMMAND test_bolt_smallsort)

# FST term dictionary — Layer 1.5 (Lucene-style minimal acyclic FST).
add_executable(test_bolt_fst test_bolt_fst.cpp)
target_link_libraries(test_bolt_fst
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_fst)
add_test(NAME test_bolt_fst COMMAND test_bolt_fst)

# PFOR-Delta scalar block kernels — Layer 2 substrate (Lemire 2015 / Lucene104).
add_executable(test_bolt_pfor test_bolt_pfor.cpp)
target_link_libraries(test_bolt_pfor
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_pfor)
add_test(NAME test_bolt_pfor COMMAND test_bolt_pfor)

# PFOR-Delta SIMD overlay — Layer 2.1b (AVX2 + SSE4.2). Bit-exact equivalence
# with the scalar path is the load-bearing correctness gate.
add_executable(test_bolt_pfor_simd test_bolt_pfor_simd.cpp)
target_link_libraries(test_bolt_pfor_simd
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_pfor_simd)
add_test(NAME test_bolt_pfor_simd COMMAND test_bolt_pfor_simd)

# Window function ranking kernels — ROW_NUMBER / RANK / DENSE_RANK / LAG /
# LEAD / FIRST_VALUE / LAST_VALUE / NTILE. Operate on already-partitioned +
# sorted input; the kernel walks the sequence and emits ranks / offsets,
# detecting partition breaks by key change. TPC-DS prerequisite.
add_executable(test_bolt_window test_bolt_window.cpp)
target_link_libraries(test_bolt_window
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_window)
add_test(NAME test_bolt_window COMMAND test_bolt_window)

# Decimal128 kernel family (add/sub/mul/div, filter_{gt,lt,eq,ne,ge,le},
# sum/min/max). Backs SQL Decimal monetary columns (l_extendedprice,
# o_totalprice, p_retailprice). Branchless scalar with cross-checks
# against __int128 where available.
add_executable(test_bolt_decimal test_bolt_decimal.cpp)
target_link_libraries(test_bolt_decimal
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_decimal)
add_test(NAME test_bolt_decimal COMMAND test_bolt_decimal)

# UTF-8 native operator kernels — compare/eq/ne/hash/LIKE (compiled)/SUBSTRING/
# TRIM/UPPER/LOWER/LENGTH/POSITION/CONCAT/REPLACE.  Backs SQL VARCHAR/TEXT
# without falling back to Arrow compute (TPC-H Q14, Q16, Q19, Q20).
add_executable(test_bolt_utf8 test_bolt_utf8.cpp)
target_link_libraries(test_bolt_utf8
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_utf8)
add_test(NAME test_bolt_utf8 COMMAND test_bolt_utf8)

# SQL INTERVAL arithmetic kernels (date32 + months/days, sub_date, mul, filter).
# Tiny semantic, ubiquitous in TPC-H predicates like
# `l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY`.
add_executable(test_bolt_interval test_bolt_interval.cpp)
target_link_libraries(test_bolt_interval
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_interval)
add_test(NAME test_bolt_interval COMMAND test_bolt_interval)

# Date32 extract kernel (year/month/day/quarter/week/dow/doy + filter wrappers).
add_executable(test_bolt_date test_bolt_date.cpp)
target_link_libraries(test_bolt_date PRIVATE bolt::bolt GTest::gtest_main)
bolt_apply_hardening(test_bolt_date PRIVATE)
add_test(NAME bolt_date COMMAND test_bolt_date)

# Lock-free queue zoo (Wave: morsel-distribution substrate).
# Borrowed from chukonu-v1 lockfree_queue.h, stripped of Arrow / shared_ptr
# and rebuilt to Bolt's Tiger-Style header-only rules.
add_executable(test_bolt_sharded_mpsc test_bolt_sharded_mpsc.cpp)
target_link_libraries(test_bolt_sharded_mpsc
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_sharded_mpsc)
add_test(NAME test_bolt_sharded_mpsc COMMAND test_bolt_sharded_mpsc)

add_executable(test_bolt_node_pool test_bolt_node_pool.cpp)
target_link_libraries(test_bolt_node_pool
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_node_pool)
add_test(NAME test_bolt_node_pool COMMAND test_bolt_node_pool)

add_executable(test_bolt_work_stealing_deque test_bolt_work_stealing_deque.cpp)
target_link_libraries(test_bolt_work_stealing_deque
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_work_stealing_deque)
add_test(NAME test_bolt_work_stealing_deque COMMAND test_bolt_work_stealing_deque)

# COUNT(DISTINCT) / SUM(DISTINCT) per-cell tracker — K-AGG-B substrate.
add_executable(test_bolt_groupby_distinct test_bolt_groupby_distinct.cpp)
target_link_libraries(test_bolt_groupby_distinct
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_groupby_distinct)
add_test(NAME test_bolt_groupby_distinct COMMAND test_bolt_groupby_distinct)

# K-AGG-A typed multi-key GROUP BY kernel — correctness on small inputs.
add_executable(test_bolt_groupby_typed test_bolt_groupby_typed.cpp)
target_link_libraries(test_bolt_groupby_typed
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_groupby_typed)
add_test(NAME test_bolt_groupby_typed COMMAND test_bolt_groupby_typed)

# Typed all-shapes SwissTable join kernel (bolt/join/bolt_joinkernel.h).
add_executable(test_bolt_joinkernel test_bolt_joinkernel.cpp)
target_link_libraries(test_bolt_joinkernel
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_joinkernel)
add_test(NAME test_bolt_joinkernel COMMAND test_bolt_joinkernel)
