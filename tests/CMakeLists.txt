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
# Byte/string hash primitive — bolt::hash_bytes (RDF term interner backbone).
# Single scalar wyhash/xxh3-style block reader finalized through swiss_mix so
# it composes with SwissTable. No SIMD divergence (predecessor join-corruption).
add_executable(test_bolt_hash_bytes test_bolt_hash_bytes.cpp)
target_link_libraries(test_bolt_hash_bytes
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_hash_bytes)
add_test(NAME test_bolt_hash_bytes COMMAND test_bolt_hash_bytes)

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

# Explicit-comparator permutation merge sort — the std::sort replacement on
# chukonu's ORDER BY / agg-spill / window hot paths (stable, no-alloc).
add_executable(test_bolt_msort_perm test_bolt_msort_perm.cpp)
target_link_libraries(test_bolt_msort_perm
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)
bolt_apply_hardening(test_bolt_msort_perm)
add_test(NAME test_bolt_msort_perm COMMAND test_bolt_msort_perm)

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

add_executable(test_promql_kernels test_promql_kernels.cpp)

target_link_libraries(test_promql_kernels
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_promql_kernels)

add_test(NAME test_promql_kernels COMMAND test_promql_kernels)

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

# Null-bitmap-aware filter kernels (SQL three-valued logic): cmp / cmp_scalar
# with validity bitmaps, cmp_is_null / cmp_is_not_null, dense + sparse paths,
# all-valid fast path == no-validity path.
add_executable(test_bolt_filter_null test_bolt_filter_null.cpp)

target_link_libraries(test_bolt_filter_null
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_filter_null)

add_test(NAME test_bolt_filter_null COMMAND test_bolt_filter_null)

# Indirect / argsort kernels — return a sorted index permutation (single
# key + lexicographic multi-key) for the triple-store index build and
# ORDER BY (sort perm, then gather columns). Stable, allocation-free.
add_executable(test_bolt_argsort test_bolt_argsort.cpp)

target_link_libraries(test_bolt_argsort
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_argsort)

add_test(NAME test_bolt_argsort COMMAND test_bolt_argsort)

# LSD radix sort — pure-int64-key fast path (direct value sort + indirect
# permutation sort). The linear-time speed upgrade promised by bolt_argsort.h
# for triple-store index build + single-int64-key ORDER BY. Scalar==SIMD
# digit-extract parity is asserted (DigitExtractParity test).
add_executable(test_bolt_radixsort test_bolt_radixsort.cpp)

target_link_libraries(test_bolt_radixsort
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_radixsort)

add_test(NAME test_bolt_radixsort COMMAND test_bolt_radixsort)

# Parallel multi-key radix argsort (G2FEAT-157) — byte-identical to stable sort.
add_executable(test_bolt_radixsort_parallel test_bolt_radixsort_parallel.cpp)

target_link_libraries(test_bolt_radixsort_parallel
    PRIVATE
        bolt::bolt
        GTest::gtest
        GTest::gtest_main
)

bolt_apply_hardening(test_bolt_radixsort_parallel)

add_test(NAME test_bolt_radixsort_parallel COMMAND test_bolt_radixsort_parallel)

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

# Scan/scatter + CSR graph kernels — exclusive_scan / histogram / scatter
# (bolt_scan.h) and csr_build / csr_expand (bolt_csr.h). The array + graph
# primitives under chukonu's CSR adjacency build and 1-hop expand operators.
add_executable(test_bolt_csr test_bolt_csr.cpp)
target_link_libraries(test_bolt_csr
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_csr)
add_test(NAME test_bolt_csr COMMAND test_bolt_csr)

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

# Sandboxed Jinja2-subset lexer for LLM chat templates — BLLM-70/71.
if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_parse_jinja test_bolt_parse_jinja.cpp)
    target_link_libraries(test_bolt_parse_jinja
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_parse_jinja)
    add_test(NAME test_bolt_parse_jinja COMMAND test_bolt_parse_jinja)
endif()

# Sandboxed Jinja2-subset expression parser/AST — BLLM-70/72.
if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_parse_jinja_ast test_bolt_parse_jinja_ast.cpp)
    target_link_libraries(test_bolt_parse_jinja_ast
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_parse_jinja_ast)
    add_test(NAME test_bolt_parse_jinja_ast COMMAND test_bolt_parse_jinja_ast)
endif()

# Sandboxed Jinja2-subset statement parser (if/for/set/macro) — BLLM-70/73.
if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_parse_jinja_stmt test_bolt_parse_jinja_stmt.cpp)
    target_link_libraries(test_bolt_parse_jinja_stmt
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_parse_jinja_stmt)
    add_test(NAME test_bolt_parse_jinja_stmt COMMAND test_bolt_parse_jinja_stmt)
endif()

# Sandboxed Jinja2-subset evaluator (Value/Scope + tree-walking render) — BLLM-70/73.
if(BOLT_BUILD_PARSE)
    add_executable(test_bolt_parse_jinja_eval test_bolt_parse_jinja_eval.cpp)
    target_link_libraries(test_bolt_parse_jinja_eval
        PRIVATE bolt::bolt bolt::parse GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_parse_jinja_eval)
    add_test(NAME test_bolt_parse_jinja_eval COMMAND test_bolt_parse_jinja_eval)
endif()

# Vulkan device/queue/memory scaffolding — BLLM-46. Runs against whatever
# real Vulkan device this machine has; self-skips (GTEST_SKIP, not a
# failure) if bolt::vulkan::available() is false.
if(BOLT_BUILD_VULKAN)
    add_executable(test_bolt_vulkan test_bolt_vulkan.cpp)
    target_link_libraries(test_bolt_vulkan
        PRIVATE bolt::bolt bolt::vulkan GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_vulkan)
    add_test(NAME test_bolt_vulkan COMMAND test_bolt_vulkan)
endif()

# ROCm/HIP matmul+dequant kernel core — BLLM-78. Runs against whatever
# real ROCm device this machine has; self-skips (GTEST_SKIP, not a
# failure) if bolt::rocm::available() is false. Only built when a real
# hipcc was found (BOLT_BUILD_ROCM) -- not the case on this dev box.
if(BOLT_BUILD_ROCM)
    add_executable(test_bolt_rocm test_bolt_rocm.cpp)
    target_link_libraries(test_bolt_rocm
        PRIVATE bolt::bolt bolt::rocm GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_rocm)
    add_test(NAME test_bolt_rocm COMMAND test_bolt_rocm)
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

# Holistic + two-input group-stats kernels — exact MEDIAN / QUANTILE
# (sort-based, reuses the integer argsort over a sortable-key transform) and
# single-pass Pearson CORRELATION / COVARIANCE moments. Back chukonu's
# PhysicalHashAggregate holistic + 2-arg aggregate path.
add_executable(test_bolt_group_stats test_bolt_group_stats.cpp)
target_link_libraries(test_bolt_group_stats
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_group_stats)
add_test(NAME test_bolt_group_stats COMMAND test_bolt_group_stats)

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

# K-AGG-C two-pass typed GROUP BY ingest — cross-path hash/group parity.
add_executable(test_bolt_groupby_twopass test_bolt_groupby_twopass.cpp)
target_link_libraries(test_bolt_groupby_twopass
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_groupby_twopass)
add_test(NAME test_bolt_groupby_twopass COMMAND test_bolt_groupby_twopass)

# G2FEAT-13 radix route-hash kernels — THE stored-hash invariant property
# test (gb_route_hash_rows == gb_merge_hash_stored for every row) +
# radix_partition_hashes invariants + GbSharedScratch begin differential.
add_executable(test_bolt_groupby_radix_hash test_bolt_groupby_radix_hash.cpp)
target_link_libraries(test_bolt_groupby_radix_hash
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_groupby_radix_hash)
add_test(NAME test_bolt_groupby_radix_hash COMMAND test_bolt_groupby_radix_hash)

# Typed all-shapes SwissTable join kernel (bolt/join/bolt_joinkernel.h).
add_executable(test_bolt_joinkernel test_bolt_joinkernel.cpp)
target_link_libraries(test_bolt_joinkernel
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_joinkernel)
add_test(NAME test_bolt_joinkernel COMMAND test_bolt_joinkernel)

# Perf Wave 2 kernels — col-literal arithmetic (numeric + Decimal128),
# scale-aware d128 col-col compare-filter, f64 mixed-compare filter and
# the widen/d128_to_f64 coercion helpers. Every kernel vs a scalar oracle
# (the oracle replicates chukonu filter_op.cpp dense_filter_decimal_col).
add_executable(test_bolt_decimal_kernels test_bolt_decimal_kernels.cpp)
target_link_libraries(test_bolt_decimal_kernels
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_decimal_kernels)
add_test(NAME test_bolt_decimal_kernels COMMAND test_bolt_decimal_kernels)

# W3 — parallel CSV parse parity vs the serial parser (value-identical
# output; spilled Utf8 compares by content since overflow offsets differ
# by design between the sequential and chunk-windowed writers).
add_executable(test_bolt_csv_parallel test_bolt_csv_parallel.cpp)
target_link_libraries(test_bolt_csv_parallel
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_csv_parallel)
add_test(NAME test_bolt_csv_parallel COMMAND test_bolt_csv_parallel)

# W-PQ — Parquet footer metadata reader (hand-rolled thrift compact subset)
# vs pyarrow goldens + corrupt-input fuzzing (truncations, byte flips).
add_executable(test_bolt_parquet_meta test_bolt_parquet_meta.cpp)
target_link_libraries(test_bolt_parquet_meta
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_parquet_meta PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_parquet_meta)
add_test(NAME test_bolt_parquet_meta COMMAND test_bolt_parquet_meta)

# W-PQ — snappy block decompressor vs cramjam vectors + corrupt fuzzing.
add_executable(test_bolt_snappy test_bolt_snappy.cpp)
target_link_libraries(test_bolt_snappy
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_snappy PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_snappy)
add_test(NAME test_bolt_snappy COMMAND test_bolt_snappy)

# W-PQ increment 2 — Parquet page reader: all-1000-row value verification
# against both goldens (snappy+dict and plain), row-group-granular reads,
# and corrupt-page fuzzing (truncate / flip / zero -> false, never crash).
# G2FEAT-134 — self-contained zstd decoder: byte-exact vs python-zstandard,
# a real zstd parquet file end to end, and malformed-input robustness.
add_executable(test_bolt_zstd_dec test_bolt_zstd_dec.cpp)
target_link_libraries(test_bolt_zstd_dec
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_zstd_dec PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_zstd_dec)
add_test(NAME test_bolt_zstd_dec COMMAND test_bolt_zstd_dec)

add_executable(test_bolt_parquet_read test_bolt_parquet_read.cpp)
target_link_libraries(test_bolt_parquet_read
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_parquet_read PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_parquet_read)
add_test(NAME test_bolt_parquet_read COMMAND test_bolt_parquet_read)

# G2FEAT-23 — Parquet page index (ColumnIndex/OffsetIndex) + split-block
# bloom filter reader vs pyarrow/duckdb goldens: analytic page-bound ground
# truth, prune-property tests, XXH64 vectors, bloom membership from two
# independent writers, hostile-input fuzzing.
add_executable(test_bolt_parquet_pageindex test_bolt_parquet_pageindex.cpp)
target_link_libraries(test_bolt_parquet_pageindex
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_parquet_pageindex PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_parquet_pageindex)
add_test(NAME test_bolt_parquet_pageindex COMMAND test_bolt_parquet_pageindex)

# W-PQ-W — Parquet writer round-trip vs the reader (Int64/Utf8 plain,
# Int64/Utf8 SNAPPY, nullable Int64).
add_executable(test_bolt_parquet_write test_bolt_parquet_write.cpp)
target_link_libraries(test_bolt_parquet_write
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_parquet_write)
add_test(NAME test_bolt_parquet_write COMMAND test_bolt_parquet_write)

# G2FEAT-46 — Parquet type conformance: TIMESTAMP (millis/micros/nanos),
# TIME, INT96 legacy timestamp, INT32-DECIMAL, un/signed IntType mapping +
# a Timestamp writer->reader round-trip.
add_executable(test_bolt_parquet_types test_bolt_parquet_types.cpp)
target_link_libraries(test_bolt_parquet_types
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_parquet_types)
add_test(NAME test_bolt_parquet_types COMMAND test_bolt_parquet_types)

# bolt::net::TlsSocket — OpenSSL 3.x TLS 1.3 client wrapper (G2CLUS-128).
# Live connect test against cloudflare.com:443; skips on offline workers.
add_executable(test_bolt_tls test_bolt_tls.cpp)
target_link_libraries(test_bolt_tls
    PRIVATE bolt::net GTest::gtest GTest::gtest_main)
if(WIN32)
    target_link_libraries(test_bolt_tls PRIVATE ws2_32)
endif()
bolt_apply_hardening(test_bolt_tls)
add_test(NAME test_bolt_tls COMMAND test_bolt_tls)

# ---------------------------------------------------------------------------
# Lakehouse W1 — Avro OCF, Roaring bitmap, codecs (zstd/gzip/lz4), SigV4,
# ObjectStore + Catalog. (G2CLUS-184)
# ---------------------------------------------------------------------------
if(BOLT_BUILD_INGEST)
    add_executable(test_bolt_avro test_bolt_avro.cpp)
    target_link_libraries(test_bolt_avro
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    # Iceberg-manifest-shaped goldens written by the stock Apache Avro JAVA
    # writer (data/golden_avro_manifest.avro, golden_avro_array_of_record.avro).
    target_compile_definitions(test_bolt_avro PRIVATE
        BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    bolt_apply_hardening(test_bolt_avro)
    add_test(NAME test_bolt_avro COMMAND test_bolt_avro)

    add_executable(test_bolt_protobuf test_bolt_protobuf.cpp)
    target_link_libraries(test_bolt_protobuf
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_protobuf)
    add_test(NAME test_bolt_protobuf COMMAND test_bolt_protobuf)

    add_executable(test_bolt_roaring test_bolt_roaring.cpp)
    target_link_libraries(test_bolt_roaring
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_roaring)
    add_test(NAME test_bolt_roaring COMMAND test_bolt_roaring)

    add_executable(test_bolt_zstd test_bolt_zstd.cpp)
    target_link_libraries(test_bolt_zstd
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_zstd)
    add_test(NAME test_bolt_zstd COMMAND test_bolt_zstd)

    add_executable(test_bolt_gzip test_bolt_gzip.cpp)
    target_link_libraries(test_bolt_gzip
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_gzip)
    add_test(NAME test_bolt_gzip COMMAND test_bolt_gzip)

    add_executable(test_bolt_lz4 test_bolt_lz4.cpp)
    target_link_libraries(test_bolt_lz4
        PRIVATE bolt::bolt bolt::ingest GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_lz4)
    add_test(NAME test_bolt_lz4 COMMAND test_bolt_lz4)
endif()

add_executable(test_bolt_sigv4 test_bolt_sigv4.cpp)
target_link_libraries(test_bolt_sigv4
    PRIVATE bolt::crypto GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_sigv4)
add_test(NAME test_bolt_sigv4 COMMAND test_bolt_sigv4)

add_executable(test_bolt_lakehouse_catalog test_bolt_lakehouse_catalog.cpp)
target_link_libraries(test_bolt_lakehouse_catalog
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog)
add_test(NAME test_bolt_lakehouse_catalog COMMAND test_bolt_lakehouse_catalog)

# W7 — Azure / GCS / prefetch.
add_executable(test_bolt_lakehouse_azure_sign test_bolt_lakehouse_azure_sign.cpp)
target_link_libraries(test_bolt_lakehouse_azure_sign
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_azure_sign)
add_test(NAME test_bolt_lakehouse_azure_sign COMMAND test_bolt_lakehouse_azure_sign)

add_executable(test_bolt_lakehouse_gcs_jwt test_bolt_lakehouse_gcs_jwt.cpp)
target_link_libraries(test_bolt_lakehouse_gcs_jwt
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
# Test exercises the same OpenSSL EVP path as the lakehouse impl when TLS is on.
find_package(OpenSSL 3.0 QUIET)
if(OpenSSL_FOUND)
    target_compile_definitions(test_bolt_lakehouse_gcs_jwt PRIVATE BOLT_WITH_TLS=1)
    target_link_libraries(test_bolt_lakehouse_gcs_jwt PRIVATE OpenSSL::Crypto)
endif()
bolt_apply_hardening(test_bolt_lakehouse_gcs_jwt)
add_test(NAME test_bolt_lakehouse_gcs_jwt COMMAND test_bolt_lakehouse_gcs_jwt)

add_executable(test_bolt_lakehouse_prefetch test_bolt_lakehouse_prefetch.cpp)
target_link_libraries(test_bolt_lakehouse_prefetch
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_prefetch)
add_test(NAME test_bolt_lakehouse_prefetch COMMAND test_bolt_lakehouse_prefetch)

# W8 — scan optimisations (selection vectors, column prune, predicate
# pushdown, DV inline filter, partition prune, SIMD dispatch, parallel
# pool lifecycle).
add_executable(test_bolt_lakehouse_scan_optim test_bolt_lakehouse_scan_optim.cpp)
target_link_libraries(test_bolt_lakehouse_scan_optim
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_scan_optim)
add_test(NAME test_bolt_lakehouse_scan_optim COMMAND test_bolt_lakehouse_scan_optim)
add_executable(test_bolt_lakehouse_catalog_iceberg_rest
    test_bolt_lakehouse_catalog_iceberg_rest.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_iceberg_rest
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_iceberg_rest)
add_test(NAME test_bolt_lakehouse_catalog_iceberg_rest
         COMMAND test_bolt_lakehouse_catalog_iceberg_rest)

add_executable(test_bolt_lakehouse_catalog_unity
    test_bolt_lakehouse_catalog_unity.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_unity
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_unity)
add_test(NAME test_bolt_lakehouse_catalog_unity
         COMMAND test_bolt_lakehouse_catalog_unity)

add_executable(test_bolt_lakehouse_catalog_hive_thrift
    test_bolt_lakehouse_catalog_hive_thrift.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_hive_thrift
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_hive_thrift)
add_test(NAME test_bolt_lakehouse_catalog_hive_thrift
         COMMAND test_bolt_lakehouse_catalog_hive_thrift)

add_executable(test_bolt_lakehouse_catalog_glue_sign
    test_bolt_lakehouse_catalog_glue_sign.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_glue_sign
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_glue_sign)
add_test(NAME test_bolt_lakehouse_catalog_glue_sign
         COMMAND test_bolt_lakehouse_catalog_glue_sign)

# G2CLUS-178 — extended Glue surface (partitions / schema registry / LF grants).
add_executable(test_bolt_lakehouse_catalog_glue_ext
    test_bolt_lakehouse_catalog_glue_ext.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_glue_ext
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_glue_ext)
add_test(NAME test_bolt_lakehouse_catalog_glue_ext
         COMMAND test_bolt_lakehouse_catalog_glue_ext)

# G2CLUS-180 — Polaris (Iceberg REST reuse + management API).
add_executable(test_bolt_lakehouse_catalog_polaris
    test_bolt_lakehouse_catalog_polaris.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_polaris
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_polaris)
add_test(NAME test_bolt_lakehouse_catalog_polaris
         COMMAND test_bolt_lakehouse_catalog_polaris)

# G2CLUS-181 — Nessie (Iceberg REST reuse + native v2 tree API).
add_executable(test_bolt_lakehouse_catalog_nessie
    test_bolt_lakehouse_catalog_nessie.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_nessie
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_nessie)
add_test(NAME test_bolt_lakehouse_catalog_nessie
         COMMAND test_bolt_lakehouse_catalog_nessie)

# G2CLUS-182 — Gravitino (metalake/catalog/schema/table CRUD).
add_executable(test_bolt_lakehouse_catalog_gravitino
    test_bolt_lakehouse_catalog_gravitino.cpp)
target_link_libraries(test_bolt_lakehouse_catalog_gravitino
    PRIVATE bolt::lakehouse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_catalog_gravitino)
add_test(NAME test_bolt_lakehouse_catalog_gravitino
         COMMAND test_bolt_lakehouse_catalog_gravitino)

# W4 — Iceberg read path.
add_executable(test_bolt_lakehouse_iceberg_read test_bolt_lakehouse_iceberg_read.cpp)
target_link_libraries(test_bolt_lakehouse_iceberg_read
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_iceberg_read)
add_test(NAME test_bolt_lakehouse_iceberg_read COMMAND test_bolt_lakehouse_iceberg_read)

# G2FEAT-76 — read a REAL Iceberg table's Avro manifests (pyiceberg-written
# fixtures, deflate-coded). Distinct from the JSON-manifest tests above.
add_executable(test_bolt_iceberg_real_avro test_bolt_iceberg_real_avro.cpp)
target_link_libraries(test_bolt_iceberg_real_avro
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_iceberg_real_avro PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_iceberg_real_avro)
add_test(NAME test_bolt_iceberg_real_avro COMMAND test_bolt_iceberg_real_avro)

# G2FEAT-125 — SCAN a real Iceberg table end to end (metadata -> manifests ->
# Parquet -> rows) over the complete pyiceberg warehouse in
# data/golden_iceberg_table. The test above only decodes manifests.
add_executable(test_bolt_iceberg_real_scan test_bolt_iceberg_real_scan.cpp)
target_link_libraries(test_bolt_iceberg_real_scan
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
target_compile_definitions(test_bolt_iceberg_real_scan PRIVATE
    BOLT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
bolt_apply_hardening(test_bolt_iceberg_real_scan)
add_test(NAME test_bolt_iceberg_real_scan COMMAND test_bolt_iceberg_real_scan)

# W5 — Iceberg write path.
add_executable(test_bolt_lakehouse_iceberg_write test_bolt_lakehouse_iceberg_write.cpp)
target_link_libraries(test_bolt_lakehouse_iceberg_write
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_iceberg_write)
add_test(NAME test_bolt_lakehouse_iceberg_write COMMAND test_bolt_lakehouse_iceberg_write)

# W2 — Delta read path.
add_executable(test_bolt_lakehouse_delta_read test_bolt_lakehouse_delta_read.cpp)
target_link_libraries(test_bolt_lakehouse_delta_read
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_delta_read)
add_test(NAME test_bolt_lakehouse_delta_read COMMAND test_bolt_lakehouse_delta_read)

# W3 — Delta write path.
add_executable(test_bolt_lakehouse_delta_write test_bolt_lakehouse_delta_write.cpp)
target_link_libraries(test_bolt_lakehouse_delta_write
    PRIVATE bolt::lakehouse bolt::ingest bolt::parse GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_lakehouse_delta_write)
add_test(NAME test_bolt_lakehouse_delta_write COMMAND test_bolt_lakehouse_delta_write)

# Plan E2 phase 2 — blocked parallel inclusive prefix scan (int64/float64):
# the single-partition cumulative-window substrate (SUM/COUNT/AVG OVER
# UNBOUNDED PRECEDING). Fixed-block determinism + exactness properties.
add_executable(test_bolt_prefix_scan test_bolt_prefix_scan.cpp)
target_link_libraries(test_bolt_prefix_scan
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_prefix_scan)
add_test(NAME test_bolt_prefix_scan COMMAND test_bolt_prefix_scan)

# RE2 common-subset anchored FULL-MATCH regex engine (bolt_regex.h re2_*).
# Thompson NFA: alternation / {m,n} / char-classes / (?i) inline flags —
# the subset PromQL =~/!~ matchers need, cross-checked against the exact
# PromQL corpus patterns (and reject-cleanly / ReDoS-safe cases).
add_executable(test_bolt_regex_re2 test_bolt_regex_re2.cpp)
target_link_libraries(test_bolt_regex_re2
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_regex_re2)
add_test(NAME test_bolt_regex_re2 COMMAND test_bolt_regex_re2)

# BLLM-12 (N0) — device-generic tensor primitive (bolt_tensor.h). Substrate
# boltllm::quant::QTensor generalizes into: proves the Tensor POD layout,
# arena/caller-owned (no-alloc) construction, and per-row-scale
# quantized row-stride formulas are exact matches for QuantTypes.h's
# existing convention (F32/Int8/Int4/Int2), not a fresh redesign.
add_executable(test_bolt_tensor test_bolt_tensor.cpp)
target_link_libraries(test_bolt_tensor
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_tensor)
add_test(NAME test_bolt_tensor COMMAND test_bolt_tensor)

# BLLM-13 — device-generic compute dispatch table (bolt_compute.h), layered
# on bolt_tensor.h's Tensor/Device (BLLM-12). CPU-tier matmul/rmsnorm/rope
# (both conventions)/softmax/swiglu ports + a net-new
# top-k/top-p/temperature sampler; Vulkan/CUDA slots reserved but hard-fail
# rather than silently no-op.
add_executable(test_bolt_compute test_bolt_compute.cpp)
target_link_libraries(test_bolt_compute
    PRIVATE bolt::bolt GTest::gtest GTest::gtest_main)
bolt_apply_hardening(test_bolt_compute)
add_test(NAME test_bolt_compute COMMAND test_bolt_compute)

# BLLM-14 — int8-activation IDOT GEMV kernels (bolt::compute_idot). Consolidates
# boltllm's two IDOT test suites into one gtest TU: every available SIMD tier
# (AVX2/AVX-512VNNI) bit-exact vs scalar, dispatch == selected tier, tolerance
# vs a dequant-dot reference, determinism. Self-guards on BOLT_ARCH_X86 + the
# runtime CPUID probes so it is green on x86 (fast tiers exercised) and ARM
# (scalar-only). Only built when the lib is.
if(BOLT_BUILD_COMPUTE_IDOT)
    add_executable(test_bolt_idot test_bolt_idot.cpp)
    target_link_libraries(test_bolt_idot
        PRIVATE bolt::bolt bolt::compute_idot GTest::gtest GTest::gtest_main)
    bolt_apply_hardening(test_bolt_idot)
    add_test(NAME test_bolt_idot COMMAND test_bolt_idot)
endif()
