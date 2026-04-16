#pragma once
// bolt_config.h — overridable compile-time constants.
//
// Every tunable in Bolt that downstream projects may want to override
// lives here as a macro with an `#ifndef` guard. Users override by
// either:
//
//   1. Defining the macro on the compiler command line:
//        -DBOLT_MAX_WORKERS=128
//        -DBOLT_LATENCY_GRAIN_BYTES=32768
//
//   2. Creating `bolt/bolt_user_config.h` on the include path with
//      `#define BOLT_X ...` lines before any Bolt header is included.
//      If that file exists, we pull it in before the default block.
//
// Every macro is also exposed as a `constexpr` in `namespace bolt::config`
// so runtime code and templates can read the effective values uniformly
// without re-checking the macro.
//
// Streaming note: latency-sensitive stream processors typically want
// tiny grains (16-32 KB), strict pinning, and low parallel break-even
// thresholds. See "Streaming defaults" cookbook at the bottom.

#include <cstdint>

// Optional user-supplied overrides. Drop a file on the include path.
#if defined(__has_include)
    #if __has_include("bolt/bolt_user_config.h")
        #include "bolt/bolt_user_config.h"
    #endif
#endif

// ---------------------------------------------------------------------------
// Infrastructure bounds (sized at init, never at hot path).
// ---------------------------------------------------------------------------
#ifndef BOLT_MAX_WORKERS
    #define BOLT_MAX_WORKERS 64u           // upper cap on scheduler thread pool
#endif
#ifndef BOLT_TASK_RING_SIZE
    #define BOLT_TASK_RING_SIZE 16384u     // SPMC ring slots (power of two)
#endif
#ifndef BOLT_SCHEDULER_POOL_CAPACITY
    #define BOLT_SCHEDULER_POOL_CAPACITY 1024u  // range/column task pool slots
#endif
#ifndef BOLT_SCHEDULER_POOL_SLOT_BYTES
    #define BOLT_SCHEDULER_POOL_SLOT_BYTES 128u // bytes per pool entry
#endif
#ifndef BOLT_DEFAULT_SPIN_COUNT
    #define BOLT_DEFAULT_SPIN_COUNT 1000u  // spin-yield iterations before yield
#endif

// ---------------------------------------------------------------------------
// TuneProfile presets (SchedulerConfig::with_profile reads these).
// Override to match your workload's trade-offs.
// ---------------------------------------------------------------------------
// Latency profile — HFT / streaming: minimize tail, burn CPU to get it.
#ifndef BOLT_LATENCY_GRAIN_BYTES
    #define BOLT_LATENCY_GRAIN_BYTES (64u * 1024u)
#endif
#ifndef BOLT_LATENCY_DISPATCH_BATCH
    #define BOLT_LATENCY_DISPATCH_BATCH 1u
#endif

// Balanced profile — default analytics behavior.
#ifndef BOLT_BALANCED_GRAIN_BYTES
    #define BOLT_BALANCED_GRAIN_BYTES (256u * 1024u)
#endif
#ifndef BOLT_BALANCED_DISPATCH_BATCH
    #define BOLT_BALANCED_DISPATCH_BATCH 1u
#endif

// Throughput profile — bulk scans, accept higher p99 for higher agg rate.
#ifndef BOLT_THROUGHPUT_GRAIN_BYTES
    #define BOLT_THROUGHPUT_GRAIN_BYTES (1024u * 1024u)
#endif
#ifndef BOLT_THROUGHPUT_DISPATCH_BATCH
    #define BOLT_THROUGHPUT_DISPATCH_BATCH 4u
#endif

// ---------------------------------------------------------------------------
// Parallel-kernel break-even thresholds.
// Below these row counts the parallel_* wrappers fall through to the
// single-threaded kernel so dispatch overhead doesn't dominate small work.
//
// Picked from `benchmarks/bench_tpch_lite.cpp` measurements on a 4-core
// laptop: dispatch amortizes around 1-5M rows for filter/sum. Streaming
// setups (thousands of sub-10K batches per second) want these LOWER so
// the scheduler stays in the hot path; drop to 1 to always parallelize.
// ---------------------------------------------------------------------------
#ifndef BOLT_PARALLEL_MIN_ROWS_FILTER
    #define BOLT_PARALLEL_MIN_ROWS_FILTER 500000
#endif
#ifndef BOLT_PARALLEL_MIN_ROWS_SUM
    #define BOLT_PARALLEL_MIN_ROWS_SUM 500000
#endif
#ifndef BOLT_PARALLEL_MIN_ROWS_GROUPBY
    #define BOLT_PARALLEL_MIN_ROWS_GROUPBY 100000
#endif

// Cardinality threshold for groupby Phase-2 merge strategy. When the sum of
// distinct groups across all worker partials is below this value, the
// parallel groupby uses a single-thread serial walk to merge partials into
// one global table. Above it, the radix-partitioned parallel merge is used.
//
// Why: 1BRC-shape inputs have low cardinality (~413 distinct stations), so
// 8 workers × 413 = ~3300 total triples. The radix scatter (atomic CAS per
// triple, P shard buffers, P parallel merges) costs more than just walking
// the partials in one thread for cases this small. High-cardinality joins
// (10k+ distinct groups) keep the radix path.
#ifndef BOLT_GROUPBY_SERIAL_MERGE_THRESHOLD
    #define BOLT_GROUPBY_SERIAL_MERGE_THRESHOLD 4096
#endif

// Lookahead prefetch distance for the groupby ingest loop. Default = 0
// (DISABLED) because the workloads we measure (1BRC-shape low
// cardinality, ~413 distinct keys hit repeatedly) keep the actual
// hot probed cache lines in L1 even though the table itself is large
// — capacity-based heuristics misjudge this. Set to >0 (typically 16)
// when you know your workload has high distinct-key spread (random
// IDs, large unique-key universe) so the probed lines aren't
// repeat-hot. The `SwissTable::prefetch` primitive remains available
// regardless of this setting.
//
// Tested in Wave M1: prefetch = 16 regressed 1BRC by ~30% on the
// i9-9980HK because the prefetch instructions issue but the lines
// were already in L1. See docs/research/design-log.md.
#ifndef BOLT_GROUPBY_PREFETCH_AHEAD
    #define BOLT_GROUPBY_PREFETCH_AHEAD 0
#endif

// Below this SwissTable capacity (slot count), the bulk-probe code
// paths that DO opt into prefetch (notably `SwissTable::find_simd` for
// hash-join probes) skip the prefetch — the table fits in L1 and the
// instructions would be pure overhead. Above it, prefetch runs. Default
// threshold = 2048 slots ≈ 34 KB.
#ifndef BOLT_PREFETCH_TABLE_CAPACITY_THRESHOLD
    #define BOLT_PREFETCH_TABLE_CAPACITY_THRESHOLD 2048
#endif

// ---------------------------------------------------------------------------
// constexpr mirror — runtime code reads these, not the macros.
// ---------------------------------------------------------------------------
namespace bolt {
namespace config {

inline constexpr uint32_t kMaxWorkers             = BOLT_MAX_WORKERS;
inline constexpr uint32_t kTaskRingSize           = BOLT_TASK_RING_SIZE;
inline constexpr uint32_t kSchedulerPoolCapacity  = BOLT_SCHEDULER_POOL_CAPACITY;
inline constexpr uint32_t kSchedulerPoolSlotBytes = BOLT_SCHEDULER_POOL_SLOT_BYTES;
inline constexpr uint32_t kDefaultSpinCount       = BOLT_DEFAULT_SPIN_COUNT;

inline constexpr uint32_t kLatencyGrainBytes      = BOLT_LATENCY_GRAIN_BYTES;
inline constexpr uint32_t kLatencyDispatchBatch   = BOLT_LATENCY_DISPATCH_BATCH;
inline constexpr uint32_t kBalancedGrainBytes     = BOLT_BALANCED_GRAIN_BYTES;
inline constexpr uint32_t kBalancedDispatchBatch  = BOLT_BALANCED_DISPATCH_BATCH;
inline constexpr uint32_t kThroughputGrainBytes   = BOLT_THROUGHPUT_GRAIN_BYTES;
inline constexpr uint32_t kThroughputDispatchBatch= BOLT_THROUGHPUT_DISPATCH_BATCH;

inline constexpr int64_t  kParallelMinRowsFilter  = BOLT_PARALLEL_MIN_ROWS_FILTER;
inline constexpr int64_t  kParallelMinRowsSum     = BOLT_PARALLEL_MIN_ROWS_SUM;
inline constexpr int64_t  kParallelMinRowsGroupby = BOLT_PARALLEL_MIN_ROWS_GROUPBY;
inline constexpr uint32_t kGroupbySerialMergeThreshold = BOLT_GROUPBY_SERIAL_MERGE_THRESHOLD;
inline constexpr uint32_t kGroupbyPrefetchAhead        = BOLT_GROUPBY_PREFETCH_AHEAD;
inline constexpr uint32_t kPrefetchTableCapacityThreshold = BOLT_PREFETCH_TABLE_CAPACITY_THRESHOLD;

// Static sanity: power-of-two ring; non-zero bounds.
static_assert((kTaskRingSize & (kTaskRingSize - 1)) == 0,
              "BOLT_TASK_RING_SIZE must be a power of two");
static_assert(kMaxWorkers > 0, "BOLT_MAX_WORKERS must be > 0");
static_assert(kSchedulerPoolCapacity > 0, "pool capacity must be > 0");
static_assert(kLatencyGrainBytes     >= 64,  "latency grain too small");
static_assert(kBalancedGrainBytes    >= 64,  "balanced grain too small");
static_assert(kThroughputGrainBytes  >= 64,  "throughput grain too small");

}  // namespace config
}  // namespace bolt

// ---------------------------------------------------------------------------
// Cookbook — drop one of these into bolt/bolt_user_config.h for common
// deployment shapes. Do NOT enable these here; they're documentation.
//
// Streaming / sub-10ms batches:
//   #define BOLT_LATENCY_GRAIN_BYTES       (16u * 1024u)
//   #define BOLT_PARALLEL_MIN_ROWS_FILTER  1
//   #define BOLT_PARALLEL_MIN_ROWS_SUM     1
//   #define BOLT_PARALLEL_MIN_ROWS_GROUPBY 1
//
// Bulk OLAP / 10-100M row scans:
//   #define BOLT_BALANCED_GRAIN_BYTES      (512u * 1024u)
//   #define BOLT_THROUGHPUT_GRAIN_BYTES    (4u * 1024u * 1024u)
//
// Constrained hosts (shared tenant, laptop):
//   #define BOLT_MAX_WORKERS               8u
//   #define BOLT_TASK_RING_SIZE            4096u
// ---------------------------------------------------------------------------
