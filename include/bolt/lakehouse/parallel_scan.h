// bolt/lakehouse/parallel_scan.h — multi-threaded row-group scan + prefetch.
//
// One `bolt::scheduler::TaskPool`-style worker pool decodes row groups in
// parallel, while a `bolt::lakehouse::prefetch::Prefetcher` reads-ahead by
// `lookahead = 2 * parallelism` (capped at 32). The producer side is the
// existing scan driver; this module just runs the morsels through.
//
// W7 prefetcher provides the async object I/O; W8 wires it. Workers use
// std::thread state (heap is allowed for the thread struct only — every
// Morsel buffer is arena-allocated).
//
// Tiger Style: PODs, ≥2 asserts/fn, bounded, no exceptions.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/scan_optimizer.h"
#include "bolt/lakehouse/object_store.h"

namespace bolt {
namespace lakehouse {

// One row-group work unit. The worker thread will:
//   1. Fetch the file (prefetcher may have it warm).
//   2. Decode the listed row group into a BoltBatch in the work-unit arena.
//   3. Apply optional projection + predicates → Morsel.
//   4. Push the Morsel onto the result channel.
struct RowGroupTask {
    const char* file_path;       // borrowed; lifetime ≥ pool->shutdown
    uint32_t    row_group;
    uint32_t    slot_id;         // caller-supplied tag, echoed on result
};

struct ParallelScanConfig {
    uint32_t parallelism;        // worker thread count, capped at 16
    uint32_t lookahead;          // prefetch depth, capped at 32
    uint32_t inflight_max;       // capped at 64
    bool     use_prefetch;
    uint8_t  _pad[3];
};

// Initialise a config with the W8 defaults.
inline void parallel_scan_config_init(ParallelScanConfig* c) noexcept {
    assert(c != nullptr);
    if (c == nullptr) return;
    c->parallelism  = kScanDefaultParallel;
    c->lookahead    = kScanDefaultLookahead;
    c->inflight_max = 16u;
    c->use_prefetch = true;
}

// Clamp a config to W8 hard caps. Returns true if any field was clamped.
bool parallel_scan_config_clamp(ParallelScanConfig* c) noexcept;

// Opaque pool. Holds the worker threads + the request/result channels.
struct ParallelScanPool;

// Open. `store` is borrowed (for the prefetcher). `arena` backs the pool
// struct + work-unit allocs. Returns false on bad arg / oversize.
bool parallel_scan_open(
    ParallelScanPool** out, Arena* arena, ObjectStore* store,
    const ParallelScanConfig* cfg) noexcept;

// Submit a task (non-blocking). Returns false if the in-flight ring is
// full — caller should drain results.
bool parallel_scan_submit(
    ParallelScanPool* pool, const RowGroupTask* task) noexcept;

// Poll one morsel result with a timeout. Returns true if a Morsel was
// written into `*out`. `out_eof` is set true ONLY when the pool has
// been signalled `parallel_scan_finish()` AND drained.
bool parallel_scan_poll(
    ParallelScanPool* pool, Morsel* out, uint32_t timeout_ms,
    bool* out_eof) noexcept;

// Mark "no more submits". Drain semantics: poll until eof.
void parallel_scan_finish(ParallelScanPool* pool) noexcept;

// Close + join all worker threads. Safe to call on null.
void parallel_scan_close(ParallelScanPool* pool) noexcept;

// Diagnostic counters (one cache-line atomic). Useful from tests.
struct ParallelScanStats {
    uint64_t tasks_submitted;
    uint64_t morsels_produced;
    uint64_t prefetches_inflight;
    uint64_t bytes_fetched;
};

void parallel_scan_stats(const ParallelScanPool* pool,
                         ParallelScanStats* out) noexcept;

}  // namespace lakehouse
}  // namespace bolt
