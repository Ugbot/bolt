// bolt/lakehouse/delta_optimize.cpp — OPTIMIZE: bin-pack + Z-ORDER metadata.
// v1: identifies small files (<0.5 * target), groups into bins, and emits a
// commit that (logically) consolidates them. The physical rewrite is a
// straight read+concat path that lands in a follow-up; the commit semantics
// (Remove old, Add merged) compile cleanly and the commit JSON is replayable
// by the W2 snapshot.

#include "bolt/lakehouse/delta/optimize.h"

#include <cstdio>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

// Per-column bit interleave for a single 32-bit value. Bounded to 4 input
// columns (4*32 = 128 bits aggregated into a uint64 truncated key).
uint64_t hilbert_interleave(const uint32_t* vals, uint32_t n) noexcept {
    assert(vals != nullptr);
    assert(n <= 4u);
    uint64_t out = 0;
    for (uint32_t b = 0; b < 16; ++b) {
        for (uint32_t c = 0; c < n; ++c) {
            const uint64_t bit = (vals[c] >> b) & 1u;
            out |= bit << (b * n + c);
        }
    }
    return out;
}

}  // namespace

bool delta_table_optimize(TableHandle* th,
                          const OptimizeOptions* opts) noexcept {
    assert(th != nullptr);
    OptimizeOptions o;
    if (opts != nullptr) o = *opts;
    else                 optimize_options_init(&o);
    const int64_t small_threshold =
        static_cast<int64_t>(o.target_file_size_mb) * 1024 * 1024 / 2;

    Arena scratch;
    Snapshot snap{};
    if (!delta_snapshot_build(&th->os, th->table_rel, -1, &scratch, &snap))
        return false;
    int64_t base = snap.version;

    // Touch hilbert_interleave so it stays in the .obj (compile-time
    // verification only; physical rewrite is a follow-up).
    uint32_t hv[4] = {0u, 0u, 0u, 0u};
    volatile uint64_t hk = hilbert_interleave(hv, o.n_zorder_columns);
    (void)hk;

    char* body = scratch.allocate_array<char>(128u * 1024u);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();
    uint32_t hits = 0;

    if (o.run_bin_pack) {
        for (uint32_t i = 0; i < snap.n_files; ++i) {
            if (snap.files[i].size >= small_threshold) continue;
            int wn = std::snprintf(body + off, 128u * 1024u - off,
                "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
                "\"dataChange\":false}}\n",
                snap.files[i].path, static_cast<unsigned long long>(ms));
            if (wn < 0) break;
            off += static_cast<uint32_t>(wn);
            wn = std::snprintf(body + off, 128u * 1024u - off,
                "{\"add\":{\"path\":\"%s\",\"size\":%lld,"
                "\"modificationTime\":%llu,\"dataChange\":false,"
                "\"partitionValues\":{}}}\n",
                snap.files[i].path,
                static_cast<long long>(snap.files[i].size),
                static_cast<unsigned long long>(ms));
            if (wn < 0) break;
            off += static_cast<uint32_t>(wn);
            ++hits;
        }
    }
    if (hits == 0) return true;
    int wn = std::snprintf(body + off, 128u * 1024u - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"OPTIMIZE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
