// bench_salt_probe.cpp — G2FEAT-89/362 go/no-go micro-benchmark.
//
// Question: does co-locating a SALT (upper hash bits) with the key+accumulator
// in ONE open-addressed entry beat bolt::SwissTable's current layout (a dense
// control-byte array + a SEPARATE key/accumulator array) for an aggregation
// INGEST probe (find-or-insert + accumulate)?
//
// This isolates exactly the lever named in G2FEAT-89/-362 and
// docs/intent/single-box-engine-strategy.md: our own profiling attributes
// ~94% of ClickBench Q16/Q10 wall time to this probe, and it is measured
// PARALLELISM-NEGATIVE (62 ns/row/worker @8t vs 32 ns/row @1t) because
// workers contend for DRAM bandwidth on a large (>L3) live table. A layout
// that is faster at 1 thread but still DRAM-bandwidth-bound at 8 threads is
// NOT a win for the real problem — so every approach below is measured at
// BOTH 1 and 8 threads, each thread owning its own full-size table (mirrors
// chukonu's per-worker SwissTable ingest shape).
//
// Four approaches, all against the REAL bolt primitives (no reimplementation
// of the control-plane structure being tested):
//
//   A — current: bolt::SwissTable.find/insert (ctrl-byte array, SIMD group
//       scan) -> slot index into a SEPARATE accum[] array. Two arrays, so a
//       hit typically costs one ctrl-line miss + one accum-line miss.
//   D — control: bolt::SwissTableInterleaved (existing "per-group co-location"
//       alternative already in bolt_swiss.h) -> slot index into a SEPARATE
//       accum[] array. This is the ALREADY-TRIED, ALREADY-MEASURED-WEAK lever
//       (~6% in prior work, attributed to "MLP already overlaps the loads") —
//       included here as a same-harness sanity check that this bench
//       reproduces known behavior before trusting a NEW number from it.
//   B — salt-in-entry (DuckDB/Photon-style): one open-addressed array of
//       {salt, key, accum} entries. A probe touches ONE cache line for the
//       common (small-scan) case instead of two.
//   C — B + windowed software prefetch (ClickHouse/StarRocks-style): compute
//       a window of hashes first, prefetch each target entry line, then probe
//       — hides miss latency across the window instead of stalling per-row.
//
// Correctness by construction: all four approaches compute COUNT(*) grouped
// by key over the IDENTICAL key stream; each approach's (distinct groups,
// total count) must match every other approach's exactly, or the benchmark
// fails loudly (assert) rather than reporting a silently-wrong number.
//
// Tiger Style: no heap allocation inside the timed region (arenas sized
// up front); every loop has a fixed bound; assertions on both pre- and
// post-conditions; no exceptions.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/join/bolt_swiss.h"

using namespace bolt;
using clk = std::chrono::steady_clock;

namespace {

constexpr int      kMaxWorkers   = 8;
constexpr int      kMaxReps      = 3;
constexpr uint64_t kMaxRowsCap   = 100'000'000ull;  // hard bound, Tiger Style

double ms_since(clk::time_point t0) noexcept {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

Arena* make_arena(size_t max_block) noexcept {
    assert(max_block > 0);
    ArenaConfig c{};
    c.initial_block_size = (max_block < (64u << 20)) ? max_block : (64u << 20);
    c.max_block_size     = max_block;
    return new Arena(c);
}

uint64_t next_pow2(uint64_t x) noexcept {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// Approach A — bolt::SwissTable (ctrl-byte array) + SEPARATE accum[] array.
// This is the REAL, unmodified bolt primitive as it exists in production.
// ---------------------------------------------------------------------------
void approach_A(const int64_t* keys, int64_t n, uint64_t distinct_hint,
                int64_t* out_distinct, int64_t* out_total, double* out_ms) noexcept {
    assert(keys != nullptr);
    assert(n > 0);
    Arena* a = make_arena(size_t(2) << 30);
    SwissTable t{};
    bool ok = SwissTable::create(&t, distinct_hint, a);
    assert(ok); (void)ok;
    int64_t* accum = a->allocate_array<int64_t>(size_t(t.capacity) + 1);
    assert(accum != nullptr);
    int32_t size = 0;

    auto t0 = clk::now();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t key = keys[i];
        int32_t slot = t.find(static_cast<uint64_t>(key));
        if (slot < 0) {
            slot = size++;
            t.insert(static_cast<uint64_t>(key), static_cast<uint32_t>(slot));
            accum[slot] = 1;
        } else {
            accum[slot]++;
        }
    }
    *out_ms = ms_since(t0);

    int64_t total = 0;
    for (int32_t si = 0; si < size; ++si) total += accum[si];
    *out_distinct = size;
    *out_total    = total;
    delete a;
}

// ---------------------------------------------------------------------------
// Approach D — bolt::SwissTableInterleaved (existing group-colocated
// alternative) + SEPARATE accum[] array. Sanity-check control: this is the
// documented ~6% dead end; if this bench doesn't roughly reproduce "small
// win, not a fix" here too, the harness itself is suspect.
// ---------------------------------------------------------------------------
void approach_D(const int64_t* keys, int64_t n, uint64_t distinct_hint,
                int64_t* out_distinct, int64_t* out_total, double* out_ms) noexcept {
    assert(keys != nullptr);
    assert(n > 0);
    Arena* a = make_arena(size_t(2) << 30);
    SwissTableInterleaved t{};
    bool ok = SwissTableInterleaved::create(&t, distinct_hint, a);
    assert(ok); (void)ok;
    int64_t* accum = a->allocate_array<int64_t>(size_t(t.capacity) + 1);
    assert(accum != nullptr);
    int32_t size = 0;

    auto t0 = clk::now();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t key = keys[i];
        int32_t slot = t.find(static_cast<uint64_t>(key));
        if (slot < 0) {
            slot = size++;
            t.insert(static_cast<uint64_t>(key), static_cast<uint32_t>(slot));
            accum[slot] = 1;
        } else {
            accum[slot]++;
        }
    }
    *out_ms = ms_since(t0);

    int64_t total = 0;
    for (int32_t si = 0; si < size; ++si) total += accum[si];
    *out_distinct = size;
    *out_total    = total;
    delete a;
}

// ---------------------------------------------------------------------------
// Approaches B/C — salt-in-entry. NOT part of bolt_swiss.h yet (Phase 1 is
// measure-first); this is a local, honestly-sized prototype entry. Real
// key+accum for a plain int64 aggregate need 8+8=16 bytes already; adding a
// salt needs 8-byte alignment for the following int64 fields, so the entry
// is 32 bytes (2 per 64B cache line, none straddling) — NOT the idealized
// 16 bytes a pointer-based (DuckDB `ht_entry_t`-style) design could reach.
// Reported honestly; the memory-footprint cost of this is part of the
// go/no-go call, not hidden from it.
// ---------------------------------------------------------------------------
struct SaltEntry {
    uint64_t salt;   // bit63 = occupied; low bits = hash tag
    int64_t  key;
    int64_t  accum;
    int64_t  _reserved;  // pads to 32B so entries never straddle a cache line
};
static_assert(sizeof(SaltEntry) == 32, "SaltEntry must be 32 bytes (2/cache line)");

constexpr uint64_t kSaltOccupied = uint64_t(1) << 63;

void approach_B(const int64_t* keys, int64_t n, uint64_t distinct_hint, bool use_prefetch,
                int64_t* out_distinct, int64_t* out_total, double* out_ms) noexcept {
    assert(keys != nullptr);
    assert(n > 0);
    const uint64_t cap  = next_pow2(distinct_hint + (distinct_hint >> 1) + 64);  // ~66% load
    const uint64_t mask = cap - 1;
    Arena* a = make_arena(size_t(2) << 30);
    auto* e = a->allocate_array<SaltEntry>(cap);
    assert(e != nullptr);
    memset(e, 0, sizeof(SaltEntry) * cap);
    int64_t distinct = 0;

    auto t0 = clk::now();
    if (!use_prefetch) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t  key  = keys[i];
            const uint64_t h    = swiss_mix(static_cast<uint64_t>(key));
            const uint64_t salt = (h & (kSaltOccupied - 1)) | kSaltOccupied;
            uint64_t slot = h & mask;
            for (;;) {
                SaltEntry& c = e[slot];  // ONE cache line: salt + key + accum
                if (c.salt == 0) { c.key = key; c.accum = 1; c.salt = salt; ++distinct; break; }
                if (c.salt == salt && c.key == key) { ++c.accum; break; }
                slot = (slot + 1) & mask;
            }
        }
    } else {
        constexpr int64_t kWindow = 32;
        uint64_t hbuf[kWindow];
        uint64_t sbuf[kWindow];
        for (int64_t base = 0; base < n; base += kWindow) {
            const int64_t m = (base + kWindow <= n) ? kWindow : (n - base);
            for (int64_t j = 0; j < m; ++j) {
                const uint64_t h = swiss_mix(static_cast<uint64_t>(keys[base + j]));
                hbuf[j] = h;
                sbuf[j] = h & mask;
                BOLT_PREFETCH_WRITE(&e[sbuf[j]]);
            }
            for (int64_t j = 0; j < m; ++j) {
                const int64_t  key  = keys[base + j];
                const uint64_t salt = (hbuf[j] & (kSaltOccupied - 1)) | kSaltOccupied;
                uint64_t slot = sbuf[j];
                for (;;) {
                    SaltEntry& c = e[slot];
                    if (c.salt == 0) { c.key = key; c.accum = 1; c.salt = salt; ++distinct; break; }
                    if (c.salt == salt && c.key == key) { ++c.accum; break; }
                    slot = (slot + 1) & mask;
                }
            }
        }
    }
    *out_ms = ms_since(t0);

    int64_t total = 0;
    for (uint64_t si = 0; si < cap; ++si) total += e[si].accum;
    *out_distinct = distinct;
    *out_total    = total;
    delete a;
}

// ---------------------------------------------------------------------------
// Driver: one (distinct, rows) size point, at a given worker count. Each
// worker gets its OWN full-size table + its own shard of the row stream —
// mirrors chukonu's per-worker independent-SwissTable ingest shape, which is
// exactly what makes the current layout parallelism-negative (workers
// contend for DRAM bandwidth on independent large tables, not on shared
// state). A layout that only wins with one live table would not transfer.
// ---------------------------------------------------------------------------
struct SizePoint {
    uint64_t distinct;
    uint64_t rows;
    int      reps;
    const char* label;
};

void run_one(const char* approach_name, int which, bool prefetch_variant,
            const int64_t* keys, uint64_t rows, uint64_t distinct,
            int workers, int reps, double* out_best_ns_per_row) noexcept {
    assert(keys != nullptr);
    assert(rows > 0);
    assert(workers >= 1 && workers <= kMaxWorkers);
    assert(reps >= 1 && reps <= kMaxReps);

    const uint64_t shard = rows / static_cast<uint64_t>(workers);
    double best_ms = 1e300;

    for (int rep = 0; rep < reps; ++rep) {
        std::thread threads[kMaxWorkers];
        int64_t dd[kMaxWorkers] = {0};
        int64_t tt[kMaxWorkers] = {0};
        double  wms[kMaxWorkers] = {0};

        auto t0 = clk::now();
        for (int w = 0; w < workers; ++w) {
            threads[w] = std::thread([&, w] {
                const int64_t* k = keys + static_cast<int64_t>(w) * static_cast<int64_t>(shard);
                switch (which) {
                    case 0: approach_A(k, static_cast<int64_t>(shard), distinct, &dd[w], &tt[w], &wms[w]); break;
                    case 1: approach_D(k, static_cast<int64_t>(shard), distinct, &dd[w], &tt[w], &wms[w]); break;
                    default: approach_B(k, static_cast<int64_t>(shard), distinct, prefetch_variant, &dd[w], &tt[w], &wms[w]); break;
                }
            });
        }
        for (int w = 0; w < workers; ++w) threads[w].join();
        const double wall_ms = ms_since(t0);

        // Correctness: every worker's total must equal its shard size.
        for (int w = 0; w < workers; ++w) {
            assert(tt[w] == static_cast<int64_t>(shard));
        }
        if (wall_ms < best_ms) best_ms = wall_ms;

        std::printf("    [rep%d] %-16s workers=%d  wall=%8.1f ms  ns/row=%6.2f\n",
                    rep, approach_name, workers, wall_ms,
                    wall_ms * 1e6 / static_cast<double>(rows));
    }
    *out_best_ns_per_row = best_ms * 1e6 / static_cast<double>(rows);
}

// Verify all approaches agree on (distinct groups, total) for a single-
// worker, single-rep pass — correctness gate before any timing is trusted.
void verify_correctness(const int64_t* keys, int64_t n, uint64_t distinct_hint) noexcept {
    assert(keys != nullptr);
    assert(n > 0);
    int64_t dA=0, tA=0, dD=0, tD=0, dB=0, tB=0, dC=0, tC=0;
    double msA=0, msD=0, msB=0, msC=0;
    approach_A(keys, n, distinct_hint, &dA, &tA, &msA);
    approach_D(keys, n, distinct_hint, &dD, &tD, &msD);
    approach_B(keys, n, distinct_hint, false, &dB, &tB, &msB);
    approach_B(keys, n, distinct_hint, true,  &dC, &tC, &msC);
    assert(tA == n && tD == n && tB == n && tC == n);
    assert(dA == dD && dA == dB && dA == dC);
    std::printf("  correctness OK: distinct=%lld total=%lld (A/D/B/C agree)\n",
                static_cast<long long>(dA), static_cast<long long>(tA));
}

}  // namespace

int main(int argc, char** argv) {
    // Optional single-size override: bench_salt_probe <distinct> <rows> <reps>
    uint64_t override_distinct = 0, override_rows = 0;
    int      override_reps = 0;
    if (argc >= 3) {
        override_distinct = static_cast<uint64_t>(std::strtoull(argv[1], nullptr, 10));
        override_rows     = static_cast<uint64_t>(std::strtoull(argv[2], nullptr, 10));
        override_reps      = (argc >= 4) ? std::atoi(argv[3]) : 2;
        if (override_rows > kMaxRowsCap) override_rows = kMaxRowsCap;
    }

    const SizePoint sweep[] = {
        {2'000,       5'000'000, 2, "L1 (2K distinct)"},
        {50'000,      8'000'000, 2, "L2 (50K distinct)"},
        {1'000'000,  20'000'000, 2, "L3-boundary (1M distinct)"},
        {4'000'000,  40'000'000, 2, "DRAM (4M distinct)"},
        {17'000'000,100'000'000, 3, "DRAM, prod-matched (17M distinct, Q16 shape)"},
    };
    const int n_sizes = override_distinct ? 1 : static_cast<int>(sizeof(sweep) / sizeof(sweep[0]));

    std::printf("== bench_salt_probe (G2FEAT-89/362 go/no-go) ==\n");
    std::printf("hardware_concurrency=%u\n", std::thread::hardware_concurrency());

    for (int si = 0; si < n_sizes; ++si) {
        const uint64_t distinct = override_distinct ? override_distinct : sweep[si].distinct;
        const uint64_t rows_req = override_distinct ? override_rows     : sweep[si].rows;
        const int      reps     = override_distinct ? (override_reps > 0 ? override_reps : 2) : sweep[si].reps;
        const char*    label    = override_distinct ? "override" : sweep[si].label;

        // Round rows down to a multiple of kMaxWorkers so every worker-count
        // divides evenly (fair shard sizes at 1 and 8 workers alike).
        const uint64_t rows = (rows_req / kMaxWorkers) * kMaxWorkers;
        assert(rows > 0);

        const uint64_t table_bytes_A = distinct * 2 * (sizeof(SwissSlot) + 1) + distinct * 2 * sizeof(int64_t);
        const uint64_t table_bytes_B = next_pow2(distinct + (distinct >> 1) + 64) * sizeof(SaltEntry);
        std::printf("\n-- size: %s  distinct=%llu rows=%llu  "
                    "(approx table bytes: A=%.1fMB B=%.1fMB) --\n",
                    label, (unsigned long long)distinct, (unsigned long long)rows,
                    table_bytes_A / 1e6, table_bytes_B / 1e6);

        int64_t* keys = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * rows));
        assert(keys != nullptr);
        uint64_t s = 0x9E3779B97F4A7C15ull ^ (distinct * 0xBF58476D1CE4E5B9ull);
        for (uint64_t i = 0; i < rows; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            keys[i] = static_cast<int64_t>(s % distinct);
        }

        verify_correctness(keys, static_cast<int64_t>(rows > 20'000'000 ? 5'000'000 : rows), distinct);

        for (int workers : {1, 8}) {
            std::printf("  -- workers=%d --\n", workers);
            double a_ns=0, d_ns=0, b_ns=0, c_ns=0;
            run_one("A-current",    0, false, keys, rows, distinct, workers, reps, &a_ns);
            run_one("D-interleaved",1, false, keys, rows, distinct, workers, reps, &d_ns);
            run_one("B-salt",       2, false, keys, rows, distinct, workers, reps, &b_ns);
            run_one("C-salt+pf",    2, true,  keys, rows, distinct, workers, reps, &c_ns);
            std::printf("  SUMMARY workers=%d  A=%.2f  D=%.2f(%.2fx)  B=%.2f(%.2fx)  C=%.2f(%.2fx) ns/row\n",
                        workers, a_ns,
                        d_ns, (d_ns > 0) ? a_ns / d_ns : 0.0,
                        b_ns, (b_ns > 0) ? a_ns / b_ns : 0.0,
                        c_ns, (c_ns > 0) ? a_ns / c_ns : 0.0);
        }

        std::free(keys);
    }

    std::printf("\n== done ==\n");
    return 0;
}
