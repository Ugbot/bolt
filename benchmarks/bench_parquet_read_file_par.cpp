// G2PQ-29 before/after benchmark: parquet_read_file's optional decode_pool.
//
// Exercises the ONE serial whole-file caller the ticket actually targets --
// a single blocking parquet_read_file() call with no outer (row-group- or
// stripe-level) parallelism of its own, exactly the shape of
// chukonu::io::load_parquet's bulk-load path (and gestaltd's
// GESTALT_TPCH_PARQUET_DIR startup loader). This is deliberately NOT the
// chukonu scan-operator shape G2PQ-34 already measured saturated.
//
// Usage: bench_parquet_read_file_par <file.parquet> [passes] [arena_gb]
//
// Correctness gate: every configuration (serial and every thread count) must
// decode to the IDENTICAL value-level checksum. A speedup number is not
// reported for a config that disagrees.
//
// Measurement hygiene (G2PQ-29/G2PQ-34 precedent): interleaved A/B/A/B across
// configs (not all-serial-then-all-parallel, which would let box load drift
// contaminate the comparison), min-of-N per config, fresh-arena reset between
// passes (not a fresh directory -- there is no file write here at all).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/ingest/bolt_parquet_read.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void mix(uint64_t& h, uint64_t v) noexcept {
    h ^= v;
    h *= 0x100000001b3ull;
}

uint64_t width_of(bolt::BoltType t) noexcept {
    using T = bolt::BoltType;
    switch (t) {
        case T::Int64: case T::Float64: case T::Decimal64:
        case T::UInt64: case T::Timestamp: case T::Duration: return 8;
        case T::Int32: case T::Float32: case T::UInt32:
        case T::Date32: return 4;
        case T::Int16: case T::UInt16: return 2;
        case T::Int8:  case T::UInt8:  case T::Bool: return 1;
        default: return 0;
    }
}

bool valid_at(const bolt::BoltColumn& c, int64_t r) noexcept {
    if (c.validity == nullptr) return true;
    const auto* v = static_cast<const uint8_t*>(static_cast<const void*>(c.validity));
    return (v[r >> 3] >> (r & 7)) & 1u;
}

// Value-level hash: type + row count + every row's value, sentinel for
// nulls, following the exact overflow-buffer indirection the reader uses
// for Utf8/Binary. Two decodes agree here only if they produced the same
// visible data -- buffer addresses and arena layout never enter the hash.
uint64_t hash_col(const bolt::BoltColumn& c, int64_t rows) {
    uint64_t h = 0xcbf29ce484222325ull;
    mix(h, static_cast<uint64_t>(c.type));
    mix(h, static_cast<uint64_t>(rows));
    if (rows <= 0) return h;
    if (c.type == bolt::BoltType::Utf8 || c.type == bolt::BoltType::Binary) {
        const auto* sv = static_cast<const bolt::StringView*>(c.data);
        for (int64_t r = 0; r < rows; ++r) {
            if (!valid_at(c, r)) { mix(h, 0xFFFFFFFFu); continue; }
            const uint32_t len = sv[r].length;
            const char* p = (len <= 12u)
                ? &sv[r].prefix[0]
                : static_cast<const char*>(c.str_overflow_base) + sv[r].ref.offset;
            mix(h, len);
            for (uint32_t k = 0; k < len; ++k) mix(h, static_cast<uint8_t>(p[k]));
        }
        return h;
    }
    const uint64_t w = width_of(c.type);
    if (w == 0 || c.data == nullptr) return h;
    const auto* b = static_cast<const uint8_t*>(c.data);
    for (int64_t r = 0; r < rows; ++r) {
        if (!valid_at(c, r)) { mix(h, 0xFFFFFFFFu); continue; }
        uint64_t v = 0;
        std::memcpy(&v, b + static_cast<uint64_t>(r) * w, w);
        mix(h, v);
    }
    return h;
}

uint64_t hash_batch(const bolt::BoltBatch& b) {
    uint64_t h = 0x100000001b3ull;
    mix(h, static_cast<uint64_t>(b.num_rows));
    mix(h, static_cast<uint64_t>(b.num_cols));
    const bolt::BoltColumn* cols = b.columns[b.read_epoch];
    for (uint32_t c = 0; c < b.num_cols; ++c) {
        h ^= hash_col(cols[c], b.num_rows) * (c + 1);
    }
    return h;
}

struct RunResult {
    bool     ok;
    double   ms;
    uint64_t checksum;
    int64_t  rows;
    uint32_t cols;
};

RunResult run_once(const uint8_t* buf, uint64_t len, uint64_t arena_gb,
                   bolt::Scheduler* pool) {
    bolt::ArenaConfig ac;
    ac.initial_block_size = 64ull << 20;
    ac.max_block_size     = (arena_gb << 30) / 32ull;
    bolt::Arena arena(ac);
    bolt::BoltBatch* b = arena.allocate_array<bolt::BoltBatch>(1);
    RunResult r{false, 0.0, 0, 0, 0};
    if (b == nullptr) return r;
    const auto t0 = Clock::now();
    r.ok = bolt::ingest::parquet::parquet_read_file(buf, len, &arena, b, pool);
    r.ms = ms_since(t0);
    if (!r.ok) return r;
    r.checksum = hash_batch(*b);
    r.rows = b->num_rows;
    r.cols = b->num_cols;
    return r;
}

int run(const char* path, int passes, uint64_t arena_gb) {
    std::vector<uint8_t> buf;
    {
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) { std::fprintf(stderr, "open failed: %s\n", path); return 2; }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        buf.resize(static_cast<size_t>(n));
        const size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (got != buf.size()) { std::fprintf(stderr, "short read\n"); return 2; }
    }
    std::fprintf(stderr, "%s: %.2f GB, %d passes, arena_gb=%llu\n", path,
                double(buf.size()) / double(1ull << 30), passes,
                (unsigned long long)arena_gb);

    // Configs: serial baseline, then 2/4/8 threads. Index 0 is always serial.
    const uint32_t thread_counts[] = {0u, 2u, 4u, 8u};
    constexpr uint32_t kConfigs = 4;
    bolt::Scheduler* pools[kConfigs] = {nullptr, nullptr, nullptr, nullptr};
    for (uint32_t i = 1; i < kConfigs; ++i) {
        pools[i] = new bolt::Scheduler();
        if (!pools[i]->init(thread_counts[i])) {
            std::fprintf(stderr, "scheduler init failed for %u threads\n",
                         thread_counts[i]);
            delete pools[i];
            pools[i] = nullptr;
        }
    }

    double best_ms[kConfigs];
    uint64_t checksum[kConfigs] = {0, 0, 0, 0};
    bool any_fail = false;
    for (uint32_t i = 0; i < kConfigs; ++i) best_ms[i] = 1e18;

    // Interleaved A/B/A/.../serial,2t,4t,8t,serial,2t,4t,8t,... across passes
    // so box-load drift hits every config equally rather than favoring
    // whichever ran first.
    for (int p = 0; p < passes; ++p) {
        for (uint32_t i = 0; i < kConfigs; ++i) {
            if (i != 0 && pools[i] == nullptr) continue;
            const RunResult r = run_once(buf.data(), buf.size(), arena_gb, pools[i]);
            if (!r.ok) {
                std::fprintf(stderr, "  pass %d config[%u threads]: DECODE FAILED\n",
                            p, thread_counts[i]);
                any_fail = true;
                continue;
            }
            if (checksum[i] == 0) checksum[i] = r.checksum;
            else if (checksum[i] != r.checksum) {
                std::fprintf(stderr, "  pass %d config[%u threads]: CHECKSUM MISMATCH "
                            "(own passes) got=%016llx want=%016llx\n", p,
                            thread_counts[i], (unsigned long long)r.checksum,
                            (unsigned long long)checksum[i]);
                any_fail = true;
            }
            if (r.ms < best_ms[i]) best_ms[i] = r.ms;
            std::fprintf(stderr, "  pass %d [%2u threads]: %9.1f ms  rows=%lld cols=%u\n",
                        p, thread_counts[i], r.ms, (long long)r.rows, r.cols);
        }
    }

    // Cross-config agreement: serial and every parallel config must have
    // decoded to the SAME value-level checksum.
    for (uint32_t i = 1; i < kConfigs; ++i) {
        if (pools[i] == nullptr) continue;
        if (checksum[i] != checksum[0]) {
            std::fprintf(stderr, "CROSS-CONFIG CHECKSUM MISMATCH: serial=%016llx "
                        "%u-threads=%016llx\n", (unsigned long long)checksum[0],
                        thread_counts[i], (unsigned long long)checksum[i]);
            any_fail = true;
        }
    }

    std::printf("checksum(serial)=%016llx  %s\n", (unsigned long long)checksum[0],
               any_fail ? "MISMATCH DETECTED" : "all configs value-identical");
    std::printf("%-10s %10s %10s\n", "threads", "best_ms", "speedup");
    for (uint32_t i = 0; i < kConfigs; ++i) {
        if (i != 0 && pools[i] == nullptr) continue;
        std::printf("%-10u %10.1f %9.2fx\n", thread_counts[i], best_ms[i],
                   best_ms[0] / best_ms[i]);
    }

    for (uint32_t i = 1; i < kConfigs; ++i) {
        if (pools[i] != nullptr) { pools[i]->shutdown(); delete pools[i]; }
    }
    return any_fail ? 3 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.parquet> [passes] [arena_gb]\n", argv[0]);
        return 1;
    }
    const int passes = (argc > 2) ? std::atoi(argv[2]) : 5;
    const uint64_t arena_gb = (argc > 3) ? static_cast<uint64_t>(std::atoi(argv[3])) : 24;
    return run(argv[1], passes, arena_gb);
}
