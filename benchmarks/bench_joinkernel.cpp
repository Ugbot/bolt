// bench_joinkernel.cpp — Card A2 typed-join kernel microbench.
//
// Mirrors bench_hashjoin_multi_match.cpp but drives the TYPED, all-shapes
// SwissTable join (bolt/join/bolt_joinkernel.h): build/probe = 100k, ns/row
// reported per probe row. Covers the four arms the kernel must keep fast:
//   1. INNER single int64 key, unique build (OneInt64 fast path, avg-chain-1)
//   2. INNER single int64 key, dup build    (OneInt64 fast path, avg-chain-4)
//   3. INNER composite 2-key Int64          (General GbCell16 path)
//   4. INNER single Utf8 key, spilled bytes (General + sv_compare resolve)
//
// Timing model: the SwissTable BUILD is done ONCE outside the timed region;
// `bench_row` times ONLY the probe loop (and reports ns per probe row). This
// isolates probe cost so it is comparable to the SwissTable::find floor.
//
// Anchor: SwissTable::find is <= 5.36 ns/op when the table fits cache
// (docs/research/22-two-stage-cascades.md / Bolt swiss microbench). At 100k
// build keys the table is >L2, so each probe's find is largely DRAM-latency-
// bound (~tens of ns/miss) — the floors below are the cache-cold regime, not
// the cache-hot find floor. A probe = 1 hash + 1 find + chain walk + per-hit
// emit. Target ceilings for THIS microbench shape (single-thread, Release,
// AVX2 tier; PROBE only; measured on the dev host as the comment baseline):
//   uniq int64 (chain-1, OneInt64)  <= 25 ns/row   (raw-value find, no cell cmp)
//   dup  int64 (chain-4, OneInt64)  <= 45 ns/row   (find + 4-node walk + 4 emits)
//   composite 2-key Int64 (General) <= 170 ns/row  (GbCell16 fold + cell cmp,
//                                                    cold 64-partition tables)
//   Utf8 spilled key (General)      <= 220 ns/row  (utf8_hash_one + sv_compare
//                                                    over ~28-byte spilled keys)
// These are ceilings for the microbench shape, NOT gates on a query. The
// OneInt64 fast path is the one that must stay tight; the General path trades
// speed for type coverage (composite/Decimal/Utf8) and is cache-miss-bound at
// this cardinality. Probe emits index pairs only (no payload gather — that is
// the operator's job).

#include "bolt/bolt.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_joinkernel.h"
#include "bolt/kernels/bolt_utf8.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using ns    = std::chrono::nanoseconds;

template <typename T>
BOLT_FORCE_INLINE void DoNotOptimize(T&& v) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    (void)v; _ReadWriteBarrier();
#else
    asm volatile("" : : "g"(v) : "memory");
#endif
}

template <typename Fn>
static double bench_row(const char* name, int64_t rows, int iters,
                        Fn&& fn) noexcept {
    fn();  // warmup
    int64_t best = INT64_MAX;
    for (int it = 0; it < iters; ++it) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        int64_t dt = std::chrono::duration_cast<ns>(t1 - t0).count();
        if (dt < best) best = dt;
    }
    double per = static_cast<double>(best) / static_cast<double>(rows);
    std::printf("  %-44s %8.2f ns/row (best %8lld ns)\n",
                name, per, static_cast<long long>(best));
    return per;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive a crash
    constexpr int64_t kBuild = 100000;
    constexpr int64_t kProbe = 100000;
    constexpr int     kIters = 5;

    std::mt19937_64 rng(0x10AD);

    // ---- int64 single-key, unique build ----
    std::vector<int64_t> bk_uniq(kBuild), pk_uniq(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) bk_uniq[i] = i;
    {
        std::uniform_int_distribution<int64_t> d(0, kBuild - 1);
        for (int64_t i = 0; i < kProbe; ++i) pk_uniq[i] = d(rng);
    }

    // ---- int64 single-key, dup build (avg chain 4) ----
    constexpr int64_t kDupKeys = kBuild / 4;
    std::vector<int64_t> bk_dup(kBuild), pk_dup(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) bk_dup[i] = i % kDupKeys;
    {
        std::uniform_int_distribution<int64_t> d(0, kDupKeys - 1);
        for (int64_t i = 0; i < kProbe; ++i) pk_dup[i] = d(rng);
    }

    // ---- composite 2-key Int64 ----
    std::vector<int64_t> bk_a(kBuild), bk_b(kBuild);
    std::vector<int64_t> pk_a(kProbe), pk_b(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) { bk_a[i] = i / 1000; bk_b[i] = i % 1000; }
    {
        std::uniform_int_distribution<int64_t> d(0, kBuild - 1);
        for (int64_t i = 0; i < kProbe; ++i) {
            int64_t r = d(rng);
            pk_a[i] = r / 1000; pk_b[i] = r % 1000;
        }
    }

    // ---- Utf8 single-key, spilled (>12-byte) strings ----
    // Unique build keys (avg-chain-1) so the emitted-pair count stays ~kProbe
    // and fits the output scratch; probe samples uniformly from the build keys.
    // Bytes spill into one contiguous pool per side; ref.offset is relative to
    // that pool (the column's str_overflow_base).
    const int kCard = static_cast<int>(kBuild);
    std::vector<std::string> dict(kCard);
    for (int i = 0; i < kCard; ++i) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "supplier_key_prefix_%08d", i);  // 28 chars
        dict[i] = buf;
    }
    auto build_utf8_side = [&](const std::vector<int>& ids,
                               std::vector<char>& pool,
                               std::vector<bolt::StringView>& views) {
        size_t total = 0;
        for (int id : ids) total += dict[id].size();
        pool.assign(total, 0);
        views.resize(ids.size());
        uint32_t cur = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const std::string& s = dict[ids[i]];
            bolt::StringView v;
            std::memset(&v, 0, sizeof(v));
            v.length = static_cast<uint32_t>(s.size());
            std::memcpy(v.prefix, s.data(), 4);
            std::memcpy(pool.data() + cur, s.data(), s.size());
            v.ref.buf_idx = 0;
            v.ref.offset  = cur;
            cur += static_cast<uint32_t>(s.size());
            views[i] = v;
        }
    };
    std::vector<int> build_ids(kBuild), probe_ids(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) build_ids[i] = static_cast<int>(i);  // unique
    {
        std::uniform_int_distribution<int> d(0, kCard - 1);
        for (int64_t i = 0; i < kProbe; ++i) probe_ids[i] = d(rng);
    }
    std::vector<char> bpool, ppool;
    std::vector<bolt::StringView> bviews, pviews;
    build_utf8_side(build_ids, bpool, bviews);
    build_utf8_side(probe_ids, ppool, pviews);

    bolt::ArenaConfig acfg{};
    acfg.initial_block_size = 64 * 1024 * 1024;
    acfg.max_block_size     = 256 * 1024 * 1024;

    // Output index-pair scratch. Dup case can emit ~4x probe rows.
    std::vector<int32_t> ob(kProbe * 8), op(kProbe * 8);

    std::printf("Card A2 typed-join microbench (build=%lld probe=%lld iters=%d)\n",
                static_cast<long long>(kBuild),
                static_cast<long long>(kProbe), kIters);
    std::printf("  (build done once, untimed; ns/row is PROBE cost per row)\n");

    // One arena holds every pre-built table (build is untimed; probe is the
    // measured hot loop). Each table's BoltColumns persist for the lambda.
    bolt::Arena arena{acfg};

    // --- 1. INNER single int64 key, unique build (OneInt64, chain-1) ---
    bolt::BoltColumn b_uniq = bolt::BoltColumn::make_flat(
        bk_uniq.data(), nullptr, kBuild, bolt::BoltType::Int64);
    bolt::BoltColumn p_uniq = bolt::BoltColumn::make_flat(
        pk_uniq.data(), nullptr, kProbe, bolt::BoltType::Int64);
    bolt::JoinBuildTyped jb_uniq{};
    if (!bolt::join_build_typed(&b_uniq, 1, kBuild, &arena, &jb_uniq)) return 1;
    bench_row("INNER 1xint64 uniq build (OneInt64, chain-1)", kProbe, kIters, [&]{
        size_t n = bolt::join_probe_typed(&jb_uniq, &p_uniq, kProbe,
                                          ob.data(), op.data(), ob.size());
        DoNotOptimize(n);
    });

    // --- 2. INNER single int64 key, dup build (OneInt64, chain-4) ---
    bolt::BoltColumn b_dup = bolt::BoltColumn::make_flat(
        bk_dup.data(), nullptr, kBuild, bolt::BoltType::Int64);
    bolt::BoltColumn p_dup = bolt::BoltColumn::make_flat(
        pk_dup.data(), nullptr, kProbe, bolt::BoltType::Int64);
    bolt::JoinBuildTyped jb_dup{};
    if (!bolt::join_build_typed(&b_dup, 1, kBuild, &arena, &jb_dup)) return 1;
    bench_row("INNER 1xint64 dup build  (OneInt64, chain-4)", kProbe, kIters, [&]{
        size_t n = bolt::join_probe_typed(&jb_dup, &p_dup, kProbe,
                                          ob.data(), op.data(), ob.size());
        DoNotOptimize(n);
    });

    // --- 3. INNER composite 2-key Int64 (General GbCell16 path) ---
    bolt::BoltColumn bcols[2] = {
        bolt::BoltColumn::make_flat(bk_a.data(), nullptr, kBuild, bolt::BoltType::Int64),
        bolt::BoltColumn::make_flat(bk_b.data(), nullptr, kBuild, bolt::BoltType::Int64)};
    bolt::BoltColumn pcols[2] = {
        bolt::BoltColumn::make_flat(pk_a.data(), nullptr, kProbe, bolt::BoltType::Int64),
        bolt::BoltColumn::make_flat(pk_b.data(), nullptr, kProbe, bolt::BoltType::Int64)};
    bolt::JoinBuildTyped jb_comp{};
    if (!bolt::join_build_typed(bcols, 2, kBuild, &arena, &jb_comp)) return 1;
    bench_row("INNER composite 2-key Int64 (General)", kProbe, kIters, [&]{
        size_t n = bolt::join_probe_typed(&jb_comp, pcols, kProbe,
                                          ob.data(), op.data(), ob.size());
        DoNotOptimize(n);
    });

    // --- 4. INNER single Utf8 key, spilled bytes (General + sv_compare) ---
    bolt::BoltColumn b_str = bolt::BoltColumn::make_flat(
        bviews.data(), nullptr, kBuild, bolt::BoltType::Utf8);
    b_str.str_overflow_base = bpool.data();
    bolt::BoltColumn p_str = bolt::BoltColumn::make_flat(
        pviews.data(), nullptr, kProbe, bolt::BoltType::Utf8);
    p_str.str_overflow_base = ppool.data();
    bolt::JoinBuildTyped jb_str{};
    if (!bolt::join_build_typed(&b_str, 1, kBuild, &arena, &jb_str)) return 1;
    bench_row("INNER 1xUtf8 spilled key (General+sv_cmp)", kProbe, kIters, [&]{
        size_t n = bolt::join_probe_typed(&jb_str, &p_str, kProbe,
                                          ob.data(), op.data(), ob.size());
        DoNotOptimize(n);
    });

    // --- 5. BUILD cost (the Q05/Q07/Q09/Q18 tail driver). The probe rows above
    //     are untimed-build; here we time join_build_typed itself. Each iter uses
    //     a dedicated arena reset to its first block so allocation is amortized
    //     out and we measure the kernel's per-row insert cost, not malloc. ---
    std::printf("  --- build cost (ns per BUILD row) ---\n");
    bolt::Arena barena{acfg};
    auto bench_build = [&](const char* name, const bolt::BoltColumn* cols,
                           uint8_t n_keys) {
        bolt::JoinBuildTyped jb{};
        // warmup (also faults in the arena blocks)
        if (!bolt::join_build_typed(cols, n_keys, kBuild, &barena, &jb)) return;
        int64_t best = INT64_MAX;
        for (int it = 0; it < kIters; ++it) {
            barena.reset();
            bolt::JoinBuildTyped jbi{};
            auto t0 = Clock::now();
            bool ok = bolt::join_build_typed(cols, n_keys, kBuild, &barena, &jbi);
            auto t1 = Clock::now();
            DoNotOptimize(ok);
            int64_t dt = std::chrono::duration_cast<ns>(t1 - t0).count();
            if (dt < best) best = dt;
        }
        std::printf("  %-44s %8.2f ns/row (best %8lld ns)\n", name,
                    static_cast<double>(best) / static_cast<double>(kBuild),
                    static_cast<long long>(best));
    };
    bench_build("BUILD 1xint64 uniq (OneInt64, chain-1)", &b_uniq, 1);
    bench_build("BUILD 1xint64 dup  (OneInt64, chain-4)", &b_dup, 1);
    bench_build("BUILD composite 2-key Int64 (General)", bcols, 2);

    std::printf("OK bench_joinkernel\n");
    return 0;
}
