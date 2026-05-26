// bench_hashjoin_multi_match.cpp — K-JOIN-MM kernel microbench.
//
// Three scenarios, ns/row reported per probe row:
//   single-key, unique build (avg-chain-1, fast path)
//   single-key, dup build (avg-chain-4)
//   composite-key (2 keys) Int64 build
//
// Floors (per card):
//   uniq build       <= 5.5 ns/row
//   dup build (4)    <= 9.0 ns/row
//   composite-2-key  <= 8.0 ns/row

#include "bolt/bolt.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/join/bolt_hashjoin.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
    constexpr int64_t kBuild = 100000;
    constexpr int64_t kProbe = 100000;
    constexpr int     kIters = 5;

    std::mt19937_64 rng(0x10AD);

    std::vector<uint64_t> bk_uniq(kBuild), pk_uniq(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) bk_uniq[i] = static_cast<uint64_t>(i);
    {
        std::uniform_int_distribution<int64_t> d(0, kBuild - 1);
        for (int64_t i = 0; i < kProbe; ++i) pk_uniq[i] = static_cast<uint64_t>(d(rng));
    }

    constexpr int64_t kDupKeys = kBuild / 4;
    std::vector<uint64_t> bk_dup(kBuild), pk_dup(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) bk_dup[i] = static_cast<uint64_t>(i % kDupKeys);
    {
        std::uniform_int_distribution<int64_t> d(0, kDupKeys - 1);
        for (int64_t i = 0; i < kProbe; ++i) pk_dup[i] = static_cast<uint64_t>(d(rng));
    }

    std::vector<uint64_t> bk_a(kBuild), bk_b(kBuild);
    std::vector<uint64_t> pk_a(kProbe), pk_b(kProbe);
    for (int64_t i = 0; i < kBuild; ++i) {
        bk_a[i] = static_cast<uint64_t>(i / 1000);
        bk_b[i] = static_cast<uint64_t>(i % 1000);
    }
    {
        std::uniform_int_distribution<int64_t> d(0, kBuild - 1);
        for (int64_t i = 0; i < kProbe; ++i) {
            int64_t r = d(rng);
            pk_a[i] = static_cast<uint64_t>(r / 1000);
            pk_b[i] = static_cast<uint64_t>(r % 1000);
        }
    }

    bolt::ArenaConfig acfg{};
    acfg.initial_block_size = 64 * 1024 * 1024;
    acfg.max_block_size     = 256 * 1024 * 1024;

    std::vector<uint32_t> pb(kProbe * 8), pp(kProbe * 8);

    std::printf("K-JOIN-MM microbench (build=%lld probe=%lld iters=%d)\n",
                static_cast<long long>(kBuild),
                static_cast<long long>(kProbe), kIters);

    bench_row("single-key uniq build (avg-chain-1)", kProbe, kIters, [&]{
        bolt::Arena arena{acfg};
        bolt::HashJoinBuildChained b{};
        bolt::HashJoinBuildCfgChained cfg{};
        cfg.n_keys = 1; cfg.keys[0] = bolt::BoltType::UInt64;
        cfg.build_rows = static_cast<uint64_t>(kBuild);
        cfg.expected_dup_factor = 1;
        cfg.max_chain_len = bolt::kHJMaxChainLen;
        const uint64_t* lk[1] = { bk_uniq.data() };
        const uint64_t* rk[1] = { pk_uniq.data() };
        bool ok = bolt::hash_join_build_chained(lk, cfg, &arena, &b);
        (void)ok;
        size_t n = bolt::hash_join_probe_multi_match(
            &b, lk, rk, kProbe, pb.data(), pp.data(), pb.size());
        DoNotOptimize(n);
    });

    bench_row("single-key dup build  (avg-chain-4)", kProbe, kIters, [&]{
        bolt::Arena arena{acfg};
        bolt::HashJoinBuildChained b{};
        bolt::HashJoinBuildCfgChained cfg{};
        cfg.n_keys = 1; cfg.keys[0] = bolt::BoltType::UInt64;
        cfg.build_rows = static_cast<uint64_t>(kBuild);
        cfg.expected_dup_factor = 4;
        cfg.max_chain_len = bolt::kHJMaxChainLen;
        const uint64_t* lk[1] = { bk_dup.data() };
        const uint64_t* rk[1] = { pk_dup.data() };
        bool ok = bolt::hash_join_build_chained(lk, cfg, &arena, &b);
        (void)ok;
        size_t n = bolt::hash_join_probe_multi_match(
            &b, lk, rk, kProbe, pb.data(), pp.data(), pb.size());
        DoNotOptimize(n);
    });

    bench_row("composite-key (2 keys) uniq Int64", kProbe, kIters, [&]{
        bolt::Arena arena{acfg};
        bolt::HashJoinBuildChained b{};
        bolt::HashJoinBuildCfgChained cfg{};
        cfg.n_keys = 2;
        cfg.keys[0] = bolt::BoltType::UInt64;
        cfg.keys[1] = bolt::BoltType::UInt64;
        cfg.build_rows = static_cast<uint64_t>(kBuild);
        cfg.expected_dup_factor = 1;
        cfg.max_chain_len = bolt::kHJMaxChainLen;
        const uint64_t* lk[2] = { bk_a.data(), bk_b.data() };
        const uint64_t* rk[2] = { pk_a.data(), pk_b.data() };
        bool ok = bolt::hash_join_build_chained(lk, cfg, &arena, &b);
        (void)ok;
        size_t n = bolt::hash_join_probe_multi_match(
            &b, lk, rk, kProbe, pb.data(), pp.data(), pb.size());
        DoNotOptimize(n);
    });

    std::printf("OK bench_hashjoin_multi_match\n");
    return 0;
}
