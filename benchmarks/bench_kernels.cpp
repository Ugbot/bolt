// bench_kernels.cpp — direct microbenchmarks for SIMD kernels.
//
// Isolates a single kernel per test (no query glue, no arena setup) and
// reports best-of-N timings so run-to-run variance on noisy hosts doesn't
// drown out real speedups. Compare against the scalar template to see the
// uplift attributable to the `bmm_*` path.

#include "bolt/bolt.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_numeric.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/join/bolt_groupby.h"

#if __has_include("bolt/kernels/bolt_parallel.h")
  #include "bolt/kernels/bolt_parallel.h"
  #define BOLT_HAS_PARALLEL 1
#else
  #define BOLT_HAS_PARALLEL 0
#endif

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

// Best-of-N timing with warmup. Returns min ns/op.
template <typename Fn>
static double bench(const char* name, int64_t ops, int iters, Fn&& fn) noexcept {
    fn();  // warmup
    int64_t best = INT64_MAX;
    for (int it = 0; it < iters; ++it) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        int64_t dt = std::chrono::duration_cast<ns>(t1 - t0).count();
        if (dt < best) best = dt;
    }
    double per = static_cast<double>(best) / static_cast<double>(ops);
    printf("  %-40s %8.2f ns/op  (best %8lld ns, %.1f M ops/s)\n",
           name, per, static_cast<long long>(best),
           static_cast<double>(ops) / (best / 1e9) / 1e6);
    return per;
}

// ------------------------------------------------------------
// Kernel microbenches
// ------------------------------------------------------------

static void bench_filter_i32(int64_t N, int iters, bolt::Scheduler* sched, bolt::Arena* arena) {
    std::vector<int32_t> data(N);
    std::vector<int32_t> out(N);
    std::mt19937 rng(42);
    for (auto& v : data) v = static_cast<int32_t>(rng() & 0x3FFFFFFF);
    const int32_t scalar = 0x1FFFFFFF;  // ~50% selectivity

    printf("filter_gt<int32_t>  (N=%lld, ~50%% selectivity):\n",
           static_cast<long long>(N));

    bench("  branchless scalar kernel", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_branchless<int32_t>(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    bench("  bmm_* SIMD kernel (compressstore)", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_avx2_i32(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });
#endif

#if BOLT_HAS_PARALLEL
    if (sched && sched->thread_count() > 1) {
        bench("  parallel (4 workers, balanced)", N, iters, [&] {
            int64_t c = bolt::kernels::parallel_filter_gt<int32_t>(
                sched, data.data(), N, scalar, out.data(), arena);
            DoNotOptimize(c);
        });
    } else {
        printf("  parallel row skipped (no scheduler / single core)\n");
    }
#else
    (void)sched; (void)arena;
    printf("  parallel row skipped (bolt_parallel.h not present)\n");
#endif
}

static void bench_filter_i64(int64_t N, int iters) {
    std::vector<int64_t> data(N);
    std::vector<int32_t> out(N);
    std::mt19937_64 rng(42);
    for (auto& v : data) v = static_cast<int64_t>(rng() & 0x3FFFFFFFFFFFFFFFLL);
    const int64_t scalar = 0x1FFFFFFFFFFFFFFFLL;  // ~50% selectivity

    printf("filter_gt<int64_t>  (N=%lld, ~50%% selectivity):\n",
           static_cast<long long>(N));

    bench("  branchless scalar kernel", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_branchless<int64_t>(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    bench("  bmm_* SIMD kernel (compressstore)", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_avx2_i64(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });
#endif
}

static void bench_filter_f64(int64_t N, int iters) {
    std::vector<double> data(N);
    std::vector<int32_t> out(N);
    std::mt19937_64 rng(42);
    for (auto& v : data) v = static_cast<double>(static_cast<int64_t>(rng())) / 1e9;
    const double scalar = 0.0;  // ~50% selectivity

    printf("filter_gt<double>  (N=%lld, ~50%% selectivity):\n",
           static_cast<long long>(N));

    bench("  branchless scalar kernel", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_branchless<double>(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    bench("  bmm_* SIMD kernel (compressstore)", N, iters, [&] {
        int64_t c = bolt::branchless::filter_gt_avx2_f64(
            data.data(), N, scalar, out.data());
        DoNotOptimize(c);
    });
#endif
}

static void bench_sum_i64(int64_t N, int iters, bolt::Scheduler* sched) {
    std::vector<int64_t> data(N);
    std::mt19937_64 rng(42);
    for (auto& v : data) v = static_cast<int64_t>(rng() & 0xFFFFFFFFULL);

    printf("sum<int64_t>  (N=%lld):\n", static_cast<long long>(N));

    // Volatile sink so the compiler can't elide repeated identical calls.
    volatile int64_t sink = 0;
    bench("  kernels::sum<int64_t> (scalar)", N, iters, [&] {
        sink = bolt::kernels::sum<int64_t>(data.data(), N);
    });

#if BOLT_SIMD_AVX2 || BOLT_SIMD_SSE42 || BOLT_SIMD_NEON
    bench("  bmm_* SIMD reduce (bmm_add_i64+hadd)", N, iters, [&] {
        sink = bolt::branchless::sum_avx2_i64(data.data(), N);
    });
#endif

#if BOLT_HAS_PARALLEL
    if (sched && sched->thread_count() > 1) {
        bench("  parallel (4 workers, balanced)", N, iters, [&] {
            sink = bolt::kernels::parallel_sum<int64_t>(sched, data.data(), N);
        });
    } else {
        printf("  parallel row skipped (no scheduler / single core)\n");
    }
#else
    (void)sched;
    printf("  parallel row skipped (bolt_parallel.h not present)\n");
#endif
    (void)sink;
}

// Match pyarrow's group_by(...).aggregate("sum") shape: 1M rows, 10 distinct
// int64 keys, int64 values. Times the parallel groupby (with serial fallback
// when sched is null) so the comparison is apples-to-apples with pyarrow,
// which is multi-threaded internally.
static void bench_groupby_i64(int64_t N, int iters, bolt::Scheduler* sched, bolt::Arena* arena) {
    std::vector<int64_t> keys(N);
    std::vector<int64_t> vals(N);
    std::mt19937_64 rng(42);
    for (int64_t i = 0; i < N; ++i) {
        keys[i] = static_cast<int64_t>(rng() % 10);
        vals[i] = static_cast<int64_t>(rng() % 10000);
    }

    bolt::BoltColumn key_col = bolt::BoltColumn::make_flat(
        keys.data(), nullptr, N, bolt::BoltType::Int64);
    bolt::BoltColumn val_col = bolt::BoltColumn::make_flat(
        vals.data(), nullptr, N, bolt::BoltType::Int64);

    printf("groupby_sum_int64  (N=%lld, 10 groups):\n", static_cast<long long>(N));

    bench("  serial groupby_sum_int64", N, iters, [&] {
        bolt::BoltColumn out_keys, out_sums;
        arena->reset();
        bool ok = bolt::groupby_sum_int64(key_col, val_col, arena, &out_keys, &out_sums);
        DoNotOptimize(ok);
        DoNotOptimize(out_keys.length);
        DoNotOptimize(out_sums.length);
    });

#if BOLT_HAS_PARALLEL
    if (sched && sched->thread_count() > 1) {
        bench("  parallel groupby (radix merge)", N, iters, [&] {
            bolt::BoltColumn out_keys, out_sums;
            arena->reset();
            bool ok = bolt::parallel_groupby_sum_int64(
                sched, key_col, val_col, arena, &out_keys, &out_sums);
            DoNotOptimize(ok);
            DoNotOptimize(out_keys.length);
            DoNotOptimize(out_sums.length);
        });
    }
#else
    (void)sched;
#endif
}

static void bench_gather_i32(int64_t N_data, int64_t N_probes, int iters) {
    std::vector<int32_t> data(N_data);
    std::mt19937 rng(42);
    for (auto& v : data) v = static_cast<int32_t>(rng());

    // Random indices — the cache-unfriendly worst case that motivates the
    // gather + prefetch machinery.
    std::vector<int32_t> indices(N_probes);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(N_data - 1));
    for (auto& v : indices) v = dist(rng);

    std::vector<int32_t> out(N_probes);
    printf("gather<int32_t>  (data=%lld, probes=%lld, random indices):\n",
           static_cast<long long>(N_data), static_cast<long long>(N_probes));
    bench("  gather_branchless (SIMD + prefetch)", N_probes, iters, [&] {
        bolt::branchless::gather_branchless<int32_t>(
            data.data(), indices.data(), N_probes, out.data());
        DoNotOptimize(out[0]);
    });
}

static void bench_gather_i64(int64_t N_data, int64_t N_probes, int iters) {
    std::vector<int64_t> data(N_data);
    std::mt19937_64 rng(42);
    for (auto& v : data) v = static_cast<int64_t>(rng());

    std::vector<int32_t> indices(N_probes);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(N_data - 1));
    std::mt19937 irng(7);
    for (auto& v : indices) v = dist(irng);

    std::vector<int64_t> out(N_probes);
    printf("gather<int64_t>  (data=%lld, probes=%lld, random indices):\n",
           static_cast<long long>(N_data), static_cast<long long>(N_probes));
    bench("  gather_branchless (scalar + prefetch) [default]", N_probes, iters, [&] {
        bolt::branchless::gather_branchless<int64_t>(
            data.data(), indices.data(), N_probes, out.data());
        DoNotOptimize(out[0]);
    });
#if BOLT_SIMD_AVX2 || BOLT_SIMD_AVX512
    // Kept-in-source alternative: hardware gather. Slower on client Intel;
    // reachable by name for future JIT dispatch or other hardware.
    bench("  gather_simd_i64 (alternative, slower on client Intel)", N_probes, iters, [&] {
        bolt::branchless::gather_simd_i64(
            data.data(), indices.data(), N_probes, out.data());
        DoNotOptimize(out[0]);
    });
#endif
}

static void bench_swiss_find(int64_t N_keys, int iters) {
    // Populate a moderately-sized table so each find does real probing.
    const uint32_t kTableSize = 16384;
    bolt::Arena arena;
    bolt::ArenaGuard g(arena);
    bolt::SwissTable st;
    bolt::SwissTable::create(&st, kTableSize, &arena);

    std::mt19937_64 rng(42);
    for (uint32_t i = 0; i < kTableSize; ++i) {
        st.insert(rng(), i);
    }

    // Probe: half hits (pick from inserted range), half misses.
    std::vector<uint64_t> probes(N_keys);
    std::mt19937_64 rng2(42);
    std::vector<uint64_t> inserted;
    inserted.reserve(kTableSize);
    for (uint32_t i = 0; i < kTableSize; ++i) inserted.push_back(rng2());
    std::mt19937_64 rng3(1337);
    for (int64_t i = 0; i < N_keys; ++i) {
        if (i & 1) probes[i] = inserted[rng3() % kTableSize];
        else       probes[i] = rng3() | (1ULL << 63);  // likely miss
    }
    std::vector<int32_t> out(N_keys);

    printf("SwissTable::find_simd  (table=%u, probes=%lld, ~50%% hit):\n",
           kTableSize, static_cast<long long>(N_keys));

    bench("  group-scanning find (bmm_cmpeq_i8)", N_keys, iters, [&] {
        st.find_simd(probes.data(), N_keys, out.data());
        DoNotOptimize(out[0]);
    });
}

int main() {
    printf("============================================================\n");
    printf("  Bolt kernel microbenchmarks (best-of-N)\n");
    printf("  SIMD tier compile-detected via bolt_port.h\n");
    printf("============================================================\n");
#if BOLT_SIMD_AVX2
    printf("  Active ISA: AVX2 (lanes_i32=8)\n");
#elif BOLT_SIMD_SSE42
    printf("  Active ISA: SSE4.2 (lanes_i32=4)\n");
#elif BOLT_SIMD_NEON
    printf("  Active ISA: NEON (lanes_i32=4)\n");
#else
    printf("  Active ISA: scalar fallback\n");
#endif
    // Build a small scheduler for the parallel rows. 4 workers, balanced.
    // If the host only has 1 logical CPU, skip construction — the bench
    // helpers will print a "skipped" line instead.
    bolt::Scheduler  parallel_sched;
    bolt::Scheduler* parallel_sched_ptr = nullptr;
    bolt::Arena      parallel_arena;
    bool sched_live = false;
    const uint32_t hw = ::bolt_get_hardware_concurrency();
    if (hw > 1) {
        bolt::SchedulerConfig scfg = bolt::SchedulerConfig::with_profile(
            bolt::TuneProfile::Balanced);
        uint32_t want = 4u;
        if (want > hw) want = hw;
        scfg.num_workers = want;
        if (parallel_sched.init(scfg)) {
            sched_live = true;
            parallel_sched_ptr = &parallel_sched;
            printf("  Parallel scheduler: %u workers (balanced profile).\n",
                   parallel_sched.thread_count());
        } else {
            printf("  Parallel scheduler: init failed, parallel rows skipped.\n");
        }
    } else {
        printf("  Parallel scheduler: host has <=1 logical CPU, skipped.\n");
    }
    printf("\n");

    constexpr int kIters = 20;
    bench_filter_i32(1'000'000, kIters, parallel_sched_ptr, &parallel_arena);
    printf("\n");
    bench_filter_i64(1'000'000, kIters);
    printf("\n");
    bench_filter_f64(1'000'000, kIters);
    printf("\n");
    bench_sum_i64   (1'000'000, kIters, parallel_sched_ptr);
    printf("\n");
    bench_groupby_i64(1'000'000, kIters, parallel_sched_ptr, &parallel_arena);
    printf("\n");
    bench_swiss_find(  500'000, kIters);
    printf("\n");
    bench_gather_i32(1'000'000, 1'000'000, kIters);
    printf("\n");
    bench_gather_i64(1'000'000, 1'000'000, kIters);
    printf("\n");

    if (sched_live) parallel_sched.shutdown();
    return 0;
}
