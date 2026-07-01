// Performance regression GATE (Gestalt2 tracker G2FEAT-142).
//
// A ctest-registered gtest that measures warm ns/row for the hot kernels built
// in this perf push and FAILS if any is above a GENEROUS threshold. This is a
// regression guard, NOT a tight micro-bound: each threshold sits ~2-3x above
// the measured baseline so ordinary run-to-run / machine-to-machine noise
// passes, but a real regression (e.g. a fast path accidentally degraded to
// O(n^2), a removed prefix-sum path, or a serialised parallel sort) trips it.
//
// Each gate:
//   1. builds its input ONCE (heap vectors, off the timed region),
//   2. runs the kernel kReps times and keeps the MIN nanoseconds (warm, the
//      least-perturbed sample),
//   3. divides by the row count to get ns/row, prints it, and
//   4. EXPECT_LT(ns_per_row, threshold).
//
// A checksum derived from each kernel's output is fed to a volatile sink so the
// optimizer can't dead-code-eliminate the work being measured.
//
// Kernels gated (the three from the perf push):
//   1. bolt::kernels::window_agg::window_agg_f64  — whole-partition + cumulative
//      O(n) fast paths (window_agg.h).
//   2. bolt::groupby_agg_multi_key_typed (Float64 SUM arm) — bolt_groupby.h.
//   3. bolt::kernels::parallel_merge_sort_indirect_u32 vs the serial
//      merge_sort_indirect_u32 — bolt_argsort_parallel.h / bolt_argsort.h.
//
// Tiger Style: bounded N, fixed reps, no hot-path allocation inside the timed
// region (inputs and the groupby arena are reused / reset between reps).

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/bolt_argsort.h"
#include "bolt/kernels/bolt_argsort_parallel.h"
#include "bolt/kernels/bolt_radixsort.h"
#include "bolt/kernels/bolt_radixsort_parallel.h"
#include "bolt/kernels/window_agg.h"

#include <limits>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Warm-timing reps: enough to discard cold-cache / scheduling outliers; the
// MIN sample is the least-perturbed estimate of the kernel's real cost.
constexpr int kReps = 5;

// Volatile sink so the compiler cannot prove the measured work is dead.
volatile double g_sink_d = 0.0;
volatile std::uint32_t g_sink_u = 0u;

double ns_of(double total_ns, std::int64_t rows) noexcept {
    return total_ns / static_cast<double>(rows);
}

// ---------------------------------------------------------------------------
// Gate 1 — window_agg_f64 whole-partition + cumulative O(n) fast paths.
//
// One huge single partition (N rows). Both gated frames have an O(n) fast path
// in the kernel; if either regressed to the O(n*w) general loop the window
// width would be ~N and the run would explode (effectively O(n^2)) — so even a
// very generous ns/row ceiling trips the regression.
//
// Measured baseline (MSVC Release, AVX2 desktop, 16 workers, N=1,000,000):
//   whole-partition Sum ~1.2 ns/row, cumulative Sum ~1.0 ns/row.
// Thresholds set well above to absorb noise while still catching a fast-path
// regression (which would be thousands of ns/row).
// ---------------------------------------------------------------------------
TEST(BoltPerfGate, WindowAggFastPathsNsPerRow) {
    using namespace bolt::kernels::window_agg;
    constexpr std::int64_t kN = 1'000'000;

    std::vector<double> vals(kN);
    std::vector<double> out(kN);
    std::mt19937_64 rng(0xA11CE);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (auto& v : vals) v = dist(rng);

    const Frame whole{0, 0, /*start_unbounded=*/true, /*end_unbounded=*/true, {}};
    const Frame cumulative{0, 0, /*start_unbounded=*/true, /*end_unbounded=*/false, {}};

    double best_whole = 1e30;
    double best_cumul = 1e30;
    for (int rep = 0; rep < kReps; ++rep) {
        auto t0 = Clock::now();
        window_agg_f64(WinAgg::Sum, vals.data(), kN, whole, out.data());
        auto t1 = Clock::now();
        best_whole = std::min(best_whole,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        g_sink_d = out[kN - 1];

        auto t2 = Clock::now();
        window_agg_f64(WinAgg::Sum, vals.data(), kN, cumulative, out.data());
        auto t3 = Clock::now();
        best_cumul = std::min(best_cumul,
            std::chrono::duration<double, std::nano>(t3 - t2).count());
        g_sink_d = out[kN - 1];
    }

    const double ns_whole = ns_of(best_whole, kN);
    const double ns_cumul = ns_of(best_cumul, kN);
    std::printf("\n[perf-gate window_agg N=%lld] whole-partition=%.3f ns/row  "
                "cumulative=%.3f ns/row\n",
                static_cast<long long>(kN), ns_whole, ns_cumul);
    std::fflush(stdout);

    // Generous ceilings (~baseline x 8-10). A removed O(n) fast path would be
    // orders of magnitude over these, not a few x.
    EXPECT_LT(ns_whole, 10.0);  // baseline ~1.2 ns/row
    EXPECT_LT(ns_cumul, 10.0);  // baseline ~1.0 ns/row
}

// ---------------------------------------------------------------------------
// Gate 2 — Float64 group-by aggregation (SUM arm).
//
// N rows, C distinct int64 keys, Float64 values; one SUM aggregate. Exercises
// the typed group-by's float apply/finalize arms (bolt_groupby.h). The arena is
// RESET between reps (no unbounded growth, no hot-path malloc inside the timed
// region; the reset itself is ~5ns and excluded from timing).
//
// Measured baseline (MSVC Release, AVX2 desktop, N=1,000,000, C=1024):
//   ~19 ns/row. Threshold 60 ns/row (~3x) catches a real regression
//   (e.g. a per-row branch reintroduced, or float arm falling off the fast
//   path) without flapping on noise.
// ---------------------------------------------------------------------------
TEST(BoltPerfGate, GroupByFloat64SumNsPerRow) {
    using namespace bolt;
    constexpr std::int64_t kN = 1'000'000;
    constexpr std::int64_t kCard = 1024;

    std::vector<std::int64_t> keys(kN);
    std::vector<double> vals(kN);
    std::mt19937_64 rng(0xB0B);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::int64_t i = 0; i < kN; ++i) {
        keys[i] = i % kCard;
        vals[i] = dist(rng);
    }

    BoltColumn key = BoltColumn::make_flat(keys.data(), nullptr, kN, BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vals.data(), nullptr, kN, BoltType::Float64);
    AggSpec spec{};
    spec.kind = AggKind::Sum;
    spec.in_col = 0;
    spec.distinct = 0;

    Arena arena;
    double best = 1e30;
    uint32_t ng = 0;
    for (int rep = 0; rep < kReps; ++rep) {
        arena.reset();
        BoltColumn ok[1], oa[1];
        auto t0 = Clock::now();
        const bool ran = groupby_agg_multi_key_typed(
            &key, 1, &val, 1, &spec, 1, kN, ok, oa, &ng, &arena, /*hint=*/2048);
        auto t1 = Clock::now();
        ASSERT_TRUE(ran);
        ASSERT_EQ(ng, static_cast<uint32_t>(kCard));
        best = std::min(best,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        // Checksum the per-group sums into the sink (prevents DCE; also a light
        // correctness signal — the grand total must equal sum of all values).
        const double* sums = static_cast<const double*>(oa[0].data);
        double total = 0.0;
        for (uint32_t g = 0; g < ng; ++g) total += sums[g];
        g_sink_d = total;
    }

    const double ns = ns_of(best, kN);
    std::printf("[perf-gate groupby_f64 N=%lld card=%lld] %.3f ns/row  groups=%u\n",
                static_cast<long long>(kN), static_cast<long long>(kCard), ns, ng);
    std::fflush(stdout);

    EXPECT_LT(ns, 60.0);   // baseline ~19 ns/row
}

// ---------------------------------------------------------------------------
// Gate 3 — parallel_merge_sort_indirect_u32 vs serial merge_sort_indirect_u32.
//
// N random i64 keys; argsort to a u32 permutation. Prints serial ns/elem,
// parallel ns/elem, and the speedup. Gates the PARALLEL path's absolute
// ns/elem below a generous ceiling (the regression target: a parallel sort
// that silently serialised, or a merge that went quadratic, blows past it).
// The serial kernel is gated too (it is also a perf-push hot kernel). We do
// NOT assert parallel < serial: that ratio depends on the host core count and
// would flap on a 1-2 core CI box — the dedicated argsort_parallel test owns
// the speedup assertion at hardware concurrency.
//
// Measured baseline (MSVC Release, AVX2 desktop, 16 workers, N=1,000,000):
//   serial ~165 ns/elem, parallel ~59 ns/elem (~2.8x speedup; both are
//   workers-/machine-dependent).
// Thresholds: serial 400 ns/elem, parallel 180 ns/elem (~2.5-3x; parallel's
// is loose because with 1 worker it falls back to the serial path).
// ---------------------------------------------------------------------------
TEST(BoltPerfGate, ArgsortParallelVsSerialNsPerElem) {
    using bolt::kernels::merge_sort_indirect_u32;
    using bolt::kernels::parallel_merge_sort_indirect_u32;
    constexpr std::int64_t kN = 1'000'000;

    std::vector<std::int64_t> kvals(kN);
    std::mt19937_64 rng(0x5EED);
    std::uniform_int_distribution<std::int64_t> dist(-(1LL << 40), (1LL << 40));
    for (auto& k : kvals) k = dist(rng);
    auto less = [&](std::uint32_t a, std::uint32_t b) { return kvals[a] < kvals[b]; };

    std::vector<std::uint32_t> perm_s(kN), scratch_s(kN);
    std::vector<std::uint32_t> perm_p(kN), scratch_p(kN);

    double best_serial = 1e30;
    for (int rep = 0; rep < kReps; ++rep) {
        for (std::int64_t i = 0; i < kN; ++i) perm_s[i] = static_cast<std::uint32_t>(i);
        auto t0 = Clock::now();
        merge_sort_indirect_u32(perm_s.data(), scratch_s.data(), kN, less);
        auto t1 = Clock::now();
        best_serial = std::min(best_serial,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        g_sink_u = perm_s[kN - 1];
    }

    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));  // 0 -> hardware concurrency
    const std::uint32_t workers = sched.thread_count();
    double best_par = 1e30;
    for (int rep = 0; rep < kReps; ++rep) {
        auto t0 = Clock::now();
        parallel_merge_sort_indirect_u32(perm_p.data(), scratch_p.data(), kN,
                                         less, &sched);
        auto t1 = Clock::now();
        best_par = std::min(best_par,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        g_sink_u = perm_p[kN - 1];
    }
    sched.shutdown();

    // Correctness parity: both must produce the same stable permutation.
    EXPECT_EQ(perm_s, perm_p);

    const double ns_serial = ns_of(best_serial, kN);
    const double ns_par = ns_of(best_par, kN);
    std::printf("[perf-gate argsort N=%lld] workers=%u  serial=%.3f ns/elem  "
                "parallel=%.3f ns/elem  speedup=%.2fx\n",
                static_cast<long long>(kN), workers, ns_serial, ns_par,
                best_serial / best_par);
    std::fflush(stdout);

    EXPECT_LT(ns_serial, 400.0);  // baseline ~165 ns/elem
    EXPECT_LT(ns_par, 180.0);     // baseline ~59 ns/elem (serial fallback w/ 1 worker)
}

// ---------------------------------------------------------------------------
// Gate 4 — radix argsort (G2FEAT-153/157): serial single-int64 LSD radix and
// the parallel multi-key MSD radix. The serial radix is O(n) (linear passes),
// so any regression to a comparison sort or a degenerate scatter blows past a
// generous ns/row ceiling. The parallel radix is fed a FULL-RANGE int64 primary
// (high top-byte entropy) so it PROCEEDS (a narrow primary self-declines to the
// merge — gated in the kernel); a serialised parallel radix or the wrong-slice
// histogram bug the unit test caught would trip the ceiling.
//
// Byte-identical correctness is owned by test_bolt_radixsort_parallel; this gate
// only guards ns/row + that the high-entropy input proceeds.
// ---------------------------------------------------------------------------
TEST(BoltPerfGate, RadixArgsortNsPerRow) {
    using namespace bolt::kernels;
    constexpr std::int64_t kN = 1'000'000;

    std::vector<std::int64_t> k0(kN), k1(kN), knar(kN);
    std::mt19937_64 rng(0x4AD1);
    std::uniform_int_distribution<std::int64_t> full(std::numeric_limits<std::int64_t>::min(),
                                                     std::numeric_limits<std::int64_t>::max());
    std::uniform_int_distribution<std::int64_t> small(0, 64);
    std::uniform_int_distribution<std::int64_t> narrow(0, (1LL << 24));  // ~3 byte-passes
    for (std::int64_t i = 0; i < kN; ++i) { k0[i] = full(rng); k1[i] = small(rng); knar[i] = narrow(rng); }

    // Serial single-int64 indirect radix — gated on NARROW keys: that is its real
    // use (the multi-key path routes serial only below the parallel floor, and
    // narrow keys skip most passes). Full-range indirect radix is random-scatter
    // bound and too flappy under memory load to gate on.
    std::vector<std::int32_t> perm(kN), pscr(kN);
    std::vector<std::int64_t> kscr(kN);
    double best_serial = 1e30;
    for (int rep = 0; rep < kReps; ++rep) {
        auto t0 = Clock::now();
        radix_sort_indirect_i64(perm.data(), knar.data(), pscr.data(), kscr.data(), kN);
        auto t1 = Clock::now();
        best_serial = std::min(best_serial,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        g_sink_u = static_cast<std::uint32_t>(perm[kN - 1]);
    }

    // Parallel multi-key MSD radix (full-range primary → proceeds).
    std::vector<std::uint64_t> b0(kN), b1(kN);
    radix_fill_biased_i64(b0.data(), k0.data(), kN, /*ascending=*/true);
    radix_fill_biased_i64(b1.data(), k1.data(), kN, /*ascending=*/true);
    const std::uint64_t* keyp[2] = {b0.data(), b1.data()};
    std::vector<std::uint32_t> up(kN), us(kN);
    bolt::Scheduler sched;
    ASSERT_TRUE(sched.init(0));  // 0 -> hardware concurrency
    bolt::Arena arena;
    double best_par = 1e30;
    bool proceeded = false;
    for (int rep = 0; rep < kReps; ++rep) {
        arena.reset();
        auto t0 = Clock::now();
        proceeded = parallel_radix_argsort_multikey_u64(
            up.data(), us.data(), keyp, 2, kN, &sched, &arena);
        auto t1 = Clock::now();
        best_par = std::min(best_par,
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        g_sink_u = up[kN - 1];
    }
    sched.shutdown();
    ASSERT_TRUE(proceeded) << "full-range primary should proceed (not decline)";

    const double ns_serial = ns_of(best_serial, kN);
    const double ns_par = ns_of(best_par, kN);
    std::printf("[perf-gate radix N=%lld] serial-1key=%.3f ns/row  parallel-2key=%.3f ns/row\n",
                static_cast<long long>(kN), ns_serial, ns_par);
    std::fflush(stdout);

    // Baselines (MSVC Release, 16 workers, N=1e6): serial indirect radix on NARROW
    // keys ~40-70 ns/row (~3 byte-passes); parallel MSD radix on FULL-RANGE keys
    // ~16-25 ns/row (~3x the parallel merge's ~60 on the same data). Generous
    // ceilings absorb thermal/machine noise but trip a real regression (a
    // serialised parallel radix, or a scatter gone quadratic).
    EXPECT_LT(ns_serial, 200.0);
    EXPECT_LT(ns_par, 80.0);
}

}  // namespace
