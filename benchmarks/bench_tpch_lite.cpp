// bench_tpch_lite.cpp — TPC-H-shaped micro-queries (Phase 6 scope).
//
// Three synthetic queries loosely modeled on TPC-H Q1/Q3/Q6. Runs on a single
// thread, single core. Data is generated deterministically and lives in
// arena-backed buffers sized at startup — the timed regions perform no heap
// allocation. Multi-worker scheduler integration is out of scope here.
//
// Build target: bench_tpch_lite (see benchmarks/CMakeLists.txt).

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "bolt/bolt.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/kernels/bolt_numeric.h"
#include "bolt/join/bolt_hashjoin.h"
#include "bolt/join/bolt_groupby.h"

#if __has_include("bolt/kernels/bolt_parallel.h")
  #include "bolt/kernels/bolt_parallel.h"
  #define BOLT_HAS_PARALLEL 1
#else
  #define BOLT_HAS_PARALLEL 0
#endif

using Clock = std::chrono::high_resolution_clock;
using ns    = std::chrono::nanoseconds;

// ---------------------------------------------------------------------------
// DoNotOptimize / ClobberMemory — MSVC-friendly variants (mirror bench_bolt).
// ---------------------------------------------------------------------------
template <typename T>
inline void DoNotOptimize(T&& val) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    (void)val;
    _ReadWriteBarrier();
#else
    asm volatile("" : : "g"(val) : "memory");
#endif
}

inline void ClobberMemory() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    _ReadWriteBarrier();
#else
    asm volatile("" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// Median / min over a small fixed-capacity sample buffer. ≤16 iterations.
// ---------------------------------------------------------------------------
static constexpr int kMaxIters = 16;

static int64_t median_ns(const int64_t* samples, int n) noexcept {
    assert(samples != nullptr);
    assert(n > 0 && n <= kMaxIters);
    int64_t buf[kMaxIters];
    for (int i = 0; i < n; ++i) buf[i] = samples[i];
    std::sort(buf, buf + n);
    return buf[n / 2];
}

static int64_t min_ns(const int64_t* samples, int n) noexcept {
    assert(samples != nullptr);
    assert(n > 0 && n <= kMaxIters);
    int64_t m = samples[0];
    for (int i = 1; i < n; ++i) if (samples[i] < m) m = samples[i];
    return m;
}

// ---------------------------------------------------------------------------
// Result buckets. One row per query for the final print-out.
// ---------------------------------------------------------------------------
struct QueryResult {
    const char* label;
    int64_t     rows_timed;    // denominator for ns/row
    int64_t     median_total;
    int64_t     min_total;
};

static void print_row(const QueryResult& r) noexcept {
    assert(r.label != nullptr);
    assert(r.rows_timed > 0);
    double per_row   = static_cast<double>(r.median_total) / static_cast<double>(r.rows_timed);
    double rows_per_s = static_cast<double>(r.rows_timed) /
                        (static_cast<double>(r.median_total) / 1e9);
    printf("  %-28s %7.2f ns/row  (%10lld ns total, %6.1f M rows/s)  [min %lld ns]\n",
           r.label, per_row,
           static_cast<long long>(r.median_total),
           rows_per_s / 1e6,
           static_cast<long long>(r.min_total));
}

// ---------------------------------------------------------------------------
// Query scales — Tiger Style constants, fixed at compile time.
// ---------------------------------------------------------------------------
// Defaults — overridden via `--rows N` (multiplies Q1/Q3-orders/Q6 by N/1M).
// Q3's customer build side is intentionally fixed: a real Q3 has a small
// customer table no matter how many orders we throw at it.
static int64_t kQ1Rows      = 1'000'000;   // lineitem rows for Q1
static int64_t kQ3Customers = 1'000;       // customer rows (build, fixed)
static int64_t kQ3Orders    = 1'000'000;   // order rows (probe)
static int64_t kQ6Rows      = 1'000'000;   // lineitem rows for Q6
static constexpr int     kIters       = 5;

// ===========================================================================
// Q1 — group-by aggregate over (l_returnflag, l_quantity, l_extendedprice).
// ===========================================================================
// We compute sum(l_extendedprice) grouped by l_returnflag via
// bolt::groupby_sum_int64, then sum(l_quantity) and count via a small manual
// pass (three distinct groups so the direct loop is trivial and keeps the
// focus on the hash aggregate that the library provides).
static QueryResult run_q1(bolt::Arena* arena,
                          bolt::BoltColumn* return_flag_col,
                          bolt::BoltColumn* quantity_col,
                          bolt::BoltColumn* price_col,
                          bolt::Scheduler* sched) noexcept {
    assert(arena != nullptr);
    assert(return_flag_col != nullptr);
    assert(price_col != nullptr);
    assert(return_flag_col->length == kQ1Rows);

    int64_t samples[kMaxIters];
    int n = 0;

    // Warm-up.
    {
        bolt::ArenaGuard g(*arena);
        bolt::BoltColumn out_keys{}, out_sums{};
#if BOLT_HAS_PARALLEL
        (void)bolt::parallel_groupby_sum_int64(
            sched, *return_flag_col, *price_col, arena, &out_keys, &out_sums);
#else
        (void)sched;
        (void)bolt::groupby_sum_int64(*return_flag_col, *price_col, arena,
                                      &out_keys, &out_sums);
#endif
        DoNotOptimize(out_sums.data);
    }

    for (int it = 0; it < kIters; ++it) {
        bolt::ArenaGuard g(*arena);
        bolt::BoltColumn out_keys{}, out_sums{};

        auto t0 = Clock::now();
#if BOLT_HAS_PARALLEL
        bool ok = bolt::parallel_groupby_sum_int64(
            sched, *return_flag_col, *price_col, arena, &out_keys, &out_sums);
#else
        bool ok = bolt::groupby_sum_int64(*return_flag_col, *price_col, arena,
                                          &out_keys, &out_sums);
#endif
        // Also aggregate sum(l_quantity) via the streaming sum kernel; this
        // does NOT group, it just exercises the sum path alongside the
        // group-by so the timed region reflects the full row scan.
        const int64_t* qty = static_cast<const int64_t*>(quantity_col->data);
#if BOLT_HAS_PARALLEL
        int64_t qty_sum = bolt::kernels::parallel_sum<int64_t>(sched, qty, kQ1Rows);
#else
        int64_t qty_sum = bolt::kernels::sum<int64_t>(qty, kQ1Rows);
#endif
        auto t1 = Clock::now();

        assert(ok);
        (void)ok;
        DoNotOptimize(out_sums.data);
        DoNotOptimize(qty_sum);
        ClobberMemory();

        samples[n++] = std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    QueryResult r{};
    r.label        = "Q1 (group-by agg):";
    r.rows_timed   = kQ1Rows;
    r.median_total = median_ns(samples, n);
    r.min_total    = min_ns(samples, n);
    return r;
}

// ===========================================================================
// Q3 — filter orders by o_totalprice > 50k, hash-join on custkey, sum price.
// ===========================================================================
static QueryResult run_q3(bolt::Arena* arena,
                          bolt::BoltColumn* cust_keys,
                          bolt::BoltColumn* order_custkey,
                          bolt::BoltColumn* order_totalprice,
                          bolt::Scheduler* sched) noexcept {
    assert(arena != nullptr);
    assert(cust_keys != nullptr);
    assert(order_custkey != nullptr);
    assert(order_totalprice != nullptr);
    assert(order_custkey->length == kQ3Orders);

    int64_t samples[kMaxIters];
    int n = 0;

    // Pre-size selection / pair output buffers from the arena *before* timing.
    // They are owned by the outer-scope arena (not the per-iter guard) so they
    // survive every iteration without reallocation.
    int32_t* filt_sel    = arena->allocate_array<int32_t>(kQ3Orders);
    int32_t* build_idx   = arena->allocate_array<int32_t>(kQ3Orders);
    int32_t* probe_idx   = arena->allocate_array<int32_t>(kQ3Orders);
    assert(filt_sel    != nullptr);
    assert(build_idx   != nullptr);
    assert(probe_idx   != nullptr);

    const int64_t* tp_data = static_cast<const int64_t*>(order_totalprice->data);
    const int64_t* ck_data = static_cast<const int64_t*>(order_custkey->data);

    // Warm-up.
    {
        bolt::ArenaGuard g(*arena);
        bolt::HashJoinBuild hb{};
        bool ok = hb.build(*cust_keys, arena);
        (void)ok;
        assert(ok);
#if BOLT_HAS_PARALLEL
        int64_t m = bolt::kernels::parallel_filter_gt<int64_t>(
            sched, tp_data, kQ3Orders, int64_t{50'000}, filt_sel, arena);
#else
        (void)sched;
        int64_t m = bolt::kernels::filter_gt<int64_t>(tp_data, kQ3Orders,
                                                      int64_t{50'000}, filt_sel);
#endif
        DoNotOptimize(m);
    }

    for (int it = 0; it < kIters; ++it) {
        bolt::ArenaGuard g(*arena);

        auto t0 = Clock::now();

        // (1) Filter: o_totalprice > 50_000.
#if BOLT_HAS_PARALLEL
        int64_t m = bolt::kernels::parallel_filter_gt<int64_t>(
            sched, tp_data, kQ3Orders, int64_t{50'000}, filt_sel, arena);
#else
        int64_t m = bolt::kernels::filter_gt<int64_t>(tp_data, kQ3Orders,
                                                      int64_t{50'000}, filt_sel);
#endif

        // (2) Gather the filtered custkeys into a compacted Flat BoltColumn
        //     via bolt::gather_to_column — the library primitive replaces
        //     the hand-rolled materialization loop that used to live here.
        bolt::BoltColumn probe_col =
            bolt::gather_to_column<int64_t>(*order_custkey, filt_sel, m, arena);

        // Totalprice still gathers to a raw buffer (needed as plain int64_t*
        // for the final acc loop, not as a BoltColumn).
        int64_t* filt_tp = arena->allocate_array<int64_t>(m > 0 ? m : 1);
        assert(filt_tp != nullptr);
        for (int64_t i = 0; i < m; ++i) {
            filt_tp[i] = tp_data[filt_sel[i]];
        }

        // (3) Build + probe. Build must live inside the timed region because
        //     a real query replan rebuilds it each invocation.
        bolt::HashJoinBuild hb{};
        bool built = hb.build(*cust_keys, arena);
        assert(built);
        (void)built;

        int64_t pairs = bolt::HashJoinProbe::probe(hb, probe_col,
                                                   build_idx, probe_idx);

        // (4) Sum o_totalprice over the joined pairs. `probe_idx[i]` indexes
        //     into the filtered column, so this is exactly sum_masked's shape:
        //     sum data[sel[i]] for i in [0, pairs). One fused gather+add pass.
        int64_t acc = bolt::kernels::sum_masked<int64_t>(filt_tp, probe_idx, pairs);

        auto t1 = Clock::now();

        DoNotOptimize(acc);
        DoNotOptimize(pairs);
        DoNotOptimize(m);
        ClobberMemory();

        samples[n++] = std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    QueryResult r{};
    r.label        = "Q3 (filter+join+agg):";
    r.rows_timed   = kQ3Orders;
    r.median_total = median_ns(samples, n);
    r.min_total    = min_ns(samples, n);
    return r;
}

// ===========================================================================
// Q6 — scan + filter (shipdate range + discount range) + sum(price*discount).
// ===========================================================================
static QueryResult run_q6(bolt::Arena* arena,
                          bolt::BoltColumn* price_col,
                          bolt::BoltColumn* discount_col,
                          bolt::BoltColumn* shipdate_col,
                          bolt::Scheduler* sched) noexcept {
    assert(arena != nullptr);
    assert(price_col != nullptr);
    assert(discount_col != nullptr);
    assert(shipdate_col != nullptr);
    assert(price_col->length == kQ6Rows);

    int64_t samples[kMaxIters];
    int n = 0;

    int32_t* sel_date = arena->allocate_array<int32_t>(kQ6Rows);
    int32_t* sel_hi   = arena->allocate_array<int32_t>(kQ6Rows);
    int32_t* sel_both = arena->allocate_array<int32_t>(kQ6Rows);
    int64_t* gather_p = arena->allocate_array<int64_t>(kQ6Rows);
    int64_t* gather_d = arena->allocate_array<int64_t>(kQ6Rows);
    int64_t* product  = arena->allocate_array<int64_t>(kQ6Rows);
    assert(sel_date != nullptr);
    assert(sel_hi   != nullptr);
    assert(sel_both != nullptr);
    assert(gather_p != nullptr);
    assert(gather_d != nullptr);
    assert(product  != nullptr);

    const int64_t* price    = static_cast<const int64_t*>(price_col->data);
    const int64_t* discount = static_cast<const int64_t*>(discount_col->data);
    const int64_t* shipdate = static_cast<const int64_t*>(shipdate_col->data);

    // Date window: [2000, 3000] in "days since epoch" units (arbitrary synthetic).
    const int64_t kDateLo = 2'000;
    const int64_t kDateHi = 3'000;
    // Discount percent window: [5, 7] inclusive.
    const int64_t kDiscLo = 5;
    const int64_t kDiscHi = 7;

    // Warm-up.
    {
#if BOLT_HAS_PARALLEL
        int64_t a = bolt::kernels::parallel_filter_gt<int64_t>(
            sched, shipdate, kQ6Rows, kDateLo, sel_date, arena);
#else
        (void)sched;
        int64_t a = bolt::kernels::filter_gt<int64_t>(shipdate, kQ6Rows, kDateLo, sel_date);
#endif
        DoNotOptimize(a);
    }

    for (int it = 0; it < kIters; ++it) {
        auto t0 = Clock::now();

        // (1) shipdate > kDateLo  AND shipdate <= kDateHi
#if BOLT_HAS_PARALLEL
        int64_t n_date = bolt::kernels::parallel_filter_gt<int64_t>(
            sched, shipdate, kQ6Rows, kDateLo, sel_date, arena);
#else
        int64_t n_date = bolt::kernels::filter_gt<int64_t>(shipdate, kQ6Rows,
                                                           kDateLo, sel_date);
#endif
        // Intersect with shipdate <= kDateHi. Cheapest: walk sel_date and
        // keep rows whose shipdate <= kDateHi. Branchless predicate.
        int64_t n_both = 0;
        for (int64_t i = 0; i < n_date; ++i) {
            int32_t row = sel_date[i];
            sel_hi[n_both] = row;
            n_both += (shipdate[row] <= kDateHi) ? 1 : 0;
        }

        // (2) discount in [kDiscLo, kDiscHi]. Filter the sel_hi survivors.
        int64_t n_final = 0;
        for (int64_t i = 0; i < n_both; ++i) {
            int32_t row = sel_hi[i];
            sel_both[n_final] = row;
            int64_t d = discount[row];
            n_final += (d >= kDiscLo && d <= kDiscHi) ? 1 : 0;
        }

        // (3) Fused gather + mul + sum: SUM(price * discount) over survivors.
        //     The previous version materialized gather_p, gather_d, product
        //     into the arena and made three passes over them. The fused form
        //     stays in registers + L1: one pass, one accumulator, no writes
        //     to intermediate buffers (which still exist for warm-up but are
        //     unused in the timed loop). Branchless predicate-free body.
        int64_t revenue_loss = 0;
        for (int64_t i = 0; i < n_final; ++i) {
            const int32_t row = sel_both[i];
            revenue_loss += price[row] * discount[row];
        }
        (void)gather_p; (void)gather_d; (void)product;  // arena-owned, kept for warm-up

        auto t1 = Clock::now();

        DoNotOptimize(revenue_loss);
        DoNotOptimize(n_final);
        ClobberMemory();

        samples[n++] = std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    QueryResult r{};
    r.label        = "Q6 (scan+filter+sum):";
    r.rows_timed   = kQ6Rows;
    r.median_total = median_ns(samples, n);
    r.min_total    = min_ns(samples, n);
    return r;
}

// ===========================================================================
// Data generation — deterministic, outside timed regions.
// ===========================================================================
static void gen_q1(int32_t* rflag, int64_t* qty, int64_t* price,
                   int64_t n, uint64_t seed) noexcept {
    assert(rflag != nullptr);
    assert(qty != nullptr);
    assert(price != nullptr);
    assert(n > 0);
    std::mt19937_64 rng(seed);
    for (int64_t i = 0; i < n; ++i) {
        rflag[i] = static_cast<int32_t>(i % 3);
        qty[i]   = static_cast<int64_t>(1 + (rng() % 50));
        price[i] = static_cast<int64_t>(1'000 + (rng() % 100'000));
    }
}

static void gen_q3(int64_t* custkey_build, int64_t cb,
                   int64_t* custkey_probe, int64_t* totalprice, int64_t cp,
                   uint64_t seed) noexcept {
    assert(custkey_build != nullptr);
    assert(custkey_probe != nullptr);
    assert(totalprice != nullptr);
    assert(cb > 0 && cp > 0);
    for (int64_t i = 0; i < cb; ++i) custkey_build[i] = i;

    std::mt19937_64 rng(seed);
    for (int64_t i = 0; i < cp; ++i) {
        custkey_probe[i] = static_cast<int64_t>(rng() % static_cast<uint64_t>(cb));
        totalprice[i]    = static_cast<int64_t>(1'000 + (rng() % 200'000));
    }
}

static void gen_q6(int64_t* price, int64_t* discount, int64_t* shipdate,
                   int64_t n, uint64_t seed) noexcept {
    assert(price != nullptr);
    assert(discount != nullptr);
    assert(shipdate != nullptr);
    assert(n > 0);
    std::mt19937_64 rng(seed);
    for (int64_t i = 0; i < n; ++i) {
        price[i]    = static_cast<int64_t>(1'000 + (rng() % 100'000));
        discount[i] = static_cast<int64_t>(rng() % 11);    // 0..10 percent
        shipdate[i] = static_cast<int64_t>(rng() % 5'000); // 0..4999 days
    }
}

// ===========================================================================
// Host metadata — best-effort; print env hint, skip silently on failure.
// ===========================================================================
static void print_host_banner() noexcept {
#if defined(_MSC_VER)
    const char* compiler = "MSVC";
#elif defined(__clang__)
    const char* compiler = "clang";
#elif defined(__GNUC__)
    const char* compiler = "gcc";
#else
    const char* compiler = "unknown";
#endif
    printf("Compiler: %s, Release build.\n", compiler);
}

static void print_usage(const char* exe) noexcept {
    printf("Usage: %s [--threads N] [--profile NAME] [--rows N] [--help]\n",
           exe ? exe : "bench_tpch_lite");
    printf("  --threads N        Number of worker threads (default 1 = serial baseline).\n");
    printf("  --profile NAME     Scheduler tuning profile: latency|balanced|throughput.\n");
    printf("  --rows N           Row count for Q1/Q3-orders/Q6 (default 1000000).\n");
    printf("                     Q3 customer build side stays fixed at 1000.\n");
    printf("  --help             Print this message.\n");
}

static const char* profile_name(bolt::TuneProfile p) noexcept {
    switch (p) {
        case bolt::TuneProfile::Latency:    return "Latency";
        case bolt::TuneProfile::Balanced:   return "Balanced";
        case bolt::TuneProfile::Throughput: return "Throughput";
    }
    return "?";
}

static bool streq(const char* a, const char* b) noexcept {
    return a && b && std::strcmp(a, b) == 0;
}

int main(int argc, char** argv) {
    // --- CLI parse ----------------------------------------------------
    int          cli_threads = 1;
    bolt::TuneProfile cli_profile = bolt::TuneProfile::Balanced;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (streq(a, "--help") || streq(a, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (streq(a, "--threads")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            cli_threads = std::atoi(argv[++i]);
            if (cli_threads < 1) cli_threads = 1;
        } else if (streq(a, "--profile")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            const char* p = argv[++i];
            if      (streq(p, "latency"))    cli_profile = bolt::TuneProfile::Latency;
            else if (streq(p, "balanced"))   cli_profile = bolt::TuneProfile::Balanced;
            else if (streq(p, "throughput")) cli_profile = bolt::TuneProfile::Throughput;
            else { print_usage(argv[0]); return 1; }
        } else if (streq(a, "--rows")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            int64_t n = std::atoll(argv[++i]);
            if (n < 1024) n = 1024;
            kQ1Rows   = n;
            kQ3Orders = n;
            kQ6Rows   = n;
        } else {
            printf("Unknown arg: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("============================================================\n");
    printf("  Bolt TPC-H-lite (v0.1) — synthetic micro-queries\n");
    printf("============================================================\n");
    print_host_banner();
    printf("\n");

    // One master arena sized for all three queries up front. Queries use
    // ArenaGuard internally so per-iter intermediates are released without
    // touching the malloc path.
    // Size the arena to the requested row count. Per-query peak is roughly:
    //   Q1: 4 cols * 8B = 32B/row
    //   Q3: 6 cols + scratch ≈ 80B/row
    //   Q6: 6 cols + scratch ≈ 64B/row
    // Allocate ~256B/row total + 64MB slack so a 100M-row run gets a 32GB
    // arena. The ArenaConfig caps at max_block_size per block; we let it
    // grow up to total need.
    const int64_t bytes_per_row = 256;
    const size_t  rows_max      = static_cast<size_t>(
        kQ1Rows > kQ6Rows ? (kQ1Rows > kQ3Orders ? kQ1Rows : kQ3Orders)
                          : (kQ6Rows > kQ3Orders ? kQ6Rows : kQ3Orders));
    const size_t  arena_bytes   = rows_max * bytes_per_row + (64ull << 20);
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = arena_bytes < (64ull << 20) ? (64ull << 20) : arena_bytes;
    cfg.max_block_size     = cfg.initial_block_size;
    bolt::Arena arena(cfg);

    // --- Q1 base columns --------------------------------------------------
    int32_t* q1_rflag = arena.allocate_array<int32_t>(kQ1Rows);
    int64_t* q1_qty   = arena.allocate_array<int64_t>(kQ1Rows);
    int64_t* q1_price = arena.allocate_array<int64_t>(kQ1Rows);
    if (!q1_rflag || !q1_qty || !q1_price) {
        printf("FATAL: Q1 arena allocation failed\n");
        return 1;
    }
    gen_q1(q1_rflag, q1_qty, q1_price, kQ1Rows, 0xB01D1ull);

    // groupby_sum_int64 expects an int64 (or uint64) key column. Promote the
    // int32 return-flag buffer to int64 at setup time — still outside timer.
    int64_t* q1_rflag64 = arena.allocate_array<int64_t>(kQ1Rows);
    if (!q1_rflag64) { printf("FATAL: Q1 key widen failed\n"); return 1; }
    for (int64_t i = 0; i < kQ1Rows; ++i) {
        q1_rflag64[i] = static_cast<int64_t>(q1_rflag[i]);
    }

    bolt::BoltColumn q1_rflag_col = bolt::BoltColumn::make_flat(
        q1_rflag64, nullptr, kQ1Rows, bolt::BoltType::Int64);
    bolt::BoltColumn q1_qty_col = bolt::BoltColumn::make_flat(
        q1_qty, nullptr, kQ1Rows, bolt::BoltType::Int64);
    bolt::BoltColumn q1_price_col = bolt::BoltColumn::make_flat(
        q1_price, nullptr, kQ1Rows, bolt::BoltType::Int64);

    // --- Q3 base columns --------------------------------------------------
    int64_t* q3_cb = arena.allocate_array<int64_t>(kQ3Customers);
    int64_t* q3_cp = arena.allocate_array<int64_t>(kQ3Orders);
    int64_t* q3_tp = arena.allocate_array<int64_t>(kQ3Orders);
    if (!q3_cb || !q3_cp || !q3_tp) {
        printf("FATAL: Q3 arena allocation failed\n");
        return 1;
    }
    gen_q3(q3_cb, kQ3Customers, q3_cp, q3_tp, kQ3Orders, 0xB01D2ull);
    bolt::BoltColumn q3_cb_col = bolt::BoltColumn::make_flat(
        q3_cb, nullptr, kQ3Customers, bolt::BoltType::Int64);
    bolt::BoltColumn q3_cp_col = bolt::BoltColumn::make_flat(
        q3_cp, nullptr, kQ3Orders, bolt::BoltType::Int64);
    bolt::BoltColumn q3_tp_col = bolt::BoltColumn::make_flat(
        q3_tp, nullptr, kQ3Orders, bolt::BoltType::Int64);

    // --- Q6 base columns --------------------------------------------------
    int64_t* q6_price = arena.allocate_array<int64_t>(kQ6Rows);
    int64_t* q6_disc  = arena.allocate_array<int64_t>(kQ6Rows);
    int64_t* q6_date  = arena.allocate_array<int64_t>(kQ6Rows);
    if (!q6_price || !q6_disc || !q6_date) {
        printf("FATAL: Q6 arena allocation failed\n");
        return 1;
    }
    gen_q6(q6_price, q6_disc, q6_date, kQ6Rows, 0xB01D3ull);
    bolt::BoltColumn q6_price_col = bolt::BoltColumn::make_flat(
        q6_price, nullptr, kQ6Rows, bolt::BoltType::Int64);
    bolt::BoltColumn q6_disc_col = bolt::BoltColumn::make_flat(
        q6_disc, nullptr, kQ6Rows, bolt::BoltType::Int64);
    bolt::BoltColumn q6_date_col = bolt::BoltColumn::make_flat(
        q6_date, nullptr, kQ6Rows, bolt::BoltType::Int64);

    // --- Scheduler (built outside timed regions) ----------------------
    bolt::Scheduler  sched_storage;
    bolt::Scheduler* sched_ptr = nullptr;
    bool sched_live = false;
    if (cli_threads > 1) {
        bolt::SchedulerConfig scfg = bolt::SchedulerConfig::with_profile(cli_profile);
        scfg.num_workers = static_cast<uint32_t>(cli_threads);
        // Leave pin_workers / numa_bind profile-driven (per spec).
        if (!sched_storage.init(scfg)) {
            printf("FATAL: scheduler init failed (threads=%d)\n", cli_threads);
            return 1;
        }
        sched_live = true;
        sched_ptr  = &sched_storage;
    }

    // Report the scheduler configuration just before the results table.
    if (sched_ptr) {
        const auto& c = sched_ptr->config();
        printf("Scheduler: %u workers, profile=%s, pinned=%s, grain=%u KB\n",
               sched_ptr->thread_count(),
               profile_name(c.tune),
               c.pin_workers ? "true" : "false",
               c.grain_bytes / 1024u);
    } else {
        printf("Scheduler: 1 worker (serial baseline, no pool), profile=%s\n",
               profile_name(cli_profile));
    }

    // ------------------------------------------------------------------
#if BOLT_HAS_PARALLEL
    printf("Parallel kernels: available (bolt_parallel.h present).\n");
#else
    printf("Parallel kernels: not present — falling back to serial kernels.\n");
#endif

    QueryResult q1 = run_q1(&arena, &q1_rflag_col, &q1_qty_col, &q1_price_col, sched_ptr);
    QueryResult q3 = run_q3(&arena, &q3_cb_col, &q3_cp_col, &q3_tp_col, sched_ptr);
    QueryResult q6 = run_q6(&arena, &q6_price_col, &q6_disc_col, &q6_date_col, sched_ptr);

    printf("\n");
    printf("TPC-H-lite (N=1M rows, synthetic):\n");
    print_row(q1);
    print_row(q3);
    print_row(q6);
    printf("\n");
    if (sched_ptr) {
        printf("Notes: %u-worker scheduler path. No heap allocation in timed regions.\n",
               sched_ptr->thread_count());
    } else {
        printf("Notes: single-thread, single-core path. No heap allocation in\n");
        printf("       timed regions. Pass --threads N for multi-worker execution.\n");
    }

    if (sched_live) sched_storage.shutdown();

    return 0;
}
