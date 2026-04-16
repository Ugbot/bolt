// bench_1brc.cpp — 1 Billion Row Challenge (1BRC) compute-shape microbench.
//
// Scope (v0.1 — preparsed path only):
//   Per-station min/max/sum aggregation over N synthetic rows, with the
//   1BRC-canonical cardinality of 413 distinct stations. Temperatures are
//   integer tenths of a degree in [-999, 999].
//
//   Mode `preparsed` skips CSV entirely: keys + int10th temps are generated
//   directly into arena-backed Flat Int64 columns, and we time the pure
//   compute path (`parallel_groupby_sum_int64`). This is the right baseline:
//   the 1BRC reference submissions spend the bulk of their time on parse —
//   isolating compute tells us the ceiling the parser has to chase.
//
//   Mode `parsed` (stub) will time CSV-bytes → columns → compute once K1's
//   `bolt::ingest::parse_csv` lands. Guarded by `__has_include` so this file
//   builds whether or not `bolt/ingest/bolt_csv.h` exists yet — when missing
//   the mode prints a "skipped" line.
//
// NOTE v0: we proxy {min, max, sum, count} with sum + count (count comes
// from the serial GroupByTable payload, not exposed by the parallel API,
// so the output table here is sum-only — TODO wire min/max once a
// `min_masked` / `max_masked` kernel or a parallel_groupby_full exists).
//
// Tiger Style:
//   - No heap inside the timed region. Arena pre-sized for N rows once at
//     startup. Per-iteration scratch is carved via `ArenaGuard`.
//   - No exceptions, no RTTI, no smart pointers. Every function noexcept.
//   - Hard caps: kMaxIters == 3 here to keep 100M-row runs sane.
//   - All preconditions asserted outside the timed region; never assert
//     inside it.
//
// Build target: bench_1brc (see benchmarks/CMakeLists.txt).

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/join/bolt_groupby.h"
#include "bolt/join/bolt_swiss.h"

#if __has_include("bolt/kernels/bolt_parallel.h")
  #include "bolt/kernels/bolt_parallel.h"
  #define BOLT_HAS_PARALLEL 1
#else
  #define BOLT_HAS_PARALLEL 0
#endif

#if __has_include("bolt/ingest/bolt_csv.h")
  #include "bolt/ingest/bolt_csv.h"
  #define BOLT_HAS_INGEST 1
#else
  #define BOLT_HAS_INGEST 0
#endif

using Clock = std::chrono::high_resolution_clock;
using ns    = std::chrono::nanoseconds;

// ---------------------------------------------------------------------------
// DoNotOptimize / ClobberMemory — mirror bench_tpch_lite.
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
// Sample bookkeeping — small fixed buffer for kMaxIters.
// ---------------------------------------------------------------------------
static constexpr int kMaxIters = 8;

static int64_t min_ns(const int64_t* samples, int n) noexcept {
    assert(samples != nullptr);
    assert(n > 0 && n <= kMaxIters);
    int64_t m = samples[0];
    for (int i = 1; i < n; ++i) if (samples[i] < m) m = samples[i];
    return m;
}

static int64_t median_ns(const int64_t* samples, int n) noexcept {
    assert(samples != nullptr);
    assert(n > 0 && n <= kMaxIters);
    int64_t buf[kMaxIters];
    for (int i = 0; i < n; ++i) buf[i] = samples[i];
    std::sort(buf, buf + n);
    return buf[n / 2];
}

// ---------------------------------------------------------------------------
// Tunables — default 100M rows, 413 stations (canonical 1BRC cardinality).
// ---------------------------------------------------------------------------
static int64_t  kRows     = 100'000'000;   // total row count
static int64_t  kStations = 413;           // distinct stations
static constexpr int kIters = 3;           // timed iterations

// ---------------------------------------------------------------------------
// Deterministic station-id generator.
// swiss_mix(i) | 1 — mix the counter, force low bit so station ids are
// distinct from each other and from a naïve 0..N index. `| 1` is cheap
// insurance that we never emit 0, which the SwissTable treats as empty.
// ---------------------------------------------------------------------------
static void gen_stations(int64_t* out, int64_t n) noexcept {
    assert(out != nullptr);
    assert(n > 0);
    for (int64_t i = 0; i < n; ++i) {
        uint64_t h = bolt::swiss_mix(static_cast<uint64_t>(i));
        out[i] = static_cast<int64_t>(h | 1ULL);
    }
}

// Generate keys[i] = stations[i % kStations], temps[i] = pseudo-uniform in
// [-999, 999] (int10ths of a degree). Both deterministic so repeated runs
// share the same data-dependent compute path.
static void gen_rows(const int64_t* stations, int64_t ns_count,
                     int64_t* keys, int64_t* temps, int64_t n) noexcept {
    assert(stations != nullptr);
    assert(keys != nullptr);
    assert(temps != nullptr);
    assert(ns_count > 0);
    assert(n > 0);
    constexpr uint64_t kMix = 0x9E3779B97F4A7C15ULL;
    for (int64_t i = 0; i < n; ++i) {
        keys[i] = stations[i % ns_count];
        uint64_t h = static_cast<uint64_t>(i) * kMix;
        int64_t v = static_cast<int64_t>((h >> 50) % 1999) - 999;
        temps[i] = v;
    }
}

// ---------------------------------------------------------------------------
// CLI plumbing.
// ---------------------------------------------------------------------------
enum class Mode { Preparsed, Parsed };

static const char* mode_name(Mode m) noexcept {
    switch (m) {
        case Mode::Preparsed: return "preparsed";
        case Mode::Parsed:    return "parsed";
    }
    return "?";
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

static void print_usage(const char* exe) noexcept {
    printf("Usage: %s [--rows N] [--threads N] [--profile NAME]\n"
           "            [--mode preparsed|parsed] [--stations N]\n"
           "            [--prefetch N] [--grain-kb N] [--help]\n",
           exe ? exe : "bench_1brc");
    printf("  --rows N           Row count (default 100000000).\n");
    printf("  --threads N        Worker threads (default 1 = serial baseline).\n");
    printf("  --profile NAME     Scheduler tuning: latency|balanced|throughput.\n");
    printf("  --mode MODE        preparsed (default) | parsed.\n");
    printf("  --stations N       Distinct station count (default 413).\n");
    printf("  --prefetch N       GroupByTable lookahead-prefetch distance\n");
    printf("                     (default 0 = off; ~16 wins for high cardinality).\n");
    printf("  --grain-kb N       Override scheduler grain_bytes (default profile).\n");
    printf("  --two-pass         Use ingest_two_pass (hoisted-probe variant).\n");
}

// ---------------------------------------------------------------------------
// Preparsed path — time only the compute (group-by sum).
// ---------------------------------------------------------------------------
struct RunResult {
    int64_t median_total_ns;
    int64_t min_total_ns;
    int64_t groups_out;
};

static bool run_preparsed(bolt::Arena* scratch,
                          bolt::BoltColumn* keys_col,
                          bolt::BoltColumn* temps_col,
                          bolt::Scheduler*  sched,
                          RunResult*        out) noexcept {
    assert(scratch != nullptr);
    assert(keys_col != nullptr);
    assert(temps_col != nullptr);
    assert(out != nullptr);
    assert(keys_col->length == temps_col->length);

    int64_t samples[kMaxIters];
    int64_t groups_last = 0;
    int n = 0;

    // Warm-up: primes the per-morsel partial tables + worker arenas.
    // scratch->reset() wipes all scratch-state between calls — it holds no
    // persistent data (base columns live in a separate data arena).
    {
        scratch->reset();
        bolt::ArenaGuard g(*scratch);
        bolt::BoltColumn ok{}, os{}, oc{}, on{}, ox{};
#if BOLT_HAS_PARALLEL
        bool ok_run = bolt::parallel_groupby_agg_int64(
            sched, *keys_col, *temps_col, scratch, &ok, &os, &oc, &on, &ox);
#else
        (void)sched;
        bool ok_run = bolt::groupby_agg_int64(*keys_col, *temps_col, scratch,
                                              &ok, &os, &oc, &on, &ox);
#endif
        if (!ok_run) {
            printf("FATAL: warm-up groupby failed\n");
            return false;
        }
        DoNotOptimize(os.data);
        DoNotOptimize(on.data);
        DoNotOptimize(ox.data);
    }

    int64_t spot_min = 0, spot_max = 0, spot_count = 0, spot_sum = 0;
    int64_t spot_key = 0;
    for (int it = 0; it < kIters; ++it) {
        scratch->reset();
        bolt::ArenaGuard g(*scratch);
        bolt::BoltColumn ok{}, os{}, oc{}, on{}, ox{};

        auto t0 = Clock::now();
#if BOLT_HAS_PARALLEL
        bool ok_run = bolt::parallel_groupby_agg_int64(
            sched, *keys_col, *temps_col, scratch, &ok, &os, &oc, &on, &ox);
#else
        bool ok_run = bolt::groupby_agg_int64(*keys_col, *temps_col, scratch,
                                              &ok, &os, &oc, &on, &ox);
#endif
        auto t1 = Clock::now();

        if (!ok_run) {
            printf("FATAL: groupby iter %d failed\n", it);
            return false;
        }
        DoNotOptimize(os.data);
        DoNotOptimize(ok.data);
        DoNotOptimize(on.data);
        DoNotOptimize(ox.data);
        DoNotOptimize(oc.data);
        ClobberMemory();

        samples[n++] = std::chrono::duration_cast<ns>(t1 - t0).count();
        groups_last = ok.length;

        // Capture spot values from the first group of the last iter for the
        // post-loop sanity print. (Output ordering is implementation-defined
        // but stable within one run.)
        if (it == kIters - 1 && ok.length > 0) {
            const int64_t* k_arr = static_cast<const int64_t*>(ok.data);
            const int64_t* s_arr = static_cast<const int64_t*>(os.data);
            const int64_t* c_arr = static_cast<const int64_t*>(oc.data);
            const int64_t* n_arr = static_cast<const int64_t*>(on.data);
            const int64_t* x_arr = static_cast<const int64_t*>(ox.data);
            spot_key   = k_arr[0];
            spot_sum   = s_arr[0];
            spot_count = c_arr[0];
            spot_min   = n_arr[0];
            spot_max   = x_arr[0];
        }
    }

    out->median_total_ns = median_ns(samples, n);
    out->min_total_ns    = min_ns(samples, n);
    out->groups_out      = groups_last;
    printf("  spot-check group[0]: key=%lld sum=%lld count=%lld min=%lld max=%lld\n",
           static_cast<long long>(spot_key),
           static_cast<long long>(spot_sum),
           static_cast<long long>(spot_count),
           static_cast<long long>(spot_min),
           static_cast<long long>(spot_max));
    if (spot_min < -999 || spot_max > 999 || spot_min > spot_max) {
        printf("  WARN: spot-check looks wrong (expected min..max in [-999,999])\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// Parsed path — stub until K1's bolt::ingest::parse_csv lands.
// ---------------------------------------------------------------------------
static bool run_parsed(bolt::Arena* /*arena*/,
                       const int64_t* /*stations*/, int64_t /*ns_count*/,
                       const int64_t* /*keys*/, const int64_t* /*temps*/,
                       int64_t /*n*/, bolt::Scheduler* /*sched*/,
                       RunResult* /*out*/) noexcept {
#if BOLT_HAS_INGEST
    // TODO(K1): format keys+temps into CSV bytes ("<key_hex>;<int10>\n"),
    // time parse_csv + parallel_groupby_sum_int64 end-to-end.
    printf("parsed mode: bolt::ingest detected, but bench_1brc glue is TODO.\n");
    printf("             Once K1 lands, this path will format synthetic CSV\n"
           "             into arena bytes and time parse+groupby.\n");
    return false;
#else
    printf("parsed mode: skipped (build with BOLT_BUILD_INGEST=ON once K1 lands).\n");
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Host banner.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int               cli_threads = 1;
    bolt::TuneProfile cli_profile = bolt::TuneProfile::Throughput;
    int               cli_prefetch = -1;     // -1 = leave default (off)
    int               cli_grain_kb = 0;      // 0 = use profile default
    int               cli_two_pass = 0;      // 1 = use ingest_two_pass
    Mode              cli_mode    = Mode::Preparsed;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (streq(a, "--help") || streq(a, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (streq(a, "--rows")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            int64_t n = std::atoll(argv[++i]);
            if (n < 1024) n = 1024;
            kRows = n;
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
        } else if (streq(a, "--mode")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            const char* m = argv[++i];
            if      (streq(m, "preparsed")) cli_mode = Mode::Preparsed;
            else if (streq(m, "parsed"))    cli_mode = Mode::Parsed;
            else { print_usage(argv[0]); return 1; }
        } else if (streq(a, "--stations")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            int64_t n = std::atoll(argv[++i]);
            if (n < 1) n = 1;
            kStations = n;
        } else if (streq(a, "--prefetch")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            cli_prefetch = std::atoi(argv[++i]);
            if (cli_prefetch < 0) cli_prefetch = 0;
            if (cli_prefetch > 64) cli_prefetch = 64;
        } else if (streq(a, "--grain-kb")) {
            if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
            cli_grain_kb = std::atoi(argv[++i]);
            if (cli_grain_kb < 1) cli_grain_kb = 1;
        } else if (streq(a, "--two-pass")) {
            cli_two_pass = 1;
        } else {
            printf("Unknown arg: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("============================================================\n");
    printf("  Bolt 1BRC-shape (v0.1) — per-station agg, %lld rows\n",
           static_cast<long long>(kRows));
    printf("============================================================\n");
    print_host_banner();
    printf("Rows=%lld  Stations=%lld  Mode=%s  Threads=%d  Profile=%s\n",
           static_cast<long long>(kRows), static_cast<long long>(kStations),
           mode_name(cli_mode), cli_threads, profile_name(cli_profile));
    printf("\n");

    // -----------------------------------------------------------------
    // Arena sizing (preparsed path):
    //   base columns:      2 * 8 B/row (keys + temps)             = 16 B/row
    //   per-morsel partials + radix merge + output:
    //                      ~kStations * (24 B MergeTriple
    //                       + slop)  — dominated by partials × morsels.
    //
    // A coarse, robust envelope: parallel path's per-morsel SwissTables are
    // bounded by `grain`, but the serial fallback hints its table at the
    // full row count — SwissTable rounds to next-pow-2 and allocates control
    // bytes + slots proportional to `hint`. At 100M rows serial that's
    // ~4 GB of table state alone, so we provision generously. The machine
    // has 16+ GB; we cap arena growth by the OS rather than the config.
    // -----------------------------------------------------------------
    // Two arenas:
    //   data_arena    — persistent, holds keys/temps/stations across all iters.
    //   scratch_arena — reset() per iteration, holds the per-call SwissTable
    //                   + groupby partials. Sized generously (parallel
    //                   radix merge can hold N intermediate triples).
    const size_t data_bytes    = static_cast<size_t>(kRows) * 16ull   // keys+temps
                               + static_cast<size_t>(kStations) * 8ull
                               + (16ull << 20);                       // 16 MB slack
    const size_t scratch_bytes = static_cast<size_t>(kRows) * 64ull   // worst-case partials
                               + (64ull << 20);                       // 64 MB slack

    bolt::ArenaConfig data_cfg{};
    data_cfg.initial_block_size = data_bytes < (64ull << 20) ? (64ull << 20) : data_bytes;
    data_cfg.max_block_size     = data_cfg.initial_block_size;
    bolt::Arena data_arena(data_cfg);

    bolt::ArenaConfig sc_cfg{};
    sc_cfg.initial_block_size = scratch_bytes < (64ull << 20) ? (64ull << 20) : scratch_bytes;
    sc_cfg.max_block_size     = sc_cfg.initial_block_size;
    bolt::Arena scratch_arena(sc_cfg);

    // -----------------------------------------------------------------
    // Data generation — outside the timed region. Uses data_arena only.
    // -----------------------------------------------------------------
    int64_t* stations = data_arena.allocate_array<int64_t>(kStations);
    int64_t* keys     = data_arena.allocate_array<int64_t>(kRows);
    int64_t* temps    = data_arena.allocate_array<int64_t>(kRows);
    if (!stations || !keys || !temps) {
        printf("FATAL: data arena allocation failed (%.2f GB requested)\n",
               static_cast<double>(data_bytes) / (1024.0 * 1024.0 * 1024.0));
        return 1;
    }
    gen_stations(stations, kStations);
    gen_rows(stations, kStations, keys, temps, kRows);

    bolt::BoltColumn keys_col = bolt::BoltColumn::make_flat(
        keys,  nullptr, kRows, bolt::BoltType::Int64);
    bolt::BoltColumn temps_col = bolt::BoltColumn::make_flat(
        temps, nullptr, kRows, bolt::BoltType::Int64);

    // -----------------------------------------------------------------
    // Scheduler — skip pool when threads == 1 (serial baseline).
    // -----------------------------------------------------------------
    bolt::Scheduler  sched_storage;
    bolt::Scheduler* sched_ptr = nullptr;
    bool             sched_live = false;
    if (cli_threads > 1) {
        bolt::SchedulerConfig scfg =
            bolt::SchedulerConfig::with_profile(cli_profile);
        scfg.num_workers = static_cast<uint32_t>(cli_threads);
        if (cli_grain_kb > 0) {
            scfg.grain_bytes = static_cast<uint32_t>(cli_grain_kb) * 1024u;
        }
        if (!sched_storage.init(scfg)) {
            printf("FATAL: scheduler init failed (threads=%d)\n", cli_threads);
            return 1;
        }
        sched_live = true;
        sched_ptr  = &sched_storage;
    }

    if (sched_ptr) {
        const auto& c = sched_ptr->config();
        printf("Scheduler: %u workers, profile=%s, pinned=%s, grain=%u KB\n",
               sched_ptr->thread_count(), profile_name(c.tune),
               c.pin_workers ? "true" : "false", c.grain_bytes / 1024u);
    } else {
        printf("Scheduler: 1 worker (serial baseline, no pool), profile=%s\n",
               profile_name(cli_profile));
    }
#if BOLT_HAS_PARALLEL
    if (cli_prefetch >= 0) {
        bolt::parallel_groupby_prefetch_ahead_override() =
            static_cast<uint16_t>(cli_prefetch);
        printf("Prefetch override: prefetch_ahead=%d\n", cli_prefetch);
    }
    if (cli_two_pass) {
        bolt::parallel_groupby_two_pass_override() = 1;
        printf("Two-pass hoisted-probe ingest: ENABLED\n");
    }
#endif
#if BOLT_HAS_PARALLEL
    printf("Parallel kernels: available.\n");
#else
    printf("Parallel kernels: not present — serial fallback.\n");
#endif
#if BOLT_HAS_INGEST
    printf("bolt::ingest: linked (parsed mode available).\n");
#else
    printf("bolt::ingest: not linked (parsed mode will skip).\n");
#endif
    printf("\n");

    // -----------------------------------------------------------------
    // Dispatch.
    // -----------------------------------------------------------------
    RunResult result{};
    bool ok = false;
    if (cli_mode == Mode::Preparsed) {
        ok = run_preparsed(&scratch_arena, &keys_col, &temps_col, sched_ptr, &result);
    } else {
        ok = run_parsed(&scratch_arena, stations, kStations, keys, temps, kRows,
                        sched_ptr, &result);
    }

    if (!ok) {
        if (sched_live) sched_storage.shutdown();
        return cli_mode == Mode::Parsed ? 0 : 1;  // parsed skip is not a failure
    }

    // -----------------------------------------------------------------
    // Reporting.
    // -----------------------------------------------------------------
    const double total_ms = static_cast<double>(result.median_total_ns) / 1e6;
    const double per_row  = static_cast<double>(result.median_total_ns) /
                            static_cast<double>(kRows);
    const double rows_s   = static_cast<double>(kRows) /
                            (static_cast<double>(result.median_total_ns) / 1e9);

    printf("1BRC-shape (N=%lld rows, %lld stations, mode=%s):\n",
           static_cast<long long>(kRows), static_cast<long long>(kStations),
           mode_name(cli_mode));
    printf("  total: median %.2f ms   min %.2f ms\n",
           total_ms,
           static_cast<double>(result.min_total_ns) / 1e6);
    printf("  per-row: %.2f ns/row    throughput: %.1f M rows/s\n",
           per_row, rows_s / 1e6);
    printf("  groups emitted: %lld (expected %lld)\n",
           static_cast<long long>(result.groups_out),
           static_cast<long long>(kStations));

    printf("\nNotes: no heap allocation in timed region. kIters=%d.\n", kIters);
    printf("       v0 reports sum-only; min/max/count require a\n"
           "       parallel_groupby_full API (follow-on).\n");

    if (sched_live) sched_storage.shutdown();
    return 0;
}
