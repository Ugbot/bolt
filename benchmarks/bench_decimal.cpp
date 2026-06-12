// bench_decimal.cpp — floors for the Perf Wave 2 kernel additions:
//
//   col-literal arithmetic     add_lit / sub_lit / lit_sub / mul_lit
//                              (int64 + double) and the Decimal128
//                              d128_add_lit / d128_sub_lit /
//                              d128_lit_sub_col / d128_mul_lit family
//   scale-aware d128 filter    d128_filter_cmp_col (same- + mixed-scale)
//   mixed-compare f64 filter   f64_filter_cmp_col + coercion helpers
//
// Harness mirrors bench_groupby: 6M rows, warmup + best-of-5, ns/row,
// printed against an honest floor (watermark of measured reality, not an
// aspiration). Also measures the MSVC 16-byte-struct hazard: the lane-wise
// d128_add_lit vs a by-value `out[i] = d128_add(a[i], lit)` loop — the
// shipped kernel must not lose to the by-value shape.
//
// Tiger Style: no heap inside timed regions; arrays pre-allocated in an
// Arena; kMaxIters fixed cap; preconditions asserted outside the loop.

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "bolt/bolt_arena.h"
#include "bolt/kernels/bolt_numeric.h"
#include "bolt/kernels/bolt_decimal.h"

using namespace bolt;
using namespace bolt::kernels;
namespace dec = bolt::kernels::decimal;

namespace {

constexpr int     kMaxIters = 15;  // warmup happens on iter 0; best-of wins
                                   // (high rep count: 6M-row streams are
                                   // DRAM-bound and the floor is the min
                                   // across quiet windows on a loaded box)
constexpr int64_t kRows     = 6'000'000;

volatile int64_t g_sink = 0;       // defeat dead-code elimination

// Best-of-kMaxIters ns/row for a callable returning an int64 checksum.
template <typename F>
double time_best(int64_t n, F&& fn) noexcept {
    assert(n > 0);
    double best = 1e30;
    for (int it = 0; it < kMaxIters; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        g_sink += fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        const double per = ns / static_cast<double>(n);
        if (per < best) best = per;
    }
    assert(best < 1e30);
    return best;
}

void report(const char* name, double measured, double floor_ns) noexcept {
    std::printf("  %-26s measured=%.3f ns/row   floor<=%.2f  %s\n",
                name, measured, floor_ns, (measured <= floor_ns) ? "ok" : "OVER");
}

struct Buffers {
    int64_t*         i64_a;
    double*          f64_a;
    dec::Decimal128* d_a;
    dec::Decimal128* d_b;
    int64_t*         i64_out;
    double*          f64_out;
    double*          f64_b;
    dec::Decimal128* d_out;
    int32_t*         sel_out;
};

void fill(Buffers* bf, int64_t n) noexcept {
    assert(bf != nullptr);
    assert(n > 0);
    uint64_t s = 0x12345678ull;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const int64_t v = static_cast<int64_t>(s >> 40) - (1 << 22);  // ±~4M
        bf->i64_a[i] = v;
        bf->f64_a[i] = static_cast<double>(v) / 100.0;
        bf->f64_b[i] = static_cast<double>(static_cast<int64_t>(s >> 41)) / 100.0;
        bf->d_a[i] = dec::d128_from_i64(v);
        bf->d_b[i] = dec::d128_from_i64(static_cast<int64_t>(s >> 41) - (1 << 21));
        // Pre-fault every OUTPUT buffer too: the first kernel measured against
        // a never-touched 96MB buffer would otherwise pay the page-fault bill
        // inside its timing window (observed: +2x on the first d128 bench).
        bf->i64_out[i] = 0;
        bf->f64_out[i] = 0.0;
        bf->d_out[i].lo = 0;
        bf->d_out[i].hi = 0;
        bf->sel_out[i] = 0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int64_t n = kRows;
    if (argc >= 2) n = std::strtoll(argv[1], nullptr, 10);
    if (n <= 0) n = kRows;

    Arena setup;
    Buffers bf{};
    bf.i64_a   = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    bf.f64_a   = setup.allocate_array<double>(static_cast<size_t>(n));
    bf.f64_b   = setup.allocate_array<double>(static_cast<size_t>(n));
    bf.d_a     = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    bf.d_b     = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    bf.i64_out = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    bf.f64_out = setup.allocate_array<double>(static_cast<size_t>(n));
    bf.d_out   = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    bf.sel_out = setup.allocate_array<int32_t>(static_cast<size_t>(n));
    assert(bf.i64_a && bf.f64_a && bf.f64_b && bf.d_a && bf.d_b);
    assert(bf.i64_out && bf.f64_out && bf.d_out && bf.sel_out);
    fill(&bf, n);

    std::printf("== bench_decimal (Perf Wave 2 kernels) ==\n(rows=%lld)\n",
                static_cast<long long>(n));

    // --- col-literal arithmetic, int64 ------------------------------------
    const double i64_add = time_best(n, [&]() noexcept {
        add_lit<int64_t>(bf.i64_a, n, 12345, bf.i64_out);
        return bf.i64_out[n - 1];
    });
    const double i64_sub = time_best(n, [&]() noexcept {
        sub_lit<int64_t>(bf.i64_a, n, 12345, bf.i64_out);
        return bf.i64_out[n - 1];
    });
    const double i64_lsub = time_best(n, [&]() noexcept {
        lit_sub<int64_t>(12345, bf.i64_a, n, bf.i64_out);
        return bf.i64_out[n - 1];
    });
    const double i64_mul = time_best(n, [&]() noexcept {
        mul_lit<int64_t>(bf.i64_a, n, 7, bf.i64_out);
        return bf.i64_out[n - 1];
    });

    // --- col-literal arithmetic, double ------------------------------------
    const double f64_add = time_best(n, [&]() noexcept {
        add_lit<double>(bf.f64_a, n, 0.94, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });
    const double f64_sub = time_best(n, [&]() noexcept {
        sub_lit<double>(bf.f64_a, n, 0.94, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });
    const double f64_lsub = time_best(n, [&]() noexcept {
        lit_sub<double>(1.0, bf.f64_a, n, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });
    const double f64_mul = time_best(n, [&]() noexcept {
        mul_lit<double>(bf.f64_a, n, 0.94, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });

    // --- Decimal128 col-literal arithmetic ---------------------------------
    const dec::Decimal128 lit = dec::d128_from_i64(94);  // pre-rescaled by caller
    const double d_add = time_best(n, [&]() noexcept {
        dec::d128_add_lit(bf.d_a, n, lit, bf.d_out);
        return bf.d_out[n - 1].lo;
    });
    // MSVC hazard cross-check: by-value d128_add round-trip per row.
    const double d_add_byval = time_best(n, [&]() noexcept {
        for (int64_t i = 0; i < n; ++i) bf.d_out[i] = dec::d128_add(bf.d_a[i], lit);
        return bf.d_out[n - 1].lo;
    });
    const double d_sub = time_best(n, [&]() noexcept {
        dec::d128_sub_lit(bf.d_a, n, lit, bf.d_out);
        return bf.d_out[n - 1].lo;
    });
    const double d_lsub = time_best(n, [&]() noexcept {
        dec::d128_lit_sub_col(lit, bf.d_a, n, bf.d_out);
        return bf.d_out[n - 1].lo;
    });
    const double d_mul = time_best(n, [&]() noexcept {
        dec::d128_mul_lit(bf.d_a, n, lit, bf.d_out);
        return bf.d_out[n - 1].lo;
    });

    // --- Decimal128 scale-aware compare-filter ------------------------------
    const double d_flt_same = time_best(n, [&]() noexcept {  // sa==sb fast path
        return dec::d128_filter_cmp_col(bf.d_a, 2, bf.d_b, 2, n, /*lt*/2,
                                        bf.sel_out);
    });
    const double d_flt_mixed = time_best(n, [&]() noexcept {  // sa<sb, hoisted mul
        return dec::d128_filter_cmp_col(bf.d_a, 0, bf.d_b, 2, n, /*lt*/2,
                                        bf.sel_out);
    });

    // --- f64 mixed-compare filter + coercion --------------------------------
    const double f_flt = time_best(n, [&]() noexcept {
        return f64_filter_cmp_col(bf.f64_a, bf.f64_b, n, /*lt*/2, bf.sel_out);
    });
    const double w_i64 = time_best(n, [&]() noexcept {
        widen_i64_to_f64(bf.i64_a, n, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });
    const double c_d128 = time_best(n, [&]() noexcept {
        dec::d128_to_f64(bf.d_a, 2, n, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });
    const double c_d64 = time_best(n, [&]() noexcept {
        dec::d64_to_f64(bf.i64_a, 2, n, bf.f64_out);
        return static_cast<int64_t>(bf.f64_out[n - 1]);
    });

    // --- floors --------------------------------------------------------------
    // Honest watermarks, NOT aspirations. Calibration (2026-06-12, AVX2 client
    // box, VS2022 /O2, best-of-15 × ≥6 process runs, concurrent build load):
    // at 6M rows every kernel streams 48-96 MB — DRAM-bound, not ALU-bound.
    // Best observed mins: 8B-stream lit ops 1.3-2.0 ns/row; d128 (16B) lit ops
    // 2.2-3.6; d128_mul_lit 2.45; filter same-scale 5.15 / mixed-scale 9.16
    // (read 32B/row + per-row 128-bit mul); f64 filter 1.45; widen 1.33;
    // d128_to_f64 3.85 (i64→f64 convert + fdiv, 24B/row); d64_to_f64 1.59.
    // Floors = min × ~1.3-2x to absorb shared-DRAM noise; the pure 8B-stream
    // lit ops get the widest margin (no compute to hide contention behind).
    // Revisit only with measured evidence (e.g. a dedicated-box calibration).
    std::printf("== floors ==\n");
    report("add_lit_i64",        i64_add, 3.0);
    report("sub_lit_i64",        i64_sub, 3.0);
    report("lit_sub_i64",        i64_lsub, 3.0);
    report("mul_lit_i64",        i64_mul, 3.0);
    report("add_lit_f64",        f64_add, 3.0);
    report("sub_lit_f64",        f64_sub, 3.0);
    report("lit_sub_f64",        f64_lsub, 3.0);
    report("mul_lit_f64",        f64_mul, 3.0);
    report("d128_add_lit",       d_add, 4.5);
    std::printf("  %-26s measured=%.3f ns/row   (lane kernel %.3f, ratio=%.2fx — lane must not lose)\n",
                "d128_add_lit_byvalue", d_add_byval, d_add,
                (d_add > 0.0) ? (d_add_byval / d_add) : 0.0);
    report("d128_sub_lit",       d_sub, 4.5);
    report("d128_lit_sub_col",   d_lsub, 4.5);
    report("d128_mul_lit",       d_mul, 4.5);
    report("d128_filter_same_scale", d_flt_same, 7.5);
    report("d128_filter_mixed_scale", d_flt_mixed, 12.5);
    report("f64_filter_cmp_col", f_flt, 2.5);
    report("widen_i64_to_f64",   w_i64, 2.5);
    report("d128_to_f64",        c_d128, 5.5);
    report("d64_to_f64",         c_d64, 2.5);
    std::printf("(sink=%lld)\n", static_cast<long long>(g_sink));
    return 0;
}
