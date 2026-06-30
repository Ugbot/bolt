// bolt/kernels/window_agg.h — per-row windowed aggregate over one sorted
// partition slice. The substrate for SQL aggregate-as-window functions
// (`SUM/AVG/MIN/MAX/COUNT/STDDEV(x) OVER (... ROWS/RANGE frame)`).
//
// One general kernel covers every standard frame by computing, per row i,
// the inclusive window [lo(i), hi(i)] and aggregating vals over it:
//   lo(i) = start_unbounded ? 0     : clamp(i - start_preceding, 0, n-1)
//   hi(i) = end_unbounded   ? (n-1) : clamp(i + end_following,   0, n-1)
// CURRENT ROW as a bound is start_preceding==0 (lo=i) or end_following==0
// (hi=i). Cumulative (UNBOUNDED PRECEDING → CURRENT ROW) is
// {start_unbounded=true, end_following=0}; trailing-n is
// {start_preceding=n, end_following=0}.
//
// Float64 throughout — the window operator widens Int64 inputs on ingest.
// The operator calls this once per partition over that partition's
// already-order-sorted contiguous value slice.
//
// Shape rules (Tiger Style): noexcept, ≥2 asserts, bounded loops, no heap,
// no exceptions/RTTI, explicit int sizes, BOLT_RESTRICT on data pointers.
//
// Complexity: O(n · w) where w is the mean window width (correctness-first;
// TAQ partitions are ≤ a few hundred rows with windows ≤ 20). A prefix-sum
// fast path for SUM/MEAN/COUNT and a monotonic-deque path for MIN/MAX are
// tracked perf follow-ups; the result values are identical.

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace window_agg {

enum class WinAgg : std::uint8_t {
    Sum      = 0,
    Mean     = 1,   // AVG
    Min      = 2,
    Max      = 3,
    Count    = 4,
    VarSamp  = 5,
    VarPop   = 6,
    StdSamp  = 7,
    StdPop   = 8,
};

// Frame bounds relative to the current row. *_preceding / *_following are
// non-negative row offsets; the *_unbounded flags override them.
struct Frame {
    std::int64_t start_preceding;  // rows before i (ignored if start_unbounded)
    std::int64_t end_following;    // rows after  i (ignored if end_unbounded)
    bool         start_unbounded;  // UNBOUNDED PRECEDING
    bool         end_unbounded;    // UNBOUNDED FOLLOWING
    std::uint8_t _pad[6];
};

// Aggregate one inclusive window [lo, hi] of `vals`. Helper kept < 70 lines.
inline double agg_window(WinAgg kind, const double* BOLT_RESTRICT vals,
                         std::int64_t lo, std::int64_t hi) noexcept {
    assert(vals != nullptr);
    assert(lo <= hi);
    const std::int64_t cnt = hi - lo + 1;
    double sum = 0.0;
    double mn  = vals[lo];
    double mx  = vals[lo];
    for (std::int64_t j = lo; j <= hi; ++j) {
        const double v = vals[j];
        sum += v;
        mn = (v < mn) ? v : mn;
        mx = (v > mx) ? v : mx;
    }
    switch (kind) {
        case WinAgg::Sum:   return sum;
        case WinAgg::Count: return static_cast<double>(cnt);
        case WinAgg::Mean:  return sum / static_cast<double>(cnt);
        case WinAgg::Min:   return mn;
        case WinAgg::Max:   return mx;
        default: break;     // variance / stddev fall through
    }
    const double mean = sum / static_cast<double>(cnt);
    double ss = 0.0;
    for (std::int64_t j = lo; j <= hi; ++j) {
        const double d = vals[j] - mean;
        ss += d * d;
    }
    const bool samp = (kind == WinAgg::VarSamp || kind == WinAgg::StdSamp);
    const double denom = samp ? static_cast<double>(cnt - 1)
                              : static_cast<double>(cnt);
    const double var = (denom > 0.0) ? (ss / denom) : 0.0;
    const bool is_std = (kind == WinAgg::StdSamp || kind == WinAgg::StdPop);
    return is_std ? std::sqrt(var) : var;
}

// Compute out[i] for i in [0, n) over one partition's value slice.
inline void window_agg_f64(WinAgg kind, const double* BOLT_RESTRICT vals,
                           std::int64_t n, Frame fr,
                           double* BOLT_RESTRICT out) noexcept {
    assert((vals != nullptr) || n == 0);
    assert((out  != nullptr) || n == 0);
    assert(fr.start_preceding >= 0 && fr.end_following >= 0);
    if (n <= 0) return;
    // Whole-partition frame (UNBOUNDED PRECEDING .. UNBOUNDED FOLLOWING — the
    // default frame of `agg(x) OVER (PARTITION BY k)` with no ORDER BY): the
    // window is the entire partition for EVERY row, so the result is constant.
    // Compute it once (O(n)) and broadcast, instead of re-aggregating the whole
    // partition per row (O(n^2) — the many-/large-partition cliff). Values are
    // identical to the general path.
    if (fr.start_unbounded && fr.end_unbounded) {
        const double v = agg_window(kind, vals, 0, n - 1);
        for (std::int64_t i = 0; i < n; ++i) out[i] = v;
        return;
    }
    // Cumulative frame (UNBOUNDED PRECEDING .. CURRENT ROW): a running fold,
    // O(n), instead of re-aggregating [0, i] per row (O(n^2) on a big
    // partition). Row semantics match the general index path. Var/Std fall
    // through (running moments are a follow-up).
    if (fr.start_unbounded && !fr.end_unbounded && fr.end_following == 0) {
        switch (kind) {
            case WinAgg::Sum: {
                double s = 0.0;
                for (std::int64_t i = 0; i < n; ++i) { s += vals[i]; out[i] = s; }
                return;
            }
            case WinAgg::Count:
                for (std::int64_t i = 0; i < n; ++i)
                    out[i] = static_cast<double>(i + 1);
                return;
            case WinAgg::Mean: {
                double s = 0.0;
                for (std::int64_t i = 0; i < n; ++i) {
                    s += vals[i];
                    out[i] = s / static_cast<double>(i + 1);
                }
                return;
            }
            case WinAgg::Min: {
                double m = vals[0];
                for (std::int64_t i = 0; i < n; ++i) {
                    if (vals[i] < m) m = vals[i];
                    out[i] = m;
                }
                return;
            }
            case WinAgg::Max: {
                double m = vals[0];
                for (std::int64_t i = 0; i < n; ++i) {
                    if (vals[i] > m) m = vals[i];
                    out[i] = m;
                }
                return;
            }
            default: break;   // VarSamp/VarPop/StdSamp/StdPop → general path
        }
    }
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t lo = fr.start_unbounded ? 0 : (i - fr.start_preceding);
        std::int64_t hi = fr.end_unbounded   ? (n - 1) : (i + fr.end_following);
        if (lo < 0)     lo = 0;
        if (hi > n - 1) hi = n - 1;
        assert(lo <= hi);   // holds: lo<=i<=hi for all standard bounds
        out[i] = agg_window(kind, vals, lo, hi);
    }
}

}  // namespace window_agg
}  // namespace kernels
}  // namespace bolt
