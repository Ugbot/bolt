// bolt/kernels/counter_rate.h: counter / gauge rate family over frames of a (ts, v) series, with an optional
// per-sample start-timestamp column (G2CHK-1173).
//
// The start-timestamp rules are Prometheus's (promql/functions.go extrapolatedRate / instantValue / funcResets,
// float path): a sample may declare when its counter started (epoch ticks, 0 = unknown). A start inside the
// frame, or a start that moved past the previous sample, marks a reset even when the value did not drop, and
// a start inside the frame lets the rate extrapolate to the start instead of the frame edge.
//
// Column shapes. A series' non-null samples sit compact in ts[] / v[] / st[] (st == nullptr: no column, the
// family below equals promql_rate.h exactly). Frame i covers compact samples [pc[lo[i] - base], pc[hi[i] - base])
// of the run view, where pc is the prefix count of valid rows. One call evaluates all m frames.
//
// Tiger Style: noexcept, no allocation, fixed-bound loops, no std:: containers.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_frame_bounds.h"
#include "bolt/kernels/promql_rate.h"

#include <cassert>
#include <cstdint>

#if defined(__clang__)
#pragma float_control(push)
#pragma clang fp contract(off)
#endif

namespace bolt {
namespace kernels {
namespace counter {

enum class RateFn : std::uint8_t { Rate = 0, Increase, Delta, Irate, Idelta, Resets };

// Did the counter restart between two samples, given their declared starts? (Prometheus isReset.)
BOLT_FORCE_INLINE bool st_is_reset(std::int64_t prev_st, std::int64_t prev_t,
                                   std::int64_t curr_st, std::int64_t curr_t) noexcept {
    assert(prev_t <= curr_t);
    if (prev_st == curr_st || curr_st == 0 || curr_st >= curr_t) return false;
    if (curr_st < prev_t) return false;
    if (curr_st > prev_t) return true;
    // curr_st == prev_t: a delta / unknown-start continuation or a real reset, decided by whether prev's own
    // start was known.
    if (prev_st > prev_t) return false;
    return prev_st != 0 && prev_st != prev_t;
}

// rate / increase / delta over n samples with starts. `ts` ascending. is_counter: reset correction;
// is_rate: per second. Fewer than two samples without a usable start, or n == 0: no sample.
inline double st_extrapolated_rate(const std::int64_t* BOLT_RESTRICT ts, const double* BOLT_RESTRICT val,
                                   const std::int64_t* BOLT_RESTRICT st, std::int64_t n,
                                   std::int64_t range_start, std::int64_t range_end,
                                   bool is_counter, bool is_rate, double ts_per_second) noexcept {
    assert(n >= 0 && range_end >= range_start && ts_per_second > 0.0);
    assert((ts != nullptr && val != nullptr && st != nullptr) || n == 0);
    if (n == 0) return promql::promql_no_sample();
    const std::int64_t first_t = ts[0];
    const std::int64_t last_t = ts[n - 1];
    double result = val[n - 1] - val[0];
    if (is_counter) {
        for (std::int64_t i = 1; i < n; ++i) {
            if (val[i] < val[i - 1] || st_is_reset(st[i - 1], ts[i - 1], st[i], ts[i])) result += val[i - 1];
        }
    }
    const std::int64_t num_minus_one = n - 1;
    double duration_to_start = static_cast<double>(first_t - range_start) / ts_per_second;
    double duration_to_end = static_cast<double>(range_end - last_t) / ts_per_second;
    double sampled_interval = static_cast<double>(last_t - first_t) / ts_per_second;
    const double avg_duration = (num_minus_one > 0) ? sampled_interval / static_cast<double>(num_minus_one) : 0.0;
    const double extrap_threshold = avg_duration * 1.1;
    if (is_counter && st[0] != 0 && st[0] > range_start && st[0] < first_t) {
        // The first sample's declared start is inside the frame: a zero sample at that start replaces the
        // leftward extrapolation.
        duration_to_start = 0.0;
        sampled_interval = static_cast<double>(last_t - st[0]) / ts_per_second;
        result += val[0];
    } else if (num_minus_one == 0) {
        return promql::promql_no_sample();
    } else {
        if (duration_to_start >= extrap_threshold) duration_to_start = avg_duration / 2.0;
        if (is_counter) {
            double duration_to_zero = duration_to_start;
            if (result > 0.0 && val[0] >= 0.0) duration_to_zero = sampled_interval * (val[0] / result);
            if (duration_to_zero < duration_to_start) duration_to_start = duration_to_zero;
        }
    }
    if (duration_to_end >= extrap_threshold) duration_to_end = avg_duration / 2.0;
    double factor = 1.0;
    if (sampled_interval != 0.0) factor = (sampled_interval + duration_to_start + duration_to_end) / sampled_interval;
    if (is_rate) {
        const double range_secs = static_cast<double>(range_end - range_start) / ts_per_second;
        if (range_secs != 0.0) factor /= range_secs;
    }
    return result * factor;
}

// irate / idelta over the last two samples, with starts.
inline double st_instant_value(const std::int64_t* BOLT_RESTRICT ts, const double* BOLT_RESTRICT val,
                               const std::int64_t* BOLT_RESTRICT st, std::int64_t n, bool is_rate,
                               double ts_per_second) noexcept {
    assert(n >= 0 && ts_per_second > 0.0);
    assert((ts != nullptr && val != nullptr && st != nullptr) || n == 0);
    if (n < 2) return promql::promql_no_sample();
    const std::int64_t t0 = ts[n - 2], t1 = ts[n - 1];
    const double v0 = val[n - 2], v1 = val[n - 1];
    const std::int64_t sampled_interval = t1 - t0;
    if (sampled_interval == 0) return promql::promql_no_sample();
    double result = v1;
    if (!is_rate || !(v1 < v0 || st_is_reset(st[n - 2], t0, st[n - 1], t1))) result = v1 - v0;
    if (is_rate) result /= (static_cast<double>(sampled_interval) / ts_per_second);
    return result;
}

// resets(): value drops plus declared-start resets.
inline std::int64_t st_resets(const std::int64_t* BOLT_RESTRICT ts, const double* BOLT_RESTRICT val,
                              const std::int64_t* BOLT_RESTRICT st, std::int64_t n) noexcept {
    assert(n >= 0);
    assert((ts != nullptr && val != nullptr && st != nullptr) || n == 0);
    std::int64_t resets = 0;
    for (std::int64_t i = 1; i < n; ++i) {
        if (val[i] < val[i - 1] || st_is_reset(st[i - 1], ts[i - 1], st[i], ts[i])) ++resets;
    }
    return resets;
}

// One sample set -> one answer; st == nullptr is the plain family of promql_rate.h.
inline double rate_one(RateFn fn, const std::int64_t* ts, const double* val, const std::int64_t* st, std::int64_t n,
                       std::int64_t range_start, std::int64_t range_end, double tps) noexcept {
    switch (fn) {
        case RateFn::Rate:
            return st != nullptr ? st_extrapolated_rate(ts, val, st, n, range_start, range_end, true, true, tps)
                                 : promql::promql_rate(ts, val, n, range_start, range_end, tps);
        case RateFn::Increase:
            return st != nullptr ? st_extrapolated_rate(ts, val, st, n, range_start, range_end, true, false, tps)
                                 : promql::promql_increase(ts, val, n, range_start, range_end, tps);
        case RateFn::Delta:
            return st != nullptr ? st_extrapolated_rate(ts, val, st, n, range_start, range_end, false, false, tps)
                                 : promql::promql_delta(ts, val, n, range_start, range_end, tps);
        case RateFn::Irate:
            return st != nullptr ? st_instant_value(ts, val, st, n, true, tps) : promql::promql_irate(ts, val, n, tps);
        case RateFn::Idelta:
            return st != nullptr ? st_instant_value(ts, val, st, n, false, tps) : promql::promql_idelta(ts, val, n, tps);
        case RateFn::Resets:
            return static_cast<double>(st != nullptr ? st_resets(ts, val, st, n) : promql::promql_resets(val, n));
    }
    return promql::promql_no_sample();
}

// m frames of one run in one call. ct / cv / cst: the run's compact samples (cst may be nullptr). pc: prefix
// counts of valid rows over the run view [base, base + len]; frame i is rows [lo[i], hi[i]) of the view and
// its evaluation time pt[i]. rows_frame: a row-count frame (no time bounds: the sample edges are the bounds).
// out[i] / present[i]: the value and whether the frame answers (no sample / wider than int64 ticks: not).
inline void rate_frames(RateFn fn, const std::int64_t* BOLT_RESTRICT ct, const double* BOLT_RESTRICT cv,
                        const std::int64_t* BOLT_RESTRICT cst, const std::uint32_t* BOLT_RESTRICT pc,
                        std::uint32_t base, const std::uint32_t* BOLT_RESTRICT lo,
                        const std::uint32_t* BOLT_RESTRICT hi, const std::int64_t* BOLT_RESTRICT pt, std::uint32_t m,
                        const frame::FrameSpec& spec, bool rows_frame, double tps, double* BOLT_RESTRICT out,
                        std::uint8_t* BOLT_RESTRICT present) noexcept {
    assert(tps > 0.0 && m > 0);
    assert(pc != nullptr && out != nullptr && present != nullptr);
    assert(lo != nullptr && hi != nullptr);   // ct / cv may be null when no frame holds a sample
    for (std::uint32_t i = 0; i < m; ++i) {
        const std::uint32_t a = pc[lo[i] - base], e = pc[hi[i] - base];
        out[i] = 0.0;
        present[i] = 0;
        if (e == a) continue;
        if (e - a >= 2 && static_cast<__int128>(ct[e - 1]) - ct[a] > INT64_MAX) continue;
        frame::FrameBound b{};
        if (!rows_frame) b = frame::frame_bound(spec, pt[i]);
        std::int64_t rlo = (rows_frame || b.lo_inf) ? ct[a] : b.lo;
        std::int64_t rhi = (rows_frame || b.hi_inf) ? ct[e - 1] : b.hi;
        if (rhi < rlo) rhi = rlo;
        const double r = rate_one(fn, ct + a, cv + a, cst != nullptr ? cst + a : nullptr,
                                  static_cast<std::int64_t>(e - a), rlo, rhi, tps);
        const bool ok = !promql::promql_is_no_sample(r);
        present[i] = ok ? 1u : 0u;
        out[i] = ok ? r : 0.0;
    }
}

}  // namespace counter
}  // namespace kernels
}  // namespace bolt

#if defined(__clang__)
#pragma float_control(pop)
#endif
