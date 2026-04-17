// bolt/kernels/fintech/column_primitives.h — Phase 1 fintech primitives.
//
// Five stateless column operations that every Tier 1/2 fintech kernel
// composes on top of:
//
//   diff_column<T>          out[0]=0, out[i]=in[i]-in[i-1]
//   lag_column<T>           out[i<k]=0, out[i>=k]=in[i-k]
//   log_column              double-only element-wise natural log
//   sum_of_squares<T>       Σ x² (int64 acc for integral, double for fp)
//   running_max_drawdown<T> out[i] = max_so_far - in[i]   (single pass)
//
// Shape rules (Tiger Style, mirrors bolt_branchless.h + bolt_numeric.h):
//   - every function noexcept, ≥2 asserts, ≤70 lines
//   - BOLT_RESTRICT on every input/output pointer
//   - caller owns all buffers; no allocation, no growth, no state
//   - branchless inner loops where it costs nothing; auto-vectorises
//
// NOTE: Phase 1 only. The `execute_<name>(in, out, range, arena, state)`
// operator wrappers land in Phase 3+ (see docs/BOLT_FINTECH_PORT.md).

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace bolt {
namespace fintech {

// ============================================================================
// P1.1 — diff_column
// ============================================================================
// First-difference of a column. Row 0 is the conventional sentinel `0`
// (no prior observation). Rows [1..n) are `in[i] - in[i-1]`.
//
// Unlocks: LogReturns, Diff, OpenInterestDelta, OFI, every change-of-price
// kernel. Works for int64_t (price-in-ticks) and double (mid/log-prices).

template <typename T>
inline void diff_column(const T* BOLT_RESTRICT in, int64_t n,
                        T* BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    out[0] = T{0};
    for (int64_t i = 1; i < n; ++i) {
        out[i] = in[i] - in[i - 1];
    }
}

// ============================================================================
// P1.2 — lag_column
// ============================================================================
// `out[i] = (i < k) ? 0 : in[i - k]`. `k == 0` is the identity (memcpy).
// Caller must pass `0 <= k <= n` — `k == n` is a fully-zero output, which
// is a legal degenerate case (window entirely before the batch).

template <typename T>
inline void lag_column(const T* BOLT_RESTRICT in, int64_t n, int64_t k,
                       T* BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(k >= 0 && k <= n);
    for (int64_t i = 0; i < k; ++i) out[i] = T{0};
    for (int64_t i = k; i < n; ++i) out[i] = in[i - k];
}

// ============================================================================
// P1.3 — log_column
// ============================================================================
// Element-wise natural log. Non-positive inputs yield NaN / -inf per IEEE
// semantics — the kernel does NOT branch to filter them. Filtering is the
// caller's job (and can be fused via filter_sum_gt etc).
//
// Unlocks: LogReturns, Parkinson, RogersSatchell, GarmanKlass.

inline void log_column(const double* BOLT_RESTRICT in, int64_t n,
                       double* BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}

// ============================================================================
// P1.4 — sum_of_squares
// ============================================================================
// Σ x². Accumulator widened for integral T (int64_t) to match bolt_numeric's
// `filter_sum_gt` shape; floating T accumulates in `double`. Branchless:
// the compiler fuses load + mul + add into a single FMA / pmaddwd chain
// under -O2 with AVX2/NEON.
//
// Overflow note: for int64 inputs whose magnitude exceeds 2^31, the square
// overflows the accumulator. Use Kahan/Neumaier if that regime matters —
// not needed for price-tick or order-book workloads where magnitudes stay
// well under sqrt(INT64_MAX).

template <typename T>
inline auto sum_of_squares(const T* BOLT_RESTRICT in, int64_t n) noexcept
    -> std::conditional_t<std::is_floating_point_v<T>, double, int64_t> {
    assert(in != nullptr || n == 0);
    assert(n >= 0);
    using Acc = std::conditional_t<std::is_floating_point_v<T>, double, int64_t>;
    Acc a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const Acc v0 = static_cast<Acc>(in[i    ]);
        const Acc v1 = static_cast<Acc>(in[i + 1]);
        const Acc v2 = static_cast<Acc>(in[i + 2]);
        const Acc v3 = static_cast<Acc>(in[i + 3]);
        a0 += v0 * v0;
        a1 += v1 * v1;
        a2 += v2 * v2;
        a3 += v3 * v3;
    }
    Acc acc = a0 + a1 + a2 + a3;
    for (; i < n; ++i) {
        const Acc v = static_cast<Acc>(in[i]);
        acc += v * v;
    }
    return acc;
}

// ============================================================================
// P1.5 — running_max_drawdown
// ============================================================================
// Single-pass running drawdown: track the maximum value seen so far; emit
// `max_so_far - in[i]` at every index. Output is always non-negative
// (the first row emits 0 since max_so_far starts at in[0]).
//
// Works for double (price paths) and int64_t (tick-quantised prices).
// Unlocks MaxDrawdown, Drawdown; composed into CornishFisherVaR upstream.

template <typename T>
inline void running_max_drawdown(const T* BOLT_RESTRICT in, int64_t n,
                                 T* BOLT_RESTRICT out) noexcept {
    assert(in  != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    if (n == 0) return;
    T max_so_far = in[0];
    out[0] = T{0};
    for (int64_t i = 1; i < n; ++i) {
        // Branchless running max: (a > b) ? a : b compiles to cmov / csel.
        max_so_far = (in[i] > max_so_far) ? in[i] : max_so_far;
        out[i] = max_so_far - in[i];
    }
}

}  // namespace fintech
}  // namespace bolt
