// bolt/kernels/fintech/cusum.h — Phase 5.S.3.
//
// Two-sided CUSUM (Page 1954) — cumulative sum change-point detector with
// upper/lower accumulators that self-reset when they cross a threshold:
//     d[i]     = x[i] - target
//     cp[i]    = max(0.0, cp[i-1] + d[i])       ("upper" CUSUM)
//     cn[i]    = min(0.0, cn[i-1] + d[i])       ("lower" CUSUM)
//     if cp[i] >  threshold OR cn[i] < -threshold:
//         emit (cp[i], cn[i]); then reset both to 0 for the NEXT row.
//     else:
//         emit (cp[i], cn[i]); carry cp/cn to next row.
//
// The plan text in BOLT_FINTECH_PORT.md P5.S.3 specifies the two-sided
// form above. Chukonu's CreateCUSUMProcessor (line 3540) ships a ONE-sided
// variant that also estimates the baseline mean from a warmup window — we
// do not embed warmup statistics, the target (baseline) is caller-supplied.
// This is the Bolt convention: state is pure, statistics computed upstream.
//
// Output shape: TWO output columns (cp_out, cn_out). The "alarm" flag in
// chukonu's shape is trivially derivable downstream as
// `(cp > threshold) | (cn < -threshold)` via filter_gt — we do not emit
// a separate boolean column here.
//
// Branchless audit:
//   - cp = max(0, cp + d)  → `(z > 0) ? z : 0`  → cmov
//   - cn = min(0, cn + d)  → `(z < 0) ? z : 0`  → cmov
//   - reset after emit     → `tripped = (cp > h) | (cn < -h)` is a bool;
//                             `cp = tripped ? 0 : cp` (cmov),
//                             `cn = tripped ? 0 : cn` (cmov).
// No data-dependent branches in the per-row loop.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

struct CUSUMOpDesc {
    int32_t in_col;
    int32_t cp_col;   // "cumulative positive" output
    int32_t cn_col;   // "cumulative negative" output
};

struct CUSUMState {
    CUSUMOpDesc desc;
    double      target;      // baseline to subtract per-row (k or mu)
    double      threshold;   // symmetric trip threshold (> 0)
    double      cum_pos;     // running upper CUSUM
    double      cum_neg;     // running lower CUSUM
};

// Execute CUSUM over `n` rows. Emits two output streams.
// Contract:
//   - state != nullptr; threshold > 0 (asserted).
//   - in != nullptr; in[desc.in_col] holds n doubles.
//   - cp_out, cn_out both point to buffers of capacity >= n (unless n == 0).
inline void execute_cusum(CUSUMState* state,
                          const double* BOLT_RESTRICT* in,
                          int64_t n,
                          double* BOLT_RESTRICT cp_out,
                          double* BOLT_RESTRICT cn_out) noexcept {
    assert(state != nullptr);
    assert(in != nullptr);
    assert(n >= 0);
    assert(cp_out != nullptr || n == 0);
    assert(cn_out != nullptr || n == 0);
    assert(state->threshold > 0.0);

    const double* BOLT_RESTRICT v = in[state->desc.in_col];
    assert(v != nullptr || n == 0);

    const double target = state->target;
    const double h      = state->threshold;
    double       cp     = state->cum_pos;
    double       cn     = state->cum_neg;

    for (int64_t i = 0; i < n; ++i) {
        const double d   = v[i] - target;
        const double zp  = cp + d;
        const double zn  = cn + d;
        // Branchless max(0, zp) and min(0, zn) — compile to cmov.
        cp               = (zp > 0.0) ? zp : 0.0;
        cn               = (zn < 0.0) ? zn : 0.0;
        // Emit pre-reset values so the caller sees the threshold crossing.
        cp_out[i]        = cp;
        cn_out[i]        = cn;
        // Branchless reset: if either side tripped, zero both accumulators.
        const bool trip  = (cp > h) || (cn < -h);
        cp               = trip ? 0.0 : cp;
        cn               = trip ? 0.0 : cn;
    }

    state->cum_pos = cp;
    state->cum_neg = cn;
}

inline CUSUMState* make_cusum_state(::bolt::Arena* arena,
                                    int32_t in_col,
                                    int32_t cp_col,
                                    int32_t cn_col,
                                    double  target,
                                    double  threshold) noexcept {
    assert(arena != nullptr);
    assert(threshold > 0.0);
    CUSUMState* s = arena->allocate_array<CUSUMState>(1);
    if (s == nullptr) return nullptr;
    s->desc.in_col = in_col;
    s->desc.cp_col = cp_col;
    s->desc.cn_col = cn_col;
    s->target      = target;
    s->threshold   = threshold;
    s->cum_pos     = 0.0;
    s->cum_neg     = 0.0;
    return s;
}

inline void cusum_reset(CUSUMState* state) noexcept {
    assert(state != nullptr);
    assert(state->threshold > 0.0);
    state->cum_pos = 0.0;
    state->cum_neg = 0.0;
}

}  // namespace fintech
}  // namespace bolt
