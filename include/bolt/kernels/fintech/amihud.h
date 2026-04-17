// bolt/kernels/fintech/amihud.h — Phase 5.S.10 Tier 2 kernel.
//
// Amihud illiquidity: |return| / volume, emitted per-row (no rolling window
// in chukonu). chukonu's CreateAmihudProcessor (line 639) simply computes
//   amihud[i] = (v > 0) ? |r[i]| / v[i] : 0.0
// with no warmup — scalar pass. We match that exactly.
//
// Note: the plan doc in BOLT_FINTECH_PORT.md describes an "Amihud rolling
// mean" flavour with `P2.1` dependency. We follow the chukonu source of
// truth (per-row, no ring) to keep numerical parity with chukonu tests.
// A rolling-mean flavour is a trivial wrapper (SMA on top of Amihud) and
// is not reproduced here.
//
// Branchless note: the `v > 0` guard is expressed as a cmov-style ternary.
// The ternary `(v > 0.0) ? ... : 0.0` compiles to a cmov on x86 — no
// data-dependent control-flow branch in the per-row loop. `std::fabs` is
// also branchless on IEEE doubles (bit-mask of the sign bit on x86).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

struct AmihudOpDesc {
    int32_t return_col;
    int32_t volume_col;
    int32_t out_col;
};

// POD state — pinned in arena at graph-compile time. No per-row state.
struct AmihudState {
    AmihudOpDesc desc;
};

// Execute Amihud per-row. `in[0]` is the return series, `in[1]` is the
// volume series — the AmihudOpDesc column indices track caller metadata
// only; we take raw typed pointers.
inline void execute_amihud(AmihudState* state,
                           const double* BOLT_RESTRICT r,
                           const double* BOLT_RESTRICT v,
                           int64_t n,
                           double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(r   != nullptr || n == 0);
    assert(v   != nullptr || n == 0);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    (void)state;  // stateless in Release builds (assert compiles out).
    // Branchless per-row body: cmov on the `v > 0` guard. fabs is a bit-
    // mask on x86, so no branch there either. The `vi > 0.0 ? vi : 1.0`
    // dodge avoids a spurious MSVC C4723 (potential divide by zero) —
    // when the guard would select 0.0 the divisor is 1.0, then the mask
    // cmov picks the 0.0 branch anyway. Loop auto-vectorises cleanly.
    for (int64_t i = 0; i < n; ++i) {
        const double vi     = v[i];
        const double ri     = std::fabs(r[i]);
        const bool   has_v  = (vi > 0.0);
        const double denom  = has_v ? vi : 1.0;
        out[i]              = has_v ? (ri / denom) : 0.0;
    }
}

inline AmihudState* make_amihud_state(::bolt::Arena* arena,
                                      int32_t return_col,
                                      int32_t volume_col,
                                      int32_t out_col) noexcept {
    assert(arena != nullptr);
    assert(return_col >= 0);
    AmihudState* s = arena->allocate_array<AmihudState>(1);
    if (s == nullptr) return nullptr;
    s->desc.return_col = return_col;
    s->desc.volume_col = volume_col;
    s->desc.out_col    = out_col;
    return s;
}

}  // namespace fintech
}  // namespace bolt
