// bolt/kernels/fintech/rolling_min_max.h — Phase 5.R.6 Tier 2 kernels.
//
// Rolling min / rolling max over a fixed window. Two kernels, one header —
// they share the IndexRing monotonic-deque primitive.
//
//   out[i] = NaN                      if seen < window     (warm-up)
//   out[i] = min(x[i-w+1..i])         for RollingMin       (otherwise)
//   out[i] = max(x[i-w+1..i])         for RollingMax       (otherwise)
//
// Source: chukonu/fintech/kernels.h ::
//   CreateRollingMinProcessor (3395), CreateRollingMaxProcessor (3438).
// Port shape: docs/research/gestalt-kernel-adapter.md (Tier 2 stateful).
//
// Both are O(1) amortised via IndexRing<kCap>, not O(w) per row like a
// naive scan. Deviation from chukonu: Bolt emits NaN during warm-up;
// chukonu emits partial-window extrema.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/fintech/monotonic_deque.h"

#include <cassert>
#include <cstdint>
#include <limits>

namespace bolt {
namespace fintech {

// ------------------------------------------------------------------
// RollingMin
// ------------------------------------------------------------------
template <uint32_t kCap>
struct RollingMinState {
    int32_t             in_col;
    int32_t             out_col;
    uint32_t            window;
    int64_t             row_counter;
    IndexRing<kCap>     dq_min;

    inline void init(int32_t in, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        in_col      = in;
        out_col     = out;
        window      = w;
        row_counter = 0;
        dq_min.init();
    }
};

template <uint32_t kCap>
inline void execute_rolling_min(RollingMinState<kCap>* state,
                                const double* BOLT_RESTRICT* in,
                                int64_t n,
                                double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT x = in[0];
    assert(x != nullptr || n == 0);
    const int64_t w = static_cast<int64_t>(state->window);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t i = state->row_counter;
        state->dq_min.pop_front_while_stale(i - w + 1);
        state->dq_min.push_min(i, x[k]);
        out[k] = (i + 1 < w) ? nan : state->dq_min.front_val();
        ++state->row_counter;
    }
}

// ------------------------------------------------------------------
// RollingMax
// ------------------------------------------------------------------
template <uint32_t kCap>
struct RollingMaxState {
    int32_t             in_col;
    int32_t             out_col;
    uint32_t            window;
    int64_t             row_counter;
    IndexRing<kCap>     dq_max;

    inline void init(int32_t in, int32_t out, uint32_t w) noexcept {
        assert(w > 0u);
        assert(w <= kCap);
        in_col      = in;
        out_col     = out;
        window      = w;
        row_counter = 0;
        dq_max.init();
    }
};

template <uint32_t kCap>
inline void execute_rolling_max(RollingMaxState<kCap>* state,
                                const double* BOLT_RESTRICT* in,
                                int64_t n,
                                double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(in    != nullptr);
    assert(n >= 0);
    assert(out != nullptr || n == 0);
    const double* BOLT_RESTRICT x = in[0];
    assert(x != nullptr || n == 0);
    const int64_t w = static_cast<int64_t>(state->window);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    assert(w > 0);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t i = state->row_counter;
        state->dq_max.pop_front_while_stale(i - w + 1);
        state->dq_max.push_max(i, x[k]);
        out[k] = (i + 1 < w) ? nan : state->dq_max.front_val();
        ++state->row_counter;
    }
}

}  // namespace fintech
}  // namespace bolt
