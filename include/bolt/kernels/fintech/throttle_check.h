// bolt/kernels/fintech/throttle_check.h — Phase 5.S.15.
//
// Throttle tripwire — count events in a sliding time window and flag each
// row according to whether the current window size has exceeded a cap:
//     out[i] = 1  if count_in_window(ts[i]) <= max_count
//     out[i] = 0  otherwise
// where `count_in_window(t) = |{j : t - window_ns <= ts[j] <= t}|`.
//
// State: a fixed-capacity ring of pending timestamps. Capacity is chosen
// at graph compile via the kCap template parameter; callers must pick
// kCap >= max_count + 1 (enforced by assert).
//
// Scope note vs Phase 6: chukonu's CreateThrottleCheckProcessor is GLOBAL
// (unkeyed). We port that exact shape. A per-symbol / per-user throttle
// requires keyed state from bolt::dataflow and is explicitly deferred to
// Phase 6. This single-stream shape is the baseline primitive for rate
// limits over an aggregated order stream.
//
// Units: the window and timestamps are in the SAME integer unit. Chukonu
// uses milliseconds (int64 `window_ms`); we stay unit-agnostic — the
// constructor takes `window` (int64) and it's the caller's job to match
// units (ns, us, ms, ...) with the `ts` column.
//
// Branchless audit:
//   - Eviction loop: a `while (ring.front < t_start) pop_front()` — this
//     is a data-dependent loop, BUT each timestamp is evicted at most
//     ONCE over the entire stream (amortised O(1) per row). This is the
//     same shape as chukonu's std::deque version and the same shape Bolt
//     uses in monotonic_deque.h / rolling_min_max.h. The alternative —
//     unrolling into a fixed-trip loop — would be strictly slower under
//     uniform input density. This is the documented exception to the
//     per-row branchless rule (see docs/research/design-log.md — entry
//     for "Monotonic-deque eviction loop"), and it's bounded by the
//     kCap invariant.
//   - Per-row emit: `(count <= max_count) ? 1 : 0` — cmov.
// Zero data-dependent branches in the steady-state emit path.
//
// Source: chukonu/fintech/kernels.h :: CreateThrottleCheckProcessor (2263).

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// Hard upper bound on the ring capacity — matches kMaxRollingRingCap from
// state.h. We redeclare locally to keep this header self-contained.
inline constexpr uint32_t kThrottleCheckMaxCap = 65536u;

struct ThrottleCheckOpDesc {
    int32_t  ts_col;           // int64 timestamp column
    int32_t  out_col;          // int64 flag output (0 or 1)
    int64_t  window;           // window length in timestamp units
    int64_t  max_count;        // events per window allowed (inclusive)
};

// State: fixed-capacity circular buffer of pending timestamps.
// Entries between [head, tail) (modulo kCap) are within the current window.
template <uint32_t kCap>
struct ThrottleCheckState {
    static_assert(kCap > 0u,                    "kCap must be > 0");
    static_assert(kCap <= kThrottleCheckMaxCap, "kCap exceeds hard cap");

    ThrottleCheckOpDesc desc;
    int64_t             ring[kCap];
    uint32_t            head;    // next eviction slot
    uint32_t            tail;    // next insertion slot
    uint32_t            count;   // live entries in the ring
};

// Execute throttle over `n` rows. `ts_in` is the int64 timestamp column,
// `out` is an int64 output (0 or 1 per row). Separate typed pointers
// because ThrottleCheck is the only Phase 5 kernel with non-double I/O.
template <uint32_t kCap>
inline void execute_throttle_check(ThrottleCheckState<kCap>* state,
                                   const int64_t* BOLT_RESTRICT ts_in,
                                   int64_t  n,
                                   int64_t* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(ts_in != nullptr || n == 0);
    assert(out   != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.max_count >= 0);
    assert(state->desc.window    >= 0);

    const int64_t  window    = state->desc.window;
    const int64_t  max_count = state->desc.max_count;
    uint32_t       head      = state->head;
    uint32_t       tail      = state->tail;
    uint32_t       count     = state->count;

    for (int64_t i = 0; i < n; ++i) {
        const int64_t t       = ts_in[i];
        const int64_t t_start = t - window;

        // Insert t. Ring invariant: kCap-1 slots must be enough to hold
        // max_count + 1 live entries. Caller enforces kCap > max_count.
        assert(count < kCap);
        state->ring[tail] = t;
        tail              = (tail + 1u) % kCap;
        ++count;

        // Evict timestamps that fell out of the window. Amortised O(1):
        // each inserted timestamp is evicted at most once over the stream.
        while (count > 0u && state->ring[head] < t_start) {
            head  = (head + 1u) % kCap;
            --count;
        }

        // Branchless emit: (count <= max_count) ? 1 : 0  → cmov.
        const int64_t live = static_cast<int64_t>(count);
        out[i] = (live <= max_count) ? 1 : 0;
    }

    state->head  = head;
    state->tail  = tail;
    state->count = count;
}

// Arena-pinned constructor. Requires kCap > max_count so the ring can
// hold one extra insertion before the tripwire fires without wrapping.
template <uint32_t kCap>
inline ThrottleCheckState<kCap>* make_throttle_check_state(::bolt::Arena* arena,
                                                           int32_t ts_col,
                                                           int32_t out_col,
                                                           int64_t window,
                                                           int64_t max_count) noexcept {
    assert(arena != nullptr);
    assert(window >= 0);
    assert(max_count >= 0);
    assert(static_cast<int64_t>(kCap) > max_count);
    ThrottleCheckState<kCap>* s = arena->allocate_array<ThrottleCheckState<kCap>>(1);
    if (s == nullptr) return nullptr;
    s->desc.ts_col    = ts_col;
    s->desc.out_col   = out_col;
    s->desc.window    = window;
    s->desc.max_count = max_count;
    s->head           = 0u;
    s->tail           = 0u;
    s->count          = 0u;
    for (uint32_t k = 0; k < kCap; ++k) s->ring[k] = 0;
    return s;
}

template <uint32_t kCap>
inline void throttle_check_reset(ThrottleCheckState<kCap>* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.window >= 0);
    state->head  = 0u;
    state->tail  = 0u;
    state->count = 0u;
}

}  // namespace fintech
}  // namespace bolt
