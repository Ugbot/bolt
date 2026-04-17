// bolt/kernels/fintech/monotonic_deque.h — Phase 5.R shared primitive.
//
// IndexRing<kCap> — a fixed-capacity monotonic deque over (index, value)
// pairs. Used by RollingMin / RollingMax / DonchianChannel / StochasticOsc
// to compute sliding-window extrema in O(1) amortised.
//
// Bolt has no std::deque available (Tiger Style bans it). This is the
// arena-pinned POD replacement: a contiguous ring of kCap slots with
// independent head/tail indices. Capacity must be >= window size.
//
// Invariants (maintained by push_max / push_min):
//   - Indices stored in the deque are strictly increasing head→tail.
//   - Values stored are strictly monotonic (decreasing for max, increasing
//     for min). The front always holds the extremum of the current window.
//
// Shape rules: POD layout, noexcept everywhere, >=2 asserts per method,
// functions <= 70 lines. No allocation. No exceptions.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace fintech {

// Upper bound on IndexRing capacity — matches kMaxRollingRingCap (state.h).
// We redeclare locally so this header is usable without pulling in state.h.
inline constexpr uint32_t kMaxIndexRingCap = 65536u;

// IndexRing<kCap>
// Fixed ring of (index, value) pairs. The deque occupies slots
// [head, tail) modulo kCap; count tracks live entries. Caller picks kCap
// >= window size; the ring never grows.
template <uint32_t kCap>
struct IndexRing {
    static_assert(kCap > 0, "IndexRing capacity must be > 0");
    static_assert(kCap <= kMaxIndexRingCap,
                  "IndexRing capacity exceeds kMaxIndexRingCap");

    int64_t  idx_buf[kCap];
    double   val_buf[kCap];
    uint32_t head;   // front of deque
    uint32_t tail;   // one-past-back of deque
    uint32_t count;  // live entries in [0, kCap]

    inline void init() noexcept {
        head = 0u;
        tail = 0u;
        count = 0u;
        // Zero the backing arrays so stale bits don't confuse debuggers.
        for (uint32_t i = 0; i < kCap; ++i) { idx_buf[i] = 0; val_buf[i] = 0.0; }
    }

    inline bool empty() const noexcept { return count == 0u; }
    inline uint32_t size() const noexcept { return count; }

    inline int64_t front_idx() const noexcept {
        assert(count > 0u);
        assert(head < kCap);
        return idx_buf[head];
    }
    inline double front_val() const noexcept {
        assert(count > 0u);
        assert(head < kCap);
        return val_buf[head];
    }

    // Drop entries from the front whose index is stale (< min_keep_idx).
    // Call exactly once per advance, before push_max / push_min.
    inline void pop_front_while_stale(int64_t min_keep_idx) noexcept {
        assert(count <= kCap);
        assert(head < kCap || count == 0u);
        while (count > 0u && idx_buf[head] < min_keep_idx) {
            head = (head + 1u) % kCap;
            --count;
        }
    }

    // Push (i, v) enforcing a monotonic-decreasing values invariant
    // (running window maximum). Drops back entries with val <= v, then
    // appends (i, v) at the tail.
    inline void push_max(int64_t i, double v) noexcept {
        assert(count <= kCap);
        assert(kCap > 0u);
        while (count > 0u) {
            const uint32_t back = (tail + kCap - 1u) % kCap;
            if (val_buf[back] > v) break;
            tail = back;
            --count;
        }
        assert(count < kCap);
        idx_buf[tail] = i;
        val_buf[tail] = v;
        tail = (tail + 1u) % kCap;
        ++count;
    }

    // Push (i, v) enforcing a monotonic-increasing values invariant
    // (running window minimum). Drops back entries with val >= v.
    inline void push_min(int64_t i, double v) noexcept {
        assert(count <= kCap);
        assert(kCap > 0u);
        while (count > 0u) {
            const uint32_t back = (tail + kCap - 1u) % kCap;
            if (val_buf[back] < v) break;
            tail = back;
            --count;
        }
        assert(count < kCap);
        idx_buf[tail] = i;
        val_buf[tail] = v;
        tail = (tail + 1u) % kCap;
        ++count;
    }
};

}  // namespace fintech
}  // namespace bolt
