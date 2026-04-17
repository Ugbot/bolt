// bolt/kernels/fintech/sorted_ring.h — Phase 5.S rolling-sorted-window primitive.
//
// SortedRing<T, kCap>: fixed-capacity rolling window that keeps the live
// samples both in insertion order (for FIFO eviction) and in ascending
// sorted order (for O(1) quantile lookups on a sorted contiguous array).
//
// Primitive shape matches RollingRing<T, kCap> in state.h — POD, trivial
// init(), push(v) per sample, no allocation. Inserts into `sorted[]` via
// std::lower_bound + memmove. Eviction locates the outgoing value the
// same way. No heap, no std::multiset, no std::priority_queue.
//
// Cost model:
//   push()     : O(window) — lower_bound (O(log w)) + memmove (O(w)).
//   quantile() : O(1)      — index into sorted[] with linear interpolation.
//   median()   : O(1)      — calls quantile(0.5).
//
// The O(w)-per-row push is the price of rolling-quantile without heap
// allocation, and is acceptable for the typical Tier-2 windows in
// Bolt's fintech set (w ≤ 512). Streaming approximations (P²/GK/t-digest)
// are strictly deferred — see design-log entry "Rolling quantile —
// SortedRing vs streaming approximations".
//
// Invariants enforced via assert:
//   - count <= window && window <= kCap
//   - sorted[0..count) is ascending on every public entry/exit
//   - raw[(pos - count) mod window .. pos) holds the live samples in
//     insertion order, used only to drive the FIFO eviction choice.
//
// No chukonu structural equivalent; chukonu per-row allocates a
// std::vector, sorts it, and throws it away. This type replaces that
// pattern in every Phase 5.S quantile/median kernel.

#pragma once

#include "bolt/bolt_port.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace fintech {

template <typename T, uint32_t kCap>
struct SortedRing {
    static_assert(kCap > 0, "SortedRing capacity must be > 0");

    T         raw[kCap];     // insertion order (FIFO; drives eviction)
    T         sorted[kCap];  // ascending sort; only [0..count) is live
    uint32_t  pos;           // next write slot in raw[]
    uint32_t  count;         // current fill, 0 <= count <= window
    uint32_t  window;        // configured window size, <= kCap

    // Zero every field and configure the window. `size` must satisfy
    // 0 < size <= kCap. Caller pins SortedRing in graph arena once.
    inline void init(uint32_t size) noexcept {
        assert(size > 0u);
        assert(size <= kCap);
        window = size;
        pos    = 0u;
        count  = 0u;
        for (uint32_t i = 0; i < kCap; ++i) { raw[i] = T{0}; sorted[i] = T{0}; }
    }

    inline bool     full()  const noexcept { return count == window; }
    inline uint32_t size()  const noexcept { return count; }

    // Append one sample. If the ring is full, the oldest sample at
    // raw[pos] is removed from sorted[] first (memmove tail forward),
    // then the new value is inserted into its sorted position
    // (memmove tail backward). If not full, just insert.
    //
    // Hot-path shape: one lower_bound + one memmove per branch. The
    // full-vs-not-full split happens once per push (not per row of an
    // outer kernel loop that walks many pushes), so the branch is not
    // a per-row branch in the Bolt sense.
    inline void push(T v) noexcept {
        assert(window > 0u);
        assert(count <= window);
        assert(pos   <  window);

        if (count < window) {
            // Not full — pure insertion.
            raw[pos] = v;
            T* BOLT_RESTRICT s = sorted;
            T* ins = std::lower_bound(s, s + count, v);
            const uint32_t ins_idx = static_cast<uint32_t>(ins - s);
            // Shift tail right by one to open the slot.
            if (ins_idx < count) {
                std::memmove(s + ins_idx + 1, s + ins_idx,
                             (count - ins_idx) * sizeof(T));
            }
            s[ins_idx] = v;
            pos = (pos + 1u) % window;
            ++count;
            return;
        }

        // Full — remove the evicted value from sorted[], then insert new.
        const T evicted = raw[pos];
        T* BOLT_RESTRICT s = sorted;
        T* ev = std::lower_bound(s, s + count, evicted);
        assert(ev != s + count);
        assert(*ev == evicted);
        const uint32_t ev_idx = static_cast<uint32_t>(ev - s);
        // Shift tail left by one to close the slot.
        if (ev_idx + 1u < count) {
            std::memmove(s + ev_idx, s + ev_idx + 1,
                         (count - ev_idx - 1u) * sizeof(T));
        }
        // count is now effectively count-1 sorted entries; insert new.
        const uint32_t n_after_evict = count - 1u;
        T* ins = std::lower_bound(s, s + n_after_evict, v);
        const uint32_t ins_idx = static_cast<uint32_t>(ins - s);
        if (ins_idx < n_after_evict) {
            std::memmove(s + ins_idx + 1, s + ins_idx,
                         (n_after_evict - ins_idx) * sizeof(T));
        }
        s[ins_idx] = v;
        raw[pos] = v;
        pos = (pos + 1u) % window;
        // count unchanged (was full, still full).
    }

    // Quantile q in [0,1] over sorted[0..count) with linear interpolation
    // between neighbours. Matches numpy's default "linear" method so the
    // tests can hand-compute against numpy formulae. Requires count > 0.
    inline double quantile(double q) const noexcept {
        assert(count > 0u);
        assert(q >= 0.0);
        assert(q <= 1.0);
        const double h = q * static_cast<double>(count - 1u);
        const double fl = std::floor(h);
        const uint32_t lo = static_cast<uint32_t>(fl);
        const uint32_t hi = (lo + 1u < count) ? (lo + 1u) : lo;
        const double frac = h - fl;
        return static_cast<double>(sorted[lo]) * (1.0 - frac)
             + static_cast<double>(sorted[hi]) *        frac;
    }

    // Median over the live window. If count is odd this is the middle
    // element; if even this is the mean of the two middle elements
    // (matches chukonu's CreateOutlierFlagMADProcessor convention and
    // numpy.median).
    inline double median() const noexcept {
        assert(count > 0u);
        if ((count & 1u) == 1u) {
            return static_cast<double>(sorted[count >> 1]);
        }
        const uint32_t hi = count >> 1;
        const uint32_t lo = hi - 1u;
        return 0.5 * (static_cast<double>(sorted[lo])
                    + static_cast<double>(sorted[hi]));
    }

    // Write the first n_out entries of sorted[] into `out`. Used by
    // the MAD kernel when it needs to scan the window to compute
    // absolute deviations from the median.
    inline void copy_sorted(T* BOLT_RESTRICT out, uint32_t n_out) const noexcept {
        assert(out   != nullptr || n_out == 0u);
        assert(n_out <= count);
        for (uint32_t i = 0; i < n_out; ++i) out[i] = sorted[i];
    }

    // Access the k-th order statistic (0-indexed into sorted[]).
    // Used by HistoricalVaR / HistoricalCVaR which need sorted[var_index]
    // directly, not an interpolated quantile.
    inline T order(uint32_t k) const noexcept {
        assert(count > 0u);
        assert(k < count);
        return sorted[k];
    }
};

}  // namespace fintech
}  // namespace bolt
