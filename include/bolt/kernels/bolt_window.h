// bolt/kernels/bolt_window.h — Window function ranking kernels.
//
// SQL window functions over a sequence ALREADY partitioned by
// `partition_keys` and ORDER BY-sorted within each partition. The
// kernel's job is to walk the sorted sequence once and emit ranks /
// offsets, detecting partition breaks by key change.
//
// Design (Tiger Style):
//   - header-only, noexcept, no exceptions, no RTTI
//   - ≥2 asserts on every entry point
//   - functions ≤70 lines
//   - BOLT_RESTRICT on all in/out pointers
//   - branchless inner bodies where the running-state dependency permits
//   - compile-time type dispatch (templates over Key / OrderKey / T)
//
// SIMD note: ROW_NUMBER / RANK / DENSE_RANK have a true scalar
// dependency on the running counter (output[i] depends on output[i-1]
// across a partition). The partition-boundary compare is vectorizable
// but the running counter is not; we keep the loop scalar +
// BOLT_FORCE_INLINE and let the compiler auto-vectorize the boundary
// compares it can. LAG / LEAD / FIRST_VALUE / LAST_VALUE / NTILE are
// memory-bound, scan-shaped, and auto-vectorize cleanly.
//
// The caller owns out[] and guarantees n ≤ capacity.

#pragma once

#include "bolt/bolt_port.h"

#include <cstdint>
#include <cassert>
#include <cstring>

namespace bolt {
namespace kernels {
namespace window {

// ============================================================================
// ROW_NUMBER() — 1, 2, 3, ... within each partition.
// ============================================================================
template <typename Key>
BOLT_FORCE_INLINE void row_number(
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    int32_t*   BOLT_RESTRICT out) noexcept
{
    assert(n >= 0);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    int32_t counter = 0;
    Key prev{};                          // value-init; first iter forces reset
    for (int64_t i = 0; i < n; ++i) {
        const Key k = partition_keys[i];
        const bool same = (i != 0) & (k == prev);
        counter = same ? (counter + 1) : 1;
        out[i] = counter;
        prev = k;
    }
}

// ============================================================================
// RANK() — 1, 2, 2, 4 (gaps after ties on order_keys).
// ============================================================================
template <typename Key, typename OrderKey>
BOLT_FORCE_INLINE void rank(
    const Key*      BOLT_RESTRICT partition_keys,
    const OrderKey* BOLT_RESTRICT order_keys,
    int64_t                       n,
    int32_t*  BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || order_keys != nullptr);
    assert(n == 0 || out != nullptr);

    int32_t row_idx = 0;                 // 1-based position within partition
    int32_t cur_rank = 0;                // rank to emit for current tie-run
    Key prev_key{};
    OrderKey prev_order{};
    for (int64_t i = 0; i < n; ++i) {
        const Key k = partition_keys[i];
        const OrderKey o = order_keys[i];
        const bool new_part = (i == 0) | (k != prev_key);
        if (new_part) {
            row_idx = 1;
            cur_rank = 1;
        } else {
            ++row_idx;
            // tie => keep cur_rank; otherwise jump to row_idx
            cur_rank = (o == prev_order) ? cur_rank : row_idx;
        }
        out[i] = cur_rank;
        prev_key = k;
        prev_order = o;
    }
}

// ============================================================================
// DENSE_RANK() — 1, 2, 2, 3 (no gaps on ties).
// ============================================================================
template <typename Key, typename OrderKey>
BOLT_FORCE_INLINE void dense_rank(
    const Key*      BOLT_RESTRICT partition_keys,
    const OrderKey* BOLT_RESTRICT order_keys,
    int64_t                       n,
    int32_t*  BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || order_keys != nullptr);
    assert(n == 0 || out != nullptr);

    int32_t cur = 0;
    Key prev_key{};
    OrderKey prev_order{};
    for (int64_t i = 0; i < n; ++i) {
        const Key k = partition_keys[i];
        const OrderKey o = order_keys[i];
        const bool new_part = (i == 0) | (k != prev_key);
        if (new_part) {
            cur = 1;
        } else {
            cur += (o != prev_order) ? 1 : 0;
        }
        out[i] = cur;
        prev_key = k;
        prev_order = o;
    }
}

// ============================================================================
// LAG(value, k, default) — value k rows back, default if before partition start.
// ============================================================================
template <typename T, typename Key>
BOLT_FORCE_INLINE void lag(
    const T*   BOLT_RESTRICT values,
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    int32_t                  k,
    T                        default_value,
    T*   BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(k >= 0);
    assert(n == 0 || values != nullptr);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    const int64_t kk = static_cast<int64_t>(k);
    // Peel: first kk rows have no in-range source; emit default.
    const int64_t head = kk < n ? kk : n;
    for (int64_t i = 0; i < head; ++i) out[i] = default_value;
    // Body: src >= 0 always; only the partition-key compare gates emission.
    for (int64_t i = head; i < n; ++i) {
        const int64_t src = i - kk;
        const Key kp = partition_keys[src];
        const Key ki = partition_keys[i];
        const T   vs = values[src];
        out[i] = (kp == ki) ? vs : default_value;
    }
}

// ============================================================================
// LEAD(value, k, default) — value k rows forward, default if past partition end.
// ============================================================================
template <typename T, typename Key>
BOLT_FORCE_INLINE void lead(
    const T*   BOLT_RESTRICT values,
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    int32_t                  k,
    T                        default_value,
    T*   BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(k >= 0);
    assert(n == 0 || values != nullptr);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    const int64_t kk = static_cast<int64_t>(k);
    const int64_t body_end = (n >= kk) ? (n - kk) : 0;
    // Body: src = i + kk < n always; only the key compare gates emission.
    for (int64_t i = 0; i < body_end; ++i) {
        const int64_t src = i + kk;
        const Key kp = partition_keys[src];
        const Key ki = partition_keys[i];
        const T   vs = values[src];
        out[i] = (kp == ki) ? vs : default_value;
    }
    // Tail: no in-range source; emit default.
    for (int64_t i = body_end; i < n; ++i) out[i] = default_value;
}

// ============================================================================
// FIRST_VALUE — broadcast values[partition_start] across the partition.
// ============================================================================
template <typename T, typename Key>
BOLT_FORCE_INLINE void first_value(
    const T*   BOLT_RESTRICT values,
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    T*   BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(n == 0 || values != nullptr);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    T cur{};
    Key prev{};
    for (int64_t i = 0; i < n; ++i) {
        const Key k = partition_keys[i];
        const bool new_part = (i == 0) | (k != prev);
        if (new_part) {
            cur = values[i];
        }
        out[i] = cur;
        prev = k;
    }
}

// ============================================================================
// LAST_VALUE — broadcast values[partition_end] across the partition.
// Two-pass: forward to find each partition's right edge, then a backward
// fill. The backward fill is the read-only hot loop.
// ============================================================================
template <typename T, typename Key>
BOLT_FORCE_INLINE void last_value(
    const T*   BOLT_RESTRICT values,
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    T*   BOLT_RESTRICT       out) noexcept
{
    assert(n >= 0);
    assert(n == 0 || values != nullptr);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    if (n == 0) return;

    // Walk backwards; on each partition's *last* row (i.e. when next is a
    // different key, or i == n-1), latch values[i]; broadcast leftward.
    T cur = values[n - 1];
    Key next_key = partition_keys[n - 1];
    out[n - 1] = cur;
    for (int64_t i = n - 2; i >= 0; --i) {
        const Key k = partition_keys[i];
        if (k != next_key) {
            cur = values[i];
        }
        out[i] = cur;
        next_key = k;
    }
}

// ============================================================================
// NTILE(n_tiles) — assign each row to one of n_tiles buckets, within partition.
//
// Two-pass per-partition: scan forward to size the partition, then emit
// bucket ids. Bucket sizes: first (size % n_tiles) buckets get ceil(size /
// n_tiles), the rest get floor(size / n_tiles). 1-based bucket ids.
// ============================================================================
template <typename Key>
BOLT_FORCE_INLINE void ntile(
    const Key* BOLT_RESTRICT partition_keys,
    int64_t                  n,
    int32_t                  n_tiles,
    int32_t*   BOLT_RESTRICT out) noexcept
{
    assert(n >= 0);
    assert(n_tiles > 0);
    assert(n == 0 || partition_keys != nullptr);
    assert(n == 0 || out != nullptr);

    int64_t i = 0;
    while (i < n) {
        // find partition extent [i, j)
        int64_t j = i + 1;
        const Key k0 = partition_keys[i];
        while (j < n && partition_keys[j] == k0) ++j;
        const int64_t size = j - i;
        const int64_t base = size / n_tiles;
        const int64_t extra = size % n_tiles;
        // first `extra` buckets have (base+1) rows; remaining have base rows.
        // Cumulative boundary for bucket b (1-based, b in [1..n_tiles]):
        //   end(b) = b*base + min(b, extra)
        int64_t pos = 0;          // index within partition
        int32_t bucket = 1;
        int64_t end = base + (extra > 0 ? 1 : 0);
        for (int64_t r = i; r < j; ++r) {
            while (pos >= end && bucket < n_tiles) {
                ++bucket;
                end += base + (bucket <= extra ? 1 : 0);
            }
            out[r] = bucket;
            ++pos;
        }
        i = j;
    }
}

// ============================================================================
// NTH_VALUE(value, nth [, ignore_nulls]) over a frame that starts at UNBOUNDED
// PRECEDING (the partition's first row), within partition.
//
//   frame end  kPartitionEnd : UNBOUNDED FOLLOWING (or no ORDER BY)
//              kCurrentRow   : ROWS .. CURRENT ROW
//              kPeerEnd      : RANGE .. CURRENT ROW; the end is the last row
//                              whose frame_keys equal the current row's
//                              (frame_keys = one id per (partition, peer run))
//
// Row r gets the nth row (1-based; nth non-null row with ignore_nulls) of its
// frame, or NULL (out_null = 1) when the frame holds fewer than nth rows. A
// NULL value at that position is NULL as well. Linear: one scan per partition
// to locate the nth position, one backward pass to bound each frame.
// ============================================================================
enum class NthFrameEnd : uint8_t { kPartitionEnd = 0, kCurrentRow = 1, kPeerEnd = 2 };

template <typename V, typename Key>
BOLT_FORCE_INLINE void nth_value(
    const V*       BOLT_RESTRICT values,
    const uint8_t* BOLT_RESTRICT value_null,      // nullable: 1 = NULL value
    const Key*     BOLT_RESTRICT partition_keys,
    const Key*     BOLT_RESTRICT frame_keys,      // only read for kPeerEnd
    int64_t                      n,
    int32_t                      nth,
    NthFrameEnd                  frame_end,
    bool                         ignore_nulls,
    V*             BOLT_RESTRICT out,
    uint8_t*       BOLT_RESTRICT out_null) noexcept
{
    assert(n >= 0);
    assert(nth >= 1);
    assert(n == 0 || (values != nullptr && partition_keys != nullptr));
    assert(n == 0 || (out != nullptr && out_null != nullptr));
    assert(frame_end != NthFrameEnd::kPeerEnd || n == 0 || frame_keys != nullptr);

    int64_t i = 0;
    while (i < n) {
        int64_t j = i + 1;
        const Key k0 = partition_keys[i];
        while (j < n && partition_keys[j] == k0) ++j;
        // p = the partition row that answers for every row whose frame reaches
        // it; j (one past the partition) when the partition has no such row.
        int64_t p = j;
        if (!ignore_nulls) {
            const int64_t at = i + static_cast<int64_t>(nth) - 1;
            p = at < j ? at : j;
        } else {
            int64_t seen = 0;
            for (int64_t r = i; r < j; ++r) {
                const bool is_null = value_null != nullptr && value_null[r] != 0;
                seen += is_null ? 0 : 1;
                if (seen == nth) { p = r; break; }
            }
        }
        const bool p_null = p < j && value_null != nullptr && value_null[p] != 0;
        const V pv = p < j ? values[p] : V{};
        int64_t end = j - 1;
        for (int64_t r = j - 1; r >= i; --r) {
            if (frame_end == NthFrameEnd::kCurrentRow) end = r;
            else if (frame_end == NthFrameEnd::kPeerEnd) {
                end = (r == j - 1 || frame_keys[r] != frame_keys[r + 1]) ? r : end;
            }
            const bool present = p <= end;
            out[r] = present ? pv : V{};
            out_null[r] = static_cast<uint8_t>((!present) | (present & p_null));
        }
        i = j;
    }
}

}  // namespace window
}  // namespace kernels
}  // namespace bolt
