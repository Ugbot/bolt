// bolt_sorted_count.h — row counts of sorted runs by position, without reading the rows (G2CHK-1198).
//
// A run of rows sorted ascending by a key column holds, for any value range of that column, one contiguous
// interval of positions; its row count is the difference of two positions found by galloping + bisection, so
// counting rows by key range or by time bucket reads O(buckets * log(run)) cells instead of every row.
//
//   gallop_lower_bound / gallop_upper_bound   first position in [lo, hi) with d[pos] >= key (> key); cost
//                                             O(log(pos - lo)): cheap when the answer is near `lo`, bounded by
//                                             O(log(hi - lo)) when it is not.
//   bucket_of_i64                             floor((v - origin) / width) without 128-bit arithmetic.
//   count_buckets_i64                         per-bucket row counts of d[lo, hi) (ascending): one (bucket,
//                                             count) pair per non-empty bucket, empty buckets absent.
//
// Everything is branch-light scalar code over int64 arrays; no allocation; callers own the output arrays.
#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace sorted {

// `d` is anything indexable by int64 (a pointer, or an accessor over paged storage: `d[i]` yields the i-th value).
template <typename T, typename A>
BOLT_FORCE_INLINE int64_t lower_bound_range(const A& d, int64_t lo, int64_t hi, T key) noexcept {
    assert(lo >= 0);
    while (lo < hi) {                                   // bounded: log2(hi - lo)
        const int64_t mid = lo + ((hi - lo) >> 1);
        if (d[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

template <typename T, typename A>
BOLT_FORCE_INLINE int64_t upper_bound_range(const A& d, int64_t lo, int64_t hi, T key) noexcept {
    assert(lo >= 0);
    while (lo < hi) {                                   // bounded: log2(hi - lo)
        const int64_t mid = lo + ((hi - lo) >> 1);
        if (d[mid] <= key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// First position in [lo, hi) with d[pos] >= key (hi when none): doubles a step from `lo`, then bisects the last
// interval, so a position near `lo` costs about 2 log2(distance) probes.
template <typename T, typename A>
BOLT_FORCE_INLINE int64_t gallop_lower_bound(const A& d, int64_t lo, int64_t hi, T key) noexcept {
    assert(lo >= 0 && lo <= hi);
    if (lo >= hi || !(d[lo] < key)) return lo;
    int64_t step = 1, prev = lo, cur = lo + 1;
    while (cur < hi && d[cur] < key) {                  // bounded: log2(hi - lo)
        prev = cur;
        step <<= 1;
        cur = (hi - cur > step) ? cur + step : hi;
    }
    return lower_bound_range<T, A>(d, prev + 1, cur < hi ? cur + 1 : hi, key);
}

template <typename T, typename A>
BOLT_FORCE_INLINE int64_t gallop_upper_bound(const A& d, int64_t lo, int64_t hi, T key) noexcept {
    assert(lo >= 0 && lo <= hi);
    if (lo >= hi || d[lo] > key) return lo;
    int64_t step = 1, prev = lo, cur = lo + 1;
    while (cur < hi && d[cur] <= key) {                 // bounded: log2(hi - lo)
        prev = cur;
        step <<= 1;
        cur = (hi - cur > step) ? cur + step : hi;
    }
    return upper_bound_range<T, A>(d, prev + 1, cur < hi ? cur + 1 : hi, key);
}

// floor((v - origin) / width), width >= 1. False: the quotient does not fit int64 (nothing written).
BOLT_FORCE_INLINE bool bucket_of_i64(int64_t v, int64_t origin, uint64_t width, int64_t* b) noexcept {
    assert(width >= 1u && b != nullptr);
    if (v >= origin) {
        const uint64_t m = static_cast<uint64_t>(v) - static_cast<uint64_t>(origin);
        const uint64_t q = m / width;
        if (q > static_cast<uint64_t>(INT64_MAX)) return false;
        *b = static_cast<int64_t>(q);
        return true;
    }
    const uint64_t m = static_cast<uint64_t>(origin) - static_cast<uint64_t>(v);
    const uint64_t q = m / width + ((m % width) != 0u ? 1u : 0u);
    if (q > static_cast<uint64_t>(INT64_MAX) + 1u) return false;
    *b = q == static_cast<uint64_t>(INT64_MAX) + 1u ? INT64_MIN : -static_cast<int64_t>(q);
    return true;
}

// The first value of the bucket after the one holding v, or false when it exceeds INT64_MAX (v's bucket is the
// last one that can hold an int64).
BOLT_FORCE_INLINE bool next_bucket_start_i64(int64_t v, int64_t origin, uint64_t width, int64_t* next) noexcept {
    assert(width >= 1u && next != nullptr);
    uint64_t rem;
    if (v >= origin) {
        rem = (static_cast<uint64_t>(v) - static_cast<uint64_t>(origin)) % width;
    } else {
        const uint64_t r = (static_cast<uint64_t>(origin) - static_cast<uint64_t>(v)) % width;
        rem = r == 0u ? 0u : width - r;
    }
    const uint64_t gap = width - rem;                    // 1 .. width
    // Order-preserving map int64 -> uint64 (add 2^63): the next start is v + gap unless that passes INT64_MAX.
    constexpr uint64_t kBias = 0x8000000000000000ull;
    const uint64_t u = static_cast<uint64_t>(v) ^ kBias;
    const uint64_t nu = u + gap;
    if (nu < u) return false;
    *next = static_cast<int64_t>(nu ^ kBias);
    return true;
}

// Per-bucket counts of the ascending run d[lo, hi): out_bucket[i], out_count[i] for the i-th non-empty bucket, at
// most `cap` of them. *end is the position after the last bucket counted (hi when the run is done); call again from
// *end for the rest. Returns the pairs written, or -1 when a bucket index does not fit int64.
template <typename A>
BOLT_FORCE_INLINE int64_t count_buckets_i64(const A& d, int64_t lo, int64_t hi, int64_t origin, uint64_t width,
                                            int64_t* out_bucket, uint64_t* out_count, int64_t cap, int64_t* end_out) noexcept {
    assert(width >= 1u && lo >= 0 && lo <= hi);
    assert(out_bucket != nullptr && out_count != nullptr && cap >= 1 && end_out != nullptr);
    int64_t n = 0, at = lo;
    while (at < hi && n < cap) {                        // bounded: buckets of the run
        int64_t b = 0;
        if (!bucket_of_i64(d[at], origin, width, &b)) return -1;
        int64_t next_start = 0, end = hi;
        if (next_bucket_start_i64(d[at], origin, width, &next_start))
            end = gallop_lower_bound<int64_t, A>(d, at + 1, hi, next_start);
        out_bucket[n] = b;
        out_count[n] = static_cast<uint64_t>(end - at);
        ++n;
        at = end;
    }
    *end_out = at;
    return n;
}

}  // namespace sorted
}  // namespace kernels
}  // namespace bolt
