// bolt_radixpart.h — radix partition of int64 keys into hash buckets.
//
// Companion to bolt_radixsort.h. Where radix_sort fully orders keys, this
// kernel only PARTITIONS rows into 2^radix_bits buckets by a hash of each
// key, producing a permutation grouped by bucket plus the bucket offsets.
// One histogram pass + one scatter pass, O(n). No allocation.
//
// Primary use — parallel hash aggregation / hash join build by KEY CLASS:
// partition the input so each bucket holds a DISJOINT set of group keys (the
// same key always hashes to the same bucket), then one worker per bucket
// builds a private table over ~n/buckets keys. Because buckets are key-
// disjoint the per-worker partials need NO merge — their outputs concatenate.
// This is the high-cardinality path the row-range partition driver can't take
// (every row-range worker would otherwise build a table over ALL keys).
//
// Hash, not raw low bits: hashing spreads keys that share low bits (e.g. all
// multiples of 16) evenly across buckets, avoiding the degenerate single-
// bucket skew a raw-radix split would hit. The mix is the SplitMix64
// finalizer — same family as the project's fuzz RNG — deterministic so a key
// always lands in one bucket.
//
// Tiger Style: noexcept, no allocation, fixed bounds, ≥2 asserts, branch-light.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"   // BOLT_FORCE_INLINE, BOLT_RESTRICT

namespace bolt {

// Max buckets = 2^kMaxRadixPartBits. 8 bits = 256 buckets covers any realistic
// worker count (the partition fans into ≥ workers buckets, workers ≤ ~64). The
// per-bucket scatter cursor lives on the stack, so the cap also bounds that.
constexpr int     kMaxRadixPartBits    = 8;
constexpr int64_t kMaxRadixPartBuckets = static_cast<int64_t>(1) << kMaxRadixPartBits;

// SplitMix64 finalizer — maps a key to a well-mixed 64-bit hash so the low
// `radix_bits` used for bucketing are uniform even for low-entropy keys.
BOLT_FORCE_INLINE std::uint64_t radixpart_mix(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

// Partition `n` rows into `2^radix_bits` buckets by hash(keys[i]).
//
//   keys     [n]              read-only int64 keys
//   perm     [n]              OUT: row indices grouped by bucket
//   offsets  [n_buckets + 1]  OUT: exclusive prefix; bucket b occupies
//                             perm[offsets[b] .. offsets[b+1]).
//
// Stable within a bucket (rows keep input order). radix_bits in [1, 8].
// O(n) time, O(2^radix_bits) stack scratch. No allocation.
BOLT_FORCE_INLINE void radix_partition_hash_i64(
        const std::int64_t* BOLT_RESTRICT keys,
        std::int64_t                      n,
        int                               radix_bits,
        std::int32_t* BOLT_RESTRICT       perm,
        std::int64_t* BOLT_RESTRICT       offsets) noexcept {
    assert(n >= 0);
    assert((keys != nullptr && perm != nullptr) || n == 0);
    assert(offsets != nullptr);
    assert(radix_bits >= 1 && radix_bits <= kMaxRadixPartBits);

    const std::int64_t  n_buckets = static_cast<std::int64_t>(1) << radix_bits;
    const std::uint64_t mask      = static_cast<std::uint64_t>(n_buckets) - 1u;

    // Histogram into offsets[1..n_buckets] (shifted by one for the exclusive
    // prefix sum below); offsets[0] stays 0.
    for (std::int64_t b = 0; b <= n_buckets; ++b) offsets[b] = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::uint64_t h = radixpart_mix(static_cast<std::uint64_t>(keys[i]));
        ++offsets[static_cast<std::int64_t>(h & mask) + 1];
    }
    // Exclusive prefix sum: offsets[b] = start index of bucket b.
    for (std::int64_t b = 0; b < n_buckets; ++b) offsets[b + 1] += offsets[b];
    assert(offsets[n_buckets] == n);

    // Scatter row indices using a private cursor so `offsets` keeps the bucket
    // starts for the caller.
    std::int64_t cursor[kMaxRadixPartBuckets];
    for (std::int64_t b = 0; b < n_buckets; ++b) cursor[b] = offsets[b];
    for (std::int64_t i = 0; i < n; ++i) {
        const std::uint64_t h = radixpart_mix(static_cast<std::uint64_t>(keys[i]));
        const std::int64_t  bkt = static_cast<std::int64_t>(h & mask);
        perm[cursor[bkt]++] = static_cast<std::int32_t>(i);
    }
    assert(n == 0 || cursor[n_buckets - 1] == offsets[n_buckets]);
}

}  // namespace bolt
