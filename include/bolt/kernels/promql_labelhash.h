// bolt/kernels/promql_labelhash.h — Prometheus's `labels.Labels.Hash()` and
// the `limit_ratio` sampler built on it.
//
// WHY THIS IS EXACT AND NOT "A HASH".
//
// `limit_ratio(r, v)` decides membership per series with
//
//     sampleOffset := float64(sample.Metric.Hash()) / float64(math.MaxUint64)
//     keep         := (r >= 0 && sampleOffset <  r) ||
//                     (r <  0 && sampleOffset >= (1.0 + r))
//
// (promql/engine.go, `ratiosampler.AddRatioSample`). Every part of that is
// observable: which series come back for a given `r` is the ANSWER, not an
// implementation detail. Substituting any other deterministic hash still
// satisfies the two properties the vendored corpus actually asserts —
// `limit_ratio(x) or limit_ratio(x-1)` is everything, `and` is nothing, and
// the count for r=0.5 lands in [3,5] of 8 — so a wrong hash scores GREEN on
// the whole of `limit.test` while returning a DIFFERENT SET OF SERIES than
// Prometheus for every real query a user writes. That is the reason this is a
// port and not an approximation, and the reason the gate beside it asserts the
// selected label SETS against prometheus v3.7.3 rather than their count.
//
// WHICH `Labels.Hash()`. THERE ARE TWO, AND THEY DISAGREE.
//
// prometheus ships two `Labels` representations behind a build tag, and their
// `Hash()` methods hash DIFFERENT BYTES:
//
//   * `labels.go` — the slice implementation, and the one anybody reads first
//     — hashes  `name 0xFF value 0xFF`  per label.
//   * `labels_stringlabels.go`, selected by the `stringlabels` build tag,
//     keeps one flat string of  `uvarint(len(name)) name uvarint(len(value))
//     value`  in name order, and hashes that.
//
// prometheus 3.x's release binaries — and `prom/prometheus:v3.7.3`, the image
// this tree gates against — are built with `stringlabels`, so the second is
// what a user's Prometheus actually computes. This port implements the second,
// and that choice is MEASURED, not read: at r=0.5 over eight series the
// shipped binary keeps {production/0, production/1, canary/1, canary/3,
// canary/10}; the varint encoding predicts exactly that set and the 0xFF
// encoding predicts a different THREE. Reproduce with
// `python3 benchmarks/promql/limit_ratio_oracle.py`, which probes one ratio
// per adjacent-offset boundary (so a hash that merely swaps two neighbours
// fails) and reports ALL BOUNDARIES AGREE.
//
// The consequence is worth stating rather than hiding: `limit_ratio`'s
// selected set is NOT portable across prometheus builds. Both builds satisfy
// every property the language documents — the complement identities, the
// stability across steps — and differ only in which series they name. We match
// the shipped one.
//
// Other load-bearing details:
//
//   * `__name__` is an ordinary label and IS hashed. A bare selector's samples
//     carry it (3.x keeps the name and flags DropName instead of stripping),
//     so dropping it here shifts every offset.
//   * The label set is sorted by NAME. Prometheus keeps `Labels` sorted as an
//     invariant; our label enumeration is in column order, so this sorts.
//   * Empty-valued labels do not exist in Prometheus's model — a label with an
//     empty value is an absent label — so they must not be serialized.
//   * Upstream has a >1KB streaming fallback that produces the SAME digest
//     (same bytes, incremental instead of buffered); the only difference is
//     allocation, so a single buffer is faithful as long as it does not
//     truncate. `label_set_hash` refuses rather than truncating.
//
// XXH64 itself is not re-implemented: bolt already carries one, pinned against
// xxhash.com's published vectors in `test_bolt_parquet_pageindex.cpp`
// (`XXH64("", 0) == 0xEF46DB3751D8E999`, "abc", the 62-byte vector, seed 42).
// A second copy would be a second thing to get wrong.
//
// Zero allocation (caller-sized stack buffer), no exceptions, bounded.

#ifndef BOLT_KERNELS_PROMQL_LABELHASH_H
#define BOLT_KERNELS_PROMQL_LABELHASH_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/ingest/bolt_parquet_bloom.h"

namespace bolt::promql {

// One (name, value) label pair, borrowed. Value "" means the label is absent.
struct LabelHashPair {
    const char* name;
    const char* value;
};

// Serialization buffer cap. Prometheus's own fast path reserves 1 KiB and
// falls back to streaming past it; 8 KiB here covers any label set this engine
// can carry (k_max_label_cols * (name_cap + value_cap) is far below it) while
// still being a bound rather than a hope.
inline constexpr std::uint32_t k_label_hash_buf = 8192u;

// Prometheus `labels.Labels.Hash()`. Returns false — never a truncated digest —
// when the serialization would not fit. `n` may be 0 (the empty label set
// hashes as XXH64("") == 0xEF46DB3751D8E999, which is what Prometheus returns).
inline bool label_set_hash(const LabelHashPair* pairs, std::uint32_t n,
                           std::uint64_t* out) noexcept {
    assert(out != nullptr);
    assert(pairs != nullptr || n == 0u);
    if (n > 256u) return false;                 // bounded: no unbounded sort
    // Index sort by name (insertion; label sets are tiny and this stays
    // branch-simple). Duplicate names cannot occur in a valid label set.
    std::uint16_t idx[256];
    std::uint32_t m = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (pairs[i].name == nullptr || pairs[i].name[0] == '\0') continue;
        if (pairs[i].value == nullptr || pairs[i].value[0] == '\0') continue;
        idx[m++] = static_cast<std::uint16_t>(i);
    }
    for (std::uint32_t a = 1; a < m; ++a) {
        const std::uint16_t cur = idx[a];
        std::int32_t b = static_cast<std::int32_t>(a) - 1;
        while (b >= 0 && std::strcmp(pairs[idx[static_cast<std::uint32_t>(b)]].name,
                                     pairs[cur].name) > 0) {
            idx[static_cast<std::uint32_t>(b) + 1] = idx[static_cast<std::uint32_t>(b)];
            --b;
        }
        idx[static_cast<std::uint32_t>(b) + 1] = cur;
    }
    std::uint8_t buf[k_label_hash_buf];
    std::uint32_t o = 0;
    // LEB128, matching stringlabels' `decodeSize`: 7 bits per byte,
    // low group first, continuation bit set on all but the last.
    auto put_uvarint = [&buf, &o](std::size_t n) noexcept {
        for (;;) {
            const std::uint8_t b = static_cast<std::uint8_t>(n & 0x7Fu);
            n >>= 7;
            buf[o++] = n ? static_cast<std::uint8_t>(b | 0x80u) : b;
            if (n == 0) return;
        }
    };
    for (std::uint32_t a = 0; a < m; ++a) {
        const LabelHashPair& p = pairs[idx[a]];
        const std::size_t ln = std::strlen(p.name);
        const std::size_t lv = std::strlen(p.value);
        // 10 bytes covers two LEB128 lengths for any string this engine can
        // hold; the check is against the true worst case, not a guess.
        if (o + ln + lv + 20u > k_label_hash_buf) return false;   // never truncate
        put_uvarint(ln);
        std::memcpy(buf + o, p.name, ln); o += static_cast<std::uint32_t>(ln);
        put_uvarint(lv);
        std::memcpy(buf + o, p.value, lv); o += static_cast<std::uint32_t>(lv);
    }
    *out = bolt::ingest::parquet::pq_xxh64(buf, o, 0);
    assert(o <= k_label_hash_buf);
    return true;
}

// `float64(hash) / float64(math.MaxUint64)`, exactly as Go computes it: the
// uint64 is converted to float64 FIRST (rounding to nearest, so the result can
// reach 1.0 for hashes within half an ulp of 2^64), then divided by the
// float64 value of MaxUint64 — which is itself 2^64 after conversion. Writing
// it as `h * (1.0 / 18446744073709551615.0)` is NOT the same expression and
// can differ in the last bit, which is exactly the kind of difference that
// flips one series in or out at a threshold.
inline double label_sample_offset(std::uint64_t h) noexcept {
    const double num = static_cast<double>(h);
    const double den = static_cast<double>(UINT64_MAX);
    return num / den;
}

// `ratiosampler.AddRatioSample`. `r` is assumed already clamped to [-1, 1] by
// the caller (upstream clamps and emits a warning; clamping silently here
// would hide that warning's trigger).
inline bool ratio_sample_keep(double r, double sample_offset) noexcept {
    return (r >= 0.0 && sample_offset < r) ||
           (r < 0.0 && sample_offset >= (1.0 + r));
}

}  // namespace bolt::promql

#endif  // BOLT_KERNELS_PROMQL_LABELHASH_H
