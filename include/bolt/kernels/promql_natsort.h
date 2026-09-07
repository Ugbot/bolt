// bolt/kernels/promql_natsort.h — Prometheus's NATURAL string ordering.
//
// The comparison `sort_by_label` / `sort_by_label_desc` use on label VALUES.
// It is NOT `strcmp`, and the difference is a silent wrong answer rather than
// an error: upstream's own answer key orders `cpu` as
//
//     0, 1, 2, 3, 10, 11, 12, 20, 21, 100
//
// where a byte comparison gives 0, 1, 10, 100, 11, 12, 2, 20, 21, 3 — the same
// ten series, the same ten values, in an order the caller asked to be
// different. `expect ordered` is the only assertion that can see it.
//
// This is a port of github.com/facette/natsort's `Compare`, which
// prometheus/promql vendors and calls from `funcSortByLabel`. The port is
// deliberately literal, including three behaviours that look like bugs and are
// load-bearing because upstream's ordering is defined by this exact code:
//
//   * it is a strict LESS-THAN predicate that never reports "equal" — the
//     caller (`funcSortByLabel`) has already established the two values differ
//     and maps false to "a after b";
//   * a numerically equal chunk pair with different text ("007" vs "7") does
//     not decide the comparison, it falls through to the NEXT chunk;
//   * when one side runs out of chunks first, the answer is decided by WHICH
//     side ran out, checked in the order A-then-B.
//
// Oracle: prometheus v3.7.3 answering `sort_by_label` over a real TSDB agrees
// with the vendored key on every discriminating cell (`cpu`, `4m5/4m600/
// 4m1000`, `1.2.3/1.11.3/1.111.3`). GreptimeDB does NOT implement the ordering
// — it returns byte order — so it is a characterised divergence here, not a
// third voice.
//
// Zero allocation, no exceptions, bounded: one forward pass over each string.

#ifndef BOLT_KERNELS_PROMQL_NATSORT_H
#define BOLT_KERNELS_PROMQL_NATSORT_H

#include <cstdint>
#include <cstring>

namespace bolt::promql {

// A maximal run of digits, or a maximal run of non-digits. `chunkify`'s
// `(\d+|\D+)` with no allocation.
struct NatChunk {
    const char*   p;
    std::uint32_t n;
    bool          numeric;
};

inline bool natsort_is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// Advance `cur` past one chunk. Returns false when the string is exhausted.
inline bool natsort_next_chunk(const char** cur, const char* end,
                               NatChunk* out) noexcept {
    if (cur == nullptr || *cur == nullptr || out == nullptr) return false;
    const char* c = *cur;
    if (c >= end) return false;
    const bool dig = natsort_is_digit(*c);
    const char* s = c;
    while (c < end && natsort_is_digit(*c) == dig) ++c;
    out->p = s;
    out->n = static_cast<std::uint32_t>(c - s);
    out->numeric = dig;
    *cur = c;
    return true;
}

// Go's `strconv.Atoi` on an all-digit chunk: succeeds iff the value fits in a
// 64-bit signed int. A 20-digit run, or a 19-digit run past INT64_MAX, ERRORS
// upstream and falls to the string branch — so overflow must be detected, not
// wrapped, or two absurdly long numeric labels would order by a truncated
// value.
inline bool natsort_atoi(const NatChunk& c, std::int64_t* out) noexcept {
    if (out == nullptr || c.n == 0u || !c.numeric) return false;
    constexpr std::int64_t k_max = 9223372036854775807LL;
    std::int64_t v = 0;
    for (std::uint32_t i = 0; i < c.n; ++i) {
        const std::int64_t d = static_cast<std::int64_t>(c.p[i] - '0');
        if (v > (k_max - d) / 10) return false;  // would overflow
        v = v * 10 + d;
    }
    *out = v;
    return true;
}

// Go string comparison over two chunks: byte-wise, then by length.
inline int natsort_chunk_cmp(const NatChunk& a, const NatChunk& b) noexcept {
    const std::uint32_t n = (a.n < b.n) ? a.n : b.n;
    const int r = (n == 0u) ? 0 : std::memcmp(a.p, b.p, n);
    if (r != 0) return r;
    if (a.n == b.n) return 0;
    return (a.n < b.n) ? -1 : 1;
}

// True iff `a` precedes `b` in natural order. NUL-terminated inputs.
inline bool promql_natsort_less(const char* a, const char* b) noexcept {
    if (a == nullptr || b == nullptr) return false;
    const char* ap = a;
    const char* ae = a + std::strlen(a);
    const char* bp = b;
    const char* be = b + std::strlen(b);
    // Bounded: each iteration consumes at least one byte of BOTH strings, so
    // the loop cannot run more times than the shorter string is long.
    for (;;) {
        NatChunk ca{}, cb{};
        // `for i := range chunksA` ended -> false.
        if (!natsort_next_chunk(&ap, ae, &ca)) return false;
        // `if i >= nChunksB { return false }`.
        if (!natsort_next_chunk(&bp, be, &cb)) return false;
        const bool a_last = (ap >= ae);
        const bool b_last = (bp >= be);
        std::int64_t ai = 0;
        std::int64_t bi = 0;
        const bool an = natsort_atoi(ca, &ai);
        const bool bn = natsort_atoi(cb, &bi);
        if (an && bn) {
            if (ai != bi) return ai < bi;
            // Numerically equal but possibly different text ("007" vs "7"):
            // upstream does NOT decide here.
            if (a_last) return true;
            if (b_last) return false;
            continue;
        }
        const int c = natsort_chunk_cmp(ca, cb);
        if (c != 0) return c < 0;
        if (a_last) return true;
        if (b_last) return false;
    }
}

}  // namespace bolt::promql

#endif  // BOLT_KERNELS_PROMQL_NATSORT_H
