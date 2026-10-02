// bolt_mergejoin_bytes.h — merge join over sorted, order-preserving byte
// keys (MSEG B10): canonical key_encode() bytes, so one kernel serves every
// key kind and every multi-column key (the columns are concatenated in the
// encoding). The int64 join is bolt_mergejoin.h.
//
// Kinds: inner, left outer (unmatched probe rows, build index -1), right
// outer (unmatched build rows, probe index -1) and full outer. Within an
// equal-key run the pairs are probe-major: for each probe row, every build
// row of the run in order. Output is in key order.
//
// Resumable: the cursor carries the walk and the position inside an equal-key
// cross product, so a bounded output buffer never drops a pair. Call
// mergejoin_bytes_next until it returns 0.
//
// Tiger Style: POD + free functions, noexcept, no allocation, every loop
// bounded by the output capacity or an input length.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_kmerge_bytes.h"

namespace bolt {

enum class MergeJoinKind : uint8_t {
    Inner      = 0,
    LeftOuter  = 1,   // + probe rows with no build match
    RightOuter = 2,   // + build rows with no probe match
    FullOuter  = 3,
};

struct MergeJoinBytesCursor {
    KeyBytesColumn build;
    KeyBytesColumn probe;
    int64_t i, j;            // next unconsumed build / probe row
    int64_t run_i_end;       // equal-key run [i, run_i_end) x [j, run_j_end)
    int64_t run_j_end;
    int64_t a, b;            // next pair inside the run (b outer, a inner)
    MergeJoinKind kind;
    uint8_t in_run;
    uint8_t _pad[6];
};

inline bool mergejoin_bytes_init(MergeJoinBytesCursor* c, const KeyBytesColumn& build,
                                 const KeyBytesColumn& probe,
                                 MergeJoinKind kind) noexcept {
    assert(c != nullptr);
    assert(build.len >= 0 && probe.len >= 0);
    if (build.len < 0 || probe.len < 0) return false;
    if (static_cast<uint8_t>(kind) > static_cast<uint8_t>(MergeJoinKind::FullOuter))
        return false;
    memset(c, 0, sizeof(*c));
    c->build = build;
    c->probe = probe;
    c->kind = kind;
    return true;
}

namespace detail {

BOLT_FORCE_INLINE int mjb_cmp(const KeyBytesColumn& x, int64_t xi,
                              const KeyBytesColumn& y, int64_t yi) noexcept {
    const uint8_t *p = nullptr, *q = nullptr;
    uint32_t pn = 0, qn = 0;
    key_bytes_at(x, xi, &p, &pn);
    key_bytes_at(y, yi, &q, &qn);
    return key_bytes_cmp(p, pn, q, qn);
}

// End of the run of rows equal to row `from` of `c`.
inline int64_t mjb_run_end(const KeyBytesColumn& c, int64_t from) noexcept {
    assert(from >= 0 && from < c.len);
    int64_t e = from + 1;
    while (e < c.len && mjb_cmp(c, from, c, e) == 0) ++e;   // bounded: c.len
    return e;
}

// Emit pairs of the current run until it ends or the buffer is full.
inline int64_t mjb_drain_run(MergeJoinBytesCursor* c, int64_t* ob, int64_t* op,
                             int64_t count, int64_t cap) noexcept {
    assert(c != nullptr && c->in_run);
    assert(count <= cap);
    while (count < cap) {                         // bounded by cap
        ob[count] = c->a;
        op[count] = c->b;
        ++count;
        if (++c->a == c->run_i_end) {
            c->a = c->i;
            if (++c->b == c->run_j_end) {
                c->i = c->run_i_end;
                c->j = c->run_j_end;
                c->in_run = 0;
                break;
            }
        }
    }
    return count;
}

}  // namespace detail

/// Emit up to `cap` (build_idx, probe_idx) pairs (-1 = no match on that
/// side). Returns pairs written; 0 when the join is complete.
inline int64_t mergejoin_bytes_next(MergeJoinBytesCursor* c,
                                    int64_t* BOLT_RESTRICT out_build_idx,
                                    int64_t* BOLT_RESTRICT out_probe_idx,
                                    int64_t cap) noexcept {
    assert(c != nullptr && cap >= 0);
    assert(out_build_idx != nullptr && out_probe_idx != nullptr);
    const bool keep_build = c->kind == MergeJoinKind::RightOuter ||
                            c->kind == MergeJoinKind::FullOuter;
    const bool keep_probe = c->kind == MergeJoinKind::LeftOuter ||
                            c->kind == MergeJoinKind::FullOuter;
    int64_t n = 0;
    // Each pass emits >= 1 pair or consumes >= 1 input row, and leaves the
    // loop when the buffer fills, so it runs at most cap + bn + pn times.
    while (n < cap) {
        if (c->in_run) { n = detail::mjb_drain_run(c, out_build_idx, out_probe_idx, n, cap); continue; }
        const bool bi = c->i < c->build.len, pj = c->j < c->probe.len;
        if (!bi && !pj) break;
        const int s = (bi && pj) ? detail::mjb_cmp(c->build, c->i, c->probe, c->j)
                                 : (bi ? -1 : 1);
        if (s < 0) {
            if (keep_build) { out_build_idx[n] = c->i; out_probe_idx[n] = -1; ++n; }
            ++c->i;
        } else if (s > 0) {
            if (keep_probe) { out_build_idx[n] = -1; out_probe_idx[n] = c->j; ++n; }
            ++c->j;
        } else {
            c->run_i_end = detail::mjb_run_end(c->build, c->i);
            c->run_j_end = detail::mjb_run_end(c->probe, c->j);
            c->a = c->i;
            c->b = c->j;
            c->in_run = 1;
        }
    }
    assert(n <= cap);
    return n;
}

}  // namespace bolt
