// bolt_csr_shortest.h — unweighted bidirectional shortest path over CSR.
//
// One kernel:
//   - csr_shortest_bidir : alternating level-synchronous BFS from s over the
//     FORWARD CSR and from t over the REVERSE CSR, meeting in the middle.
//     Returns the hop length of a shortest s->t path and, optionally, ONE
//     witness path (its node sequence and, if asked, its relationship ids).
//
// This is the kernel chukonu's `PhysicalShortestPathBidir` has been costed
// for since Stage 4 without ever existing. Declared-but-never-built is a
// documented failure mode in this tree (PathExpandPayload::semantics was
// designed, carried through the canonical layer, and never passed — a silent
// wrong answer), so the kernel lands with its own oracle-checked tests before
// anything is wired to it.
//
// ---------------------------------------------------------------------------
// WHY BIDIRECTIONAL, AND WHY IT IS CORRECT
// ---------------------------------------------------------------------------
// One-directional BFS to depth d touches O(b^d) nodes; two searches meeting at
// d/2 touch O(2*b^(d/2)). The saving is the whole point of the operator.
//
// Correctness rests on ONE property, which is stronger than the usual folklore
// and is what lets the search stop at the FIRST meeting:
//
//   Both sides expand FULL LEVELS. `df` / `dr` are the deepest levels each side
//   has COMPLETED, so every node within df of s (resp. dr of t) carries its
//   TRUE distance. A meeting is recorded whenever a side labels a node v the
//   other side has already labelled: cand = dist_this[v] + dist_other[v].
//
//   CLAIM: the first recorded meeting already gives the true distance D.
//   PROOF. Say the first double-labelling happens while side f expands to
//   level nd, with side r having completed level dr. Every meeting cand is the
//   length of a real s->v->t walk, so mu >= D always. For the other direction,
//   suppose D <= (nd-1) + dr. Take the shortest path P and its node w at
//   position D - dr (position D, i.e. t itself, if D < dr). On a shortest path
//   dist_f[w] = D - dr <= nd-1 and dist_r[w] = dr, so w was ALREADY labelled by
//   both sides — before this level — contradicting "first". Hence
//   D >= nd + dr >= nd + dist_r[v] = cand >= mu, and with mu >= D, mu = D. []
//
//   Consequences worth stating because they are easy to get wrong:
//   - Stopping at the first meeting is exact. So are the conservative variants
//     `mu <= df+dr` and the folklore `mu <= df+dr+1`; they merely do MORE work.
//     This header previously asserted that `mu <= df+dr+1` was unsafe. That
//     claim was FALSE and was refuted by measurement, not by re-reading: a
//     differential against plain BFS over 237,152 (s,t) pairs on 4,000 random
//     graphs found ZERO disagreement for any of those rules, and 48,000 more
//     pairs found zero for stop-at-first-meeting. The claim is corrected here
//     rather than quietly deleted.
//   - A pair whose true distance is <= max_hops is ALWAYS resolved: the level
//     that finds the first meeting has nd + dr <= D <= max_hops, so the cap
//     check (which tests the depths BEFORE that expansion) cannot pre-empt it.
//   - Stop when either frontier empties: that side's BFS is complete over the
//     component, so if t is reachable it was labelled and its meeting (against
//     the other side's seed) was recorded. mu is then exact, or the pair is
//     genuinely unreachable.
//
// ---------------------------------------------------------------------------
// LOUD REFUSALS, NEVER A QUIET WRONG NUMBER
// ---------------------------------------------------------------------------
//   - `max_hops` bounds the search. If it is reached before a shortest path is
//     PROVEN, the status is CsrShortestStatus::HopCapReached and length is
//     written as -1. -1 under Ok means "proven unreachable"; -1 under
//     HopCapReached means "not determined within the cap". The caller must
//     distinguish them — they are different answers, and conflating them is
//     exactly the wrong-answer class this tree keeps paying for.
//   - A found path whose witness does not fit out_path_cap returns
//     WitnessTruncated. `length` is still valid and proven; the PATH is the
//     part that did not fit, and nothing is written past the cap.
//
// ---------------------------------------------------------------------------
// SHAPE RULES (Tiger Style)
// ---------------------------------------------------------------------------
//   - >=2 asserts/fn, <=70-line fns, noexcept, no exceptions, no RTTI
//   - NO heap allocation: the caller owns every scratch array (n_nodes slots)
//   - visited marking uses a monotone EPOCH stamp, so a query costs O(1) to
//     reset instead of O(n_nodes) — the caller zeroes the stamps ONCE at
//     init and the kernel manages the epoch (including a bounded re-zero on
//     the 2^31 wrap).
//   - every loop has a fixed upper bound: level expansion is bounded by
//     n_nodes (each node is queued at most once), the level alternation by
//     2*max_hops + 4, witness reconstruction by max_hops.
//
// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------
// Frontier nodes are processed in queue order and their neighbours in CSR
// insertion order, so the meeting node chosen for a given `mu` — and hence the
// witness path — is deterministic for a given graph and (s, t). When several
// shortest paths exist the kernel returns ONE of them; which one is stable
// across runs but is not specified as "the smallest" by any ordering.
//
// ---------------------------------------------------------------------------
// ns budget — MEASURED, not estimated
// ---------------------------------------------------------------------------
// `test_bolt_csr_shortest.cpp`'s CsrShortestRandom.PerfSmokeNotGated prints it
// on every run:
//
//   csr_shortest_bidir, 1,000 nodes / 10,000 edges (avg out-degree ~10, mean
//   distance ~3), 6,000 queries:  178 ns per QUERY
//   (Apple M4, clang -O3 -g, asserts LIVE — a shipping -DNDEBUG build is
//   faster than this, not slower.)
//
// Per EDGE examined that is one neighbour load, the branch-free label keep, a
// stamp compare, and — only on first discovery — four stores. The win over a
// single-ended BFS is in edges EXAMINED, not in per-edge cost: on a b-ary
// shape at distance d it examines O(2*b^(d/2)) instead of O(b^d), and the
// epoch stamp keeps per-query setup at O(1) rather than O(n_nodes).
// PRINTED, NOT GATED this wave (W1-L1).

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_bfs.h"   // CsrBfsGraph + csr_edge_label_keep

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// Upper bound on the hop cap a caller may ask for. Bounds the alternation
// loop and the witness walk.
constexpr int32_t k_csr_shortest_max_hops = 1 << 20;

enum class CsrShortestStatus : int32_t {
    Ok               = 0,   // length is PROVEN (-1 == proven unreachable)
    HopCapReached    = 1,   // undetermined within max_hops; length = -1
    WitnessTruncated = 2,   // length proven; the witness did not fit
    InvalidArgument  = 3,
};

struct CsrShortestResult {
    int32_t length;        // hops, or -1 (see the status for which -1 it is)
    int32_t witness_len;   // node count written to out_path_nodes (length + 1)
};

// Caller-owned scratch. Every array needs n_nodes slots. `parent_*` /
// `pedge_*` are needed ONLY when a witness path is requested; pass all four
// null (or all four non-null) — a mixed set is rejected. `epoch` must be
// zero-initialised together with stamp_f / stamp_r on first use and then left
// to the kernel.
struct CsrShortestScratch {
    int32_t* stamp_f;    // visited epoch, forward side
    int32_t* stamp_r;    // visited epoch, reverse side
    int32_t* dist_f;     // hop distance from s   (valid iff stamp_f == epoch)
    int32_t* dist_r;     // hop distance from t   (valid iff stamp_r == epoch)
    int64_t* queue_f;    // BFS queue, forward
    int64_t* queue_r;    // BFS queue, reverse
    int64_t* parent_f;   // predecessor toward s, or null
    int64_t* parent_r;   // predecessor toward t, or null
    int64_t* pedge_f;    // relationship id of the parent_f edge, or null
    int64_t* pedge_r;    // relationship id of the parent_r edge, or null
    int32_t  epoch;      // kernel-managed; caller zeroes once
    int32_t  _pad;
};

// Slots needed per scratch array, for a graph with n_nodes dense node ids.
constexpr int64_t csr_shortest_scratch_slots(int64_t n_nodes) noexcept {
    return n_nodes;
}

// ============================================================================
// internals
// ============================================================================

// One side of the meet-in-the-middle search.
struct CsrSpSide {
    const CsrBfsGraph* g;
    int32_t* stamp;
    int32_t* dist;
    int64_t* queue;
    int64_t* parent;    // may be null (no witness wanted)
    int64_t* pedge;     // non-null iff parent is non-null
    int64_t  lvl_begin; // queue range [lvl_begin, lvl_end) = current level
    int64_t  lvl_end;
    int64_t  tail;      // next queue write slot
    int32_t  depth;     // deepest COMPLETED level
    int32_t  _pad;
};

// Bump the visit epoch. Re-zeroes both stamp arrays on the (bounded) wrap so
// a stale stamp can never be mistaken for a live one. O(1) per query except
// once every 2^31 queries.
BOLT_FORCE_INLINE void csr_sp_begin_epoch(CsrShortestScratch* sc,
                                          int64_t n_nodes) noexcept {
    assert(sc != nullptr && n_nodes > 0);
    assert(sc->stamp_f != nullptr && sc->stamp_r != nullptr);
    if (sc->epoch >= 0x7ffffffe) {
        for (int64_t i = 0; i < n_nodes; ++i) {
            sc->stamp_f[i] = 0;
            sc->stamp_r[i] = 0;
        }
        sc->epoch = 0;
    }
    ++sc->epoch;
    assert(sc->epoch > 0);
}

// Expand one full BFS level on side `a`, checking every newly-labelled node
// against side `b`'s stamps and keeping the best meeting in (*mu, *meet).
// Each node enters a queue at most once (the stamp guarantees it), so the
// writes are bounded by n_nodes.
inline void csr_sp_expand_level(CsrSpSide* BOLT_RESTRICT a,
                                const CsrSpSide* BOLT_RESTRICT b,
                                int32_t epoch, int64_t n_nodes,
                                int32_t* BOLT_RESTRICT mu,
                                int64_t* BOLT_RESTRICT meet) noexcept {
    assert(a != nullptr && b != nullptr && mu != nullptr && meet != nullptr);
    assert(a->lvl_begin <= a->lvl_end && a->lvl_end <= a->tail);
    const int32_t nd = a->depth + 1;
    for (int64_t qi = a->lvl_begin; qi < a->lvl_end; ++qi) {
        const int64_t u = a->queue[qi];
        assert(u >= 0 && u < n_nodes && "csr_shortest: queued id out of range");
        const int64_t e = a->g->off[u + 1];
        for (int64_t j = a->g->off[u]; j < e; ++j) {
            if (csr_edge_label_keep(a->g->edge_labels, a->g->want_label, j) == 0)
                continue;
            const int64_t v = a->g->neighbors[j];
            assert(v >= 0 && v < n_nodes && "csr_shortest: neighbour out of range");
            if (a->stamp[v] == epoch) continue;
            a->stamp[v] = epoch;
            a->dist[v]  = nd;
            if (a->parent != nullptr) {
                a->parent[v] = u;
                a->pedge[v]  = a->g->edge_ids[j];
            }
            assert(a->tail < n_nodes && "csr_shortest: queue overflow");
            a->queue[a->tail++] = v;
            if (b->stamp[v] == epoch) {
                const int32_t cand = nd + b->dist[v];
                if (cand < *mu) { *mu = cand; *meet = v; }
            }
        }
    }
    a->lvl_begin = a->lvl_end;
    a->lvl_end   = a->tail;
    a->depth     = nd;
}

// Seed one side with its start node at distance 0.
BOLT_FORCE_INLINE void csr_sp_seed(CsrSpSide* side, int64_t start,
                                   int32_t epoch) noexcept {
    assert(side != nullptr && start >= 0);
    assert(side->stamp != nullptr && side->queue != nullptr);
    side->stamp[start] = epoch;
    side->dist[start]  = 0;
    side->queue[0]     = start;
    side->lvl_begin    = 0;
    side->lvl_end      = 1;
    side->tail         = 1;
    side->depth        = 0;
}

// Reconstruct the witness: walk parent_f from `meet` back to s (reversed into
// the prefix) then parent_r forward to t. Writes length+1 node ids and, when
// out_edges is non-null, `length` relationship ids. Both loops are bounded by
// the recorded distances, which are <= max_hops.
inline void csr_sp_build_witness(const CsrShortestScratch* sc, int64_t meet,
                                 int64_t* BOLT_RESTRICT out_nodes,
                                 int64_t* BOLT_RESTRICT out_edges) noexcept {
    assert(sc != nullptr && out_nodes != nullptr && meet >= 0);
    assert(sc->parent_f != nullptr && sc->parent_r != nullptr);
    const int32_t df = sc->dist_f[meet];
    const int32_t dr = sc->dist_r[meet];
    assert(df >= 0 && dr >= 0);
    int64_t cur = meet;
    for (int32_t i = df; i > 0; --i) {          // prefix: meet -> ... -> s
        out_nodes[i] = cur;
        if (out_edges != nullptr) out_edges[i - 1] = sc->pedge_f[cur];
        cur = sc->parent_f[cur];
    }
    out_nodes[0] = cur;                          // == s
    cur = meet;
    for (int32_t i = 0; i < dr; ++i) {           // suffix: meet -> ... -> t
        if (out_edges != nullptr) out_edges[df + i] = sc->pedge_r[cur];
        cur = sc->parent_r[cur];
        out_nodes[df + 1 + i] = cur;
    }
}

// Contract validation. Fails closed on every mixed/absent buffer combination.
BOLT_FORCE_INLINE bool csr_shortest_args_ok(const CsrBfsGraph* fwd,
                                            const CsrBfsGraph* rev,
                                            int64_t s, int64_t t,
                                            int32_t max_hops,
                                            const CsrShortestScratch* sc) noexcept {
    assert(max_hops >= 0);
    assert(sc != nullptr || fwd == nullptr);
    if (fwd == nullptr || rev == nullptr || sc == nullptr) return false;
    if (fwd->off == nullptr || rev->off == nullptr) return false;
    if (fwd->n_nodes <= 0 || fwd->n_nodes != rev->n_nodes) return false;
    if (s < 0 || s >= fwd->n_nodes || t < 0 || t >= fwd->n_nodes) return false;
    if (max_hops < 0 || max_hops > k_csr_shortest_max_hops) return false;
    if (sc->stamp_f == nullptr || sc->stamp_r == nullptr ||
        sc->dist_f == nullptr || sc->dist_r == nullptr ||
        sc->queue_f == nullptr || sc->queue_r == nullptr)
        return false;
    const bool want_witness = (sc->parent_f != nullptr);
    if (want_witness != (sc->parent_r != nullptr)) return false;
    if (want_witness != (sc->pedge_f != nullptr)) return false;
    if (want_witness != (sc->pedge_r != nullptr)) return false;
    return true;
}

// Run the alternating expansion until a meeting, an empty frontier, or the cap.
// Returns true when `mu` is PROVEN minimal (including mu == INT32_MAX meaning
// proven unreachable); false means the hop cap stopped the search first.
// The stop-at-first-meeting rule is exact — see the CLAIM at the top of this
// header for the proof and the differential that backs it.
inline bool csr_sp_search(CsrSpSide* f, CsrSpSide* r, int32_t epoch,
                          int64_t n_nodes, int32_t max_hops,
                          int32_t* BOLT_RESTRICT mu,
                          int64_t* BOLT_RESTRICT meet) noexcept {
    assert(f != nullptr && r != nullptr && mu != nullptr && meet != nullptr);
    assert(max_hops >= 0 && max_hops <= k_csr_shortest_max_hops);
    const int64_t iter_cap = 2LL * max_hops + 4;
    for (int64_t it = 0; it < iter_cap; ++it) {
        const int64_t fn = f->lvl_end - f->lvl_begin;
        const int64_t rn = r->lvl_end - r->lvl_begin;
        if (fn <= 0 || rn <= 0) return true;            // component exhausted
        if (*mu != INT32_MAX) return true;              // first meeting is exact
        if (f->depth + r->depth >= max_hops) return false;   // cap
        if (fn <= rn) csr_sp_expand_level(f, r, epoch, n_nodes, mu, meet);
        else          csr_sp_expand_level(r, f, epoch, n_nodes, mu, meet);
    }
    assert(false && "csr_shortest: alternation exceeded its bound");
    return false;
}

// ============================================================================
// csr_shortest_bidir
// ============================================================================
//
// Shortest UNWEIGHTED s->t path over `fwd` (out-edges) and `rev` (in-edges).
// `rev` must be the reverse CSR of the SAME edge set, carrying the SAME edge
// ids, so a witness path names real relationships. Both sides apply the same
// label filter (each graph carries its own want_label; pass the same value).
//
// Witness: pass out_path_nodes non-null with out_path_cap >= length + 1 and
// scratch parent_*/pedge_* non-null. out_path_edges may be null (nodes only).
// With no witness wanted, pass out_path_nodes null and all four parent/pedge
// scratch pointers null.
//
// Returns Ok with out->length == -1 for a PROVEN unreachable pair, and
// HopCapReached with out->length == -1 when max_hops stopped the search
// first. These are different answers; a caller that conflates them is wrong.
inline CsrShortestStatus csr_shortest_bidir(
        const CsrBfsGraph* BOLT_RESTRICT fwd,
        const CsrBfsGraph* BOLT_RESTRICT rev, int64_t s, int64_t t,
        int32_t max_hops, CsrShortestScratch* BOLT_RESTRICT sc,
        int64_t* BOLT_RESTRICT out_path_nodes,
        int64_t* BOLT_RESTRICT out_path_edges, int32_t out_path_cap,
        CsrShortestResult* BOLT_RESTRICT out) noexcept {
    assert(out != nullptr);
    assert(out_path_cap >= 0);
    if (out == nullptr) return CsrShortestStatus::InvalidArgument;
    out->length = -1;
    out->witness_len = 0;
    if (!csr_shortest_args_ok(fwd, rev, s, t, max_hops, sc) || out_path_cap < 0)
        return CsrShortestStatus::InvalidArgument;
    if (out_path_nodes != nullptr && sc->parent_f == nullptr)
        return CsrShortestStatus::InvalidArgument;

    const int64_t n_nodes = fwd->n_nodes;
    csr_sp_begin_epoch(sc, n_nodes);
    const int32_t epoch = sc->epoch;
    if (s == t) {                                  // zero-length path
        out->length = 0;
        if (out_path_nodes == nullptr) return CsrShortestStatus::Ok;
        if (out_path_cap < 1) return CsrShortestStatus::WitnessTruncated;
        out_path_nodes[0] = s;
        out->witness_len  = 1;
        return CsrShortestStatus::Ok;
    }

    CsrSpSide f{fwd, sc->stamp_f, sc->dist_f, sc->queue_f, sc->parent_f,
                sc->pedge_f, 0, 0, 0, 0, 0};
    CsrSpSide r{rev, sc->stamp_r, sc->dist_r, sc->queue_r, sc->parent_r,
                sc->pedge_r, 0, 0, 0, 0, 0};
    csr_sp_seed(&f, s, epoch);
    csr_sp_seed(&r, t, epoch);

    int32_t mu   = INT32_MAX;
    int64_t meet = -1;
    const bool proven = csr_sp_search(&f, &r, epoch, n_nodes, max_hops,
                                      &mu, &meet);
    // csr_sp_search only returns false via the hop cap, and it tests the proof
    // condition BEFORE the cap each round — so !proven means `mu` (found or
    // not) is undetermined, never merely unproven-but-right.
    if (!proven) return CsrShortestStatus::HopCapReached;
    if (mu == INT32_MAX) return CsrShortestStatus::Ok;   // proven unreachable
    assert(meet >= 0 && meet < n_nodes);
    assert(mu == sc->dist_f[meet] + sc->dist_r[meet]);
    out->length = mu;
    if (out_path_nodes == nullptr) return CsrShortestStatus::Ok;
    if (out_path_cap < mu + 1) return CsrShortestStatus::WitnessTruncated;
    csr_sp_build_witness(sc, meet, out_path_nodes, out_path_edges);
    out->witness_len = mu + 1;
    return CsrShortestStatus::Ok;
}

}  // namespace kernels
}  // namespace bolt
