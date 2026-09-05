// bolt_csr_bfs.h — variable-length CSR traversal kernels.
//
// Two kernels over the CSR built by bolt_csr.h's csr_build:
//   - csr_bfs_expand : variable-length expansion. For each source s, emit one
//                      (s, endpoint, hops) row per path of hop-length in
//                      [min_hops, max_hops] under an explicit path-semantics
//                      enum (WALK / TRAIL / SIMPLE). RESUMABLE mid-source.
//   - csr_bfs_count  : the aggregate-pushdown twin. Same traversal, same
//                      semantics, but counts endpoints per source WITHOUT
//                      materialising a single output row.
//
// These lift the semantics out of chukonu's hand-rolled
// src/operators/impl/bfs_expand_op.cpp (bfs_dfs + step_forbidden +
// materialize_src). The enum here is the single source of WALK/TRAIL/SIMPLE
// truth; the operator becomes morsel buffering + cursors only.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A DEPTH-FIRST WALK AND NOT A LEVEL-SYNCHRONOUS FRONTIER
// ---------------------------------------------------------------------------
// The kernel is named for the operator it serves (PhysicalBFSExpand), but the
// traversal is a bounded DEPTH-FIRST pre-order walk with an explicit stack.
// That is not an implementation accident, it is forced by the contract:
//   - TRAIL ("never repeat a relationship") and SIMPLE ("never repeat a node")
//     are PATH-DEPENDENT predicates. Whether an edge may be taken depends on
//     the particular path that reached the current node, not on the node. A
//     level-synchronous frontier holds a set of NODES, so it cannot evaluate
//     either predicate without materialising every path — which is the
//     memory blow-up this kernel family exists to avoid.
//   - Emission order is therefore DFS pre-order with neighbours visited in CSR
//     insertion order. This is BYTE-IDENTICAL to the order bfs_expand_op.cpp
//     produces today, which matters: a kernel swap that reorders rows silently
//     changes any LIMIT / first-N answer downstream.
// WALK semantics alone could use a frontier; shipping two traversals to save
// nothing on the shape that matters would be two perf surfaces and two
// correctness surfaces. One default (Tiger Style), documented.
//
// ---------------------------------------------------------------------------
// SHAPE RULES (Tiger Style, matching bolt_csr.h / bolt_scan.h)
// ---------------------------------------------------------------------------
//   - >=2 asserts/fn, <=70-line fns, noexcept, no RTTI, no exceptions
//   - NO heap allocation: the caller owns EVERY output and scratch buffer
//   - NO unbounded loop: every call is bounded by `visit_budget` (edges
//     examined + rows emitted) as well as by out_cap. A traversal that would
//     run for a second returns and is resumed, so a driver stays responsive.
//   - Per-source materialisation cap is a PARAMETER whose overflow is a
//     RETURN CODE (CsrBfsStatus::PerSourceCapExceeded), NEVER truncation. A
//     silently short answer under a success code is the failure mode this
//     whole family is built to make impossible.
//   - explicit integer sizes, BOLT_RESTRICT on disjoint pointer params
//
// ---------------------------------------------------------------------------
// ns/row budget — MEASURED, not estimated
// ---------------------------------------------------------------------------
// `test_bolt_csr_bfs.cpp`'s CsrBfsRandom.PerfSmokeNotGated prints the number
// on every run, so this comment cannot drift away from the code:
//
//   csr_bfs_expand, WALK [*1..2], 1,000 nodes / 10,000 edges, 200 sources,
//   22,647 emitted rows:  5.1 ns per EMITTED row
//   (Apple M4, clang -O3 -g, asserts LIVE — this tree does not define NDEBUG,
//   so a shipping -DNDEBUG build is faster than the figure above, not slower.)
//
// Per emitted row that is: one indexed neighbour load, the branch-free label
// keep, three stores, and the O(1) explicit-stack push/pop.
//   - SIMPLE / TRAIL add an O(depth) on-path scan per CANDIDATE edge (depth <=
//     k_csr_bfs_max_hops = 15, all L1-resident scratch). At the depth-2/3
//     shapes LSQB and the Cypher ladder actually use, that is <= 3 compares.
//   - csr_bfs_count runs the same traversal minus the three output stores. It
//     allocates and writes NOTHING per endpoint, which is the entire point: an
//     aggregate over a 10^10-endpoint expansion costs TIME, not MEMORY.
// The perf smoke is PRINTED and NOT GATED this wave (W1-L1); it is a floor to
// notice a 10x regression by, not a ratchet.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// Hard cap on traversal depth. Bounds the DFS stack and the on-path scratch,
// and therefore the SIMPLE/TRAIL per-edge scan. Mirrors chukonu's
// k_bfs_max_hops so an operator rewire cannot widen the contract silently.
constexpr int32_t k_csr_bfs_max_hops = 15;

// Path semantics. Identical meaning to chukonu's PathExpandPayload::semantics
// byte (0/1/2) so the operator can memcpy the value across.
//   Walk   — nothing forbidden. Cypher `-[*m..n]->` lowers here (this is what
//            LadybugDB 0.20.2 does, and the only choice under which
//            `-[*1..1]->` equals a plain `-[]->`: SIMPLE rejects a self-loop
//            because its one-hop endpoint IS its start node).
//   Trail  — never reuse a RELATIONSHIP already on the current path.
//   Simple — never revisit a NODE already on the current path.
enum class CsrPathSemantics : uint8_t {
    Walk   = 0,
    Trail  = 1,
    Simple = 2,
};

// Status. Ok covers every normal outcome INCLUDING suspension; use
// csr_bfs_done() to tell "this call filled up" from "the whole run finished".
enum class CsrBfsStatus : int32_t {
    Ok                   = 0,
    PerSourceCapExceeded = 1,   // loud refusal — never a short result
    InvalidArgument      = 2,
    // A neighbour id in the CSR is outside [0, n_nodes). Unlike csr_expand,
    // which only COPIES a neighbour to its output, this kernel DEREFERENCES it
    // (`off[nb]` to descend), so a malformed adjacency would be an
    // out-of-bounds READ rather than a bad output value. Checked, and refused
    // by name, on every descent.
    MalformedGraph       = 3,
};

// The CSR being walked, plus the relationship-type filter. `edge_labels` may
// be null (unlabeled CSR => every edge matches); `want_label < 0` is a
// wildcard. Predicate is csr_edge_label_keep() from bolt_csr.h — the SAME
// one csr_expand and chukonu's edge_label_keep use.
struct CsrBfsGraph {
    const int64_t* off;           // n_nodes + 1
    const int64_t* neighbors;     // n_edges
    const int64_t* edge_ids;      // n_edges
    const int32_t* edge_labels;   // n_edges, or null
    int64_t        n_nodes;
    int32_t        want_label;    // < 0 => wildcard
    int32_t        _pad;
};

struct CsrBfsParams {
    // Endpoints materialised (or counted) for ONE source before the kernel
    // gives up. Overflow is CsrBfsStatus::PerSourceCapExceeded, not a cut.
    int64_t          per_source_cap;
    // Work budget for ONE call: edges examined + rows emitted. Must be > 0.
    // This is what makes every call bounded even when out_cap is huge or the
    // traversal prunes for a long time without emitting.
    int64_t          visit_budget;
    int32_t          min_hops;    // >= 0; 0 emits the zero-length pair (s, s)
    int32_t          max_hops;    // >= min_hops, <= k_csr_bfs_max_hops
    CsrPathSemantics semantics;
    uint8_t          _pad[7];
};

// Caller-owned DFS scratch. Each array needs csr_bfs_scratch_slots(max_hops)
// int64 slots. The KERNEL owns the layout (what each array means and how it
// is indexed); the CALLER owns the memory (arena, stack, wherever).
struct CsrBfsScratch {
    int64_t* path;    // path[0..depth]   — nodes on the current path
    int64_t* epath;   // epath[0..depth-1]— relationship ids on the current path
    int64_t* iter;    // iter[d]          — next CSR index to try at depth d
};

constexpr int64_t csr_bfs_scratch_slots(int32_t max_hops) noexcept {
    return static_cast<int64_t>(max_hops) + 1;
}

// Resume point. Caller ZERO-INITIALISES before the first call and then passes
// the same cursor back unchanged. `active == 0` means "start the source at
// src_index"; a zeroed cursor therefore starts cleanly at source 0.
struct CsrBfsCursor {
    int64_t src_index;         // next/current index into src_ids[]
    int64_t emitted_for_src;   // endpoints produced for the current source
    int32_t depth;             // current DFS depth (valid iff active)
    uint8_t active;            // 1 => mid-source, the stack in scratch is live
    uint8_t pending_emit;      // 1 => path[depth] still owes an output row
    uint8_t _pad[2];
};

// The whole run is finished when this is true on return.
BOLT_FORCE_INLINE bool csr_bfs_done(const CsrBfsCursor* cursor,
                                    int64_t n) noexcept {
    assert(cursor != nullptr);
    assert(n >= 0);
    return cursor->src_index >= n && cursor->active == 0;
}

// ============================================================================
// internals
// ============================================================================

// Would extending the current path (nodes path[0..depth]) by relationship
// `edge_id` to node `nb` violate the semantics? WALK forbids nothing; TRAIL
// scans the depth relationship ids already on the path; SIMPLE scans the
// depth+1 nodes. Both loops are bounded by k_csr_bfs_max_hops.
BOLT_FORCE_INLINE bool csr_bfs_step_forbidden(
        const CsrBfsScratch* BOLT_RESTRICT sc, CsrPathSemantics sem,
        int64_t edge_id, int64_t nb, int32_t depth) noexcept {
    assert(sc != nullptr && sc->path != nullptr && sc->epath != nullptr);
    assert(depth >= 0 && depth <= k_csr_bfs_max_hops);
    if (sem == CsrPathSemantics::Walk) return false;
    if (sem == CsrPathSemantics::Trail) {
        for (int32_t k = 0; k < depth; ++k)
            if (sc->epath[k] == edge_id) return true;
        return false;
    }
    assert(sem == CsrPathSemantics::Simple);
    for (int32_t k = 0; k <= depth; ++k)
        if (sc->path[k] == nb) return true;
    return false;
}

// Sentinels returned by csr_bfs_next_edge (never valid CSR indices).
constexpr int64_t k_csr_bfs_edge_exhausted = -1;   // block finished
constexpr int64_t k_csr_bfs_edge_budget    = -2;   // work budget spent

// Next admissible CSR edge index out of path[depth], starting at iter[depth].
// Skips label-rejected and semantics-forbidden edges, charging one unit of
// budget per edge examined, and PARKS iter[depth] at the exact resume point
// in every exit path. Returns the CSR index, or a sentinel above.
BOLT_FORCE_INLINE int64_t csr_bfs_next_edge(
        const CsrBfsGraph* BOLT_RESTRICT g, CsrBfsScratch* BOLT_RESTRICT sc,
        CsrPathSemantics sem, int32_t depth,
        int64_t* BOLT_RESTRICT budget) noexcept {
    assert(g != nullptr && sc != nullptr && budget != nullptr);
    assert(depth >= 0 && depth <= k_csr_bfs_max_hops);
    const int64_t node = sc->path[depth];
    assert(node >= 0 && node < g->n_nodes && "csr_bfs: node id out of CSR range");
    const int64_t end = g->off[node + 1];
    int64_t j = sc->iter[depth];
    assert(j >= g->off[node] && j <= end && "csr_bfs: iter left its block");
    while (j < end) {
        if (*budget <= 0) { sc->iter[depth] = j; return k_csr_bfs_edge_budget; }
        --(*budget);
        if (csr_edge_label_keep(g->edge_labels, g->want_label, j) != 0 &&
            !csr_bfs_step_forbidden(sc, sem, g->edge_ids[j],
                                    g->neighbors[j], depth)) {
            sc->iter[depth] = j;
            return j;
        }
        ++j;
    }
    sc->iter[depth] = end;
    return k_csr_bfs_edge_exhausted;
}

// Outcome of one source's (partial) traversal.
enum class CsrBfsStep : int32_t {
    SourceDone = 0, Suspended = 1, CapHit = 2, BadGraph = 3
};

// Drive the explicit-stack DFS for the CURRENT source until it finishes, the
// output fills, the budget runs out, or the per-source cap is hit. All state
// lives in `cursor` + `sc`, so any exit is resumable by calling again.
// out_src / out_dst / out_hops may each be null (csr_bfs_count passes all
// three null and an out_cap of INT64_MAX).
//
// FORCE_INLINE, like every other kernel in this family, and it is load-bearing
// rather than stylistic. Left as a plain `inline` the compiler emits it
// out-of-line: ten pointer parameters, so `w`, `budget` and every `cursor`
// field stay in MEMORY across the innermost loop and the caller's constant
// `out_hops == nullptr` cannot fold away its per-row branch and store.
// Measured through chukonu's PhysicalBFSExpand on LSQB SF1
// (`MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN count(*)`, 11,806,176
// emitted rows, min-of-5, order-alternated): out-of-line 36.3-42.9 ms vs
// 30.3-30.5 ms inlined, against a 30.5-31.2 ms hand-rolled baseline — i.e.
// out-of-line is a 1.2-1.4x LOSS and inlined is parity. Two copies of the
// body (csr_bfs_expand + csr_bfs_count) is the price.
BOLT_FORCE_INLINE CsrBfsStep csr_bfs_run_source(
        const CsrBfsGraph* BOLT_RESTRICT g, CsrBfsScratch* BOLT_RESTRICT sc,
        const CsrBfsParams* BOLT_RESTRICT p, CsrBfsCursor* BOLT_RESTRICT cursor,
        int64_t* BOLT_RESTRICT out_src, int64_t* BOLT_RESTRICT out_dst,
        int32_t* BOLT_RESTRICT out_hops, int64_t out_cap,
        int64_t* BOLT_RESTRICT w, int64_t* BOLT_RESTRICT budget) noexcept {
    assert(g != nullptr && sc != nullptr && p != nullptr && cursor != nullptr);
    assert(w != nullptr && budget != nullptr && *w >= 0);
    while (cursor->depth >= 0) {
        assert(cursor->depth <= p->max_hops);
        if (cursor->pending_emit != 0) {
            if (*w >= out_cap) return CsrBfsStep::Suspended;
            if (cursor->emitted_for_src >= p->per_source_cap)
                return CsrBfsStep::CapHit;
            if (out_src  != nullptr) out_src[*w]  = sc->path[0];
            if (out_dst  != nullptr) out_dst[*w]  = sc->path[cursor->depth];
            if (out_hops != nullptr) out_hops[*w] = cursor->depth;
            ++(*w);
            ++cursor->emitted_for_src;
            cursor->pending_emit = 0;
            if (--(*budget) <= 0) return CsrBfsStep::Suspended;
        }
        if (cursor->depth >= p->max_hops) { --cursor->depth; continue; }
        const int64_t j = csr_bfs_next_edge(g, sc, p->semantics,
                                            cursor->depth, budget);
        if (j == k_csr_bfs_edge_budget) return CsrBfsStep::Suspended;
        if (j == k_csr_bfs_edge_exhausted) { --cursor->depth; continue; }
        assert(j >= 0);
        const int64_t nb = g->neighbors[j];
        // Fail closed BEFORE dereferencing off[nb]: a neighbour outside the
        // dense node range is a malformed CSR, and the next line would read
        // out of bounds in a -DNDEBUG build where the assert is gone.
        if (nb < 0 || nb >= g->n_nodes) return CsrBfsStep::BadGraph;
        sc->iter[cursor->depth]  = j + 1;          // resume AFTER this edge
        sc->epath[cursor->depth] = g->edge_ids[j];
        ++cursor->depth;
        sc->path[cursor->depth] = nb;
        sc->iter[cursor->depth] = g->off[nb];
        cursor->pending_emit =
            (cursor->depth >= p->min_hops) ? uint8_t{1} : uint8_t{0};
    }
    assert(cursor->depth == -1);
    return CsrBfsStep::SourceDone;
}

// Contract validation shared by both public kernels. Cheap, always on.
BOLT_FORCE_INLINE bool csr_bfs_args_ok(
        const int64_t* src_ids, int64_t n, const CsrBfsGraph* g,
        const CsrBfsParams* p, const CsrBfsScratch* sc,
        const CsrBfsCursor* cursor) noexcept {
    assert(n >= 0);
    assert(g != nullptr || n == 0);
    if (n < 0 || g == nullptr || p == nullptr || sc == nullptr ||
        cursor == nullptr)
        return false;
    if (n > 0 && src_ids == nullptr) return false;
    if (g->off == nullptr || g->n_nodes <= 0) return false;
    if (sc->path == nullptr || sc->epath == nullptr || sc->iter == nullptr)
        return false;
    if (p->min_hops < 0 || p->max_hops < p->min_hops ||
        p->max_hops > k_csr_bfs_max_hops)
        return false;
    if (p->per_source_cap <= 0 || p->visit_budget <= 0) return false;
    return true;
}

// Begin the source at cursor->src_index. Returns false for a source id that
// is not a dense CSR node (fail closed rather than read out of bounds).
BOLT_FORCE_INLINE bool csr_bfs_start_source(
        const int64_t* BOLT_RESTRICT src_ids, const CsrBfsGraph* g,
        const CsrBfsParams* p, CsrBfsScratch* sc,
        CsrBfsCursor* cursor) noexcept {
    assert(src_ids != nullptr && cursor != nullptr && cursor->active == 0);
    assert(g != nullptr && p != nullptr && sc != nullptr);
    const int64_t s = src_ids[cursor->src_index];
    if (s < 0 || s >= g->n_nodes) return false;
    sc->path[0]             = s;
    sc->iter[0]             = g->off[s];
    cursor->depth           = 0;
    cursor->emitted_for_src = 0;
    cursor->pending_emit    = (p->min_hops == 0) ? uint8_t{1} : uint8_t{0};
    cursor->active          = 1;
    return true;
}

// ============================================================================
// csr_bfs_expand
// ============================================================================
//
// For each s in src_ids[0..n), emit one row per path from s of hop-length in
// [min_hops, max_hops] under `semantics`. Columns are written to whichever of
// out_src / out_dst / out_hops are non-null, at most out_cap rows per call.
//
// RESUMABILITY CONTRACT:
//   - Caller owns a CsrBfsCursor, zeroed before the first call, and scratch of
//     csr_bfs_scratch_slots(max_hops) int64 per array. Both must be passed
//     back UNCHANGED and must outlive the run — the DFS stack lives there.
//   - Call repeatedly with the same arguments. Each call writes up to out_cap
//     rows and parks the exact resume point (mid-source, mid-block, even
//     mid-emit). The run is finished when csr_bfs_done(cursor, n) is true.
//   - out_cap == 0 is a valid no-op poll: 0 rows, no cursor movement.
//   - *out_rows is always written (0 on every error path too).
// FAILURE IS LOUD, AND THERE ARE THREE KINDS, EACH NAMED:
//   InvalidArgument  — the contract was broken (bad hop window, zero cap, a
//                      source id outside the dense node range).
//   MalformedGraph   — a NEIGHBOUR id left the dense node range. Note this is
//                      unavoidable by asking for less: even `[*1..1]` pushes
//                      the neighbour onto the path, so no hop window reaches a
//                      bad neighbour without dereferencing it.
//   PerSourceCapExceeded — see below.
// PerSourceCapExceeded means one source's expansion needed
// more than per_source_cap endpoints. Rows already written this call are
// reported in *out_rows but the RESULT IS INCOMPLETE — the caller must fail
// the query, never ship the prefix.
inline CsrBfsStatus csr_bfs_expand(
        const int64_t* BOLT_RESTRICT src_ids, int64_t n,
        const CsrBfsGraph* BOLT_RESTRICT g,
        const CsrBfsParams* BOLT_RESTRICT p, CsrBfsScratch* BOLT_RESTRICT sc,
        int64_t* BOLT_RESTRICT out_src, int64_t* BOLT_RESTRICT out_dst,
        int32_t* BOLT_RESTRICT out_hops, int64_t out_cap,
        CsrBfsCursor* BOLT_RESTRICT cursor,
        int64_t* BOLT_RESTRICT out_rows) noexcept {
    assert(out_rows != nullptr);
    assert(out_cap >= 0);
    if (out_rows == nullptr) return CsrBfsStatus::InvalidArgument;
    *out_rows = 0;
    if (!csr_bfs_args_ok(src_ids, n, g, p, sc, cursor) || out_cap < 0)
        return CsrBfsStatus::InvalidArgument;
    // out_cap == 0 is a no-op poll and returns BEFORE touching the cursor, so
    // a zero-capacity call can never burn budget on work it cannot emit.
    if (out_cap == 0) return CsrBfsStatus::Ok;

    int64_t w      = 0;
    int64_t budget = p->visit_budget;
    while (cursor->src_index < n) {
        if (cursor->active == 0 &&
            !csr_bfs_start_source(src_ids, g, p, sc, cursor)) {
            *out_rows = w;
            return CsrBfsStatus::InvalidArgument;
        }
        const CsrBfsStep step = csr_bfs_run_source(
            g, sc, p, cursor, out_src, out_dst, out_hops, out_cap, &w, &budget);
        if (step == CsrBfsStep::CapHit) {
            *out_rows = w;
            return CsrBfsStatus::PerSourceCapExceeded;
        }
        if (step == CsrBfsStep::BadGraph) {
            *out_rows = w;
            return CsrBfsStatus::MalformedGraph;
        }
        if (step == CsrBfsStep::Suspended) break;
        cursor->active = 0;
        ++cursor->src_index;
        if (budget <= 0) break;
    }
    assert(w <= out_cap);
    *out_rows = w;
    return CsrBfsStatus::Ok;
}

// ============================================================================
// csr_bfs_count — the aggregate-pushdown twin
// ============================================================================
//
// Same traversal, same semantics, same cursor discipline — but nothing is
// materialised. out_counts[i] receives the endpoint count for src_ids[i] and
// is written ONLY when source i completes, so a partially-traversed source
// never exposes a misleadingly small number: until csr_bfs_done() is true,
// out_counts[cursor->src_index] is UNWRITTEN, not partial. The running
// partial lives in cursor->emitted_for_src.
//
// Caller owns out_counts[0..n) and need not pre-zero it (every completed
// source is assigned, not accumulated). per_source_cap still applies: set it
// to INT64_MAX when an aggregate genuinely wants an unbounded count.
inline CsrBfsStatus csr_bfs_count(
        const int64_t* BOLT_RESTRICT src_ids, int64_t n,
        const CsrBfsGraph* BOLT_RESTRICT g,
        const CsrBfsParams* BOLT_RESTRICT p, CsrBfsScratch* BOLT_RESTRICT sc,
        int64_t* BOLT_RESTRICT out_counts,
        CsrBfsCursor* BOLT_RESTRICT cursor) noexcept {
    assert(out_counts != nullptr || n == 0);
    assert(cursor != nullptr);
    if (!csr_bfs_args_ok(src_ids, n, g, p, sc, cursor))
        return CsrBfsStatus::InvalidArgument;
    if (n > 0 && out_counts == nullptr) return CsrBfsStatus::InvalidArgument;

    int64_t sink   = 0;                       // total emitted this call (unused)
    int64_t budget = p->visit_budget;
    while (cursor->src_index < n) {
        if (cursor->active == 0 &&
            !csr_bfs_start_source(src_ids, g, p, sc, cursor))
            return CsrBfsStatus::InvalidArgument;
        const CsrBfsStep step = csr_bfs_run_source(
            g, sc, p, cursor, nullptr, nullptr, nullptr, INT64_MAX,
            &sink, &budget);
        if (step == CsrBfsStep::CapHit) return CsrBfsStatus::PerSourceCapExceeded;
        if (step == CsrBfsStep::BadGraph) return CsrBfsStatus::MalformedGraph;
        if (step == CsrBfsStep::Suspended) break;
        out_counts[cursor->src_index] = cursor->emitted_for_src;
        cursor->active = 0;
        ++cursor->src_index;
        if (budget <= 0) break;
    }
    assert(sink >= 0);
    return CsrBfsStatus::Ok;
}

}  // namespace kernels
}  // namespace bolt
