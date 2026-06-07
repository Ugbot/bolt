// bolt_csr.h — CSR (compressed-sparse-row) graph primitives.
//
// Two graph kernels built on the scan/scatter substrate in bolt_scan.h:
//   - csr_build  : edge list (grouped by source) -> CSR offsets / neighbors
//                  / edge_ids, via degree-histogram -> exclusive-scan ->
//                  insertion-order scatter.
//   - csr_expand : 1-hop neighbour walk with a branch-free relationship-type
//                  (label) mask, emitting (src, edge, dst) rows into a
//                  bounded output buffer, RESUMABLE across calls.
//
// These mirror chukonu's hand-rolled loops (src/graph/csr_adjacency.cpp
// degree->scan->scatter, and src/operators/impl/csr_expand_op.cpp neighbour
// walk + branch-free `edge_label_keep` mask). Outputs are BYTE-IDENTICAL:
//   - csr_build emits neighbours in INSERTION ORDER within each source's
//     block (the per-node write cursor starts at offsets[k] and increments;
//     edge i is placed before edge j>i whenever src[i]==src[j]). Determin-
//     istic. Same as csr_adjacency.cpp Pass 3.
//   - csr_expand's label mask is the SAME predicate as edge_label_keep:
//     keep iff (want_label < 0)  // wildcard
//           || (edge_labels == nullptr)  // unlabeled CSR -> all match
//           || (edge_labels[j] == want_label). The write cursor advances by
//     the 0/1 keep mask (no data-dependent branch in the inner loop) — a
//     non-matching edge writes into the current slot but does not advance,
//     so it never reaches the output (identical to a residual filter drop).
//
// Shape rules (Tiger Style, matching bolt_binsearch.h / bolt_scan.h):
//   - >=2 asserts/fn, <=70-line fns (helpers split out), noexcept, no RTTI
//   - NO heap allocation: the caller owns EVERY output + scratch buffer
//   - BOLT_RESTRICT on disjoint pointer params, explicit int sizes
//   - branch-free inner loop in csr_expand (the label-mask advance)
//
// ns/row floors (scalar, /O2, L1/L2-resident, client Intel):
//   - csr_build  : ~2-3 ns/EDGE — two passes over the edge list (degree
//     histogram + scatter, each a random-access write keyed by node id) plus
//     one exclusive_scan over the node array. Bound by the scatter store
//     (random-access write into out_neighbors / out_edge_ids).
//   - csr_expand : ~1-1.5 ns/EMITTED-row in the steady state (one indexed
//     load of neighbors[j] / edge_ids[j] + the branch-free keep-advance +
//     three stores). Per source there is fixed O(1) block-bound setup; the
//     amortised floor is dominated by the per-neighbour inner loop.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_scan.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// ============================================================================
// csr_build
// ============================================================================
//
// Build CSR from an edge list grouped by source node. Caller owns ALL
// outputs AND scratch (no allocation here):
//   offsets[n_nodes + 1]   — node block boundaries (final result)
//   out_neighbors[n_edges] — neighbour (dst) per CSR slot
//   out_edge_ids[n_edges]  — edge id per CSR slot (== input index if edge_id
//                            is null)
//   scratch_nodes[n_nodes] — caller scratch: holds the degree histogram for
//                            the histogram+scan, then is re-seeded from
//                            offsets as the per-node insertion cursor.
// A separate scratch buffer (rather than overwriting offsets) keeps the
// final offsets intact while the scatter cursor advances — and keeps the
// kernel allocation-free. Returns false on a contract violation (null
// buffers with positive n_edges, or n_nodes <= 0); true on success.
//
// Determinism: insertion order within each node's block (edge i before
// edge j whenever i < j and src[i] == src[j]) — byte-identical to
// csr_adjacency.cpp Pass 3.

// Internal helper: degree histogram + exclusive scan into offsets. After
// this, offsets[k]..offsets[k+1] is node k's block. <=70 lines.
BOLT_FORCE_INLINE void csr_offsets_from_degrees(
        const int64_t* BOLT_RESTRICT src, int64_t n_edges, int64_t n_nodes,
        int64_t* BOLT_RESTRICT degrees, int64_t* BOLT_RESTRICT offsets) noexcept {
    assert(n_nodes > 0 && n_edges >= 0);
    assert(degrees != nullptr && offsets != nullptr);
    for (int64_t k = 0; k < n_nodes; ++k) degrees[k] = 0;
    histogram_i64(src, n_edges, n_nodes, degrees);     // Pass 1 — degree count
    exclusive_scan_i64(degrees, n_nodes, offsets);     // Pass 2 — prefix sum
    assert(offsets[0] == 0);
    assert(offsets[n_nodes] == n_edges && "csr_build: offsets total must == n_edges");
}

// Internal helper: scatter neighbours + edge_ids into CSR order using a
// per-node insertion cursor seeded from offsets. `cursor` is caller scratch
// of n_nodes slots. <=70 lines, branch-free body (label is not applied here
// — csr_build stores all edges; labels are filtered at expand time).
BOLT_FORCE_INLINE void csr_scatter_edges(
        const int64_t* BOLT_RESTRICT src, const int64_t* BOLT_RESTRICT dst,
        const int64_t* BOLT_RESTRICT edge_id, int64_t n_edges,
        const int64_t* BOLT_RESTRICT offsets, int64_t* BOLT_RESTRICT cursor,
        int64_t n_nodes, int64_t* BOLT_RESTRICT out_neighbors,
        int64_t* BOLT_RESTRICT out_edge_ids) noexcept {
    assert(n_nodes > 0 && n_edges >= 0);
    assert(offsets != nullptr && cursor != nullptr);
    for (int64_t k = 0; k < n_nodes; ++k) cursor[k] = offsets[k];
    for (int64_t i = 0; i < n_edges; ++i) {                // Pass 3 — scatter
        const int64_t k = src[i];
        assert(k >= 0 && k < n_nodes && "csr_build: non-dense source id");
        const int64_t w = cursor[k];
        assert(w >= offsets[k] && w < offsets[k + 1]);
        out_neighbors[w] = dst[i];
        out_edge_ids[w]  = (edge_id != nullptr) ? edge_id[i] : i;
        cursor[k] = w + 1;
    }
}

// Public csr_build. The single n_nodes-slot `scratch_nodes` buffer is reused
// for both the degree histogram and (after the exclusive scan) the per-node
// insertion cursor — the cursor is re-seeded from `offsets`, so reusing the
// degree array is safe. <=70 lines.
BOLT_FORCE_INLINE bool csr_build(
        const int64_t* BOLT_RESTRICT src, const int64_t* BOLT_RESTRICT dst,
        const int64_t* BOLT_RESTRICT edge_id, int64_t n_edges, int64_t n_nodes,
        int64_t* BOLT_RESTRICT offsets, int64_t* BOLT_RESTRICT out_neighbors,
        int64_t* BOLT_RESTRICT out_edge_ids,
        int64_t* BOLT_RESTRICT scratch_nodes) noexcept {
    assert(n_nodes > 0 && "csr_build: dense node table must be non-empty");
    assert(n_edges >= 0);
    if (n_nodes <= 0 || n_edges < 0) return false;
    if (offsets == nullptr || scratch_nodes == nullptr) return false;
    if (n_edges > 0 && (src == nullptr || dst == nullptr ||
                        out_neighbors == nullptr || out_edge_ids == nullptr))
        return false;

    // scratch_nodes holds degrees for the histogram+scan, then is re-seeded
    // as the per-node write cursor (from offsets) inside csr_scatter_edges.
    csr_offsets_from_degrees(src, n_edges, n_nodes, scratch_nodes, offsets);
    csr_scatter_edges(src, dst, edge_id, n_edges, offsets, scratch_nodes,
                      n_nodes, out_neighbors, out_edge_ids);
    assert(offsets[n_nodes] == n_edges);
    return true;
}

// ============================================================================
// csr_expand
// ============================================================================
//
// Branch-free relationship-type keep mask for CSR edge index `j`. Returns 1
// (keep) or 0 (drop). Identical predicate to chukonu's edge_label_keep:
// wildcard (want < 0) OR unlabeled CSR (edge_labels == nullptr) OR match.
// The two pointer/sign tests are loop-invariant (op-identity constant) and
// the compiler hoists them out of the per-edge loop, leaving a single
// compare in the body.
BOLT_FORCE_INLINE int64_t csr_edge_label_keep(
        const int32_t* BOLT_RESTRICT edge_labels, int32_t want_label,
        int64_t j) noexcept {
    assert(j >= 0);
    // Both inputs are loop-invariant; the per-edge cost is one compare. A
    // null edge_labels array means an unlabeled CSR (every edge matches).
    const bool wildcard = (want_label < 0) | (edge_labels == nullptr);
    assert((wildcard || edge_labels != nullptr) && "label probe needs an array");
    const int32_t have = (edge_labels != nullptr) ? edge_labels[j] : want_label;
    return static_cast<int64_t>(wildcard | (have == want_label));
}

// Cursor for resumable expansion. Two fields cover the exact resume point:
//   src_index   : next index into src_ids[] to process (outer loop bound)
//   neighbor_j  : the CSR edge index to resume at within the current source's
//                 block; 0 means "start at the block begin". This mirrors the
//                 chukonu op's (cur_in_row, cur_neighbor_j) pair.
// Caller zero-initialises before the first call; csr_expand updates it in
// place. Expansion is COMPLETE when cursor.src_index == n on return.
struct CsrExpandCursor {
    int64_t src_index;
    int64_t neighbor_j;
};

// Expand: for each src in src_ids[0..n), emit one (src, edge, dst) row per
// outgoing edge whose label matches want_label (want_label < 0 => wildcard).
// edge_labels may be null (unlabeled => all match). Writes into caller
// out_* buffers (capacity out_cap rows). Returns rows written this call.
//
// RESUMABILITY CONTRACT (what csr_expand_op.cpp uses):
//   - Caller owns a CsrExpandCursor, zeroed before the first call.
//   - Call repeatedly with the SAME src_ids / CSR / cursor. Each call fills
//     up to out_cap rows and advances the cursor to the exact resume point
//     (a high-degree source whose fan-out exceeds out_cap resumes mid-block
//     via cursor->neighbor_j; the next call continues from that edge).
//   - Expansion is finished when cursor->src_index == n on return.
//   - Returns the number of rows written this call (>=0, <= out_cap). It
//     never returns -1: out_cap exhaustion is the normal resume path, not an
//     error — the cursor simply isn't fully advanced. (The op slices nothing;
//     it just calls again.) out_cap == 0 is a valid no-op poll (returns 0).
//
// Branch-free inner loop: append at slot `w`, advance w += keep. <=70 lines.
BOLT_FORCE_INLINE int64_t csr_expand(
        const int64_t* BOLT_RESTRICT src_ids, int64_t n,
        const int64_t* BOLT_RESTRICT csr_off,
        const int64_t* BOLT_RESTRICT csr_neighbors,
        const int64_t* BOLT_RESTRICT csr_edge_ids,
        const int32_t* BOLT_RESTRICT edge_labels, int32_t want_label,
        int64_t* BOLT_RESTRICT out_src, int64_t* BOLT_RESTRICT out_edge,
        int64_t* BOLT_RESTRICT out_dst, int64_t out_cap,
        CsrExpandCursor* BOLT_RESTRICT cursor) noexcept {
    assert(cursor != nullptr && n >= 0 && out_cap >= 0);
    assert(n == 0 || (src_ids != nullptr && csr_off != nullptr));

    int64_t w = 0;                              // rows written this call
    while (cursor->src_index < n && w < out_cap) {
        const int64_t s = src_ids[cursor->src_index];
        assert(s >= 0 && "csr_expand: source id out of CSR range (non-dense?)");
        const int64_t begin = csr_off[s];
        const int64_t end   = csr_off[s + 1];
        assert(begin <= end);
        int64_t j = (cursor->neighbor_j > 0) ? cursor->neighbor_j : begin;
        assert(j >= begin && j <= end);
        for (; j < end && w < out_cap; ++j) {
            const int64_t keep =
                csr_edge_label_keep(edge_labels, want_label, j);
            out_src[w]  = s;                    // speculative write at slot w
            out_edge[w] = csr_edge_ids[j];
            out_dst[w]  = csr_neighbors[j];
            w += keep;                          // advance only on a match
        }
        if (j < end) {                          // out_cap hit mid-block: resume
            cursor->neighbor_j = j;
            break;
        }
        cursor->neighbor_j = 0;                 // block done -> next source
        ++cursor->src_index;
    }
    assert(w <= out_cap);
    return w;
}

}  // namespace kernels
}  // namespace bolt
