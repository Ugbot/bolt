// bolt_csr_levels.h — single-source hop distances (level-synchronous BFS).
//
//   csr_bfs_levels : out_depth[v] = the fewest kept edges on a walk from
//                    `source` to v, or -1 when v is unreached (or deeper than
//                    max_depth). out_depth[source] = 0. Direction is the
//                    view's: pass the forward CSR for out-edges, the reverse
//                    one for in-edges, an undirected one for both.
//
// Unlike csr_bfs_expand's Reach semantics this reports each node's LEVEL, and
// unlike Dijkstra it needs no weight column and no heap: one FIFO of n_nodes
// slots, each node enqueued at most once. O(n_nodes + num_edges).
//
// Scratch: csr_bfs_levels_scratch_bytes(n) — the queue.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

constexpr uint64_t csr_bfs_levels_scratch_bytes(int64_t n_nodes) noexcept {
    return static_cast<uint64_t>(n_nodes) * sizeof(int64_t);
}

// max_depth < 0 = unbounded. *out_reached = nodes with a depth (source incl.).
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_bfs_levels_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                      int64_t source, int32_t max_depth,
                                      int64_t* BOLT_RESTRICT out_depth,
                                      int64_t* BOLT_RESTRICT queue,
                                      int64_t* BOLT_RESTRICT out_reached) noexcept {
    assert(out_reached != nullptr);
    assert(out_depth != queue || out_depth == nullptr);
    if (out_reached == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_reached = 0;
    if (!csr_algo_graph_ok(g) || out_depth == nullptr || queue == nullptr)
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n = g->n_nodes;
    if (source < 0 || source >= n) return CsrAlgoStatus::InvalidArgument;
    for (int64_t i = 0; i < n; ++i) out_depth[i] = -1;
    out_depth[source] = 0;
    queue[0] = source;
    int64_t head = 0;
    int64_t tail = 1;
    while (head < tail) {                         // each node enqueued once
        const int64_t u = queue[head++];
        const int64_t du = out_depth[u];
        if (max_depth >= 0 && du >= max_depth) continue;
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        for (int64_t j = b; j < e; ++j) {
            if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0) continue;
            const int64_t v = static_cast<int64_t>(g->neighbors[j]);
            if (v < 0 || v >= n) return CsrAlgoStatus::MalformedGraph;
            if (out_depth[v] != -1) continue;
            out_depth[v] = du + 1;
            assert(tail < n);
            queue[tail++] = v;
        }
    }
    assert(tail >= 1 && tail <= n);
    *out_reached = tail;
    return CsrAlgoStatus::Ok;
}

#define BOLT_CSR_LEVELS_ENTRY(G)                                               \
    inline CsrAlgoStatus csr_bfs_levels(const G* BOLT_RESTRICT g,              \
                                        int64_t source, int32_t max_depth,     \
                                        int64_t* BOLT_RESTRICT out_depth,      \
                                        int64_t* BOLT_RESTRICT queue,          \
                                        int64_t* BOLT_RESTRICT out_reached) noexcept { \
        return csr_bfs_levels_t(g, source, max_depth, out_depth, queue,        \
                                out_reached);                                  \
    }
BOLT_CSR_LEVELS_ENTRY(CsrGraph64)
BOLT_CSR_LEVELS_ENTRY(CsrGraph32)
#undef BOLT_CSR_LEVELS_ENTRY

}  // namespace kernels
}  // namespace bolt
