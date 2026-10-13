// Opt-in indexed-property Dijkstra. The contiguous hot path remains in
// bolt_csr_dijkstra.h with its original restrict-pointer contract.
// This variant trades random property reads for avoiding an O(edges) gather;
// it is not the full-traversal default (see graph-weight-kernel audit).
#pragma once

#include "bolt/kernels/bolt_csr_dijkstra.h"
#include "bolt/kernels/bolt_csr_weights.h"

namespace bolt::kernels {

template <typename Nbr, typename Eid, typename Value>
inline CsrAlgoStatus csr_dj_relax_indexed(
        const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
        const CsrIndexedWeights<Value, Eid>& weights, int64_t u,
        double* BOLT_RESTRICT dist, int64_t* BOLT_RESTRICT parent,
        int64_t* BOLT_RESTRICT parent_slot, CsrDijkstraHeap* BOLT_RESTRICT h) noexcept {
    assert(g != nullptr && dist != nullptr && h != nullptr);
    assert(u >= 0 && u < g->n_nodes);
    int64_t begin = 0, end = 0;
    csr_graph_block(g, u, &begin, &end);
    for (int64_t slot = begin; slot < end; ++slot) {
        if (csr_edge_label_keep(g->edge_labels, g->want_label, slot) == 0) continue;
        const int64_t v = static_cast<int64_t>(g->neighbors[slot]);
        if (v < 0 || v >= g->n_nodes) return CsrAlgoStatus::MalformedGraph;
        double weight = 0;
        const auto read = csr_dj_weight(weights, slot, &weight);
        if (read != CsrAlgoStatus::Ok) return read;
        if (!(weight >= 0)) return CsrAlgoStatus::NegativeWeight;
        if (h->pos[v] == -2) continue;
        const double next = dist[u] + weight;
        if (!(next < dist[v])) continue;
        dist[v] = next;
        if (parent != nullptr) { parent[v] = u; parent_slot[v] = slot; }
        if (h->pos[v] == -1) {
            h->pos[v] = h->size;
            h->heap[h->size++] = v;
        }
        csr_dj_sift_up(h, h->pos[v]);
    }
    return CsrAlgoStatus::Ok;
}

// Same output/scratch contract and reached-edge validation as contiguous
// Dijkstra. Caller owns immutable property storage for the search lifetime.
// No type dispatch, heap allocation or materialization inside the search.
template <typename Nbr, typename Eid, typename Value>
inline CsrAlgoStatus csr_dijkstra_indexed(
        const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
        const CsrIndexedWeights<Value, Eid>& weights,
        int64_t source, int64_t target, double* BOLT_RESTRICT dist,
        int64_t* BOLT_RESTRICT parent, int64_t* BOLT_RESTRICT parent_slot,
        int64_t* BOLT_RESTRICT scratch,
        int64_t* BOLT_RESTRICT out_settled) noexcept {
    assert(out_settled != nullptr);
    assert((parent == nullptr) == (parent_slot == nullptr));
    if (out_settled == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_settled = 0;
    if (!csr_algo_graph_ok(g) || dist == nullptr || scratch == nullptr ||
        !csr_dj_weights_ok(weights, csr_algo_num_edges(g)) ||
        (parent == nullptr) != (parent_slot == nullptr)) return CsrAlgoStatus::InvalidArgument;
    const int64_t n = g->n_nodes;
    if (source < 0 || source >= n || target >= n) return CsrAlgoStatus::InvalidArgument;
    CsrDijkstraHeap h{scratch, scratch + n, dist, 0};
    for (int64_t i = 0; i < n; ++i) {
        dist[i] = std::numeric_limits<double>::infinity();
        h.pos[i] = -1;
        if (parent != nullptr) { parent[i] = -1; parent_slot[i] = -1; }
    }
    dist[source] = 0;
    h.pos[source] = 0;
    h.heap[h.size++] = source;
    int64_t settled = 0;
    while (h.size > 0) { // Each node settles at most once.
        const int64_t u = csr_dj_pop(&h);
        ++settled;
        assert(settled <= n);
        if (u == target) break;
        const auto status = csr_dj_relax_indexed(g, weights, u, dist, parent, parent_slot, &h);
        if (status != CsrAlgoStatus::Ok) { *out_settled = settled; return status; }
    }
    *out_settled = settled;
    return CsrAlgoStatus::Ok;
}

} // namespace bolt::kernels
