// bolt_csr_dijkstra.h — single-source weighted shortest paths over a CSR.
//
// Dijkstra with an INDEXED binary heap (decrease-key in place), so the heap
// never holds more than n_nodes entries and needs no growth: heap[n] + pos[n]
// of caller scratch. Weights are a per-edge column indexed by CSR slot
// (parallel to `neighbors`), which is how a relationship property column is
// laid out against the CSR. A negative or NaN weight on an edge the search
// reaches is refused (NegativeWeight) — Dijkstra would silently return wrong
// distances, and this tree refuses rather than answers wrongly.
//
// dist[v] = +inf for an unreached node. parent / parent_slot (optional)
// record one shortest-path tree; parent_slot is the CSR slot of the tree
// edge so the caller can name the relationship. target >= 0 stops the search
// once the target is settled (its distance is then final; others may not
// be); target < 0 settles every reachable node.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace bolt {
namespace kernels {

constexpr uint64_t csr_dijkstra_scratch_bytes(int64_t n_nodes) noexcept {
    return static_cast<uint64_t>(n_nodes) * 2u * sizeof(int64_t);
}

// pos[v]: -1 never queued, -2 settled, else the heap index.
struct CsrDijkstraHeap {
    int64_t*      heap;
    int64_t*      pos;
    const double* dist;
    int64_t       size;
};

BOLT_FORCE_INLINE void csr_dj_place(CsrDijkstraHeap* h, int64_t i,
                                    int64_t v) noexcept {
    assert(h != nullptr && i >= 0 && i < h->size);
    assert(v >= 0);
    h->heap[i] = v;
    h->pos[v]  = i;
}

// Strict order: distance, then node id, so ties resolve deterministically.
BOLT_FORCE_INLINE bool csr_dj_less(const CsrDijkstraHeap* h, int64_t a,
                                   int64_t b) noexcept {
    assert(h != nullptr && a >= 0 && b >= 0);
    const double da = h->dist[a];
    const double db = h->dist[b];
    return da < db || (da == db && a < b);
}

inline void csr_dj_sift_up(CsrDijkstraHeap* h, int64_t i) noexcept {
    assert(h != nullptr && i >= 0 && i < h->size);
    const int64_t v = h->heap[i];
    while (i > 0) {                              // bounded by log2(size)
        const int64_t p = (i - 1) / 2;
        if (!csr_dj_less(h, v, h->heap[p])) break;
        csr_dj_place(h, i, h->heap[p]);
        i = p;
    }
    csr_dj_place(h, i, v);
}

inline void csr_dj_sift_down(CsrDijkstraHeap* h, int64_t i) noexcept {
    assert(h != nullptr && i >= 0 && i < h->size);
    const int64_t v = h->heap[i];
    for (;;) {                                   // bounded by log2(size)
        const int64_t l = 2 * i + 1;
        if (l >= h->size) break;
        const int64_t r = l + 1;
        const int64_t c =
            (r < h->size && csr_dj_less(h, h->heap[r], h->heap[l])) ? r : l;
        if (!csr_dj_less(h, h->heap[c], v)) break;
        csr_dj_place(h, i, h->heap[c]);
        i = c;
    }
    csr_dj_place(h, i, v);
}

BOLT_FORCE_INLINE int64_t csr_dj_pop(CsrDijkstraHeap* h) noexcept {
    assert(h != nullptr && h->size > 0);
    const int64_t top = h->heap[0];
    --h->size;
    if (h->size > 0) {
        h->heap[0] = h->heap[h->size];
        h->pos[h->heap[0]] = 0;
        csr_dj_sift_down(h, 0);
    }
    h->pos[top] = -2;
    return top;
}

// Relax every kept out-edge of u. Returns Ok, NegativeWeight or MalformedGraph.
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_dj_relax(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                  const double* BOLT_RESTRICT w, int64_t u,
                                  double* BOLT_RESTRICT dist,
                                  int64_t* BOLT_RESTRICT parent,
                                  int64_t* BOLT_RESTRICT parent_slot,
                                  CsrDijkstraHeap* BOLT_RESTRICT h) noexcept {
    assert(g != nullptr && w != nullptr && dist != nullptr && h != nullptr);
    assert(u >= 0 && u < g->n_nodes);
    int64_t b = 0;
    int64_t e = 0;
    csr_graph_block(g, u, &b, &e);
    for (int64_t j = b; j < e; ++j) {
        if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0) continue;
        const int64_t v = static_cast<int64_t>(g->neighbors[j]);
        if (v < 0 || v >= g->n_nodes) return CsrAlgoStatus::MalformedGraph;
        const double wj = w[j];
        if (!(wj >= 0.0)) return CsrAlgoStatus::NegativeWeight;   // NaN too
        if (h->pos[v] == -2) continue;
        const double nd = dist[u] + wj;
        if (!(nd < dist[v])) continue;
        dist[v] = nd;
        if (parent != nullptr) { parent[v] = u; parent_slot[v] = j; }
        if (h->pos[v] == -1) {
            h->pos[v] = h->size;
            h->heap[h->size++] = v;
        }
        csr_dj_sift_up(h, h->pos[v]);
    }
    return CsrAlgoStatus::Ok;
}

template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_dijkstra_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                    const double* BOLT_RESTRICT weights,
                                    int64_t source, int64_t target,
                                    double* BOLT_RESTRICT dist,
                                    int64_t* BOLT_RESTRICT parent,
                                    int64_t* BOLT_RESTRICT parent_slot,
                                    int64_t* BOLT_RESTRICT scratch,
                                    int64_t* BOLT_RESTRICT out_settled) noexcept {
    assert(out_settled != nullptr);
    assert((parent == nullptr) == (parent_slot == nullptr));
    if (out_settled == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_settled = 0;
    if (!csr_algo_graph_ok(g) || dist == nullptr || scratch == nullptr ||
        (weights == nullptr && csr_algo_num_edges(g) > 0) ||
        (parent == nullptr) != (parent_slot == nullptr))
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n = g->n_nodes;
    if (source < 0 || source >= n || target >= n) return CsrAlgoStatus::InvalidArgument;
    CsrDijkstraHeap h{scratch, scratch + n, dist, 0};
    for (int64_t i = 0; i < n; ++i) {
        dist[i]  = std::numeric_limits<double>::infinity();
        h.pos[i] = -1;
        if (parent != nullptr) { parent[i] = -1; parent_slot[i] = -1; }
    }
    dist[source] = 0.0;
    h.pos[source] = 0;
    h.heap[h.size++] = source;
    int64_t settled = 0;
    while (h.size > 0) {                          // each node settles once
        const int64_t u = csr_dj_pop(&h);
        ++settled;
        assert(settled <= n);
        if (u == target) break;
        const CsrAlgoStatus s = csr_dj_relax(g, weights, u, dist, parent,
                                             parent_slot, &h);
        if (s != CsrAlgoStatus::Ok) { *out_settled = settled; return s; }
    }
    *out_settled = settled;
    return CsrAlgoStatus::Ok;
}

#define BOLT_CSR_DIJKSTRA_ENTRY(G)                                             \
    inline CsrAlgoStatus csr_dijkstra(                                         \
            const G* BOLT_RESTRICT g, const double* BOLT_RESTRICT weights,     \
            int64_t source, int64_t target, double* BOLT_RESTRICT dist,        \
            int64_t* BOLT_RESTRICT parent, int64_t* BOLT_RESTRICT parent_slot, \
            int64_t* BOLT_RESTRICT scratch,                                    \
            int64_t* BOLT_RESTRICT out_settled) noexcept {                     \
        return csr_dijkstra_t(g, weights, source, target, dist, parent,        \
                              parent_slot, scratch, out_settled);              \
    }
BOLT_CSR_DIJKSTRA_ENTRY(CsrGraph64)
BOLT_CSR_DIJKSTRA_ENTRY(CsrGraph32)
#undef BOLT_CSR_DIJKSTRA_ENTRY

}  // namespace kernels
}  // namespace bolt
