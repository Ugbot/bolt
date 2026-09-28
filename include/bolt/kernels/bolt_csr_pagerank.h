// bolt_csr_pagerank.h — PageRank by power iteration over a CSR.
//
// Semantics are networkx.pagerank's on a MultiDiGraph with unit weights (the
// oracle the tests use): uniform teleport, uniform redistribution of the
// mass of dangling nodes (no kept out-edge), parallel edges each carry an
// equal share of their source's rank, self-loops are ordinary edges. Stops
// when the L1 change of an iteration is below n_nodes * tol.
//
// Push formulation over the FORWARD CSR only: each iteration walks every
// block once, so one iteration is O(n_nodes + num_edges) with no reverse
// adjacency and no degree array. Ranks are double: at Pokec scale (1.6 M
// nodes) a float accumulator's rounding is the same order as tol * n, and
// the scratch is only 8 B/node more.
//
// Scratch: csr_pagerank_scratch_bytes(n) — one double per node.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace kernels {

constexpr uint64_t csr_pagerank_scratch_bytes(int64_t n_nodes) noexcept {
    return static_cast<uint64_t>(n_nodes) * sizeof(double);
}

// One push sweep: next[v] += x[u] / outdeg(u) for every kept edge u->v.
// Returns the rank mass of dangling nodes, or -1 on a malformed neighbour.
template <typename Nbr, typename Eid>
inline double csr_pagerank_push(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                const double* BOLT_RESTRICT x,
                                double* BOLT_RESTRICT next) noexcept {
    assert(g != nullptr && x != nullptr && next != nullptr);
    assert(g->n_nodes > 0);
    const int64_t n = g->n_nodes;
    for (int64_t i = 0; i < n; ++i) next[i] = 0.0;
    double dangling = 0.0;
    for (int64_t u = 0; u < n; ++u) {
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        int64_t kept = 0;
        for (int64_t j = b; j < e; ++j)
            kept += csr_edge_label_keep(g->edge_labels, g->want_label, j);
        if (kept == 0) { dangling += x[u]; continue; }
        const double share = x[u] / static_cast<double>(kept);
        for (int64_t j = b; j < e; ++j) {
            if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0)
                continue;
            const int64_t v = static_cast<int64_t>(g->neighbors[j]);
            if (v < 0 || v >= n) return -1.0;
            next[v] += share;
        }
    }
    assert(dangling >= 0.0);
    return dangling;
}

// rank[n_nodes] receives the result; scratch[n_nodes] is caller-owned.
// *out_iters = iterations run. NotConverged means max_iter was spent: rank
// then holds the last iterate, which is NOT the answer.
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_pagerank_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                    double damping, int32_t max_iter, double tol,
                                    double* BOLT_RESTRICT rank,
                                    double* BOLT_RESTRICT scratch,
                                    int32_t* BOLT_RESTRICT out_iters) noexcept {
    assert(out_iters != nullptr);
    assert(rank != scratch || rank == nullptr);
    if (out_iters == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_iters = 0;
    if (!csr_algo_graph_ok(g) || rank == nullptr || scratch == nullptr ||
        !(damping >= 0.0 && damping <= 1.0) || !(tol > 0.0) ||
        max_iter <= 0)
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n  = g->n_nodes;
    const double  dn = static_cast<double>(n);
    for (int64_t i = 0; i < n; ++i) rank[i] = 1.0 / dn;
    for (int32_t it = 0; it < max_iter; ++it) {
        const double dangling = csr_pagerank_push(g, rank, scratch);
        if (dangling < 0.0) return CsrAlgoStatus::MalformedGraph;
        const double base = (1.0 - damping) / dn + damping * dangling / dn;
        double err = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double nv = damping * scratch[i] + base;
            err += std::fabs(nv - rank[i]);
            rank[i] = nv;
        }
        *out_iters = it + 1;
        if (err < dn * tol) return CsrAlgoStatus::Ok;
    }
    return CsrAlgoStatus::NotConverged;
}

#define BOLT_CSR_PAGERANK_ENTRY(G)                                             \
    inline CsrAlgoStatus csr_pagerank(const G* BOLT_RESTRICT g, double damping,\
                                      int32_t max_iter, double tol,           \
                                      double* BOLT_RESTRICT rank,             \
                                      double* BOLT_RESTRICT scratch,          \
                                      int32_t* BOLT_RESTRICT out_iters) noexcept { \
        return csr_pagerank_t(g, damping, max_iter, tol, rank, scratch,        \
                              out_iters);                                      \
    }
BOLT_CSR_PAGERANK_ENTRY(CsrGraph64)
BOLT_CSR_PAGERANK_ENTRY(CsrGraph32)
#undef BOLT_CSR_PAGERANK_ENTRY

}  // namespace kernels
}  // namespace bolt
