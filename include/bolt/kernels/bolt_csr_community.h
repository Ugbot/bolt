// bolt_csr_community.h — community detection over a CSR.
//
//   csr_label_propagation : LDBC Graphalytics CDLP (community detection by
//       label propagation), the deterministic synchronous form.
//         * Every node starts with its own id as its label.
//         * Each iteration, every node takes the label that occurs most often
//           among its neighbours' labels of the PREVIOUS iteration; a tie
//           goes to the smallest label. All nodes update at once.
//         * A node with no kept neighbour keeps its label.
//         * Exactly max_iter iterations (Graphalytics fixes the count; there
//           is no convergence test). Once an iteration changes nothing every
//           later one is the identity, so the kernel stops early there and
//           the answer is the same.
//       Neighbours are a multiset gathered from up to two views: `a` and,
//       when non-null, `b`. Graphalytics' directed CDLP is a = forward,
//       b = reverse: a node counts the labels of its out- AND in-neighbours,
//       so a neighbour linked both ways counts twice, a parallel edge counts
//       once per edge and a self-loop counts the node's own label once per
//       direction. a = forward alone (or reverse alone) is the one-direction
//       variant. Both views must cover the same n_nodes.
//
//       MAGE's label propagation (community_detection_online, LabelRankT) is
//       a different, weighted and incremental algorithm; this kernel is the
//       Graphalytics one and does not claim MAGE parity.
//
// Scratch: csr_label_propagation_scratch_bytes(n, cap) — one int64 per node
// (the next-iteration labels) plus two int64 buffers of `cap` slots for the
// neighbour labels and their radix-sort ping-pong. cap must be at least
// csr_lp_buf_cap(a, b): the largest per-node block of a plus block of b.
// Every iteration is O(n + edges · radix passes); nothing is allocated.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"
#include "bolt/kernels/bolt_radixsort.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// The hard cap on CDLP iterations (Graphalytics runs ~10; this bounds a
// mistyped parameter, not a real workload).
constexpr int32_t k_csr_lp_max_iterations = 1000000;

constexpr uint64_t csr_label_propagation_scratch_bytes(int64_t n_nodes,
                                                       int64_t buf_cap) noexcept {
    return (static_cast<uint64_t>(n_nodes) + 2u * static_cast<uint64_t>(buf_cap)) *
           sizeof(int64_t);
}

struct CsrLpScratch {
    int64_t* next;     // n_nodes: labels of the iteration being computed
    int64_t* buf;      // buf_cap: one node's neighbour labels
    int64_t* tmp;      // buf_cap: radix-sort ping-pong
    int64_t  buf_cap;
};

// Largest block of a plus block of b over all nodes (b may be null).
template <typename Nbr, typename Eid>
inline int64_t csr_lp_buf_cap(const CsrGraphT<Nbr, Eid>* a,
                              const CsrGraphT<Nbr, Eid>* b) noexcept {
    assert(a != nullptr && a->n_nodes >= 0);
    assert(b == nullptr || b->n_nodes == a->n_nodes);
    int64_t mx = 0;
    for (int64_t u = 0; u < a->n_nodes; ++u) {
        int64_t lo = 0, hi = 0, d = 0;
        csr_graph_block(a, u, &lo, &hi);
        d = hi - lo;
        if (b != nullptr) {
            csr_graph_block(b, u, &lo, &hi);
            d += hi - lo;
        }
        mx = (d > mx) ? d : mx;
    }
    return mx;
}

// Append the labels of u's kept neighbours in g to buf[*k ..]. 0 = ok,
// -1 = a malformed neighbour, -2 = the buffer is smaller than the block.
template <typename Nbr, typename Eid>
BOLT_FORCE_INLINE int32_t csr_lp_gather(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                     int64_t u, const int64_t* BOLT_RESTRICT label,
                                     int64_t* BOLT_RESTRICT buf, int64_t cap,
                                     int64_t* BOLT_RESTRICT k) noexcept {
    assert(g != nullptr && label != nullptr && buf != nullptr && k != nullptr);
    assert(u >= 0 && u < g->n_nodes && *k >= 0);
    int64_t b = 0;
    int64_t e = 0;
    csr_graph_block(g, u, &b, &e);
    if (*k + (e - b) > cap) return -2;
    const int64_t n = g->n_nodes;
    for (int64_t j = b; j < e; ++j) {
        if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0) continue;
        const int64_t v = static_cast<int64_t>(g->neighbors[j]);
        if (v < 0 || v >= n) return -1;
        buf[(*k)++] = label[v];
    }
    return 0;
}

// The most frequent value of sorted[0, k); the smallest one on a tie.
BOLT_FORCE_INLINE int64_t csr_lp_mode(const int64_t* BOLT_RESTRICT sorted,
                                      int64_t k) noexcept {
    assert(sorted != nullptr && k > 0);
    int64_t best = sorted[0];
    int64_t best_run = 0;
    int64_t run = 0;
    for (int64_t i = 0; i < k; ++i) {
        run = (i > 0 && sorted[i] == sorted[i - 1]) ? run + 1 : 1;
        // Strictly greater: an equal run later in ascending order is a
        // larger label and loses the tie.
        if (run > best_run) { best_run = run; best = sorted[i]; }
    }
    assert(best_run >= 1 && best_run <= k);
    return best;
}

// One synchronous sweep label -> sc->next. Returns the number of nodes whose
// label changed, -1 on a malformed graph, -2 on an undersized buffer.
template <typename Nbr, typename Eid>
inline int64_t csr_lp_sweep(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT a,
                            const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT b,
                            const int64_t* BOLT_RESTRICT label,
                            CsrLpScratch* BOLT_RESTRICT sc) noexcept {
    assert(a != nullptr && label != nullptr && sc != nullptr);
    assert(sc->next != nullptr && sc->buf_cap >= 0);
    const int64_t n = a->n_nodes;
    int64_t changed = 0;
    for (int64_t u = 0; u < n; ++u) {
        int64_t k = 0;
        int32_t rc = csr_lp_gather(a, u, label, sc->buf, sc->buf_cap, &k);
        if (rc == 0 && b != nullptr)
            rc = csr_lp_gather(b, u, label, sc->buf, sc->buf_cap, &k);
        if (rc != 0) return rc;
        int64_t nl = label[u];
        if (k > 0) {
            radix_sort_i64(sc->buf, sc->tmp, k);
            nl = csr_lp_mode(sc->buf, k);
        }
        sc->next[u] = nl;
        changed += static_cast<int64_t>(nl != label[u]);
    }
    assert(changed >= 0 && changed <= n);
    return changed;
}

// labels[n_nodes] receives the result. *out_iters = sweeps that changed at
// least one label (<= max_iter). b may be null; when set, it must cover the
// same nodes as a. max_iter in [0, k_csr_lp_max_iterations].
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_label_propagation_t(
        const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT a,
        const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT b, int32_t max_iter,
        CsrLpScratch* BOLT_RESTRICT sc, int64_t* BOLT_RESTRICT labels,
        int32_t* BOLT_RESTRICT out_iters) noexcept {
    assert(out_iters != nullptr);
    assert(sc == nullptr || sc->next != labels || labels == nullptr);
    if (out_iters == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_iters = 0;
    if (!csr_algo_graph_ok(a) || (b != nullptr && !csr_algo_graph_ok(b)) ||
        (b != nullptr && b->n_nodes != a->n_nodes) || sc == nullptr ||
        labels == nullptr || sc->next == nullptr || sc->buf_cap < 0 ||
        (sc->buf_cap > 0 && (sc->buf == nullptr || sc->tmp == nullptr)) ||
        max_iter < 0 || max_iter > k_csr_lp_max_iterations)
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n = a->n_nodes;
    for (int64_t i = 0; i < n; ++i) labels[i] = i;
    for (int32_t it = 0; it < max_iter; ++it) {
        const int64_t changed = csr_lp_sweep(a, b, labels, sc);
        if (changed == -2) return CsrAlgoStatus::InvalidArgument;
        if (changed < 0) return CsrAlgoStatus::MalformedGraph;
        if (changed == 0) break;                 // fixpoint: the rest are identity
        for (int64_t i = 0; i < n; ++i) labels[i] = sc->next[i];
        *out_iters = it + 1;
    }
    assert(*out_iters >= 0 && *out_iters <= max_iter);
    return CsrAlgoStatus::Ok;
}

#define BOLT_CSR_COMMUNITY_ENTRY(G)                                            \
    inline CsrAlgoStatus csr_label_propagation(                               \
            const G* BOLT_RESTRICT a, const G* BOLT_RESTRICT b,               \
            int32_t max_iter, CsrLpScratch* BOLT_RESTRICT sc,                 \
            int64_t* BOLT_RESTRICT labels,                                    \
            int32_t* BOLT_RESTRICT out_iters) noexcept {                      \
        return csr_label_propagation_t(a, b, max_iter, sc, labels, out_iters);\
    }
BOLT_CSR_COMMUNITY_ENTRY(CsrGraph64)
BOLT_CSR_COMMUNITY_ENTRY(CsrGraph32)
#undef BOLT_CSR_COMMUNITY_ENTRY

}  // namespace kernels
}  // namespace bolt
