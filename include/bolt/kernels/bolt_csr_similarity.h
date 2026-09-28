// bolt_csr_similarity.h — degree and neighbourhood-similarity kernels.
//
//   csr_degree            : kept-edge count per node over one CSR view.
//                           Out-degree over the forward CSR, in-degree over
//                           the reverse one; parallel edges count, a
//                           self-loop counts once per direction (networkx
//                           MultiDiGraph out_degree / in_degree).
//   csr_max_degree        : the largest kept block — sizes the similarity
//                           scratch before the call.
//   csr_node_similarity   : Jaccard |N(u) ∩ N(v)| / |N(u) ∪ N(v)| or overlap
//                           |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|) for a batch of
//                           (u, v) pairs, N = the SET of kept out-neighbours
//                           (parallel edges collapse). An empty denominator
//                           gives 0. The intersection is set_intersect_sorted
//                           over the two blocks — directly when the CSR is
//                           dst-sorted, after a radix sort into scratch
//                           otherwise.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"
#include "bolt/kernels/bolt_radixsort.h"
#include "bolt/kernels/bolt_setop.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

enum class CsrSimilarity : uint8_t { Jaccard = 0, Overlap = 1 };

template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_degree_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                                  int64_t* BOLT_RESTRICT out_deg) noexcept {
    assert(g == nullptr || g->n_nodes >= 0);
    assert(out_deg != nullptr || g == nullptr);
    if (!csr_algo_graph_ok(g) || out_deg == nullptr)
        return CsrAlgoStatus::InvalidArgument;
    for (int64_t u = 0; u < g->n_nodes; ++u) {
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        int64_t d = 0;
        for (int64_t j = b; j < e; ++j)
            d += csr_edge_label_keep(g->edge_labels, g->want_label, j);
        out_deg[u] = d;
    }
    return CsrAlgoStatus::Ok;
}

template <typename Nbr, typename Eid>
inline int64_t csr_max_degree_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g) noexcept {
    assert(g != nullptr && g->off != nullptr);
    assert(g->n_nodes > 0);
    int64_t mx = 0;
    for (int64_t u = 0; u < g->n_nodes; ++u) {
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        mx = (e - b > mx) ? e - b : mx;
    }
    return mx;
}

// Kept neighbours of u into buf (sorted ascending, duplicates kept); returns
// the count, or -1 on a malformed neighbour / -2 when buf_cap is too small.
template <typename Nbr, typename Eid>
inline int64_t csr_sim_gather(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                              int64_t u, bool dst_sorted,
                              int64_t* BOLT_RESTRICT buf,
                              int64_t* BOLT_RESTRICT tmp,
                              int64_t buf_cap) noexcept {
    assert(g != nullptr && buf != nullptr && tmp != nullptr);
    assert(u >= 0 && u < g->n_nodes);
    int64_t b = 0;
    int64_t e = 0;
    csr_graph_block(g, u, &b, &e);
    if (e - b > buf_cap) return -2;
    int64_t k = 0;
    for (int64_t j = b; j < e; ++j) {
        if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0) continue;
        const int64_t v = static_cast<int64_t>(g->neighbors[j]);
        if (v < 0 || v >= g->n_nodes) return -1;
        buf[k++] = v;
    }
    if (!dst_sorted) radix_sort_i64(buf, tmp, k);
    return k;
}

BOLT_FORCE_INLINE int64_t csr_sim_distinct(const int64_t* a, int64_t n) noexcept {
    assert(a != nullptr || n == 0);
    assert(n >= 0);
    int64_t d = 0;
    for (int64_t i = 0; i < n; ++i) d += static_cast<int64_t>(i == 0 || a[i] != a[i - 1]);
    return d;
}

// Scratch: three int64 buffers of buf_cap >= csr_max_degree(g) slots each.
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_node_similarity_t(
        const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g, const int64_t* BOLT_RESTRICT us,
        const int64_t* BOLT_RESTRICT vs, int64_t n_pairs, CsrSimilarity kind,
        bool dst_sorted, int64_t* BOLT_RESTRICT buf_a, int64_t* BOLT_RESTRICT buf_b,
        int64_t* BOLT_RESTRICT buf_tmp, int64_t buf_cap,
        double* BOLT_RESTRICT out) noexcept {
    assert(n_pairs >= 0 && buf_cap >= 0);
    assert(kind == CsrSimilarity::Jaccard || kind == CsrSimilarity::Overlap);
    if (!csr_algo_graph_ok(g) || n_pairs < 0 || buf_cap < 0 ||
        (n_pairs > 0 && (us == nullptr || vs == nullptr || out == nullptr ||
                         buf_a == nullptr || buf_b == nullptr || buf_tmp == nullptr)))
        return CsrAlgoStatus::InvalidArgument;
    for (int64_t i = 0; i < n_pairs; ++i) {
        const int64_t u = us[i];
        const int64_t v = vs[i];
        if (u < 0 || u >= g->n_nodes || v < 0 || v >= g->n_nodes)
            return CsrAlgoStatus::InvalidArgument;
        const int64_t na = csr_sim_gather(g, u, dst_sorted, buf_a, buf_tmp, buf_cap);
        const int64_t nb = csr_sim_gather(g, v, dst_sorted, buf_b, buf_tmp, buf_cap);
        if (na == -1 || nb == -1) return CsrAlgoStatus::MalformedGraph;
        if (na < 0 || nb < 0) return CsrAlgoStatus::InvalidArgument;
        const int64_t inter = setop::set_intersect_sorted(buf_a, na, buf_b, nb, buf_tmp);
        const int64_t da = csr_sim_distinct(buf_a, na);
        const int64_t db = csr_sim_distinct(buf_b, nb);
        const int64_t den = (kind == CsrSimilarity::Jaccard)
                                ? da + db - inter
                                : (da < db ? da : db);
        assert(inter >= 0 && inter <= den);
        out[i] = (den == 0) ? 0.0
                            : static_cast<double>(inter) / static_cast<double>(den);
    }
    return CsrAlgoStatus::Ok;
}

#define BOLT_CSR_SIMILARITY_ENTRY(G)                                           \
    inline CsrAlgoStatus csr_degree(const G* BOLT_RESTRICT g,                  \
                                    int64_t* BOLT_RESTRICT out_deg) noexcept { \
        return csr_degree_t(g, out_deg);                                       \
    }                                                                          \
    inline int64_t csr_max_degree(const G* BOLT_RESTRICT g) noexcept {         \
        return csr_max_degree_t(g);                                            \
    }                                                                          \
    inline CsrAlgoStatus csr_node_similarity(                                  \
            const G* BOLT_RESTRICT g, const int64_t* BOLT_RESTRICT us,         \
            const int64_t* BOLT_RESTRICT vs, int64_t n_pairs,                  \
            CsrSimilarity kind, bool dst_sorted, int64_t* BOLT_RESTRICT buf_a, \
            int64_t* BOLT_RESTRICT buf_b, int64_t* BOLT_RESTRICT buf_tmp,      \
            int64_t buf_cap, double* BOLT_RESTRICT out) noexcept {             \
        return csr_node_similarity_t(g, us, vs, n_pairs, kind, dst_sorted,     \
                                     buf_a, buf_b, buf_tmp, buf_cap, out);     \
    }
BOLT_CSR_SIMILARITY_ENTRY(CsrGraph64)
BOLT_CSR_SIMILARITY_ENTRY(CsrGraph32)
#undef BOLT_CSR_SIMILARITY_ENTRY

}  // namespace kernels
}  // namespace bolt
