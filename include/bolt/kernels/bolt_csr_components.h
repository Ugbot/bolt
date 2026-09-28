// bolt_csr_components.h — weakly and strongly connected components.
//
//   csr_wcc : union-find over the kept edges, direction ignored. Linking the
//             larger root under the smaller keeps every root the minimum id
//             of its set, so the label written for a node IS the smallest
//             node id in its component — a canonical, order-free labelling.
//             Path halving keeps finds short; no rank array is needed.
//   csr_scc : Tarjan, iterative. The DFS call stack is an explicit pair of
//             arrays (node, next CSR slot), so a million-node path cannot
//             overflow the machine stack. Labels are again the smallest node
//             id of each component.
//
// Both write out_comp[n_nodes] and the component count. Nodes outside the
// view's window have no edges and are singleton components.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_algo.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

// ---------------------------------------------------------------------------
// WCC
// ---------------------------------------------------------------------------

BOLT_FORCE_INLINE int64_t csr_uf_find(int64_t* BOLT_RESTRICT parent, int64_t x,
                                      int64_t n) noexcept {
    assert(parent != nullptr && x >= 0 && x < n);
    for (int64_t steps = 0; parent[x] != x; ++steps) {
        assert(steps < n && "csr_wcc: union-find cycle");
        (void)steps;
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// out_comp doubles as the union-find parent array: no scratch beyond it.
template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_wcc_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                               int64_t* BOLT_RESTRICT out_comp,
                               int64_t* BOLT_RESTRICT out_n_comp) noexcept {
    assert(out_n_comp != nullptr);
    assert(g == nullptr || g->n_nodes >= 0);
    if (out_n_comp == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_n_comp = 0;
    if (!csr_algo_graph_ok(g) || out_comp == nullptr)
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n = g->n_nodes;
    for (int64_t i = 0; i < n; ++i) out_comp[i] = i;
    for (int64_t u = 0; u < n; ++u) {
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        for (int64_t j = b; j < e; ++j) {
            if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0)
                continue;
            const int64_t v = static_cast<int64_t>(g->neighbors[j]);
            if (v < 0 || v >= n) return CsrAlgoStatus::MalformedGraph;
            const int64_t ru = csr_uf_find(out_comp, u, n);
            const int64_t rv = csr_uf_find(out_comp, v, n);
            if (ru < rv) out_comp[rv] = ru;
            else if (rv < ru) out_comp[ru] = rv;
        }
    }
    int64_t comps = 0;
    for (int64_t i = 0; i < n; ++i) {
        out_comp[i] = csr_uf_find(out_comp, i, n);
        comps += static_cast<int64_t>(out_comp[i] == i);
    }
    assert(comps >= 1 && comps <= n);
    *out_n_comp = comps;
    return CsrAlgoStatus::Ok;
}

// ---------------------------------------------------------------------------
// SCC
// ---------------------------------------------------------------------------

constexpr uint64_t csr_scc_scratch_bytes(int64_t n_nodes) noexcept {
    return static_cast<uint64_t>(n_nodes) * 5u * sizeof(int64_t) +
           static_cast<uint64_t>((n_nodes + 63) / 64) * sizeof(uint64_t);
}

// Caller-owned: five int64 arrays of n_nodes and one bitmap of
// (n_nodes + 63) / 64 words. The kernel initialises all of them.
struct CsrSccScratch {
    int64_t*  index;     // DFS discovery index, -1 = unvisited
    int64_t*  low;       // Tarjan lowlink
    int64_t*  stack;     // Tarjan component stack
    int64_t*  cs_node;   // explicit call stack: node
    int64_t*  cs_edge;   // explicit call stack: next CSR slot to try
    uint64_t* on_stack;  // bitmap
};

struct CsrSccState {
    int64_t next_index;
    int64_t sp;          // Tarjan stack depth
    int64_t csp;         // call stack depth
    int64_t comps;
};

BOLT_FORCE_INLINE bool csr_scc_on(const uint64_t* bm, int64_t v) noexcept {
    assert(bm != nullptr && v >= 0);
    return ((bm[v >> 6] >> (static_cast<uint64_t>(v) & 63u)) & 1u) != 0;
}

template <typename Nbr, typename Eid>
BOLT_FORCE_INLINE void csr_scc_push(const CsrGraphT<Nbr, Eid>* g,
                                    CsrSccScratch* sc, CsrSccState* st,
                                    int64_t v) noexcept {
    assert(g != nullptr && sc != nullptr && st != nullptr);
    assert(st->sp < g->n_nodes && st->csp < g->n_nodes);
    sc->index[v] = st->next_index;
    sc->low[v]   = st->next_index;
    ++st->next_index;
    sc->stack[st->sp++] = v;
    sc->on_stack[v >> 6] |= uint64_t{1} << (static_cast<uint64_t>(v) & 63u);
    int64_t b = 0;
    int64_t e = 0;
    csr_graph_block(g, v, &b, &e);
    sc->cs_node[st->csp] = v;
    sc->cs_edge[st->csp] = b;
    ++st->csp;
}

// u is finished and is a root: pop its component, label it by its min id.
BOLT_FORCE_INLINE void csr_scc_pop_component(CsrSccScratch* BOLT_RESTRICT sc,
                                             CsrSccState* BOLT_RESTRICT st,
                                             int64_t u,
                                             int64_t* BOLT_RESTRICT out) noexcept {
    assert(sc != nullptr && st != nullptr && out != nullptr);
    assert(st->sp > 0);
    int64_t first = st->sp - 1;
    while (sc->stack[first] != u) { assert(first > 0); --first; }
    int64_t mn = u;
    for (int64_t k = first; k < st->sp; ++k)
        mn = (sc->stack[k] < mn) ? sc->stack[k] : mn;
    for (int64_t k = first; k < st->sp; ++k) {
        const int64_t w = sc->stack[k];
        out[w] = mn;
        sc->on_stack[w >> 6] &= ~(uint64_t{1} << (static_cast<uint64_t>(w) & 63u));
    }
    st->sp = first;
    ++st->comps;
}

// Run the DFS rooted at r to completion. False on a malformed neighbour.
template <typename Nbr, typename Eid>
inline bool csr_scc_from(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                         CsrSccScratch* BOLT_RESTRICT sc,
                         CsrSccState* BOLT_RESTRICT st, int64_t r,
                         int64_t* BOLT_RESTRICT out) noexcept {
    assert(g != nullptr && sc != nullptr && st != nullptr && out != nullptr);
    assert(sc->index[r] == -1 && st->csp == 0);
    const int64_t n = g->n_nodes;
    csr_scc_push(g, sc, st, r);
    while (st->csp > 0) {
        const int64_t u = sc->cs_node[st->csp - 1];
        int64_t b = 0;
        int64_t e = 0;
        csr_graph_block(g, u, &b, &e);
        int64_t j = sc->cs_edge[st->csp - 1];
        bool descended = false;
        for (; j < e; ++j) {
            if (csr_edge_label_keep(g->edge_labels, g->want_label, j) == 0)
                continue;
            const int64_t v = static_cast<int64_t>(g->neighbors[j]);
            if (v < 0 || v >= n) return false;
            if (sc->index[v] == -1) {
                sc->cs_edge[st->csp - 1] = j + 1;
                csr_scc_push(g, sc, st, v);
                descended = true;
                break;
            }
            if (csr_scc_on(sc->on_stack, v) && sc->index[v] < sc->low[u])
                sc->low[u] = sc->index[v];
        }
        if (descended) continue;
        if (sc->low[u] == sc->index[u]) csr_scc_pop_component(sc, st, u, out);
        --st->csp;
        if (st->csp > 0) {
            const int64_t w = sc->cs_node[st->csp - 1];
            if (sc->low[u] < sc->low[w]) sc->low[w] = sc->low[u];
        }
    }
    return true;
}

template <typename Nbr, typename Eid>
inline CsrAlgoStatus csr_scc_t(const CsrGraphT<Nbr, Eid>* BOLT_RESTRICT g,
                               CsrSccScratch* BOLT_RESTRICT sc,
                               int64_t* BOLT_RESTRICT out_comp,
                               int64_t* BOLT_RESTRICT out_n_comp) noexcept {
    assert(out_n_comp != nullptr);
    assert(g == nullptr || g->n_nodes >= 0);
    if (out_n_comp == nullptr) return CsrAlgoStatus::InvalidArgument;
    *out_n_comp = 0;
    if (!csr_algo_graph_ok(g) || sc == nullptr || out_comp == nullptr ||
        sc->index == nullptr || sc->low == nullptr || sc->stack == nullptr ||
        sc->cs_node == nullptr || sc->cs_edge == nullptr ||
        sc->on_stack == nullptr)
        return CsrAlgoStatus::InvalidArgument;
    const int64_t n = g->n_nodes;
    for (int64_t i = 0; i < n; ++i) sc->index[i] = -1;
    for (int64_t i = 0; i < (n + 63) / 64; ++i) sc->on_stack[i] = 0;
    CsrSccState st{0, 0, 0, 0};
    for (int64_t r = 0; r < n; ++r) {
        if (sc->index[r] != -1) continue;
        if (!csr_scc_from(g, sc, &st, r, out_comp))
            return CsrAlgoStatus::MalformedGraph;
    }
    assert(st.sp == 0 && st.next_index == n);
    *out_n_comp = st.comps;
    return CsrAlgoStatus::Ok;
}

#define BOLT_CSR_COMPONENTS_ENTRY(G)                                           \
    inline CsrAlgoStatus csr_wcc(const G* BOLT_RESTRICT g,                     \
                                 int64_t* BOLT_RESTRICT out_comp,              \
                                 int64_t* BOLT_RESTRICT out_n_comp) noexcept { \
        return csr_wcc_t(g, out_comp, out_n_comp);                             \
    }                                                                          \
    inline CsrAlgoStatus csr_scc(const G* BOLT_RESTRICT g,                     \
                                 CsrSccScratch* BOLT_RESTRICT sc,              \
                                 int64_t* BOLT_RESTRICT out_comp,              \
                                 int64_t* BOLT_RESTRICT out_n_comp) noexcept { \
        return csr_scc_t(g, sc, out_comp, out_n_comp);                         \
    }
BOLT_CSR_COMPONENTS_ENTRY(CsrGraph64)
BOLT_CSR_COMPONENTS_ENTRY(CsrGraph32)
#undef BOLT_CSR_COMPONENTS_ENTRY

}  // namespace kernels
}  // namespace bolt
