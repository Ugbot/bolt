// bolt_csr_algo.h — shared contract of the whole-graph CSR algorithm kernels.
//
// The kernels (bolt_csr_pagerank.h, bolt_csr_components.h,
// bolt_csr_dijkstra.h, bolt_csr_similarity.h) run over any CsrGraphT view —
// the int64 layout or the compact u32 one — so they are source-agnostic: the
// CSR may have been built from MarbleDB, parquet, Iceberg or a connector.
//
// Memory: a kernel never allocates. Each publishes `*_scratch_bytes(...)`,
// and the caller carves that from an Arena after charging a Budget —
// CsrAlgoAlloc below does both, so a graph too large for the query's budget
// is refused with ResourceExhausted naming the knob, before any work. Every
// loop is bounded by n_nodes, num_edges or an explicit iteration cap.
//
// Edges honour the view's label filter (csr_edge_label_keep): a typed
// algorithm call sees only its relationship type. Parallel edges and
// self-loops are real edges (multigraph semantics) unless a kernel says
// otherwise.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_csr.h"

#include <cassert>
#include <cstdint>

namespace bolt {
namespace kernels {

enum class CsrAlgoStatus : int32_t {
    Ok              = 0,
    InvalidArgument = 1,   // contract broken (null buffer, id out of range)
    MalformedGraph  = 2,   // a neighbour id outside [0, n_nodes)
    NotConverged    = 3,   // iteration cap hit; the output is NOT the answer
    NegativeWeight  = 4,   // Dijkstra met a negative or NaN weight
};

// Number of CSR slots a view covers (off[win] of its window).
template <typename Nbr, typename Eid>
BOLT_FORCE_INLINE int64_t csr_algo_num_edges(const CsrGraphT<Nbr, Eid>* g) noexcept {
    assert(g != nullptr && g->off != nullptr);
    const int64_t win = (g->n_win < 0) ? g->n_nodes : g->n_win;
    assert(win >= 0);
    return g->off[win];
}

template <typename Nbr, typename Eid>
BOLT_FORCE_INLINE bool csr_algo_graph_ok(const CsrGraphT<Nbr, Eid>* g) noexcept {
    assert(g == nullptr || g->n_nodes >= 0);
    if (g == nullptr || g->off == nullptr || g->n_nodes <= 0) return false;
    if (g->neighbors == nullptr && csr_algo_num_edges(g) > 0) return false;
    return true;
}

// Budget-charged scratch carving. try_alloc charges `budget` (when non-null)
// then the arena; on refusal it returns null with ResourceExhausted set and
// nothing charged. release() returns every byte this object charged.
struct CsrAlgoAlloc {
    Arena*   arena;
    Budget*  budget;
    uint64_t charged;

    void* try_alloc(uint64_t bytes) noexcept {
        assert(arena != nullptr);
        assert(bytes > 0);
        if (budget != nullptr && !budget->try_reserve(bytes)) return nullptr;
        void* p = arena->allocate(static_cast<size_t>(bytes), 64);
        if (p == nullptr) {
            if (budget != nullptr) budget->release(bytes);
            set_resource_exhausted("arena_max_blocks", bytes, 0);
            return nullptr;
        }
        charged += (budget != nullptr) ? bytes : 0;
        return p;
    }

    void release() noexcept {
        assert(budget != nullptr || charged == 0);
        if (budget != nullptr && charged != 0) budget->release(charged);
        charged = 0;
    }
};

}  // namespace kernels
}  // namespace bolt
