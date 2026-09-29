// test_bolt_csr_algo.cpp — pinned cases for the CSR algorithm kernels and the
// Reach path semantics. Expected values are hand-derived from the definition
// (and agree with networkx / pyoxigraph); the random-graph oracle run is
// graph_algo_fuzz.py.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/kernels/bolt_csr_bfs.h"
#include "bolt/kernels/bolt_csr_components.h"
#include "bolt/kernels/bolt_csr_dijkstra.h"
#include "bolt/kernels/bolt_csr_levels.h"
#include "bolt/kernels/bolt_csr_pagerank.h"
#include "bolt/kernels/bolt_csr_similarity.h"

#include <cmath>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace bk = bolt::kernels;

namespace {

struct Graph {
    std::vector<int64_t> off, nbr, eid;
    bk::CsrGraph64 g{};
    Graph(int64_t n, const std::vector<std::pair<int64_t, int64_t>>& edges) {
        std::vector<int64_t> s, d, scratch(static_cast<size_t>(n));
        for (const auto& e : edges) { s.push_back(e.first); d.push_back(e.second); }
        const size_t m = edges.size();
        off.assign(static_cast<size_t>(n) + 1, 0);
        nbr.assign(m ? m : 1, 0);
        eid.assign(m ? m : 1, 0);
        EXPECT_TRUE(bk::csr_build(s.data(), d.data(), nullptr,
                                  static_cast<int64_t>(m), n, off.data(),
                                  nbr.data(), eid.data(), scratch.data()));
        g.off = off.data(); g.neighbors = nbr.data(); g.edge_ids = eid.data();
        g.n_nodes = n; g.want_label = -1;
    }
};

std::set<std::pair<int64_t, int64_t>> reach_all(const bk::CsrGraph64& g,
                                                int32_t lo, int32_t hi) {
    const int64_t n = g.n_nodes;
    std::vector<int64_t> src(static_cast<size_t>(n)), q(src.size());
    for (int64_t i = 0; i < n; ++i) src[static_cast<size_t>(i)] = i;
    std::vector<uint64_t> seen(static_cast<size_t>(bk::csr_reach_seen_words(n)), 0);
    bk::CsrBfsScratch sc{nullptr, nullptr, nullptr, seen.data(), q.data()};
    bk::CsrBfsParams p{};
    p.per_source_cap = INT64_MAX; p.visit_budget = 3;
    p.min_hops = lo; p.max_hops = hi; p.semantics = bk::CsrPathSemantics::Reach;
    bk::CsrBfsCursor cur{};
    std::set<std::pair<int64_t, int64_t>> out;
    int64_t os[2], od[2];
    for (int guard = 0; !bk::csr_bfs_done(&cur, n) && guard < 100000; ++guard) {
        int64_t got = 0;
        EXPECT_EQ(bk::csr_bfs_expand(src.data(), n, &g, &p, &sc, os, od, nullptr,
                                     2, &cur, &got),
                  bk::CsrBfsStatus::Ok);
        for (int64_t r = 0; r < got; ++r) {
            EXPECT_TRUE(out.emplace(os[r], od[r]).second) << "duplicate row";
        }
    }
    return out;
}

// G2GRAPH-328 fixture: a p a . a p b . b p c . c p c . p p x (a..x = 0..4).
const std::vector<std::pair<int64_t, int64_t>> k328 = {
    {0, 0}, {0, 1}, {1, 2}, {2, 2}, {3, 4}};

}  // namespace

TEST(CsrReach, PlusKeepsCyclesBackToTheStart) {
    Graph g(5, k328);
    const auto got = reach_all(g.g, 1, bk::k_csr_reach_unbounded);
    const std::set<std::pair<int64_t, int64_t>> want = {
        {0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 2}, {3, 4}};
    EXPECT_EQ(got, want);
}

TEST(CsrReach, StarAddsEveryZeroLengthPairOnce) {
    Graph g(5, k328);
    const auto got = reach_all(g.g, 0, bk::k_csr_reach_unbounded);
    EXPECT_EQ(got.size(), 9u);   // the 6 above plus (b,b) (p,p) (x,x), each once
    for (int64_t v = 0; v < 5; ++v) EXPECT_TRUE(got.count({v, v}));
}

TEST(CsrReach, HopBoundAndLongChain) {
    std::vector<std::pair<int64_t, int64_t>> chain;
    for (int64_t i = 0; i < 39; ++i) chain.emplace_back(i, i + 1);
    chain.emplace_back(39, 0);                      // one 40-cycle
    Graph g(40, chain);
    EXPECT_EQ(reach_all(g.g, 1, bk::k_csr_reach_unbounded).size(), 1600u);
    EXPECT_EQ(reach_all(g.g, 1, 39).size(), 40u * 39u);   // start needs 40 hops
    EXPECT_EQ(reach_all(g.g, 0, 0).size(), 40u);
}

TEST(CsrReach, RefusesHopsColumnAndMinAboveOne) {
    Graph g(2, {{0, 1}});
    int64_t src = 0, q[2], os[4], od[4], got = 0;
    int32_t hops[4];
    uint64_t seen[1] = {0};
    bk::CsrBfsScratch sc{nullptr, nullptr, nullptr, seen, q};
    bk::CsrBfsParams p{};
    p.per_source_cap = 10; p.visit_budget = 10;
    p.min_hops = 0; p.max_hops = 3; p.semantics = bk::CsrPathSemantics::Reach;
    bk::CsrBfsCursor cur{};
    EXPECT_EQ(bk::csr_bfs_expand(&src, 1, &g.g, &p, &sc, os, od, hops, 4, &cur, &got),
              bk::CsrBfsStatus::InvalidArgument);
    p.min_hops = 2;
    EXPECT_EQ(bk::csr_bfs_expand(&src, 1, &g.g, &p, &sc, os, od, nullptr, 4, &cur, &got),
              bk::CsrBfsStatus::InvalidArgument);
    p.min_hops = 0; p.per_source_cap = 1;          // 2 rows owed, cap 1: loud
    EXPECT_EQ(bk::csr_bfs_expand(&src, 1, &g.g, &p, &sc, os, od, nullptr, 4, &cur, &got),
              bk::CsrBfsStatus::PerSourceCapExceeded);
}

TEST(CsrAlgo, PageRankTwoCycleAndDangling) {
    Graph g(3, {{0, 1}, {1, 0}});                  // node 2 dangling and isolated
    std::vector<double> r(3), s(3);
    int32_t it = 0;
    ASSERT_EQ(bk::csr_pagerank(&g.g, 0.85, 1000, 1e-14, r.data(), s.data(), &it),
              bk::CsrAlgoStatus::Ok);
    EXPECT_NEAR(r[0] + r[1] + r[2], 1.0, 1e-12);
    EXPECT_NEAR(r[0], r[1], 1e-12);
    EXPECT_NEAR(r[2], 1.0 / 3.0 * (0.15 + 0.85 * r[2]), 1e-12);
    EXPECT_EQ(bk::csr_pagerank(&g.g, 0.85, 1, 1e-14, r.data(), s.data(), &it),
              bk::CsrAlgoStatus::NotConverged);
}

TEST(CsrAlgo, ComponentsLabelByMinId) {
    Graph g(6, {{1, 2}, {2, 1}, {3, 1}, {5, 4}, {4, 4}});
    std::vector<int64_t> w(6), s(6), ix(6), lo(6), st(6), cn(6), ce(6);
    uint64_t on[1];
    int64_t nw = 0, ns = 0;
    ASSERT_EQ(bk::csr_wcc(&g.g, w.data(), &nw), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(nw, 3);
    EXPECT_EQ(w, (std::vector<int64_t>{0, 1, 1, 1, 4, 4}));
    bk::CsrSccScratch sc{ix.data(), lo.data(), st.data(), cn.data(), ce.data(), on};
    ASSERT_EQ(bk::csr_scc(&g.g, &sc, s.data(), &ns), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(ns, 5);
    EXPECT_EQ(s, (std::vector<int64_t>{0, 1, 1, 3, 4, 5}));
}

TEST(CsrAlgo, SccDeepChainNoRecursion) {
    const int64_t n = 200000;                       // recursion would blow the stack
    std::vector<std::pair<int64_t, int64_t>> e;
    for (int64_t i = 0; i + 1 < n; ++i) e.emplace_back(i, i + 1);
    e.emplace_back(n - 1, 0);
    Graph g(n, e);
    const size_t z = static_cast<size_t>(n);
    std::vector<int64_t> s(z), ix(z), lo(z), st(z), cn(z), ce(z);
    std::vector<uint64_t> on((z + 63) / 64);
    bk::CsrSccScratch sc{ix.data(), lo.data(), st.data(), cn.data(), ce.data(), on.data()};
    int64_t ns = 0;
    ASSERT_EQ(bk::csr_scc(&g.g, &sc, s.data(), &ns), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(ns, 1);
    EXPECT_EQ(s[z - 1], 0);
}

TEST(CsrAlgo, DijkstraParallelEdgesAndNegativeRefused) {
    Graph g(4, {{0, 1}, {0, 1}, {1, 2}, {0, 2}});
    std::vector<double> w = {5.0, 1.0, 3.0, 1.0}, dist(4);   // by CSR slot
    std::vector<int64_t> par(4), ps(4), scratch(8);
    int64_t settled = 0;
    ASSERT_EQ(bk::csr_dijkstra(&g.g, w.data(), 0, -1, dist.data(), par.data(),
                               ps.data(), scratch.data(), &settled),
              bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(dist[1], 1.0);
    EXPECT_EQ(dist[2], 2.0);
    EXPECT_TRUE(std::isinf(dist[3]));
    EXPECT_EQ(settled, 3);
    w[3] = -1.0;
    EXPECT_EQ(bk::csr_dijkstra(&g.g, w.data(), 0, -1, dist.data(), nullptr, nullptr,
                               scratch.data(), &settled),
              bk::CsrAlgoStatus::NegativeWeight);
}

TEST(CsrAlgo, SimilaritySetsNotMultisets) {
    Graph g(4, {{0, 2}, {0, 2}, {0, 3}, {1, 3}});
    const int64_t us[3] = {0, 0, 2}, vs[3] = {1, 0, 3};
    int64_t a[4], b[4], t[4];
    double j[3], o[3];
    ASSERT_EQ(bk::csr_node_similarity(&g.g, us, vs, 3, bk::CsrSimilarity::Jaccard,
                                      false, a, b, t, 4, j),
              bk::CsrAlgoStatus::Ok);
    ASSERT_EQ(bk::csr_node_similarity(&g.g, us, vs, 3, bk::CsrSimilarity::Overlap,
                                      false, a, b, t, 4, o),
              bk::CsrAlgoStatus::Ok);
    EXPECT_DOUBLE_EQ(j[0], 0.5);   // {2,3} vs {3}
    EXPECT_DOUBLE_EQ(o[0], 1.0);
    EXPECT_DOUBLE_EQ(j[1], 1.0);
    EXPECT_DOUBLE_EQ(j[2], 0.0);   // both empty
    std::vector<int64_t> deg(4);
    ASSERT_EQ(bk::csr_degree(&g.g, deg.data()), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(deg, (std::vector<int64_t>{3, 1, 0, 0}));
    EXPECT_EQ(bk::csr_node_similarity(&g.g, us, vs, 3, bk::CsrSimilarity::Jaccard,
                                      false, a, b, t, 2, j),
              bk::CsrAlgoStatus::InvalidArgument);   // scratch smaller than a block
}

TEST(CsrAlgo, ContractRefusals) {
    bk::CsrGraph64 empty{};
    std::vector<int64_t> out(4);
    int64_t n = 0;
    EXPECT_EQ(bk::csr_wcc(&empty, out.data(), &n), bk::CsrAlgoStatus::InvalidArgument);
    EXPECT_EQ(bk::csr_degree(&empty, out.data()), bk::CsrAlgoStatus::InvalidArgument);
    Graph g(2, {{0, 5}});                           // neighbour outside the node range
    EXPECT_EQ(bk::csr_wcc(&g.g, out.data(), &n), bk::CsrAlgoStatus::MalformedGraph);
}

TEST(CsrAlgo, ScratchIsBudgetCharged) {
    bolt::Arena arena;
    bolt::Budget budget("graph_algo_test_mb", 1024);
    bk::CsrAlgoAlloc alloc{&arena, &budget, 0};
    EXPECT_NE(alloc.try_alloc(bk::csr_pagerank_scratch_bytes(100)), nullptr);
    EXPECT_EQ(alloc.try_alloc(bk::csr_scc_scratch_bytes(100)), nullptr);
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "graph_algo_test_mb");
    EXPECT_EQ(budget.used(), 800u);
    alloc.release();
    EXPECT_EQ(budget.used(), 0u);
}

TEST(CsrAlgo, BfsLevelsDepthBoundAndUnreached) {
    // 0->1->2->3, 0->2 shortcut, 4->0 (0 cannot reach 4), 3->3 self-loop.
    Graph g(5, {{0, 1}, {1, 2}, {2, 3}, {0, 2}, {4, 0}, {3, 3}});
    int64_t depth[5], q[5], reached = 0;
    ASSERT_EQ(bk::csr_bfs_levels(&g.g, 0, -1, depth, q, &reached), bk::CsrAlgoStatus::Ok);
    const int64_t want[5] = {0, 1, 1, 2, -1};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(depth[i], want[i]) << i;
    EXPECT_EQ(reached, 4);
    ASSERT_EQ(bk::csr_bfs_levels(&g.g, 0, 1, depth, q, &reached), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(depth[3], -1);
    EXPECT_EQ(reached, 3);
    ASSERT_EQ(bk::csr_bfs_levels(&g.g, 0, 0, depth, q, &reached), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(reached, 1);
    EXPECT_EQ(bk::csr_bfs_levels(&g.g, 5, -1, depth, q, &reached),
              bk::CsrAlgoStatus::InvalidArgument);
    EXPECT_EQ(bk::csr_bfs_levels(&g.g, 0, -1, depth, nullptr, &reached),
              bk::CsrAlgoStatus::InvalidArgument);
}
