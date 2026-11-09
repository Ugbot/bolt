// test_bolt_csr_community.cpp — pinned cases for csr_label_propagation
// (G2GRAPH-438). Expected labels are hand-derived from the LDBC Graphalytics
// CDLP definition; the random-graph oracle run is graph_algo_fuzz.py.

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/kernels/bolt_csr_community.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace bk = bolt::kernels;

namespace {

struct Graph {
    std::vector<int64_t> off, nbr, eid;
    bk::CsrGraph64 g{};
    Graph(int64_t n, const std::vector<std::pair<int64_t, int64_t>>& edges, bool rev) {
        std::vector<int64_t> s, d, scratch(static_cast<size_t>(n));
        for (const auto& e : edges) {
            s.push_back(rev ? e.second : e.first);
            d.push_back(rev ? e.first : e.second);
        }
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

struct LpRun {
    bk::CsrAlgoStatus st;
    int32_t iters;
    std::vector<int64_t> labels;
};

LpRun lp(const bk::CsrGraph64* a, const bk::CsrGraph64* b, int32_t max_iter,
       int64_t cap_override = -1) {
    const int64_t n = a->n_nodes;
    const int64_t cap = cap_override >= 0 ? cap_override : bk::csr_lp_buf_cap(a, b);
    std::vector<int64_t> next(static_cast<size_t>(n)), buf(static_cast<size_t>(cap) + 1),
        tmp(buf.size());
    bk::CsrLpScratch sc{next.data(), buf.data(), tmp.data(), cap};
    LpRun r{bk::CsrAlgoStatus::Ok, -1, std::vector<int64_t>(static_cast<size_t>(n), -9)};
    r.st = bk::csr_label_propagation(a, b, max_iter, &sc, r.labels.data(), &r.iters);
    return r;
}

}  // namespace

// 0->1->2 seen both ways is bipartite: synchronous CDLP oscillates, so the
// iteration count is part of the answer.
TEST(CsrLabelPropagation, PathOscillatesWithIterationCount) {
    const std::vector<std::pair<int64_t, int64_t>> e = {{0, 1}, {1, 2}};
    Graph f(3, e, false), r(3, e, true);
    LpRun x = lp(&f.g, &r.g, 0);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(x.iters, 0);
    x = lp(&f.g, &r.g, 1);                  // node 1 sees {2, 0}: tie -> 0
    EXPECT_EQ(x.labels, (std::vector<int64_t>{1, 0, 1}));
    x = lp(&f.g, &r.g, 2);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{0, 1, 0}));
    x = lp(&f.g, &r.g, 3);
    EXPECT_EQ(x.st, bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{1, 0, 1}));
    EXPECT_EQ(x.iters, 3);
}

// A 4-clique settles on its smallest id in two sweeps and then stops early;
// an isolated node and a self-loop-only node keep their own labels.
TEST(CsrLabelPropagation, CliqueConvergesIsolatedKeepLabel) {
    const std::vector<std::pair<int64_t, int64_t>> e = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}, {5, 5}};
    Graph f(6, e, false), r(6, e, true);
    LpRun x = lp(&f.g, &r.g, 1);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{1, 0, 0, 0, 4, 5}));
    x = lp(&f.g, &r.g, 10);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{0, 0, 0, 0, 4, 5}));
    EXPECT_EQ(x.iters, 2);
}

// Parallel edges are counted per edge (multigraph): {1, 2, 2} -> 2, where a
// deduplicated neighbour set {1, 2} would tie to 1. Forward-only view.
TEST(CsrLabelPropagation, ParallelEdgesCountPerEdge) {
    Graph f(3, {{0, 2}, {0, 2}, {0, 1}}, false);
    const LpRun x = lp(&f.g, nullptr, 5);
    EXPECT_EQ(x.labels, (std::vector<int64_t>{2, 1, 2}));
    EXPECT_EQ(x.iters, 1);
}

TEST(CsrLabelPropagation, CompactLayoutAgrees) {
    const std::vector<std::pair<int64_t, int64_t>> e = {
        {0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 3}, {2, 3}, {5, 5}, {5, 1}};
    Graph f(6, e, false), r(6, e, true);
    std::vector<uint32_t> fn(f.nbr.begin(), f.nbr.end()), rn(r.nbr.begin(), r.nbr.end());
    bk::CsrGraph32 f32{}, r32{};
    f32.off = f.off.data(); f32.neighbors = fn.data(); f32.n_nodes = 6; f32.want_label = -1;
    r32.off = r.off.data(); r32.neighbors = rn.data(); r32.n_nodes = 6; r32.want_label = -1;
    const LpRun x = lp(&f.g, &r.g, 7);
    const int64_t cap = bk::csr_lp_buf_cap(&f32, &r32);
    std::vector<int64_t> next(6), buf(static_cast<size_t>(cap)), tmp(buf.size()), lab(6);
    bk::CsrLpScratch sc{next.data(), buf.data(), tmp.data(), cap};
    int32_t it = -1;
    ASSERT_EQ(bk::csr_label_propagation(&f32, &r32, 7, &sc, lab.data(), &it),
              bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(lab, x.labels);
    EXPECT_EQ(it, x.iters);
}

TEST(CsrLabelPropagation, ContractRefusals) {
    Graph f(3, {{0, 1}, {0, 2}}, false), r(3, {{0, 1}, {0, 2}}, true);
    EXPECT_EQ(lp(&f.g, &r.g, -1).st, bk::CsrAlgoStatus::InvalidArgument);
    EXPECT_EQ(lp(&f.g, &r.g, bk::k_csr_lp_max_iterations + 1).st,
              bk::CsrAlgoStatus::InvalidArgument);
    EXPECT_EQ(lp(&f.g, &r.g, 3, 1).st, bk::CsrAlgoStatus::InvalidArgument);  // cap < 2
    Graph small(2, {{0, 1}}, false);
    EXPECT_EQ(lp(&f.g, &small.g, 3).st, bk::CsrAlgoStatus::InvalidArgument);  // n differs
    Graph bad(2, {{0, 1}}, false);
    bad.nbr[0] = 7;                                    // neighbour outside the range
    EXPECT_EQ(lp(&bad.g, nullptr, 3).st, bk::CsrAlgoStatus::MalformedGraph);
    bk::CsrGraph64 empty{};
    int64_t lab[1];
    int32_t it = 0;
    bk::CsrLpScratch sc{lab, lab, lab, 0};
    EXPECT_EQ(bk::csr_label_propagation(&empty, nullptr, 1, &sc, lab, &it),
              bk::CsrAlgoStatus::InvalidArgument);
}

TEST(CsrLabelPropagation, ScratchIsBudgetCharged) {
    bolt::Arena arena;
    bolt::Budget budget("graph_algo_test_mb", 1000);
    bk::CsrAlgoAlloc alloc{&arena, &budget, 0};
    EXPECT_EQ(bk::csr_label_propagation_scratch_bytes(100, 10), 960u);
    EXPECT_NE(alloc.try_alloc(bk::csr_label_propagation_scratch_bytes(100, 10)), nullptr);
    EXPECT_EQ(alloc.try_alloc(bk::csr_label_propagation_scratch_bytes(10, 0)), nullptr);  // 960 + 80 > 1000
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "graph_algo_test_mb");
    alloc.release();
    EXPECT_EQ(budget.used(), 0u);
}
