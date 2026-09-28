// G2GRAPH-178: csr_expand_bounded_sorted narrows a sorted block to the bound
// destination's equal range. It must write exactly the rows the linear
// bound walk writes, in the same order, under every resume point. The
// seeded differential loop at the bottom is the fuzz test: random sorted
// CSRs with parallel edges, self loops, labels, exclusions and output caps.

#include "bolt/kernels/bolt_csr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using bolt::kernels::csr_block_narrow_to_dst;
using bolt::kernels::csr_expand_bounded_sorted;
using bolt::kernels::CsrExpandCursor;
using bolt::kernels::kCsrExpandOutOfRange;

struct Rows {
    std::vector<int64_t> src, edge, dst;
    bool failed = false;
};

struct Graph {
    int64_t n_nodes = 0;
    std::vector<int64_t> off, nbr, eid;
    std::vector<int32_t> lbl;
};

struct Query {
    std::vector<int64_t> srcs, bounds, excl;
    int32_t want_label = -1;
    bool use_labels = false;
    int64_t cap = 1;
};

Rows run(const Graph& g, const Query& q, bool sorted) {
    Rows r;
    std::vector<int64_t> os(static_cast<size_t>(q.cap)),
        oe(static_cast<size_t>(q.cap)), od(static_cast<size_t>(q.cap));
    CsrExpandCursor cur{0, 0};
    const int64_t n = static_cast<int64_t>(q.srcs.size());
    int64_t max_calls = n + 2;              // each call emits cap rows or ends
    for (int64_t s : q.srcs) {
        if (s >= 0 && s < g.n_nodes) {
            max_calls += g.off[static_cast<size_t>(s + 1)] -
                         g.off[static_cast<size_t>(s)];
        }
    }
    for (int64_t call = 0; call < max_calls && cur.src_index < n; ++call) {
        const int64_t got = csr_expand_bounded_sorted(
            q.srcs.data(), n, g.n_nodes, g.off.data(), g.nbr.data(),
            g.eid.data(), q.use_labels ? g.lbl.data() : nullptr, q.want_label,
            q.excl.empty() ? nullptr : q.excl.data(),
            static_cast<int32_t>(q.excl.size()), q.bounds.data(), sorted,
            os.data(), oe.data(), od.data(), q.cap, &cur);
        if (got == kCsrExpandOutOfRange) { r.failed = true; return r; }
        for (int64_t i = 0; i < got; ++i) {
            r.src.push_back(os[static_cast<size_t>(i)]);
            r.edge.push_back(oe[static_cast<size_t>(i)]);
            r.dst.push_back(od[static_cast<size_t>(i)]);
        }
    }
    EXPECT_EQ(cur.src_index, n);
    return r;
}

// 4 nodes; node 0 has a sorted block with a run of parallel edges to 2.
Graph fixed_graph() {
    Graph g;
    g.n_nodes = 4;
    g.off = {0, 6, 6, 7, 8};
    g.nbr = {0, 1, 2, 2, 2, 3, 0, 1};
    g.eid = {10, 11, 12, 13, 14, 15, 16, 17};
    g.lbl = {0, 1, 0, 1, 0, 0, 1, 1};
    return g;
}

TEST(CsrDstProbe, NarrowFindsEqualRange) {
    const Graph g = fixed_graph();
    int64_t b = 0, e = 6;
    csr_block_narrow_to_dst(g.nbr.data(), 2, &b, &e);
    EXPECT_EQ(b, 2);
    EXPECT_EQ(e, 5);
    b = 0; e = 6;
    csr_block_narrow_to_dst(g.nbr.data(), 7, &b, &e);
    EXPECT_EQ(b, e);
    b = 0; e = 6;
    csr_block_narrow_to_dst(g.nbr.data(), -1, &b, &e);
    EXPECT_EQ(b, 0);
    EXPECT_EQ(e, 0);
    b = 6; e = 6;
    csr_block_narrow_to_dst(g.nbr.data(), 2, &b, &e);
    EXPECT_EQ(b, 6);
    EXPECT_EQ(e, 6);
}

TEST(CsrDstProbe, ParallelEdgesResumeAcrossCapOne) {
    const Graph g = fixed_graph();
    Query q;
    q.srcs = {0, 0, 1, 2};
    q.bounds = {2, 3, 0, 0};
    q.cap = 1;
    const Rows lin = run(g, q, false);
    const Rows srt = run(g, q, true);
    ASSERT_FALSE(lin.failed);
    EXPECT_EQ(lin.edge, (std::vector<int64_t>{12, 13, 14, 15, 16}));
    EXPECT_EQ(srt.src, lin.src);
    EXPECT_EQ(srt.edge, lin.edge);
    EXPECT_EQ(srt.dst, lin.dst);
}

TEST(CsrDstProbe, LabelAndExclusionStillApply) {
    const Graph g = fixed_graph();
    Query q;
    q.srcs = {0};
    q.bounds = {2};
    q.use_labels = true;
    q.want_label = 0;
    q.excl = {14};
    q.cap = 4;
    const Rows srt = run(g, q, true);
    EXPECT_EQ(srt.edge, (std::vector<int64_t>{12}));
    EXPECT_EQ(run(g, q, false).edge, srt.edge);
}

// The flag is trusted: only the binary-searched range is walked. On a block
// that violates the contract the linear walk still finds the edge and the
// probe does not, which is what proves the probe actually narrowed.
TEST(CsrDstProbe, SortedFlagIsTrusted) {
    Graph g;
    g.n_nodes = 10;
    g.off = {0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    g.nbr = {9, 1, 2, 3};
    g.eid = {20, 21, 22, 23};
    Query q;
    q.srcs = {0};
    q.bounds = {9};
    q.cap = 4;
    EXPECT_EQ(run(g, q, false).edge, (std::vector<int64_t>{20}));
    EXPECT_TRUE(run(g, q, true).edge.empty());
}

TEST(CsrDstProbe, OutOfRangeSourceStillFailsClosed) {
    const Graph g = fixed_graph();
    Query q;
    q.srcs = {9};
    q.bounds = {2};
    q.cap = 4;
    EXPECT_TRUE(run(g, q, true).failed);
}

// Fuzz: seeded, deterministic. A failure prints the seed and iteration.
uint64_t next_rand(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

Graph random_sorted_graph(uint64_t* rng) {
    Graph g;
    g.n_nodes = 1 + static_cast<int64_t>(next_rand(rng) % 40);
    g.off.push_back(0);
    int64_t next_eid = 1000;
    for (int64_t s = 0; s < g.n_nodes; ++s) {
        const int64_t deg = static_cast<int64_t>(next_rand(rng) % 24);
        std::vector<int64_t> blk;
        const int64_t span = 1 + static_cast<int64_t>(next_rand(rng) % 6);
        for (int64_t k = 0; k < deg; ++k) {
            blk.push_back(static_cast<int64_t>(next_rand(rng) %
                static_cast<uint64_t>(std::min(g.n_nodes, span * 4))));
        }
        std::stable_sort(blk.begin(), blk.end());
        for (int64_t d : blk) {
            g.nbr.push_back(d);
            g.eid.push_back(next_eid++);
            g.lbl.push_back(static_cast<int32_t>(next_rand(rng) % 3));
        }
        g.off.push_back(static_cast<int64_t>(g.nbr.size()));
    }
    return g;
}

Query random_query(const Graph& g, uint64_t* rng) {
    Query q;
    const int64_t n = static_cast<int64_t>(next_rand(rng) % 20);
    for (int64_t i = 0; i < n; ++i) {
        q.srcs.push_back(static_cast<int64_t>(
            next_rand(rng) % static_cast<uint64_t>(g.n_nodes)));
        const int64_t pick = static_cast<int64_t>(next_rand(rng) % 4);
        const int64_t s = q.srcs.back();
        const int64_t deg = g.off[static_cast<size_t>(s + 1)] -
                            g.off[static_cast<size_t>(s)];
        if (pick != 0 && deg > 0) {
            q.bounds.push_back(g.nbr[static_cast<size_t>(
                g.off[static_cast<size_t>(s)] +
                static_cast<int64_t>(next_rand(rng) %
                                     static_cast<uint64_t>(deg)))]);
        } else {
            q.bounds.push_back(static_cast<int64_t>(next_rand(rng) %
                static_cast<uint64_t>(g.n_nodes + 3)) - 1);
        }
    }
    q.use_labels = (next_rand(rng) % 2) == 0;
    q.want_label = static_cast<int32_t>(next_rand(rng) % 4) - 1;
    const int64_t ne = static_cast<int64_t>(next_rand(rng) % 3);
    for (int64_t k = 0; k < ne && !g.eid.empty(); ++k) {
        q.excl.push_back(g.eid[static_cast<size_t>(
            next_rand(rng) % g.eid.size())]);
    }
    q.cap = 1 + static_cast<int64_t>(next_rand(rng) % 9);
    return q;
}

TEST(CsrDstProbe, FuzzSortedProbeMatchesLinearWalk) {
    constexpr int kIters = 20000;
    for (int it = 0; it < kIters; ++it) {
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(it + 1);
        const Graph g = random_sorted_graph(&rng);
        const Query q = random_query(g, &rng);
        const Rows lin = run(g, q, false);
        const Rows srt = run(g, q, true);
        ASSERT_EQ(lin.failed, srt.failed) << "iteration " << it;
        ASSERT_EQ(srt.src, lin.src) << "iteration " << it;
        ASSERT_EQ(srt.edge, lin.edge) << "iteration " << it;
        ASSERT_EQ(srt.dst, lin.dst) << "iteration " << it;
    }
}

}  // namespace
