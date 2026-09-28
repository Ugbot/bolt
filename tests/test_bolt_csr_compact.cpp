// Compact CSR (CsrGraph32: uint32 neighbours/edge ids, int64 offsets over a
// node WINDOW) walks exactly like the int64 whole-id-space layout. Every
// kernel is checked against the int64 run on the networkx-oracled random
// graph and against the oracle itself, with the window trimmed so nodes at
// both ends of the id space fall outside it.

#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_bfs.h"
#include "bolt/kernels/bolt_csr_shortest.h"
#include "csr_bfs_fixture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace bk = bolt::kernels;

namespace {

struct Compact {
    std::vector<int64_t>  off;
    std::vector<uint32_t> nbr, eid;
    int64_t node_lo = 0;
    int64_t n_win   = 0;
    int64_t n_nodes = 0;

    bk::CsrGraph32 graph() const {
        bk::CsrGraph32 g{};
        g.off = off.data(); g.neighbors = nbr.data(); g.edge_ids = eid.data();
        g.n_nodes = n_nodes; g.want_label = -1;
        g.node_lo = node_lo; g.n_win = n_win;
        return g;
    }
};

// Narrow an int64 CSR to its populated node window.
Compact compact_of(const std::vector<int64_t>& off,
                   const std::vector<int64_t>& nbr,
                   const std::vector<int64_t>& ids, int64_t n_nodes) {
    Compact c;
    c.n_nodes = n_nodes;
    int64_t lo = n_nodes, hi = -1;
    for (int64_t k = 0; k < n_nodes; ++k) {
        if (off[k + 1] > off[k]) { if (lo == n_nodes) lo = k; hi = k; }
    }
    c.node_lo = (hi < 0) ? 0 : lo;
    c.n_win   = (hi < 0) ? 0 : hi - lo + 1;
    c.off.resize(static_cast<size_t>(c.n_win + 1));
    const int64_t base = (hi < 0) ? 0 : off[lo];
    for (int64_t r = 0; r <= c.n_win; ++r)
        c.off[static_cast<size_t>(r)] = off[static_cast<size_t>(c.node_lo + r)] - base;
    c.nbr.assign(nbr.begin() + base, nbr.begin() + off[static_cast<size_t>(c.node_lo + c.n_win)]);
    c.eid.clear();
    for (int64_t j = base; j < off[static_cast<size_t>(c.node_lo + c.n_win)]; ++j)
        c.eid.push_back(static_cast<uint32_t>(ids[static_cast<size_t>(j)]));
    return c;
}

RandomGraph& rg() {
    static RandomGraph g;
    static bool tried = false;
    if (!tried) { tried = true; load_random_graph(random_graph_path().c_str(), &g); }
    return g;
}

struct Emit { std::vector<int64_t> s, e, d; };

Emit expand_all(const int64_t* src, int64_t n, const auto* g, int64_t cap) {
    Emit out;
    std::vector<int64_t> os(static_cast<size_t>(cap)), oe(static_cast<size_t>(cap)),
        od(static_cast<size_t>(cap));
    bk::CsrExpandCursor cur{};
    for (int guard = 0; guard < 1000000 && cur.src_index < n; ++guard) {
        const int64_t w = bk::csr_expand_graph(src, n, g, nullptr, 0, nullptr,
                                               os.data(), oe.data(), od.data(),
                                               cap, &cur);
        EXPECT_GE(w, 0);
        if (w < 0) break;
        for (int64_t i = 0; i < w; ++i) {
            out.s.push_back(os[static_cast<size_t>(i)]);
            out.e.push_back(oe[static_cast<size_t>(i)]);
            out.d.push_back(od[static_cast<size_t>(i)]);
        }
    }
    return out;
}

}  // namespace

TEST(CsrCompact, WindowExcludesBothEnds) {
    // Nodes 0..9, edges only out of 3..6: the window must be [3, 7).
    std::vector<int64_t> src = {3, 3, 5, 6}, dst = {9, 0, 4, 3}, eid = {0, 1, 2, 3};
    std::vector<int64_t> off(11), nbr(4), ids(4), scr(10);
    ASSERT_TRUE(bk::csr_build(src.data(), dst.data(), eid.data(), 4, 10,
                              off.data(), nbr.data(), ids.data(), scr.data()));
    const Compact c = compact_of(off, nbr, ids, 10);
    EXPECT_EQ(c.node_lo, 3);
    EXPECT_EQ(c.n_win, 4);
    const bk::CsrGraph32 g = c.graph();
    int64_t b = -1, e = -1;
    bk::csr_graph_block(&g, 0, &b, &e);  EXPECT_EQ(e - b, 0);
    bk::csr_graph_block(&g, 9, &b, &e);  EXPECT_EQ(e - b, 0);
    bk::csr_graph_block(&g, 3, &b, &e);  EXPECT_EQ(e - b, 2);
    bk::csr_graph_block(&g, 4, &b, &e);  EXPECT_EQ(e - b, 0);
    const int64_t all[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const Emit got = expand_all(all, 10, &g, 3);
    ASSERT_EQ(got.s.size(), 4u);
    // A source outside [0, n_nodes) is still an error, not an empty block.
    const int64_t bad[1] = {10};
    bk::CsrExpandCursor cur{};
    int64_t os[4], oe[4], od[4];
    EXPECT_EQ(bk::csr_expand_graph(bad, 1, &g, nullptr, 0, nullptr, os, oe, od,
                                   4, &cur),
              bk::kCsrExpandOutOfRange);
}

TEST(CsrCompact, ExpandMatchesInt64OnRandomGraph) {
    RandomGraph& r = rg();
    ASSERT_TRUE(r.loaded) << "missing " << random_graph_path();
    const Compact c = compact_of(r.off, r.nbr, r.ids, r.n_nodes);
    ASSERT_GT(c.node_lo + (r.n_nodes - c.node_lo - c.n_win), -1);
    const bk::CsrGraph32 g32 = c.graph();
    const bk::CsrBfsGraph g64 = r.fwd();
    std::vector<int64_t> all(static_cast<size_t>(r.n_nodes));
    for (int64_t i = 0; i < r.n_nodes; ++i) all[static_cast<size_t>(i)] = i;
    const Emit a = expand_all(all.data(), r.n_nodes, &g64, 7);
    const Emit b = expand_all(all.data(), r.n_nodes, &g32, 7);
    ASSERT_EQ(a.s.size(), static_cast<size_t>(r.src.size()));
    EXPECT_EQ(a.s, b.s);
    EXPECT_EQ(a.e, b.e);
    EXPECT_EQ(a.d, b.d);
}

TEST(CsrCompact, BfsCountsMatchInt64AndOracle) {
    RandomGraph& r = rg();
    ASSERT_TRUE(r.loaded);
    const Compact c = compact_of(r.off, r.nbr, r.ids, r.n_nodes);
    const bk::CsrGraph32 g32 = c.graph();
    const bk::CsrBfsGraph g64 = r.fwd();
    ASSERT_FALSE(r.paths.empty());
    for (const PathsExpect& e : r.paths) {
        const int64_t src[1] = {e.src};
        const int64_t want[3] = {e.walk, e.trail, e.simple};
        const bk::CsrPathSemantics sem[3] = {bk::CsrPathSemantics::Walk,
                                             bk::CsrPathSemantics::Trail,
                                             bk::CsrPathSemantics::Simple};
        for (int k = 0; k < 3; ++k) {
            const bk::CsrBfsParams p = bfs_params(1, e.hops, sem[k], 1 << 26);
            BfsScratchBuf b1, b2;
            bk::CsrBfsScratch s1 = b1.view(), s2 = b2.view();
            bk::CsrBfsCursor c1{}, c2{};
            int64_t n1 = -1, n2 = -1;
            ASSERT_EQ(bk::csr_bfs_count(src, 1, &g64, &p, &s1, &n1, &c1),
                      bk::CsrBfsStatus::Ok);
            ASSERT_EQ(bk::csr_bfs_count(src, 1, &g32, &p, &s2, &n2, &c2),
                      bk::CsrBfsStatus::Ok);
            EXPECT_EQ(n1, n2);
            EXPECT_EQ(n2, want[k]) << "src=" << e.src << " hops=" << e.hops;
        }
    }
}

TEST(CsrCompact, ShortestMatchesOracle) {
    RandomGraph& r = rg();
    ASSERT_TRUE(r.loaded);
    const Compact f = compact_of(r.off, r.nbr, r.ids, r.n_nodes);
    const Compact v = compact_of(r.roff, r.rnbr, r.rids, r.n_nodes);
    const bk::CsrGraph32 fwd = f.graph();
    const bk::CsrGraph32 rev = v.graph();
    const size_t n = static_cast<size_t>(r.n_nodes);
    std::vector<int32_t> sf(n, 0), sr(n, 0), df(n), dr(n);
    std::vector<int64_t> qf(n), qr(n), pf(n), pr(n), ef(n), er(n);
    bk::CsrShortestScratch sc{};
    sc.stamp_f = sf.data(); sc.stamp_r = sr.data();
    sc.dist_f = df.data(); sc.dist_r = dr.data();
    sc.queue_f = qf.data(); sc.queue_r = qr.data();
    sc.parent_f = pf.data(); sc.parent_r = pr.data();
    sc.pedge_f = ef.data(); sc.pedge_r = er.data();
    ASSERT_FALSE(r.sp.empty());
    for (const SpExpect& e : r.sp) {
        int64_t nodes[64];
        bk::CsrShortestResult res{};
        ASSERT_EQ(bk::csr_shortest_bidir(&fwd, &rev, e.s, e.t, 64, &sc, nodes,
                                         nullptr, 64, &res),
                  bk::CsrShortestStatus::Ok);
        EXPECT_EQ(res.length, e.len) << e.s << "->" << e.t;
    }
}
