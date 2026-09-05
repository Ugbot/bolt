// test_bolt_csr_shortest.cpp — GTest suite for bolt_csr_shortest.h.
//
// ORACLE PROVENANCE:
//   - The deterministic fixture's distances were printed by
//     tests/csr_bfs_enumerate.py (plain Python BFS, written from the
//     definition of a shortest path, not from this kernel).
//   - The randomized 1,000-node / 10,000-edge cases compare against
//     data/csr_bfs_random.txt, whose SP lines come from **networkx** —
//     an engine that shares no code with bolt. 300 pairs, of which 7 are
//     genuinely unreachable, so the -1 branch is exercised by real data.
//
// Covered: exact lengths, s == t, proven-unreachable vs cap-reached (the two
// meanings of -1, which MUST NOT be conflated), witness path validity against
// the real CSR, WitnessTruncated, scratch reuse across queries (the epoch
// stamp), and argument rejection.

#include <gtest/gtest.h>

#include "csr_bfs_fixture.h"
#include "bolt/kernels/bolt_csr_shortest.h"

#include <chrono>
#include <cstdint>
#include <vector>

using bolt::kernels::CsrBfsGraph;
using bolt::kernels::CsrShortestResult;
using bolt::kernels::CsrShortestScratch;
using bolt::kernels::CsrShortestStatus;
using bolt::kernels::csr_shortest_bidir;

namespace {

// Caller-owned scratch for a graph of n_nodes. `witness` selects whether the
// parent/pedge arrays are supplied at all — the kernel rejects a mixed set.
struct SpScratch {
    std::vector<int32_t> stamp_f, stamp_r, dist_f, dist_r;
    std::vector<int64_t> qf, qr, pf, pr, ef, er;
    CsrShortestScratch sc{};

    SpScratch(int64_t n, bool witness) {
        const size_t z = static_cast<size_t>(n);
        stamp_f.assign(z, 0); stamp_r.assign(z, 0);
        dist_f.assign(z, 0);  dist_r.assign(z, 0);
        qf.assign(z, 0);      qr.assign(z, 0);
        sc.stamp_f = stamp_f.data(); sc.stamp_r = stamp_r.data();
        sc.dist_f  = dist_f.data();  sc.dist_r  = dist_r.data();
        sc.queue_f = qf.data();      sc.queue_r = qr.data();
        sc.epoch   = 0;
        if (witness) {
            pf.assign(z, 0); pr.assign(z, 0); ef.assign(z, 0); er.assign(z, 0);
            sc.parent_f = pf.data(); sc.parent_r = pr.data();
            sc.pedge_f  = ef.data(); sc.pedge_r  = er.data();
        }
    }
};

// Build the reverse CSR of the KnownBfsCsr fixture (same edge ids).
struct KnownRevCsr {
    // reverse edges: 1->0(100), 1->0(101), 2->1(102), 0->2(103), 4->2(105),
    // 3->3(104). Grouped by the reversed source (the original dst).
    int64_t off[6] = {0, 1, 3, 4, 5, 6};
    int64_t nbr[6] = {2, 0, 0, 1, 3, 2};
    int64_t eid[6] = {103, 100, 101, 102, 104, 105};

    CsrBfsGraph graph() const {
        CsrBfsGraph g{};
        g.off = off; g.neighbors = nbr; g.edge_ids = eid;
        g.edge_labels = nullptr; g.n_nodes = KnownBfsCsr::kNodes;
        g.want_label = -1;
        return g;
    }
};

// Walk the witness against the FORWARD CSR: every consecutive node pair must
// be joined by the named relationship. A length that is right while the path
// is fabricated is exactly the class of wrong answer this checks for.
void expect_witness_is_a_real_path(const CsrBfsGraph& fwd, int64_t s, int64_t t,
                                   const int64_t* nodes, const int64_t* edges,
                                   int32_t len) {
    ASSERT_GE(len, 0);
    EXPECT_EQ(nodes[0], s);
    EXPECT_EQ(nodes[len], t);
    for (int32_t i = 0; i < len; ++i) {
        const int64_t u = nodes[i], v = nodes[i + 1], e = edges[i];
        bool found = false;
        for (int64_t j = fwd.off[u]; j < fwd.off[u + 1]; ++j)
            if (fwd.neighbors[j] == v && fwd.edge_ids[j] == e) found = true;
        EXPECT_TRUE(found) << "witness step " << i << ": no edge " << e
                           << " from " << u << " to " << v;
    }
}

RandomGraph& random_graph() {
    static RandomGraph g;
    static bool tried = false;
    if (!tried) { tried = true; load_random_graph(random_graph_path().c_str(), &g); }
    return g;
}

}  // namespace

// ============================================================================
// Deterministic fixture — distances from csr_bfs_enumerate.py
// ============================================================================
//
//   from 0: [0, 1, 2, -1, 3]
//   from 1: [2, 0, 1, -1, 2]
//   from 2: [1, 2, 0, -1, 1]
//   from 3: [-1, -1, -1, 0, -1]
//   from 4: [-1, -1, -1, -1, 0]
TEST(CsrShortestBidir, FixtureDistanceMatrix) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    const int32_t want[5][5] = {
        { 0,  1,  2, -1,  3},
        { 2,  0,  1, -1,  2},
        { 1,  2,  0, -1,  1},
        {-1, -1, -1,  0, -1},
        {-1, -1, -1, -1,  0},
    };
    SpScratch buf(KnownBfsCsr::kNodes, /*witness=*/true);
    std::vector<int64_t> nodes(32), edges(32);
    for (int64_t s = 0; s < KnownBfsCsr::kNodes; ++s) {
        for (int64_t t = 0; t < KnownBfsCsr::kNodes; ++t) {
            SCOPED_TRACE("s=" + std::to_string(s) + " t=" + std::to_string(t));
            CsrShortestResult out{};
            const CsrShortestStatus st =
                csr_shortest_bidir(&fwd, &rev, s, t, /*max_hops=*/8, &buf.sc,
                                   nodes.data(), edges.data(), 32, &out);
            ASSERT_EQ(st, CsrShortestStatus::Ok)
                << "every pair is decided well inside max_hops=8";
            EXPECT_EQ(out.length, want[s][t]);
            if (out.length < 0) {
                EXPECT_EQ(out.witness_len, 0);
                continue;
            }
            EXPECT_EQ(out.witness_len, out.length + 1);
            expect_witness_is_a_real_path(fwd, s, t, nodes.data(), edges.data(),
                                          out.length);
        }
    }
}

// s == t is length 0 with a one-node witness, not "unreachable".
TEST(CsrShortestBidir, SameNodeIsZeroLength) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    SpScratch buf(KnownBfsCsr::kNodes, true);
    int64_t nodes[4] = {-1, -1, -1, -1};
    CsrShortestResult out{};
    ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, 4, 4, 8, &buf.sc, nodes, nullptr,
                                 4, &out), CsrShortestStatus::Ok);
    EXPECT_EQ(out.length, 0);
    EXPECT_EQ(out.witness_len, 1);
    EXPECT_EQ(nodes[0], 4);
}

// THE distinction: -1 under Ok means PROVEN unreachable; -1 under
// HopCapReached means undetermined. Same pair, two caps, two statuses.
TEST(CsrShortestBidir, CapReachedIsNotUnreachable) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    SpScratch buf(KnownBfsCsr::kNodes, false);
    CsrShortestResult out{};

    // 0 -> 4 is genuinely 3 hops. A cap of 2 must NOT report "no path".
    ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 4, /*max_hops=*/2, &buf.sc,
                                 nullptr, nullptr, 0, &out),
              CsrShortestStatus::HopCapReached);
    EXPECT_EQ(out.length, -1);

    ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 4, /*max_hops=*/3, &buf.sc,
                                 nullptr, nullptr, 0, &out),
              CsrShortestStatus::Ok);
    EXPECT_EQ(out.length, 3);

    // 0 -> 3 has no path at all: Ok, length -1, PROVEN by frontier exhaustion
    // even though the cap is generous.
    ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 3, /*max_hops=*/64, &buf.sc,
                                 nullptr, nullptr, 0, &out),
              CsrShortestStatus::Ok);
    EXPECT_EQ(out.length, -1);
}

// A witness that does not fit is a named refusal; the length still stands and
// nothing is written past the cap.
TEST(CsrShortestBidir, WitnessTruncatedIsNamedNotSilent) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    SpScratch buf(KnownBfsCsr::kNodes, true);
    int64_t nodes[8];
    for (int i = 0; i < 8; ++i) nodes[i] = -7;
    CsrShortestResult out{};
    // 0 -> 4 needs 4 node slots; give it 3.
    ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 4, 8, &buf.sc, nodes, nullptr,
                                 /*out_path_cap=*/3, &out),
              CsrShortestStatus::WitnessTruncated);
    EXPECT_EQ(out.length, 3) << "the LENGTH is still proven";
    EXPECT_EQ(out.witness_len, 0);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(nodes[i], -7) << "wrote past the cap";
}

// The epoch stamp must isolate consecutive queries sharing one scratch.
// Running the whole matrix twice on ONE scratch must give the same answers.
TEST(CsrShortestBidir, ScratchIsReusableAcrossQueries) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    SpScratch buf(KnownBfsCsr::kNodes, false);
    int32_t first[25], second[25];
    for (int pass = 0; pass < 2; ++pass) {
        int32_t* dst = (pass == 0) ? first : second;
        for (int64_t s = 0; s < 5; ++s)
            for (int64_t t = 0; t < 5; ++t) {
                CsrShortestResult out{};
                ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, s, t, 8, &buf.sc,
                                             nullptr, nullptr, 0, &out),
                          CsrShortestStatus::Ok);
                dst[s * 5 + t] = out.length;
            }
    }
    for (int i = 0; i < 25; ++i) EXPECT_EQ(first[i], second[i]) << i;
    EXPECT_GE(buf.sc.epoch, 50) << "each query must consume its own epoch";
}

TEST(CsrShortestBidir, RejectsBadArguments) {
    KnownBfsCsr f; KnownRevCsr r;
    const CsrBfsGraph fwd = f.graph(), rev = r.graph();
    SpScratch buf(KnownBfsCsr::kNodes, false);
    CsrShortestResult out{};
    EXPECT_EQ(csr_shortest_bidir(nullptr, &rev, 0, 1, 8, &buf.sc, nullptr,
                                 nullptr, 0, &out),
              CsrShortestStatus::InvalidArgument);
    EXPECT_EQ(csr_shortest_bidir(&fwd, &rev, -1, 1, 8, &buf.sc, nullptr,
                                 nullptr, 0, &out),
              CsrShortestStatus::InvalidArgument);
    EXPECT_EQ(csr_shortest_bidir(&fwd, &rev, 0, KnownBfsCsr::kNodes, 8, &buf.sc,
                                 nullptr, nullptr, 0, &out),
              CsrShortestStatus::InvalidArgument);
    // Witness asked for, but no parent scratch supplied -> refused, not
    // silently answered without a path.
    int64_t nodes[8];
    EXPECT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 1, 8, &buf.sc, nodes, nullptr,
                                 8, &out),
              CsrShortestStatus::InvalidArgument);
    // A half-supplied witness scratch is also refused.
    SpScratch mixed(KnownBfsCsr::kNodes, true);
    mixed.sc.pedge_r = nullptr;
    EXPECT_EQ(csr_shortest_bidir(&fwd, &rev, 0, 1, 8, &mixed.sc, nodes, nullptr,
                                 8, &out),
              CsrShortestStatus::InvalidArgument);
}

// ============================================================================
// Randomized graph vs networkx
// ============================================================================

TEST(CsrShortestRandom, MatchesNetworkxOnEveryPair) {
    RandomGraph& rg = random_graph();
    ASSERT_TRUE(rg.loaded) << "missing " << random_graph_path()
                           << " — regenerate with csr_bfs_enumerate.py "
                              "--emit-random; a missing fixture must FAIL, "
                              "never silently skip";
    ASSERT_GE(rg.sp.size(), 100u);
    const CsrBfsGraph fwd = rg.fwd(), rev = rg.rev();
    SpScratch buf(rg.n_nodes, /*witness=*/true);
    std::vector<int64_t> nodes(64), edges(64);
    int reachable = 0, unreachable = 0;
    for (const SpExpect& e : rg.sp) {
        SCOPED_TRACE("s=" + std::to_string(e.s) + " t=" + std::to_string(e.t));
        CsrShortestResult out{};
        ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, e.s, e.t, /*max_hops=*/64,
                                     &buf.sc, nodes.data(), edges.data(), 64,
                                     &out),
                  CsrShortestStatus::Ok);
        ASSERT_EQ(out.length, e.len);
        if (e.len < 0) { ++unreachable; continue; }
        ++reachable;
        expect_witness_is_a_real_path(fwd, e.s, e.t, nodes.data(), edges.data(),
                                      out.length);
    }
    // Both branches must be exercised, or the -1 path is untested.
    EXPECT_GT(reachable, 100);
    EXPECT_GT(unreachable, 0);
}

// ============================================================================
// Randomized differential vs a textbook single-source BFS
// ============================================================================
//
// SECONDARY oracle, and labelled as such: networkx (above) is the independent
// engine, this is a DIFFERENT ALGORITHM (single-ended BFS, no reverse CSR, no
// meeting logic) run over thousands of small graphs to cover shapes the 300
// networkx pairs cannot reach — tiny components, self-loops, parallel edges,
// wildly asymmetric in/out degree.
//
// It exists because of a measured finding: the bidirectional STOP RULE turns
// out to be insensitive to its constant (stop-at-first-meeting, `mu<=df+dr`
// and `mu<=df+dr+1` all give identical answers — see the CLAIM in
// bolt_csr_shortest.h), so no single hand-written pair can discriminate a
// broken termination. Breadth is what discriminates here, not cleverness.
namespace {

struct TinyGraph {
    int64_t n = 0;
    std::vector<int64_t> src, dst, eid, off, nbr, ids, scratch;
    std::vector<int64_t> roff, rnbr, rids, rscratch;
};

// Deterministic 64-bit LCG (same constants the Python enumerator uses).
struct Lcg {
    uint64_t s;
    uint64_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s >> 11;
    }
};

bool make_tiny(Lcg* rnd, TinyGraph* g) {
    g->n = static_cast<int64_t>(rnd->next() % 40) + 2;
    const int64_t m = static_cast<int64_t>(
        rnd->next() % static_cast<uint64_t>(3 * g->n)) + 1;
    g->src.clear(); g->dst.clear(); g->eid.clear();
    for (int64_t i = 0; i < m; ++i) {
        g->src.push_back(static_cast<int64_t>(rnd->next() % static_cast<uint64_t>(g->n)));
        g->dst.push_back(static_cast<int64_t>(rnd->next() % static_cast<uint64_t>(g->n)));
        g->eid.push_back(1000 + i);
    }
    const size_t zn = static_cast<size_t>(g->n), zm = static_cast<size_t>(m);
    g->off.assign(zn + 1, 0);  g->nbr.assign(zm, 0);  g->ids.assign(zm, 0);
    g->scratch.assign(zn, 0);
    g->roff.assign(zn + 1, 0); g->rnbr.assign(zm, 0); g->rids.assign(zm, 0);
    g->rscratch.assign(zn, 0);
    return bolt::kernels::csr_build(g->src.data(), g->dst.data(), g->eid.data(),
                                    m, g->n, g->off.data(), g->nbr.data(),
                                    g->ids.data(), g->scratch.data()) &&
           bolt::kernels::csr_build(g->dst.data(), g->src.data(), g->eid.data(),
                                    m, g->n, g->roff.data(), g->rnbr.data(),
                                    g->rids.data(), g->rscratch.data());
}

// Textbook single-source BFS over the forward CSR. Deliberately shares no
// structure with the kernel: one queue, one direction, no stamps, no meeting.
void bfs_from(const TinyGraph& g, int64_t s, std::vector<int32_t>* dist) {
    dist->assign(static_cast<size_t>(g.n), -1);
    std::vector<int64_t> q;
    q.push_back(s);
    (*dist)[static_cast<size_t>(s)] = 0;
    for (size_t head = 0; head < q.size(); ++head) {
        const int64_t u = q[head];
        for (int64_t j = g.off[static_cast<size_t>(u)];
             j < g.off[static_cast<size_t>(u) + 1]; ++j) {
            const int64_t v = g.nbr[static_cast<size_t>(j)];
            if ((*dist)[static_cast<size_t>(v)] >= 0) continue;
            (*dist)[static_cast<size_t>(v)] =
                (*dist)[static_cast<size_t>(u)] + 1;
            q.push_back(v);
        }
    }
}

}  // namespace

TEST(CsrShortestRandom, DifferentialAgainstPlainBfsOnManySmallGraphs) {
    Lcg rnd{0xC0FFEE1234567ULL};
    TinyGraph g;
    std::vector<int32_t> dist;
    std::vector<int64_t> nodes(256), edges(256);
    int64_t pairs = 0, reachable = 0, unreachable = 0, self_loops = 0;
    for (int trial = 0; trial < 1500; ++trial) {
        ASSERT_TRUE(make_tiny(&rnd, &g)) << "trial " << trial;
        const CsrBfsGraph fwd = [&] {
            CsrBfsGraph x{}; x.off = g.off.data(); x.neighbors = g.nbr.data();
            x.edge_ids = g.ids.data(); x.edge_labels = nullptr;
            x.n_nodes = g.n; x.want_label = -1; return x; }();
        const CsrBfsGraph rev = [&] {
            CsrBfsGraph x{}; x.off = g.roff.data(); x.neighbors = g.rnbr.data();
            x.edge_ids = g.rids.data(); x.edge_labels = nullptr;
            x.n_nodes = g.n; x.want_label = -1; return x; }();
        for (size_t i = 0; i < g.src.size(); ++i)
            if (g.src[i] == g.dst[i]) ++self_loops;
        SpScratch buf(g.n, /*witness=*/true);
        for (int64_t s = 0; s < g.n; ++s) {
            bfs_from(g, s, &dist);
            for (int64_t t = 0; t < g.n; ++t) {
                CsrShortestResult out{};
                ASSERT_EQ(csr_shortest_bidir(&fwd, &rev, s, t,
                                             /*max_hops=*/1024, &buf.sc,
                                             nodes.data(), edges.data(), 256,
                                             &out),
                          CsrShortestStatus::Ok)
                    << "trial " << trial << " s=" << s << " t=" << t;
                ASSERT_EQ(out.length, dist[static_cast<size_t>(t)])
                    << "trial " << trial << " s=" << s << " t=" << t;
                ++pairs;
                if (out.length < 0) { ++unreachable; continue; }
                ++reachable;
                expect_witness_is_a_real_path(fwd, s, t, nodes.data(),
                                              edges.data(), out.length);
            }
        }
    }
    // The sweep must actually have covered the interesting shapes.
    EXPECT_GT(pairs, 100000);
    EXPECT_GT(reachable, 1000);
    EXPECT_GT(unreachable, 1000);
    EXPECT_GT(self_loops, 0);
    std::printf("[diff] %lld pairs (%lld reachable / %lld unreachable), "
                "%lld self-loop edges\n",
                static_cast<long long>(pairs), static_cast<long long>(reachable),
                static_cast<long long>(unreachable),
                static_cast<long long>(self_loops));
}

// Perf smoke — PRINTED, NOT GATED (this wave).
TEST(CsrShortestRandom, PerfSmokeNotGated) {
    RandomGraph& rg = random_graph();
    ASSERT_TRUE(rg.loaded) << "missing " << random_graph_path();
    const CsrBfsGraph fwd = rg.fwd(), rev = rg.rev();
    SpScratch buf(rg.n_nodes, false);
    int64_t decided = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 20; ++rep)
        for (const SpExpect& e : rg.sp) {
            CsrShortestResult out{};
            if (csr_shortest_bidir(&fwd, &rev, e.s, e.t, 64, &buf.sc, nullptr,
                                   nullptr, 0, &out) == CsrShortestStatus::Ok)
                ++decided;
        }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    ASSERT_GT(decided, 0);
    std::printf("[perf] csr_shortest_bidir: %lld queries, %.0f ns/query\n",
                static_cast<long long>(decided), ns / static_cast<double>(decided));
}
