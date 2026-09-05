// test_bolt_csr_bfs.cpp — GTest suite for bolt_csr_bfs.h.
//
// ORACLE PROVENANCE (this is the point of the suite, not a footnote):
//   - Every literal row in the deterministic pins below was PRINTED BY
//     tests/csr_bfs_enumerate.py, a pure-Python re-derivation written from the
//     WALK/TRAIL/SIMPLE definitions. Run it and diff:
//         python3 tests/csr_bfs_enumerate.py
//   - The randomized 1,000-node / 10,000-edge cases compare against
//     data/csr_bfs_random.txt, whose REACH and SP expectations come from
//     **networkx** — an engine that shares no line of code with bolt.
//   The kernel and its expectation therefore do not have a common author.
//
// Covered: WALK / TRAIL / SIMPLE separation, zero-length paths, the hop
// window, the label filter, multi-source drain, out_cap resume, visit_budget
// suspension, the per-source cap as a LOUD return code, csr_bfs_count parity
// with csr_bfs_expand, and argument rejection.

#include <gtest/gtest.h>

#include "csr_bfs_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <set>
#include <vector>

using bolt::kernels::CsrBfsCursor;
using bolt::kernels::CsrBfsGraph;
using bolt::kernels::CsrBfsParams;
using bolt::kernels::CsrBfsScratch;
using bolt::kernels::CsrBfsStatus;
using bolt::kernels::CsrPathSemantics;
using bolt::kernels::csr_bfs_count;
using bolt::kernels::csr_bfs_done;
using bolt::kernels::csr_bfs_expand;

namespace {

struct Row { int64_t src, dst; int32_t hops; };

// Drain the whole expansion in ONE call with a generous cap.
CsrBfsStatus expand_all(const int64_t* src_ids, int64_t n, const CsrBfsGraph& g,
                        const CsrBfsParams& p, std::vector<Row>* out,
                        int64_t out_cap = 4096) {
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    std::vector<int64_t> os(static_cast<size_t>(out_cap));
    std::vector<int64_t> od(static_cast<size_t>(out_cap));
    std::vector<int32_t> oh(static_cast<size_t>(out_cap));
    int64_t rows = -1;
    const CsrBfsStatus st = csr_bfs_expand(src_ids, n, &g, &p, &sc, os.data(),
                                           od.data(), oh.data(), out_cap, &cur,
                                           &rows);
    EXPECT_GE(rows, 0) << "*out_rows must be written on every path";
    out->clear();
    for (int64_t i = 0; i < rows; ++i)
        out->push_back(Row{os[static_cast<size_t>(i)], od[static_cast<size_t>(i)],
                           oh[static_cast<size_t>(i)]});
    if (st == CsrBfsStatus::Ok) EXPECT_TRUE(csr_bfs_done(&cur, n));
    return st;
}

// Drain in `chunk`-row slices, exercising the resume cursor.
std::vector<Row> expand_chunked(const int64_t* src_ids, int64_t n,
                                const CsrBfsGraph& g, const CsrBfsParams& p,
                                int64_t chunk) {
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    std::vector<Row> out;
    std::vector<int64_t> os(static_cast<size_t>(chunk));
    std::vector<int64_t> od(static_cast<size_t>(chunk));
    std::vector<int32_t> oh(static_cast<size_t>(chunk));
    for (int guard = 0; guard < 100000 && !csr_bfs_done(&cur, n); ++guard) {
        int64_t rows = -1;
        const CsrBfsStatus st =
            csr_bfs_expand(src_ids, n, &g, &p, &sc, os.data(), od.data(),
                           oh.data(), chunk, &cur, &rows);
        EXPECT_EQ(st, CsrBfsStatus::Ok);
        if (st != CsrBfsStatus::Ok) break;
        for (int64_t i = 0; i < rows; ++i)
            out.push_back(Row{os[static_cast<size_t>(i)],
                              od[static_cast<size_t>(i)],
                              oh[static_cast<size_t>(i)]});
    }
    EXPECT_TRUE(csr_bfs_done(&cur, n)) << "chunked drain never finished";
    return out;
}

void expect_rows(const std::vector<Row>& got, const std::vector<Row>& want) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(got[i].src, want[i].src)  << "row " << i;
        EXPECT_EQ(got[i].dst, want[i].dst)  << "row " << i;
        EXPECT_EQ(got[i].hops, want[i].hops) << "row " << i;
    }
}

}  // namespace

// ============================================================================
// Deterministic fixture — literals from csr_bfs_enumerate.py
// ============================================================================

// WALK [*0..1] from node 0. Two PARALLEL edges 0->1 give TWO rows for the same
// endpoint: a var-length match is per PATH, not per endpoint.
TEST(CsrBfsExpand, WalkZeroToOneFromNodeZero) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(0, 1, CsrPathSemantics::Walk);
    const int64_t src[1] = {0};
    std::vector<Row> got;
    ASSERT_EQ(expand_all(src, 1, g, p, &got), CsrBfsStatus::Ok);
    expect_rows(got, {{0, 0, 0}, {0, 1, 1}, {0, 1, 1}});
}

// The zero-length binding: `-[*0..0]->` binds (a, a) for every source.
TEST(CsrBfsExpand, ZeroLengthBindsSelfPair) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(0, 0, CsrPathSemantics::Walk);
    const int64_t src[5] = {0, 1, 2, 3, 4};
    std::vector<Row> got;
    ASSERT_EQ(expand_all(src, 5, g, p, &got), CsrBfsStatus::Ok);
    expect_rows(got, {{0, 0, 0}, {1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0}});
}

// WALK vs SIMPLE on the 0->1->2->0 cycle: WALK returns to 0 at hop 3, SIMPLE
// refuses to revisit a node already on the path. 8 rows vs 6.
TEST(CsrBfsExpand, WalkAndSimpleDivergeOnACycle) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[1] = {0};
    std::vector<Row> walk, simple;
    ASSERT_EQ(expand_all(src, 1, g, bfs_params(1, 3, CsrPathSemantics::Walk),
                         &walk), CsrBfsStatus::Ok);
    expect_rows(walk, {{0, 1, 1}, {0, 2, 2}, {0, 0, 3}, {0, 4, 3},
                       {0, 1, 1}, {0, 2, 2}, {0, 0, 3}, {0, 4, 3}});
    ASSERT_EQ(expand_all(src, 1, g, bfs_params(1, 3, CsrPathSemantics::Simple),
                         &simple), CsrBfsStatus::Ok);
    expect_rows(simple, {{0, 1, 1}, {0, 2, 2}, {0, 4, 3},
                         {0, 1, 1}, {0, 2, 2}, {0, 4, 3}});
}

// The self-loop 3 -104-> 3 is where the three semantics all differ, and it is
// the exact shape SIMPLE silently dropped before G2GRAPH-L3(a):
//   WALK   walks the loop over and over -> 3 rows at [*1..3]
//   TRAIL  may take relationship 104 once -> 1 row
//   SIMPLE cannot even take it once (the endpoint IS the start node) -> 0 rows
TEST(CsrBfsExpand, SelfLoopSeparatesAllThreeSemantics) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[1] = {3};
    std::vector<Row> walk, trail, simple;
    ASSERT_EQ(expand_all(src, 1, g, bfs_params(1, 3, CsrPathSemantics::Walk),
                         &walk), CsrBfsStatus::Ok);
    expect_rows(walk, {{3, 3, 1}, {3, 3, 2}, {3, 3, 3}});
    ASSERT_EQ(expand_all(src, 1, g, bfs_params(1, 3, CsrPathSemantics::Trail),
                         &trail), CsrBfsStatus::Ok);
    expect_rows(trail, {{3, 3, 1}});
    ASSERT_EQ(expand_all(src, 1, g, bfs_params(1, 3, CsrPathSemantics::Simple),
                         &simple), CsrBfsStatus::Ok);
    EXPECT_TRUE(simple.empty());
}

// Multi-source drain in one call: 13 rows across five sources, one of which
// (node 4) has no out-edges at all.
TEST(CsrBfsExpand, MultiSourceWalkOneToTwo) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(1, 2, CsrPathSemantics::Walk);
    const int64_t src[5] = {0, 1, 2, 3, 4};
    std::vector<Row> got;
    ASSERT_EQ(expand_all(src, 5, g, p, &got), CsrBfsStatus::Ok);
    expect_rows(got, {{0, 1, 1}, {0, 2, 2}, {0, 1, 1}, {0, 2, 2},
                      {1, 2, 1}, {1, 0, 2}, {1, 4, 2},
                      {2, 0, 1}, {2, 1, 2}, {2, 1, 2}, {2, 4, 1},
                      {3, 3, 1}, {3, 3, 2}});
}

// Relationship-type filter: edge 101 (0->1) carries label 9, every other edge
// carries 7. want_label=7 therefore removes one of the two parallel 0->1
// edges and the whole branch under it.
TEST(CsrBfsExpand, LabelFilterPrunesABranch) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph(/*want_label=*/7, /*labeled=*/true);
    const CsrBfsParams p = bfs_params(1, 2, CsrPathSemantics::Walk);
    const int64_t src[1] = {0};
    std::vector<Row> got;
    ASSERT_EQ(expand_all(src, 1, g, p, &got), CsrBfsStatus::Ok);
    expect_rows(got, {{0, 1, 1}, {0, 2, 2}});

    // A label nothing carries yields nothing, cleanly.
    const CsrBfsGraph g42 = f.graph(/*want_label=*/42, /*labeled=*/true);
    std::vector<Row> none;
    ASSERT_EQ(expand_all(src, 1, g42, p, &none), CsrBfsStatus::Ok);
    EXPECT_TRUE(none.empty());
}

// ============================================================================
// Resumability — the same rows, in the same order, however finely sliced
// ============================================================================

TEST(CsrBfsExpand, OutCapResumeIsRowIdentical) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(1, 3, CsrPathSemantics::Walk);
    const int64_t src[5] = {0, 1, 2, 3, 4};
    std::vector<Row> whole;
    ASSERT_EQ(expand_all(src, 5, g, p, &whole), CsrBfsStatus::Ok);
    ASSERT_FALSE(whole.empty());
    for (int64_t chunk : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{7}}) {
        SCOPED_TRACE(chunk);
        expect_rows(expand_chunked(src, 5, g, p, chunk), whole);
    }
}

TEST(CsrBfsExpand, VisitBudgetSuspendsWithoutLosingRows) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[5] = {0, 1, 2, 3, 4};
    std::vector<Row> whole;
    ASSERT_EQ(expand_all(src, 5, g, bfs_params(1, 3, CsrPathSemantics::Walk),
                         &whole), CsrBfsStatus::Ok);
    // One unit of work per call: every call must still make progress and the
    // concatenated result must be byte-identical to the single-call answer.
    const CsrBfsParams tiny = bfs_params(1, 3, CsrPathSemantics::Walk,
                                         /*per_source_cap=*/1 << 20,
                                         /*visit_budget=*/1);
    expect_rows(expand_chunked(src, 5, g, tiny, /*chunk=*/64), whole);
}

TEST(CsrBfsExpand, ZeroOutCapIsANoOpPoll) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(1, 2, CsrPathSemantics::Walk);
    const int64_t src[1] = {0};
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    int64_t rows = -1;
    ASSERT_EQ(csr_bfs_expand(src, 1, &g, &p, &sc, nullptr, nullptr, nullptr, 0,
                             &cur, &rows), CsrBfsStatus::Ok);
    EXPECT_EQ(rows, 0);
    EXPECT_EQ(cur.src_index, 0);
    EXPECT_EQ(cur.active, 0);
    EXPECT_FALSE(csr_bfs_done(&cur, 1));
}

// ============================================================================
// The per-source cap is a LOUD refusal, never a short answer
// ============================================================================

TEST(CsrBfsExpand, PerSourceCapIsAnErrorNotATruncation) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    // Node 0 at [*1..3] WALK produces 8 rows; cap the source at 3.
    const CsrBfsParams p = bfs_params(1, 3, CsrPathSemantics::Walk,
                                      /*per_source_cap=*/3);
    const int64_t src[1] = {0};
    std::vector<Row> got;
    EXPECT_EQ(expand_all(src, 1, g, p, &got),
              CsrBfsStatus::PerSourceCapExceeded);
    // Rows already produced are reported so the caller can account for them,
    // but the status says the result is INCOMPLETE — shipping this prefix as
    // an answer is the failure this return code exists to prevent.
    EXPECT_EQ(got.size(), 3u);

    // One more than the true count is enough: the cap is not off-by-one.
    const CsrBfsParams ok = bfs_params(1, 3, CsrPathSemantics::Walk, 8);
    std::vector<Row> all;
    EXPECT_EQ(expand_all(src, 1, g, ok, &all), CsrBfsStatus::Ok);
    EXPECT_EQ(all.size(), 8u);
}

// ============================================================================
// csr_bfs_count — same traversal, nothing materialised
// ============================================================================

TEST(CsrBfsCount, MatchesExpandRowCountsPerSource) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[5] = {0, 1, 2, 3, 4};
    for (auto sem : {CsrPathSemantics::Walk, CsrPathSemantics::Trail,
                     CsrPathSemantics::Simple}) {
        for (int32_t hi : {1, 2, 3}) {
            SCOPED_TRACE(static_cast<int>(sem) * 10 + hi);
            const CsrBfsParams p = bfs_params(1, hi, sem);
            std::vector<Row> rows;
            ASSERT_EQ(expand_all(src, 5, g, p, &rows), CsrBfsStatus::Ok);
            int64_t want[5] = {0, 0, 0, 0, 0};
            for (const Row& r : rows) ++want[r.src];

            BfsScratchBuf buf;
            CsrBfsScratch sc = buf.view();
            CsrBfsCursor cur{};
            int64_t counts[5] = {-1, -1, -1, -1, -1};
            ASSERT_EQ(csr_bfs_count(src, 5, &g, &p, &sc, counts, &cur),
                      CsrBfsStatus::Ok);
            ASSERT_TRUE(csr_bfs_done(&cur, 5));
            for (int i = 0; i < 5; ++i) EXPECT_EQ(counts[i], want[i]) << i;
        }
    }
}

// Counting is resumable, and a partially-traversed source is NOT written —
// so a caller can never read a small number and mistake it for the answer.
TEST(CsrBfsCount, PartialSourceIsUnwrittenNotPartial) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[5] = {0, 1, 2, 3, 4};
    const CsrBfsParams tiny = bfs_params(1, 3, CsrPathSemantics::Walk,
                                         1 << 20, /*visit_budget=*/1);
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    int64_t counts[5] = {-1, -1, -1, -1, -1};
    // First call cannot possibly finish source 0 with a budget of one unit.
    ASSERT_EQ(csr_bfs_count(src, 5, &g, &tiny, &sc, counts, &cur),
              CsrBfsStatus::Ok);
    EXPECT_FALSE(csr_bfs_done(&cur, 5));
    EXPECT_EQ(counts[0], -1) << "an incomplete source must stay UNWRITTEN";
    EXPECT_EQ(cur.active, 1) << "the cursor must be parked INSIDE source 0";
    EXPECT_GE(cur.emitted_for_src, 0);

    for (int guard = 0; guard < 100000 && !csr_bfs_done(&cur, 5); ++guard)
        ASSERT_EQ(csr_bfs_count(src, 5, &g, &tiny, &sc, counts, &cur),
                  CsrBfsStatus::Ok);
    ASSERT_TRUE(csr_bfs_done(&cur, 5));
    const int64_t want[5] = {8, 5, 6, 3, 0};   // csr_bfs_enumerate.py, [*1..3]
    for (int i = 0; i < 5; ++i) EXPECT_EQ(counts[i], want[i]) << i;
}

TEST(CsrBfsCount, PerSourceCapAlsoRefusesLoudly) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    const int64_t src[1] = {0};
    const CsrBfsParams p = bfs_params(1, 3, CsrPathSemantics::Walk, 3);
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    int64_t counts[1] = {-1};
    EXPECT_EQ(csr_bfs_count(src, 1, &g, &p, &sc, counts, &cur),
              CsrBfsStatus::PerSourceCapExceeded);
    EXPECT_EQ(counts[0], -1);
}

// ============================================================================
// Argument rejection — fail closed, never read out of bounds
// ============================================================================

TEST(CsrBfsExpand, RejectsBadArguments) {
    KnownBfsCsr f;
    const CsrBfsGraph g = f.graph();
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    const int64_t src[1] = {0};
    int64_t os[8], od[8]; int32_t oh[8];
    int64_t rows = -1;

    auto call = [&](const CsrBfsParams& p, const int64_t* ids, int64_t n) {
        CsrBfsCursor cur{};
        return csr_bfs_expand(ids, n, &g, &p, &sc, os, od, oh, 8, &cur, &rows);
    };
    EXPECT_EQ(call(bfs_params(2, 1, CsrPathSemantics::Walk), src, 1),
              CsrBfsStatus::InvalidArgument);              // min > max
    EXPECT_EQ(call(bfs_params(0, 99, CsrPathSemantics::Walk), src, 1),
              CsrBfsStatus::InvalidArgument);              // max > cap
    EXPECT_EQ(call(bfs_params(1, 2, CsrPathSemantics::Walk, 0), src, 1),
              CsrBfsStatus::InvalidArgument);              // zero per-source cap
    EXPECT_EQ(call(bfs_params(1, 2, CsrPathSemantics::Walk, 8, 0), src, 1),
              CsrBfsStatus::InvalidArgument);              // zero budget
    EXPECT_EQ(call(bfs_params(1, 2, CsrPathSemantics::Walk), nullptr, 3),
              CsrBfsStatus::InvalidArgument);              // null ids, n > 0

    // A source id outside the dense CSR range is refused, not read.
    const int64_t bad[1] = {KnownBfsCsr::kNodes};
    EXPECT_EQ(call(bfs_params(1, 2, CsrPathSemantics::Walk), bad, 1),
              CsrBfsStatus::InvalidArgument);
    EXPECT_EQ(rows, 0);

    // n == 0 is a clean no-op, not an error.
    EXPECT_EQ(call(bfs_params(1, 2, CsrPathSemantics::Walk), nullptr, 0),
              CsrBfsStatus::Ok);
}

// A CSR whose NEIGHBOUR ids leave the dense range is refused by name. This
// kernel dereferences off[nb] to descend (csr_expand only copies nb to its
// output), so an unchecked malformed adjacency would be an out-of-bounds READ
// in a -DNDEBUG build, not merely a bad value.
TEST(CsrBfsExpand, MalformedNeighbourIsRefusedNotDereferenced) {
    KnownBfsCsr f;
    f.nbr[0] = KnownBfsCsr::kNodes + 7;    // edge 0 -100-> <out of range>
    const CsrBfsGraph g = f.graph();
    const CsrBfsParams p = bfs_params(1, 2, CsrPathSemantics::Walk);
    const int64_t src[1] = {0};
    std::vector<Row> got;
    EXPECT_EQ(expand_all(src, 1, g, p, &got), CsrBfsStatus::MalformedGraph);

    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    int64_t counts[1] = {-1};
    EXPECT_EQ(csr_bfs_count(src, 1, &g, &p, &sc, counts, &cur),
              CsrBfsStatus::MalformedGraph);
    EXPECT_EQ(counts[0], -1);

    // MEASURED, and it corrects a first guess: even `[*1..1]` refuses, because
    // emitting a 1-hop endpoint still PUSHES the neighbour onto the path (and
    // seeds iter[1] = off[nb]). There is no hop window that reaches a bad
    // neighbour without dereferencing it, so the refusal is not avoidable by
    // asking for less.
    const CsrBfsParams one = bfs_params(1, 1, CsrPathSemantics::Walk);
    std::vector<Row> shallow;
    EXPECT_EQ(expand_all(src, 1, g, one, &shallow),
              CsrBfsStatus::MalformedGraph);

    // The refusal is TARGETED, not blanket: a source whose reachable set never
    // touches the bad edge is still answered normally.
    const int64_t clean[1] = {1};          // 1 -> 2 -> {0, 4}, all in range
    std::vector<Row> ok_rows;
    EXPECT_EQ(expand_all(clean, 1, g, bfs_params(1, 2, CsrPathSemantics::Walk),
                         &ok_rows), CsrBfsStatus::Ok);
    expect_rows(ok_rows, {{1, 2, 1}, {1, 0, 2}, {1, 4, 2}});
}

// ============================================================================
// Randomized graph vs networkx (data/csr_bfs_random.txt)
// ============================================================================

namespace {
RandomGraph& random_graph() {
    static RandomGraph g;
    static bool tried = false;
    if (!tried) { tried = true; load_random_graph(random_graph_path().c_str(), &g); }
    return g;
}
}  // namespace

// Reachability: the DISTINCT endpoint set of a WALK expansion over [0, h] is
// exactly {v : dist(src, v) <= h}. networkx supplies the expected size.
TEST(CsrBfsRandom, WalkReachabilityMatchesNetworkx) {
    RandomGraph& rg = random_graph();
    ASSERT_TRUE(rg.loaded) << "missing " << random_graph_path()
                           << " — regenerate with csr_bfs_enumerate.py "
                              "--emit-random; a missing fixture must FAIL, "
                              "never silently skip";
    ASSERT_FALSE(rg.reach.empty());
    const CsrBfsGraph g = rg.fwd();
    for (const ReachExpect& e : rg.reach) {
        SCOPED_TRACE("src=" + std::to_string(e.src) +
                     " hops=" + std::to_string(e.hops));
        const CsrBfsParams p = bfs_params(0, e.hops, CsrPathSemantics::Walk,
                                          /*per_source_cap=*/1 << 24);
        const int64_t src[1] = {e.src};
        std::vector<Row> rows;
        ASSERT_EQ(expand_all(src, 1, g, p, &rows, /*out_cap=*/1 << 16),
                  CsrBfsStatus::Ok);
        std::set<int64_t> distinct;
        for (const Row& r : rows) distinct.insert(r.dst);
        EXPECT_EQ(static_cast<int64_t>(distinct.size()), e.count);
    }
}

// Path counts under each semantics, against the enumerator's own DFS. This is
// the check that WALK != TRAIL != SIMPLE on a graph large enough to have real
// cycles and repeated relationships.
TEST(CsrBfsRandom, PathCountsMatchTheEnumerator) {
    RandomGraph& rg = random_graph();
    ASSERT_TRUE(rg.loaded) << "missing " << random_graph_path();
    ASSERT_FALSE(rg.paths.empty());
    const CsrBfsGraph g = rg.fwd();
    bool saw_semantics_difference = false;
    for (const PathsExpect& e : rg.paths) {
        SCOPED_TRACE("src=" + std::to_string(e.src) +
                     " hops=" + std::to_string(e.hops));
        const int64_t src[1] = {e.src};
        const int64_t want[3] = {e.walk, e.trail, e.simple};
        int i = 0;
        for (auto sem : {CsrPathSemantics::Walk, CsrPathSemantics::Trail,
                         CsrPathSemantics::Simple}) {
            const CsrBfsParams p = bfs_params(1, e.hops, sem, 1 << 24);
            BfsScratchBuf buf;
            CsrBfsScratch sc = buf.view();
            CsrBfsCursor cur{};
            int64_t got = -1;
            ASSERT_EQ(csr_bfs_count(src, 1, &g, &p, &sc, &got, &cur),
                      CsrBfsStatus::Ok);
            EXPECT_EQ(got, want[i]) << "semantics " << i;
            ++i;
        }
        if (e.walk != e.simple) saw_semantics_difference = true;
    }
    EXPECT_TRUE(saw_semantics_difference)
        << "the random fixture must contain at least one case where WALK and "
           "SIMPLE disagree, or it cannot discriminate the semantics at all";
}

// Perf smoke — PRINTED, NOT GATED (this wave). Reports ns per emitted row.
TEST(CsrBfsRandom, PerfSmokeNotGated) {
    RandomGraph& rg = random_graph();
    ASSERT_TRUE(rg.loaded) << "missing " << random_graph_path();
    const CsrBfsGraph g = rg.fwd();
    std::vector<int64_t> srcs;
    for (int64_t i = 0; i < 200; ++i) srcs.push_back(i);
    const CsrBfsParams p = bfs_params(1, 2, CsrPathSemantics::Walk, 1 << 24);
    BfsScratchBuf buf;
    CsrBfsScratch sc = buf.view();
    CsrBfsCursor cur{};
    std::vector<int64_t> os(1 << 20), od(1 << 20);
    int64_t total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int guard = 0; guard < 100000 &&
         !csr_bfs_done(&cur, static_cast<int64_t>(srcs.size())); ++guard) {
        int64_t rows = 0;
        ASSERT_EQ(csr_bfs_expand(srcs.data(),
                                 static_cast<int64_t>(srcs.size()), &g, &p, &sc,
                                 os.data(), od.data(), nullptr,
                                 static_cast<int64_t>(os.size()), &cur, &rows),
                  CsrBfsStatus::Ok);
        total += rows;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    ASSERT_GT(total, 0);
    std::printf("[perf] csr_bfs_expand WALK [*1..2]: %lld rows, %.2f ns/row\n",
                static_cast<long long>(total), ns / static_cast<double>(total));
}
