// Bounds contract for csr_expand_bounded / csr_expand_excluding / csr_expand
// (G2CHK-92).
//
// WHY THIS FILE IS COMPILED WITH -DNDEBUG
// ----------------------------------------
// Before this fix, the ONLY guard on the current source id was
//     assert(s >= 0 && "csr_expand: source id out of CSR range (non-dense?)");
// with NO UPPER BOUND at all — `csr_off[s]` / `csr_off[s + 1]` were read
// unconditionally, and `s` is prior-operator output (chukonu's csr_expand_op.cpp
// feeds it straight from an upstream scan/join/expand column) on the live
// Cypher/SPARQL graph-traversal path. In any -DNDEBUG build (a stock Release)
// that assert compiles to nothing, so an out-of-range `s` was an out-of-bounds
// READ whose garbage begin/end then drove further out-of-bounds reads in the
// inner neighbour-walk loop. This file forces NDEBUG so it exercises exactly
// the build configuration where the old assert vanished — mirroring
// test_bolt_join_bounds.cpp (G2GRAPH-27).
//
// THE FIX
// -------
// csr_expand_bounded (and its two forwarding entry points) now take a
// required `n_nodes` parameter and return kCsrExpandOutOfRange (-1) instead
// of reading csr_off[s] when `s` is outside [0, n_nodes) — a real runtime
// check, not just an assert.

#ifndef NDEBUG
#error "test_bolt_csr_bounds.cpp must be compiled with -DNDEBUG (see header comment)"
#endif

#include "bolt/kernels/bolt_csr.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using bolt::kernels::csr_expand;
using bolt::kernels::csr_expand_bounded;
using bolt::kernels::csr_expand_excluding;
using bolt::kernels::CsrExpandCursor;
using bolt::kernels::kCsrExpandOutOfRange;

// Same 4-node / 4-edge fixture as test_bolt_csr.cpp.
struct KnownCsr {
    static constexpr int64_t n_nodes = 4;
    int64_t off[5] = {0, 2, 2, 3, 4};
    int64_t nbr[4] = {1, 2, 3, 0};
    int64_t eid[4] = {100, 101, 102, 103};
};

// A source id at/past n_nodes must fail closed, not read csr_off[s].
TEST(CsrExpandBounds, SourceAtOrPastNNodesFailsClosed) {
    KnownCsr g;
    for (const int64_t bad_src : {g.n_nodes, g.n_nodes + 1, int64_t{1000000}}) {
        const int64_t src_ids[1] = {bad_src};
        // Canary sentinel: if the kernel wrote anything, these would change.
        int64_t o_src[4]  = {-777, -777, -777, -777};
        int64_t o_edge[4] = {-777, -777, -777, -777};
        int64_t o_dst[4]  = {-777, -777, -777, -777};
        CsrExpandCursor cur{};
        const int64_t w = csr_expand(src_ids, 1, g.n_nodes, g.off, g.nbr,
                                     g.eid, nullptr, -1, o_src, o_edge, o_dst,
                                     4, &cur);
        EXPECT_EQ(w, kCsrExpandOutOfRange) << "bad_src=" << bad_src;
        // Nothing written; cursor left exactly where it was (not advanced
        // past the bad row, so the caller's own diagnostics see the real
        // failing index rather than a silently-skipped one).
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(o_src[i], -777);
            EXPECT_EQ(o_edge[i], -777);
            EXPECT_EQ(o_dst[i], -777);
        }
        EXPECT_EQ(cur.src_index, 0);
        EXPECT_EQ(cur.neighbor_j, 0);
    }
}

// A negative source id must fail closed too (the pre-fix assert covered
// this half; the fix keeps it, and this proves the check is one condition,
// not two separate ones that could drift).
TEST(CsrExpandBounds, NegativeSourceFailsClosed) {
    KnownCsr g;
    const int64_t src_ids[1] = {-1};
    int64_t o_src[2] = {0}, o_edge[2] = {0}, o_dst[2] = {0};
    CsrExpandCursor cur{};
    const int64_t w = csr_expand(src_ids, 1, g.n_nodes, g.off, g.nbr, g.eid,
                                 nullptr, -1, o_src, o_edge, o_dst, 2, &cur);
    EXPECT_EQ(w, kCsrExpandOutOfRange);
}

// The last VALID source id (n_nodes - 1) must still work — the fix must not
// have tightened the valid range by one.
TEST(CsrExpandBounds, LastValidSourceIdStillWorks) {
    KnownCsr g;
    const int64_t src_ids[1] = {g.n_nodes - 1};   // node 3: one edge, eid 103
    int64_t o_src[4] = {0}, o_edge[4] = {0}, o_dst[4] = {0};
    CsrExpandCursor cur{};
    const int64_t w = csr_expand(src_ids, 1, g.n_nodes, g.off, g.nbr, g.eid,
                                 nullptr, -1, o_src, o_edge, o_dst, 4, &cur);
    ASSERT_EQ(w, 1);
    EXPECT_EQ(o_edge[0], 103);
    EXPECT_EQ(o_dst[0], 0);
}

// csr_expand_excluding and csr_expand_bounded forward the same check.
TEST(CsrExpandBounds, ExcludingAndBoundedForwardTheSameCheck) {
    KnownCsr g;
    const int64_t src_ids[1] = {999};
    int64_t o_src[2] = {0}, o_edge[2] = {0}, o_dst[2] = {0};

    CsrExpandCursor c1{};
    EXPECT_EQ(csr_expand_excluding(src_ids, 1, g.n_nodes, g.off, g.nbr, g.eid,
                                   nullptr, -1, nullptr, 0, o_src, o_edge,
                                   o_dst, 2, &c1),
             kCsrExpandOutOfRange);

    CsrExpandCursor c2{};
    EXPECT_EQ(csr_expand_bounded(src_ids, 1, g.n_nodes, g.off, g.nbr, g.eid,
                                 nullptr, -1, nullptr, 0, /*dst_bounds=*/nullptr,
                                 o_src, o_edge, o_dst, 2, &c2),
             kCsrExpandOutOfRange);
}

// Multi-source batch: an EARLIER source in the same call is valid, a LATER
// one is out of range. Documents the actual (and only production-reachable
// via n==1 calls) contract: the whole call reports failure rather than a
// partial success, even though rows for the earlier valid source were
// written into the caller's output buffer before the bad row was reached —
// the caller MUST discard the call's output entirely on a negative return,
// which is exactly what chukonu's csr_expand_op.cpp now does (fails the
// operator rather than trusting a partial batch).
TEST(CsrExpandBounds, MultiSourceBatchFailsClosedOnLaterBadSource) {
    KnownCsr g;
    const int64_t src_ids[2] = {0, 1000000};   // node 0 valid, then garbage
    int64_t o_src[8] = {0}, o_edge[8] = {0}, o_dst[8] = {0};
    CsrExpandCursor cur{};
    const int64_t w = csr_expand(src_ids, 2, g.n_nodes, g.off, g.nbr, g.eid,
                                 nullptr, -1, o_src, o_edge, o_dst, 8, &cur);
    EXPECT_EQ(w, kCsrExpandOutOfRange);
    // cursor parked exactly at the bad source (index 1), never past it.
    EXPECT_EQ(cur.src_index, 1);
}

}  // namespace
