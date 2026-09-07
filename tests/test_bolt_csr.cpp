// test_bolt_csr.cpp — GTest suite for the scan/scatter + CSR graph kernels.
//
// Covers:
//   - bolt::kernels::exclusive_scan_i64 (empty, single, monotone, total)
//   - bolt::kernels::histogram_i64      (bucket counts, empty buckets)
//   - bolt::kernels::scatter_i64        (permutation round-trip vs gather)
//   - bolt::kernels::csr_build          (known 4-node/4-edge graph, exact
//                                        offsets / neighbors / edge_ids)
//   - bolt::kernels::csr_expand         (wildcard vs label-filtered, multi-
//                                        src, out_cap-exceeded resume,
//                                        empty adjacency)
//
// Deterministic data only. Zero dependency beyond GTest — matches the rest
// of the tests/ shape.

#include <gtest/gtest.h>

#include "bolt/kernels/bolt_scan.h"
#include "bolt/kernels/bolt_csr.h"
#include "bolt/bolt_branchless.h"   // gather_branchless for the scatter round-trip

#include <cstdint>
#include <vector>

using bolt::kernels::exclusive_scan_i64;
using bolt::kernels::histogram_i64;
using bolt::kernels::scatter_i64;
using bolt::kernels::csr_build;
using bolt::kernels::csr_expand;
using bolt::kernels::CsrExpandCursor;

// ============================================================================
// exclusive_scan_i64
// ============================================================================

TEST(ExclusiveScan, Empty) {
    int64_t out[1] = {-1};
    exclusive_scan_i64(nullptr, 0, out);
    EXPECT_EQ(out[0], 0);   // out[0] is always 0; out[n]==out[0]==0
}

TEST(ExclusiveScan, Single) {
    const int64_t in[1] = {7};
    int64_t out[2] = {-1, -1};
    exclusive_scan_i64(in, 1, out);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 7);   // grand total
}

TEST(ExclusiveScan, MonotoneWithTotal) {
    const int64_t in[5] = {3, 0, 2, 5, 1};
    int64_t out[6] = {0};
    exclusive_scan_i64(in, 5, out);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 3);
    EXPECT_EQ(out[2], 3);   // +0
    EXPECT_EQ(out[3], 5);
    EXPECT_EQ(out[4], 10);
    EXPECT_EQ(out[5], 11);  // grand total = 3+0+2+5+1
    // Strictly non-decreasing (non-negative inputs).
    for (int i = 0; i < 5; ++i) EXPECT_LE(out[i], out[i + 1]);
}

// ============================================================================
// histogram_i64
// ============================================================================

TEST(Histogram, BucketCounts) {
    const int64_t keys[8] = {0, 2, 1, 2, 0, 2, 3, 1};
    int64_t counts[4] = {0, 0, 0, 0};
    histogram_i64(keys, 8, 4, counts);
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 3);
    EXPECT_EQ(counts[3], 1);
    int64_t total = counts[0] + counts[1] + counts[2] + counts[3];
    EXPECT_EQ(total, 8);
}

TEST(Histogram, EmptyBuckets) {
    const int64_t keys[3] = {0, 0, 4};   // buckets 1,2,3 stay empty
    int64_t counts[5] = {0, 0, 0, 0, 0};
    histogram_i64(keys, 3, 5, counts);
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 0);
    EXPECT_EQ(counts[2], 0);
    EXPECT_EQ(counts[3], 0);
    EXPECT_EQ(counts[4], 1);
}

TEST(Histogram, EmptyInput) {
    int64_t counts[3] = {0, 0, 0};
    histogram_i64(nullptr, 0, 3, counts);
    EXPECT_EQ(counts[0], 0);
    EXPECT_EQ(counts[1], 0);
    EXPECT_EQ(counts[2], 0);
}

// ============================================================================
// scatter_i64 — inverse of gather_branchless
// ============================================================================

TEST(Scatter, PermutationRoundTripWithGather) {
    // src[i] lands at out[perm[i]]; gathering out back through perm
    // reproduces src exactly (perm is a bijection).
    const int64_t src[5]   = {10, 20, 30, 40, 50};
    const int32_t perm[5]  = {4, 0, 3, 1, 2};
    int64_t out[5] = {0};
    scatter_i64(src, perm, 5, out);
    // out[4]=10, out[0]=20, out[3]=30, out[1]=40, out[2]=50
    EXPECT_EQ(out[0], 20);
    EXPECT_EQ(out[1], 40);
    EXPECT_EQ(out[2], 50);
    EXPECT_EQ(out[3], 30);
    EXPECT_EQ(out[4], 10);

    int64_t back[5] = {0};
    bolt::branchless::gather_branchless<int64_t>(out, perm, 5, back);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(back[i], src[i]);
}

TEST(Scatter, Empty) {
    int64_t out[1] = {99};
    scatter_i64(nullptr, nullptr, 0, out);
    EXPECT_EQ(out[0], 99);   // untouched
}

// ============================================================================
// csr_build — known 4-node / 4-edge graph
// ============================================================================
//
// Graph (grouped by source, insertion order preserved within a block):
//   edge 0: 0 -> 1   (eid 100)
//   edge 1: 0 -> 2   (eid 101)
//   edge 2: 2 -> 3   (eid 102)
//   edge 3: 3 -> 0   (eid 103)
// Node 1 has out-degree 0 (empty block). Expected:
//   offsets      = [0, 2, 2, 3, 4]
//   neighbors    = [1, 2, 3, 0]
//   edge_ids     = [100, 101, 102, 103]

TEST(CsrBuild, KnownGraph) {
    const int64_t src[4]     = {0, 0, 2, 3};
    const int64_t dst[4]     = {1, 2, 3, 0};
    const int64_t eid[4]     = {100, 101, 102, 103};
    const int64_t n_edges = 4, n_nodes = 4;

    int64_t offsets[5]   = {0};
    int64_t neighbors[4] = {0};
    int64_t edge_ids[4]  = {0};
    int64_t scratch[4]   = {0};

    bool ok = csr_build(src, dst, eid, n_edges, n_nodes,
                        offsets, neighbors, edge_ids, scratch);
    ASSERT_TRUE(ok);

    const int64_t exp_off[5] = {0, 2, 2, 3, 4};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(offsets[i], exp_off[i]) << "offset " << i;
    const int64_t exp_nbr[4] = {1, 2, 3, 0};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(neighbors[i], exp_nbr[i]) << "nbr " << i;
    const int64_t exp_eid[4] = {100, 101, 102, 103};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(edge_ids[i], exp_eid[i]) << "eid " << i;
}

TEST(CsrBuild, NullEdgeIdDefaultsToIndex) {
    const int64_t src[3] = {1, 0, 1};
    const int64_t dst[3] = {2, 1, 0};
    int64_t offsets[4] = {0}, neighbors[3] = {0}, edge_ids[3] = {0}, scratch[3] = {0};
    ASSERT_TRUE(csr_build(src, dst, /*edge_id=*/nullptr, 3, 3,
                          offsets, neighbors, edge_ids, scratch));
    // offsets: deg(0)=1, deg(1)=2, deg(2)=0 => [0,1,3,3]
    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 1);
    EXPECT_EQ(offsets[2], 3);
    EXPECT_EQ(offsets[3], 3);
    // node 0 block: edge index 1 (0->1). node 1 block: edge indices 0,2.
    EXPECT_EQ(neighbors[0], 1);   // dst of edge 1
    EXPECT_EQ(edge_ids[0], 1);    // edge_id defaults to input index
    EXPECT_EQ(neighbors[1], 2);   // dst of edge 0
    EXPECT_EQ(edge_ids[1], 0);
    EXPECT_EQ(neighbors[2], 0);   // dst of edge 2
    EXPECT_EQ(edge_ids[2], 2);
}

TEST(CsrBuild, ZeroEdges) {
    int64_t offsets[3] = {-1, -1, -1};
    int64_t scratch[2] = {0};
    ASSERT_TRUE(csr_build(nullptr, nullptr, nullptr, 0, 2,
                          offsets, nullptr, nullptr, scratch));
    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 0);
    EXPECT_EQ(offsets[2], 0);
}

TEST(CsrBuild, RejectsBadArgs) {
    // REPAIRED W16-L1, and stated rather than quietly dropped. This file had
    // never been registered in the house suite (see tests/CMakeLists.txt), and
    // registering it showed that this case ABORTS in an asserts-live build:
    // `n_nodes <= 0` is a PROGRAMMER ERROR, so csr_build asserts it (Tiger
    // Style) and the `return false` beneath is only the release-build net. A
    // case that can only ever abort is not a check, so it is compiled where it
    // is executable and named where it is not. The coverage is not weakened —
    // the null-scratch arm below runs in every configuration and is the one
    // that exercises the `return false` path with the assert satisfied.
#ifdef NDEBUG
    int64_t off[2] = {0}, scratch[1] = {0};
    EXPECT_FALSE(csr_build(nullptr, nullptr, nullptr, 0, 0, off, nullptr,
                           nullptr, scratch));            // n_nodes <= 0
#endif
    int64_t off2[2] = {0};
    EXPECT_FALSE(csr_build(nullptr, nullptr, nullptr, 0, 1, off2, nullptr,
                           nullptr, nullptr));            // null scratch
}

// ============================================================================
// csr_expand
// ============================================================================
//
// Reuse the known 4-node graph. CSR:
//   offsets   = [0, 2, 2, 3, 4]
//   neighbors = [1, 2, 3, 0]
//   edge_ids  = [100, 101, 102, 103]
//   labels    = [ 7,  9,  7,  7]   (edge 1 has label 9, rest label 7)

namespace {
struct KnownCsr {
    int64_t off[5]  = {0, 2, 2, 3, 4};
    int64_t nbr[4]  = {1, 2, 3, 0};
    int64_t eid[4]  = {100, 101, 102, 103};
    int32_t lbl[4]  = {7, 9, 7, 7};
};
}  // namespace

TEST(CsrExpand, WildcardMultiSrc) {
    KnownCsr g;
    const int64_t src_ids[3] = {0, 1, 3};   // node 1 is empty
    int64_t o_src[16] = {0}, o_edge[16] = {0}, o_dst[16] = {0};
    CsrExpandCursor cur{};
    int64_t w = csr_expand(src_ids, 3, g.off, g.nbr, g.eid,
                           /*edge_labels=*/nullptr, /*want=*/-1,
                           o_src, o_edge, o_dst, 16, &cur);
    // node 0 -> (1 via 100), (2 via 101); node 1 -> none; node 3 -> (0 via 103)
    ASSERT_EQ(w, 3);
    EXPECT_EQ(cur.src_index, 3);   // fully drained
    EXPECT_EQ(o_src[0], 0);  EXPECT_EQ(o_edge[0], 100); EXPECT_EQ(o_dst[0], 1);
    EXPECT_EQ(o_src[1], 0);  EXPECT_EQ(o_edge[1], 101); EXPECT_EQ(o_dst[1], 2);
    EXPECT_EQ(o_src[2], 3);  EXPECT_EQ(o_edge[2], 103); EXPECT_EQ(o_dst[2], 0);
}

TEST(CsrExpand, LabelFiltered) {
    KnownCsr g;
    const int64_t src_ids[1] = {0};   // edges: (1,eid100,lbl7) (2,eid101,lbl9)
    int64_t o_src[8] = {0}, o_edge[8] = {0}, o_dst[8] = {0};
    CsrExpandCursor cur{};
    // want label 7 => only edge 100 (to node 1) survives.
    int64_t w = csr_expand(src_ids, 1, g.off, g.nbr, g.eid, g.lbl, /*want=*/7,
                           o_src, o_edge, o_dst, 8, &cur);
    ASSERT_EQ(w, 1);
    EXPECT_EQ(o_edge[0], 100);
    EXPECT_EQ(o_dst[0], 1);

    // want label 9 => only edge 101 (to node 2) survives.
    CsrExpandCursor cur2{};
    int64_t w2 = csr_expand(src_ids, 1, g.off, g.nbr, g.eid, g.lbl, /*want=*/9,
                            o_src, o_edge, o_dst, 8, &cur2);
    ASSERT_EQ(w2, 1);
    EXPECT_EQ(o_edge[0], 101);
    EXPECT_EQ(o_dst[0], 2);

    // want a label that matches nothing => 0 rows.
    CsrExpandCursor cur3{};
    int64_t w3 = csr_expand(src_ids, 1, g.off, g.nbr, g.eid, g.lbl, /*want=*/42,
                            o_src, o_edge, o_dst, 8, &cur3);
    EXPECT_EQ(w3, 0);
}

TEST(CsrExpand, OutCapResume) {
    KnownCsr g;
    const int64_t src_ids[3] = {0, 1, 3};   // wildcard emits 3 rows total
    int64_t o_src[8] = {0}, o_edge[8] = {0}, o_dst[8] = {0};
    CsrExpandCursor cur{};

    // First call: cap 1 => exactly 1 row, cursor parked mid-block of node 0.
    int64_t w1 = csr_expand(src_ids, 3, g.off, g.nbr, g.eid, nullptr, -1,
                            o_src, o_edge, o_dst, /*out_cap=*/1, &cur);
    ASSERT_EQ(w1, 1);
    EXPECT_EQ(o_edge[0], 100);
    EXPECT_LT(cur.src_index, 3);   // not finished

    // Second call: cap 1 => the second edge of node 0.
    int64_t w2 = csr_expand(src_ids, 3, g.off, g.nbr, g.eid, nullptr, -1,
                            o_src, o_edge, o_dst, 1, &cur);
    ASSERT_EQ(w2, 1);
    EXPECT_EQ(o_edge[0], 101);

    // Third call: drain the rest (node 1 empty, node 3 one edge).
    int64_t w3 = csr_expand(src_ids, 3, g.off, g.nbr, g.eid, nullptr, -1,
                            o_src, o_edge, o_dst, 8, &cur);
    ASSERT_EQ(w3, 1);
    EXPECT_EQ(o_edge[0], 103);
    EXPECT_EQ(cur.src_index, 3);   // fully drained now

    // Fourth call: nothing left.
    int64_t w4 = csr_expand(src_ids, 3, g.off, g.nbr, g.eid, nullptr, -1,
                            o_src, o_edge, o_dst, 8, &cur);
    EXPECT_EQ(w4, 0);
}

TEST(CsrExpand, EmptyAdjacencyAndZeroCap) {
    KnownCsr g;
    // Source node 1 has an empty block: expand yields 0 rows but drains.
    const int64_t src_ids[1] = {1};
    int64_t o_src[4] = {0}, o_edge[4] = {0}, o_dst[4] = {0};
    CsrExpandCursor cur{};
    int64_t w = csr_expand(src_ids, 1, g.off, g.nbr, g.eid, nullptr, -1,
                           o_src, o_edge, o_dst, 4, &cur);
    EXPECT_EQ(w, 0);
    EXPECT_EQ(cur.src_index, 1);   // drained

    // out_cap == 0 is a valid no-op poll.
    CsrExpandCursor cur2{};
    const int64_t src2[1] = {0};
    int64_t w0 = csr_expand(src2, 1, g.off, g.nbr, g.eid, nullptr, -1,
                            o_src, o_edge, o_dst, /*out_cap=*/0, &cur2);
    EXPECT_EQ(w0, 0);
    EXPECT_EQ(cur2.src_index, 0);  // made no progress, resumes later
}

// n == 0 (no source ids) is a clean no-op.
TEST(CsrExpand, NoSources) {
    KnownCsr g;
    int64_t o_src[2] = {0}, o_edge[2] = {0}, o_dst[2] = {0};
    CsrExpandCursor cur{};
    int64_t w = csr_expand(nullptr, 0, g.off, g.nbr, g.eid, nullptr, -1,
                           o_src, o_edge, o_dst, 2, &cur);
    EXPECT_EQ(w, 0);
    EXPECT_EQ(cur.src_index, 0);
}

// ============================================================================
// csr_expand_excluding — relationship ISOMORPHISM at expansion time (W16-L1)
// ============================================================================
//
// The exclusion list carries relationship ids EARLIER HOPS of the same pattern
// already traversed. An excluded edge must be dropped by exactly the same
// keep-advance the label mask uses: it must not appear in the output, must not
// consume an output slot, and must not disturb the resume cursor.
//
// The 4-node fixture's node 0 has TWO out-edges (100 -> 1, 101 -> 2), which is
// what makes "the right one was dropped" distinguishable from "a row was
// dropped": excluding 100 must leave 101 and excluding 101 must leave 100. A
// test that only counted rows could not tell those apart.

TEST(CsrExpandExcluding, DropsExactlyTheNamedEdge) {
    KnownCsr g;
    const int64_t src_ids[1] = {0};
    int64_t o_src[8] = {0}, o_edge[8] = {0}, o_dst[8] = {0};

    // Exclude 100 => only 101 (to node 2) survives.
    const int64_t ex_a[1] = {100};
    CsrExpandCursor c1{};
    int64_t w1 = bolt::kernels::csr_expand_excluding(
        src_ids, 1, g.off, g.nbr, g.eid, nullptr, -1, ex_a, 1,
        o_src, o_edge, o_dst, 8, &c1);
    ASSERT_EQ(w1, 1);
    EXPECT_EQ(o_edge[0], 101);
    EXPECT_EQ(o_dst[0], 2);
    EXPECT_EQ(c1.src_index, 1);

    // Exclude 101 => only 100 (to node 1) survives. The mirror image, so a
    // kernel that dropped "the first edge" rather than "the named edge" fails.
    const int64_t ex_b[1] = {101};
    CsrExpandCursor c2{};
    int64_t w2 = bolt::kernels::csr_expand_excluding(
        src_ids, 1, g.off, g.nbr, g.eid, nullptr, -1, ex_b, 1,
        o_src, o_edge, o_dst, 8, &c2);
    ASSERT_EQ(w2, 1);
    EXPECT_EQ(o_edge[0], 100);
    EXPECT_EQ(o_dst[0], 1);

    // Both excluded => 0 rows, still fully drained (not a stall).
    const int64_t ex_both[2] = {101, 100};
    CsrExpandCursor c3{};
    int64_t w3 = bolt::kernels::csr_expand_excluding(
        src_ids, 1, g.off, g.nbr, g.eid, nullptr, -1, ex_both, 2,
        o_src, o_edge, o_dst, 8, &c3);
    EXPECT_EQ(w3, 0);
    EXPECT_EQ(c3.src_index, 1);

    // An id that names no edge of this block excludes nothing.
    const int64_t ex_none[1] = {999};
    CsrExpandCursor c4{};
    int64_t w4 = bolt::kernels::csr_expand_excluding(
        src_ids, 1, g.off, g.nbr, g.eid, nullptr, -1, ex_none, 1,
        o_src, o_edge, o_dst, 8, &c4);
    EXPECT_EQ(w4, 2);
}

// n_excl == 0 must be BYTE-IDENTICAL to csr_expand. This is the contract that
// lets csr_expand forward here instead of keeping a second copy of the walk,
// so it is asserted rather than assumed.
TEST(CsrExpandExcluding, ZeroExclusionsEqualsPlainExpand) {
    KnownCsr g;
    const int64_t src_ids[3] = {0, 1, 3};
    int64_t a_src[16] = {0}, a_edge[16] = {0}, a_dst[16] = {0};
    int64_t b_src[16] = {0}, b_edge[16] = {0}, b_dst[16] = {0};
    CsrExpandCursor ca{}, cb{};
    const int64_t wa = csr_expand(src_ids, 3, g.off, g.nbr, g.eid, g.lbl, 7,
                                  a_src, a_edge, a_dst, 16, &ca);
    const int64_t wb = bolt::kernels::csr_expand_excluding(
        src_ids, 3, g.off, g.nbr, g.eid, g.lbl, 7, nullptr, 0,
        b_src, b_edge, b_dst, 16, &cb);
    ASSERT_EQ(wa, wb);
    EXPECT_EQ(ca.src_index, cb.src_index);
    EXPECT_EQ(ca.neighbor_j, cb.neighbor_j);
    for (int64_t i = 0; i < wa; ++i) {
        EXPECT_EQ(a_src[i], b_src[i]);
        EXPECT_EQ(a_edge[i], b_edge[i]);
        EXPECT_EQ(a_dst[i], b_dst[i]);
    }
}

// The exclusion must compose with the LABEL mask and survive an out_cap
// resume. Node 0's block is walked one row per call with edge 100 excluded, so
// the resume point has to be parked past a DROPPED edge — the case where a
// keep-advance kernel most easily loses or repeats a row.
TEST(CsrExpandExcluding, ComposesWithLabelAndResumes) {
    KnownCsr g;
    const int64_t src_ids[2] = {0, 3};
    const int64_t ex[1] = {100};
    int64_t o_src[8] = {0}, o_edge[8] = {0}, o_dst[8] = {0};
    CsrExpandCursor cur{};
    int64_t total = 0;
    // out_cap == 1 forces a resume after every emitted row; bounded by the
    // number of CSR edges plus one poll per source.
    for (int guard = 0; guard < 16 && cur.src_index < 2; ++guard) {
        const int64_t w = bolt::kernels::csr_expand_excluding(
            src_ids, 2, g.off, g.nbr, g.eid, nullptr, -1, ex, 1,
            o_src + total, o_edge + total, o_dst + total, 1, &cur);
        total += w;
    }
    // node 0: 100 excluded, 101 kept; node 3: 103 kept.
    ASSERT_EQ(total, 2);
    EXPECT_EQ(o_edge[0], 101);
    EXPECT_EQ(o_edge[1], 103);
    EXPECT_EQ(cur.src_index, 2);

    // Same, but the surviving edge is also label-rejected => nothing at all.
    CsrExpandCursor cur2{};
    int64_t w2 = bolt::kernels::csr_expand_excluding(
        src_ids, 1, g.off, g.nbr, g.eid, g.lbl, /*want=*/7, ex, 1,
        o_src, o_edge, o_dst, 8, &cur2);
    EXPECT_EQ(w2, 0);          // edge 100 is label 7 but excluded; 101 is label 9
    EXPECT_EQ(cur2.src_index, 1);
}
