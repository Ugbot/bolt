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
    int64_t off[2] = {0}, scratch[1] = {0};
    EXPECT_FALSE(csr_build(nullptr, nullptr, nullptr, 0, 0, off, nullptr,
                           nullptr, scratch));            // n_nodes <= 0
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
