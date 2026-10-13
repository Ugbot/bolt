#include <gtest/gtest.h>
#include "bolt/kernels/bolt_csr_dijkstra_indexed.h"

#include <array>
#include <cmath>
#include <limits>

namespace bk = bolt::kernels;
namespace {

template <typename Eid, typename Value>
void indexed_cases() {
    // Property order differs from CSR order; includes parallel edges,
    // a cycle, a self-loop, a disconnected node, and a nonzero ID base.
    const int64_t off[5] = {0, 3, 5, 6, 6};
    const Eid nbr[6] = {1, 1, 2, 2, 0, 2};
    Eid ids[6] = {104, 102, 100, 103, 101, 105};
    const Value values[6] = {Value(7), Value(0), Value(1), Value(2), Value(5), Value(0)};
    bk::CsrGraphT<Eid, Eid> graph{};
    graph.off = off; graph.neighbors = nbr; graph.edge_ids = ids;
    graph.n_nodes = 4; graph.want_label = -1;
    bk::CsrIndexedWeights<Value, Eid> weights{values, nullptr, ids, 100, 6};
    double dist[4]{}; int64_t scratch[8]{}, parent[4]{}, slots[4]{}, settled = 0;
    ASSERT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, -1, dist, parent,
               slots, scratch, &settled), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(dist[0], 0); EXPECT_EQ(dist[1], 1); EXPECT_EQ(dist[2], 3);
    EXPECT_TRUE(std::isinf(dist[3])); EXPECT_EQ(settled, 3);
    EXPECT_EQ(parent[2], 1); EXPECT_EQ(slots[2], 3);
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, 1, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(dist[1], 1); EXPECT_EQ(settled, 2);
    uint8_t validity = 0xff;
    weights.validity = &validity;
    validity &= static_cast<uint8_t>(~(1u << 4));
    double gathered[6]{};
    EXPECT_EQ(bk::csr_gather_indexed_weights(weights, 6, gathered, 6), bk::CsrAlgoStatus::Ok);
    EXPECT_TRUE(std::isnan(gathered[0])); EXPECT_EQ(gathered[1], 1);
    EXPECT_EQ(bk::csr_gather_indexed_weights(weights, 6, gathered, 5), bk::CsrAlgoStatus::InvalidArgument);
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, -1, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::NegativeWeight);
    // Existing contract: no weights are visited when source is the target.
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, 0, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(dist[0], 0); EXPECT_EQ(settled, 1);
    weights.validity = nullptr;
    ids[0] = 99;
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, -1, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::MalformedGraph);
    ids[0] = 106;
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, -1, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::MalformedGraph);
    weights.count = -1;
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, 0, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::InvalidArgument);
}

TEST(CsrIndexedWeights, WidthAndTypeSpecializations) {
    indexed_cases<int64_t, int64_t>(); indexed_cases<uint32_t, int64_t>();
    indexed_cases<int64_t, double>(); indexed_cases<uint32_t, double>();
}

TEST(CsrIndexedWeights, FloatingSpecialValuesAndOverflowMatchContiguous) {
    const int64_t off[4] = {0, 1, 2, 2};
    const uint32_t nbr[2] = {1, 2}, ids[2] = {1, 0};
    bk::CsrGraph32 graph{};
    graph.off = off; graph.neighbors = nbr; graph.edge_ids = ids;
    graph.n_nodes = 3; graph.want_label = -1;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double max = std::numeric_limits<double>::max();
    const std::array<std::array<double, 2>, 8> cases{{
        {0.125, 0.375}, {-0.0, 0.25}, {inf, 1}, {1, inf},
        {nan, 1}, {-0.25, 1}, {max, max}, {1, nan}}};
    for (const auto& row : cases) {
        double indexed[2] = {row[1], row[0]};
        const bk::CsrIndexedWeights<double, uint32_t> w{indexed, nullptr, ids, 0, 2};
        double a[3]{}, b[3]{}; int64_t sa[6]{}, sb[6]{}, na = 0, nb = 0;
        const auto actual = bk::csr_dijkstra_indexed(&graph, w, 0, -1, a,
                                                    nullptr, nullptr, sa, &na);
        const auto reference = bk::csr_dijkstra(&graph, row.data(), 0, -1, b,
                                               nullptr, nullptr, sb, &nb);
        EXPECT_EQ(actual, reference); EXPECT_EQ(na, nb);
        for (int i = 0; i < 3; ++i) EXPECT_EQ(a[i], b[i]);
    }
}

TEST(CsrIndexedWeights, LabelFilterDoesNotReadOtherTypeAndByteStorageIsValid) {
    const int64_t off[4] = {0, 2, 3, 3};
    const uint32_t nbr[3] = {1, 2, 2}, ids[3] = {700, 3, 701};
    const int32_t labels[3] = {2, 8, 2};
    bk::CsrGraph32 graph{};
    graph.off = off; graph.neighbors = nbr; graph.edge_ids = ids;
    graph.edge_labels = labels; graph.want_label = 2; graph.n_nodes = 3;
    const double values[2] = {0.125, 0.375};
    int64_t bits[2]{};
    std::memcpy(bits, values, sizeof(bits));
    const bk::CsrIndexedWeights<double, uint32_t> weights{bits, nullptr, ids, 700, 2};
    double dist[3]{}; int64_t scratch[6]{}, settled = 0;
    EXPECT_EQ(bk::csr_dijkstra_indexed(&graph, weights, 0, -1, dist, nullptr,
               nullptr, scratch, &settled), bk::CsrAlgoStatus::Ok);
    EXPECT_EQ(dist[2], 0.5); EXPECT_EQ(settled, 3);
}
} // namespace
