// test_bolt_topology.cpp — smoke tests for bolt::bolt_detect_topology.
//
// Asserts only host-agnostic invariants so the suite is green on any CI host.

#include <gtest/gtest.h>

#include "bolt/bolt_topology.h"

using bolt::CpuTopology;
using bolt::bolt_detect_topology;
using bolt::bolt_topology_fallback;
using bolt::kTopologyMaxCpus;

TEST(BoltTopology, DetectReturnsTrue) {
    CpuTopology t{};
    EXPECT_TRUE(bolt_detect_topology(&t));
}

TEST(BoltTopology, LogicalCpusSensible) {
    CpuTopology t{};
    ASSERT_TRUE(bolt_detect_topology(&t));
    EXPECT_GE(t.logical_cpus, 1u);
    EXPECT_LE(t.logical_cpus, kTopologyMaxCpus);
}

TEST(BoltTopology, NumaNodesAtLeastOne) {
    CpuTopology t{};
    ASSERT_TRUE(bolt_detect_topology(&t));
    EXPECT_GE(t.numa_nodes, 1u);
}

TEST(BoltTopology, PhysicalCoresNotMoreThanLogical) {
    CpuTopology t{};
    ASSERT_TRUE(bolt_detect_topology(&t));
    EXPECT_LE(t.physical_cores, t.logical_cpus);
}

TEST(BoltTopology, FallbackProducesSingleNode) {
    CpuTopology t{};
    bolt_topology_fallback(&t);
    EXPECT_EQ(t.numa_nodes, 1u);
    EXPECT_EQ(t.performance_cores, t.logical_cpus);
    EXPECT_EQ(t.efficiency_cores, 0u);
    EXPECT_GE(t.logical_cpus, 1u);
    for (uint32_t i = 0; i < t.logical_cpus; ++i) {
        EXPECT_EQ(t.cpu_to_node[i], 0u);
        EXPECT_EQ(t.cpu_is_perf[i], 1u);
    }
}
