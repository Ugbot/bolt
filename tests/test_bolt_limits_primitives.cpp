// test_bolt_limits_primitives.cpp — ArenaVec, ArenaRingBuffer, Budget,
// ResourceExhausted, BOLT_BOUND_CHECK and the Limits registry.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "bolt/bolt_arena_vec.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_limits.h"
#include "bolt/bolt_resource.h"

// Tables for the registry tests. Environment is set before first access.
#define TEST_LIMITS_OK(X)                                                     \
    X(widgets, kFixedAtAlloc, "count", 10, 1, 100, "BOLT_TEST_WIDGETS",      \
      "BOLT_TEST_WIDGETS_LEGACY", "widgets")                                 \
    X(gizmos, kBudget, "bytes", 5, 0, 1000, "BOLT_TEST_GIZMOS", nullptr,     \
      "gizmo budget")                                                        \
    X(format_bytes, kInvariant, "bytes", 64, 64, 64, nullptr, nullptr, "wire")
BOLT_LIMITS_TABLE(test_limits_ok, "test_ok", TEST_LIMITS_OK)

#define TEST_LIMITS_BAD(X)                                                    \
    X(sprockets, kFixedAtAlloc, "count", 4, 1, 8, "BOLT_TEST_SPROCKETS",     \
      nullptr, "sprockets")
BOLT_LIMITS_TABLE(test_limits_bad, "test_bad", TEST_LIMITS_BAD)

namespace {

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    ::_putenv_s(k, v);
#else
    ::setenv(k, v, 1);
#endif
}

bolt::ArenaConfig small_arena() {
    bolt::ArenaConfig c{};
    c.initial_block_size = 64 * 1024;
    c.max_block_size = 1024 * 1024;
    return c;
}

TEST(ArenaVec, GrowsGeometricallyOnlyAtBoundary) {
    bolt::Arena arena{small_arena()};
    bolt::ArenaVec<uint64_t> v;
    ASSERT_TRUE(v.init(&arena, nullptr, 0));
    EXPECT_EQ(v.capacity(), 0u);
    ASSERT_TRUE(v.try_push(1));
    EXPECT_EQ(v.capacity(), 8u);   // 64 bytes of uint64_t
    std::vector<size_t> caps;
    for (uint64_t i = 2; i <= 1000; ++i) {
        const size_t before = v.capacity();
        ASSERT_TRUE(v.try_push(i));
        if (v.capacity() != before) caps.push_back(v.capacity());
    }
    const std::vector<size_t> want = {16, 32, 64, 128, 256, 512, 1024};
    EXPECT_EQ(caps, want);
    for (uint64_t i = 0; i < 1000; ++i) ASSERT_EQ(v[i], i + 1);
    ASSERT_TRUE(v.reserve(5000));
    EXPECT_EQ(v.capacity(), 5000u);   // n wins over 2×cap
    v.destroy();
}

TEST(ArenaVec, BudgetRefusalLeavesVectorIntactAndNamesKnob) {
    bolt::Arena arena{small_arena()};
    bolt::Budget budget("test_vec_budget", 1024);
    bolt::ArenaVec<uint32_t> v;
    ASSERT_TRUE(v.init(&arena, &budget, 16));
    EXPECT_EQ(budget.used(), 64u);
    for (uint32_t i = 0; i < 16; ++i) ASSERT_TRUE(v.try_push(i));
    ASSERT_TRUE(v.try_push(16));      // 32 elems = 128 B, total 192
    ASSERT_TRUE(v.reserve(64));       // 256 B, total 448
    bolt::clear_resource_exhausted();
    EXPECT_FALSE(v.reserve(1000));    // 4000 B > budget
    const bolt::ResourceExhausted re = bolt::last_resource_exhausted();
    ASSERT_NE(re.knob, nullptr);
    EXPECT_STREQ(re.knob, "test_vec_budget");
    EXPECT_EQ(re.requested, 4000u);
    EXPECT_EQ(re.limit, 1024u);
    EXPECT_EQ(re.in_use, 448u);
    EXPECT_EQ(v.size(), 17u);
    EXPECT_EQ(v.capacity(), 64u);
    for (uint32_t i = 0; i < 17; ++i) EXPECT_EQ(v[i], i);
    EXPECT_EQ(budget.used(), 448u);
    v.destroy();
    EXPECT_EQ(budget.used(), 0u);
}

TEST(ArenaVec, ResizeZeroFills) {
    bolt::Arena arena{small_arena()};
    bolt::ArenaVec<int32_t> v;
    ASSERT_TRUE(v.init(&arena, nullptr, 2));
    ASSERT_TRUE(v.try_push(-1));
    ASSERT_TRUE(v.resize(100));
    EXPECT_EQ(v[0], -1);
    for (size_t i = 1; i < 100; ++i) EXPECT_EQ(v[i], 0);
    ASSERT_TRUE(v.resize(3));
    EXPECT_EQ(v.size(), 3u);
    v.destroy();
}

TEST(ArenaRingBuffer, FifoWrapAndFullRefusal) {
    bolt::Arena arena{small_arena()};
    bolt::Budget budget("test_ring_budget", 1 << 20);
    bolt::ArenaRingBuffer<uint64_t> r;
    ASSERT_TRUE(r.init(&arena, &budget, 5));
    EXPECT_EQ(r.capacity(), 8u);
    EXPECT_EQ(budget.used(), 64u);
    uint64_t next_in = 0, next_out = 0, out = 0;
    for (int round = 0; round < 100; ++round) {
        while (r.try_push(next_in)) ++next_in;
        EXPECT_TRUE(r.full());
        EXPECT_EQ(r.size(), 8u);
        for (int k = 0; k < 3; ++k) {
            ASSERT_TRUE(r.try_pop(&out));
            EXPECT_EQ(out, next_out++);
        }
    }
    while (r.try_pop(&out)) EXPECT_EQ(out, next_out++);
    EXPECT_EQ(next_in, next_out);
    r.destroy();
    EXPECT_EQ(budget.used(), 0u);
}

TEST(Budget, HierarchicalRefusalRollsBackEveryLevel) {
    bolt::Budget root("root_knob", 1000);
    bolt::Budget query("query_knob", 600, &root);
    bolt::Budget op("op_knob", 10000, &query);
    ASSERT_TRUE(op.try_reserve(500));
    EXPECT_EQ(op.used(), 500u);
    EXPECT_EQ(query.used(), 500u);
    EXPECT_EQ(root.used(), 500u);
    bolt::clear_resource_exhausted();
    EXPECT_FALSE(op.try_reserve(200));   // query would be 700 > 600
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "query_knob");
    EXPECT_EQ(op.used(), 500u);
    EXPECT_EQ(query.used(), 500u);
    EXPECT_EQ(root.used(), 500u);
    bolt::Budget sibling("sibling_knob", 10000, &root);
    EXPECT_FALSE(sibling.try_reserve(600));   // root would be 1100
    EXPECT_STREQ(bolt::last_resource_exhausted().knob, "root_knob");
    EXPECT_EQ(sibling.used(), 0u);
    op.release(500);
    EXPECT_EQ(root.used(), 0u);
    EXPECT_EQ(root.peak(), 500u);
    EXPECT_FALSE(root.try_reserve(UINT64_MAX));   // overflow-safe
    EXPECT_EQ(root.used(), 0u);
}

TEST(Budget, ContentionNeverExceedsLimit) {
    constexpr uint64_t kLimit = 10'000;
    constexpr uint64_t kChunk = 7;
    bolt::Budget root("contended", kLimit);
    bolt::Budget child("contended_child", UINT64_MAX, &root);
    std::atomic<uint64_t> refused{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&] {
            uint64_t held = 0;
            for (int i = 0; i < 100'000; ++i) {
                if ((i & 3) != 3) {
                    if (child.try_reserve(kChunk)) held += kChunk;
                    else refused.fetch_add(1, std::memory_order_relaxed);
                } else if (held != 0) {
                    child.release(kChunk);
                    held -= kChunk;
                }
            }
            if (held != 0) child.release(held);
        });
    }
    for (auto& t : ts) t.join();
    EXPECT_EQ(root.used(), 0u);
    EXPECT_EQ(child.used(), 0u);
    EXPECT_LE(root.peak(), kLimit);
    EXPECT_GT(refused.load(), 0u);   // the limit was actually reached
}

int bound_checked(uint64_t n, uint64_t cap) {
    BOLT_BOUND_CHECK(n <= cap, -1);
    return 0;
}

int cap_checked(uint64_t n) {
    BOLT_BOUND_CHECK_CAP(n, 32, "test_keys", -2);
    return static_cast<int>(n);
}

TEST(BoundCheck, EnforcedAndRecordsResourceExhausted) {
    EXPECT_EQ(bound_checked(8, 8), 0);
    EXPECT_EQ(bound_checked(9, 8), -1);
    EXPECT_EQ(cap_checked(32), 32);
    bolt::clear_resource_exhausted();
    EXPECT_EQ(cap_checked(33), -2);
    const bolt::ResourceExhausted re = bolt::last_resource_exhausted();
    EXPECT_STREQ(re.knob, "test_keys");
    EXPECT_EQ(re.requested, 33u);
    EXPECT_EQ(re.limit, 32u);
}

TEST(LimitsRegistry, ResolvesEnvAliasAndReportsSource) {
    set_env("BOLT_TEST_WIDGETS", "");   // empty = unset: the alias wins
    set_env("BOLT_TEST_WIDGETS_LEGACY", "42");
    set_env("BOLT_TEST_GIZMOS", "999");
    bolt::LimitTable& t = test_limits_ok();
    EXPECT_EQ(t.error[0], '\0');
    EXPECT_EQ(test_limits_ok_value(test_limits_ok_id::widgets), 42u);
    EXPECT_EQ(bolt::limits_source(t, 0), bolt::LimitSource::kEnv);
    EXPECT_EQ(test_limits_ok_value(test_limits_ok_id::gizmos), 999u);
    EXPECT_EQ(test_limits_ok_value(test_limits_ok_id::format_bytes), 64u);
    EXPECT_EQ(bolt::limits_source(t, 2), bolt::LimitSource::kDefault);

    char err[256] = {};
    EXPECT_FALSE(bolt::limits_set(&t, 2, 128, err, sizeof(err)));
    EXPECT_NE(std::string(err).find("invariant"), std::string::npos);
    EXPECT_FALSE(bolt::limits_set(&t, 0, 101, err, sizeof(err)));
    EXPECT_NE(std::string(err).find("outside"), std::string::npos);
    EXPECT_TRUE(bolt::limits_set(&t, 0, 100, err, sizeof(err)));
    EXPECT_EQ(bolt::limits_source(t, 0), bolt::LimitSource::kSet);

    const bolt::LimitTable* ft = nullptr;
    uint32_t fi = 0;
    ASSERT_TRUE(bolt::limits_find("gizmos", &ft, &fi));
    EXPECT_EQ(ft, &t);
    EXPECT_EQ(fi, 1u);
    EXPECT_STREQ(bolt::limits_env_for("gizmos"), "BOLT_TEST_GIZMOS");
}

TEST(LimitsRegistry, OutOfRangeEnvIsAStartupErrorAndKeepsDefault) {
    set_env("BOLT_TEST_SPROCKETS", "9");
    bolt::LimitTable& t = test_limits_bad();
    EXPECT_EQ(test_limits_bad_value(test_limits_bad_id::sprockets), 4u);
    EXPECT_EQ(bolt::limits_source(t, 0), bolt::LimitSource::kDefault);
    ASSERT_NE(t.error[0], '\0');
    EXPECT_NE(std::string(t.error).find("BOLT_TEST_SPROCKETS=9"), std::string::npos);
    const char* first = bolt::limits_first_error();
    ASSERT_NE(first, nullptr);
    EXPECT_NE(std::string(first).find("sprockets"), std::string::npos);
}

TEST(LimitsRegistry, ParseRejectsJunkAndOverflow) {
    uint64_t v = 0;
    EXPECT_TRUE(bolt::limits_parse_u64("18446744073709551615", &v));
    EXPECT_EQ(v, UINT64_MAX);
    EXPECT_FALSE(bolt::limits_parse_u64("18446744073709551616", &v));
    EXPECT_FALSE(bolt::limits_parse_u64("12k", &v));
    EXPECT_FALSE(bolt::limits_parse_u64("-1", &v));
    EXPECT_FALSE(bolt::limits_parse_u64("", &v));
}

TEST(LimitsRegistry, BoltTableRegistersAndProcessBudgetUsesIt) {
    const bolt::LimitTable& t = bolt::bolt_limits();
    EXPECT_STREQ(t.owner, "bolt");
    EXPECT_EQ(bolt::limits_value(t, static_cast<uint32_t>(bolt::bolt_limits_id::max_workers)),
              static_cast<uint64_t>(BOLT_MAX_WORKERS));
    EXPECT_EQ(bolt::process_budget().limit(), bolt::configured_mem_budget_bytes());
    EXPECT_STREQ(bolt::process_budget().knob(), "mem_budget_mb");
    bolt::ResourceExhausted re{};
    re.knob = "mem_budget_mb";
    re.requested = 10;
    re.limit = 5;
    char buf[256];
    bolt::format_resource_exhausted(re, buf, sizeof(buf));
    EXPECT_NE(std::string(buf).find("raise it with BOLT_MEM_BUDGET_MB"), std::string::npos);
    char tiny[8];
    EXPECT_EQ(bolt::format_resource_exhausted(re, tiny, sizeof(tiny)), 7u);
    EXPECT_EQ(tiny[7], '\0');
}

}  // namespace
