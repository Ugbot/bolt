// test_bolt_tls_scratch.cpp — bolt::tls_scratch (G2CHK-145).

#include <gtest/gtest.h>

#include "bolt/bolt_tls_scratch.h"

#include <cstdint>
#include <thread>

namespace {

struct TagA {};
struct TagB {};

TEST(BoltTlsScratch, SameThreadReusesAndZeroInits) {
    std::uint32_t* a = bolt::tls_scratch<TagA, std::uint32_t, 4096>();
    ASSERT_NE(a, nullptr);
    for (int i = 0; i < 4096; ++i) ASSERT_EQ(a[i], 0u);
    a[7] = 42u;
    std::uint32_t* again = bolt::tls_scratch<TagA, std::uint32_t, 4096>();
    EXPECT_EQ(again, a);
    EXPECT_EQ(a[7], 42u);
}

TEST(BoltTlsScratch, TagsAreDistinctTables) {
    std::uint32_t* a = bolt::tls_scratch<TagA, std::uint32_t, 4096>();
    std::uint32_t* b = bolt::tls_scratch<TagB, std::uint32_t, 4096>();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
}

TEST(BoltTlsScratch, ThreadsGetTheirOwnTable) {
    std::uint32_t* mine = bolt::tls_scratch<TagA, std::uint32_t, 4096>();
    mine[0] = 1u;
    std::uint32_t* theirs = nullptr;
    std::uint32_t theirs0 = 99u;
    std::thread t([&] {
        theirs = bolt::tls_scratch<TagA, std::uint32_t, 4096>();
        theirs0 = theirs[0];
        theirs[0] = 2u;
    });
    t.join();
    ASSERT_NE(theirs, nullptr);
    EXPECT_NE(theirs, mine);
    EXPECT_EQ(theirs0, 0u);
    EXPECT_EQ(mine[0], 1u);
}

}  // namespace
