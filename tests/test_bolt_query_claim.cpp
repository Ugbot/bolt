// test_bolt_query_claim.cpp — operators share a query's memory limit.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "bolt/bolt_query_claim.h"

namespace {

bolt::QueryCtx* acquire_limited(uint64_t limit) {
    bolt::QueryLimits l{};
    l.memory_bytes = limit;
    bolt::QueryCtx* q = bolt::query_ctx_acquire(l, nullptr);
    EXPECT_NE(q, nullptr);
    return q;
}

}  // namespace

TEST(QueryClaim, NoLimitGrantsWantAndRecordsNothing) {
    bolt::QueryClaim c{};
    EXPECT_EQ(bolt::query_claim(&c, 12345u), 12345u);
    EXPECT_EQ(c.q, nullptr);
    bolt::QueryCtx* q = bolt::query_ctx_acquire(bolt::QueryLimits{}, nullptr);
    bolt::QueryCtxScope scope(q);
    bolt::QueryClaim d{};
    EXPECT_EQ(bolt::query_claim(&d, 777u), 777u);
    EXPECT_EQ(d.q, nullptr);
    EXPECT_EQ(q->mem_claimed.load(), 0u);
    bolt::query_ctx_release(q);
}

TEST(QueryClaim, SharesHalveAndNeverExceedTheLimit) {
    constexpr uint64_t kLimit = 1ull << 30;
    bolt::QueryCtx* q = acquire_limited(kLimit);
    bolt::QueryCtxScope scope(q);
    bolt::QueryClaim c[8]{};
    uint64_t sum = 0;
    for (auto& x : c) sum += bolt::query_claim(&x, kLimit);
    EXPECT_EQ(c[0].bytes, kLimit / 2);
    EXPECT_EQ(c[1].bytes, kLimit / 4);
    EXPECT_LT(sum, kLimit);
    EXPECT_EQ(q->mem_claimed.load(), sum);
    // A small want is granted whole.
    bolt::QueryClaim s{};
    EXPECT_EQ(bolt::query_claim(&s, 4096u), 4096u);
    bolt::query_release(&s);
    for (auto& x : c) bolt::query_release(&x);
    EXPECT_EQ(q->mem_claimed.load(), 0u);
    bolt::query_ctx_release(q);
}

TEST(QueryClaim, UsedMemoryShrinksTheShare) {
    constexpr uint64_t kLimit = 1ull << 30;
    bolt::QueryCtx* q = acquire_limited(kLimit);
    bolt::QueryCtxScope scope(q);
    q->charge(kLimit / 2);
    bolt::QueryClaim c{};
    EXPECT_EQ(bolt::query_claim(&c, kLimit), kLimit / 4);
    bolt::query_release(&c);
    q->credit(kLimit / 2);
    bolt::query_ctx_release(q);
}

TEST(QueryClaim, TrimAndReleaseAreIdempotentAndScopedToTheQuery) {
    constexpr uint64_t kLimit = 1ull << 20;
    bolt::QueryCtx* q = acquire_limited(kLimit);
    bolt::QueryClaim c{};
    {
        bolt::QueryCtxScope scope(q);
        EXPECT_EQ(bolt::query_claim(&c, kLimit), kLimit / 2);
    }
    bolt::query_claim_trim(&c, 1000u);
    EXPECT_EQ(q->mem_claimed.load(), 1000u);
    bolt::query_release(&c);
    bolt::query_release(&c);
    EXPECT_EQ(q->mem_claimed.load(), 0u);
    // A claim outliving its query never touches the slot's next query.
    bolt::QueryClaim stale{};
    {
        bolt::QueryCtxScope scope(q);
        bolt::query_claim(&stale, 100u);
    }
    bolt::query_ctx_release(q);
    bolt::QueryCtx* q2 = acquire_limited(kLimit);
    bolt::QueryClaim live{};
    {
        bolt::QueryCtxScope scope(q2);
        bolt::query_claim(&live, 200u);
    }
    bolt::query_release(&stale);
    EXPECT_EQ(q2->mem_claimed.load(), 200u);
    bolt::query_release(&live);
    bolt::query_ctx_release(q2);
}

TEST(QueryClaim, ConcurrentClaimsStayUnderTheLimit) {
    constexpr uint64_t kLimit = 1ull << 32;
    bolt::QueryCtx* q = acquire_limited(kLimit);
    constexpr int kThreads = 8, kEach = 64;
    std::vector<bolt::QueryClaim> claims(kThreads * kEach);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t] {
            bolt::QueryCtxScope scope(q);
            for (int i = 0; i < kEach; ++i) bolt::query_claim(&claims[t * kEach + i], kLimit / 16);
        });
    }
    for (auto& t : ts) t.join();
    uint64_t sum = 0;
    for (auto& c : claims) sum += c.bytes;
    EXPECT_EQ(q->mem_claimed.load(), sum);
    EXPECT_LE(sum, kLimit);
    for (auto& c : claims) bolt::query_release(&c);
    EXPECT_EQ(q->mem_claimed.load(), 0u);
    bolt::query_ctx_release(q);
}
