#include <gtest/gtest.h>
#include "bolt/kernels/fintech/order_sim.h"
#include <cstring>

namespace {
using namespace bolt::fintech;
struct SimTest : testing::Test {
    OrderSimSlot slots[8]{};
    OrderSim book{};
    OrderSimFill fills[8]{};
    std::uint64_t scratch[8]{};
    OrderSimMatchResult result{};
    void SetUp() override { ASSERT_EQ(order_sim_init(&book, slots, 8), OrderSimStatus::Ok); }
    void add(std::uint64_t id, std::int64_t p, std::uint32_t q, BookSide s = BookSide::Ask) {
        ASSERT_EQ(order_sim_add(&book, id, s, p, q), OrderSimStatus::Ok);
    }
    template<OrderSimPolicy P = OrderSimPolicy::Fifo>
    OrderSimStatus match(std::uint32_t q, std::int64_t p = 100,
        std::uint32_t minimum = 1, std::uint32_t cap = 8, BookSide side = BookSide::Bid) {
        return order_sim_match<P>(&book, side, p, q, minimum, fills, cap, scratch, 8, &result);
    }
};
TEST_F(SimTest, FifoPricesAndPartialFill) {
    add(1, 100, 7); add(2, 99, 4); add(3, 100, 8); add(4, 101, 9);
    ASSERT_EQ(match(15), OrderSimStatus::Ok);
    ASSERT_EQ(result.fills, 3u);
    EXPECT_EQ(fills[0].id, 2u); EXPECT_EQ(fills[0].qty, 4u);
    EXPECT_EQ(fills[1].id, 1u); EXPECT_EQ(fills[1].qty, 7u);
    EXPECT_EQ(fills[2].id, 3u); EXPECT_EQ(fills[2].qty, 4u);
    EXPECT_EQ(slots[2].qty, 4u); EXPECT_EQ(slots[3].qty, 9u);
    EXPECT_EQ(result.remaining, 0u);
}
TEST_F(SimTest, SellConsumesHighestBidAndHonorsLimit) {
    add(1, 100, 3, BookSide::Bid); add(2, 102, 2, BookSide::Bid);
    add(3, 101, 4, BookSide::Bid); add(4, 99, 8, BookSide::Ask);
    ASSERT_EQ(match(9, 101, 1, 8, BookSide::Ask), OrderSimStatus::Ok);
    EXPECT_EQ(result.executed, 6u); EXPECT_EQ(result.remaining, 3u);
    EXPECT_EQ(fills[0].id, 2u); EXPECT_EQ(fills[1].id, 3u);
    EXPECT_EQ(slots[0].qty, 3u); EXPECT_EQ(slots[3].qty, 8u);
}
TEST_F(SimTest, AmendmentPriorityRulesAndCancellation) {
    add(1, 100, 10); add(2, 100, 10); add(3, 101, 4);
    const auto first_priority = slots[0].priority;
    ASSERT_EQ(order_sim_amend(&book, 1, 100, 5), OrderSimStatus::Ok);
    EXPECT_EQ(slots[0].priority, first_priority);
    ASSERT_EQ(order_sim_amend(&book, 1, 100, 6), OrderSimStatus::Ok);
    ASSERT_EQ(order_sim_amend(&book, 3, 100, 4), OrderSimStatus::Ok);
    ASSERT_EQ(match(17), OrderSimStatus::Ok);
    EXPECT_EQ(fills[0].id, 2u); EXPECT_EQ(fills[1].id, 1u); EXPECT_EQ(fills[2].id, 3u);
    EXPECT_EQ(slots[2].qty, 3u);
    ASSERT_EQ(order_sim_amend(&book, 3, 100, 0), OrderSimStatus::Ok);
    EXPECT_EQ(order_sim_live_count(book), 0u);
    EXPECT_EQ(order_sim_cancel(&book, 3), OrderSimStatus::NotFound);
}
TEST_F(SimTest, ProRataFloorAndFifoResidual) {
    add(1, 100, 10); add(2, 100, 30); add(3, 100, 60);
    ASSERT_EQ(match<OrderSimPolicy::ProRata>(17), OrderSimStatus::Ok);
    EXPECT_EQ(fills[0].qty, 2u); EXPECT_EQ(fills[1].qty, 5u); EXPECT_EQ(fills[2].qty, 10u);
    EXPECT_EQ(result.executed, 17u);
}
TEST_F(SimTest, ProRataMinimumFiltersBaseOnly) {
    add(1, 100, 10); add(2, 100, 30); add(3, 100, 60);
    ASSERT_EQ(match<OrderSimPolicy::ProRata>(17, 100, 6), OrderSimStatus::Ok);
    ASSERT_EQ(result.fills, 2u);
    EXPECT_EQ(fills[0].id, 1u); EXPECT_EQ(fills[0].qty, 7u);
    EXPECT_EQ(fills[1].id, 3u); EXPECT_EQ(fills[1].qty, 10u);
    EXPECT_EQ(slots[1].qty, 30u);
}
TEST_F(SimTest, ProRataSweepsMultipleLevelsWithoutOverfill) {
    add(1, 99, 3); add(2, 100, 10); add(3, 100, 30); add(4, 101, 9);
    ASSERT_EQ(match<OrderSimPolicy::ProRata>(12), OrderSimStatus::Ok);
    EXPECT_EQ(fills[0].qty, 3u); EXPECT_EQ(fills[1].qty, 3u); EXPECT_EQ(fills[2].qty, 6u);
    EXPECT_EQ(slots[3].qty, 9u); EXPECT_EQ(result.executed, 12u);
}
TEST_F(SimTest, OutputCapacityFailureIsAtomic) {
    add(1, 100, 3); add(2, 100, 4);
    OrderSimSlot before[8]; std::memcpy(before, slots, sizeof(slots));
    fills[0] = {123, 456, 789}; result = {8, 9, 10};
    EXPECT_EQ(match(7, 100, 1, 1), OrderSimStatus::Capacity);
    EXPECT_EQ(std::memcmp(before, slots, sizeof(slots)), 0);
    EXPECT_EQ(fills[0].id, 123u); EXPECT_EQ(fills[0].qty, 789u);
    EXPECT_EQ(result.fills, 8u); EXPECT_EQ(result.remaining, 10u);
    EXPECT_EQ(book.next_priority, 3u);
}
TEST_F(SimTest, AddFailuresAndPriorityOverflowAreAtomic) {
    add(1, 100, 2);
    EXPECT_EQ(order_sim_add(&book, 1, BookSide::Ask, 101, 3), OrderSimStatus::Duplicate);
    book.next_priority = UINT64_MAX;
    EXPECT_EQ(order_sim_amend(&book, 1, 101, 2), OrderSimStatus::Overflow);
    EXPECT_EQ(slots[0].price, 100); EXPECT_EQ(slots[0].priority, 1u);
    EXPECT_EQ(order_sim_add(&book, 2, BookSide::Ask, 101, 3), OrderSimStatus::Overflow);
    EXPECT_EQ(order_sim_amend(&book, 1, 100, 1), OrderSimStatus::Ok);
    EXPECT_EQ(slots[0].priority, 1u);
}
TEST_F(SimTest, FullCapacityCancelAndReuse) {
    for (std::uint64_t i = 1; i <= 8; ++i) add(i, 100, 1);
    EXPECT_EQ(order_sim_add(&book, 9, BookSide::Ask, 100, 1), OrderSimStatus::Capacity);
    ASSERT_EQ(order_sim_cancel(&book, 1), OrderSimStatus::Ok);
    add(9, 100, 1);
    ASSERT_EQ(match(2), OrderSimStatus::Ok);
    EXPECT_EQ(fills[0].id, 2u); EXPECT_EQ(fills[1].id, 3u);
}
TEST_F(SimTest, WideProductAndLevelTotal) {
    add(1, 100, UINT32_MAX); add(2, 100, UINT32_MAX);
    EXPECT_EQ(order_sim_level_quantity(book, BookSide::Ask, 100), std::uint64_t{UINT32_MAX} * 2);
    ASSERT_EQ(match<OrderSimPolicy::ProRata>(UINT32_MAX), OrderSimStatus::Ok);
    EXPECT_EQ(fills[0].qty, 2147483648u); EXPECT_EQ(fills[1].qty, 2147483647u);
    EXPECT_EQ(result.executed, UINT32_MAX);
}
TEST_F(SimTest, InvalidAndUnsupportedPolicyAreExplicit) {
    add(1, 100, 3);
    EXPECT_EQ(match<static_cast<OrderSimPolicy>(99)>(2), OrderSimStatus::Unsupported);
    EXPECT_EQ(match<OrderSimPolicy::ProRata>(2, 100, 0), OrderSimStatus::Invalid);
    EXPECT_EQ(match(0), OrderSimStatus::Invalid);
    EXPECT_EQ(slots[0].qty, 3u);
    EXPECT_EQ(order_sim_add(&book, 0, BookSide::Ask, 100, 1), OrderSimStatus::Invalid);
}
TEST_F(SimTest, AliasAndScratchCapacityRejectedBeforeMutation) {
    add(1, 100, 3);
    EXPECT_EQ(order_sim_match<OrderSimPolicy::Fifo>(&book, BookSide::Bid, 100, 2, 1,
        reinterpret_cast<OrderSimFill*>(slots), 1, scratch, 8, &result), OrderSimStatus::Invalid);
    EXPECT_EQ(order_sim_match<OrderSimPolicy::Fifo>(&book, BookSide::Bid, 100, 2, 1,
        fills, 8, scratch, 7, &result), OrderSimStatus::Capacity);
    EXPECT_EQ(slots[0].qty, 3u);
    EXPECT_EQ(order_sim_init(&book, reinterpret_cast<OrderSimSlot*>(&book), 1), OrderSimStatus::Invalid);
    EXPECT_EQ(book.slots, slots);
}
TEST_F(SimTest, QueriesFollowLiveState) {
    add(1, -4, 5, BookSide::Bid); add(2, -2, 3, BookSide::Bid); add(3, 7, 2);
    EXPECT_EQ(order_sim_best(book, BookSide::Bid)->id, 2u);
    EXPECT_EQ(order_sim_best(book, BookSide::Ask)->id, 3u);
    EXPECT_EQ(order_sim_select_live(book, 1)->id, 2u);
    EXPECT_EQ(order_sim_select_live(book, 3), nullptr);
    ASSERT_EQ(order_sim_cancel(&book, 2), OrderSimStatus::Ok);
    EXPECT_EQ(order_sim_live_count(book), 2u);
    EXPECT_EQ(order_sim_best(book, BookSide::Bid)->id, 1u);
}
} // namespace
