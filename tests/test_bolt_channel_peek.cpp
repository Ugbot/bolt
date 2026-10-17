#include "bolt/bolt_channel.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <utility>

namespace {
struct MoveOnly {
    std::uint64_t value=0;
    unsigned* moves=nullptr;
    MoveOnly() noexcept=default;
    MoveOnly(std::uint64_t v,unsigned* count) noexcept:value(v),moves(count) {}
    MoveOnly(const MoveOnly&)=delete;
    MoveOnly& operator=(const MoveOnly&)=delete;
    MoveOnly(MoveOnly&& other) noexcept { *this=static_cast<MoveOnly&&>(other); }
    MoveOnly& operator=(MoveOnly&& other) noexcept {
        value=other.value;moves=other.moves;
        if(moves)++*moves;
        other.value=0;
        return *this;
    }
};
static_assert(!std::is_copy_constructible_v<MoveOnly>);
static_assert(std::is_same_v<decltype(std::declval<const bolt::SPSCChannel<MoveOnly,4>&>().try_peek()),const MoveOnly*>);
TEST(BoltChannelPeek, EmptyFullRepeatedBorrowAndMoveOnlyPayload) {
    bolt::SPSCChannel<MoveOnly,4> channel;unsigned moves=0;MoveOnly output;
    EXPECT_EQ(channel.try_peek(),nullptr);EXPECT_FALSE(channel.try_pop(&output));
    for(std::uint64_t i=1;i<=4;++i)ASSERT_TRUE(channel.try_push(MoveOnly{i,&moves}));
    const auto* front=channel.try_peek();ASSERT_NE(front,nullptr);EXPECT_EQ(front->value,1u);
    const auto before=moves;
    for(unsigned i=0;i<16;++i){EXPECT_EQ(channel.try_peek(),front);EXPECT_EQ(front->value,1u);}
    MoveOnly refused{99,&moves};EXPECT_FALSE(channel.try_push(static_cast<MoveOnly&&>(refused)));
    EXPECT_EQ(refused.value,99u);EXPECT_EQ(moves,before);EXPECT_EQ(channel.approx_size(),4u);
    ASSERT_TRUE(channel.try_pop(&output));EXPECT_EQ(output.value,1u);
    EXPECT_EQ(moves,before+1); // Peek performed no move; pop does exactly one.
    for(std::uint64_t i=2;i<=4;++i) {
        ASSERT_NE(channel.try_peek(),nullptr);EXPECT_EQ(channel.try_peek()->value,i);
        ASSERT_TRUE(channel.try_pop(&output));EXPECT_EQ(output.value,i);
    }
    EXPECT_EQ(channel.try_peek(),nullptr);
}
TEST(BoltChannelPeek, WrapPreservesBorrowUntilPop) {
    bolt::SPSCChannel<std::uint64_t,4> channel;
    for(std::uint64_t i=1;i<=1024;++i) {
        ASSERT_TRUE(channel.try_push(std::uint64_t{i}));
        const auto* first=channel.try_peek();ASSERT_NE(first,nullptr);EXPECT_EQ(*first,i);
        EXPECT_EQ(channel.try_peek(),first);
        std::uint64_t output=0;ASSERT_TRUE(channel.try_pop(&output));EXPECT_EQ(output,i);
        EXPECT_EQ(channel.try_peek(),nullptr);
    }
}
struct Payload {std::uint64_t sequence=0,inverse=0,pattern=0;};
TEST(BoltChannelPeek, ConcurrentProducerTwentyThousandIntegrityChecks) {
    constexpr std::uint64_t count=20000,attempt_limit=50000000;
    bolt::SPSCChannel<Payload,64> channel;
    std::atomic<bool> stop{false},producer_failed{false};
    std::thread producer([&] {
        std::uint64_t next=1;
        for(std::uint64_t attempt=0;attempt<attempt_limit && next<=count;++attempt) {
            if(stop.load(std::memory_order_relaxed))break;
            Payload value{next,~next,next^0xfedcba9876543210ull};
            if(channel.try_push(static_cast<Payload&&>(value)))++next;
            else bolt::cpu_pause();
        }
        producer_failed.store(next!=count+1,std::memory_order_relaxed);
    });
    bool valid=true;std::uint64_t consumed=0;
    for(std::uint64_t attempt=0;attempt<attempt_limit && consumed<count;++attempt) {
        const auto* view=channel.try_peek();
        if(!view){bolt::cpu_pause();continue;}
        const auto expected=consumed+1;
        if(view->sequence!=expected || view->inverse!=~expected ||
            view->pattern!=(expected^0xfedcba9876543210ull) || channel.try_peek()!=view){valid=false;break;}
        Payload popped{};
        if(!channel.try_pop(&popped) || popped.sequence!=expected || popped.inverse!=~expected ||
            popped.pattern!=(expected^0xfedcba9876543210ull)){valid=false;break;}
        ++consumed;
    }
    stop.store(true,std::memory_order_relaxed);producer.join();
    EXPECT_TRUE(valid);EXPECT_FALSE(producer_failed.load(std::memory_order_relaxed));
    EXPECT_EQ(consumed,count);EXPECT_EQ(channel.try_peek(),nullptr);
}
} // namespace
