#include <gtest/gtest.h>
#include <atomic>
#include <cstddef>
#include <limits>
#include <utility>
#include "bolt/api/core/stacked_thread.h"

namespace {
using bolt::api::core::kDefaultStackBytes;
using bolt::api::core::StackedThread;

struct Counter {
    alignas(64) std::atomic<unsigned> value{0};
};

void increment(void* user) noexcept {
    static_cast<Counter*>(user)->value.fetch_add(1, std::memory_order_relaxed);
}

void without_context(void* user) noexcept {
    // nullptr is a valid context for callbacks that do not need one.
    EXPECT_EQ(user, nullptr);
}

struct GatedCounter {
    alignas(64) std::atomic<bool> release{false};
    alignas(64) std::atomic<unsigned> value{0};
};

void gated_increment(void* user) noexcept {
    auto* state = static_cast<GatedCounter*>(user);
    state->release.wait(false, std::memory_order_acquire);
    state->value.fetch_add(1, std::memory_order_relaxed);
}

struct SelfJoin {
    StackedThread* owner = nullptr;
    alignas(64) std::atomic<bool> published{false};
    bool attempted = false;
    bool result = true;
};

void attempt_self_join(void* user) noexcept {
    auto* state = static_cast<SelfJoin*>(user);
    state->published.wait(false, std::memory_order_acquire);
    state->result = state->owner->try_join();
    state->attempted = true;
}

TEST(StackedThreadChecked, LaunchJoinAndReuseRunRealCallbacks) {
    StackedThread thread;
    Counter counter;
    EXPECT_FALSE(thread.try_join());
    for (unsigned i = 0; i < 3; ++i) {
        ASSERT_TRUE(thread.try_start(kDefaultStackBytes, &increment, &counter));
        EXPECT_TRUE(thread.joinable());
        ASSERT_TRUE(thread.try_join());
        EXPECT_EQ(counter.value.load(std::memory_order_relaxed), i + 1);
        EXPECT_FALSE(thread.joinable());
        EXPECT_FALSE(thread.try_join());
    }
}

TEST(StackedThreadChecked, InvalidArgumentsLeaveOwnerReusable) {
    StackedThread thread;
    Counter counter;
    EXPECT_FALSE(thread.try_start(kDefaultStackBytes, nullptr, &counter));
    EXPECT_FALSE(thread.try_start(0, &increment, &counter));
    EXPECT_FALSE(thread.try_start(1, &increment, &counter));
    EXPECT_FALSE(thread.try_start(64u * 1024u - 1, &increment, &counter));
    if constexpr ((std::numeric_limits<std::size_t>::max)() >
                  (std::numeric_limits<unsigned>::max)()) {
        const auto over_cap =
            static_cast<std::size_t>((std::numeric_limits<unsigned>::max)()) + 1;
        EXPECT_FALSE(thread.try_start(over_cap, &increment, &counter));
    }
    EXPECT_FALSE(thread.joinable());
    EXPECT_EQ(counter.value.load(std::memory_order_relaxed), 0u);
    ASSERT_TRUE(thread.try_start(kDefaultStackBytes, &increment, &counter));
    ASSERT_TRUE(thread.try_join());
    EXPECT_EQ(counter.value.load(std::memory_order_relaxed), 1u);
}

TEST(StackedThreadChecked, NullContextIsAccepted) {
    StackedThread thread;
    ASSERT_TRUE(thread.try_start(kDefaultStackBytes, &without_context, nullptr));
    ASSERT_TRUE(thread.try_join());
}

TEST(StackedThreadChecked, RestartWhileActivePreservesOriginalThread) {
    StackedThread thread;
    GatedCounter original;
    Counter rejected;
    ASSERT_TRUE(thread.try_start(kDefaultStackBytes, &gated_increment, &original));
    EXPECT_FALSE(thread.try_start(kDefaultStackBytes, &increment, &rejected));
    original.release.store(true, std::memory_order_release);
    original.release.notify_one();
    ASSERT_TRUE(thread.try_join());
    EXPECT_EQ(original.value.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(rejected.value.load(std::memory_order_relaxed), 0u);
}

TEST(StackedThreadChecked, SelfJoinFailsAndExternalJoinReclaimsThread) {
    StackedThread thread;
    SelfJoin state;
    state.owner = &thread;
    ASSERT_TRUE(thread.try_start(kDefaultStackBytes, &attempt_self_join, &state));
    state.published.store(true, std::memory_order_release);
    state.published.notify_one();
    ASSERT_TRUE(thread.try_join());
    EXPECT_TRUE(state.attempted);
    EXPECT_FALSE(state.result);
    EXPECT_FALSE(thread.joinable());
}

TEST(StackedThreadChecked, MovingLiveCheckedThreadPreservesCallbackAndOwnership) {
    StackedThread original;
    GatedCounter counter;
    ASSERT_TRUE(original.try_start(kDefaultStackBytes, &gated_increment, &counter));
    StackedThread moved(std::move(original));
    StackedThread assigned;
    assigned = std::move(moved);
    EXPECT_FALSE(original.try_join());
    EXPECT_FALSE(moved.try_join());
    counter.release.store(true, std::memory_order_release);
    counter.release.notify_one();
    ASSERT_TRUE(assigned.try_join());
    EXPECT_EQ(counter.value.load(std::memory_order_relaxed), 1u);
}
} // namespace
