#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../src/api/net/event_handler_registry.h"

using bolt::api::net::EventHandlerData;
using bolt::api::net::EventHandlerRegistry;
using bolt::api::net::HandlerDispatchResult;
using bolt::api::net::IOEvent;

namespace {

struct DestructionProbe {
    EventHandlerRegistry* registry = nullptr;
    bool* in_callback = nullptr;
    bool* destroyed_during_callback = nullptr;
    int fd = -1;

    void operator()(int, IOEvent, void*) const {
        *in_callback = true;
        EXPECT_TRUE(registry->remove(fd));
        *in_callback = false;
    }

    ~DestructionProbe() {
        if (in_callback != nullptr && *in_callback) {
            *destroyed_during_callback = true;
        }
    }
};

bool add_handler(
    EventHandlerRegistry* registry,
    int fd,
    bolt::api::net::EventHandler handler) {
    EventHandlerData* data = registry->insert(fd);
    if (data == nullptr) return false;
    data->handler = std::move(handler);
    data->user_data = nullptr;
    data->events = IOEvent::READ;
    return true;
}

}  // namespace

TEST(EventHandlerRegistry, SelfRemovalPinsCallableUntilReturn) {
    EventHandlerRegistry registry;
    constexpr int kFd = 17;
    bool in_callback = false;
    bool destroyed_during_callback = false;

    ASSERT_TRUE(add_handler(
        &registry, kFd,
        DestructionProbe{
            &registry, &in_callback, &destroyed_during_callback, kFd}));
    EventHandlerData* data = registry.find(kFd);
    ASSERT_NE(data, nullptr);
    const uint64_t token = data->token;

    EXPECT_EQ(registry.dispatch(kFd, token, IOEvent::READ),
              HandlerDispatchResult::dispatched);
    EXPECT_FALSE(in_callback);
    EXPECT_FALSE(destroyed_during_callback);
    EXPECT_EQ(registry.find(kFd), nullptr);
}

TEST(EventHandlerRegistry, InPlaceReplacementKeepsToken) {
    EventHandlerRegistry registry;
    constexpr int kFd = 23;
    uint32_t old_calls = 0;
    uint32_t new_calls = 0;
    ASSERT_TRUE(add_handler(
        &registry, kFd,
        [&](int, IOEvent, void*) {
            ++old_calls;
            EXPECT_TRUE(add_handler(
                &registry, kFd,
                [&](int, IOEvent, void*) { ++new_calls; }));
        }));
    const uint64_t token = registry.find(kFd)->token;

    EXPECT_EQ(registry.dispatch(kFd, token, IOEvent::READ),
              HandlerDispatchResult::dispatched);
    ASSERT_NE(registry.find(kFd), nullptr);
    EXPECT_EQ(registry.find(kFd)->token, token);
    EXPECT_EQ(registry.dispatch(kFd, token, IOEvent::READ),
              HandlerDispatchResult::dispatched);
    EXPECT_EQ(old_calls, 1u);
    EXPECT_EQ(new_calls, 1u);
}

TEST(EventHandlerRegistry, SameFdRemoveReaddCannotRestoreOldHandler) {
    EventHandlerRegistry registry;
    constexpr int kFd = 31;
    uint32_t old_calls = 0;
    uint32_t new_calls = 0;
    uint64_t new_token = 0;
    EventHandlerData* reused_slot = nullptr;

    ASSERT_TRUE(add_handler(
        &registry, kFd,
        [&](int, IOEvent, void*) {
            ++old_calls;
            EventHandlerData* const old_slot = registry.find(kFd);
            ASSERT_NE(old_slot, nullptr);
            EXPECT_TRUE(registry.remove(kFd));
            EXPECT_TRUE(add_handler(
                &registry, kFd,
                [&](int, IOEvent, void*) { ++new_calls; }));
            reused_slot = registry.find(kFd);
            ASSERT_NE(reused_slot, nullptr);
            EXPECT_EQ(reused_slot, old_slot);
            new_token = reused_slot->token;
        }));
    EventHandlerData* old_slot = registry.find(kFd);
    ASSERT_NE(old_slot, nullptr);
    const uint64_t old_token = old_slot->token;

    EXPECT_EQ(registry.dispatch(kFd, old_token, IOEvent::READ),
              HandlerDispatchResult::dispatched);
    ASSERT_NE(reused_slot, nullptr);
    EXPECT_NE(new_token, old_token);
    EXPECT_EQ(registry.dispatch(kFd, old_token, IOEvent::READ),
              HandlerDispatchResult::stale);
    EXPECT_EQ(registry.dispatch(kFd, new_token, IOEvent::READ),
              HandlerDispatchResult::dispatched);
    EXPECT_EQ(old_calls, 1u);
    EXPECT_EQ(new_calls, 1u);
}

TEST(EventHandlerRegistry, BusyAndDepthRefusalAreExplicit) {
    EventHandlerRegistry registry;
    constexpr uint32_t kCount = EventHandlerRegistry::kMaxDispatchDepth + 1u;
    uint64_t tokens[kCount] = {};
    HandlerDispatchResult same_fd_result = HandlerDispatchResult::missing;
    HandlerDispatchResult depth_result = HandlerDispatchResult::missing;

    for (uint32_t i = 0; i < kCount; ++i) {
        ASSERT_TRUE(add_handler(
            &registry, static_cast<int>(i),
            [&, i](int, IOEvent, void*) {
                if (i == 0) {
                    same_fd_result = registry.dispatch(
                        0, tokens[0], IOEvent::READ);
                }
                if (i + 1u < kCount) {
                    const HandlerDispatchResult result = registry.dispatch(
                        static_cast<int>(i + 1u), tokens[i + 1u], IOEvent::READ);
                    if (i + 1u == EventHandlerRegistry::kMaxDispatchDepth) {
                        depth_result = result;
                    }
                }
            }));
        tokens[i] = registry.find(static_cast<int>(i))->token;
    }

    EXPECT_EQ(registry.dispatch(0, tokens[0], IOEvent::READ),
              HandlerDispatchResult::dispatched);
    EXPECT_EQ(same_fd_result, HandlerDispatchResult::busy);
    EXPECT_EQ(depth_result, HandlerDispatchResult::depth_refused);
}

TEST(EventHandlerRegistry, InitialAllocationFailureRefusesInsert) {
    bolt::ArenaConfig config;
    config.initial_block_size = std::numeric_limits<size_t>::max();
    config.max_block_size = std::numeric_limits<size_t>::max();
    config.alignment = 64u;
    EventHandlerRegistry registry(config);
    EXPECT_EQ(registry.insert(59), nullptr);
    EXPECT_EQ(registry.size(), 0u);
}
