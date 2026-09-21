// event_handler_registry.h — bounded fd handler storage and dispatch pins.
//
// This module is single-threaded. Event-loop registration, removal, and
// dispatch must all run on the loop owner thread. Dispatch moves the active
// std::function into a fixed pin stack, so callbacks may remove or replace
// their own registration without destroying the callable while it executes.

#pragma once

#include "fd_registry.h"

#include "bolt/api/net/event_loop.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <utility>

namespace bolt::api::net {

static_assert(noexcept(std::declval<EventHandler&>().swap(
    std::declval<EventHandler&>())));

enum class HandlerDispatchResult : uint8_t {
    dispatched,
    missing,
    stale,
    busy,
    depth_refused,
};

struct EventHandlerData {
    EventHandler handler;
    void* user_data = nullptr;
    IOEvent events = static_cast<IOEvent>(0);
    uint64_t token = 0;
};

class EventHandlerRegistry {
public:
    static constexpr uint32_t kMaxDispatchDepth = 16;

    EventHandlerRegistry() noexcept = default;
    explicit EventHandlerRegistry(ArenaConfig arena_config) noexcept
        : handlers_(arena_config) {}

    EventHandlerData* insert(int fd) noexcept {
        if (fd < 0) return nullptr;
        EventHandlerData* data = handlers_.find(static_cast<uint64_t>(fd));
        if (data != nullptr) return data;
        if (next_token_ == 0) return nullptr;
        data = handlers_.insert(static_cast<uint64_t>(fd));
        if (data == nullptr) return nullptr;
        assert(data->token == 0);
        data->token = next_token_++;
        assert(data->token != 0);
        return data;
    }

    EventHandlerData* find(int fd) noexcept {
        if (fd < 0) return nullptr;
        return handlers_.find(static_cast<uint64_t>(fd));
    }

    bool remove(int fd) noexcept {
        if (fd < 0) return false;
        return handlers_.erase(static_cast<uint64_t>(fd));
    }

    HandlerDispatchResult dispatch(
        int fd, uint64_t expected_token, IOEvent events) noexcept {
        assert(fd >= 0);
        assert(expected_token != 0);

        EventHandlerData* data = handlers_.find(static_cast<uint64_t>(fd));
        if (data == nullptr) return HandlerDispatchResult::missing;
        if (data->token != expected_token) return HandlerDispatchResult::stale;
        if (!data->handler) return HandlerDispatchResult::busy;
        if (pin_depth_ >= kMaxDispatchDepth) {
            return HandlerDispatchResult::depth_refused;
        }

        const uint32_t pin_index = pin_depth_++;
        EventHandler& pinned = pins_[pin_index];
        assert(!pinned);
        EventHandlerData* const pinned_slot = data;
        void* const user_data = data->user_data;
        pinned.swap(data->handler);
        assert(static_cast<bool>(pinned));

        pinned(fd, events, user_data);

        assert(pin_depth_ == pin_index + 1u);
        data = handlers_.find(static_cast<uint64_t>(fd));
        if (data == pinned_slot && data->token == expected_token && !data->handler) {
            data->handler.swap(pinned);
        }
        if (pinned) pinned = nullptr;
        --pin_depth_;
        assert(!pinned);
        return HandlerDispatchResult::dispatched;
    }

    uint32_t size() const noexcept { return handlers_.size(); }

private:
    FdRegistry<EventHandlerData> handlers_;
    std::array<EventHandler, kMaxDispatchDepth> pins_{};
    uint64_t next_token_ = 1;
    uint32_t pin_depth_ = 0;
};

}  // namespace bolt::api::net
