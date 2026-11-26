// reactor_timers.h — the per-reactor timer heap behind IODispatcher::sleep_for
// (G2CHK-1273). A 4-ary min-heap by deadline, fixed capacity, owner thread only.
// Fires early by at most the timer slack, never late; the reactor's wait
// timeout is the earliest deadline, at microsecond resolution.
#pragma once

#include <cassert>
#include <coroutine>
#include <cstdint>
#include <new>

namespace bolt::api::net {

struct ReactorTimer {
    uint64_t deadline_ns;
    std::coroutine_handle<> handle;
};

class ReactorTimerHeap {
public:
    ReactorTimerHeap() noexcept = default;
    ReactorTimerHeap(const ReactorTimerHeap&) = delete;
    ReactorTimerHeap& operator=(const ReactorTimerHeap&) = delete;
    ~ReactorTimerHeap() { delete[] items_; }

    bool init(uint32_t capacity) noexcept {
        assert(items_ == nullptr && capacity > 0);
        items_ = new (std::nothrow) ReactorTimer[capacity];
        cap_ = items_ != nullptr ? capacity : 0;
        return items_ != nullptr;
    }

    bool push(uint64_t deadline_ns, std::coroutine_handle<> h) noexcept {
        assert(h != nullptr);
        if (n_ == cap_) return false;
        uint32_t i = n_++;
        items_[i] = {deadline_ns, h};
        while (i > 0) {
            const uint32_t p = (i - 1) / 4;
            if (items_[p].deadline_ns <= items_[i].deadline_ns) break;
            swap(p, i);
            i = p;
        }
        return true;
    }

    bool empty() const noexcept { return n_ == 0; }
    uint32_t size() const noexcept { return n_; }
    uint64_t next_deadline() const noexcept {
        assert(n_ > 0);
        return items_[0].deadline_ns;
    }

    std::coroutine_handle<> pop() noexcept {
        assert(n_ > 0);
        std::coroutine_handle<> h = items_[0].handle;
        items_[0] = items_[--n_];
        uint32_t i = 0;
        for (;;) {
            uint32_t m = i;
            for (uint32_t c = 4 * i + 1; c <= 4 * i + 4 && c < n_; ++c) {
                if (items_[c].deadline_ns < items_[m].deadline_ns) m = c;
            }
            if (m == i) break;
            swap(i, m);
            i = m;
        }
        return h;
    }

private:
    void swap(uint32_t a, uint32_t b) noexcept {
        const ReactorTimer t = items_[a];
        items_[a] = items_[b];
        items_[b] = t;
    }
    ReactorTimer* items_ = nullptr;
    uint32_t cap_ = 0;
    uint32_t n_ = 0;
};

}  // namespace bolt::api::net
