// test_bolt_park_stress.h — lost-wake stress for bolt::park (G2CHK-1293).
// A consumer waits on a counter through a park::Slot; a producer publishes the
// next value at varied gaps, wakes, and waits for the consumer to take it
// (so a later publish cannot rescue a lost wake). A wake is lost when the
// consumer resumes long after the value it waited for was published.
#pragma once

#include "bolt/bolt_park.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace park_stress {

struct Result {
    uint64_t lost = 0;           // consumer resumed > kLostAfterNs after the publish
    uint64_t parks = 0;
    uint64_t kernel_wakes = 0;
    uint64_t max_late_ns = 0;
};

inline constexpr uint64_t kLostAfterNs = 50u * 1000u * 1000u;

inline Result run(uint32_t items, bolt::park::WaitMode mode, bool stop_on_lost = false) {
    alignas(64) std::atomic<uint64_t> value{0};
    alignas(64) std::atomic<uint64_t> consumed{0};
    alignas(64) std::atomic<uint64_t> published_at{0};
    std::atomic<bool> stop{false};
    bolt::park::Slot slot;
    bolt::park::WaitPolicy p;
    p.mode = mode;
    p.spin_ns = 2000;
    p.deadline_ms = 200;
    Result r;
    std::thread consumer([&] {
        for (uint64_t want = 1; want <= items; ++want) {
            (void)bolt::park::wait(&slot, [&]() noexcept {
                return value.load(std::memory_order_acquire) >= want;
            }, 0, p);
            const uint64_t at = published_at.load(std::memory_order_acquire);
            const uint64_t now = bolt::park::now_ns();
            const uint64_t late = now > at ? now - at : 0;
            if (late > kLostAfterNs) {
                ++r.lost;
                if (stop_on_lost) stop.store(true, std::memory_order_release);
            }
            if (late > r.max_late_ns) r.max_late_ns = late;
            consumed.store(want, std::memory_order_release);
            if (stop.load(std::memory_order_acquire)) break;
        }
    });
    for (uint64_t v = 1; v <= items && !stop.load(std::memory_order_acquire); ++v) {
        const uint32_t gap = static_cast<uint32_t>((v * 2654435761u) % 40u);
        if (gap > 30) std::this_thread::sleep_for(std::chrono::microseconds(gap));
        else for (uint32_t i = 0; i < gap * 50u; ++i) BOLT_PAUSE();
        published_at.store(bolt::park::now_ns(), std::memory_order_release);
        value.store(v, std::memory_order_release);
        (void)bolt::park::wake(&slot);
        while (consumed.load(std::memory_order_acquire) < v) std::this_thread::yield();
    }
    consumer.join();
    r.parks = slot.announces;
    r.kernel_wakes = slot.kernel_wakes.load();
    return r;
}

}  // namespace park_stress
