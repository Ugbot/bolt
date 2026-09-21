// Compare the four isolation/slot policies in separate, clean builds.
// Unpinned diagnostic: batch latency includes scheduler interference.
#include "bolt/bolt_channel.h"
#include "bolt/bolt_sequence.h"
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t kItems = 1u << 20;
constexpr uint64_t kBatch = 1024;
constexpr uint32_t kRetries = 100000000;
using Channel = bolt::SPSCChannel<uint64_t, 1024>;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool channel_bench() {
    Channel channel;
    std::barrier start(2);
    std::atomic<bool> failed{false};
    std::array<double, kItems / kBatch> samples{};
    std::thread producer([&] {
        start.arrive_and_wait();
        for (uint64_t i = 0; i < kItems; ++i) {
            bool pushed = false;
            for (uint32_t n = 0; n < kRetries && !pushed; ++n) {
                uint64_t value = i;
                pushed = channel.try_push(static_cast<uint64_t&&>(value));
                if (!pushed) bolt::cpu_pause();
                if ((n & 4095u) == 0 && failed.load()) return;
            }
            if (!pushed) { failed.store(true); return; }
        }
    });
    start.arrive_and_wait();
    const auto begin = Clock::now();
    auto batch_begin = begin;
    for (uint64_t i = 0; i < kItems; ++i) {
        uint64_t value = kItems;
        bool popped = false;
        for (uint32_t n = 0; n < kRetries && !popped; ++n) {
            popped = channel.try_pop(&value);
            if (!popped) bolt::cpu_pause();
            if ((n & 4095u) == 0 && failed.load()) break;
        }
        if (!popped || value != i) { failed.store(true); break; }
        if ((i + 1) % kBatch == 0) {
            samples[i / kBatch] = seconds(batch_begin) * 1e6;
            batch_begin = Clock::now();
        }
    }
    const double elapsed = seconds(begin);
    producer.join();
    if (failed.load()) return false;
    std::sort(samples.begin(), samples.end());
    std::printf("spsc,items=%llu,seconds=%.6f,Mitems_s=%.3f,batch_1024_p50_us=%.3f,batch_1024_p99_us=%.3f\n",
        static_cast<unsigned long long>(kItems), elapsed, kItems / elapsed / 1e6,
        samples[samples.size()/2], samples[samples.size()*99/100]);
    return true;
}

bool counter_bench(uint32_t workers) {
    assert(workers > 0);
    assert(workers <= 8);
    bolt::Sequence counters[8];
    std::thread threads[8];
    std::barrier start(static_cast<std::ptrdiff_t>(workers + 1));
    for (uint32_t t = 0; t < workers; ++t) {
        threads[t] = std::thread([&, t] {
            start.arrive_and_wait();
            for (uint64_t i = 0; i < kItems; ++i)
                counters[t].fetch_add_acq_rel(1);
        });
    }
    const auto begin = Clock::now();
    start.arrive_and_wait();
    for (uint32_t t = 0; t < workers; ++t) threads[t].join();
    const double elapsed = seconds(begin);
    for (uint32_t t = 0; t < workers; ++t)
        if (counters[t].load_relaxed() != kItems) return false;
    std::printf("sequence,workers=%u,seconds=%.6f,Mops_s=%.3f\n",
                workers, elapsed, kItems * workers / elapsed / 1e6);
    return true;
}
} // namespace

int main() {
    std::printf("layout,isolation=%zu,slot=%zu,sequence_bytes=%zu,channel_bytes=%zu,hardware_threads=%u\n",
        bolt::config::kCacheIsolationBytes, bolt::config::kChannelSlotAlignmentBytes,
        sizeof(bolt::Sequence), sizeof(Channel), std::thread::hardware_concurrency());
    for (uint32_t workers : {1u, 2u, 4u, 8u})
        if (!counter_bench(workers)) return 1;
    return channel_bench() ? 0 : 1;
}
