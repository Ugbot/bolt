#include <gtest/gtest.h>
#include "bolt/bolt_seqlock.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
template<class Predicate> bool wait_for(Predicate ready) {
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (!ready() && Clock::now() < deadline) std::this_thread::yield();
    return ready();
}
template<size_t N> void exact_bytes() {
    struct Bytes { uint8_t bytes[N]; };
    Bytes initial{}, next{}, output{};
    for (size_t i = 0u; i < N; ++i) { initial.bytes[i] = static_cast<uint8_t>(i + 1u); next.bytes[i] = 0xa5u; }
    bolt::Seqlock<Bytes> lock; lock.init(initial);
    ASSERT_TRUE(lock.try_read(&output)); EXPECT_EQ(std::memcmp(&initial, &output, N), 0);
    lock.write(next); ASSERT_TRUE(lock.try_read(&output)); EXPECT_EQ(std::memcmp(&next, &output, N), 0);
    EXPECT_EQ(output.bytes[N - 1u], 0xa5u);
}
TEST(Seqlock, TinyAndOddTailRepresentationsAreExact) {
    ASSERT_NO_FATAL_FAILURE(exact_bytes<1u>());
    ASSERT_NO_FATAL_FAILURE(exact_bytes<3u>());
    ASSERT_NO_FATAL_FAILURE(exact_bytes<7u>());
    ASSERT_NO_FATAL_FAILURE(exact_bytes<9u>());
    ASSERT_NO_FATAL_FAILURE(exact_bytes<137u>());
}
TEST(Seqlock, InitializedPaddingAlignmentAndPointerBitsSurvive) {
    struct alignas(2u * bolt::config::kCacheIsolationBytes) Padded {
        uint8_t tag; uint64_t count; const uint64_t* pointer; uint8_t tail;
    };
    const uint64_t pointee = 77u;
    Padded input; std::memset(&input, 0x5a, sizeof(input));
    input.tag = 3u; input.count = 1234u; input.pointer = &pointee; input.tail = 0xf1u;
    bolt::Seqlock<Padded> lock; lock.init(input); Padded output{};
    ASSERT_TRUE(lock.try_read(&output)); EXPECT_EQ(std::memcmp(&input, &output, sizeof(input)), 0);
    EXPECT_EQ(output.pointer, &pointee); EXPECT_EQ(*output.pointer, 77u); EXPECT_EQ(output.tail, 0xf1u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&lock.value) % alignof(Padded), 0u);
    input.count = 5678u; input.tail = 0x12u; lock.write(input);
    ASSERT_TRUE(lock.try_read(&output)); EXPECT_EQ(std::memcmp(&input, &output, sizeof(input)), 0);
}
struct Correlated { uint64_t epoch; uint64_t cells[40]; uint8_t tail[3]; };
Correlated make_value(uint64_t epoch) {
    Correlated value{}; value.epoch = epoch;
    for (size_t i = 0u; i < 40u; ++i) value.cells[i] = epoch ^ (0x9e3779b97f4a7c15ull * (i + 1u));
    for (size_t i = 0u; i < 3u; ++i) value.tail[i] = static_cast<uint8_t>(epoch + i);
    return value;
}
bool consistent(const Correlated& value) {
    for (size_t i = 0u; i < 40u; ++i)
        if (value.cells[i] != (value.epoch ^ (0x9e3779b97f4a7c15ull * (i + 1u)))) return false;
    for (size_t i = 0u; i < 3u; ++i)
        if (value.tail[i] != static_cast<uint8_t>(value.epoch + i)) return false;
    return true;
}
TEST(Seqlock, SingleWriterAndMultipleReadersAcceptOnlyCorrelatedEpochs) {
    constexpr uint32_t kReaders = 4u, kWrites = 30000u, kReads = 40000u;
    bolt::Seqlock<Correlated> lock; lock.init(make_value(0u));
    std::atomic<uint32_t> ready{0u}; std::atomic<bool> start{false}, bad{false}, timed_out{false};
    std::atomic<bool> writer_done{false};
    std::array<std::atomic<uint32_t>, kReaders> accepted{};
    std::array<std::atomic<uint64_t>, kReaders> final_epochs{};
    std::array<std::thread, kReaders> readers;
    for (uint32_t i = 0u; i < kReaders; ++i) readers[i] = std::thread([&, i] {
        ready.fetch_add(1u, std::memory_order_release);
        if (!wait_for([&] { return start.load(std::memory_order_acquire); })) { timed_out.store(true); return; }
        for (uint32_t n = 0u; n < kReads; ++n) {
            Correlated output{};
            if (!lock.try_read(&output)) continue;
            accepted[i].fetch_add(1u, std::memory_order_relaxed);
            if (!consistent(output)) bad.store(true, std::memory_order_relaxed);
        }
        // Continuous writing may legitimately refuse every concurrent attempt.
        // Require each reader to make real progress in a stable opportunity.
        if (!wait_for([&] { return writer_done.load(std::memory_order_acquire); })) { timed_out.store(true); return; }
        Correlated output{};
        if (!lock.try_read(&output)) { bad.store(true); return; }
        accepted[i].fetch_add(1u, std::memory_order_relaxed);
        if (!consistent(output)) bad.store(true, std::memory_order_relaxed);
        final_epochs[i].store(output.epoch, std::memory_order_relaxed);
    });
    const bool all_ready = wait_for([&] { return ready.load(std::memory_order_acquire) == kReaders; });
    start.store(true, std::memory_order_release);
    for (uint32_t n = 1u; n <= kWrites; ++n) lock.write(make_value(n));
    writer_done.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();
    EXPECT_TRUE(all_ready); EXPECT_FALSE(timed_out.load()); EXPECT_FALSE(bad.load());
    for (const auto& count : accepted) EXPECT_GT(count.load(), 0u);
    for (const auto& epoch : final_epochs) EXPECT_EQ(epoch.load(), kWrites);
    const auto final = lock.read(); EXPECT_EQ(final.epoch, kWrites); EXPECT_TRUE(consistent(final));
}
TEST(Seqlock, ActualOddWriterPublicationRefusesBoundedlyAndRecoversExactly) {
    bolt::Seqlock<Correlated> lock; lock.init(make_value(5u));
    std::atomic<bool> odd{false}, release{false}, timed_out{false};
    std::thread writer([&] {
        // Same actual sequence publication used by write. The owner pauses
        // before changing cells, ends that epoch, then performs a normal write.
        const auto before = lock.seq.fetch_add_acq_rel(1u); EXPECT_EQ(before & 1u, 0u);
        odd.store(true, std::memory_order_release);
        if (!wait_for([&] { return release.load(std::memory_order_acquire); })) timed_out.store(true);
        lock.seq.fetch_add_acq_rel(1u); lock.write(make_value(19u));
    });
    const bool published = wait_for([&] { return odd.load(std::memory_order_acquire); });
    Correlated output{}; const auto before = Clock::now(); const bool read = lock.try_read(&output);
    const auto elapsed = Clock::now() - before;
    release.store(true, std::memory_order_release); writer.join();
    ASSERT_TRUE(published); EXPECT_FALSE(read); EXPECT_LT(elapsed, std::chrono::seconds(1));
    EXPECT_FALSE(timed_out.load()); ASSERT_TRUE(lock.try_read(&output));
    EXPECT_EQ(output.epoch, 19u); EXPECT_TRUE(consistent(output));
}
} // namespace
