#include <gtest/gtest.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <thread>
#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_sequence.h"
#include "bolt/bolt_seqlock.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_row_view.h"
#include "bolt/kernels/bolt_decimal.h"
#include "bolt/bolt_disruptor.h"

namespace {
constexpr size_t kIsolation = bolt::config::kCacheIsolationBytes;
struct Owner { uint8_t prefix; bolt::Sequence counters[2]; uint8_t suffix; };
struct alignas(128) Wide { uint64_t words[16]; };

void check_owner(Owner* owner) {
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(owner) % alignof(Owner), 0u);
    const auto first = reinterpret_cast<uintptr_t>(&owner->counters[0]);
    const auto second = reinterpret_cast<uintptr_t>(&owner->counters[1]);
    EXPECT_EQ(first % kIsolation, 0u);
    EXPECT_EQ(second - first, kIsolation);
    EXPECT_NE(first / kIsolation, second / kIsolation);
}

TEST(BoltAlignment, EmbeddedAndArrayOwners) {
    static_assert(sizeof(bolt::Sequence) == kIsolation);
    static_assert(alignof(bolt::Sequence) == kIsolation);
    static_assert(alignof(bolt::BoltBatch) == 64);
    static_assert(alignof(bolt::RowView) == 64);
    static_assert(sizeof(bolt::ColumnStats) == 64);
    static_assert(alignof(bolt::kernels::decimal::Decimal128) == 16);
    using Lock = bolt::Seqlock<uint64_t>;
    static_assert(alignof(Lock) == kIsolation);
    EXPECT_EQ(offsetof(Lock, value), kIsolation);
    Owner owners[2];
    for (auto& owner : owners) check_owner(&owner);
    using Ring = bolt::Disruptor<uint64_t, 16>;
    EXPECT_EQ(offsetof(Ring, published) - offsetof(Ring, cursor), kIsolation);
    EXPECT_EQ(offsetof(Ring, slots) % kIsolation, 0u);
}

TEST(BoltAlignment, HeapAndPlacementOwners) {
    Owner* heap = new (std::nothrow) Owner;
    check_owner(heap);
    delete heap;
    void* raw = bolt_aligned_alloc(alignof(Owner), sizeof(Owner) * 2u);
    ASSERT_NE(raw, nullptr);
    auto* first = new (raw) Owner;
    auto* second = new (static_cast<uint8_t*>(raw) + sizeof(Owner)) Owner;
    check_owner(first);
    check_owner(second);
    first->~Owner();
    second->~Owner();
    bolt_aligned_free(raw);
}

TEST(BoltAlignment, ArenaRolloverOversizeResetAndExhaustion) {
    bolt::ArenaConfig cfg;
    cfg.initial_block_size = 512;
    cfg.max_block_size = 1024;
    cfg.alignment = 64; // Deliberately weaker than the requested type.
    bolt::Arena arena(cfg);
    for (unsigned epoch = 0; epoch < 2; ++epoch) {
        for (unsigned i = 0; i < 80; ++i) {
            ASSERT_NE(arena.allocate(7, 1), nullptr);
            auto* p = arena.allocate_array<Wide>(1);
            ASSERT_NE(p, nullptr);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % alignof(Wide), 0u);
            p->words[0] = i;
        }
        auto* large = arena.allocate_array<Wide>(64);
        ASSERT_NE(large, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(large) % alignof(Wide), 0u);
        arena.reset();
        // A near-SIZE_MAX request must not wrap while checking reusable blocks.
        EXPECT_EQ(arena.allocate(SIZE_MAX - 256u, 128), nullptr);
    }
    unsigned allocations = 0;
    for (; allocations < bolt::kArenaMaxBlocks + 1u; ++allocations)
        if (!arena.allocate(2048, 128)) break;
    EXPECT_LE(allocations, bolt::kArenaMaxBlocks);
    EXPECT_EQ(arena.allocate(SIZE_MAX, 128), nullptr);
    EXPECT_EQ(arena.allocate_array<Wide>(SIZE_MAX), nullptr);
    EXPECT_EQ(bolt_aligned_alloc(128, SIZE_MAX), nullptr);
}

TEST(BoltAlignment, ChannelSlotPolicyIndependentOfCursors) {
    using Channel = bolt::SPSCChannel<uint64_t, 16>;
    constexpr size_t slots = 16 * bolt::config::kChannelSlotAlignmentBytes;
    constexpr size_t owner_align = kIsolation > bolt::config::kChannelSlotAlignmentBytes
        ? kIsolation : bolt::config::kChannelSlotAlignmentBytes;
    constexpr size_t bytes = slots + kIsolation + sizeof(size_t);
    constexpr size_t expected = (bytes + owner_align - 1u) & ~(owner_align - 1u);
    EXPECT_EQ(sizeof(Channel), expected);
    Channel channel;
    for (uint64_t i = 0; i < 16; ++i) {
        uint64_t value = i;
        ASSERT_TRUE(channel.try_push(static_cast<uint64_t&&>(value)));
    }
    uint64_t extra = 17;
    EXPECT_FALSE(channel.try_push(static_cast<uint64_t&&>(extra)));
    for (uint64_t i = 0; i < 16; ++i) {
        uint64_t value = 99;
        ASSERT_TRUE(channel.try_pop(&value));
        EXPECT_EQ(value, i);
    }
    EXPECT_FALSE(channel.try_pop(&extra));
}

TEST(BoltAlignment, ConcurrentHandoffPreservesOrder) {
    bolt::SPSCChannel<uint64_t, 256> channel;
    constexpr uint64_t count = 100000;
    std::atomic<bool> failed{false};
    std::thread producer([&] {
        for (uint64_t i = 0; i < count; ++i) {
            bool pushed = false;
            for (uint32_t retry = 0; retry < 10000000u && !pushed; ++retry) {
                uint64_t value = i;
                pushed = channel.try_push(static_cast<uint64_t&&>(value));
                if (!pushed && failed.load(std::memory_order_relaxed)) return;
                if (!pushed) bolt::cpu_pause();
            }
            if (!pushed) { failed.store(true); return; }
        }
    });
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t value = count;
        bool popped = false;
        for (uint32_t retry = 0; retry < 10000000u && !popped; ++retry) {
            popped = channel.try_pop(&value);
            if (!popped && failed.load(std::memory_order_relaxed)) break;
            if (!popped) bolt::cpu_pause();
        }
        if (!popped || value != i) { failed.store(true); break; }
    }
    producer.join();
    EXPECT_FALSE(failed.load());
}
} // namespace
