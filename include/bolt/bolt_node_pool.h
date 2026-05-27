// bolt_node_pool.h — Lock-free pre-allocated object pool with ABA counter.
//
// Pre-allocates `Capacity` slots at construction; allocate/free push and
// pop indices on a lock-free freelist guarded by a 16-bit ABA counter
// packed into the high bits of a 64-bit atomic. Zero-allocation in
// steady state.
//
// Layout of free_head_ (uint64_t):
//   bits  0..47 : freelist head index (or kNullIndex sentinel)
//   bits 48..63 : ABA counter
//
// RULES (per CLAUDE.md):
//   - Header-only. No exceptions. No RTTI. No smart pointers.
//   - No std::string / std::vector / std::map.
//   - >=2 asserts per hot-path function. Functions <= 70 LOC.
//   - All shared atomics cache-line padded.
//
// Borrow target: chukonu-v1 NodePool (std::vector + Arrow-free).

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_port.h"

namespace bolt {

#ifndef BOLT_KCACHELINE_DEFINED
#define BOLT_KCACHELINE_DEFINED
static constexpr size_t kCacheLineSize = 64;
#endif

template <typename T, size_t Capacity = 4096>
class NodePool {
    static_assert(Capacity > 0,              "Capacity must be positive");
    static_assert(Capacity < (1ull << 47),   "Capacity must fit in 47 bits");

    static constexpr uint64_t kIndexBits      = 48;
    static constexpr uint64_t kIndexMask      = (1ull << kIndexBits) - 1;
    static constexpr uint64_t kCounterIncr    = 1ull << kIndexBits;
    static constexpr uint64_t kNullIndex      = kIndexMask; // sentinel

public:
    NodePool() noexcept {
        // Link every slot into the freelist: slot[i].next = i + 1, last = null.
        for (size_t i = 0; i + 1 < Capacity; ++i) {
            slots_[i].next_free = static_cast<uint64_t>(i + 1);
        }
        slots_[Capacity - 1].next_free = kNullIndex;
        // Head starts at index 0, counter 0.
        free_head_.store(0, std::memory_order_relaxed);
        in_use_.store(0, std::memory_order_relaxed);
    }

    NodePool(const NodePool&)            = delete;
    NodePool& operator=(const NodePool&) = delete;

    // Pop the head of the freelist. Returns nullptr if exhausted.
    T* allocate() noexcept {
        uint64_t head = free_head_.load(std::memory_order_acquire);
        for (;;) {
            const uint64_t idx = head & kIndexMask;
            if (idx == kNullIndex) {
                return nullptr; // pool exhausted
            }
            assert(idx < Capacity);
            // Read the link BEFORE the CAS. If we lose the CAS we restart;
            // the slot's `next_free` is only ever written by whoever holds
            // the slot off the freelist, so this read races safely with
            // other allocators (they read different slots).
            const uint64_t next = slots_[idx].next_free;
            const uint64_t new_head =
                (next & kIndexMask) | ((head + kCounterIncr) & ~kIndexMask);
            if (free_head_.compare_exchange_weak(head, new_head,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                in_use_.fetch_add(1, std::memory_order_relaxed);
                assert(idx < Capacity);
                return &slots_[idx].data;
            }
        }
    }

    // Push back onto the freelist.
    void deallocate(T* ptr) noexcept {
        if (ptr == nullptr) return;
        // Compute slot index from offset. UB if ptr was not from this pool.
        Slot* s = reinterpret_cast<Slot*>(
            reinterpret_cast<char*>(ptr) - offsetof(Slot, data));
        const ptrdiff_t signed_idx = s - slots_.data();
        assert(signed_idx >= 0);
        assert(static_cast<size_t>(signed_idx) < Capacity);
        const uint64_t idx = static_cast<uint64_t>(signed_idx);

        uint64_t head = free_head_.load(std::memory_order_acquire);
        for (;;) {
            s->next_free = head & kIndexMask;
            const uint64_t new_head =
                idx | ((head + kCounterIncr) & ~kIndexMask);
            if (free_head_.compare_exchange_weak(head, new_head,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                in_use_.fetch_sub(1, std::memory_order_relaxed);
                return;
            }
        }
    }

    size_t in_use() const noexcept {
        return in_use_.load(std::memory_order_relaxed);
    }

    bool exhausted() const noexcept {
        return (free_head_.load(std::memory_order_relaxed) & kIndexMask) == kNullIndex;
    }

    constexpr size_t capacity() const noexcept { return Capacity; }

private:
    struct Slot {
        T        data;
        uint64_t next_free; // index into slots_, or kNullIndex
    };

    std::array<Slot, Capacity> slots_;
    alignas(kCacheLineSize) std::atomic<uint64_t> free_head_;
    alignas(kCacheLineSize) std::atomic<size_t>   in_use_;
};

} // namespace bolt
