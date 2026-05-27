// bolt_sharded_mpsc.h — Sharded multi-producer / single-consumer queue.
//
// Many producers route by thread-id hash to one of N independent MPSC
// ring buffers; a single consumer drains them round-robin. Reduces
// producer contention vs a single MPSC by ~N.
//
// RULES (per CLAUDE.md):
//   - Header-only. No exceptions. No RTTI. No smart pointers.
//   - No std::string / std::vector / std::map.
//   - Cache-line padded shared atomics.
//   - Bounded; no allocation in steady state.
//   - All functions noexcept, <= 70 LOC, >= 2 asserts on hot paths.
//
// Borrow target: chukonu-v1 ShardedMPSCQueue (Arrow + shared_ptr stripped).

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>

#include "bolt/bolt_port.h"

namespace bolt {

#ifndef BOLT_KCACHELINE_DEFINED
#define BOLT_KCACHELINE_DEFINED
static constexpr size_t kCacheLineSize = 64;
#endif

// One MPSC shard: bounded ring buffer, multi-producer claim via CAS,
// single-consumer pop. Same structure as MPSCChannel in bolt_channel.h
// but exposed as a free-standing primitive so the sharded queue can hold
// an array of shards inline (no heap, no smart pointers).
template <typename T, size_t Capacity>
class MPSCShard {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr size_t kMask = Capacity - 1;

public:
    MPSCShard() noexcept {
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
        }
        wpos_.store(0, std::memory_order_relaxed);
        rpos_.store(0, std::memory_order_relaxed);
    }

    MPSCShard(const MPSCShard&)            = delete;
    MPSCShard& operator=(const MPSCShard&) = delete;

    // Producer side. Multi-thread safe. Returns false if shard is full.
    bool try_push(T&& item) noexcept {
        size_t pos = wpos_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s   = slots_[pos & kMask];
            size_t sq = s.seq.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(sq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (wpos_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
                    s.data = static_cast<T&&>(item);
                    s.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false; // full
            } else {
                pos = wpos_.load(std::memory_order_relaxed);
            }
        }
    }

    // Consumer side. Single-thread only.
    bool try_pop(T* out) noexcept {
        assert(out != nullptr);
        size_t pos = rpos_.load(std::memory_order_relaxed);
        Slot& s    = slots_[pos & kMask];
        size_t sq  = s.seq.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(sq) - static_cast<intptr_t>(pos + 1);
        if (diff < 0) return false; // empty
        assert(diff == 0); // single consumer: must be exactly the next slot
        *out = static_cast<T&&>(s.data);
        s.seq.store(pos + Capacity, std::memory_order_release);
        rpos_.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        size_t r = rpos_.load(std::memory_order_acquire);
        size_t w = wpos_.load(std::memory_order_acquire);
        return r == w;
    }

private:
    struct alignas(kCacheLineSize) Slot {
        std::atomic<size_t> seq;
        T data;
    };
    std::array<Slot, Capacity> slots_;
    alignas(kCacheLineSize) std::atomic<size_t> wpos_;
    alignas(kCacheLineSize) std::atomic<size_t> rpos_;
};

// Sharded multi-producer / single-consumer queue.
// NShards must be a power of two so we can mask the hash.
template <typename T, size_t NShards = 16, size_t ShardCapacity = 1024>
class ShardedMPSCQueue {
    static_assert(NShards > 0, "NShards must be positive");
    static_assert((NShards & (NShards - 1)) == 0, "NShards must be power of 2");
    static constexpr size_t kShardMask = NShards - 1;

public:
    ShardedMPSCQueue() noexcept : drain_cursor_(0) {}

    ShardedMPSCQueue(const ShardedMPSCQueue&)            = delete;
    ShardedMPSCQueue& operator=(const ShardedMPSCQueue&) = delete;

    // Push to a caller-chosen shard (any thread). Returns false if shard full.
    bool try_push(T&& item, size_t shard_hint) noexcept {
        const size_t s = shard_hint & kShardMask;
        assert(s < NShards);
        return shards_[s].try_push(static_cast<T&&>(item));
    }

    // Push to the shard derived from this thread's id. Hash computed once
    // per thread via thread_local cache (no allocation, no syscall on hot
    // path).
    bool try_push_local(T&& item) noexcept {
        return shards_[my_shard()].try_push(static_cast<T&&>(item));
    }

    // Consumer side. Round-robin scan from drain_cursor_. Returns true if
    // any shard yielded an item.
    bool try_pop(T* out) noexcept {
        assert(out != nullptr);
        size_t start = drain_cursor_;
        for (size_t i = 0; i < NShards; ++i) {
            size_t s = (start + i) & kShardMask;
            if (shards_[s].try_pop(out)) {
                drain_cursor_ = (s + 1) & kShardMask;
                return true;
            }
        }
        return false;
    }

    // Drain everything currently visible. Returns number of items handed off.
    template <typename Fn>
    size_t drain(Fn&& fn) noexcept {
        size_t count = 0;
        T item;
        for (size_t s = 0; s < NShards; ++s) {
            while (shards_[s].try_pop(&item)) {
                fn(static_cast<T&&>(item));
                ++count;
            }
        }
        return count;
    }

    bool empty() const noexcept {
        for (size_t s = 0; s < NShards; ++s) {
            if (!shards_[s].empty()) return false;
        }
        return true;
    }

    constexpr size_t num_shards() const noexcept { return NShards; }

private:
    static size_t my_shard() noexcept {
        thread_local size_t cached =
            std::hash<std::thread::id>{}(std::this_thread::get_id()) & kShardMask;
        return cached;
    }

    std::array<MPSCShard<T, ShardCapacity>, NShards> shards_;
    alignas(kCacheLineSize) size_t drain_cursor_; // single-consumer state
};

} // namespace bolt
