// bolt_sequence.h — cache-line-padded monotonic sequence counter.
//
// RULES: No exceptions. No RTTI. POD-ish (has an atomic member). Single
// cache line (64 B). Zero allocation — embed by value in other structs.
//
// Motivation. Every lock-free primitive we build — Disruptor, Seqlock,
// RetireQueue, the scheduler's park/notify, producer-consumer cursors
// — needs the same shape: an aligned 64-bit atomic counter with
// release/acquire semantics and C++20 `atomic::wait`/`notify_one`
// parking. Today the pattern is open-coded in several places
// (`scheduler.submit_seq_`, `Zone::readers_` conceptually, WAL's
// `reserved_len` / `committed_len`). This header gives it a name and
// a single implementation.
//
// Design:
//   - `alignas(64)` on both the struct and its atomic field guarantees
//     no false sharing across neighbouring Sequences.
//   - All operations are `inline` and `BOLT_FORCE_INLINE` so the
//     compiler inlines through to a single atomic op + branch.
//   - `wait(expected)` uses `std::atomic<uint64_t>::wait` (C++20);
//     backed by `WaitOnAddress` on Windows and `futex(FUTEX_WAIT)` on
//     Linux. No mutex, no condition_variable.

#pragma once

#include "bolt/bolt_port.h"

#include <atomic>
#include <cstdint>

namespace bolt {

struct alignas(64) Sequence {
    alignas(64) std::atomic<uint64_t> value;

    BOLT_FORCE_INLINE Sequence() noexcept : value(0) {}

    BOLT_FORCE_INLINE uint64_t load_relaxed() const noexcept {
        return value.load(std::memory_order_relaxed);
    }
    BOLT_FORCE_INLINE uint64_t load_acquire() const noexcept {
        return value.load(std::memory_order_acquire);
    }
    BOLT_FORCE_INLINE void store_relaxed(uint64_t v) noexcept {
        value.store(v, std::memory_order_relaxed);
    }
    BOLT_FORCE_INLINE void store_release(uint64_t v) noexcept {
        value.store(v, std::memory_order_release);
    }
    BOLT_FORCE_INLINE uint64_t fetch_add_acq_rel(uint64_t n) noexcept {
        return value.fetch_add(n, std::memory_order_acq_rel);
    }
    BOLT_FORCE_INLINE bool cas(uint64_t& expected, uint64_t desired) noexcept {
        return value.compare_exchange_strong(
            expected, desired,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    BOLT_FORCE_INLINE bool cas_weak(uint64_t& expected, uint64_t desired) noexcept {
        return value.compare_exchange_weak(
            expected, desired,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    // Park the calling thread until `value != expected`. Backed by
    // WaitOnAddress / futex.
    BOLT_FORCE_INLINE void wait(uint64_t expected) noexcept {
        value.wait(expected, std::memory_order_acquire);
    }
    BOLT_FORCE_INLINE void notify_one() noexcept { value.notify_one(); }
    BOLT_FORCE_INLINE void notify_all() noexcept { value.notify_all(); }

    // Non-copyable, non-movable — embedded by value.
    Sequence(const Sequence&)            = delete;
    Sequence& operator=(const Sequence&) = delete;
    Sequence(Sequence&&)                 = delete;
    Sequence& operator=(Sequence&&)      = delete;
};

static_assert(sizeof(Sequence) == 64, "Sequence must be cache-line sized");
static_assert(alignof(Sequence) == 64, "Sequence must be cache-line aligned");

}  // namespace bolt
