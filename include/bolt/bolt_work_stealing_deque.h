// bolt_work_stealing_deque.h — Chase-Lev work-stealing deque.
//
// Owner thread pushes/pops at the *bottom* (relaxed). Thief threads
// steal from the *top* (acquire/release + seq_cst fence on the race).
// Bounded capacity, no allocation in steady state.
//
// Reference: Chase & Lev, "Dynamic Circular Work-Stealing Deque" (SPAA'05),
// with the simpler bounded variant. Memory model follows Le et al.,
// "Correct and Efficient Work-Stealing for Weak Memory Models" (PPoPP'13).
//
// RULES (per CLAUDE.md):
//   - Header-only. No exceptions. No RTTI. No smart pointers.
//   - All shared atomics cache-line padded.
//   - Functions <= 70 LOC, >= 2 asserts on hot paths.
//   - Stores T* (raw pointer); ownership is lexical, not refcounted.
//
// Borrow target: chukonu-v1 WorkStealingDeque (no changes needed; it was
// already pointer-only).

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
class WorkStealingDeque {
    static_assert(Capacity > 0,                          "Capacity must be positive");
    static_assert((Capacity & (Capacity - 1)) == 0,      "Capacity must be power of 2");
    static constexpr size_t kMask = Capacity - 1;

public:
    WorkStealingDeque() noexcept {
        top_.store(0,    std::memory_order_relaxed);
        bottom_.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    WorkStealingDeque(const WorkStealingDeque&)            = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    // Owner only. Returns false if full.
    bool push(T* item) noexcept {
        assert(item != nullptr);
        const int64_t b = bottom_.load(std::memory_order_relaxed);
        const int64_t t = top_.load(std::memory_order_acquire);
        assert(b >= t);
        if (b - t >= static_cast<int64_t>(Capacity)) {
            return false; // full
        }
        buffer_[b & kMask].store(item, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
        return true;
    }

    // Owner only. Returns nullptr if empty (or if a steal won the race
    // for the last item).
    T* pop() noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = top_.load(std::memory_order_relaxed);

        T* item = nullptr;
        if (t <= b) {
            // Non-empty: at least one element. Read it speculatively.
            item = buffer_[b & kMask].load(std::memory_order_relaxed);
            assert(item != nullptr);
            if (t == b) {
                // Last item — race against thieves to claim it.
                if (!top_.compare_exchange_strong(t, t + 1,
                                                  std::memory_order_seq_cst,
                                                  std::memory_order_relaxed)) {
                    item = nullptr; // lost the race
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        }
        // Empty — restore bottom.
        bottom_.store(b + 1, std::memory_order_relaxed);
        return nullptr;
    }

    // Thief threads. Returns nullptr on empty or lost race.
    T* steal() noexcept {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const int64_t b = bottom_.load(std::memory_order_acquire);
        if (t < b) {
            T* item = buffer_[t & kMask].load(std::memory_order_relaxed);
            assert(item != nullptr);
            if (top_.compare_exchange_strong(t, t + 1,
                                             std::memory_order_seq_cst,
                                             std::memory_order_relaxed)) {
                return item;
            }
            return nullptr; // lost the race
        }
        return nullptr; // empty
    }

    bool empty() const noexcept {
        const int64_t t = top_.load(std::memory_order_acquire);
        const int64_t b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }

    size_t approx_size() const noexcept {
        const int64_t t = top_.load(std::memory_order_relaxed);
        const int64_t b = bottom_.load(std::memory_order_relaxed);
        return b > t ? static_cast<size_t>(b - t) : 0;
    }

    constexpr size_t capacity() const noexcept { return Capacity; }

private:
    alignas(kCacheLineSize) std::atomic<int64_t> top_;
    alignas(kCacheLineSize) std::atomic<int64_t> bottom_;
    alignas(kCacheLineSize) std::array<std::atomic<T*>, Capacity> buffer_;
};

} // namespace bolt
