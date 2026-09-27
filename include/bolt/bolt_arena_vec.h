// bolt_arena_vec.h — ArenaVec<T> (kind C) and ArenaRingBuffer<T> (kind B).
//
// ArenaVec: an arena-backed, budget-charged growable array. It grows ×2, and
// ONLY in reserve()/try_push()/resize() — call those at an open/plan/batch
// boundary and hand per-row loops the span (data(), size()). Past its Budget
// (or when the arena cannot supply the block) growth fails with
// ResourceExhausted set and the vector unchanged; it never grows silently and
// never truncates.
//
// The arena cannot free, so a grown-out-of block stays allocated until the
// arena resets; the budget is charged for every block the vector has taken
// (≤ 2× the final capacity) and destroy() returns the whole charge.
//
// ArenaRingBuffer: a fixed-capacity FIFO sized once at init (kind B). Full is
// a refusal the caller turns into backpressure; it never overwrites.
//
// Single-threaded. T must be trivially copyable. No RAII: call destroy().
// Named ArenaRingBuffer because bolt::ArenaRing is the pool of Arenas in
// bolt_arena_ring.h.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_resource.h"

namespace bolt {

template <typename T>
class ArenaVec {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ArenaVec<T> copies elements with memcpy");

public:
    ArenaVec() noexcept = default;
    ArenaVec(const ArenaVec&) = delete;
    ArenaVec& operator=(const ArenaVec&) = delete;

    // budget may be nullptr (arena-bounded only). initial_cap may be 0.
    bool init(Arena* arena, Budget* budget, size_t initial_cap) noexcept {
        assert(arena != nullptr);
        assert(arena_ == nullptr && "ArenaVec::init called twice");
        arena_ = arena;
        budget_ = budget;
        data_ = nullptr;
        size_ = 0;
        cap_ = 0;
        charged_ = 0;
        return initial_cap == 0 || grow_to(initial_cap);
    }

    // Return every byte charged to the budget. The storage itself goes back
    // with the arena.
    void destroy() noexcept {
        if (budget_ != nullptr && charged_ != 0) budget_->release(charged_);
        charged_ = 0;
        data_ = nullptr;
        size_ = 0;
        cap_ = 0;
        arena_ = nullptr;
        budget_ = nullptr;
    }

    // Ensure capacity >= n. Grows to max(n, 2 × capacity).
    bool reserve(size_t n) noexcept {
        assert(arena_ != nullptr);
        if (n <= cap_) return true;
        size_t want = cap_ > (kMaxElems / 2) ? kMaxElems : cap_ * 2;
        if (want < n) want = n;
        if (want < kMinGrow) want = kMinGrow;
        return grow_to(want);
    }

    bool try_push(const T& v) noexcept {
        if (BOLT_UNLIKELY(size_ == cap_) && !reserve(size_ + 1)) return false;
        assert(size_ < cap_);
        data_[size_++] = v;
        return true;
    }

    // Grow or shrink the logical size; new elements are zero-filled.
    bool resize(size_t n) noexcept {
        if (!reserve(n)) return false;
        if (n > size_) std::memset(data_ + size_, 0, (n - size_) * sizeof(T));
        size_ = n;
        assert(size_ <= cap_);
        return true;
    }

    void clear() noexcept { size_ = 0; }
    void pop_back() noexcept { assert(size_ > 0); --size_; }

    T& operator[](size_t i) noexcept { assert(i < size_); return data_[i]; }
    const T& operator[](size_t i) const noexcept { assert(i < size_); return data_[i]; }
    T& back() noexcept { assert(size_ > 0); return data_[size_ - 1]; }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return cap_; }
    bool empty() const noexcept { return size_ == 0; }
    uint64_t charged_bytes() const noexcept { return charged_; }

private:
    static constexpr size_t kMaxElems = SIZE_MAX / sizeof(T);
    static constexpr size_t kMinGrow  = 64 / sizeof(T) > 0 ? 64 / sizeof(T) : 1;

    bool grow_to(size_t want) noexcept {
        assert(want > cap_);
        if (want > kMaxElems) {
            set_resource_exhausted("arena_vec_elements", want, kMaxElems);
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(want) * sizeof(T);
        if (budget_ != nullptr && !budget_->try_reserve(bytes)) return false;
        T* next = static_cast<T*>(arena_->allocate(static_cast<size_t>(bytes), alignof(T)));
        if (next == nullptr) {
            if (budget_ != nullptr) budget_->release(bytes);
            set_resource_exhausted("arena_max_blocks", bytes, 0);
            return false;
        }
        if (size_ != 0) std::memcpy(next, data_, size_ * sizeof(T));
        data_ = next;
        cap_ = want;
        charged_ += budget_ != nullptr ? bytes : 0;
        assert(size_ <= cap_);
        return true;
    }

    Arena*   arena_   = nullptr;
    Budget*  budget_  = nullptr;
    T*       data_    = nullptr;
    size_t   size_    = 0;
    size_t   cap_     = 0;
    uint64_t charged_ = 0;
};

template <typename T>
class ArenaRingBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ArenaRingBuffer<T> copies elements by value");

public:
    ArenaRingBuffer() noexcept = default;
    ArenaRingBuffer(const ArenaRingBuffer&) = delete;
    ArenaRingBuffer& operator=(const ArenaRingBuffer&) = delete;

    // Capacity is rounded up to a power of two. Charges budget (may be null).
    bool init(Arena* arena, Budget* budget, size_t capacity) noexcept {
        assert(arena != nullptr);
        assert(capacity > 0);
        size_t cap = 1;
        while (cap < capacity) {
            if (cap > (SIZE_MAX / sizeof(T)) / 2) {
                set_resource_exhausted("arena_ring_elements", capacity, cap);
                return false;
            }
            cap <<= 1;
        }
        const uint64_t bytes = static_cast<uint64_t>(cap) * sizeof(T);
        if (budget != nullptr && !budget->try_reserve(bytes)) return false;
        T* slots = static_cast<T*>(arena->allocate(static_cast<size_t>(bytes), alignof(T)));
        if (slots == nullptr) {
            if (budget != nullptr) budget->release(bytes);
            set_resource_exhausted("arena_max_blocks", bytes, 0);
            return false;
        }
        slots_ = slots;
        mask_ = cap - 1;
        head_ = tail_ = 0;
        budget_ = budget;
        charged_ = budget != nullptr ? bytes : 0;
        return true;
    }

    void destroy() noexcept {
        if (budget_ != nullptr && charged_ != 0) budget_->release(charged_);
        slots_ = nullptr;
        budget_ = nullptr;
        charged_ = 0;
        mask_ = 0;
        head_ = tail_ = 0;
    }

    bool try_push(const T& v) noexcept {
        assert(slots_ != nullptr);
        if (full()) return false;
        slots_[tail_ & mask_] = v;
        ++tail_;
        return true;
    }

    bool try_pop(T* out) noexcept {
        assert(slots_ != nullptr && out != nullptr);
        if (empty()) return false;
        *out = slots_[head_ & mask_];
        ++head_;
        return true;
    }

    T& front() noexcept { assert(!empty()); return slots_[head_ & mask_]; }
    size_t size() const noexcept { return static_cast<size_t>(tail_ - head_); }
    size_t capacity() const noexcept { return slots_ != nullptr ? mask_ + 1 : 0; }
    bool empty() const noexcept { return head_ == tail_; }
    bool full() const noexcept { return size() == capacity(); }

private:
    T*       slots_   = nullptr;
    Budget*  budget_  = nullptr;
    uint64_t charged_ = 0;
    size_t   mask_    = 0;
    uint64_t head_    = 0;
    uint64_t tail_    = 0;
};

}  // namespace bolt
