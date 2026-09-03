// bolt_arena.h — Per-thread bump allocator with epoch-based lifetime
//
// RULES: No exceptions. No RTTI. No smart pointers. No std::vector in hot path.
// Pre-allocated block table. Returns nullptr on OOM (caller checks).
// Effectively C17 with namespaces, templates, and constexpr.
//
// Measured: ~3ns/alloc vs ~25,000ns for malloc+free (9,600x faster)
// Measured: reset() ~5ns

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {

struct ArenaConfig {
    size_t initial_block_size = 4 * 1024 * 1024;   // 4MB
    size_t max_block_size     = 64 * 1024 * 1024;  // 64MB
    size_t alignment          = 64;                 // Cache-line
    bool   poison_on_reset    = false;              // Debug: 0xDE fill
};

/// Maximum number of backing blocks an arena can hold.
/// NOTE: doubling is CAPPED at config.max_block_size, so total capacity is
/// roughly 32 x max_block_size (default 64 MB => ~1.8 GB), NOT the "128 GB"
/// an uncapped doubling would suggest. Long-lived arenas that back operator
/// state (the scheduler's worker arenas) pass a larger max_block_size.
static constexpr uint32_t kArenaMaxBlocks = 32;

/// Arena: Bump allocator. One per thread. Reset per morsel epoch.
///
/// No exceptions — returns nullptr on OOM.
/// No std::vector — fixed-size block table.
/// No smart pointers — raw malloc/free in ctor/dtor only.
class Arena {
public:
    explicit Arena(ArenaConfig config = {}) noexcept
        : config_(config)
        , num_blocks_(0)
        , current_idx_(0)
        , cursor_(0)
        , end_(0)
        , total_allocated_(0)
        , peak_usage_(0) {
        memset(blocks_, 0, sizeof(blocks_));
        memset(block_sizes_, 0, sizeof(block_sizes_));
        grow(config_.initial_block_size);
    }

    ~Arena() noexcept {
        for (uint32_t i = 0; i < num_blocks_; ++i) {
            aligned_free(blocks_[i]);
        }
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    /// Opt-in shared-across-threads mode. An Arena is single-thread by design
    /// ("Arena per Slot"); the bump cursor is a plain uintptr_t. When more than
    /// one thread legitimately shares ONE arena — e.g. chukonu's parallel
    /// driver, which runs several operators concurrently but hands them all the
    /// same fragment arena — concurrent allocate() calls race the cursor and
    /// return torn / overlapping blocks (a wild BoltColumn pointer, then an
    /// EXC_BAD_ACCESS in a downstream memcpy). Enabling this serializes the
    /// allocate() body with a spinlock. OFF by default, so every per-slot arena
    /// keeps the lock-free ~3ns fast path (one predictable-false branch). The
    /// caller MUST guarantee no allocate() is in flight when it flips the flag.
    void set_concurrent(bool on) noexcept { concurrent_ = on; }
    bool concurrent() const noexcept { return concurrent_; }

    /// Bump-allocate. Returns nullptr if OOM or block table full.
    /// Cost: ~3ns fast path (pointer arithmetic + branch).
    void* allocate(size_t size, size_t alignment = 0) noexcept {
        if (!concurrent_) return allocate_unlocked(size, alignment);
        // Serialize concurrent sharers of ONE arena (see set_concurrent).
        // Short critical section (a bump, occasionally a grow), so a bare
        // test-and-set spin is the right primitive — no OS wait, no fairness
        // needed; contention is per-buffer, not per-row.
        while (alloc_lock_.test_and_set(std::memory_order_acquire)) {
            BOLT_PAUSE();
        }
        void* p = allocate_unlocked(size, alignment);
        alloc_lock_.clear(std::memory_order_release);
        return p;
    }

    /// Typed allocation. Returns nullptr on failure.
    template <typename T>
    T* allocate_array(size_t count) noexcept {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    /// Allocate and zero-fill.
    void* allocate_zeroed(size_t size, size_t alignment = 0) noexcept {
        void* p = allocate(size, alignment);
        if (p) memset(p, 0, size);
        return p;
    }

    /// Reset all allocations. Blocks stay allocated for reuse.
    /// Cost: ~5ns (pointer reset + atomic-free counter zero).
    void reset() noexcept {
        if (config_.poison_on_reset) {
            for (uint32_t i = 0; i < num_blocks_; ++i) {
                size_t used = (i < current_idx_) ? block_sizes_[i]
                    : (i == current_idx_) ? (cursor_ - reinterpret_cast<uintptr_t>(blocks_[i]))
                    : 0;
                if (used > 0) memset(blocks_[i], 0xDE, used);
            }
        }
        if (num_blocks_ > 0) {
            current_idx_ = 0;
            cursor_ = reinterpret_cast<uintptr_t>(blocks_[0]);
            end_ = cursor_ + block_sizes_[0];
        }
        // New epoch: blocks the oversize lane consumed become plain empty
        // blocks again, visible to the normal lane's forward scan.
        oversize_used_ = 0;
        total_allocated_ = 0;
    }

    /// Release all blocks except the first.
    void compact() noexcept {
        for (uint32_t i = 1; i < num_blocks_; ++i) {
            aligned_free(blocks_[i]);
            blocks_[i] = nullptr;
            block_sizes_[i] = 0;
        }
        if (num_blocks_ > 1) num_blocks_ = 1;
        if (num_blocks_ > 0) {
            current_idx_ = 0;
            cursor_ = reinterpret_cast<uintptr_t>(blocks_[0]);
            end_ = cursor_ + block_sizes_[0];
        }
        oversize_used_ = 0;
        total_allocated_ = 0;
    }

    /// reset() plus block-table trimming: free tail blocks until at most
    /// `keep_bytes` stays reserved (block 0 always survives). The warm pool's
    /// worker arenas live for the PROCESS, and their 32-slot block table only
    /// ever grew — a heterogeneous query mix ratchets it to the cap within a
    /// few heavy queries, after which allocations fail (surfacing as bogus
    /// downstream errors) and, before that, throughput quietly degrades as
    /// most of the reserved bytes sit in stranded blocks. Trimming at the
    /// between-queries reset point keeps a bounded warm working set AND
    /// returns table slots. Same no-tasks-in-flight contract as reset().
    void reset_keep(size_t keep_bytes) noexcept {
        assert(num_blocks_ >= 1);
        size_t reserved = total_reserved();
        while (num_blocks_ > 1 && reserved > keep_bytes) {
            num_blocks_--;
            reserved -= block_sizes_[num_blocks_];
            aligned_free(blocks_[num_blocks_]);
            blocks_[num_blocks_] = nullptr;
            block_sizes_[num_blocks_] = 0;
        }
        assert(num_blocks_ >= 1);
        reset();
    }

    /// Copy data into arena.
    void* copy_into(const void* src, size_t size, size_t alignment = 0) noexcept {
        void* dst = allocate(size, alignment);
        if (dst && src) memcpy(dst, src, size);
        return dst;
    }

    size_t total_allocated() const noexcept { return total_allocated_; }
    size_t peak_usage()      const noexcept { return peak_usage_; }
    uint32_t num_blocks()    const noexcept { return num_blocks_; }

    size_t total_reserved() const noexcept {
        size_t t = 0;
        for (uint32_t i = 0; i < num_blocks_; ++i) t += block_sizes_[i];
        return t;
    }

private:
    static void* aligned_alloc_impl(size_t alignment, size_t size) noexcept {
        return bolt_aligned_alloc(alignment, size);
    }

    static void aligned_free(void* p) noexcept {
        bolt_aligned_free(p);
    }

    bool grow(size_t min_size) noexcept {
        if (num_blocks_ >= kArenaMaxBlocks) return false;

        size_t sz = config_.initial_block_size;
        if (num_blocks_ > 0) {
            sz = block_sizes_[num_blocks_ - 1] * 2;
            if (sz > config_.max_block_size) sz = config_.max_block_size;
        }
        if (sz < min_size) sz = min_size;

        void* block = aligned_alloc_impl(config_.alignment, sz);
        if (!block) return false;

        blocks_[num_blocks_] = block;
        block_sizes_[num_blocks_] = sz;
        current_idx_ = num_blocks_;
        num_blocks_++;
        cursor_ = reinterpret_cast<uintptr_t>(block);
        end_ = cursor_ + sz;
        return true;
    }

    // The original lock-free bump body. Called directly on the fast path
    // (concurrent_ == false) and under alloc_lock_ when shared.
    void* allocate_unlocked(size_t size, size_t alignment) noexcept {
        if (alignment == 0) alignment = config_.alignment;

        uintptr_t aligned = (cursor_ + alignment - 1) & ~(alignment - 1);

        if (aligned + size <= end_) {
            cursor_ = aligned + size;
            total_allocated_ += size;
            if (total_allocated_ > peak_usage_) peak_usage_ = total_allocated_;
            return reinterpret_cast<void*>(aligned);
        }
        return allocate_slow(size, alignment);
    }

    void* allocate_slow(size_t size, size_t alignment) noexcept {
        assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
        assert(size > 0);
        // Oversize lane. A request bigger than the steady-state block size gets
        // a bespoke block WITHOUT moving the bump cursor. The old code fell
        // through to grow(), which jumps current_idx_ to the tail — a single
        // 128 MB request at block 11 of 30 permanently stranded blocks 12..29
        // for the rest of the epoch (measured on a warm-pool worker arena:
        // 2.69 GB reserved, ~600 MB reachable, then a 208 KB allocation FAILED
        // at the 32-block table cap). The bespoke block is consumed whole by
        // this request, so there is nothing in it for the cursor to use anyway.
        if (size + alignment > config_.max_block_size) {
            // Reuse first: a bespoke block minted by an earlier epoch is a
            // plain empty block after reset(), and re-malloc'ing 16-128 MB per
            // epoch is both slow and exactly the table ratchet this lane
            // exists to prevent. Any not-yet-used block past the cursor that
            // fits is taken whole.
            for (uint32_t i = current_idx_ + 1; i < num_blocks_; ++i) {
                if ((oversize_used_ & (1u << i)) != 0u) continue;
                const uintptr_t base = reinterpret_cast<uintptr_t>(blocks_[i]);
                const uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
                if (aligned + size <= base + block_sizes_[i]) {
                    oversize_used_ |= (1u << i);
                    total_allocated_ += size;
                    if (total_allocated_ > peak_usage_) peak_usage_ = total_allocated_;
                    return reinterpret_cast<void*>(aligned);
                }
            }
            if (num_blocks_ >= kArenaMaxBlocks) return nullptr;
            const size_t sz = size + alignment;
            void* block = aligned_alloc_impl(config_.alignment, sz);
            if (block == nullptr) return nullptr;
            blocks_[num_blocks_] = block;
            block_sizes_[num_blocks_] = sz;
            // The bespoke block sits PAST the cursor while fully in use, so it
            // must be excluded from the forward scan below for the rest of
            // this epoch — otherwise its memory would be handed out twice.
            oversize_used_ |= (1u << num_blocks_);
            num_blocks_++;
            const uintptr_t base = reinterpret_cast<uintptr_t>(block);
            const uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
            assert(aligned + size <= base + sz);
            total_allocated_ += size;
            if (total_allocated_ > peak_usage_) peak_usage_ = total_allocated_;
            return reinterpret_cast<void*>(aligned);
        }
        // Normal lane: scan FORWARD for the first pre-existing block that fits.
        // Every block past current_idx_ is untouched this epoch, so skipping to
        // the first fit reuses capacity the old probe-exactly-one code (and its
        // grow() fallback) abandoned. Blocks skipped over are smaller than this
        // request — bounded early-progression blocks, not the big tail ones.
        for (uint32_t i = current_idx_ + 1; i < num_blocks_; ++i) {
            if ((oversize_used_ & (1u << i)) != 0u) continue;   // in use this epoch
            const uintptr_t base = reinterpret_cast<uintptr_t>(blocks_[i]);
            const uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
            if (aligned + size <= base + block_sizes_[i]) {
                current_idx_ = i;
                end_ = base + block_sizes_[i];
                cursor_ = aligned + size;
                total_allocated_ += size;
                if (total_allocated_ > peak_usage_) peak_usage_ = total_allocated_;
                return reinterpret_cast<void*>(aligned);
            }
        }
        // Grow
        if (!grow(size + alignment)) return nullptr;
        uintptr_t aligned = (cursor_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > end_) return nullptr;  // Shouldn't happen
        cursor_ = aligned + size;
        total_allocated_ += size;
        if (total_allocated_ > peak_usage_) peak_usage_ = total_allocated_;
        return reinterpret_cast<void*>(aligned);
    }

    ArenaConfig config_;
    void*    blocks_[kArenaMaxBlocks];
    size_t   block_sizes_[kArenaMaxBlocks];
    uint32_t num_blocks_;
    uint32_t current_idx_;
    uintptr_t cursor_;
    uintptr_t end_;
    size_t   total_allocated_;
    size_t   peak_usage_;
    // Bitmask over blocks_ (kArenaMaxBlocks == 32 fits uint32_t exactly):
    // blocks consumed whole by the oversize lane THIS epoch, so the normal
    // lane's forward scan must not reuse them until the next reset().
    uint32_t oversize_used_ = 0;

    // Opt-in shared-arena support (see set_concurrent / allocate). concurrent_
    // is set once before any sharing thread runs and cleared after they join,
    // so it needs no atomicity itself; alloc_lock_ serializes the bump.
    bool concurrent_ = false;
    std::atomic_flag alloc_lock_{};   // C++20: value-init is the clear state
};
static_assert(kArenaMaxBlocks <= 32, "oversize_used_ bitmask is 32 bits");

// Thread-local arena pointer. Set by slot, reset at morsel boundary.
inline thread_local Arena* tl_arena = nullptr;

inline Arena& current_arena() noexcept {
    assert(tl_arena != nullptr);
    return *tl_arena;
}

/// RAII guard. No heap. No exceptions.
///
/// Contract: swaps `tl_arena` to the supplied arena for the guard's lifetime
/// and restores the previous value on destruction. The pointer-accepting
/// overload treats `nullptr` as "no-op": `tl_arena` is left untouched, and
/// the dtor still restores `prev_` (which equals the entry value of
/// `tl_arena`). This lets call sites pass an optional arena without branching.
class ArenaGuard {
public:
    explicit ArenaGuard(Arena& arena) noexcept : prev_(tl_arena) { tl_arena = &arena; }

    // Pointer overload — tolerates nullptr (leaves tl_arena unchanged).
    // Not delegated to the reference ctor because *nullptr is UB.
    explicit ArenaGuard(Arena* arena) noexcept : prev_(tl_arena) {
        assert(prev_ == tl_arena);       // Precondition: captured entry value.
        if (arena != nullptr) tl_arena = arena;
        assert(arena == nullptr || tl_arena == arena);  // Postcondition.
    }

    ~ArenaGuard() noexcept { tl_arena = prev_; }
    ArenaGuard(const ArenaGuard&) = delete;
    ArenaGuard& operator=(const ArenaGuard&) = delete;
private:
    Arena* prev_;
};

}  // namespace bolt
