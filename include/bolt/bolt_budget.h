// bolt_budget.h — hierarchical memory budget.
//
// A Budget is a byte (or count) allowance with an optional parent: process ->
// query -> operator/scan. try_reserve charges every level up the chain and
// refuses, with ResourceExhausted naming the level's knob, if any level would
// exceed its limit. Growth of data-sized structures (kind C) charges a Budget
// at open/plan/batch boundaries — never inside a per-row loop.
//
// Lock-free: one fetch_add per level, undone on refusal. A reserve racing a
// refusal may see a transient overshoot of at most the concurrent requests,
// never a lasting one.
//
// process_budget() is the root. Its limit is the `mem_budget_mb` limit
// (BOLT_MEM_BUDGET_MB, legacy CHUKONU_MEM_BUDGET_MB); 0 = machine-relative:
// 40% of physical RAM clamped to [2 GiB, 64 GiB]. This is the budget chukonu's
// exec/mem_budget.h used to own.

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_config.h"
#include "bolt/bolt_limits.h"
#include "bolt/bolt_resource.h"
#include "bolt/bolt_system_memory.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_vector_limits.h"
#include "bolt/wire/bolt_wire_limits.h"

// ---------------------------------------------------------------------------
// bolt's own Limits table.
// ---------------------------------------------------------------------------
#define BOLT_LIMITS(X)                                                          \
    X(mem_budget_mb, kBudget, "MiB", 0, 0, (UINT64_MAX >> 20),                  \
      "BOLT_MEM_BUDGET_MB", "CHUKONU_MEM_BUDGET_MB",                            \
      "process memory budget; 0 = 40% of physical RAM clamped to [2, 64] GiB")  \
    X(max_workers, kInvariant, "threads", BOLT_MAX_WORKERS, BOLT_MAX_WORKERS,   \
      BOLT_MAX_WORKERS, nullptr, nullptr,                                       \
      "scheduler worker validation ceiling (BOLT_MAX_WORKERS at build time)")   \
    X(workers, kFixedAtAlloc, "threads", 0, 0, BOLT_MAX_WORKERS, "BOLT_WORKERS", \
      nullptr, "workers of an auto-sized bolt::Scheduler; 0 = hardware threads") \
    X(arena_max_blocks, kInvariant, "blocks", 256, 256, 256, nullptr, nullptr,  \
      "backing blocks per bolt::Arena")                                         \
    X(stack_array_lint_bytes, kInvariant, "bytes", 16384, 16384, 16384,         \
      nullptr, nullptr, "largest function-scope array the stack lint allows")    \
    X(lake_metadata_budget_mb, kBudget, "MiB", 1024, 1, (UINT64_MAX >> 20),     \
      "BOLT_LAKE_METADATA_BUDGET_MB", nullptr,                                  \
      "Iceberg/Delta metadata held by one table handle or one scan")          \
    X(max_vector_dim, kInvariant, "dims", 65535, 65535, 65535, nullptr,         \
      nullptr, "vector dimension validation ceiling (bolt::kMaxVectorDim)")      \
    X(max_columns, kInvariant, "columns", 32768, 32768, 32768, nullptr,         \
      nullptr, "column-count validation ceiling (bolt::kMaxColumns)")           \
    X(gb_hash_chain_max, kInvariant, "links", 64, 64, 64, nullptr, nullptr,     \
      "group-by hash table: bounded probe chain a colliding hash follows "     \
      "before the query is refused (kGbHashChainMax)")                         \
    X(wire_buf_align, kInvariant, "bytes", BOLT_WIRE_BUF_ALIGN,                 \
      BOLT_WIRE_BUF_ALIGN, BOLT_WIRE_BUF_ALIGN, nullptr, nullptr,              \
      "frame / buffer alignment this build writes (recorded per file)")        \
    X(wire_chunk_align, kInvariant, "bytes", BOLT_WIRE_CHUNK_ALIGN,             \
      BOLT_WIRE_CHUNK_ALIGN, BOLT_WIRE_CHUNK_ALIGN, nullptr, nullptr,          \
      "frame-file index/footer alignment this build writes (recorded per file)") \
    X(wire_stripe_align, kInvariant, "bytes", BOLT_WIRE_STRIPE_ALIGN,           \
      BOLT_WIRE_STRIPE_ALIGN, BOLT_WIRE_STRIPE_ALIGN, nullptr, nullptr,        \
      "segment data-region alignment this build writes (recorded per file)")   \
    X(wire_io_align, kInvariant, "bytes", BOLT_WIRE_IO_ALIGN,                   \
      BOLT_WIRE_IO_ALIGN, BOLT_WIRE_IO_ALIGN, nullptr, nullptr,                \
      "writer I/O unit this build records (informational)")

namespace bolt {

BOLT_LIMITS_TABLE(bolt_limits, "bolt", BOLT_LIMITS)

static_assert(kArenaMaxBlocks == 256, "update BOLT_LIMITS arena_max_blocks");
static_assert(kMaxVectorDim == 65535u, "update BOLT_LIMITS max_vector_dim");
static_assert(kMaxColumns == 32768u, "update BOLT_LIMITS max_columns");

// The worker count an auto-sized pool uses: BOLT_WORKERS when set, else the
// hardware thread count, capped at the ceiling with a stderr line (the
// machine chose that number, not the caller). 0 when BOLT_WORKERS is invalid;
// the caller refuses to start and bolt_limits().error says why.
inline uint32_t bolt_auto_workers() noexcept {
    static_assert(BOLT_MAX_WORKERS >= 1u, "worker ceiling");
    const LimitTable& t = bolt_limits();
    const uint32_t wi = static_cast<uint32_t>(bolt_limits_id::workers);
    if (limits_source(t, wi) == LimitSource::kInvalid) return 0;
    uint64_t w = limits_value(t, wi);
    if (w != 0) {
        assert(w <= BOLT_MAX_WORKERS);
        return static_cast<uint32_t>(w);
    }
    w = bolt_get_hardware_concurrency();
    if (w == 0) w = 1;
    if (w > BOLT_MAX_WORKERS) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr, "bolt: %llu hardware threads, using the "
                         "BOLT_MAX_WORKERS ceiling %u\n",
                         static_cast<unsigned long long>(w), BOLT_MAX_WORKERS);
        }
        w = BOLT_MAX_WORKERS;
    }
    assert(w >= 1u && w <= BOLT_MAX_WORKERS);
    return static_cast<uint32_t>(w);
}

inline constexpr uint32_t kBudgetMaxDepth = 16;

// Query-failure fallback and the machine-relative clamp (unchanged from
// chukonu's mem_budget.h).
inline constexpr uint64_t kDefaultMemBudgetBytes = 8ull << 30;
inline constexpr uint64_t kMemBudgetFloorBytes   = 2ull << 30;
inline constexpr uint64_t kMemBudgetCapBytes     = 64ull << 30;

class alignas(64) Budget {
public:
    Budget(const char* knob, uint64_t limit, Budget* parent = nullptr) noexcept
        : used_(0), limit_(limit), peak_(0), knob_(knob), parent_(parent),
          depth_(parent != nullptr ? parent->depth_ + 1 : 0) {
        assert(knob != nullptr);
        assert(depth_ < kBudgetMaxDepth);
    }

    Budget(const Budget&) = delete;
    Budget& operator=(const Budget&) = delete;

    // Charge n at this level and every ancestor. On refusal nothing stays
    // charged and ResourceExhausted names the refusing level.
    bool try_reserve(uint64_t n) noexcept {
        Budget* charged[kBudgetMaxDepth];
        uint32_t k = 0;
        for (Budget* b = this; b != nullptr; b = b->parent_) {
            assert(k < kBudgetMaxDepth);
            if (!b->reserve_one(n)) {
                for (uint32_t j = 0; j < k; ++j) charged[j]->release_one(n);
                return false;
            }
            charged[k++] = b;
        }
        assert(k == depth_ + 1);
        return true;
    }

    void release(uint64_t n) noexcept {
        for (Budget* b = this; b != nullptr; b = b->parent_) b->release_one(n);
    }

    uint64_t used() const noexcept { return used_.load(std::memory_order_relaxed); }
    uint64_t limit() const noexcept { return limit_.load(std::memory_order_relaxed); }
    uint64_t peak() const noexcept { return peak_.load(std::memory_order_relaxed); }
    uint64_t available() const noexcept {
        const uint64_t u = used(), l = limit();
        return u >= l ? 0 : l - u;
    }
    const char* knob() const noexcept { return knob_; }
    Budget* parent() const noexcept { return parent_; }
    void set_limit(uint64_t l) noexcept { limit_.store(l, std::memory_order_relaxed); }

private:
    bool reserve_one(uint64_t n) noexcept {
        const uint64_t lim = limit();
        const uint64_t prev = used_.fetch_add(n, std::memory_order_acq_rel);
        const bool overflow = prev > UINT64_MAX - n;
        if (overflow || prev + n > lim) {
            used_.fetch_sub(n, std::memory_order_acq_rel);
            set_resource_exhausted(knob_, n, lim, prev);
            return false;
        }
        note_peak(prev + n);
        return true;
    }

    void release_one(uint64_t n) noexcept {
        const uint64_t prev = used_.fetch_sub(n, std::memory_order_acq_rel);
        assert(prev >= n && "Budget released more than it reserved");
        (void)prev;
    }

    void note_peak(uint64_t v) noexcept {
        uint64_t p = peak_.load(std::memory_order_relaxed);
        for (uint32_t i = 0; i < 64 && v > p; ++i) {
            if (peak_.compare_exchange_weak(p, v, std::memory_order_relaxed)) return;
        }
    }

    std::atomic<uint64_t> used_;
    std::atomic<uint64_t> limit_;
    std::atomic<uint64_t> peak_;
    const char*           knob_;
    Budget*               parent_;
    uint32_t              depth_;
};

// 40% of physical RAM clamped to [2 GiB, 64 GiB]; 8 GiB if RAM is unknown.
inline uint64_t machine_mem_budget_bytes() noexcept {
    static const uint64_t b = [] {
        uint64_t ram = 0;
        if (!query_total_physical_ram_bytes(&ram) || ram == 0) {
            return kDefaultMemBudgetBytes;
        }
        uint64_t budget = (ram / 5ull) * 2ull;
        if (budget < kMemBudgetFloorBytes) budget = kMemBudgetFloorBytes;
        if (budget > kMemBudgetCapBytes) budget = kMemBudgetCapBytes;
        return budget;
    }();
    return b;
}

// The limit `mem_budget_mb` resolves to, in bytes.
inline uint64_t configured_mem_budget_bytes() noexcept {
    const uint64_t mb = bolt_limits_value(bolt_limits_id::mem_budget_mb);
    return mb != 0 ? (mb << 20) : machine_mem_budget_bytes();
}

inline Budget& process_budget() noexcept {
    static Budget b("mem_budget_mb", configured_mem_budget_bytes());
    return b;
}

}  // namespace bolt
