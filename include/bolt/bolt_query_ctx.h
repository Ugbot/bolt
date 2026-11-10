// bolt_query_ctx.h — the running query, as every layer below the frontends
// sees it: why it must stop, its memory charge and its parallelism.
//
// RULES: No exceptions. No heap. noexcept throughout.
//
// One QueryCtx per running query, in a fixed static table, published on a
// thread by QueryCtxScope. Scheduler tasks carry the submitter's context to
// the worker that runs them, so a kernel, a scan or a spill loop anywhere under
// a query reads it with no parameter threaded through.
//
// query_should_stop() is the safe-point check: one thread-local load and one
// relaxed load of a flag that stays unset, so the branch predicts. Every 64th
// check on a thread also looks at the cancel flag and the clock. Coarse sites
// (per morsel, per file, per I/O) use query_poll_now(), which always does.
//
// Memory is charged where an Arena gets or frees a block, never per
// allocation. Crossing the limit stops the query at its next safe point. The
// allocation still succeeds: code under a spill or a merge treats a failed
// allocation as a bug, and a limit must end the query, not the process.
// Slots are recycled by generation: an arena that outlives its query (worker
// arenas do) finds the generation changed and stops crediting.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {

enum class QueryStop : uint32_t {
    None      = 0,
    Cancelled = 1,
    Deadline  = 2,
    Memory    = 3,
    Rows      = 4,
};

/// What a query asks of the engine. Zero means "not set" in every field.
struct QueryLimits {
    uint64_t deadline_steady_ns = 0;   // absolute steady_clock ns; 0 = none
    uint64_t memory_bytes       = 0;   // 0 = none
    uint64_t row_limit          = 0;   // result rows; 0 = none
    int64_t  morsel_rows        = 0;   // source emit size; 0 = the mode's
    uint32_t workers            = 0;   // 0 = the pool; 1 = serial
    uint32_t _pad               = 0;
};

inline uint64_t query_steady_now_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline constexpr uint32_t kQueryFailureBytes = 384;   // why an operator failed, in words

struct alignas(64) QueryCtx {
    std::atomic<uint32_t>            stop{0};
    std::atomic<uint32_t>            generation{0};   // odd while in use
    const std::atomic<uint32_t>*     cancel_flag = nullptr;
    QueryLimits                      limits{};
    alignas(64) std::atomic<uint64_t> mem_used{0};
    std::atomic<uint64_t>            mem_peak{0};
    std::atomic<uint64_t>            rows{0};
    // The first operator failure's text, from whichever worker thread ran it:
    // 0 empty, 1 being written, 2 published. Read on the submitting thread.
    std::atomic<uint32_t>            failure_state{0};
    char                             failure[kQueryFailureBytes] = {};

    // First writer wins; later failures (usually consequences) are dropped.
    void note_failure(const char* text) noexcept {
        if (text == nullptr || text[0] == '\0') return;
        uint32_t empty = 0;
        if (!failure_state.compare_exchange_strong(empty, 1u, std::memory_order_acquire)) return;
        size_t n = std::strlen(text);
        if (n >= kQueryFailureBytes) n = kQueryFailureBytes - 1u;
        std::memcpy(failure, text, n);
        failure[n] = '\0';
        failure_state.store(2u, std::memory_order_release);
    }
    // Null until a failure is published.
    const char* failure_detail() const noexcept {
        return failure_state.load(std::memory_order_acquire) == 2u ? failure : nullptr;
    }

    // First reason wins, so the error names what actually stopped the query.
    void request_stop(QueryStop why) noexcept {
        uint32_t none = 0;
        stop.compare_exchange_strong(none, static_cast<uint32_t>(why), std::memory_order_relaxed);
    }
    QueryStop stopped() const noexcept {
        return static_cast<QueryStop>(stop.load(std::memory_order_relaxed));
    }
    // Reads the cancel flag and the clock.
    bool poll_now() noexcept {
        if (stop.load(std::memory_order_relaxed) != 0u) return true;
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed) != 0u) {
            request_stop(QueryStop::Cancelled);
            return true;
        }
        if (limits.deadline_steady_ns != 0u && query_steady_now_ns() >= limits.deadline_steady_ns) {
            request_stop(QueryStop::Deadline);
            return true;
        }
        return false;
    }
    void charge(uint64_t bytes) noexcept {
        const uint64_t used = mem_used.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        if (limits.memory_bytes != 0u && used > limits.memory_bytes) request_stop(QueryStop::Memory);
        uint64_t p = mem_peak.load(std::memory_order_relaxed);
        while (used > p && !mem_peak.compare_exchange_weak(p, used, std::memory_order_relaxed)) {}
    }
    void credit(uint64_t bytes) noexcept {
        uint64_t u = mem_used.load(std::memory_order_relaxed);
        const uint64_t n = bytes < u ? bytes : u;
        mem_used.fetch_sub(n, std::memory_order_relaxed);
    }
    // Result rows produced; past row_limit the query stops. False when it has.
    bool add_rows(uint64_t n) noexcept {
        const uint64_t total = rows.fetch_add(n, std::memory_order_relaxed) + n;
        if (limits.row_limit == 0u || total <= limits.row_limit) return true;
        request_stop(QueryStop::Rows);
        return false;
    }
};

inline constexpr uint32_t kQueryCtxSlots = 4096;
inline constexpr uint32_t kQueryCtxProbes = 256;
inline constexpr uint32_t kQueryPollEvery = 64;   // checks between clock reads

namespace detail {
inline QueryCtx                g_query_ctx[kQueryCtxSlots];
inline std::atomic<uint32_t>   g_query_ctx_next{0};
inline thread_local QueryCtx*  tl_query_ctx = nullptr;
inline thread_local uint32_t   tl_query_ticks = 0;
}  // namespace detail

inline QueryCtx* current_query_ctx() noexcept { return detail::tl_query_ctx; }

/// Install `q` on this thread; returns what was there. Null clears.
inline QueryCtx* set_current_query_ctx(QueryCtx* q) noexcept {
    QueryCtx* prev = detail::tl_query_ctx;
    detail::tl_query_ctx = q;
    return prev;
}

/// Claim a slot. Null when every probed slot is in use.
inline QueryCtx* query_ctx_acquire(const QueryLimits& limits,
                                   const std::atomic<uint32_t>* cancel_flag) noexcept {
    const uint32_t start = detail::g_query_ctx_next.fetch_add(1u, std::memory_order_relaxed);
    for (uint32_t i = 0; i < kQueryCtxProbes; ++i) {
        QueryCtx* q = &detail::g_query_ctx[(start + i) % kQueryCtxSlots];
        uint32_t g = q->generation.load(std::memory_order_relaxed);
        if ((g & 1u) != 0u ||
            !q->generation.compare_exchange_strong(g, g + 1u, std::memory_order_acq_rel)) continue;
        q->stop.store(0u, std::memory_order_relaxed);
        q->cancel_flag = cancel_flag;
        q->limits = limits;
        q->mem_used.store(0u, std::memory_order_relaxed);
        q->mem_peak.store(0u, std::memory_order_relaxed);
        q->rows.store(0u, std::memory_order_relaxed);
        q->failure[0] = '\0';
        q->failure_state.store(0u, std::memory_order_release);
        return q;
    }
    return nullptr;
}

inline void query_ctx_release(QueryCtx* q) noexcept {
    q->cancel_flag = nullptr;
    q->generation.fetch_add(1u, std::memory_order_release);
}

/// The safe-point check for inner loops. True means leave now.
BOLT_FORCE_INLINE bool query_should_stop() noexcept {
    QueryCtx* q = detail::tl_query_ctx;
    if (q == nullptr) return false;
    if (q->stop.load(std::memory_order_relaxed) != 0u) return true;
    if (++detail::tl_query_ticks < kQueryPollEvery) return false;
    detail::tl_query_ticks = 0;
    return q->poll_now();
}

/// The check for coarse safe points: per morsel, per file, around blocking I/O.
inline bool query_poll_now() noexcept {
    QueryCtx* q = detail::tl_query_ctx;
    return q != nullptr && q->poll_now();
}

/// Record why the running query fails (an operator's own words), from any
/// thread that runs part of it. No-op outside a query.
inline void query_note_failure(const char* text) noexcept {
    QueryCtx* q = detail::tl_query_ctx;
    if (q != nullptr) q->note_failure(text);
}

/// Time left for blocking I/O, in ms: `fallback_ms` when the query has no
/// deadline or there is no query, 0 when it has passed.
inline uint32_t query_remaining_ms(uint32_t fallback_ms) noexcept {
    QueryCtx* q = detail::tl_query_ctx;
    if (q == nullptr || q->limits.deadline_steady_ns == 0u) return fallback_ms;
    const uint64_t now = query_steady_now_ns();
    if (now >= q->limits.deadline_steady_ns) return 0u;
    const uint64_t ms = (q->limits.deadline_steady_ns - now) / 1000000u + 1u;
    return ms < fallback_ms ? static_cast<uint32_t>(ms) : fallback_ms;
}

/// Publishes `q` on this thread for a scope. Tasks submitted inside inherit it.
class QueryCtxScope {
public:
    explicit QueryCtxScope(QueryCtx* q) noexcept : prev_(set_current_query_ctx(q)) {}
    ~QueryCtxScope() { detail::tl_query_ctx = prev_; }
    QueryCtxScope(const QueryCtxScope&) = delete;
    QueryCtxScope& operator=(const QueryCtxScope&) = delete;
private:
    QueryCtx* prev_;
};

}  // namespace bolt
