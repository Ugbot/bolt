// bolt_query_claim.h — operators share a query's memory limit before they
// use it.
//
// RULES: No exceptions. No heap. noexcept throughout.
//
// A spill ceiling, a presized table or a decode ring is sized when its
// operator starts, before any of it is charged. Sized from the limit alone,
// every such operator in a query can plan for nearly all of it and the query
// trips the limit before any of them spills. Each one instead claims a share
// of what is neither claimed nor used (query_claim_share_pct of it) from
// QueryCtx::mem_claimed, and releases it when it is done. With no query or no
// limit a claim grants what was asked and records nothing.
//
// One atomic, CAS-updated a bounded number of times; a claim that keeps
// losing the race is granted 0 and the operator falls back to its floor.
// Releases are idempotent and ignore a claim from an earlier query in a
// recycled slot.

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>

#include "bolt/bolt_budget.h"
#include "bolt/bolt_query_ctx.h"

namespace bolt {

inline constexpr uint32_t kQueryClaimRetries = 64;

struct QueryClaim {
    QueryCtx* q          = nullptr;   // null: nothing recorded
    uint32_t  generation = 0;         // q's generation when claimed
    uint32_t  _pad       = 0;
    uint64_t  bytes      = 0;         // released with atomic_ref exchange
};

/// Bytes of the running query's limit neither claimed nor used. ~0 with no
/// query or no limit.
inline uint64_t query_unclaimed_bytes() noexcept {
    const QueryCtx* q = current_query_ctx();
    if (q == nullptr || q->limits.memory_bytes == 0u) return ~0ull;
    const uint64_t claimed = q->mem_claimed.load(std::memory_order_relaxed);
    const uint64_t used    = q->mem_used.load(std::memory_order_relaxed);
    const uint64_t taken   = claimed > used ? claimed : used;
    return taken < q->limits.memory_bytes ? q->limits.memory_bytes - taken : 0u;
}

/// Draw up to `want` bytes from the running query's limit into `c` (which
/// holds no claim). Returns the bytes granted.
inline uint64_t query_claim(QueryClaim* c, uint64_t want) noexcept {
    assert(c != nullptr && c->q == nullptr && c->bytes == 0u);
    QueryCtx* q = current_query_ctx();
    if (q == nullptr || q->limits.memory_bytes == 0u) return want;
    const uint64_t pct = bolt_limits_value(bolt_limits_id::query_claim_share_pct);
    assert(pct >= 1u && pct <= 100u);
    const uint64_t limit = q->limits.memory_bytes;
    uint64_t cur = q->mem_claimed.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < kQueryClaimRetries; ++i) {
        const uint64_t used  = q->mem_used.load(std::memory_order_relaxed);
        const uint64_t taken = cur > used ? cur : used;
        const uint64_t room  = taken < limit ? limit - taken : 0u;
        uint64_t grant = room / 100u * pct + room % 100u * pct / 100u;
        if (grant > want) grant = want;
        if (q->mem_claimed.compare_exchange_weak(cur, cur + grant,
                                                 std::memory_order_relaxed)) {
            c->q          = q;
            c->generation = q->generation.load(std::memory_order_acquire);
            c->bytes      = grant;
            assert(grant <= want && grant <= limit);
            return grant;
        }
    }
    return 0u;
}

/// Give back all but `keep` bytes of `c` (an operator that claimed before it
/// knew its size).
inline void query_claim_trim(QueryClaim* c, uint64_t keep) noexcept {
    assert(c != nullptr);
    if (c->q == nullptr || c->bytes <= keep) return;
    const uint64_t give = c->bytes - keep;
    c->bytes = keep;
    if (c->q->generation.load(std::memory_order_acquire) != c->generation) return;
    uint64_t cur = c->q->mem_claimed.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < kQueryClaimRetries; ++i) {
        const uint64_t next = cur > give ? cur - give : 0u;
        if (c->q->mem_claimed.compare_exchange_weak(cur, next, std::memory_order_relaxed)) return;
    }
}

/// Give the whole claim back. Safe to call more than once, from any thread.
inline void query_release(QueryClaim* c) noexcept {
    assert(c != nullptr);
    if (c->q == nullptr) return;
    const uint64_t give = std::atomic_ref<uint64_t>(c->bytes).exchange(0u, std::memory_order_acq_rel);
    if (give == 0u || c->q->generation.load(std::memory_order_acquire) != c->generation) return;
    uint64_t cur = c->q->mem_claimed.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < kQueryClaimRetries; ++i) {
        const uint64_t next = cur > give ? cur - give : 0u;
        if (c->q->mem_claimed.compare_exchange_weak(cur, next, std::memory_order_relaxed)) return;
    }
}

}  // namespace bolt
