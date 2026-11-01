// bolt_mem_owner.h — who holds arena memory: per-owner reserved-byte counters.
//
// RULES: No exceptions. No RTTI. No heap. noexcept throughout.
//
// An Arena charges the MemOwner named in its ArenaConfig (and always the
// process-wide total) for every backing block it holds: + on grow, - on free
// and on destruction. The counters move only on block grow/free, never per
// allocation, so the bump fast path is untouched.
//
// Long-lived owners that are not per-table (worker pools, scratch arenas)
// register once under a static name; /metrics walks the registry. Per-table
// owners (MarbleDB memtables) live in their table's runtime and are read
// through that library's own stats call instead.
//
// `reserved` is backing bytes (virtual size of the blocks). Pages of a mmap'd
// block count toward the footprint only once touched, so reserved is an upper
// bound on the dirty bytes an owner can hold.

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_config.h"
#include "bolt/bolt_port_limits.h"

namespace bolt {

struct MemOwner {
    std::atomic<std::uint64_t> reserved{0};   // backing bytes held now
    std::atomic<std::uint64_t> blocks{0};     // backing blocks held now
    std::atomic<std::uint64_t> arenas{0};     // live arenas charging this owner

    void charge(std::size_t bytes) noexcept {
        reserved.fetch_add(bytes, std::memory_order_relaxed);
        blocks.fetch_add(1u, std::memory_order_relaxed);
    }
    void release(std::size_t bytes) noexcept {
        const std::uint64_t prev = reserved.fetch_sub(bytes, std::memory_order_relaxed);
        assert(prev >= bytes);
        (void)prev;
        const std::uint64_t pb = blocks.fetch_sub(1u, std::memory_order_relaxed);
        assert(pb >= 1u);
        (void)pb;
    }
};

// Every bolt::Arena in the process, whatever its own owner. Charged on block
// grow/free only (never on arena construction), and on its own cache line so
// those updates do not false-share with neighbouring globals. Its `arenas`
// field stays 0: a process-wide live-arena count would put a shared atomic on
// every short-lived arena's construct/destroy path (R0 review).
struct alignas(config::kCacheIsolationBytes) PaddedMemOwner : MemOwner {};
inline PaddedMemOwner g_arena_total_owner{};

inline constexpr std::uint32_t kMemOwnerRegistryCap = kBoltMemOwnerRegistryCapacity;

struct MemOwnerEntry {
    const char*     name;    // static storage; never freed
    const MemOwner* owner;
};

namespace detail {
inline MemOwnerEntry           g_mem_owner_entries[kMemOwnerRegistryCap] = {};
inline std::atomic<std::uint32_t> g_mem_owner_claimed{0};
inline std::atomic<std::uint32_t> g_mem_owner_published{0};
}  // namespace detail

// Register a process-lifetime owner under a static name. False when the
// registry is full (the owner still counts; it is just not listed).
inline bool mem_owner_register(const char* name, const MemOwner* owner) noexcept {
    assert(name != nullptr && owner != nullptr);
    const std::uint32_t i =
        detail::g_mem_owner_claimed.fetch_add(1u, std::memory_order_acq_rel);
    if (i >= kMemOwnerRegistryCap) return false;
    detail::g_mem_owner_entries[i] = MemOwnerEntry{name, owner};
    // Publish in claim order so a reader never sees a half-written entry.
    std::uint32_t expect = i;
    while (!detail::g_mem_owner_published.compare_exchange_weak(
               expect, i + 1u, std::memory_order_release, std::memory_order_relaxed)) {
        expect = i;   // bounded by concurrent registrations (startup only)
    }
    return true;
}

// Copy up to `cap` registered owners into `out`; returns how many.
inline std::uint32_t mem_owner_list(MemOwnerEntry* out, std::uint32_t cap) noexcept {
    assert(out != nullptr || cap == 0u);
    std::uint32_t n = detail::g_mem_owner_published.load(std::memory_order_acquire);
    if (n > kMemOwnerRegistryCap) n = kMemOwnerRegistryCap;
    if (n > cap) n = cap;
    for (std::uint32_t i = 0; i < n; ++i) out[i] = detail::g_mem_owner_entries[i];
    return n;
}

}  // namespace bolt
