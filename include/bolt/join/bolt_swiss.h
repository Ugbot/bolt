// bolt_swiss.h — SwissTable-style hash index (Tiger Style, arena-allocated).
//
// 16-slot groups. One control byte per slot; high bit set = empty; low 7 bits
// = hash tag. Payload slots store {uint64_t key; uint32_t value;} in a parallel
// array. Lookup scans 16 control bytes per group via a SIMD cmpeq (SSE4.2/AVX2/
// NEON/scalar) plus `bolt_ctz32` for match enumeration.
//
// Rules: noexcept, no exceptions, no RTTI, no growth on the hot path. Capacity
// rounded to a power of two at `create()` time and never changes.
//
// Reference: docs/BOLT_ROADMAP.md Phase 3.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_config.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
inline constexpr uint32_t kSwissGroupWidth   = 16;
inline constexpr uint8_t  kSwissCtrlEmpty    = 0x80;  // high bit = empty
inline constexpr uint32_t kSwissMinCapacity  = 16;
inline constexpr uint32_t kSwissMaxProbeCaps = 1u << 28;  // 256M slots cap

struct SwissSlot {
    uint64_t key;
    uint32_t value;
    uint32_t _pad;
};

static_assert(sizeof(SwissSlot) == 16, "SwissSlot must be 16 bytes");

// ---------------------------------------------------------------------------
// Hash mixer — wyhash-style 3-op mix.
//
// Was Murmur3 finalizer (6 ops, ~5-7 cycle dep chain). The 3-op form
// keeps independent high/low halves (so SwissTable's tag/idx split keeps
// its selectivity) at roughly half the cost. Avalanche margin is lower
// than Murmur3 but adequate for open-addressing on adversarial-but-not-
// pathological inputs.
//
// Trade-off rationale and rejected alternatives (plain Fibonacci hash,
// PtrHash, pre-hashed contracts) are documented in
// docs/research/hash-functions.md. Constant from the wyhash / foldhash
// family (Wang Yi, https://github.com/wangyi-fudan/wyhash).
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint64_t swiss_mix(uint64_t k) noexcept {
    k ^= k >> 32;
    k *= 0xE7037ED1A0B428DBULL;
    k ^= k >> 32;
    return k;
}

BOLT_FORCE_INLINE uint8_t swiss_tag(uint64_t h) noexcept {
    // Low 7 bits of the upper word; high bit 0 so we never collide with Empty.
    return static_cast<uint8_t>((h >> 57) & 0x7Fu);
}

BOLT_FORCE_INLINE uint32_t swiss_round_up_pow2(uint64_t x) noexcept {
    if (x <= kSwissMinCapacity) return kSwissMinCapacity;
    if (x >= kSwissMaxProbeCaps) return kSwissMaxProbeCaps;
    uint32_t r = 1;
    while (r < x) r <<= 1;
    return r;
}

// ---------------------------------------------------------------------------
// SwissTable
// ---------------------------------------------------------------------------
// Open-addressing hash table with a 16-slot SIMD-scanned control array. The
// ctrl/slots arrays are each `capacity` elements; capacity is a power of two,
// so index masking is `h & (cap-1)`. On insert we linear-probe one slot at a
// time (not groups) — correctness-first for Phase 3; SIMD is still used inside
// `find*` to scan 16 control bytes at once.
struct SwissTable {
    uint8_t*   ctrl;      // capacity bytes, arena-allocated, all Empty init
    SwissSlot* slots;     // capacity entries, arena-allocated
    uint32_t   capacity;  // power of two, >= kSwissMinCapacity
    uint32_t   mask;      // capacity - 1
    uint32_t   size;      // live entries

    // ------------------------------------------------------------------
    // Factory: allocate ctrl + slots from arena. Returns false on OOM.
    // capacity_hint is max expected entries; we size to >= 2x for <=50% load.
    // ------------------------------------------------------------------
    static bool create(SwissTable* out, uint64_t capacity_hint, Arena* arena) noexcept {
        return create_with(out, capacity_hint, arena, /*tight_sizing=*/false);
    }

    // ------------------------------------------------------------------
    // Cardinality-aware factory: when caller knows the exact upper-bound
    // number of distinct keys (e.g. 1BRC = 413 stations), pass tight_sizing
    // = true to round capacity to the smallest power-of-two >= capacity_hint
    // instead of the default 2x oversize. Faster lookups under known load
    // (better cache density), but catastrophic if real cardinality exceeds
    // the hint. Default false preserves existing behavior.
    // ------------------------------------------------------------------
    static bool create_with(SwissTable* out, uint64_t capacity_hint,
                            Arena* arena, bool tight_sizing) noexcept {
        assert(out != nullptr);
        assert(arena != nullptr);

        uint64_t target = tight_sizing ? capacity_hint : capacity_hint * 2;
        if (target < kSwissMinCapacity) target = kSwissMinCapacity;
        uint32_t cap = swiss_round_up_pow2(target);
        assert(cap >= kSwissMinCapacity);
        assert((cap & (cap - 1)) == 0);

        // Over-allocate ctrl by kSwissGroupWidth so SIMD scans at the tail
        // don't read past the array. We mirror the first group at the end.
        uint8_t*   c = static_cast<uint8_t*>(arena->allocate(cap + kSwissGroupWidth, 16));
        SwissSlot* s = arena->allocate_array<SwissSlot>(cap);
        if (!c || !s) return false;

        memset(c, kSwissCtrlEmpty, cap + kSwissGroupWidth);
        memset(s, 0, cap * sizeof(SwissSlot));

        out->ctrl     = c;
        out->slots    = s;
        out->capacity = cap;
        out->mask     = cap - 1;
        out->size     = 0;
        return true;
    }

    // ------------------------------------------------------------------
    // Prefetch the control + slot cache lines for `key` into L1, ahead of
    // a future find()/insert(). The prefetched lines hold the first probe
    // group (16 ctrl bytes + 16-slot SwissSlot[] segment). Cheap when the
    // line is already cached; one cache-miss avoided when not. Used by
    // bulk callers (parallel groupby morsel, find_simd) that know future
    // keys ahead of time.
    // ------------------------------------------------------------------
    void prefetch(uint64_t key) const noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);
        const uint64_t h   = swiss_mix(key);
        const uint32_t idx = static_cast<uint32_t>(h) & mask;
        BOLT_PREFETCH_READ(&ctrl[idx]);
        BOLT_PREFETCH_READ(&slots[idx]);
    }

    // ------------------------------------------------------------------
    // Insert. Linear-probes for an empty slot. Returns false if the table
    // is full (size would exceed capacity) — Tiger Style: no growth.
    // If the key already exists, its value is overwritten and size unchanged.
    // ------------------------------------------------------------------
    bool insert(uint64_t key, uint32_t value) noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);
        if (size >= capacity) return false;

        uint64_t h = swiss_mix(key);
        uint8_t  tag = swiss_tag(h);
        uint32_t idx = static_cast<uint32_t>(h) & mask;

        constexpr uint32_t kMaxProbes = kSwissMaxProbeCaps;
        for (uint32_t probe = 0; probe < kMaxProbes; ++probe) {
            uint8_t c = ctrl[idx];
            if (c == kSwissCtrlEmpty) {
                ctrl[idx]        = tag;
                slots[idx].key   = key;
                slots[idx].value = value;
                if (idx < kSwissGroupWidth) ctrl[capacity + idx] = tag;
                ++size;
                return true;
            }
            if (c == tag && slots[idx].key == key) {
                slots[idx].value = value;  // update in place
                return true;
            }
            idx = (idx + 1) & mask;
        }
        return false;  // unreachable given size check above
    }

    // ------------------------------------------------------------------
    // Find. Returns value index, or -1 if not present.
    //
    // Scans 16 control bytes per group using `bmm_cmpeq_i8` + movemask —
    // both the tag-match and empty-slot checks happen in parallel. Advances
    // by 16 per group (matches the linear-probe insertion semantics because
    // we over-allocated `ctrl` by kSwissGroupWidth and mirrored the first
    // group at the tail, so group loads never run past the buffer).
    // ------------------------------------------------------------------
    int32_t find(uint64_t key) const noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);

        using namespace bolt::simd;
        const uint64_t h   = swiss_mix(key);
        const uint8_t  tag = swiss_tag(h);

        // Build broadcast vectors for tag / empty once.
        const bmm_vec_i8 vtag   = bmm_set1_i8(static_cast<int8_t>(tag));
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));

        uint32_t base = static_cast<uint32_t>(h) & mask;
        // kSwissGroupWidth == 16; bmm_lanes_i8 is 32 on AVX2, 16 elsewhere.
        // We only use the low 16 lanes so every ISA works with one load.
        constexpr uint32_t kGroup = kSwissGroupWidth;

        for (uint32_t probes = 0; probes < capacity; probes += kGroup) {
            const bmm_vec_i8 cgrp = bmm_loadu_i8(
                reinterpret_cast<const int8_t*>(ctrl + base));

            uint32_t tag_mask   = static_cast<uint32_t>(bmm_movemask_i8(
                                      bmm_cmpeq_i8(cgrp, vtag))) & 0xFFFFu;
            const uint32_t empty_mask = static_cast<uint32_t>(bmm_movemask_i8(
                                      bmm_cmpeq_i8(cgrp, vempty))) & 0xFFFFu;

            // Check each tag match in order; key compare confirms.
            while (tag_mask) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(tag_mask));
                const uint32_t p    = (base + lane) & mask;
                if (slots[p].key == key) {
                    return static_cast<int32_t>(slots[p].value);
                }
                tag_mask &= tag_mask - 1;
            }

            // If an empty exists in this group, the key cannot be further.
            if (empty_mask != 0) return -1;

            base = (base + kGroup) & mask;
        }
        return -1;
    }

    // ------------------------------------------------------------------
    // Vectorized probe: find `n` keys, write value-or-(-1) per key.
    // Cardinality-adaptive prefetch — only when the table itself exceeds
    // L1 (~2048 slots). Below that the cache lines are already hot and
    // prefetch issues are pure overhead.
    // ------------------------------------------------------------------
    void find_simd(const uint64_t* BOLT_RESTRICT keys, int64_t n,
                   int32_t* BOLT_RESTRICT out) const noexcept {
        assert(keys != nullptr || n == 0);
        assert(out != nullptr  || n == 0);
        constexpr int64_t kPF = 16;
        if (capacity > config::kPrefetchTableCapacityThreshold) {
            for (int64_t i = 0; i < n; ++i) {
                if (i + kPF < n) prefetch(keys[i + kPF]);
                out[i] = find(keys[i]);
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                out[i] = find(keys[i]);
            }
        }
    }
};

}  // namespace bolt
