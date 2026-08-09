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
#include "bolt/bolt_hash.h"
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
// Hash mixer — see `bolt_hash.h`.
//
// `swiss_mix` resolves to the variant selected by `BOLT_HASH_TIER_*`
// (WYHASH3 default, XXH3 and MURMUR3 also compiled in). Rationale,
// alternatives, and measured trade-offs live in
// docs/research/hash-functions.md and
// docs/research/design-log.md ("Hash mixer — …"). No runtime branch.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE uint8_t swiss_tag(uint64_t h) noexcept {
    // 7 bits from the upper-middle of the hash; high ctrl bit stays 0 so a
    // tag never collides with Empty.
    //
    // WHY bits 49..55 and not 57..63 (the historical choice): the hash join
    // partitions its tables by hj_partition_of(h) == bits 58..63 of the SAME
    // mixed hash. With the tag at 57..63, six of its seven bits were the
    // partition selector — constant within a partition — leaving ONE bit of
    // tag entropy: every SIMD group scan passed ~50% of slots to the full key
    // compare instead of ~1/128 (found reading the DaMoN'24 unchained-hash-
    // table paper against this file; TPC-H probe profiles bear the cost).
    // Bits 49..55 are disjoint from BOTH the partition selector (58..63) and
    // the table index (low 32 bits), restoring the full 1/128 filter inside
    // partitioned and standalone tables alike.
    return static_cast<uint8_t>((h >> 49) & 0x7Fu);
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

    // Same, for a caller that already holds `swiss_mix(key)` (see find_mixed).
    void prefetch_mixed(uint64_t h) const noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);
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
    // Upsert: ONE probe sequence that does what a find()+insert() pair does.
    // If `key` exists, its value is replaced with `value` and the old value
    // is written to *prev; if not, the pair is inserted at the first empty
    // slot and *prev is set to -1. Returns false only when the table is full
    // (nothing written).
    //
    // Motivation: the hash-join chain build wants "link to the previous head,
    // then make me the head" — historically a full find() followed by a full
    // insert(), i.e. two probe walks over the same group chain per build row.
    // Measured on TPC-H SF10 Q18 (build side 15M rows) the build was 23% of
    // query self-time with find+insert as separate walks. Uses find()'s
    // 16-lane group scan (not insert()'s byte walk) for both halves.
    // ------------------------------------------------------------------
    bool upsert(uint64_t key, uint32_t value, int32_t* prev) noexcept {
        assert(ctrl != nullptr && slots != nullptr);
        assert(prev != nullptr);
        using namespace bolt::simd;
        const uint64_t h   = swiss_mix(key);
        const uint8_t  tag = swiss_tag(h);
        const bmm_vec_i8 vtag   = bmm_set1_i8(static_cast<int8_t>(tag));
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));
        uint32_t base = static_cast<uint32_t>(h) & mask;
        constexpr uint32_t kGroup = kSwissGroupWidth;
        for (uint32_t probes = 0; probes < capacity; probes += kGroup) {
            const bmm_vec_i8 cgrp = bmm_loadu_i8(
                reinterpret_cast<const int8_t*>(ctrl + base));
            uint32_t tag_mask = static_cast<uint32_t>(bmm_movemask_i8(
                                    bmm_cmpeq_i8(cgrp, vtag))) & 0xFFFFu;
            const uint32_t empty_mask = static_cast<uint32_t>(bmm_movemask_i8(
                                    bmm_cmpeq_i8(cgrp, vempty))) & 0xFFFFu;
            // A tag match only counts if it sits BEFORE the first empty in
            // probe order — matches past an empty belong to other probe chains.
            const uint32_t first_empty =
                empty_mask ? static_cast<uint32_t>(bolt_ctz32(empty_mask)) : kGroup;
            while (tag_mask) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(tag_mask));
                if (lane > first_empty) break;
                const uint32_t p = (base + lane) & mask;
                if (slots[p].key == key) {
                    *prev = static_cast<int32_t>(slots[p].value);
                    slots[p].value = value;
                    return true;
                }
                tag_mask &= tag_mask - 1;
            }
            if (empty_mask != 0) {
                if (size >= capacity) return false;
                const uint32_t p = (base + first_empty) & mask;
                assert(ctrl[p] == kSwissCtrlEmpty);
                ctrl[p]        = tag;
                slots[p].key   = key;
                slots[p].value = value;
                if (p < kSwissGroupWidth) ctrl[capacity + p] = tag;
                ++size;
                *prev = -1;
                return true;
            }
            base = (base + kGroup) & mask;
        }
        return false;
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
        return find_mixed(key, swiss_mix(key));
    }

    // ------------------------------------------------------------------
    // find() for a caller that ALREADY holds `swiss_mix(key)`.
    //
    // The partitioned hash join computes the mixed hash itself (to pick the
    // partition and to test the bloom) and then called find(), which mixed it
    // a second time. Behaviour is identical by construction — find() is now
    // literally this function with the mix applied — but the join's probe can
    // hand the hash straight through. `h` MUST equal swiss_mix(key); passing
    // anything else silently searches the wrong group.
    //
    // That precondition is deliberately NOT asserted: this repo's Release
    // build keeps asserts ON, and `assert(h == swiss_mix(key))` would put a
    // redundant hash on every find() in the codebase — an assert that costs
    // the thing it guards. The invariant is instead pinned by construction
    // (find() supplies it) and by the probe equivalence test, which fails
    // loudly on any mismatch of the resulting pairs.
    // ------------------------------------------------------------------
    int32_t find_mixed(uint64_t key, uint64_t h) const noexcept {
        assert(ctrl != nullptr);
        assert(slots != nullptr);

        using namespace bolt::simd;
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

// ---------------------------------------------------------------------------
// SwissTableInterleaved (C3) — read-only interleaved layout alternative.
//
// FLAT layout (`SwissTable`): ctrl[capacity] and slots[capacity] live in
// separate arrays.  A probe loads one ctrl cache line and, on a tag
// match, loads one SwissSlot cache line (may be 4 lines away in memory).
// On >L3 tables the ctrl↔slot pair traverse different TLB entries.
//
// INTERLEAVED layout (`SwissTableInterleaved`): each group of 16 control
// bytes + 16 SwissSlot entries sits in one contiguous 272-byte block.
// The ctrl cache line and the first few slot cache lines fall under the
// same TLB mapping — probes that match in the first few lanes save a
// TLB walk on large tables.
//
// Build-once, probe-many: `build_from` copies a populated `SwissTable`
// into the interleaved layout. No insert/update/delete in this wave —
// hash-join build is insert-once, so this covers the main use-case.
// Kept-code-paths policy: FLAT remains the default
// (`BOLT_SWISS_LAYOUT_FLAT=1` from `bolt_apply_feature_toggles`). This
// alternative ships alongside and is reachable by name (or by CMake
// switch once a caller auto-selects per table size).
// ---------------------------------------------------------------------------

// Local cache-line constant (bolt_channel.h defines the canonical one but
// we avoid the include-cycle by duplicating it here).
inline constexpr uint32_t kSwissCacheLine = 64;

// One packed group: 16 ctrl bytes + 16 slots. Cache-line-aligned so the
// group lands on a line boundary regardless of the previous group's end.
struct alignas(kSwissCacheLine) SwissInterleavedGroup {
    uint8_t   ctrl[kSwissGroupWidth];   // 16 bytes
    uint8_t   pad[kSwissCacheLine - kSwissGroupWidth]; // pad before slots
    SwissSlot slots[kSwissGroupWidth];  // 16 × 16 = 256 bytes = 4 lines
};
static_assert(sizeof(SwissInterleavedGroup) ==
              kSwissCacheLine + sizeof(SwissSlot) * kSwissGroupWidth,
              "SwissInterleavedGroup layout");

struct SwissTableInterleaved {
    SwissInterleavedGroup* groups;  // capacity / kGroupWidth entries
    uint32_t               num_groups;  // capacity / kSwissGroupWidth
    uint32_t               group_mask;  // num_groups - 1
    uint32_t               capacity;
    uint32_t               size;

    // Create an empty interleaved table sized for `capacity_hint` entries.
    // Internal probing is group-aligned (probe-by-group, not probe-by-slot),
    // so the layout is self-contained — key placement will differ from FLAT
    // but every key that was inserted here is findable here.
    static bool create(SwissTableInterleaved* out, uint64_t capacity_hint,
                       Arena* arena) noexcept {
        assert(out != nullptr);
        assert(arena != nullptr);

        uint64_t target = capacity_hint * 2;
        if (target < kSwissMinCapacity) target = kSwissMinCapacity;
        uint32_t cap = swiss_round_up_pow2(target);
        const uint32_t ng = cap / kSwissGroupWidth;

        SwissInterleavedGroup* g = arena->allocate_array<SwissInterleavedGroup>(ng);
        if (!g) return false;

        // Initialize every ctrl byte to Empty; slots zero-init.
        for (uint32_t i = 0; i < ng; ++i) {
            memset(g[i].ctrl, kSwissCtrlEmpty, kSwissGroupWidth);
            memset(g[i].slots, 0, sizeof(SwissSlot) * kSwissGroupWidth);
        }

        out->groups      = g;
        out->num_groups  = ng;
        out->group_mask  = ng - 1u;
        out->capacity    = cap;
        out->size        = 0;
        return true;
    }

    // Insert. Linear-probe across groups (group-aligned).  Within a group,
    // fill the first Empty lane.  Returns false if the table is full.
    bool insert(uint64_t key, uint32_t value) noexcept {
        assert(groups != nullptr);
        if (size >= capacity) return false;

        const uint64_t h = swiss_mix(key);
        const uint8_t tag = swiss_tag(h);
        uint32_t gi = (static_cast<uint32_t>(h) & (capacity - 1u)) / kSwissGroupWidth;

        for (uint32_t p = 0; p < num_groups; ++p) {
            SwissInterleavedGroup& grp = groups[gi];
            // Scan for matching tag first — update-on-existing-key.
            for (uint32_t lane = 0; lane < kSwissGroupWidth; ++lane) {
                const uint8_t c = grp.ctrl[lane];
                if (c == tag && grp.slots[lane].key == key) {
                    grp.slots[lane].value = value;
                    return true;
                }
            }
            // No existing match — look for an Empty.
            for (uint32_t lane = 0; lane < kSwissGroupWidth; ++lane) {
                if (grp.ctrl[lane] == kSwissCtrlEmpty) {
                    grp.ctrl[lane] = tag;
                    grp.slots[lane].key   = key;
                    grp.slots[lane].value = value;
                    ++size;
                    return true;
                }
            }
            gi = (gi + 1u) & group_mask;
        }
        return false;
    }

    // Build from a populated FLAT SwissTable by reinserting every occupied
    // slot. Capacity matches the source so collisions don't force growth.
    static bool build_from(SwissTableInterleaved* out, const SwissTable& src,
                           Arena* arena) noexcept {
        assert(out != nullptr);
        assert(arena != nullptr);
        assert(src.capacity > 0);

        // Size for >=src.size entries; interleaved's own 2x oversize runs
        // inside create().
        if (!create(out, src.size, arena)) return false;

        for (uint32_t i = 0; i < src.capacity; ++i) {
            if (src.ctrl[i] != kSwissCtrlEmpty) {
                if (!out->insert(src.slots[i].key, src.slots[i].value)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Find. Returns value index, or -1.
    int32_t find(uint64_t key) const noexcept {
        assert(groups != nullptr);

        using namespace bolt::simd;
        const uint64_t h   = swiss_mix(key);
        const uint8_t  tag = swiss_tag(h);

        const bmm_vec_i8 vtag   = bmm_set1_i8(static_cast<int8_t>(tag));
        const bmm_vec_i8 vempty = bmm_set1_i8(static_cast<int8_t>(kSwissCtrlEmpty));

        uint32_t gi = (static_cast<uint32_t>(h) & (capacity - 1u)) / kSwissGroupWidth;

        for (uint32_t p = 0; p < num_groups; ++p) {
            const SwissInterleavedGroup& grp = groups[gi];
            const bmm_vec_i8 cgrp = bmm_loadu_i8(
                reinterpret_cast<const int8_t*>(grp.ctrl));

            uint32_t tag_mask = static_cast<uint32_t>(bmm_movemask_i8(
                                    bmm_cmpeq_i8(cgrp, vtag))) & 0xFFFFu;
            const uint32_t empty_mask = static_cast<uint32_t>(bmm_movemask_i8(
                                    bmm_cmpeq_i8(cgrp, vempty))) & 0xFFFFu;

            while (tag_mask) {
                const uint32_t lane = static_cast<uint32_t>(bolt_ctz32(tag_mask));
                if (grp.slots[lane].key == key) {
                    return static_cast<int32_t>(grp.slots[lane].value);
                }
                tag_mask &= tag_mask - 1;
            }
            if (empty_mask != 0) return -1;

            gi = (gi + 1u) & group_mask;
        }
        return -1;
    }
};

// ---------------------------------------------------------------------------
// SwissTableSalted (G2FEAT-89/362) — salt-in-entry open-addressed table with
// an inline int64 payload.
//
// FLAT `SwissTable` (above) stores {ctrl byte} in one array and {key, value}
// in a separate array; a caller building an aggregate table typically adds a
// THIRD array (accumulators, indexed by SwissTable's uint32 value) since the
// 32-bit value field is meant as an index, not a payload. A probe hit then
// touches up to three cache lines: ctrl, slot, accumulator.
//
// SwissTableSalted co-locates {salt, key, int64 payload} in ONE 32-byte open-
// addressed entry (two per 64-byte cache line, never straddling — the salt
// and the trailing pad exist only to keep 8-byte-aligned `key`/`value` on a
// clean boundary). A probe hit touches ONE cache line: the salt filters
// candidates cheaply, the full key compare only runs on a salt match, and
// the payload is already resident for in-place accumulate — no second array.
//
// Micro-bench-justified go/no-go (`benchmarks/bench_salt_probe.cpp`,
// G2FEAT-89): vs FLAT `SwissTable` + a separate accumulator array, measured
// 2-4x faster for an aggregation-ingest-shaped find-or-insert-and-accumulate
// workload once the table exceeds L2/L3 residency (no win, sometimes a
// regression, below that — the extra 32B-vs-~17B entry size and branchier
// linear probe cost more than a cache-resident table's near-zero miss rate
// saves). Crucially, the win SURVIVES at 8 independent per-thread tables
// under DRAM-bandwidth contention (the parallelism-negative regime that
// motivated this work): at the 17M-distinct / 100M-row production-matched
// shape, `find_or_insert_batch` measured 2.3x faster than FLAT+separate-
// array at both 1 and 8 threads. Full numbers, sizes swept, and the exact
// methodology: docs/research/design-log.md "G2FEAT-89 — salt-in-entry
// SwissTable probe".
//
// Trade-off (do not hide this from a future caller): ~1.9x the per-slot
// memory footprint of FLAT (32B/entry vs FLAT's ~17B/slot including ctrl).
// The measured win holds even after paying it, but it is a real cost at
// >L3 table sizes and should be budgeted for, not assumed free.
//
// Scope: single inline int64 payload (sum/count/min/max-shaped) — the
// common case for a plain int64 aggregate. A composite or wide payload
// still wants FLAT `SwissTable` mapping key -> a caller-owned wider record.
//
// Kept-code-paths policy (mirrors `SwissTableInterleaved` / BOLT_SWISS_LAYOUT
// above): FLAT `SwissTable` remains the default for general key->slot
// lookups and is UNCHANGED by this addition. `SwissTableSalted` is reachable
// by name; wiring a cardinality-based picker into a real aggregate operator
// is a chukonu-side call (G2FEAT-362 productionization) and explicitly out
// of scope here — this is the bolt-primitive half only, provided tested and
// correct, not wired into any call site.
// ---------------------------------------------------------------------------
struct SwissEntrySalted {
    uint64_t salt;       // bit63 = occupied marker; low 63 bits = hash tag
    int64_t  key;
    int64_t  value;
    int64_t  _reserved;  // pads to 32B so entries never straddle a cache line
};
static_assert(sizeof(SwissEntrySalted) == 32, "SwissEntrySalted must be 32 bytes");

inline constexpr uint64_t kSwissSaltedOccupied = uint64_t(1) << 63;

struct SwissTableSalted {
    SwissEntrySalted* entries;   // capacity entries, arena-allocated
    uint64_t          capacity;  // power of two
    uint64_t          mask;      // capacity - 1
    uint64_t          size;      // live entries

    // ------------------------------------------------------------------
    // Factory: allocate `entries` from arena, all zero (unoccupied).
    // capacity_hint is the max expected distinct keys; sized to ~66% load
    // (matches the micro-bench-validated shape), rounded to a power of two.
    // ------------------------------------------------------------------
    static bool create(SwissTableSalted* out, uint64_t capacity_hint,
                       Arena* arena) noexcept {
        assert(out != nullptr);
        assert(arena != nullptr);

        uint64_t target = capacity_hint + (capacity_hint >> 1) + kSwissMinCapacity;
        uint32_t cap = swiss_round_up_pow2(target);
        assert(cap >= kSwissMinCapacity);
        assert((cap & (cap - 1)) == 0);

        SwissEntrySalted* e = arena->allocate_array<SwissEntrySalted>(cap);
        if (!e) return false;
        memset(e, 0, sizeof(SwissEntrySalted) * cap);

        out->entries  = e;
        out->capacity = cap;
        out->mask     = cap - 1;
        out->size     = 0;
        return true;
    }

    // Prefetch the entry line a future find()/find_or_insert() will touch.
    void prefetch(uint64_t key) const noexcept {
        assert(entries != nullptr);
        assert(capacity > 0);
        const uint64_t h = swiss_mix(key);
        BOLT_PREFETCH_WRITE(&entries[h & mask]);
    }

    // ------------------------------------------------------------------
    // Find-or-insert. Returns a pointer to the entry's `value` field —
    // freshly set to `initial_value` on first sight of `key`, or the
    // existing accumulator otherwise (caller does `*slot += x`, `*slot =
    // max(*slot, x)`, etc. in place — no second lookup, no second array).
    //
    // Returns nullptr only if the table is completely full and `key` is
    // not already present — Tiger Style: bounded probe, no growth on the
    // hot path, fail rather than loop forever or grow silently. Caller
    // must size `capacity_hint` generously at create() time.
    // ------------------------------------------------------------------
    int64_t* find_or_insert(uint64_t key, int64_t initial_value) noexcept {
        return find_or_insert_hashed(key, swiss_mix(key), initial_value);
    }

    // Same as find_or_insert but takes a pre-computed hash — the primitive
    // find_or_insert_batch() below builds on, and any caller doing its own
    // windowed prefetch (bench_salt_probe.cpp approach "C") can reuse.
    int64_t* find_or_insert_hashed(uint64_t key, uint64_t h,
                                   int64_t initial_value) noexcept {
        assert(entries != nullptr);
        assert(capacity > 0);
        const uint64_t salt = (h & (kSwissSaltedOccupied - 1)) | kSwissSaltedOccupied;
        uint64_t slot = h & mask;

        for (uint64_t probe = 0; probe < capacity; ++probe) {
            SwissEntrySalted& c = entries[slot];
            if (c.salt == 0) {
                if (size >= capacity) return nullptr;
                c.key   = static_cast<int64_t>(key);
                c.value = initial_value;
                c.salt  = salt;
                ++size;
                return &c.value;
            }
            if (c.salt == salt && c.key == static_cast<int64_t>(key)) {
                return &c.value;
            }
            slot = (slot + 1) & mask;
        }
        return nullptr;  // table full, key not present — bounded, no growth
    }

    // Pure lookup. Returns nullptr if `key` was never inserted.
    int64_t* find(uint64_t key) const noexcept {
        assert(entries != nullptr);
        assert(capacity > 0);
        const uint64_t h    = swiss_mix(key);
        const uint64_t salt = (h & (kSwissSaltedOccupied - 1)) | kSwissSaltedOccupied;
        uint64_t slot = h & mask;

        for (uint64_t probe = 0; probe < capacity; ++probe) {
            SwissEntrySalted& c = entries[slot];
            if (c.salt == 0) return nullptr;
            if (c.salt == salt && c.key == static_cast<int64_t>(key)) return &c.value;
            slot = (slot + 1) & mask;
        }
        return nullptr;
    }

    // ------------------------------------------------------------------
    // Batched find-or-insert with windowed software prefetch — the
    // strongest measured variant in bench_salt_probe.cpp ("approach C"):
    // precompute a window of hashes, prefetch each target entry line, then
    // probe — hides DRAM miss latency across the window instead of
    // stalling one row at a time. `out_slots[i]` receives the pointer
    // find_or_insert_hashed() would have returned for `keys[i]`; nullptr
    // entries mean the table filled (same bounded-failure contract as
    // find_or_insert). Caller drives accumulation from `out_slots` (kept
    // separate from the prefetch/probe loop so this stays payload-agnostic
    // — SUM/COUNT/MIN/MAX all reduce to a different write into *out_slots[i]).
    // ------------------------------------------------------------------
    static constexpr int64_t kPrefetchWindow = 32;

    void find_or_insert_batch(const uint64_t* BOLT_RESTRICT keys, int64_t n,
                              int64_t initial_value,
                              int64_t** BOLT_RESTRICT out_slots) noexcept {
        assert(keys != nullptr || n == 0);
        assert(out_slots != nullptr || n == 0);
        assert(n >= 0);

        uint64_t hbuf[kPrefetchWindow];
        for (int64_t base = 0; base < n; base += kPrefetchWindow) {
            const int64_t m = (base + kPrefetchWindow <= n) ? kPrefetchWindow : (n - base);
            for (int64_t j = 0; j < m; ++j) {
                const uint64_t h = swiss_mix(keys[base + j]);
                hbuf[j] = h;
                BOLT_PREFETCH_WRITE(&entries[h & mask]);
            }
            for (int64_t j = 0; j < m; ++j) {
                out_slots[base + j] =
                    find_or_insert_hashed(keys[base + j], hbuf[j], initial_value);
            }
        }
    }
};

}  // namespace bolt
