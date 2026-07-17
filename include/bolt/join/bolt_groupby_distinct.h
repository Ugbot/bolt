// bolt_groupby_distinct.h — Per-cell DISTINCT-value tracker for grouped
// aggregates (Tiger Style).
//
// Purpose
// -------
// SQL `COUNT(DISTINCT col)` / `SUM(DISTINCT col)` need to dedupe input values
// *per group* before accumulating. A naive implementation either (a) hashes
// the full (group_key, value) into one big SwissTable — large memory tax on
// low-cardinality cases — or (b) maintains a per-group std::unordered_set —
// banned (heap, RAII, mutex). This header gives a third path:
//
//   - A fixed-capacity inline cell holding up to `k_distinct_cell_cap` 64-bit
//     values. Lookup is a small linear scan (typically branch-predicted hot).
//   - On overflow the cell flips to an HLL-style "sticky-saturated" mode
//     where every new value is *probabilistically* counted (currently:
//     conservative — overflowed cells stop deduping further inserts and
//     return a `count >= cap` signal so the caller can fall back to a
//     full hash set in a follow-up wave). Phase-A scope: correct up to the
//     cap; over-cap behaviour is documented and asserts in debug.
//
// ns/row floor (single-thread, RelWithDebInfo, AVX2 tier):
//   - DistinctCell::insert with cell occupancy ≤ 8  : ≤ 4.0 ns/row
//   - DistinctCell::insert with cell occupancy ≤ 32 : ≤ 12.0 ns/row
//   - over-cap (saturated)                          : ≤ 1.0 ns/row (early-out)
// (Not measured in this commit — see K-AGG-B follow-up bench.)
//
// Tiger Style: noexcept, no heap, ≥ 2 asserts/fn, ≤ 70 lines/fn, POD.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_hash.h"     // swiss_mix — spill-set hashing
#include "bolt/bolt_arena.h"    // Arena — exact spill on cell saturation

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// Per-cell inline distinct-set capacity. Sized for TPC-H-style cardinalities
// where most COUNT(DISTINCT) groups have < 100 unique values. Each cell costs
// `k_distinct_cell_cap * 8` bytes of state + 4 bytes of bookkeeping; we keep
// it ≤ a cache line slab so a column of cells is dense in memory.
constexpr std::uint32_t k_distinct_cell_cap = 64;

// ---------------------------------------------------------------------------
// DistinctSet64 — arena-backed, growable, EXACT int64 hash-set used as the
// overflow (spill) store when a DistinctCell exceeds its inline cap. Open
// addressing with a separate 1-byte occupancy array (so any int64 value,
// including INT64_MIN-adjacent keys, is storable without a reserved sentinel).
// Grows by doubling + rehash from the arena at 0.75 load. This is what makes
// COUNT(DISTINCT)/SUM(DISTINCT) exact for high-cardinality inputs instead of
// silently saturating at k_distinct_cell_cap.
// ---------------------------------------------------------------------------
struct DistinctSet64 {
    std::int64_t* keys;   // [cap], valid only where occ==1
    std::uint8_t* occ;    // [cap], 0 = empty, 1 = occupied
    std::uint32_t cap;    // power of two
    std::uint32_t size;   // live entries

    static DistinctSet64* create(std::uint32_t cap_hint, bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        DistinctSet64* s = arena->allocate_array<DistinctSet64>(1);
        if (s == nullptr) return nullptr;
        std::uint32_t cap = 128;
        while (cap < cap_hint) cap <<= 1;
        s->keys = arena->allocate_array<std::int64_t>(cap);
        s->occ  = arena->allocate_array<std::uint8_t>(cap);
        if (s->keys == nullptr || s->occ == nullptr) return nullptr;
        std::memset(s->occ, 0, cap);
        s->cap = cap; s->size = 0;
        return s;
    }

    BOLT_FORCE_INLINE bool grow(bolt::Arena* arena) noexcept {
        const std::uint32_t ncap = cap << 1;
        std::int64_t* nk = arena->allocate_array<std::int64_t>(ncap);
        std::uint8_t* no = arena->allocate_array<std::uint8_t>(ncap);
        if (nk == nullptr || no == nullptr) return false;
        std::memset(no, 0, ncap);
        const std::uint32_t nmask = ncap - 1;
        for (std::uint32_t j = 0; j < cap; ++j) {
            if (!occ[j]) continue;
            std::uint32_t i = static_cast<std::uint32_t>(
                bolt::swiss_mix(static_cast<std::uint64_t>(keys[j]))) & nmask;
            while (no[i]) i = (i + 1) & nmask;
            no[i] = 1; nk[i] = keys[j];
        }
        keys = nk; occ = no; cap = ncap;
        return true;
    }

    // Returns true iff v was newly inserted (not previously present). Returns
    // false on duplicate, or on true arena exhaustion at grow time (the only
    // residual undercount path — a condition under which the whole query is
    // already failing elsewhere).
    BOLT_FORCE_INLINE bool insert(std::int64_t v, bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        if (static_cast<std::uint64_t>(size + 1) * 4 >=
            static_cast<std::uint64_t>(cap) * 3) {            // 0.75 load
            if (size + 1 >= cap && !grow(arena)) return false;  // full + OOM
            if (size * 4 >= cap * 3) { if (!grow(arena)) { /* keep probing */ } }
        }
        const std::uint32_t mask = cap - 1;
        std::uint32_t i = static_cast<std::uint32_t>(
            bolt::swiss_mix(static_cast<std::uint64_t>(v))) & mask;
        while (occ[i]) {
            if (keys[i] == v) return false;
            i = (i + 1) & mask;
        }
        occ[i] = 1; keys[i] = v; ++size;
        return true;
    }
};

// ---------------------------------------------------------------------------
// DistinctSetUtf8 — arena-backed, growable, EXACT set of variable-length byte
// strings, deduped BY CONTENT (not by StringView offset). This is what makes
// COUNT(DISTINCT <utf8_col>) correct for spilled (>12-byte) strings: two equal
// strings stored at different overflow offsets have different StringViews but
// identical content, so an offset/view compare overcounts. Open addressing on
// an FNV-1a content hash; content bytes are copied into an arena-backed byte
// pool so the set owns them past the source batch. Grows by doubling + rehash
// at 0.75 load (content pool grows geometrically, same accepted pattern as the
// typed-agg key_str_buf / utf8_acc_buf).
// ---------------------------------------------------------------------------
struct DistinctSetU8Slot {
    std::uint64_t hash;    // FNV-1a of content
    std::uint32_t off;     // byte offset into content pool
    std::uint32_t len;     // content length
};
struct DistinctSetUtf8 {
    DistinctSetU8Slot* slots;   // [cap]
    std::uint8_t*      occ;     // [cap]
    std::uint32_t      cap;     // power of two
    std::uint32_t      size;    // live entries
    char*              pool;    // content byte pool
    std::uint32_t      pool_used;
    std::uint32_t      pool_cap;

    static BOLT_FORCE_INLINE std::uint64_t hash_bytes(const char* p,
                                                      std::uint32_t len) noexcept {
        std::uint64_t h = 1469598103934665603ull;          // FNV-1a offset basis
        for (std::uint32_t i = 0; i < len; ++i) {
            h ^= static_cast<std::uint8_t>(p[i]);
            h *= 1099511628211ull;
        }
        return h;
    }

    static DistinctSetUtf8* create(std::uint32_t cap_hint, bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        DistinctSetUtf8* s = arena->allocate_array<DistinctSetUtf8>(1);
        if (s == nullptr) return nullptr;
        std::uint32_t cap = 128;
        while (cap < cap_hint) cap <<= 1;
        s->slots = arena->allocate_array<DistinctSetU8Slot>(cap);
        s->occ   = arena->allocate_array<std::uint8_t>(cap);
        s->pool_cap  = 1u << 16;
        s->pool      = arena->allocate_array<char>(s->pool_cap);
        if (s->slots == nullptr || s->occ == nullptr || s->pool == nullptr) return nullptr;
        std::memset(s->occ, 0, cap);
        s->cap = cap; s->size = 0; s->pool_used = 0;
        return s;
    }

    BOLT_FORCE_INLINE bool grow_slots(bolt::Arena* arena) noexcept {
        const std::uint32_t ncap = cap << 1;
        auto* ns = arena->allocate_array<DistinctSetU8Slot>(ncap);
        auto* no = arena->allocate_array<std::uint8_t>(ncap);
        if (ns == nullptr || no == nullptr) return false;
        std::memset(no, 0, ncap);
        const std::uint32_t nmask = ncap - 1;
        for (std::uint32_t j = 0; j < cap; ++j) {
            if (!occ[j]) continue;
            std::uint32_t i = static_cast<std::uint32_t>(slots[j].hash) & nmask;
            while (no[i]) i = (i + 1) & nmask;
            no[i] = 1; ns[i] = slots[j];
        }
        slots = ns; occ = no; cap = ncap;
        return true;
    }

    // Returns true iff the content was newly inserted.
    BOLT_FORCE_INLINE bool insert(const char* bytes, std::uint32_t len,
                                  bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(bytes != nullptr || len == 0);
        if (static_cast<std::uint64_t>(size + 1) * 4 >=
            static_cast<std::uint64_t>(cap) * 3) {
            if (!grow_slots(arena) && size + 1 >= cap) return false;
        }
        const std::uint64_t h = hash_bytes(bytes, len);
        const std::uint32_t mask = cap - 1;
        std::uint32_t i = static_cast<std::uint32_t>(h) & mask;
        while (occ[i]) {
            if (slots[i].hash == h && slots[i].len == len &&
                std::memcmp(pool + slots[i].off, bytes, len) == 0) {
                return false;                                   // duplicate content
            }
            i = (i + 1) & mask;
        }
        // Store content in the pool (grow geometrically if needed).
        if (pool_used + len > pool_cap) {
            std::uint32_t ncap = pool_cap * 2u;
            while (ncap < pool_used + len) ncap *= 2u;
            char* np = arena->allocate_array<char>(ncap);
            if (np == nullptr) return false;
            std::memcpy(np, pool, pool_used);
            pool = np; pool_cap = ncap;
        }
        std::memcpy(pool + pool_used, bytes, len);
        occ[i] = 1;
        slots[i].hash = h; slots[i].off = pool_used; slots[i].len = len;
        pool_used += len;
        ++size;
        return true;
    }
};

// 64-byte aligned inline distinct-value tracker for ONE (group, agg) cell.
// `n` holds the live entry count; once it reaches `k_distinct_cell_cap` the
// cell is marked saturated and future inserts increment `overflow_seen`
// (caller can detect overcount with `saturated()`).
struct DistinctCell {
    std::uint32_t   n;                            // inline live entries (≤ cap)
    std::uint32_t   overflow_seen;                // legacy: over-cap drops when no arena
    std::int64_t    values[k_distinct_cell_cap];  // inline dedup set (fast path)
    DistinctSet64*  spill;                         // null until inline cap exceeded

    BOLT_FORCE_INLINE void init() noexcept {
        n = 0; overflow_seen = 0; spill = nullptr;
        // values[] is logically unused until n grows; no need to zero.
    }

    BOLT_FORCE_INLINE bool saturated() const noexcept {
        return spill == nullptr && n >= k_distinct_cell_cap;
    }

    // Exact live distinct count (inline entries, or spill size once promoted).
    BOLT_FORCE_INLINE std::uint32_t count() const noexcept {
        return spill != nullptr ? spill->size : n;
    }

    // Insert `v` if not already present. Returns true iff it was a NEW value
    // (caller then does `sum += v` / `count += 1`). Returns false on duplicate.
    //
    // With `arena != nullptr` the set is EXACT: once the inline cap is reached
    // the cell promotes to an arena-backed DistinctSet64 (migrating the inline
    // values) and keeps deduping without bound. With `arena == nullptr` the
    // legacy capped behaviour is preserved (drops past cap) for callers/tests
    // that don't supply one.
    BOLT_FORCE_INLINE bool insert(std::int64_t v, bolt::Arena* arena = nullptr) noexcept {
        if (spill != nullptr) return spill->insert(v, arena);
        // Linear-scan dedupe. For n ≤ ~32 this beats a hash probe by a wide
        // margin (no hash compute, no indirection, SIMD-friendly).
        const std::uint32_t cur = n;
        for (std::uint32_t i = 0; i < cur; ++i) {
            if (values[i] == v) return false;
        }
        if (cur < k_distinct_cell_cap) {
            values[cur] = v;
            n = cur + 1;
            return true;
        }
        // Inline cap reached and v is genuinely new.
        if (arena == nullptr) { overflow_seen += 1; return false; }  // legacy drop
        spill = DistinctSet64::create(k_distinct_cell_cap * 4, arena);
        if (spill == nullptr) { overflow_seen += 1; return false; }  // arena OOM
        for (std::uint32_t i = 0; i < cur; ++i) (void)spill->insert(values[i], arena);
        return spill->insert(v, arena);  // v not in inline set → newly inserted
    }
};

static_assert(sizeof(DistinctCell) == 8 + k_distinct_cell_cap * 8 + sizeof(void*),
              "DistinctCell layout — 8B header + N*8B inline values + spill ptr");

// Flat array of DistinctCells, one per (group, agg) slot. The chukonu typed
// hash-agg operator allocates one of these in its arena when ANY agg is
// flagged `distinct == 1`, sized [n_distinct_aggs * entry_cap]. Layout
// mirrors `accums_flat`: cell for (agg j, group g) lives at
//   distinct_cells[j * entry_cap + g]
// Caller seeds via `cell_at(...)->init()` lazily on first touch.
// 16-byte-wide DistinctCell variant for inline Utf8 keys (and any other
// 16-byte-key DISTINCT semantics that arrive). Same linear-scan dedup as
// the int64 version; 64-entry inline cap; saturates silently after cap.
// Callers pad unused bytes to zero (e.g. StringView's inline_data tail) —
// equality is a flat 16-byte memcmp.
struct DistinctCell16 {
    std::uint32_t n;                              // live entries (≤ cap)
    std::uint32_t overflow_seen;                  // inserts attempted past cap
    std::uint8_t values[k_distinct_cell_cap * 16];   // 64 * 16 B (no align pad)

    BOLT_FORCE_INLINE void init() noexcept {
        n = 0; overflow_seen = 0;
    }

    BOLT_FORCE_INLINE bool saturated() const noexcept {
        return n >= k_distinct_cell_cap;
    }

    // Insert a 16-byte value if not already present. Returns true if it
    // was a NEW value, false on duplicate or saturation.
    BOLT_FORCE_INLINE bool insert(const std::uint8_t value[16]) noexcept {
        assert(value != nullptr);
        assert(n <= k_distinct_cell_cap);
        const std::uint32_t cur = n;
        for (std::uint32_t i = 0; i < cur; ++i) {
            if (std::memcmp(&values[i * 16], value, 16) == 0) return false;
        }
        if (cur >= k_distinct_cell_cap) {
            overflow_seen += 1;
            return false;
        }
        std::memcpy(&values[cur * 16], value, 16);
        n = cur + 1;
        assert(n <= k_distinct_cell_cap);
        return true;
    }
};

static_assert(sizeof(DistinctCell16) == 8 + k_distinct_cell_cap * 16,
              "DistinctCell16 layout — packed (8B header + N*16B values)");

// Free-function helpers — match the API shape described in the K-AGG-B plan
// (some callers prefer C-style; the methods above are kept for parity with
// DistinctCell).
BOLT_FORCE_INLINE bool distinct_cell16_insert(DistinctCell16* c,
                                              const std::uint8_t value[16]) noexcept {
    assert(c != nullptr);
    return c->insert(value);
}
BOLT_FORCE_INLINE void distinct_cell16_reset(DistinctCell16* c) noexcept {
    assert(c != nullptr);
    c->init();
}

struct DistinctCellArray {
    DistinctCell* cells;     // [n_aggs_distinct * entry_cap]
    std::uint32_t entry_cap; // rows per agg
    std::uint16_t n_aggs;    // number of distinct-flagged aggs

    BOLT_FORCE_INLINE DistinctCell* cell_at(std::uint16_t j,
                                            std::uint32_t g) noexcept {
        assert(cells != nullptr);
        assert(j < n_aggs);
        assert(g < entry_cap);
        return &cells[static_cast<std::size_t>(j) * entry_cap + g];
    }
};

}  // namespace bolt
