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
#include "bolt/bolt_types.h"    // StringView — content-exact Utf8 dedup

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

    BOLT_FORCE_INLINE bool contains(std::int64_t v) const noexcept {
        assert(cap > 0 && (cap & (cap - 1)) == 0);
        const std::uint32_t mask = cap - 1;
        std::uint32_t i = static_cast<std::uint32_t>(
            bolt::swiss_mix(static_cast<std::uint64_t>(v))) & mask;
        for (std::uint32_t probes = 0; probes < cap && occ[i]; ++probes) {
            if (keys[i] == v) return true;
            i = (i + 1) & mask;
        }
        return false;
    }

    // 1 new, 0 duplicate, -1 arena exhausted (never read as a duplicate).
    BOLT_FORCE_INLINE int insert_exact(std::int64_t v, bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        if (insert(v, arena)) return 1;
        return contains(v) ? 0 : -1;
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

    // `min_cap` / `pool_bytes` let a per-group cell start small; the
    // global (one set per query) default keeps the historical sizing.
    static DistinctSetUtf8* create(std::uint32_t cap_hint, bolt::Arena* arena,
                                   std::uint32_t pool_bytes = 1u << 16,
                                   std::uint32_t min_cap = 128) noexcept {
        assert(arena != nullptr);
        assert(pool_bytes > 0 && min_cap > 0 && (min_cap & (min_cap - 1)) == 0);
        DistinctSetUtf8* s = arena->allocate_array<DistinctSetUtf8>(1);
        if (s == nullptr) return nullptr;
        std::uint32_t cap = min_cap;
        while (cap < cap_hint) cap <<= 1;
        s->slots = arena->allocate_array<DistinctSetU8Slot>(cap);
        s->occ   = arena->allocate_array<std::uint8_t>(cap);
        s->pool_cap  = pool_bytes;
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

    BOLT_FORCE_INLINE bool contains(const char* bytes, std::uint32_t len) const noexcept {
        assert(cap > 0 && (cap & (cap - 1)) == 0);
        assert(bytes != nullptr || len == 0);
        const std::uint64_t h = hash_bytes(bytes, len);
        const std::uint32_t mask = cap - 1;
        std::uint32_t i = static_cast<std::uint32_t>(h) & mask;
        for (std::uint32_t probes = 0; probes < cap && occ[i]; ++probes) {
            if (slots[i].hash == h && slots[i].len == len &&
                std::memcmp(pool + slots[i].off, bytes, len) == 0) return true;
            i = (i + 1) & mask;
        }
        return false;
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

    // Exact insert: 1 new, 0 duplicate, -1 arena exhausted. The caller fails
    // the query on -1 instead of counting the value as a duplicate.
    int insert_exact(std::int64_t v, bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(n <= k_distinct_cell_cap);
        if (spill != nullptr) return spill->insert_exact(v, arena);
        for (std::uint32_t i = 0; i < n; ++i) {
            if (values[i] == v) return 0;
        }
        if (n < k_distinct_cell_cap) {
            values[n] = v;
            n += 1;
            return 1;
        }
        DistinctSet64* s = DistinctSet64::create(k_distinct_cell_cap * 4, arena);
        if (s == nullptr) return -1;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (s->insert_exact(values[i], arena) != 1) return -1;
        }
        spill = s;
        return spill->insert_exact(v, arena);
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
// Per-(group, agg) COUNT(DISTINCT utf8) tracker. Short (<=12-byte) values
// live canonicalised (length + zero-padded bytes) in the inline array and
// are deduped by a 16-byte memcmp. Spilled values never compare by their
// view (it holds a buffer offset, meaningless across offsets or morsels):
// they go by content into an arena-backed DistinctSetUtf8 that owns its
// bytes. A short value can never equal a spilled one (lengths differ), so
// the two stores are disjoint until the inline array fills; then every
// inline value migrates into the set (`promoted`) and all values route there.
struct DistinctCell16 {
    std::uint32_t n;                              // live inline entries (<= cap)
    std::uint32_t overflow_seen;                  // legacy raw-insert drops
    std::uint32_t promoted;                       // 1: all values in `spill`
    std::uint32_t _pad;
    std::uint8_t values[k_distinct_cell_cap * 16];   // 64 * 16 B
    DistinctSetUtf8* spill;                       // content set; lazily created

    BOLT_FORCE_INLINE void init() noexcept {
        n = 0; overflow_seen = 0; promoted = 0; _pad = 0; spill = nullptr;
    }

    BOLT_FORCE_INLINE bool saturated() const noexcept {
        return n >= k_distinct_cell_cap;
    }

    // Raw 16-byte insert — exact only for canonical inline values. Kept for
    // callers that dedup fixed 16-byte cells; saturates (drops) past cap.
    BOLT_FORCE_INLINE bool insert(const std::uint8_t value[16]) noexcept {
        assert(value != nullptr);
        assert(n <= k_distinct_cell_cap);
        const std::uint32_t cur = n;
        for (std::uint32_t i = 0; i < cur; ++i) {
            if (std::memcmp(&values[i * 16], value, 16) == 0) return false;  // sv-memcmp-ok: caller-canonical cells
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

    BOLT_FORCE_INLINE bool ensure_spill(bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        if (spill == nullptr) spill = DistinctSetUtf8::create(8, arena, 256, 8);
        return spill != nullptr;
    }

    // Moves every inline value into the content set. False on arena OOM.
    bool promote(bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(promoted == 0 && n == k_distinct_cell_cap);
        if (!ensure_spill(arena)) return false;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t len = 0;
            std::memcpy(&len, &values[i * 16], 4);
            assert(len <= 12u);
            const char* b = reinterpret_cast<const char*>(&values[i * 16 + 4]);
            if (!spill->insert(b, len, arena)) return false;   // disjoint: only OOM
        }
        promoted = 1;
        return true;
    }

    // Content-exact insert of a Utf8 value whose spilled bytes resolve through
    // `base`. Returns 1 if new, 0 if already present, -1 on arena OOM.
    int insert_sv(const StringView& v, const char* base,
                  bolt::Arena* arena) noexcept {
        assert(arena != nullptr);
        assert(n <= k_distinct_cell_cap);
        if (v.length > 12u || promoted != 0) {
            assert(v.length <= 12u || base != nullptr);
            if (!ensure_spill(arena)) return -1;
            const char* bytes = (v.length <= 12u) ? v.prefix : base + v.ref.offset;
            return insert_bytes(bytes, v.length, arena);
        }
        std::uint8_t canon[16] = {};
        std::memcpy(canon, &v.length, 4);
        std::memcpy(canon + 4, v.prefix, v.length);   // prefix+inline_data contiguous
        for (std::uint32_t i = 0; i < n; ++i) {
            if (std::memcmp(&values[i * 16], canon, 16) == 0) return 0;  // sv-memcmp-ok: canonical inline view
        }
        if (n < k_distinct_cell_cap) {
            std::memcpy(&values[n * 16], canon, 16);
            n += 1;
            return 1;
        }
        if (!promote(arena)) return -1;
        return insert_bytes(v.prefix, v.length, arena);
    }

    // Content-exact insert of a fixed 16-byte value (Decimal128). The inline
    // store holds the raw cells; past the cap they move into the content set
    // as 16-byte strings. A cell carries one agg, so it never mixes this with
    // insert_sv. 1 new, 0 duplicate, -1 arena exhausted.
    int insert_raw16(const std::uint8_t value[16], bolt::Arena* arena) noexcept {
        assert(value != nullptr && arena != nullptr);
        assert(n <= k_distinct_cell_cap);
        const char* bytes = reinterpret_cast<const char*>(value);
        if (promoted != 0) return insert_bytes(bytes, 16, arena);
        for (std::uint32_t i = 0; i < n; ++i) {
            if (std::memcmp(&values[i * 16], value, 16) == 0) return 0;  // sv-memcmp-ok: fixed Decimal128
        }
        if (n < k_distinct_cell_cap) {
            std::memcpy(&values[n * 16], value, 16);
            n += 1;
            return 1;
        }
        if (!ensure_spill(arena)) return -1;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (insert_bytes(reinterpret_cast<const char*>(&values[i * 16]), 16,
                             arena) != 1) return -1;
        }
        promoted = 1;
        return insert_bytes(bytes, 16, arena);
    }

private:
    // DistinctSetUtf8::insert reports OOM as "not inserted"; a miss that is
    // not a duplicate is OOM.
    int insert_bytes(const char* bytes, std::uint32_t len,
                     bolt::Arena* arena) noexcept {
        assert(spill != nullptr);
        assert(bytes != nullptr || len == 0);
        if (spill->insert(bytes, len, arena)) return 1;
        return spill->contains(bytes, len) ? 0 : -1;
    }
};

static_assert(sizeof(DistinctCell16) == 16 + k_distinct_cell_cap * 16 + sizeof(void*),
              "DistinctCell16 layout — 16B header + N*16B values + spill ptr");

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

    // G2CHK-92: `assert(g < entry_cap)` (and the `j < n_aggs` assert above
    // it) used to be the ONLY guard between an out-of-range (j, g) and a
    // pointer the caller then WRITES THROUGH — absent from any -DNDEBUG
    // build, an out-of-bounds write. Audited: no production caller in
    // bolt, marbledb, or chukonu calls `cell_at` today (the hash-agg
    // operator indexes `distinct_cells[j * entry_cap + g]` directly rather
    // than through this helper — see bolt_groupby.h), so there is nothing
    // to break by making the contract real rather than asserted. The check
    // is one comparison per call, not per row, so there is no perf reason
    // to leave it assert-only. Returns nullptr on an out-of-range index;
    // callers MUST check before dereferencing.
    BOLT_FORCE_INLINE DistinctCell* cell_at(std::uint16_t j,
                                            std::uint32_t g) noexcept {
        assert(cells != nullptr);
        assert(j < n_aggs);
        assert(g < entry_cap);
        if (cells == nullptr || j >= n_aggs || g >= entry_cap) return nullptr;
        return &cells[static_cast<std::size_t>(j) * entry_cap + g];
    }
};

}  // namespace bolt
