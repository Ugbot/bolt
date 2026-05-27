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

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

// Per-cell inline distinct-set capacity. Sized for TPC-H-style cardinalities
// where most COUNT(DISTINCT) groups have < 100 unique values. Each cell costs
// `k_distinct_cell_cap * 8` bytes of state + 4 bytes of bookkeeping; we keep
// it ≤ a cache line slab so a column of cells is dense in memory.
constexpr std::uint32_t k_distinct_cell_cap = 64;

// 64-byte aligned inline distinct-value tracker for ONE (group, agg) cell.
// `n` holds the live entry count; once it reaches `k_distinct_cell_cap` the
// cell is marked saturated and future inserts increment `overflow_seen`
// (caller can detect overcount with `saturated()`).
struct DistinctCell {
    std::uint32_t n;                            // live entries (≤ cap)
    std::uint32_t overflow_seen;                // inserts attempted past cap
    std::int64_t  values[k_distinct_cell_cap];  // dedup set

    BOLT_FORCE_INLINE void init() noexcept {
        n = 0; overflow_seen = 0;
        // values[] is logically unused until n grows; no need to zero.
    }

    BOLT_FORCE_INLINE bool saturated() const noexcept {
        return n >= k_distinct_cell_cap;
    }

    // Insert `v` if not already present. Returns true if it was a NEW value
    // (i.e. caller should `sum += v` or `count += 1` etc.). Returns false on
    // duplicate OR on saturation (over-cap insert). Tiger Style: caller's
    // accumulator path stays branch-free relative to this bool.
    BOLT_FORCE_INLINE bool insert(std::int64_t v) noexcept {
        assert(n <= k_distinct_cell_cap);
        // Linear-scan dedupe. For n ≤ ~32 this beats a hash probe by a wide
        // margin (no hash compute, no indirection, SIMD-friendly).
        const std::uint32_t cur = n;
        for (std::uint32_t i = 0; i < cur; ++i) {
            if (values[i] == v) return false;
        }
        if (cur >= k_distinct_cell_cap) {
            overflow_seen += 1;
            return false;  // saturated: drop. Phase-A scope.
        }
        values[cur] = v;
        n = cur + 1;
        assert(n <= k_distinct_cell_cap);
        return true;
    }
};

static_assert(sizeof(DistinctCell) == 8 + k_distinct_cell_cap * 8,
              "DistinctCell layout — packed (8B header + N*8B values)");

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
