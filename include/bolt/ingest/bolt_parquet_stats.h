// bolt/ingest/bolt_parquet_stats.h — rowgroup statistics decode + pruning
// predicates over PqChunk min/max/null_count (G2FEAT-21).
//
// A scan may skip a rowgroup's decode entirely when the chunk statistics
// PROVE no row can satisfy a predicate. Everything here is conservative:
// any absent / malformed / untrusted stat makes the "excludes" predicate
// return false ("cannot prove absence — decode the rowgroup"), so a skip
// can never drop a matching row.
//
// Trust rules (documented against the parquet-format spec):
//   * INT32 / INT64 stats are fixed-width PLAIN little-endian payloads.
//     Signed order is the one defined sort order for these physical types,
//     so both the modern min_value/max_value fields AND the legacy min/max
//     fields are trusted (DuckDB applies the same rule). A reversed pair
//     (min > max) is rejected as corrupt.
//   * BYTE_ARRAY (Utf8) stats are trusted ONLY when they came from the
//     modern order-aware min_value/max_value fields (PqChunk::stats_flags
//     carries the provenance) — legacy byte-array min/max were written
//     with inconsistent orderings by old writers. Writer-side truncation
//     of the modern fields is bound-preserving by spec (min_value <= all
//     values, max_value >= all values), and bolt drops >kPqMaxStatBytes
//     stats outright at parse (copy_stat), so a present modern stat is a
//     valid bound as-is.
//   * NULL handling is the caller's job: pq_chunk_no_nulls() says whether
//     every row of the chunk is a real (non-null) value, which is the
//     precondition for equating "stats exclude the predicate" with "the
//     engine's filter would keep zero rows of this rowgroup".
//
// Tiger Style: header-only inline, noexcept, no allocation, >=2 asserts
// per non-trivial function, bounded loops.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {
namespace ingest {
namespace parquet {

// Every row of this chunk is a real value (no nulls) — either the writer
// recorded null_count == 0, or the column is REQUIRED (optional == 0), in
// which case nulls are structurally impossible even when the writer left
// null_count absent (-1). bolt's own writer does exactly that for
// non-nullable columns.
inline bool pq_chunk_no_nulls(const PqColumn& col, const PqChunk& ch) noexcept {
    return ch.null_count == 0 || col.optional == 0;
}

// True when this column's INT32/INT64 payload is an UNSIGNED integer, i.e.
// its statistics bytes must be ZERO-extended rather than sign-extended
// (G2FEAT-111). This has to match parquet_map_type/init_col_ctx's extension
// policy exactly: a zone-map range decoded on a different convention than
// the column values is worse than no zone map at all — it prunes rowgroups
// that do contain matches.
inline bool pq_col_is_unsigned_int(const PqColumn& col) noexcept {
    return col.logical == static_cast<int32_t>(PqLogical::Int) &&
           col.int_signed == 0;
}

// Decode the chunk's min/max statistics as a signed int64 range. Handles
// physical INT64 (8-byte LE) and INT32 (4-byte LE) — sign-extended for
// signed columns, ZERO-extended for unsigned ones, matching how the read
// path materialises the values. Returns false when stats are absent, the
// payload width is wrong for the physical type, or the pair is reversed
// (corrupt / untrusted) — callers must then treat the rowgroup as "may
// contain anything".
//
// The reversed-pair guard is what makes the unsigned arm safe against
// LEGACY statistics: the pre-`min_value` min/max fields were written with
// SIGNED ordering even for unsigned columns, so a chunk whose values
// straddle 2^31 records a "min" that is the larger unsigned value. Zero-
// extending that yields lo > hi and the range is rejected rather than
// trusted. Chunks entirely below or entirely above 2^31 order identically
// under both conventions, so they stay prunable.
//
// u64 (INT64 physical, unsigned) above 2^63-1 still has no lossless int64
// representation; it is rejected here rather than silently compared as a
// negative, so those rowgroups are simply never pruned.
inline bool pq_stat_range_i64(const PqColumn& col, const PqChunk& ch,
                              int64_t* lo, int64_t* hi) noexcept {
    assert(lo != nullptr && hi != nullptr);
    const bool uns = pq_col_is_unsigned_int(col);
    switch (col.physical) {
        case PqType::Int64: {
            if (ch.min_len != 8u || ch.max_len != 8u) return false;
            std::memcpy(lo, ch.min_bytes, 8);
            std::memcpy(hi, ch.max_bytes, 8);
            // Unsigned 64: no wider signed lane. Only trust the range when
            // both ends are non-negative, i.e. genuinely below 2^63.
            if (uns && (*lo < 0 || *hi < 0)) return false;
            break;
        }
        case PqType::Int32: {
            if (ch.min_len != 4u || ch.max_len != 4u) return false;
            int32_t lo32 = 0, hi32 = 0;
            std::memcpy(&lo32, ch.min_bytes, 4);
            std::memcpy(&hi32, ch.max_bytes, 4);
            if (uns) {
                *lo = static_cast<int64_t>(static_cast<uint32_t>(lo32));
                *hi = static_cast<int64_t>(static_cast<uint32_t>(hi32));
            } else {
                *lo = static_cast<int64_t>(lo32);
                *hi = static_cast<int64_t>(hi32);
            }
            break;
        }
        default:
            return false;   // strings/floats/bools: not an i64 range
    }
    return *lo <= *hi;      // reversed pair => corrupt, never prune on it
}

// Unsigned byte-wise lexicographic compare (the UTF8/BYTE_ARRAY sort order
// the modern min_value/max_value fields are defined against). Shorter
// string that is a prefix of the longer compares less.
inline int pq_bytes_lex_cmp(const uint8_t* a, uint32_t a_len,
                            const uint8_t* b, uint32_t b_len) noexcept {
    assert(a != nullptr || a_len == 0);
    assert(b != nullptr || b_len == 0);
    const uint32_t n = (a_len < b_len) ? a_len : b_len;
    const int c = (n != 0) ? std::memcmp(a, b, n) : 0;
    if (c != 0) return c;
    return (a_len < b_len) ? -1 : (a_len > b_len ? 1 : 0);
}

// Can the chunk statistics PROVE that no value equals `q` (byte-wise)?
// True only when q lies strictly outside [min_value, max_value] AND both
// bounds came from the modern order-aware fields (legacy byte-array stats
// are never trusted — see the header comment). NULL semantics are the
// caller's job (pq_chunk_no_nulls).
inline bool pq_stat_utf8_excludes_eq(const PqChunk& ch,
                                     const uint8_t* q,
                                     uint32_t q_len) noexcept {
    assert(q != nullptr || q_len == 0);
    if (ch.min_len == 0u || ch.max_len == 0u) return false;      // absent
    if ((ch.stats_flags & kPqStatMinIsValueField) == 0u) return false;
    if ((ch.stats_flags & kPqStatMaxIsValueField) == 0u) return false;
    if (pq_bytes_lex_cmp(ch.min_bytes, ch.min_len,
                         ch.max_bytes, ch.max_len) > 0) {
        return false;                                            // reversed
    }
    if (pq_bytes_lex_cmp(q, q_len, ch.min_bytes, ch.min_len) < 0) return true;
    if (pq_bytes_lex_cmp(q, q_len, ch.max_bytes, ch.max_len) > 0) return true;
    return false;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
