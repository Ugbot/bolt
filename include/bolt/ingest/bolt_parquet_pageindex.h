// bolt/ingest/bolt_parquet_pageindex.h — Parquet PAGE INDEX reader +
// page-granular pruning helpers (G2FEAT-23, bolt half).
//
// The page index (ColumnIndex + OffsetIndex, parquet-format PageIndex.md)
// lives OUTSIDE the footer FileMetaData, located via the ColumnChunk's
// column_index_offset/length + offset_index_offset/length fields that
// bolt_parquet_meta.h now captures on PqChunk. Per column chunk:
//
//   ColumnIndex  — per-PAGE min/max value bytes, null_pages flags,
//                  null_counts, boundary_order. Thrift compact struct:
//                    1: list<bool>   null_pages
//                    2: list<binary> min_values
//                    3: list<binary> max_values
//                    4: i32          boundary_order (0 UNORD / 1 ASC / 2 DESC)
//                    5: list<i64>    null_counts        (optional)
//                    6/7: level histograms              (skipped)
//   OffsetIndex  — per-page file offset + compressed size + first row:
//                    1: list<PageLocation> page_locations
//                       PageLocation { 1: i64 offset,
//                                      2: i32 compressed_page_size,
//                                      3: i64 first_row_index }
//                    2: list<i64> unencoded_byte_array_data_bytes (skipped)
//
// Trust rules mirror bolt_parquet_stats.h (G2FEAT-21) with ONE difference:
// ColumnIndex min/max are DEFINED against the column's logical sort order
// (there is no legacy/ambiguous variant of the page index), so Utf8 page
// bounds are trusted as written — writer-side truncation of page bounds is
// bound-preserving by spec, and bolt still drops >kPqMaxStatBytes values
// outright (kept-stat = writer's full bytes, never self-truncated).
// Everything stays conservative: absent / oversize / malformed stats or a
// null page make "excludes" return false ("cannot prove — keep the page").
// NULL semantics remain the caller's job: apply the pq_chunk_no_nulls()
// gate before treating "stats-disjoint" as "filter keeps zero rows" (the
// engine's INT64_MIN null-sentinel bridge, see G2FEAT-21).
//
// Bounded everything: at most kPqMaxPagesPerChunk pages per chunk — a
// chunk with more pages fails the parse and the caller falls back to
// rowgroup-level pruning (perf loss, never a correctness loss). The
// caller supplies the page arrays (arena-allocate; PqPageStat is 152 B so
// a full-cap array is ~608 KiB of scratch).
//
// Tiger Style: POD outputs, noexcept, no allocation, >=2 asserts per
// non-trivial function, bounded loops; hostile input returns false,
// never UB (every read bounds-checked against the supplied slice).

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_stats.h"

namespace bolt {
namespace ingest {
namespace parquet {

// Cap on pages per column chunk. parquet-mr's default page limit is
// 20,000 rows -> 4096 pages cover an 80M-row rowgroup; pyarrow's default
// ~1 MiB pages sit far under this. Oversize => parse fails => caller
// falls back to chunk-level stats.
inline constexpr uint32_t kPqMaxPagesPerChunk = 4096;

// One page's ColumnIndex entry. Same stat-byte convention as PqChunk:
// a >kPqMaxStatBytes bound is recorded ABSENT (len 0), never truncated.
struct PqPageStat {
    uint8_t  min_bytes[kPqMaxStatBytes];
    uint8_t  max_bytes[kPqMaxStatBytes];
    uint32_t min_len;                 // 0 = absent
    uint32_t max_len;                 // 0 = absent
    int64_t  null_count;              // -1 = absent (no null_counts list)
    uint8_t  null_page;               // 1 = page is entirely null
    uint8_t  _pad[7];
};
static_assert(sizeof(PqPageStat) == 152, "PqPageStat layout");

// Parsed ColumnIndex for one chunk. `pages`/`pages_cap` are caller
// storage (set before the parse), like PqMeta::chunks.
struct PqColumnIndex {
    PqPageStat* pages;
    uint32_t    pages_cap;
    uint32_t    n_pages;
    int32_t     boundary_order;       // 0 unordered / 1 asc / 2 desc
    uint8_t     has_null_counts;
    uint8_t     _pad[3];
};

// One page's OffsetIndex entry.
struct PqPageLocation {
    int64_t offset;                   // absolute file offset of page header
    int64_t first_row_index;          // first row of page, rowgroup-relative
    int32_t compressed_page_size;
    int32_t _pad;
};

struct PqOffsetIndex {
    PqPageLocation* pages;            // caller storage
    uint32_t        pages_cap;
    uint32_t        n_pages;
};

// Parse one ColumnIndex thrift struct from `buf[0..len)`. `out->pages` /
// `pages_cap` must be set beforehand. Validates: list lengths agree
// (null_pages / min_values / max_values / null_counts-if-present all equal),
// n_pages in [1, min(pages_cap, kPqMaxPagesPerChunk)]. Returns false on
// any corrupt/oversize shape — never UB on hostile input.
bool pq_parse_column_index(const uint8_t* buf, uint32_t len,
                           PqColumnIndex* out) noexcept;

// Parse one OffsetIndex thrift struct from `buf[0..len)`. Validates:
// first_row_index[0] == 0, first_row_index strictly increasing, offsets
// positive and increasing, compressed_page_size > 0.
bool pq_parse_offset_index(const uint8_t* buf, uint32_t len,
                           PqOffsetIndex* out) noexcept;

// Convenience: locate the chunk's index slice inside the whole-file buffer
// (bounds-checked against file_len) and parse. False when the chunk has no
// page index (offset/length absent) or the slice escapes the file.
bool pq_read_column_index(const uint8_t* file, uint64_t file_len,
                          const PqChunk& ch, PqColumnIndex* out) noexcept;
bool pq_read_offset_index(const uint8_t* file, uint64_t file_len,
                          const PqChunk& ch, PqOffsetIndex* out) noexcept;

// ---- evaluation helpers (mirror bolt_parquet_stats.h) ----------------------

// Decode one page's min/max as a signed int64 range (INT64 8-byte LE /
// INT32 4-byte LE) — sign-extended for signed columns, ZERO-extended for
// unsigned ones, exactly as pq_stat_range_i64 does for the chunk level
// (G2FEAT-111; see that function for why the reversed-pair guard makes the
// unsigned arm safe against legacy signed-ordered statistics). False when
// absent, wrong width, a null page, or a reversed pair — caller must then
// keep the page.
inline bool pq_page_range_i64(const PqColumn& col, const PqPageStat& ps,
                              int64_t* lo, int64_t* hi) noexcept {
    assert(lo != nullptr && hi != nullptr);
    if (ps.null_page != 0) return false;
    const bool uns = pq_col_is_unsigned_int(col);
    switch (col.physical) {
        case PqType::Int64: {
            if (ps.min_len != 8u || ps.max_len != 8u) return false;
            std::memcpy(lo, ps.min_bytes, 8);
            std::memcpy(hi, ps.max_bytes, 8);
            if (uns && (*lo < 0 || *hi < 0)) return false;   // u64 >= 2^63
            break;
        }
        case PqType::Int32: {
            if (ps.min_len != 4u || ps.max_len != 4u) return false;
            int32_t lo32 = 0, hi32 = 0;
            std::memcpy(&lo32, ps.min_bytes, 4);
            std::memcpy(&hi32, ps.max_bytes, 4);
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
            return false;
    }
    return *lo <= *hi;
}

// Can this page's bounds PROVE no value falls in [q_lo, q_hi]? A null
// page never "excludes" (the engine's null-sentinel bridge — see header
// comment); the caller's pq_chunk_no_nulls gate makes null pages moot.
inline bool pq_page_excludes_range_i64(const PqColumn& col,
                                       const PqPageStat& ps,
                                       int64_t q_lo, int64_t q_hi) noexcept {
    assert(q_lo <= q_hi);
    int64_t lo = 0, hi = 0;
    if (!pq_page_range_i64(col, ps, &lo, &hi)) return false;
    return q_hi < lo || q_lo > hi;
}

// Can this page's bounds PROVE no value equals `q` byte-wise? Unlike the
// chunk-level helper there is no legacy-field provenance to distrust —
// ColumnIndex bounds are order-aware by definition.
inline bool pq_page_utf8_excludes_eq(const PqPageStat& ps,
                                     const uint8_t* q,
                                     uint32_t q_len) noexcept {
    assert(q != nullptr || q_len == 0);
    if (ps.null_page != 0) return false;
    if (ps.min_len == 0u || ps.max_len == 0u) return false;     // absent
    if (pq_bytes_lex_cmp(ps.min_bytes, ps.min_len,
                         ps.max_bytes, ps.max_len) > 0) {
        return false;                                           // reversed
    }
    if (pq_bytes_lex_cmp(q, q_len, ps.min_bytes, ps.min_len) < 0) return true;
    if (pq_bytes_lex_cmp(q, q_len, ps.max_bytes, ps.max_len) > 0) return true;
    return false;
}

// Surviving-page set: bit i set => page i must be read.
struct PqPageSet {
    uint64_t bits[kPqMaxPagesPerChunk / 64];
    uint32_t n_pages;
    uint32_t n_survivors;
};

// Prune pages against an inclusive [q_lo, q_hi] range predicate. Every
// page starts surviving; a page is cleared only when its bounds PROVE
// exclusion. Returns the survivor count (== out->n_survivors).
uint32_t pq_prune_pages_i64(const PqColumn& col, const PqColumnIndex& ci,
                            int64_t q_lo, int64_t q_hi,
                            PqPageSet* out) noexcept;

// Prune pages against a byte-wise equality predicate on a BYTE_ARRAY/Utf8
// column. Same conservative contract.
uint32_t pq_prune_pages_utf8_eq(const PqColumnIndex& ci,
                                const uint8_t* q, uint32_t q_len,
                                PqPageSet* out) noexcept;

// A contiguous run of rows to read (rowgroup-relative), from coalescing
// adjacent surviving pages via the OffsetIndex row locations.
struct PqRowRange {
    int64_t first_row;
    int64_t row_count;
};

// Convert a survivor set into coalesced row ranges. `oi.n_pages` must
// equal `s.n_pages` (the ColumnIndex/OffsetIndex pair of one chunk) and
// `rg_num_rows` bounds the final page. Worst case emits ceil(n_pages/2)
// ranges — pass out_cap >= (n_pages + 1) / 2 and overflow is impossible;
// an insufficient out_cap returns 0 (assert in debug). Returns the range
// count (0 also when nothing survives — check s.n_survivors).
uint32_t pq_surviving_row_ranges(const PqOffsetIndex& oi, const PqPageSet& s,
                                 int64_t rg_num_rows,
                                 PqRowRange* out, uint32_t out_cap) noexcept;

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
