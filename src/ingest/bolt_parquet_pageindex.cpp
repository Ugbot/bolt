// bolt_parquet_pageindex.cpp — Parquet page-index (ColumnIndex/OffsetIndex)
// reader + page pruning drivers. See bolt_parquet_pageindex.h for the
// scope + safety contract (G2FEAT-23, bolt half).

#include "bolt/ingest/bolt_parquet_pageindex.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {

// Same oversize-is-absent stat copy rule as the footer parser (copy_stat
// in bolt_parquet_meta.cpp): a kept stat is always the writer's full
// bytes, never a self-truncation.
void copy_page_stat(const uint8_t* p, uint32_t n, uint8_t* dst,
                    uint32_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr);
    if (n == 0 || n > kPqMaxStatBytes) { *out_len = 0; return; }
    std::memcpy(dst, p, n);
    *out_len = n;
}

// list<bool> element (thrift compact writes one byte per collection bool:
// 1 = true, 2 = false — the field-type codes reused as values).
bool tc_bool_elem(TcCursor* c, bool* out) noexcept {
    assert(c != nullptr && out != nullptr);
    uint8_t b;
    if (!tc_u8(c, &b)) return false;
    if (b == 1u) { *out = true;  return true; }
    if (b == 2u || b == 0u) { *out = false; return true; }
    return false;   // anything else: corrupt
}

// Effective page cap for a parse: min(caller cap, compile-time cap).
uint32_t page_cap(uint32_t pages_cap) noexcept {
    return (pages_cap < kPqMaxPagesPerChunk) ? pages_cap
                                             : kPqMaxPagesPerChunk;
}

// ---- ColumnIndex field parsers (each bounded by out->pages_cap) -----------

bool ci_parse_null_pages(TcCursor* c, PqColumnIndex* out) noexcept {
    assert(c != nullptr && out != nullptr);
    uint8_t et; uint32_t n;
    if (!tc_list(c, &et, &n)) return false;
    if (et != kTcTrue && et != kTcFalse) return false;
    if (n == 0 || n > page_cap(out->pages_cap)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        bool v = false;
        if (!tc_bool_elem(c, &v)) return false;
        out->pages[i].null_page = v ? 1u : 0u;
    }
    out->n_pages = n;
    return true;
}

bool ci_parse_values(TcCursor* c, PqColumnIndex* out, bool is_min) noexcept {
    assert(c != nullptr && out != nullptr);
    uint8_t et; uint32_t n;
    if (!tc_list(c, &et, &n)) return false;
    if (et != kTcBinary) return false;
    if (n == 0 || n > page_cap(out->pages_cap)) return false;
    if (out->n_pages != 0 && n != out->n_pages) return false;   // must agree
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* p; uint32_t len;
        if (!tc_binary(c, &p, &len)) return false;
        PqPageStat* ps = &out->pages[i];
        if (is_min) copy_page_stat(p, len, ps->min_bytes, &ps->min_len);
        else        copy_page_stat(p, len, ps->max_bytes, &ps->max_len);
    }
    if (out->n_pages == 0) out->n_pages = n;
    return true;
}

bool ci_parse_null_counts(TcCursor* c, PqColumnIndex* out) noexcept {
    assert(c != nullptr && out != nullptr);
    uint8_t et; uint32_t n;
    if (!tc_list(c, &et, &n)) return false;
    if (et != kTcI64 && et != kTcI32 && et != kTcI16) return false;
    if (n == 0 || n > page_cap(out->pages_cap)) return false;
    if (out->n_pages != 0 && n != out->n_pages) return false;
    for (uint32_t i = 0; i < n; ++i) {
        int64_t v = 0;
        if (!tc_zigzag(c, &v)) return false;
        out->pages[i].null_count = v;
    }
    out->has_null_counts = 1;
    return true;
}

}  // namespace

bool pq_parse_column_index(const uint8_t* buf, uint32_t len,
                           PqColumnIndex* out) noexcept {
    assert(out != nullptr);
    if (buf == nullptr || len == 0) return false;
    if (out->pages == nullptr || out->pages_cap == 0) return false;
    out->n_pages = 0;
    out->boundary_order = 0;
    out->has_null_counts = 0;
    for (uint32_t i = 0; i < page_cap(out->pages_cap); ++i) {
        PqPageStat* ps = &out->pages[i];
        std::memset(ps, 0, sizeof(*ps));
        ps->null_count = -1;
    }
    TcCursor c{buf, buf + len};
    int16_t fid = 0;
    uint8_t ft;
    bool saw_null_pages = false, saw_min = false, saw_max = false;
    while (tc_field(&c, &fid, &ft)) {
        switch (fid) {
            case 1:   // null_pages: list<bool>
                if (!ci_parse_null_pages(&c, out)) return false;
                saw_null_pages = true;
                break;
            case 2:   // min_values: list<binary>
                if (!ci_parse_values(&c, out, /*is_min=*/true)) return false;
                saw_min = true;
                break;
            case 3:   // max_values: list<binary>
                if (!ci_parse_values(&c, out, /*is_min=*/false)) return false;
                saw_max = true;
                break;
            case 4: { // boundary_order: i32 enum
                int64_t v = 0;
                if (!tc_zigzag(&c, &v)) return false;
                if (v < 0 || v > 2) return false;
                out->boundary_order = static_cast<int32_t>(v);
                break;
            }
            case 5:   // null_counts: list<i64>
                if (!ci_parse_null_counts(&c, out)) return false;
                break;
            default:  // 6/7 level histograms + future fields: skip
                if (!tc_skip(&c, ft, 0)) return false;
                break;
        }
    }
    if (!saw_null_pages || !saw_min || !saw_max) return false;   // required
    if (out->n_pages == 0) return false;
    // A null page carries empty min/max by spec; a NON-null page with both
    // bounds absent is legal too (oversize) — nothing further to enforce.
    return true;
}

namespace {

// ---- OffsetIndex ------------------------------------------------------------
// PageLocation { 1: i64 offset, 2: i32 compressed_page_size,
//                3: i64 first_row_index }
bool parse_page_location(TcCursor* c, PqPageLocation* pl) noexcept {
    assert(c != nullptr && pl != nullptr);
    std::memset(pl, 0, sizeof(*pl));
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1:
                if (!tc_zigzag(c, &v)) return false;
                pl->offset = v;
                break;
            case 2:
                if (!tc_zigzag(c, &v)) return false;
                if (v <= 0 || v > 0x7FFFFFFF) return false;
                pl->compressed_page_size = static_cast<int32_t>(v);
                break;
            case 3:
                if (!tc_zigzag(c, &v)) return false;
                pl->first_row_index = v;
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

}  // namespace

bool pq_parse_offset_index(const uint8_t* buf, uint32_t len,
                           PqOffsetIndex* out) noexcept {
    assert(out != nullptr);
    if (buf == nullptr || len == 0) return false;
    if (out->pages == nullptr || out->pages_cap == 0) return false;
    out->n_pages = 0;
    TcCursor c{buf, buf + len};
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(&c, &fid, &ft)) {
        switch (fid) {
            case 1: { // page_locations: list<PageLocation>
                uint8_t et; uint32_t n;
                if (!tc_list(&c, &et, &n)) return false;
                if (et != kTcStruct) return false;
                if (n == 0 || n > page_cap(out->pages_cap)) return false;
                for (uint32_t i = 0; i < n; ++i) {
                    if (!parse_page_location(&c, &out->pages[i])) return false;
                }
                out->n_pages = n;
                break;
            }
            default:  // 2 unencoded_byte_array_data_bytes + future: skip
                if (!tc_skip(&c, ft, 0)) return false;
                break;
        }
    }
    if (out->n_pages == 0) return false;
    // Sanity: offsets positive/increasing, first_row_index starts at 0 and
    // is strictly increasing. Reject otherwise (corrupt index — the caller
    // then simply reads without page pruning).
    if (out->pages[0].first_row_index != 0) return false;
    for (uint32_t i = 0; i < out->n_pages; ++i) {
        if (out->pages[i].offset <= 0) return false;
        if (i > 0) {
            if (out->pages[i].first_row_index <=
                out->pages[i - 1].first_row_index) return false;
            if (out->pages[i].offset <= out->pages[i - 1].offset) return false;
        }
    }
    return true;
}

namespace {

// Bounds-check a chunk's index slice against the whole file.
bool index_slice(const uint8_t* file, uint64_t file_len,
                 int64_t off, int32_t len,
                 const uint8_t** buf, uint32_t* n) noexcept {
    assert(buf != nullptr && n != nullptr);
    if (file == nullptr || off <= 0 || len <= 0) return false;
    const uint64_t uoff = static_cast<uint64_t>(off);
    const uint64_t ulen = static_cast<uint64_t>(len);
    if (uoff >= file_len || ulen > file_len - uoff) return false;
    *buf = file + uoff;
    *n = static_cast<uint32_t>(ulen);
    return true;
}

}  // namespace

bool pq_read_column_index(const uint8_t* file, uint64_t file_len,
                          const PqChunk& ch, PqColumnIndex* out) noexcept {
    assert(out != nullptr);
    const uint8_t* buf = nullptr;
    uint32_t n = 0;
    if (!index_slice(file, file_len, ch.column_index_offset,
                     ch.column_index_length, &buf, &n)) return false;
    return pq_parse_column_index(buf, n, out);
}

bool pq_read_offset_index(const uint8_t* file, uint64_t file_len,
                          const PqChunk& ch, PqOffsetIndex* out) noexcept {
    assert(out != nullptr);
    const uint8_t* buf = nullptr;
    uint32_t n = 0;
    if (!index_slice(file, file_len, ch.offset_index_offset,
                     ch.offset_index_length, &buf, &n)) return false;
    return pq_parse_offset_index(buf, n, out);
}

// ---- pruning drivers --------------------------------------------------------

namespace {

void page_set_init(PqPageSet* out, uint32_t n_pages) noexcept {
    assert(out != nullptr && n_pages <= kPqMaxPagesPerChunk);
    std::memset(out->bits, 0, sizeof(out->bits));
    out->n_pages = n_pages;
    out->n_survivors = 0;
    for (uint32_t i = 0; i < n_pages; ++i) {          // all survive initially
        out->bits[i >> 6] |= (1ull << (i & 63u));
    }
    out->n_survivors = n_pages;
}

void page_set_clear(PqPageSet* out, uint32_t i) noexcept {
    assert(out != nullptr && i < out->n_pages);
    const uint64_t mask = 1ull << (i & 63u);
    if ((out->bits[i >> 6] & mask) != 0ull) {
        out->bits[i >> 6] &= ~mask;
        assert(out->n_survivors > 0);
        --out->n_survivors;
    }
}

bool page_set_test(const PqPageSet& s, uint32_t i) noexcept {
    assert(i < s.n_pages);
    return (s.bits[i >> 6] & (1ull << (i & 63u))) != 0ull;
}

}  // namespace

uint32_t pq_prune_pages_i64(const PqColumn& col, const PqColumnIndex& ci,
                            int64_t q_lo, int64_t q_hi,
                            PqPageSet* out) noexcept {
    assert(out != nullptr);
    assert(ci.n_pages > 0 && ci.n_pages <= kPqMaxPagesPerChunk);
    page_set_init(out, ci.n_pages);
    if (q_lo > q_hi) return out->n_survivors;         // empty query: keep all
    for (uint32_t i = 0; i < ci.n_pages; ++i) {
        if (pq_page_excludes_range_i64(col, ci.pages[i], q_lo, q_hi)) {
            page_set_clear(out, i);
        }
    }
    return out->n_survivors;
}

uint32_t pq_prune_pages_utf8_eq(const PqColumnIndex& ci,
                                const uint8_t* q, uint32_t q_len,
                                PqPageSet* out) noexcept {
    assert(out != nullptr);
    assert(ci.n_pages > 0 && ci.n_pages <= kPqMaxPagesPerChunk);
    page_set_init(out, ci.n_pages);
    for (uint32_t i = 0; i < ci.n_pages; ++i) {
        if (pq_page_utf8_excludes_eq(ci.pages[i], q, q_len)) {
            page_set_clear(out, i);
        }
    }
    return out->n_survivors;
}

uint32_t pq_surviving_row_ranges(const PqOffsetIndex& oi, const PqPageSet& s,
                                 int64_t rg_num_rows,
                                 PqRowRange* out, uint32_t out_cap) noexcept {
    assert(out != nullptr);
    assert(oi.n_pages == s.n_pages);
    if (oi.n_pages != s.n_pages || oi.n_pages == 0) return 0;
    if (rg_num_rows <= 0) return 0;
    if (out_cap < (s.n_pages + 1u) / 2u) {            // cannot overflow below
        assert(false && "out_cap too small for worst-case range count");
        return 0;
    }
    uint32_t n_ranges = 0;
    for (uint32_t i = 0; i < oi.n_pages; ++i) {
        if (!page_set_test(s, i)) continue;
        const int64_t first = oi.pages[i].first_row_index;
        const int64_t next = (i + 1 < oi.n_pages)
                                 ? oi.pages[i + 1].first_row_index
                                 : rg_num_rows;
        if (next <= first || first >= rg_num_rows) return 0;   // corrupt
        const int64_t count = ((next < rg_num_rows) ? next : rg_num_rows)
                              - first;
        if (n_ranges > 0 &&
            out[n_ranges - 1].first_row + out[n_ranges - 1].row_count ==
                first) {
            out[n_ranges - 1].row_count += count;      // coalesce adjacent
        } else {
            assert(n_ranges < out_cap);
            out[n_ranges].first_row = first;
            out[n_ranges].row_count = count;
            ++n_ranges;
        }
    }
    return n_ranges;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
