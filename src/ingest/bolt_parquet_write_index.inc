// bolt_parquet_write_index.inc — ColumnIndex / OffsetIndex emission.
//
// Included ONCE from bolt_parquet_write.cpp, inside
// namespace bolt::ingest::parquet { namespace { ... } }, after the thrift
// codec, StatBuf/PageRec/ChunkRec and the ParquetWriter definition. A size
// seam, not a public header.
//
// Layout: both structs are written between the last row group's data and the
// FileMetaData footer, and located by ColumnChunk fields 4-7. That placement
// is what lets a reader fetch the whole index region in one range read
// without touching page data -- the entire point of the feature.
//
//   struct ColumnIndex {
//     1: required list<bool>   null_pages
//     2: required list<binary> min_values
//     3: required list<binary> max_values
//     4: required BoundaryOrder boundary_order
//     5: optional list<i64>    null_counts
//   }
//   struct OffsetIndex { 1: required list<PageLocation> page_locations }
//   struct PageLocation { 1: i64 offset, 2: i32 compressed_page_size,
//                         3: i64 first_row_index }
//
// Thrift compact detail that is easy to get wrong: a `list<bool>` element is
// NOT a 0/1 byte. Apache Thrift writes container booleans as the compact
// type codes themselves -- 1 for true, 2 for false -- and declares the list
// element type as 1. tc_put_bool_elem below is that rule in one place.

// ===== typed statistic comparison =========================================

// Compare two statistic payloads in the ORDER THE COLUMN'S TYPE DEFINES,
// which for every numeric type is NOT the lexicographic order of the
// little-endian bytes those payloads are stored in. Used only to decide
// boundary_order; getting it wrong would make a reader's binary search skip
// pages that do match, so this dispatches on the real type and declines
// (returns kStatCmpUnknown) for anything it cannot order.
constexpr int kStatCmpUnknown = 2;

template <typename T>
inline int stat_cmp_num(const std::uint8_t* a, const std::uint8_t* b) noexcept {
    assert(a != nullptr && b != nullptr);
    T x, y;
    std::memcpy(&x, a, sizeof(T));
    std::memcpy(&y, b, sizeof(T));
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

int stat_cmp(BoltType t, const std::uint8_t* a, std::uint32_t alen,
             const std::uint8_t* b, std::uint32_t blen) noexcept {
    assert(a != nullptr && b != nullptr);
    switch (t) {
        case BoltType::Int32:
        case BoltType::Date32:
            if (alen != 4u || blen != 4u) return kStatCmpUnknown;
            return stat_cmp_num<std::int32_t>(a, b);
        case BoltType::Int64:
        case BoltType::Timestamp:
            if (alen != 8u || blen != 8u) return kStatCmpUnknown;
            return stat_cmp_num<std::int64_t>(a, b);
        case BoltType::Float32:
            if (alen != 4u || blen != 4u) return kStatCmpUnknown;
            return stat_cmp_num<float>(a, b);
        case BoltType::Float64:
            if (alen != 8u || blen != 8u) return kStatCmpUnknown;
            return stat_cmp_num<double>(a, b);
        case BoltType::Bool:
            if (alen != 1u || blen != 1u) return kStatCmpUnknown;
            return (a[0] < b[0]) ? -1 : ((a[0] > b[0]) ? 1 : 0);
        case BoltType::Utf8:
        case BoltType::Binary:
            return stat_lex_cmp(a, alen, b, blen);
        default:
            // Decimal128 emits no min/max today, so this is unreachable for
            // any page that carries bounds; decline rather than guess.
            return kStatCmpUnknown;
    }
}

// BoundaryOrder enum (parquet.thrift).
constexpr std::int32_t kBoundaryUnordered = 0;
constexpr std::int32_t kBoundaryAscending = 1;
constexpr std::int32_t kBoundaryDescending = 2;

// Decide boundary_order across a chunk's pages. Only pages that actually
// carry bounds participate; a null page breaks nothing because it has no
// bounds to be out of order. Falls back to UNORDERED -- always correct, just
// unhelpful -- the moment anything is ambiguous.
std::int32_t boundary_order_for(const PageRec* pages, std::uint32_t n,
                                BoltType t) noexcept {
    assert(pages != nullptr || n == 0u);
    bool asc = true, desc = true, seen = false;
    const StatBuf* prev = nullptr;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (pages[i].all_null || !pages[i].st.have_stats) continue;
        const StatBuf& s = pages[i].st;
        if (prev != nullptr) {
            const int cmin = stat_cmp(t, s.min_buf, s.min_len,
                                      prev->min_buf, prev->min_len);
            const int cmax = stat_cmp(t, s.max_buf, s.max_len,
                                      prev->max_buf, prev->max_len);
            if (cmin == kStatCmpUnknown || cmax == kStatCmpUnknown) {
                return kBoundaryUnordered;
            }
            if (cmin < 0 || cmax < 0) asc = false;
            if (cmin > 0 || cmax > 0) desc = false;
        }
        prev = &s;
        seen = true;
    }
    if (!seen) return kBoundaryUnordered;
    // Both true means every bound compared equal; ASCENDING is the
    // conventional label for that and is a valid non-strict ordering.
    if (asc) return kBoundaryAscending;
    if (desc) return kBoundaryDescending;
    return kBoundaryUnordered;
}

// ===== thrift helpers specific to the index structs =======================

// Container booleans: the element IS the compact type code (1 true, 2 false).
inline void tc_put_bool_elem(TcOut* o, bool v) noexcept {
    assert(o != nullptr);
    tc_put_u8(o, v ? kFTrue : kFFalse);
}

// ===== ColumnIndex ========================================================

void build_column_index(std::vector<std::uint8_t>* dst,
                        const PageRec* pages, std::uint32_t n,
                        BoltType t) noexcept {
    assert(dst != nullptr);
    assert(pages != nullptr || n == 0u);
    dst->clear();
    TcOut o{dst};
    // 1 null_pages : list<bool>
    tc_put_field(&o, 1, kFList);
    tc_put_list_hdr(&o, kFTrue, n);
    for (std::uint32_t i = 0; i < n; ++i) tc_put_bool_elem(&o, pages[i].all_null);
    // 2 min_values, 3 max_values : list<binary>. A page with no usable
    // bounds (all-null, or a string longer than the stat buffer) contributes
    // an EMPTY binary, which is what the spec requires -- the lists are
    // required and must stay index-aligned with null_pages.
    tc_put_field(&o, 2, kFList);
    tc_put_list_hdr(&o, kFBinary, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool ok = pages[i].st.have_stats && !pages[i].all_null;
        tc_put_binary(&o, pages[i].st.min_buf, ok ? pages[i].st.min_len : 0u);
    }
    tc_put_field(&o, 3, kFList);
    tc_put_list_hdr(&o, kFBinary, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool ok = pages[i].st.have_stats && !pages[i].all_null;
        tc_put_binary(&o, pages[i].st.max_buf, ok ? pages[i].st.max_len : 0u);
    }
    // 4 boundary_order
    tc_put_field(&o, 4, kFI32);
    tc_put_zigzag(&o, boundary_order_for(pages, n, t));
    // 5 null_counts : list<i64>
    tc_put_field(&o, 5, kFList);
    tc_put_list_hdr(&o, kFI64, n);
    for (std::uint32_t i = 0; i < n; ++i) tc_put_zigzag(&o, pages[i].null_count);
    tc_put_stop(&o);
}

// ===== OffsetIndex ========================================================

void build_offset_index(std::vector<std::uint8_t>* dst,
                        const PageRec* pages, std::uint32_t n) noexcept {
    assert(dst != nullptr);
    assert(pages != nullptr || n == 0u);
    dst->clear();
    TcOut o{dst};
    tc_put_field(&o, 1, kFList);
    tc_put_list_hdr(&o, kFStruct, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        assert(pages[i].offset >= 0);
        assert(pages[i].comp_size >= 0);
        tc_put_field(&o, 1, kFI64);
        tc_put_zigzag(&o, pages[i].offset);
        tc_put_field(&o, 2, kFI32);
        tc_put_zigzag(&o, pages[i].comp_size);
        tc_put_field(&o, 3, kFI64);
        tc_put_zigzag(&o, pages[i].first_row);
        tc_put_stop(&o);
    }
    tc_put_stop(&o);
}

// ===== emission ===========================================================

// Write every chunk's ColumnIndex, then every chunk's OffsetIndex, recording
// where each landed. Called once, after the last row group and before the
// footer. Grouping by kind (all ColumnIndexes, then all OffsetIndexes) is
// what parquet-mr and Arrow do: a reader that only wants bounds fetches one
// contiguous region.
bool write_page_index(ParquetWriter* w) noexcept {
    assert(w != nullptr);
    if (!w->opts.emit_page_index) return true;
    if (w->pages.empty()) return true;
    std::vector<std::uint8_t> buf;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t g = 0; g < w->row_groups.size(); ++g) {
            const RowGroupRec& rg = w->row_groups[g];
            for (std::uint32_t c = 0; c < rg.chunk_count; ++c) {
                ChunkRec& rec = w->chunks[rg.chunk_off + c];
                if (rec.page_count == 0u) continue;
                // bolt's own index PARSER caps a chunk at
                // kPqMaxPagesPerChunk (bolt_parquet_pageindex.h) because the
                // page-stat array is caller-supplied scratch. Emitting an
                // index past that would produce a file bolt cannot read its
                // own index out of, so the chunk goes without one. Omission
                // is spec-legal (the index is optional per chunk) and costs
                // only the page-level pruning -- the data still reads. A
                // caller that wants the index for a chunk this large should
                // raise data_page_target_bytes or lower row_group_max_rows.
                if (rec.page_count > kPqMaxPagesPerChunk) continue;
                assert(rec.page_off + rec.page_count <= w->pages.size());
                const PageRec* pp = w->pages.data() + rec.page_off;
                if (pass == 0) {
                    build_column_index(&buf, pp, rec.page_count,
                                       w->opts.columns[c].type);
                } else {
                    build_offset_index(&buf, pp, rec.page_count);
                }
                if (buf.size() > 0x7FFFFFFFull) return false;
                const std::int64_t off = w->file_pos;
                if (!sink_write(w, buf.data(), buf.size())) return false;
                w->file_pos += static_cast<std::int64_t>(buf.size());
                if (pass == 0) {
                    rec.col_index_off = off;
                    rec.col_index_len = static_cast<std::int32_t>(buf.size());
                } else {
                    rec.off_index_off = off;
                    rec.off_index_len = static_cast<std::int32_t>(buf.size());
                }
            }
        }
    }
    return sink_ok(w);
}
