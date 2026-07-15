// bolt_parquet_meta.cpp — W-PQ: Parquet footer metadata reader.
// See include/bolt/ingest/bolt_parquet_meta.h for scope + safety contract.

#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {
constexpr uint32_t kTcMaxDepth = 16;   // corrupt nesting cap
}

// Structurally skip one compact-protocol value. Bounded depth.
bool tc_skip(TcCursor* c, uint8_t type, uint32_t depth) noexcept {
    assert(c != nullptr);
    if (depth >= kTcMaxDepth) return false;
    switch (type) {
        case kTcTrue:
        case kTcFalse: return true;            // value lives in the type nibble
        case kTcByte: { uint8_t b; return tc_u8(c, &b); }
        case kTcI16:
        case kTcI32:
        case kTcI64: { int64_t v; return tc_zigzag(c, &v); }
        case kTcDouble: {
            if (c->end - c->p < 8) return false;
            c->p += 8;
            return true;
        }
        case kTcBinary: {
            const uint8_t* p; uint32_t n;
            return tc_binary(c, &p, &n);
        }
        case kTcList:
        case kTcSet: {
            uint8_t et; uint32_t n;
            if (!tc_list(c, &et, &n)) return false;
            for (uint32_t i = 0; i < n; ++i) {
                if (!tc_skip(c, et, depth + 1)) return false;
            }
            return true;
        }
        case kTcMap: {
            uint8_t b;
            if (!tc_u8(c, &b)) return false;     // size byte or varint+kv
            uint32_t n = b >> 4;                 // compact maps: size first
            uint8_t kv = 0;
            if (b == 0) return true;             // empty map encodes as 0
            // Non-empty: byte was the first varint byte of size; re-parse.
            // (Compact map header = varint size, then 1 byte kv-types.)
            c->p -= 1;
            uint64_t sz = 0;
            if (!tc_varint(c, &sz) || sz > 0xFFFFFF) return false;
            if (!tc_u8(c, &kv)) return false;
            n = static_cast<uint32_t>(sz);
            const uint8_t kt = kv >> 4, vt = kv & 0x0Fu;
            for (uint32_t i = 0; i < n; ++i) {
                if (!tc_skip(c, kt, depth + 1)) return false;
                if (!tc_skip(c, vt, depth + 1)) return false;
            }
            return true;
        }
        case kTcStruct: {
            int16_t fid = 0;
            uint8_t ft;
            // Bounded: each tc_field consumes >=1 byte from a finite slice.
            while (tc_field(c, &fid, &ft)) {
                if (!tc_skip(c, ft, depth + 1)) return false;
            }
            return true;   // tc_field returned false at STOP (or slice end)
        }
        default:
            return false;
    }
}

namespace {

// Copy a thrift binary into a bounded NUL-terminated char field.
void copy_name(const uint8_t* p, uint32_t n, char* dst,
               uint32_t cap) noexcept {
    assert(dst != nullptr && cap > 0);
    const uint32_t w = (n < cap - 1) ? n : cap - 1;
    std::memcpy(dst, p, w);
    dst[w] = '\0';
}

// Copy statistics value bytes into a bounded buffer (oversize = absent —
// a >64-byte min/max is a string we will not use for pruning anyway).
void copy_stat(const uint8_t* p, uint32_t n, uint8_t* dst,
               uint32_t* out_len) noexcept {
    assert(dst != nullptr && out_len != nullptr);
    if (n == 0 || n > kPqMaxStatBytes) { *out_len = 0; return; }
    std::memcpy(dst, p, n);
    *out_len = n;
}

// ---- SchemaElement ---------------------------------------------------------
// fields: 1 type(i32) 2 type_length 3 repetition 4 name 5 num_children
//         6 converted_type 7 scale 8 precision (9 field_id, 10 logicalType: skipped)
struct SchemaElem {
    PqColumn col;
    int32_t  num_children;   // >0 => group node (root or nested)
    bool     has_type;
};

bool parse_schema_element(TcCursor* c, SchemaElem* out) noexcept {
    assert(c != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->col.converted = -1;
    out->num_children = 0;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1:
                if (!tc_zigzag(c, &v)) return false;
                out->col.physical = static_cast<PqType>(v);
                out->has_type = true;
                break;
            case 2:
                if (!tc_zigzag(c, &v)) return false;
                out->col.type_length = static_cast<int32_t>(v);
                break;
            case 3:
                if (!tc_zigzag(c, &v)) return false;
                out->col.optional = (v == 1) ? 1 : 0;   // OPTIONAL
                break;
            case 4: {
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                copy_name(p, n, out->col.name, kPqMaxNameBytes);
                break;
            }
            case 5:
                if (!tc_zigzag(c, &v)) return false;
                out->num_children = static_cast<int32_t>(v);
                break;
            case 6:
                if (!tc_zigzag(c, &v)) return false;
                out->col.converted = static_cast<int32_t>(v);
                break;
            case 7:    // parquet.thrift SchemaElement: 7 = scale
                if (!tc_zigzag(c, &v)) return false;
                out->col.scale = static_cast<int32_t>(v);
                break;
            case 8:    // 8 = precision (9 field_id / 10 logicalType skip)
                if (!tc_zigzag(c, &v)) return false;
                out->col.precision = static_cast<int32_t>(v);
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

// ---- Statistics ------------------------------------------------------------
// fields: 1 max 2 min 3 null_count 4 distinct_count 5 max_value 6 min_value
bool parse_statistics(TcCursor* c, PqChunk* ch) noexcept {
    assert(c != nullptr && ch != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    bool have_new_min = false, have_new_max = false;
    while (tc_field(c, &fid, &ft)) {
        switch (fid) {
            case 1: {   // legacy max — use only if max_value absent
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                if (!have_new_max) copy_stat(p, n, ch->max_bytes, &ch->max_len);
                break;
            }
            case 2: {   // legacy min
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                if (!have_new_min) copy_stat(p, n, ch->min_bytes, &ch->min_len);
                break;
            }
            case 3: {
                int64_t v;
                if (!tc_zigzag(c, &v)) return false;
                ch->null_count = v;
                break;
            }
            case 5: {   // max_value (new, order-aware) — preferred
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                copy_stat(p, n, ch->max_bytes, &ch->max_len);
                have_new_max = true;
                // Flag only when actually kept (oversize => len 0 = absent).
                if (ch->max_len != 0) ch->stats_flags |= kPqStatMaxIsValueField;
                break;
            }
            case 6: {   // min_value
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                copy_stat(p, n, ch->min_bytes, &ch->min_len);
                have_new_min = true;
                if (ch->min_len != 0) ch->stats_flags |= kPqStatMinIsValueField;
                break;
            }
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

// ---- ColumnMetaData ---------------------------------------------------------
// fields: 1 type 2 encodings 3 path_in_schema 4 codec 5 num_values
//         6 total_uncompressed_size 7 total_compressed_size
//         9 data_page_offset 10 index_page_offset 11 dictionary_page_offset
//         12 statistics
bool parse_column_meta(TcCursor* c, PqChunk* ch) noexcept {
    assert(c != nullptr && ch != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 4:
                if (!tc_zigzag(c, &v)) return false;
                ch->codec = static_cast<PqCodec>(v);
                break;
            case 5:
                if (!tc_zigzag(c, &v)) return false;
                ch->num_values = v;
                break;
            case 6:
                if (!tc_zigzag(c, &v)) return false;
                ch->total_uncompressed_size = v;
                break;
            case 7:
                if (!tc_zigzag(c, &v)) return false;
                ch->total_compressed_size = v;
                break;
            case 9:
                if (!tc_zigzag(c, &v)) return false;
                ch->data_page_offset = v;
                break;
            case 11:
                if (!tc_zigzag(c, &v)) return false;
                ch->dictionary_page_offset = v;
                break;
            case 12:
                if (ft != kTcStruct) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_statistics(c, ch)) return false;
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

// ---- ColumnChunk -----------------------------------------------------------
// fields: 1 file_path 2 file_offset 3 meta_data
bool parse_column_chunk(TcCursor* c, PqChunk* ch) noexcept {
    assert(c != nullptr && ch != nullptr);
    std::memset(ch, 0, sizeof(*ch));
    ch->null_count = -1;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        switch (fid) {
            case 1: {
                const uint8_t* p; uint32_t n;
                if (!tc_binary(c, &p, &n)) return false;
                if (n != 0) return false;   // external files unsupported
                break;
            }
            case 3:
                if (ft != kTcStruct) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_column_meta(c, ch)) return false;
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

// ---- RowGroup ---------------------------------------------------------------
// fields: 1 columns(list<ColumnChunk>) 2 total_byte_size 3 num_rows
bool parse_row_group(TcCursor* c, PqMeta* m, PqRowGroup* rg) noexcept {
    assert(c != nullptr && m != nullptr && rg != nullptr);
    std::memset(rg, 0, sizeof(*rg));
    rg->chunk_off = m->n_chunks;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1: {
                uint8_t et; uint32_t n;
                if (!tc_list(c, &et, &n)) return false;
                if (et != kTcStruct) return false;
                if (n > kPqMaxColumns) return false;
                for (uint32_t i = 0; i < n; ++i) {
                    if (m->n_chunks >= m->chunks_cap) return false;
                    if (!parse_column_chunk(c, &m->chunks[m->n_chunks])) {
                        return false;
                    }
                    ++m->n_chunks;
                }
                rg->chunk_count = n;
                break;
            }
            case 2:
                if (!tc_zigzag(c, &v)) return false;
                rg->total_byte_size = v;
                break;
            case 3:
                if (!tc_zigzag(c, &v)) return false;
                rg->num_rows = v;
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

}  // namespace

bool pq_locate_footer(const uint8_t* buf, uint64_t len,
                      uint64_t* meta_off, uint32_t* meta_len) noexcept {
    assert(meta_off != nullptr && meta_len != nullptr);
    if (buf == nullptr || len < 12) return false;   // PAR1 + len + PAR1
    if (std::memcmp(buf, "PAR1", 4) != 0) return false;
    if (std::memcmp(buf + len - 4, "PAR1", 4) != 0) return false;
    uint32_t ml;
    std::memcpy(&ml, buf + len - 8, 4);             // little-endian
    if (ml == 0 || static_cast<uint64_t>(ml) + 8 > len - 4) return false;
    *meta_len = ml;
    *meta_off = len - 8 - ml;
    return true;
}

bool pq_parse_file_meta(const uint8_t* meta, uint32_t meta_len,
                        PqMeta* out) noexcept {
    assert(out != nullptr);
    if (meta == nullptr || meta_len == 0) return false;
    if (out->chunks == nullptr || out->chunks_cap == 0) return false;
    out->num_rows = 0;
    out->n_columns = 0;
    out->n_row_groups = 0;
    out->n_chunks = 0;
    out->version = 0;

    TcCursor c{meta, meta + meta_len};
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(&c, &fid, &ft)) {
        int64_t v = 0;
        switch (fid) {
            case 1:
                if (!tc_zigzag(&c, &v)) return false;
                out->version = static_cast<int32_t>(v);
                break;
            case 2: {   // schema: list<SchemaElement>; FLAT only
                uint8_t et; uint32_t n;
                if (!tc_list(&c, &et, &n)) return false;
                if (et != kTcStruct || n == 0 || n > kPqMaxColumns + 1) {
                    return false;
                }
                for (uint32_t i = 0; i < n; ++i) {
                    SchemaElem se;
                    if (!parse_schema_element(&c, &se)) return false;
                    if (i == 0) {
                        // Root group node. Children count must equal the
                        // remaining elements or the schema is NESTED.
                        if (se.num_children !=
                            static_cast<int32_t>(n - 1)) return false;
                        continue;
                    }
                    if (se.num_children != 0 || !se.has_type) {
                        return false;   // nested group: unsupported v1
                    }
                    if (out->n_columns >= kPqMaxColumns) return false;
                    out->columns[out->n_columns++] = se.col;
                }
                break;
            }
            case 3:
                if (!tc_zigzag(&c, &v)) return false;
                out->num_rows = v;
                break;
            case 4: {   // row_groups: list<RowGroup>
                uint8_t et; uint32_t n;
                if (!tc_list(&c, &et, &n)) return false;
                if (et != kTcStruct || n > kPqMaxRowGroups) return false;
                for (uint32_t i = 0; i < n; ++i) {
                    if (!parse_row_group(&c, out,
                                         &out->row_groups[i])) return false;
                }
                out->n_row_groups = n;
                break;
            }
            default:
                if (!tc_skip(&c, ft, 0)) return false;
                break;
        }
    }
    // Shape checks: every row group must carry one chunk per column.
    if (out->n_columns == 0) return false;
    for (uint32_t g = 0; g < out->n_row_groups; ++g) {
        if (out->row_groups[g].chunk_count != out->n_columns) return false;
    }
    return true;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
