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

// ---- SchemaElement + LogicalType (G2FEAT-46) -------------------------------
// SchemaElement fields: 1 type(i32) 2 type_length 3 repetition 4 name
//   5 num_children 6 converted_type 7 scale 8 precision 9 field_id
//   10 logicalType(union).
struct SchemaElem {
    PqColumn col;
    int32_t  num_children;   // >0 => group node (root or nested)
    bool     has_type;
    bool     repeated;       // repetition_type == REPEATED (list/map element)
};

// ConvertedType numeric values (parquet.thrift ConvertedType enum).
constexpr int32_t kCtJson = 24, kCtBson = 25;
constexpr int32_t kCtDecimal = 5, kCtDate = 6, kCtTimeMillis = 7,
                  kCtTimeMicros = 8, kCtTsMillis = 9, kCtTsMicros = 10,
                  kCtUint8 = 11, kCtUint16 = 12, kCtUint32 = 13,
                  kCtUint64 = 14, kCtInt8 = 15, kCtInt16 = 16,
                  kCtInt32 = 17, kCtInt64 = 18;

// TimeUnit union (parquet.thrift): field id 1 MILLIS / 2 MICROS / 3 NANOS,
// each an empty struct. Returns 0 (none) when unrecognized/absent.
int32_t parse_time_unit(TcCursor* c) noexcept {
    assert(c != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    int32_t unit = 0;
    while (tc_field(c, &fid, &ft)) {           // bounded: finite slice
        if (fid >= 1 && fid <= 3) unit = fid;
        if (!tc_skip(c, ft, 0)) return unit;   // skip the empty struct value
    }
    return unit;
}

// TimestampType / TimeType struct: 1 isAdjustedToUTC(bool) 2 unit(TimeUnit).
void parse_ts_or_time(TcCursor* c, uint8_t* utc, int32_t* unit) noexcept {
    assert(c != nullptr && utc != nullptr && unit != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {           // bounded: finite slice
        if (fid == 1) {                        // bool value in the type nibble
            *utc = (ft == kTcTrue) ? 1 : 0;
        } else if (fid == 2 && ft == kTcStruct) {
            *unit = parse_time_unit(c);
        } else if (!tc_skip(c, ft, 0)) {
            return;
        }
    }
}

// IntType struct: 1 bitWidth(byte) 2 isSigned(bool).
void parse_int_type(TcCursor* c, uint8_t* bits, uint8_t* is_signed) noexcept {
    assert(c != nullptr && bits != nullptr && is_signed != nullptr);
    *is_signed = 1;                            // thrift: required, default safe
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {           // bounded: finite slice
        if (fid == 1 && ft == kTcByte) {
            uint8_t b = 0;
            if (tc_u8(c, &b)) *bits = b;
        } else if (fid == 2) {                 // bool value in the type nibble
            *is_signed = (ft == kTcTrue) ? 1 : 0;
        } else if (!tc_skip(c, ft, 0)) {
            return;
        }
    }
}

// DecimalType struct: 1 scale(i32) 2 precision(i32).
void parse_decimal_type(TcCursor* c, int32_t* scale, int32_t* prec) noexcept {
    assert(c != nullptr && scale != nullptr && prec != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {           // bounded: finite slice
        int64_t v = 0;
        if (fid == 1) { if (tc_zigzag(c, &v)) *scale = static_cast<int32_t>(v); }
        else if (fid == 2) { if (tc_zigzag(c, &v)) *prec = static_cast<int32_t>(v); }
        else if (!tc_skip(c, ft, 0)) return;
    }
}

// LogicalType union (SchemaElement field 10). The set field id selects the
// member: 5 DECIMAL, 6 DATE, 7 TIME, 8 TIMESTAMP, 10 INTEGER, 1 STRING.
void parse_logical_type(TcCursor* c, PqColumn* col) noexcept {
    assert(c != nullptr && col != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {           // bounded: union = one member
        if (ft != kTcStruct) { (void)tc_skip(c, ft, 0); continue; }
        switch (fid) {
            case 1: col->logical = static_cast<int32_t>(PqLogical::String);
                    (void)tc_skip(c, ft, 0); break;
            case 5: col->logical = static_cast<int32_t>(PqLogical::Decimal);
                    parse_decimal_type(c, &col->scale, &col->precision); break;
            case 6: col->logical = static_cast<int32_t>(PqLogical::Date);
                    (void)tc_skip(c, ft, 0); break;
            case 7: col->logical = static_cast<int32_t>(PqLogical::Time);
                    parse_ts_or_time(c, &col->ts_utc, &col->time_unit); break;
            case 8: col->logical = static_cast<int32_t>(PqLogical::Timestamp);
                    parse_ts_or_time(c, &col->ts_utc, &col->time_unit); break;
            case 10: {
                col->logical = static_cast<int32_t>(PqLogical::Int);
                uint8_t bits = 0, sgn = 1;
                parse_int_type(c, &bits, &sgn);
                col->int_bits = bits;
                col->int_signed = sgn;
                break;
            }
            // parquet.thrift LogicalType union ids: 12 JSON, 13 BSON,
            // 14 UUID, 16 VARIANT. All carry an empty struct.
            case 12: col->logical = static_cast<int32_t>(PqLogical::Json);
                     (void)tc_skip(c, ft, 0); break;
            case 13: col->logical = static_cast<int32_t>(PqLogical::Bson);
                     (void)tc_skip(c, ft, 0); break;
            case 16: col->logical = static_cast<int32_t>(PqLogical::Variant);
                     (void)tc_skip(c, ft, 0); break;
            default: (void)tc_skip(c, ft, 0); break;
        }
    }
}

// Fallback: fill the normalized logical fields from the legacy ConvertedType
// when SchemaElement had no LogicalType (field 10) — nanos have no
// ConvertedType, but millis/micros + un/signed ints do.
void derive_logical_from_converted(PqColumn* col) noexcept {
    assert(col != nullptr);
    switch (col->converted) {
        case kCtJson:       col->logical = static_cast<int32_t>(PqLogical::Json); break;
        case kCtBson:       col->logical = static_cast<int32_t>(PqLogical::Bson); break;
        case kCtTsMillis:   col->logical = static_cast<int32_t>(PqLogical::Timestamp); col->time_unit = 1; break;
        case kCtTsMicros:   col->logical = static_cast<int32_t>(PqLogical::Timestamp); col->time_unit = 2; break;
        case kCtTimeMillis: col->logical = static_cast<int32_t>(PqLogical::Time);      col->time_unit = 1; break;
        case kCtTimeMicros: col->logical = static_cast<int32_t>(PqLogical::Time);      col->time_unit = 2; break;
        case kCtUint8:  col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 8;  col->int_signed = 0; break;
        case kCtUint16: col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 16; col->int_signed = 0; break;
        case kCtUint32: col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 32; col->int_signed = 0; break;
        case kCtUint64: col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 64; col->int_signed = 0; break;
        case kCtInt8:   col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 8;  col->int_signed = 1; break;
        case kCtInt16:  col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 16; col->int_signed = 1; break;
        case kCtInt32:  col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 32; col->int_signed = 1; break;
        case kCtInt64:  col->logical = static_cast<int32_t>(PqLogical::Int); col->int_bits = 64; col->int_signed = 1; break;
        case kCtDecimal: col->logical = static_cast<int32_t>(PqLogical::Decimal); break;
        case kCtDate:    col->logical = static_cast<int32_t>(PqLogical::Date); break;
        default: break;
    }
}

bool parse_schema_element(TcCursor* c, SchemaElem* out) noexcept {
    assert(c != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->col.converted = -1;
    out->col.logical = static_cast<int32_t>(PqLogical::None);
    out->col.int_signed = 1;
    out->num_children = 0;
    out->repeated = false;
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
                out->repeated = (v == 2);                // REPEATED
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
            case 8:    // 8 = precision
                if (!tc_zigzag(c, &v)) return false;
                out->col.precision = static_cast<int32_t>(v);
                break;
            case 10:   // logicalType union (preferred over ConvertedType)
                if (ft != kTcStruct) { if (!tc_skip(c, ft, 0)) return false; break; }
                parse_logical_type(c, &out->col);
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    // No explicit LogicalType: normalize from the legacy ConvertedType.
    if (out->col.logical == static_cast<int32_t>(PqLogical::None)) {
        derive_logical_from_converted(&out->col);
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
//         12 statistics 14 bloom_filter_offset 15 bloom_filter_length
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
            case 14:    // bloom_filter_offset (absolute file offset)
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0) ch->bloom_filter_offset = v;
                break;
            case 15:    // bloom_filter_length (header + bitset bytes)
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0 && v <= 0x7FFFFFFF) {
                    ch->bloom_filter_length = static_cast<int32_t>(v);
                }
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
//         4 offset_index_offset 5 offset_index_length
//         6 column_index_offset 7 column_index_length
bool parse_column_chunk(TcCursor* c, PqChunk* ch) noexcept {
    assert(c != nullptr && ch != nullptr);
    std::memset(ch, 0, sizeof(*ch));
    ch->null_count = -1;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        int64_t v = 0;
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
            case 4:     // offset_index_offset
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0) ch->offset_index_offset = v;
                break;
            case 5:     // offset_index_length
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0 && v <= 0x7FFFFFFF) {
                    ch->offset_index_length = static_cast<int32_t>(v);
                }
                break;
            case 6:     // column_index_offset
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0) ch->column_index_offset = v;
                break;
            case 7:     // column_index_length
                if (!tc_zigzag(c, &v)) return false;
                if (v > 0 && v <= 0x7FFFFFFF) {
                    ch->column_index_length = static_cast<int32_t>(v);
                }
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
                // Depth-first walk of the schema tree. SchemaElements are
                // serialised in DFS order, so one stack of open groups is
                // enough -- no second pass, no recursion.
                //
                // This replaces a check that required the root's child count to
                // equal every remaining element, which is true only of a
                // completely flat schema. A struct is not flat, yet each of its
                // leaves still holds exactly one value per row: it raises the
                // leaf's max_def (one more level to say "the struct is
                // present") and leaves max_rep at zero. Only a REPEATED
                // ancestor -- a list or map -- makes a leaf produce a variable
                // number of values per row, and that is the case still refused.
                struct Frame {
                    int32_t  left;     // children still expected
                    uint8_t  def;      // max_def accumulated to here
                    uint8_t  rep;      // max_rep accumulated to here
                    uint8_t  ldef;     // def at the innermost REPEATED node's
                                       // parent (0 if none seen yet)
                    uint8_t  rdef;     // def AT the innermost REPEATED node
                    uint16_t plen;     // dotted-path length so far
                };
                Frame stk[16];
                int32_t sp = 0;
                char path[kPqMaxNameBytes];
                for (uint32_t i = 0; i < n; ++i) {
                    SchemaElem se;
                    if (!parse_schema_element(&c, &se)) return false;
                    if (i == 0) {                       // root group
                        stk[0].left = se.num_children;
                        stk[0].def = 0; stk[0].rep = 0; stk[0].plen = 0;
                        stk[0].ldef = 0; stk[0].rdef = 0;
                        sp = 1;
                        continue;
                    }
                    if (sp == 0) return false;          // element past the root
                    const uint8_t def = static_cast<uint8_t>(
                        stk[sp - 1].def + (se.col.optional ? 1 : 0) +
                        (se.repeated ? 1 : 0));
                    const uint8_t rep = static_cast<uint8_t>(
                        stk[sp - 1].rep + (se.repeated ? 1 : 0));
                    // Dotted path, so a struct field is addressable as "s.x"
                    // exactly as every other parquet tool names it.
                    uint16_t plen = stk[sp - 1].plen;
                    const uint16_t base = plen;
                    if (plen != 0u && plen + 1u < kPqMaxNameBytes) {
                        path[plen++] = '.';
                    }
                    for (uint32_t k = 0; se.col.name[k] != '\0' &&
                                         plen + 1u < kPqMaxNameBytes; ++k) {
                        path[plen++] = se.col.name[k];
                    }
                    if (se.num_children > 0) {          // group: descend
                        if (sp >= 16) return false;     // depth cap
                        stk[sp].left = se.num_children;
                        stk[sp].def = def;
                        stk[sp].rep = rep;
                        // A REPEATED group opens a new list level: remember
                        // the def AT it and the def at its parent. Otherwise
                        // inherit whatever the enclosing list level was.
                        stk[sp].ldef = se.repeated ? stk[sp - 1].def
                                                   : stk[sp - 1].ldef;
                        stk[sp].rdef = se.repeated ? def : stk[sp - 1].rdef;
                        stk[sp].plen = plen;
                        ++sp;
                        continue;
                    }
                    if (!se.has_type) return false;     // leaf without a type
                    // A REPEATED leaf (list/map) is recorded, NOT rejected.
                    // Rejecting it here failed the whole FILE: a table with
                    // three scalar columns and one list field could not be
                    // opened at all, though every scalar column was readable.
                    // The leaf must still occupy a slot because a row group's
                    // ColumnChunks are indexed by leaf position -- dropping it
                    // would silently misalign every column after it. Decode
                    // refuses it individually instead, so a projection that
                    // does not ask for the list reads normally.
                    if (out->n_columns >= kPqMaxColumns) return false;
                    PqColumn col = se.col;
                    col.max_def = def;
                    col.max_rep = rep;
                    // A REPEATED LEAF is itself the repeated node (the legacy
                    // 2-level list shape); otherwise the levels come from the
                    // innermost repeated ancestor.
                    col.list_def = se.repeated ? stk[sp - 1].def
                                               : stk[sp - 1].ldef;
                    col.rep_def  = se.repeated ? def : stk[sp - 1].rdef;
                    // `optional` drives the def-level path; a leaf below an
                    // optional ancestor has levels even if it is itself
                    // REQUIRED.
                    col.optional = (def != 0u) ? 1u : 0u;
                    uint16_t w = 0;
                    for (; w < plen && w + 1u < kPqMaxNameBytes; ++w) {
                        col.name[w] = path[w];
                    }
                    col.name[w] = '\0';
                    out->columns[out->n_columns++] = col;
                    (void)base;
                    // close finished groups
                    while (sp > 0 && --stk[sp - 1].left == 0) --sp;
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
