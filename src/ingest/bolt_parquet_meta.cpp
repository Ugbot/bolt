// bolt_parquet_meta.cpp — W-PQ: Parquet footer metadata reader.
// See include/bolt/ingest/bolt_parquet_meta.h for scope + safety contract.

#include "bolt/ingest/bolt_parquet_meta.h"

namespace bolt {
namespace ingest {
namespace parquet {

namespace {
constexpr uint32_t kTcMaxDepth = 16;   // corrupt nesting cap
bool strict_field(TcCursor*, int16_t*, uint8_t*, bool*) noexcept;
bool strict_varint(TcCursor* c,uint64_t* value) noexcept {
    assert(c && value);assert(c->p<=c->end);
    const uint8_t* begin=c->p;
    if(!tc_varint(c,value))return false;
    // The tenth ULEB128 byte contains only bit 63. The tolerant primitive
    // historically discards its other payload bits; exact admission must not.
    return c->p-begin<10 || (c->p-begin==10 && begin[9]<=1);
}
bool strict_zigzag(TcCursor* c,int64_t* value) noexcept {
    assert(c && value);assert(c->p<=c->end);
    uint64_t raw=0;if(!strict_varint(c,&raw))return false;
    *value=static_cast<int64_t>((raw>>1)^(~(raw&1)+1));return true;
}
bool strict_binary(TcCursor* c,const uint8_t** bytes,uint32_t* length) noexcept {
    assert(c && bytes && length);assert(c->p<=c->end);
    uint64_t size=0;if(!strict_varint(c,&size) || size>INT32_MAX ||
        size>static_cast<uint64_t>(c->end-c->p))return false;
    *bytes=c->p;*length=static_cast<uint32_t>(size);c->p+=size;return true;
}
bool strict_list(TcCursor* c,uint8_t* type,uint32_t* count) noexcept {
    assert(c && type && count);assert(c->p<=c->end);
    uint8_t header=0;if(!tc_u8(c,&header))return false;
    *type=header&15u;if(*type==kTcStop || *type>kTcStruct)return false;
    uint64_t size=header>>4;
    if(size==15 && !strict_varint(c,&size))return false;
    if(size>INT32_MAX)return false;
    *count=static_cast<uint32_t>(size);return true;
}
}

// Structurally skip one compact-protocol value. Bounded depth.
static bool tc_skip_impl(TcCursor* c, uint8_t type, uint32_t depth, bool strict) noexcept {
    assert(c != nullptr);
    if (depth >= kTcMaxDepth) return false;
    switch (type) {
        case kTcTrue:
        case kTcFalse: return true;            // value lives in the type nibble
        case kTcByte: { uint8_t b; return tc_u8(c, &b); }
        case kTcI16:
        case kTcI32:
        case kTcI64: {
            int64_t v;
            if(!strict)return tc_zigzag(c,&v);
            if(!strict_zigzag(c,&v))return false;
            return type==kTcI64 || (type==kTcI32?v>=INT32_MIN && v<=INT32_MAX:v>=INT16_MIN && v<=INT16_MAX);
        }
        case kTcDouble: {
            if (c->end - c->p < 8) return false;
            c->p += 8;
            return true;
        }
        case kTcBinary: {
            const uint8_t* p; uint32_t n;
            return strict?strict_binary(c,&p,&n):tc_binary(c,&p,&n);
        }
        case kTcList:
        case kTcSet: {
            uint8_t et; uint32_t n;
            if (!(strict?strict_list(c,&et,&n):tc_list(c,&et,&n))) return false;
            if (strict && n > static_cast<uint64_t>(c->end-c->p)) return false;
            for (uint32_t i = 0; i < n; ++i) {
                if (strict && (et==kTcTrue || et==kTcFalse)) {
                    uint8_t value=0;
                    if (!tc_u8(c,&value) || (value!=kTcTrue && value!=kTcFalse)) return false;
                    continue;
                }
                if (!tc_skip_impl(c, et, depth + 1, strict)) return false;
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
            if (!(strict?strict_varint(c,&sz):tc_varint(c,&sz)) || sz > 0xFFFFFF) return false;
            if (!tc_u8(c, &kv)) return false;
            n = static_cast<uint32_t>(sz);
            const uint8_t kt = kv >> 4, vt = kv & 0x0Fu;
            if (strict && n > static_cast<uint64_t>(c->end-c->p)/2u) return false;
            for (uint32_t i = 0; i < n; ++i) {
                const uint8_t items[2]={kt,vt};
                for (const uint8_t item : items) {
                    if (strict && (item==kTcTrue || item==kTcFalse)) {
                        uint8_t value=0;
                        if (!tc_u8(c,&value) || (value!=kTcTrue && value!=kTcFalse)) return false;
                    } else if (!tc_skip_impl(c,item,depth+1,strict)) return false;
                }
            }
            return true;
        }
        case kTcStruct: {
            int16_t fid = 0;
            uint8_t ft;
            // Bounded: each tc_field consumes >=1 byte from a finite slice.
            for (;;) {
                if (strict) {
                    bool done=false;
                    if (!strict_field(c,&fid,&ft,&done)) return false;
                    if (done) return true;
                    if (!tc_skip_impl(c,ft,depth+1,true)) return false;
                    continue;
                }
                const uint8_t* before = c->p;
                if (!tc_field(c, &fid, &ft))
                    return !strict || (before < c->end && *before == kTcStop);
                if (strict && fid <= 0) return false;
                if (!tc_skip_impl(c, ft, depth + 1, strict)) return false;
            }
        }
        default:
            return false;
    }
}

bool tc_skip(TcCursor* c, uint8_t type, uint32_t depth) noexcept {
    return tc_skip_impl(c, type, depth, false);
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
constexpr int32_t kCtEnum = 4;   // G2PQ-17: legacy ConvertedType.ENUM
constexpr int32_t kCtDecimal = 5, kCtDate = 6, kCtTimeMillis = 7,
                  kCtTimeMicros = 8, kCtTsMillis = 9, kCtTsMicros = 10,
                  kCtUint8 = 11, kCtUint16 = 12, kCtUint32 = 13,
                  kCtUint64 = 14, kCtInt8 = 15, kCtInt16 = 16,
                  kCtInt32 = 17, kCtInt64 = 18;
constexpr int32_t kCtInterval = 21;  // deprecated, FLBA(12) months/days/millis

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
// member: 5 DECIMAL, 6 DATE, 7 TIME, 8 TIMESTAMP, 10 INTEGER, 1 STRING,
// 12 JSON, 13 BSON, 14 UUID, 15 FLOAT16, 16 VARIANT (G2PQ-16).
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
            // G2PQ-17: union id 4 EnumType (BYTE_ARRAY, closed string set).
            case 4: col->logical = static_cast<int32_t>(PqLogical::Enum);
                    (void)tc_skip(c, ft, 0); break;
            // parquet.thrift LogicalType union ids: 12 JSON, 13 BSON,
            // 14 UUID, 15 FLOAT16, 16 VARIANT. All carry an empty struct
            // (UUIDType/Float16Type have no fields -- the union id alone
            // is the whole annotation).
            case 12: col->logical = static_cast<int32_t>(PqLogical::Json);
                     (void)tc_skip(c, ft, 0); break;
            case 13: col->logical = static_cast<int32_t>(PqLogical::Bson);
                     (void)tc_skip(c, ft, 0); break;
            case 14: col->logical = static_cast<int32_t>(PqLogical::Uuid);
                     (void)tc_skip(c, ft, 0); break;
            case 15: col->logical = static_cast<int32_t>(PqLogical::Float16);
                     (void)tc_skip(c, ft, 0); break;
            case 16: col->logical = static_cast<int32_t>(PqLogical::Variant);
                     (void)tc_skip(c, ft, 0); break;
            // G2PQ-17: union id 11 NullType (UNKNOWN) -- every value null,
            // no compatible ConvertedType exists for it.
            case 11: col->logical = static_cast<int32_t>(PqLogical::Unknown);
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
        // G2PQ-17: legacy-only writers (pre-parquet-1.9) set ConvertedType
        // ENUM with no LogicalType union at all.
        case kCtEnum:       col->logical = static_cast<int32_t>(PqLogical::Enum); break;
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
        case kCtInterval: col->logical = static_cast<int32_t>(PqLogical::Interval); break;
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
                out->col.name_truncated = (n >= kPqMaxNameBytes) ? 1u : 0u;
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

// ---- KeyValue / key_value_metadata (G2PQ-23) --------------------------------
// KeyValue struct: field 1 key (binary, required), field 2 value (binary,
// optional). Always fully consumes the struct so the cursor stays in sync
// even when the pair is dropped for being oversized or keyless.
bool parse_one_kv(TcCursor* c, char* key_out, uint32_t key_cap,
                  char* val_out, uint32_t val_cap, bool* kept) noexcept {
    assert(c != nullptr && key_out != nullptr && val_out != nullptr);
    assert(val_cap > 0);
    *kept = false;
    key_out[0] = '\0';
    val_out[0] = '\0';
    bool have_key = false;
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        const uint8_t* p; uint32_t n;
        switch (fid) {
            case 1:
                if (!tc_binary(c, &p, &n)) return false;
                if (n > 0 && n < key_cap) {
                    std::memcpy(key_out, p, n);
                    key_out[n] = '\0';
                    have_key = true;
                }
                break;
            case 2:
                if (!tc_binary(c, &p, &n)) return false;
                if (n < val_cap) {
                    std::memcpy(val_out, p, n);
                    val_out[n] = '\0';
                }
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    *kept = have_key;
    return true;
}

// list<KeyValue> -> a bounded array of KV (PqFileKeyValue / PqColKeyValue).
// A pair whose key/value overflowed its cap is dropped, never truncated;
// dropping still consumes its struct bytes above, so later pairs parse fine.
template <typename KV>
bool parse_kv_list(TcCursor* c, KV* out, uint32_t out_cap,
                   uint32_t* n_out) noexcept {
    assert(c != nullptr && out != nullptr && n_out != nullptr);
    uint8_t et; uint32_t n;
    if (!tc_list(c, &et, &n)) return false;
    if (n > kPqMaxKvListLen) return false;   // hostile/corrupt count claim
    uint32_t kept = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (et != kTcStruct) { if (!tc_skip(c, et, 0)) return false; continue; }
        KV tmp{};
        bool ok = false;
        if (!parse_one_kv(c, tmp.key, static_cast<uint32_t>(sizeof(tmp.key)),
                          tmp.value, static_cast<uint32_t>(sizeof(tmp.value)),
                          &ok)) {
            return false;
        }
        if (ok && kept < out_cap) out[kept++] = tmp;
    }
    *n_out = kept;
    return true;
}

// Read a list<i64> into buf[0..kPqMaxLevelHistBuckets). *out_len is the
// bucket count when it fits; 0 (absent) when the list is empty, oversized, or
// the wrong element type -- exotic/corrupt histogram metadata never fails
// the whole chunk parse, same convention as copy_stat's oversized-stat drop.
// The element loop is bounded by `n`, which is itself capped below: every
// tc_zigzag/tc_skip call bounds-checks against the cursor, so a truncated
// buffer with a bogus huge n fails immediately rather than spinning.
bool parse_i64_hist(TcCursor* c, int64_t* buf, uint8_t* out_len) noexcept {
    assert(c != nullptr && buf != nullptr && out_len != nullptr);
    *out_len = 0;
    uint8_t et; uint32_t n;
    if (!tc_list(c, &et, &n)) return false;
    if (n > (1u << 16)) return false;   // no legal histogram is this long
    if (et != kTcI64) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!tc_skip(c, et, 0)) return false;
        }
        return true;
    }
    const bool fits = (n <= kPqMaxLevelHistBuckets);
    for (uint32_t i = 0; i < n; ++i) {
        int64_t v = 0;
        if (!tc_zigzag(c, &v)) return false;
        if (fits) buf[i] = v;
    }
    if (fits) *out_len = static_cast<uint8_t>(n);
    return true;
}

// ---- SizeStatistics (G2PQ-25) ----------------------------------------------
// fields: 1 unencoded_byte_array_data_bytes 2 repetition_level_histogram
//         3 definition_level_histogram
// Verified against apache/parquet-format's parquet.thrift directly (field
// numbers below), not against prose: ColumnIndex does NOT nest a
// SizeStatistics struct -- it carries its OWN flattened, per-page
// repetition_level_histograms/definition_level_histograms (fields 6/7,
// concatenated across pages) plus OffsetIndex's own per-page
// unencoded_byte_array_data_bytes (field 2). Neither of those page-level
// echoes is populated by this change (bolt's writer already marks both
// "(skipped)" in bolt_parquet_pageindex.h) -- only the chunk-level
// ColumnMetaData.size_statistics (field 16) below is in scope.
bool parse_size_statistics(TcCursor* c, PqChunk* ch) noexcept {
    assert(c != nullptr && ch != nullptr);
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        switch (fid) {
            case 1: {
                int64_t v = 0;
                if (!tc_zigzag(c, &v)) return false;
                if (v >= 0) ch->byte_array_data_bytes = v;
                break;
            }
            case 2:
                if (ft != kTcList) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_i64_hist(c, ch->rep_hist, &ch->rep_hist_len)) return false;
                break;
            case 3:
                if (ft != kTcList) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_i64_hist(c, ch->def_hist, &ch->def_hist_len)) return false;
                break;
            default:
                if (!tc_skip(c, ft, 0)) return false;
                break;
        }
    }
    return true;
}

// ---- ColumnMetaData ---------------------------------------------------------
// fields: 1 type 2 encodings 3 path_in_schema 4 codec 5 num_values
//         6 total_uncompressed_size 7 total_compressed_size 8 key_value_metadata
//         9 data_page_offset 10 index_page_offset 11 dictionary_page_offset
//         12 statistics 14 bloom_filter_offset 15 bloom_filter_length
//         16 size_statistics
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
            case 8:
                if (ft != kTcList) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_kv_list(c, ch->col_kv, kPqMaxColKv, &ch->n_col_kv)) {
                    return false;
                }
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
            case 16:
                if (ft != kTcStruct) { if (!tc_skip(c, ft, 0)) return false; break; }
                if (!parse_size_statistics(c, ch)) return false;
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
    ch->byte_array_data_bytes = -1;
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

// ---- SortingColumn (B6) ------------------------------------------------------
// fields: 1 column_idx(i32) 2 descending(bool) 3 nulls_first(bool). A compact
// bool field carries its value in the field-header type nibble, same as the
// isAdjustedToUTC/isSigned bools parsed above -- no separate value byte.
bool parse_sorting_column(TcCursor* c, PqSortingColumn* sc) noexcept {
    assert(c != nullptr && sc != nullptr);
    std::memset(sc, 0, sizeof(*sc));
    int16_t fid = 0;
    uint8_t ft;
    while (tc_field(c, &fid, &ft)) {
        switch (fid) {
            case 1: {
                int64_t v;
                if (!tc_zigzag(c, &v)) return false;
                sc->column_idx = static_cast<int32_t>(v);
                break;
            }
            case 2:
                sc->descending = (ft == kTcTrue);
                break;
            case 3:
                sc->nulls_first = (ft == kTcTrue);
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
//         4 sorting_columns(list<SortingColumn>, B6)
//         5 file_offset 6 total_compressed_size 7 ordinal (all optional --
//         G2PQ-19-style: absent means "writer didn't emit it", not an error)
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
                if (n > kMaxColumns) return false;
                for (uint32_t i = 0; i < n; ++i) {
                    if (m->n_chunks >= m->chunks_cap) {
                        // Schema came after the row groups (legal, rare): we
                        // cannot size exactly, so ask for double.
                        const uint64_t want = static_cast<uint64_t>(m->chunks_cap) * 2u + n;
                        m->need_chunks = want > UINT32_MAX ? UINT32_MAX
                                                           : static_cast<uint32_t>(want);
                        return false;
                    }
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
            case 4: {   // sorting_columns: list<SortingColumn> (B6, optional)
                uint8_t et; uint32_t n;
                if (!tc_list(c, &et, &n)) return false;
                if (et != kTcStruct) return false;
                for (uint32_t i = 0; i < n; ++i) {
                    PqSortingColumn tmp;
                    if (!parse_sorting_column(c, &tmp)) return false;
                    if (rg->n_sorting_columns < kPqMaxSortingColumns) {
                        rg->sorting_columns[rg->n_sorting_columns++] = tmp;
                    }
                }
                break;
            }
            case 5:
                if (!tc_zigzag(c, &v)) return false;
                rg->file_offset = v;
                break;
            case 6:
                if (!tc_zigzag(c, &v)) return false;
                rg->total_compressed_size = v;
                break;
            case 7:
                if (!tc_zigzag(c, &v)) return false;
                rg->ordinal = static_cast<int16_t>(v);
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
    if (out->columns == nullptr || out->columns_cap == 0) return false;
    out->need_columns = 0;
    out->need_chunks = 0;
    out->num_rows = 0;
    out->n_columns = 0;
    out->n_row_groups = 0;
    out->n_chunks = 0;
    out->version = 0;
    out->n_file_kv = 0;

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
                if (et != kTcStruct || n == 0) return false;
                // Leaves <= elements - 1 (the root). Checked before any
                // column is written so a too-small array fails cleanly.
                if (n - 1u > kMaxColumns) return false;
                if (n - 1u > out->columns_cap) {
                    out->need_columns = n - 1u;
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
                    bool     cut;      // the dotted path was truncated
                    // G2PQ-15: one (ldef,rdef) pair per repeated ancestor seen
                    // so far, index k-1 for level k. ldef/rdef above track
                    // only the INNERMOST (most recent) repeated ancestor, same
                    // as pre-G2PQ-15 -- ldefs[rep-1]/rdefs[rep-1] below always
                    // agree with them. A level past kPqMaxRepLevels is simply
                    // never written into the array -- the leaf's true max_rep
                    // still records the real depth, so build_list_column
                    // refuses that ONE column explicitly instead of this walk
                    // failing the whole file over it.
                    uint8_t  ldefs[kPqMaxRepLevels];
                    uint8_t  rdefs[kPqMaxRepLevels];
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
                        stk[0].cut = false;
                        for (uint32_t z = 0; z < kPqMaxRepLevels; ++z) {
                            stk[0].ldefs[z] = 0; stk[0].rdefs[z] = 0;
                        }
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
                    bool cut = stk[sp - 1].cut || se.col.name_truncated != 0u;
                    if (plen != 0u) {
                        if (plen + 1u < kPqMaxNameBytes) path[plen++] = '.';
                        else cut = true;
                    }
                    for (uint32_t k = 0; se.col.name[k] != '\0'; ++k) {
                        if (plen + 1u >= kPqMaxNameBytes) { cut = true; break; }
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
                        for (uint32_t z = 0; z < kPqMaxRepLevels; ++z) {
                            stk[sp].ldefs[z] = stk[sp - 1].ldefs[z];
                            stk[sp].rdefs[z] = stk[sp - 1].rdefs[z];
                        }
                        if (se.repeated && rep >= 1u && rep <= kPqMaxRepLevels) {
                            stk[sp].ldefs[rep - 1u] = stk[sp - 1].def;
                            stk[sp].rdefs[rep - 1u] = def;
                        }
                        stk[sp].plen = plen;
                        stk[sp].cut = cut;
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
                    // does not ask for the list reads normally. The same
                    // applies to a leaf past kPqMaxRepLevels (G2PQ-15): its
                    // max_rep records the real depth so the assembler refuses
                    // it by name, but sibling columns still open.
                    if (out->n_columns >= out->columns_cap) return false;
                    PqColumn col = se.col;
                    col.max_def = def;
                    col.max_rep = rep;
                    // A REPEATED LEAF is itself the repeated node (the legacy
                    // 2-level list shape); otherwise the levels come from the
                    // innermost repeated ancestor.
                    col.list_def = se.repeated ? stk[sp - 1].def
                                               : stk[sp - 1].ldef;
                    col.rep_def  = se.repeated ? def : stk[sp - 1].rdef;
                    for (uint32_t z = 0; z < kPqMaxRepLevels; ++z) {
                        col.list_defs[z] = stk[sp - 1].ldefs[z];
                        col.rep_defs[z] = stk[sp - 1].rdefs[z];
                    }
                    if (se.repeated && rep >= 1u && rep <= kPqMaxRepLevels) {
                        col.list_defs[rep - 1u] = stk[sp - 1].def;
                        col.rep_defs[rep - 1u] = def;
                    }
                    // `optional` drives the def-level path; a leaf below an
                    // optional ancestor has levels even if it is itself
                    // REQUIRED.
                    col.optional = (def != 0u) ? 1u : 0u;
                    uint16_t w = 0;
                    for (; w < plen && w + 1u < kPqMaxNameBytes; ++w) {
                        col.name[w] = path[w];
                    }
                    col.name[w] = '\0';
                    col.name_truncated = cut ? 1u : 0u;
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
            case 5:
                if (ft != kTcList) { if (!tc_skip(&c, ft, 0)) return false; break; }
                if (!parse_kv_list(&c, out->file_kv, kPqMaxFileKv,
                                   &out->n_file_kv)) {
                    return false;
                }
                break;
            case 4: {   // row_groups: list<RowGroup>
                uint8_t et; uint32_t n;
                if (!tc_list(&c, &et, &n)) return false;
                if (et != kTcStruct || n > kPqMaxRowGroups) return false;
                if (out->n_columns != 0u) {
                    const uint64_t want = static_cast<uint64_t>(n) * out->n_columns;
                    if (want > out->chunks_cap) {
                        out->need_chunks = want > UINT32_MAX
                                               ? UINT32_MAX
                                               : static_cast<uint32_t>(want);
                        return false;
                    }
                }
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

namespace {
using StrictStatus = PqMetaParseStatus;
// This preflight shares the compact primitives and SchemaElem decoder with the
// tolerant reader. Raw presence/error information cannot be recovered from a
// PqColumn (notably required groups and absent UTC booleans). No data page work.
struct StrictSchema {
    StrictStatus status = StrictStatus::Invalid;
    uint32_t columns = 0, max_columns = 0, max_groups = 0;
    PqColumn* fields = nullptr;
};
bool strict_refuse(StrictSchema& s, StrictStatus status) noexcept {
    s.status = status; return false;
}
// Distinguish an actual STOP from the tolerant helper's end/error return.
bool strict_field(TcCursor* c, int16_t* id, uint8_t* type, bool* done) noexcept {
    assert(c && id && type && done); assert(c->p <= c->end);
    *done = false;
    if (c->p == c->end) return false;
    if (*c->p == kTcStop) { ++c->p; *done = true; return true; }
    uint8_t header=0;if(!tc_u8(c,&header))return false;
    *type=header & 15u;int64_t next=0;
    if(header>>4)next=int64_t(*id)+(header>>4);
    else {
        const uint8_t* start=c->p;
        if(!strict_zigzag(c,&next) || c->p-start>3)return false;
    }
    if(next<=0 || next>INT16_MAX || *type==kTcStop || *type>kTcStruct)return false;
    *id=static_cast<int16_t>(next);return true;
}
bool strict_i32(TcCursor* c, int32_t* out) noexcept {
    assert(c && out); assert(c->p <= c->end);
    const uint8_t* start=c->p; int64_t value=0;
    if (!strict_zigzag(c,&value) || c->p-start>5 || value<INT32_MIN || value>INT32_MAX) return false;
    *out=static_cast<int32_t>(value); return true;
}
bool strict_empty(TcCursor* c) noexcept {
    assert(c); assert(c->p <= c->end);
    if (c->p==c->end || *c->p!=kTcStop) return false;
    ++c->p; return true;
}
bool strict_i64(TcCursor* c,int64_t* out) noexcept {
    assert(c && out);assert(c->p<=c->end);
    return strict_zigzag(c,out);
}
bool strict_time_unit(TcCursor* c) noexcept {
    assert(c); assert(c->p <= c->end);
    int16_t id=0;uint8_t type=0;bool done=false;
    if (!strict_field(c,&id,&type,&done) || done || id<1 || id>3 || type!=kTcStruct ||
        !strict_empty(c)) return false;
    return strict_field(c,&id,&type,&done) && done;
}
bool strict_logical_members(TcCursor* c, int16_t member) noexcept {
    assert(c); assert(member>0);
    if (member==1 || member==6) return strict_empty(c); // STRING / DATE
    uint8_t seen=0;int16_t id=0;uint8_t type=0;bool done=false;
    for (;;) {
        if (!strict_field(c,&id,&type,&done)) return false;
        if (done) return seen==3;
        if (id<1 || id>2 || (seen & (1u<<(id-1)))) return false;
        seen=static_cast<uint8_t>(seen | (1u<<(id-1)));
        if (member==5) { // DECIMAL: scale and precision
            int32_t ignored=0;if(type!=kTcI32 || !strict_i32(c,&ignored))return false;
        } else if (member==7 || member==8) { // TIME / TIMESTAMP
            if (id==1) {if(type!=kTcTrue && type!=kTcFalse)return false;}
            else if(type!=kTcStruct || !strict_time_unit(c))return false;
        } else { // INTEGER: bitWidth and isSigned
            if(id==1){uint8_t bits=0;if(type!=kTcByte || !tc_u8(c,&bits) ||
                (bits!=8 && bits!=16 && bits!=32 && bits!=64))return false;}
            else if(type!=kTcTrue && type!=kTcFalse)return false;
        }
    }
}
bool strict_logical(TcCursor* c, StrictSchema& s, int16_t* member) noexcept {
    assert(c && member); assert(c->p <= c->end);
    int16_t id=0;uint8_t type=0;bool done=false;
    if(!strict_field(c,&id,&type,&done) || done || type!=kTcStruct)return false;
    if(id!=1 && id!=5 && id!=6 && id!=7 && id!=8 && id!=10)
        return strict_refuse(s,StrictStatus::Unsupported);
    *member=id;
    if(!strict_logical_members(c,id))return false;
    return strict_field(c,&id,&type,&done) && done; // a union has exactly one member
}
bool strict_annotation_pair(const SchemaElem& e,int16_t logical) noexcept {
    const auto& c=e.col;const auto old=c.converted;
    if(old<0 || !logical)return true;
    if(logical==1)return old==0;
    if(logical==5)return old==kCtDecimal;
    if(logical==6)return old==kCtDate;
    if(logical==7)return c.ts_utc && old==(c.time_unit==1?kCtTimeMillis:c.time_unit==2?kCtTimeMicros:-1);
    if(logical==8)return c.ts_utc && old==(c.time_unit==1?kCtTsMillis:c.time_unit==2?kCtTsMicros:-1);
    assert(logical==10);assert(c.int_bits>0);
    const int32_t index=c.int_bits==8?0:c.int_bits==16?1:c.int_bits==32?2:3;
    return old==(c.int_signed?kCtInt8:kCtUint8)+index;
}
bool strict_type(const PqColumn& c,StrictSchema& s) noexcept {
    const auto logical=static_cast<PqLogical>(c.logical);
    if(logical==PqLogical::Int)return
        (c.int_bits==8 || c.int_bits==16 || c.int_bits==32 || c.int_bits==64) &&
        c.physical==(c.int_bits==64?PqType::Int64:PqType::Int32);
    if(logical==PqLogical::String || (logical==PqLogical::None && c.converted==0))return c.physical==PqType::ByteArray;
    if(logical==PqLogical::Date)return c.physical==PqType::Int32;
    if(logical==PqLogical::Timestamp)return c.physical==PqType::Int64 && c.time_unit>=1 && c.time_unit<=3;
    if(logical==PqLogical::Time)return c.time_unit>=1 && c.time_unit<=3 &&
        c.physical==(c.time_unit==1?PqType::Int32:PqType::Int64);
    if(logical==PqLogical::Decimal) {
        int32_t precision=0;
        if(c.physical==PqType::Int32)precision=9;
        else if(c.physical==PqType::Int64)precision=18;
        else if(c.physical==PqType::FixedLenByteArray) {
            if(c.type_length==4)precision=9;else if(c.type_length==8)precision=18;
            else if(c.type_length==16)precision=38;else if(c.type_length==32)precision=76;
            else return strict_refuse(s,StrictStatus::Unsupported);
        } else return false;
        return c.precision>0 && c.precision<=precision && c.scale>=0 && c.scale<=c.precision;
    }
    if(logical!=PqLogical::None)return strict_refuse(s,StrictStatus::Unsupported);
    if(c.physical==PqType::Int96)return strict_refuse(s,StrictStatus::Unsupported);
    return c.physical!=PqType::FixedLenByteArray || c.type_length>0;
}
bool strict_element(TcCursor* c,StrictSchema& s,SchemaElem* element,bool root) noexcept {
    assert(c && element);assert(c->p <= c->end);
    TcCursor decoded=*c;uint16_t seen=0;int16_t id=0,logical=0;uint8_t type=0;bool done=false;
    int32_t legacy_scale=0,legacy_precision=0;
    PqColumn logical_column{};
    for (;;) {
        if(!strict_field(c,&id,&type,&done))return false;
        if(done)break;
        if(id>10 || (seen & (1u<<(id-1))))return false;
        seen=static_cast<uint16_t>(seen | (1u<<(id-1)));
        if(id==4) {
            const uint8_t* name=nullptr;uint32_t size=0;
            if(type!=kTcBinary || !strict_binary(c,&name,&size) || !size || std::memchr(name,0,size))return false;
            if(size>=kPqMaxNameBytes)return strict_refuse(s,StrictStatus::Unsupported);
        } else if(id==10) {
            TcCursor logical_cursor=*c;
            if(type!=kTcStruct || !strict_logical(c,s,&logical))return false;
            parse_logical_type(&logical_cursor,&logical_column);
        } else {
            int32_t value=0;if(type!=kTcI32 || !strict_i32(c,&value))return false;
            if((id==1 && (value<0 || value>7)) || (id==2 && value<=0) ||
                (id==3 && (value<0 || value>2)) || (id==5 && value<0) ||
                (id==6 && value<0))return false;
            if(id==7)legacy_scale=value;
            if(id==8)legacy_precision=value;
        }
    }
    if(!(seen & (1u<<3)) || !parse_schema_element(&decoded,element) || decoded.p!=c->p)return false;
    if(root)return !element->has_type && seen==((1u<<3)|(1u<<4)) && element->num_children>0;
    if(element->num_children || element->repeated)return strict_refuse(s,StrictStatus::Unsupported);
    if(!element->has_type || !(seen & (1u<<2)))return false;
    const auto old=element->col.converted;
    if(old>=0 && old!=0 && old!=kCtDecimal && old!=kCtDate && (old<kCtTimeMillis || old>kCtInt64))
        return strict_refuse(s,StrictStatus::Unsupported);
    // Converted TIME/TIMESTAMP alone loses the required explicit UTC member.
    if(!logical && old>=kCtTimeMillis && old<=kCtTsMicros)return strict_refuse(s,StrictStatus::Unsupported);
    if(!logical && old==kCtDecimal && (seen & ((1u<<6)|(1u<<7)))!=((1u<<6)|(1u<<7)))return false;
    if((seen & ((1u<<6)|(1u<<7))) && element->col.logical!=static_cast<int32_t>(PqLogical::Decimal))return false;
    if(logical==5 && (((seen & (1u<<6)) && legacy_scale!=logical_column.scale) ||
        ((seen & (1u<<7)) && legacy_precision!=logical_column.precision)))return false;
    if((seen & (1u<<1)) && element->col.physical!=PqType::FixedLenByteArray)return false;
    if(element->col.physical==PqType::FixedLenByteArray && !(seen & (1u<<1)))return false;
    return strict_annotation_pair(*element,logical) && strict_type(element->col,s);
}
bool strict_schema_list(TcCursor* c,StrictSchema& s) noexcept {
    assert(c);assert(s.max_columns>0);
    uint8_t type=0;uint32_t count=0;
    if(!strict_list(c,&type,&count) || type!=kTcStruct || count<2)return false;
    if(count-1>s.max_columns)return strict_refuse(s,StrictStatus::Capacity);
    SchemaElem root{};if(!strict_element(c,s,&root,true))return false;
    if(static_cast<uint32_t>(root.num_children)!=count-1)return strict_refuse(s,StrictStatus::Unsupported);
    for(uint32_t i=1;i<count;++i) {
        SchemaElem column{};if(!strict_element(c,s,&column,false))return false;
        s.fields[i-1]=column.col;
    }
    s.columns=count-1;return true;
}
bool strict_chunk_path(TcCursor* c,const PqColumn& column) noexcept {
    assert(c);assert(column.name[0]);
    uint8_t type=0;uint32_t count=0;const uint8_t* name=nullptr;uint32_t size=0;
    return strict_list(c,&type,&count) && type==kTcBinary && count==1 &&
        strict_binary(c,&name,&size) && size==std::strlen(column.name) &&
        std::memcmp(name,column.name,size)==0;
}
bool strict_column_meta(TcCursor* c,StrictSchema& s,uint32_t column) noexcept {
    assert(c);assert(column<s.columns);
    uint32_t seen=0;int16_t id=0;uint8_t type=0;bool done=false;
    for(;;) {
        if(!strict_field(c,&id,&type,&done))return false;
        if(done)return (seen & 0x17fu)==0x17fu;
        if(id<=16){if(seen & (1u<<(id-1)))return false;seen |= (1u<<(id-1));}
        if(id==1 || id==4) {
            int32_t value=0;if(type!=kTcI32 || !strict_i32(c,&value))return false;
            if(id==1 && value!=static_cast<int32_t>(s.fields[column].physical))return false;
            if(id==4 && (value<0 || value>7))return strict_refuse(s,StrictStatus::Unsupported);
        } else if(id==2) {
            uint8_t element=0;uint32_t count=0;
            if(type!=kTcList || !strict_list(c,&element,&count) || element!=kTcI32 || !count)return false;
            for(uint32_t i=0;i<count;++i){int32_t encoding=0;if(!strict_i32(c,&encoding) || encoding<0)return false;}
        } else if(id==3) {if(type!=kTcList || !strict_chunk_path(c,s.fields[column]))return false;}
        else if((id>=5 && id<=7) || id==9 || id==11) {
            int64_t value=0;if(type!=kTcI64 || !strict_i64(c,&value) || value<0 || ((id==9 || id==11) && value<4))return false;
        } else if(!tc_skip_impl(c,type,0,true))return false;
    }
}
bool strict_column_chunk(TcCursor* c,StrictSchema& s,uint32_t column) noexcept {
    assert(c);assert(column<s.columns);
    uint8_t seen=0;int16_t id=0;uint8_t type=0;bool done=false;
    for(;;) {
        if(!strict_field(c,&id,&type,&done))return false;
        if(done)return (seen & 6u)==6u;
        if(id<=3){if(seen & (1u<<(id-1)))return false;seen=static_cast<uint8_t>(seen | (1u<<(id-1)));}
        if(id==1) {
            const uint8_t* path=nullptr;uint32_t size=0;
            if(type!=kTcBinary || !strict_binary(c,&path,&size))return false;
            if(size)return strict_refuse(s,StrictStatus::Unsupported); // external data file
        } else if(id==2) {int64_t offset=0;if(type!=kTcI64 || !strict_i64(c,&offset) || offset<0)return false;}
        else if(id==3){if(type!=kTcStruct || !strict_column_meta(c,s,column))return false;}
        else if(!tc_skip_impl(c,type,0,true))return false;
    }
}
bool strict_row_group(TcCursor* c,StrictSchema& s) noexcept {
    assert(c);assert(s.columns>0);
    uint8_t seen=0;int16_t id=0;uint8_t type=0;bool done=false;
    for(;;) {
        if(!strict_field(c,&id,&type,&done))return false;
        if(done)return (seen & 7u)==7u;
        if(id<=3){if(seen & (1u<<(id-1)))return false;seen=static_cast<uint8_t>(seen | (1u<<(id-1)));}
        if(id==1) {
            uint8_t element=0;uint32_t count=0;
            if(type!=kTcList || !strict_list(c,&element,&count) || element!=kTcStruct || count!=s.columns)return false;
            for(uint32_t i=0;i<count;++i)if(!strict_column_chunk(c,s,i))return false;
        } else if(id==2 || id==3) {
            int64_t value=0;if(type!=kTcI64 || !strict_i64(c,&value) || value<0)return false;
        } else if(!tc_skip_impl(c,type,0,true))return false;
    }
}
bool strict_preflight(const uint8_t* bytes,uint32_t size,StrictSchema& s) noexcept {
    assert(bytes);assert(size>0);
    TcCursor c{bytes,bytes+size};int16_t id=0;uint8_t type=0,seen=0;bool done=false;
    TcCursor groups{};uint32_t group_count=0;
    for (;;) {
        if(!strict_field(&c,&id,&type,&done))return false;
        if(done) {
            if(seen!=15 || c.p!=c.end)return false;
            for(uint32_t i=0;i<group_count;++i)if(!strict_row_group(&groups,s))return false;
            return true;
        }
        if(id<=4) {
            if(seen & (1u<<(id-1)))return false;
            seen=static_cast<uint8_t>(seen | (1u<<(id-1)));
        }
        if(id==1){int32_t version=0;if(type!=kTcI32 || !strict_i32(&c,&version))return false;
            if(version!=1 && version!=2)return strict_refuse(s,StrictStatus::Unsupported);}
        else if(id==2){if(type!=kTcList || !strict_schema_list(&c,s))return false;}
        else if(id==3){int64_t rows=0;if(type!=kTcI64 || !strict_i64(&c,&rows) || rows<0)return false;}
        else if(id==4) {
            uint8_t element=0;uint32_t count=0;
            if(type!=kTcList || !strict_list(&c,&element,&count) || element!=kTcStruct)return false;
            if(count>s.max_groups)return strict_refuse(s,StrictStatus::Capacity);
            groups=c;group_count=count;
            for(uint32_t i=0;i<count;++i)if(!tc_skip_impl(&c,kTcStruct,0,true))return false;
        } else if(!tc_skip_impl(&c,type,0,true))return false;
    }
}
} // namespace

PqMetaParseStatus pq_parse_file_meta_strict(const uint8_t* bytes,uint32_t size,
        PqMeta* out,uint32_t max_row_groups) noexcept {
    if(!bytes || !size || !out || !out->columns || !out->columns_cap ||
        out->columns_cap>kMaxColumns || !out->chunks || !out->chunks_cap ||
        !max_row_groups || max_row_groups>kPqMaxRowGroups)return StrictStatus::Invalid;
    assert(out->columns_cap<=kMaxColumns);assert(max_row_groups<=kPqMaxRowGroups);
    StrictSchema strict{StrictStatus::Invalid,0,out->columns_cap,max_row_groups,out->columns};
    if(!strict_preflight(bytes,size,strict))return strict.status;
    if(!pq_parse_file_meta(bytes,size,out))return
        out->need_columns>out->columns_cap || out->need_chunks>out->chunks_cap?StrictStatus::Capacity:StrictStatus::Invalid;
    if(out->num_rows<0 || out->n_columns!=strict.columns)return StrictStatus::Invalid;
    uint64_t rows=0;
    for(uint32_t g=0;g<out->n_row_groups;++g) {
        const auto& group=out->row_groups[g];
        if(group.num_rows<0 || group.total_byte_size<0 ||
            rows>UINT64_MAX-static_cast<uint64_t>(group.num_rows))return StrictStatus::Invalid;
        rows+=static_cast<uint64_t>(group.num_rows);
        for(uint32_t i=0;i<group.chunk_count;++i) {
            const auto& chunk=out->chunks[group.chunk_off+i];
            if(chunk.num_values!=group.num_rows || chunk.total_compressed_size<0 ||
                chunk.total_uncompressed_size<0)return StrictStatus::Invalid;
        }
    }
    if(rows!=static_cast<uint64_t>(out->num_rows))return StrictStatus::Invalid;
    for(uint32_t i=0;i<out->n_columns;++i)for(uint32_t j=0;j<i;++j)
        if(std::strcmp(out->columns[i].name,out->columns[j].name)==0)return StrictStatus::Unsupported;
    return StrictStatus::Ok;
}

}  // namespace parquet
}  // namespace ingest
}  // namespace bolt
