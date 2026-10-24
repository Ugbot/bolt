// bolt/lakehouse/iceberg_partition_values.cpp — typed partition values: result types, per-row computation, order,
// binary single-value encoding and the Avro declaration (G2ICE-292). See partition_values.h.

#include "bolt/lakehouse/iceberg/partition_values.h"

#include "bolt/lakehouse/iceberg/metadata.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

constexpr uint32_t kMaxValue = kLakeMaxValBytes - 1u;   // a PartitionValue's str stays NUL-terminated

bool name_is(const char* s, const char* k) noexcept {
    assert(s != nullptr && k != nullptr);
    return std::strcmp(s, k) == 0;
}

// Parse "<prefix><digits><sep><digits><close>" forms: fixed[16], decimal(9,2) / decimal(9, 2).
bool parse_params(const char* s, const char* prefix, char close, int32_t* a, int32_t* b) noexcept {
    assert(s != nullptr && prefix != nullptr && a != nullptr);
    const size_t pl = std::strlen(prefix);
    if (std::strncmp(s, prefix, pl) != 0) return false;
    const char* p = s + pl;
    int32_t* dst[2] = {a, b};
    for (uint32_t k = 0; k < (b != nullptr ? 2u : 1u); ++k) {           // bounded: 2
        while (*p == ' ') ++p;
        int64_t v = 0;
        uint32_t digits = 0;
        while (*p >= '0' && *p <= '9' && digits < 10u) { v = v * 10 + (*p - '0'); ++p; ++digits; }
        if (digits == 0 || v > INT32_MAX) return false;
        *dst[k] = static_cast<int32_t>(v);
        while (*p == ' ') ++p;
        if (k == 0 && b != nullptr) { if (*p != ',') return false; ++p; }
    }
    return p[0] == close && p[1] == '\0';
}

bool is_intlike(IcebergType t) noexcept {
    return t == IcebergType::kBoolean || t == IcebergType::kInt || t == IcebergType::kLong ||
           t == IcebergType::kDate || t == IcebergType::kTime || t == IcebergType::kTimestamp ||
           t == IcebergType::kTimestampTz || t == IcebergType::kTimestampNs || t == IcebergType::kTimestampTzNs;
}

bool is_bytes(IcebergType t) noexcept {
    return t == IcebergType::kString || t == IcebergType::kBinary || t == IcebergType::kUuid ||
           t == IcebergType::kFixed || t == IcebergType::kDecimal;
}

bool is_ts(IcebergType t) noexcept {
    return t == IcebergType::kTimestamp || t == IcebergType::kTimestampTz || t == IcebergType::kTimestampNs ||
           t == IcebergType::kTimestampTzNs;
}

TimeUnit unit_of(IcebergType t) noexcept {
    if (t == IcebergType::kDate) return TimeUnit::kDays;
    return (t == IcebergType::kTimestampNs || t == IcebergType::kTimestampTzNs) ? TimeUnit::kNanos : TimeUnit::kMicros;
}

// The transform is defined for the source type (spec table "Partition Transforms").
bool transform_defined(TransformKind k, IcebergType s) noexcept {
    switch (k) {
        case TransformKind::kIdentity: return s != IcebergType::kUnknown;
        case TransformKind::kVoid:     return true;
        case TransformKind::kBucket:
            return s == IcebergType::kInt || s == IcebergType::kLong || s == IcebergType::kDecimal ||
                   s == IcebergType::kDate || s == IcebergType::kTime || is_ts(s) || s == IcebergType::kString ||
                   s == IcebergType::kUuid || s == IcebergType::kFixed || s == IcebergType::kBinary;
        case TransformKind::kTruncate:
            return s == IcebergType::kInt || s == IcebergType::kLong || s == IcebergType::kDecimal ||
                   s == IcebergType::kString || s == IcebergType::kBinary;
        case TransformKind::kYear:
        case TransformKind::kMonth:
        case TransformKind::kDay:      return s == IcebergType::kDate || is_ts(s);
        case TransformKind::kHour:     return is_ts(s);
        case TransformKind::kUnknown:  return false;
    }
    return false;
}

void put_bytes(PartitionValue* out, const uint8_t* b, uint32_t n) noexcept {
    assert(out != nullptr && n <= kMaxValue);
    assert(b != nullptr || n == 0u);
    out->is_str = true;
    out->str_len = static_cast<uint8_t>(n);
    if (n != 0u) std::memcpy(out->str, b, n);
    out->str[n] = '\0';
}

// A decimal's 128-bit unscaled value as `width` big-endian two's complement bytes (sign-extended / narrowed).
bool put_decimal(PartitionValue* out, int64_t hi, uint64_t lo, uint32_t width) noexcept {
    assert(out != nullptr);
    assert(width >= 1u && width <= 16u);
    uint8_t min[16];
    const uint32_t n = decimal_min_be_bytes(hi, lo, min);
    if (n > width) return false;                                          // the value does not fit the type
    uint8_t be[16];
    std::memset(be, hi < 0 ? 0xff : 0x00, width - n);
    std::memcpy(be + (width - n), min, n);
    put_bytes(out, be, width);
    return true;
}

// The 128-bit value of big-endian two's complement bytes.
void decimal_from_be(const uint8_t* b, uint32_t n, int64_t* hi, uint64_t* lo) noexcept {
    assert(b != nullptr && hi != nullptr && lo != nullptr);
    assert(n >= 1u && n <= 16u);
    uint8_t be[16];
    std::memset(be, (b[0] & 0x80u) != 0u ? 0xff : 0x00, 16u - n);
    std::memcpy(be + (16u - n), b, n);
    uint64_t h = 0, l = 0;
    for (uint32_t i = 0; i < 8u; ++i) { h = (h << 8) | be[i]; l = (l << 8) | be[8u + i]; }
    *hi = static_cast<int64_t>(h);
    *lo = l;
}

// The source value of a row, in the transform's working form.
struct SrcVal {
    int64_t        i64;
    double         f64;
    const uint8_t* bytes;
    uint32_t       len;
    int64_t        dec_hi;
    uint64_t       dec_lo;
};

bool read_int(const BoltColumn& c, int64_t row, int64_t* v) noexcept {
    assert(v != nullptr);
    assert(c.format == ColumnFormat::Flat);
    const uint8_t* p = static_cast<const uint8_t*>(c.data);
    const uint32_t w = c.type_size_bytes != 0u ? c.type_size_bytes : static_cast<uint32_t>(type_size(c.type));
    if (p == nullptr) return false;
    switch (w) {
        case 1: { int8_t x; std::memcpy(&x, p + row, 1); *v = c.type == BoltType::Bool ? (x != 0) : x; return true; }
        case 2: { int16_t x; std::memcpy(&x, p + row * 2, 2); *v = x; return true; }
        case 4: { int32_t x; std::memcpy(&x, p + row * 4, 4); *v = x; return true; }
        case 8: { std::memcpy(v, p + row * 8, 8); return true; }
        default: return false;
    }
}

bool read_source(const IcebergTypeInfo& src, const BoltColumn& c, int64_t row, SrcVal* v) noexcept {
    assert(v != nullptr);
    assert(row >= 0);
    *v = SrcVal{};
    const bool flat = c.format == ColumnFormat::Flat;
    if (is_intlike(src.type)) {
        const bool ok_t = is_integer(c.type) || c.type == BoltType::Bool || c.type == BoltType::Date32 ||
                          c.type == BoltType::Timestamp;
        return ok_t && flat && read_int(c, row, &v->i64);
    }
    if (src.type == IcebergType::kFloat || src.type == IcebergType::kDouble) {
        if (!flat || c.data == nullptr) return false;
        if (c.type == BoltType::Float64) { std::memcpy(&v->f64, static_cast<const uint8_t*>(c.data) + row * 8, 8); return true; }
        if (c.type != BoltType::Float32) return false;
        float f = 0; std::memcpy(&f, static_cast<const uint8_t*>(c.data) + row * 4, 4);
        v->f64 = static_cast<double>(f);
        return true;
    }
    if (src.type == IcebergType::kDecimal) {
        if (!flat || c.data == nullptr) return false;
        const uint8_t* p = static_cast<const uint8_t*>(c.data);
        if (c.type == BoltType::Decimal64) { std::memcpy(&v->dec_lo, p + row * 8, 8); v->dec_hi = static_cast<int64_t>(v->dec_lo) < 0 ? -1 : 0; return true; }
        if (c.type != BoltType::Decimal128) return false;
        std::memcpy(&v->dec_lo, p + row * 16, 8);                         // two's complement, lo word first
        std::memcpy(&v->dec_hi, p + row * 16 + 8, 8);
        return true;
    }
    if (c.type == BoltType::UUID || c.type == BoltType::FixedSizeBinary) {
        if (!flat || c.data == nullptr) return false;
        v->bytes = static_cast<const uint8_t*>(c.data) + row * 16;
        v->len = c.type == BoltType::UUID ? 16u : (c.fixed_width != 0u ? c.fixed_width : 16u);
        return true;
    }
    if (c.type != BoltType::Utf8 && c.type != BoltType::Binary) return false;
    if (c.format != ColumnFormat::Flat && c.format != ColumnFormat::View && c.format != ColumnFormat::VarBinary) return false;
    int32_t n = 0;
    c.utf8_at(row, &v->bytes, &n);
    v->len = static_cast<uint32_t>(n);
    return n >= 0;
}

bool apply_bucket(const IcebergTypeInfo& src, const SrcVal& v, int32_t N, PartitionValue* out) noexcept {
    assert(out != nullptr && N > 0);
    int32_t h = 0;
    if (src.type == IcebergType::kDecimal) h = hash_decimal(v.dec_hi, v.dec_lo);
    else if (is_bytes(src.type)) h = hash_bytes(v.bytes, v.len);
    else if (unit_of(src.type) == TimeUnit::kNanos) h = hash_long(micros_from_nanos(v.i64));
    else h = hash_long(v.i64);
    out->is_int = true;
    out->i64 = bucket_of_hash(h, N);
    assert(out->i64 >= 0 && out->i64 < N);
    return true;
}

bool apply_truncate(const IcebergTypeInfo& src, const SrcVal& v, int32_t W, PartitionValue* out) noexcept {
    assert(out != nullptr);
    if (W <= 0) return false;
    switch (src.type) {
        case IcebergType::kInt:  out->is_int = true; out->i64 = truncate_i32(static_cast<int32_t>(v.i64), W); return true;
        case IcebergType::kLong: out->is_int = true; out->i64 = truncate_i64(v.i64, W); return true;
        case IcebergType::kDecimal: {
            int64_t hi = 0; uint64_t lo = 0;
            truncate_decimal(v.dec_hi, v.dec_lo, W, &hi, &lo);
            return put_decimal(out, hi, lo, src.fixed_len);
        }
        case IcebergType::kString: {
            const uint32_t n = truncate_utf8_len(reinterpret_cast<const char*>(v.bytes), v.len, W);
            if (n > kMaxValue) return false;
            put_bytes(out, v.bytes, n);
            return true;
        }
        case IcebergType::kBinary: {
            const uint32_t n = truncate_binary_len(v.len, W);
            if (n > kMaxValue) return false;
            put_bytes(out, v.bytes, n);
            return true;
        }
        default: return false;
    }
}

bool apply_identity(const IcebergTypeInfo& src, const SrcVal& v, PartitionValue* out) noexcept {
    assert(out != nullptr);
    assert(src.type != IcebergType::kUnknown);
    if (is_intlike(src.type)) { out->is_int = true; out->i64 = v.i64; return true; }
    if (src.type == IcebergType::kFloat || src.type == IcebergType::kDouble) {
        out->is_float = true;
        out->f64 = src.type == IcebergType::kFloat ? static_cast<double>(static_cast<float>(v.f64)) : v.f64;
        return true;
    }
    if (src.type == IcebergType::kDecimal) return put_decimal(out, v.dec_hi, v.dec_lo, src.fixed_len);
    if ((src.type == IcebergType::kUuid && v.len != 16u) || (src.type == IcebergType::kFixed && v.len != src.fixed_len))
        return false;
    if (v.len > kMaxValue) return false;
    put_bytes(out, v.bytes, v.len);
    return true;
}

}  // namespace

uint32_t decimal_required_bytes(uint32_t precision) noexcept {
    // Max precision a decimal of n bytes holds: floor(log10(2^(8n-1) - 1)), n = 1..16.
    static constexpr uint8_t kMaxPrec[16] = {2, 4, 6, 9, 11, 14, 16, 18, 21, 23, 26, 28, 31, 33, 35, 38};
    for (uint32_t n = 0; n < 16u; ++n)                                    // bounded: 16
        if (precision <= kMaxPrec[n]) return n + 1u;
    assert(precision > 38u);
    return 0u;
}

bool iceberg_type_parse(const char* name, IcebergTypeInfo* out) noexcept {
    assert(name != nullptr && out != nullptr);
    *out = IcebergTypeInfo{};
    static const struct { const char* n; IcebergType t; } kNames[] = {
        {"boolean", IcebergType::kBoolean}, {"int", IcebergType::kInt}, {"long", IcebergType::kLong},
        {"float", IcebergType::kFloat}, {"double", IcebergType::kDouble}, {"date", IcebergType::kDate},
        {"time", IcebergType::kTime}, {"timestamp", IcebergType::kTimestamp},
        {"timestamptz", IcebergType::kTimestampTz}, {"timestamp_ns", IcebergType::kTimestampNs},
        {"timestamptz_ns", IcebergType::kTimestampTzNs}, {"string", IcebergType::kString},
        {"binary", IcebergType::kBinary}, {"uuid", IcebergType::kUuid},
    };
    for (const auto& e : kNames)                                          // bounded: 14
        if (name_is(name, e.n)) { out->type = e.t; out->fixed_len = e.t == IcebergType::kUuid ? 16u : 0u; return true; }
    int32_t a = 0, b = 0;
    if (parse_params(name, "fixed[", ']', &a, nullptr) && a > 0) {
        out->type = IcebergType::kFixed;
        out->fixed_len = static_cast<uint32_t>(a);
        return true;
    }
    if (parse_params(name, "decimal(", ')', &a, &b) && a >= 1 && a <= 38 && b <= a) {
        out->type = IcebergType::kDecimal;
        out->precision = static_cast<uint8_t>(a);
        out->scale = static_cast<uint8_t>(b);
        out->fixed_len = decimal_required_bytes(static_cast<uint32_t>(a));
        return true;
    }
    return false;
}

bool partition_result_type(Transform t, const IcebergTypeInfo& src, IcebergTypeInfo* out) noexcept {
    assert(out != nullptr);
    *out = IcebergTypeInfo{};
    if (!transform_defined(t.kind, src.type)) return false;
    if (t.kind == TransformKind::kBucket || t.kind == TransformKind::kTruncate) {
        if (t.param <= 0) return false;
    }
    switch (t.kind) {
        case TransformKind::kIdentity:
        case TransformKind::kTruncate: *out = src; return true;
        case TransformKind::kVoid:
            *out = src;
            if (src.type == IcebergType::kUnknown) out->type = IcebergType::kInt;   // "source type or int"
            return true;
        case TransformKind::kDay: out->type = IcebergType::kDate; return true;
        default: out->type = IcebergType::kInt; return true;               // bucket, year, month, hour
    }
}

bool partition_spec_types(const PartitionSpec* spec, const Schema* schema, IcebergTypeInfo* src,
                          IcebergTypeInfo* result) noexcept {
    assert(spec != nullptr && src != nullptr && result != nullptr);
    if (schema == nullptr || spec->n_fields > kIcebergMaxFieldsPerSpec) return false;
    for (uint32_t i = 0; i < spec->n_fields; ++i) {                       // bounded: kIcebergMaxFieldsPerSpec
        const PartitionField& f = spec->fields[i];
        src[i] = IcebergTypeInfo{};
        for (uint32_t s = 0; s < schema->n_fields; ++s)                   // bounded: n_fields
            if (schema->fields[s].id == f.source_id) { (void)iceberg_type_parse(schema->fields[s].type, &src[i]); break; }
        if (!partition_result_type(f.transform, src[i], &result[i])) return false;
    }
    return true;
}

bool partition_value_of(const PartitionField& f, const IcebergTypeInfo& src, const BoltColumn& c, int64_t row,
                        PartitionValue* out) noexcept {
    assert(out != nullptr);
    assert(row >= 0 && row < c.length);
    std::memset(out, 0, sizeof(*out));
    out->field_id = f.field_id;
    if (f.transform.kind == TransformKind::kVoid || c.is_null(row)) { out->is_null = true; return true; }
    if (!transform_defined(f.transform.kind, src.type)) return false;
    SrcVal v{};
    if (!read_source(src, c, row, &v)) return false;
    const TimeUnit u = unit_of(src.type);
    switch (f.transform.kind) {
        case TransformKind::kIdentity: return apply_identity(src, v, out);
        case TransformKind::kBucket:   return f.transform.param > 0 && apply_bucket(src, v, f.transform.param, out);
        case TransformKind::kTruncate: return apply_truncate(src, v, f.transform.param, out);
        case TransformKind::kYear:     out->is_int = true; out->i64 = transform_year(v.i64, u);  return true;
        case TransformKind::kMonth:    out->is_int = true; out->i64 = transform_month(v.i64, u); return true;
        case TransformKind::kDay:      out->is_int = true; out->i64 = transform_day(v.i64, u);   return true;
        case TransformKind::kHour:     out->is_int = true; out->i64 = transform_hour(v.i64, u);  return true;
        default:                       return false;
    }
}

int partition_value_compare(const IcebergTypeInfo& t, const PartitionValue& a, const PartitionValue& b,
                            bool uuid_signed) noexcept {
    assert(!a.is_null && !b.is_null);
    if (a.is_float || b.is_float) {
        if (a.f64 < b.f64) return -1;
        if (a.f64 > b.f64) return 1;
        const bool na = std::signbit(a.f64), nb = std::signbit(b.f64);   // -0.0 sorts before 0.0
        return na == nb ? 0 : (na ? -1 : 1);
    }
    if (!a.is_str || !b.is_str) return a.i64 < b.i64 ? -1 : (a.i64 > b.i64 ? 1 : 0);
    const uint32_t la = partition_value_bytes_len(a), lb = partition_value_bytes_len(b);
    const auto* pa = reinterpret_cast<const uint8_t*>(a.str);
    const auto* pb = reinterpret_cast<const uint8_t*>(b.str);
    if (t.type == IcebergType::kDecimal && la >= 1u && lb >= 1u && la <= 16u && lb <= 16u) {
        int64_t ha = 0, hb = 0; uint64_t xa = 0, xb = 0;
        decimal_from_be(pa, la, &ha, &xa);
        decimal_from_be(pb, lb, &hb, &xb);
        if (ha != hb) return ha < hb ? -1 : 1;
        return xa < xb ? -1 : (xa > xb ? 1 : 0);
    }
    if (t.type == IcebergType::kUuid && uuid_signed && la == 16u && lb == 16u) {
        for (uint32_t half = 0; half < 2u; ++half) {                      // two signed longs, msb first
            int64_t x = 0, y = 0;
            for (uint32_t i = 0; i < 8u; ++i) { x = (x << 8) | pa[half * 8u + i]; y = (y << 8) | pb[half * 8u + i]; }
            if (x != y) return x < y ? -1 : 1;
        }
        return 0;
    }
    const uint32_t n = la < lb ? la : lb;
    const int c = n == 0u ? 0 : std::memcmp(pa, pb, n);
    if (c != 0) return c < 0 ? -1 : 1;
    return la == lb ? 0 : (la < lb ? -1 : 1);
}

bool partition_value_equal(const PartitionValue& a, const PartitionValue& b) noexcept {
    if (a.is_null || b.is_null) return a.is_null == b.is_null;
    if (a.is_float != b.is_float || a.is_str != b.is_str) return false;
    if (a.is_float) return std::memcmp(&a.f64, &b.f64, sizeof(double)) == 0;
    if (!a.is_str) return a.i64 == b.i64;
    const uint32_t la = partition_value_bytes_len(a);
    return la == partition_value_bytes_len(b) && std::memcmp(a.str, b.str, la) == 0;
}

namespace {

uint32_t encode_value(const IcebergTypeInfo& t, const PartitionValue& v, uint8_t* out, uint32_t cap) noexcept {
    assert(out != nullptr);
    assert(!v.is_null);
    uint32_t w = 0;
    switch (t.type) {
        case IcebergType::kBoolean: w = 1u; break;
        case IcebergType::kInt: case IcebergType::kDate: w = 4u; break;
        case IcebergType::kFloat: w = 4u; break;
        case IcebergType::kDouble: case IcebergType::kLong: case IcebergType::kTime: case IcebergType::kTimestamp:
        case IcebergType::kTimestampTz: case IcebergType::kTimestampNs: case IcebergType::kTimestampTzNs: w = 8u; break;
        case IcebergType::kUnknown: return 0u;
        default: break;
    }
    if (w != 0u) {
        if (cap < w) return 0u;
        if (t.type == IcebergType::kFloat) { const float f = static_cast<float>(v.f64); std::memcpy(out, &f, 4u); return 4u; }
        if (t.type == IcebergType::kDouble) { std::memcpy(out, &v.f64, 8u); return 8u; }
        for (uint32_t i = 0; i < w; ++i) out[i] = static_cast<uint8_t>((static_cast<uint64_t>(v.i64) >> (8u * i)) & 0xffu);
        return w;
    }
    const uint32_t n = partition_value_bytes_len(v);
    if (t.type == IcebergType::kDecimal) {                                // minimal big-endian unscaled bytes
        if (n < 1u || n > 16u) return 0u;
        int64_t hi = 0; uint64_t lo = 0;
        decimal_from_be(reinterpret_cast<const uint8_t*>(v.str), n, &hi, &lo);
        uint8_t m[16];
        const uint32_t mn = decimal_min_be_bytes(hi, lo, m);
        if (mn > cap) return 0u;
        std::memcpy(out, m, mn);
        return mn;
    }
    if (n > cap) return 0u;
    if (n != 0u) std::memcpy(out, v.str, n);
    return n;
}

}  // namespace

bool partition_value_encode(const IcebergTypeInfo& t, const PartitionValue& v, uint8_t* out, uint32_t cap,
                            uint32_t* n) noexcept {
    assert(out != nullptr && n != nullptr);
    *n = 0;
    if (v.is_null || v.is_partial || t.type == IcebergType::kUnknown) return false;
    const bool may_be_empty = (t.type == IcebergType::kString || t.type == IcebergType::kBinary) &&
                              v.is_str && partition_value_bytes_len(v) == 0u;
    *n = encode_value(t, v, out, cap);
    assert(*n <= cap);
    return *n != 0u || may_be_empty;
}

namespace {

// Bounds of one partition field over the entries; false = this field cannot be summarised faithfully.
bool summarise_field(const DataFileRef* files, uint32_t n, uint32_t fi, const IcebergTypeInfo& t,
                     PartitionFieldSummary* ps) noexcept {
    assert(ps != nullptr);
    assert(files != nullptr || n == 0u);
    std::memset(ps, 0, sizeof(*ps));
    ps->has_nan = true;
    const PartitionValue* lo[2] = {nullptr, nullptr};   // [0] unsigned / natural order, [1] signed uuid order
    const PartitionValue* hi[2] = {nullptr, nullptr};
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        if (fi >= files[i].n_partition) return false;
        const PartitionValue& v = files[i].partition[fi];
        if (v.is_partial) return false;
        if (v.is_null) { ps->contains_null = true; continue; }
        if (v.is_float && v.f64 != v.f64) { ps->contains_nan = true; continue; }
        for (uint32_t o = 0; o < 2u; ++o) {                               // bounded: 2
            if (lo[o] == nullptr || partition_value_compare(t, v, *lo[o], o == 1u) < 0) lo[o] = &v;
            if (hi[o] == nullptr || partition_value_compare(t, v, *hi[o], o == 1u) > 0) hi[o] = &v;
        }
    }
    if (lo[0] == nullptr) return true;                                    // every value null / NaN: no bounds
    if (t.type == IcebergType::kUuid && (!partition_value_equal(*lo[0], *lo[1]) || !partition_value_equal(*hi[0], *hi[1])))
        return false;
    uint32_t ln = 0, hn = 0;
    if (!partition_value_encode(t, *lo[0], reinterpret_cast<uint8_t*>(ps->lower), kIcebergMaxSummaryBound, &ln) ||
        !partition_value_encode(t, *hi[0], reinterpret_cast<uint8_t*>(ps->upper), kIcebergMaxSummaryBound, &hn))
        return false;
    ps->lower_len = static_cast<uint8_t>(ln);
    ps->upper_len = static_cast<uint8_t>(hn);
    ps->has_lower = ps->has_upper = true;
    return true;
}

}  // namespace

void manifest_partition_summaries(const DataFileRef* files, uint32_t n, const PartitionSpec* spec,
                                  const IcebergTypeInfo* types, ManifestListEntry* e) noexcept {
    assert(e != nullptr);
    assert(files != nullptr || n == 0u);
    e->n_partitions = 0;
    if (spec == nullptr || types == nullptr || spec->n_fields == 0u || spec->n_fields > kIcebergMaxSummaries) return;
    for (uint32_t f = 0; f < spec->n_fields; ++f)                         // bounded: kIcebergMaxSummaries
        if (!summarise_field(files, n, f, types[f], &e->partitions[f])) return;
    e->n_partitions = spec->n_fields;
}

bool partition_value_decode(const IcebergTypeInfo& t, const uint8_t* b, uint32_t len, PartitionValue* out) noexcept {
    assert(b != nullptr || len == 0u);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    switch (t.type) {
        case IcebergType::kBoolean: if (len != 1u) return false; out->is_int = true; out->i64 = b[0] != 0u; return true;
        case IcebergType::kFloat: { if (len != 4u) return false; float f; std::memcpy(&f, b, 4u); out->is_float = true; out->f64 = f; return true; }
        case IcebergType::kDouble: if (len != 8u) return false; out->is_float = true; std::memcpy(&out->f64, b, 8u); return true;
        case IcebergType::kDecimal: {
            if (len < 1u || len > 16u || t.fixed_len < 1u || t.fixed_len > 16u) return false;
            int64_t hi = 0; uint64_t lo = 0;
            decimal_from_be(b, len, &hi, &lo);
            return put_decimal(out, hi, lo, t.fixed_len);
        }
        case IcebergType::kUnknown: return false;
        default: break;
    }
    if (is_intlike(t.type)) {
        if (len != 4u && len != 8u) return false;
        if (len == 4u) { int32_t x; std::memcpy(&x, b, 4u); out->i64 = x; }
        else std::memcpy(&out->i64, b, 8u);
        out->is_int = true;
        return true;
    }
    if (len > kMaxValue) return false;
    put_bytes(out, b, len);
    return true;
}

uint32_t partition_avro_type_json(const IcebergTypeInfo& t, int32_t field_id, char* out, uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    assert(field_id >= 0);
    const char* simple = nullptr;
    int n = 0;
    switch (t.type) {
        case IcebergType::kBoolean: simple = "\"boolean\""; break;
        case IcebergType::kInt:     simple = "\"int\""; break;
        case IcebergType::kLong:    simple = "\"long\""; break;
        case IcebergType::kFloat:   simple = "\"float\""; break;
        case IcebergType::kDouble:  simple = "\"double\""; break;
        case IcebergType::kString:  simple = "\"string\""; break;
        case IcebergType::kBinary:  simple = "\"bytes\""; break;
        case IcebergType::kDate:    simple = "{\"type\":\"int\",\"logicalType\":\"date\"}"; break;
        case IcebergType::kTime:    simple = "{\"type\":\"long\",\"logicalType\":\"time-micros\"}"; break;
        case IcebergType::kTimestamp:
            simple = "{\"type\":\"long\",\"logicalType\":\"timestamp-micros\",\"adjust-to-utc\":false}"; break;
        case IcebergType::kTimestampTz:
            simple = "{\"type\":\"long\",\"logicalType\":\"timestamp-micros\",\"adjust-to-utc\":true}"; break;
        case IcebergType::kTimestampNs:
            simple = "{\"type\":\"long\",\"logicalType\":\"timestamp-nanos\",\"adjust-to-utc\":false}"; break;
        case IcebergType::kTimestampTzNs:
            simple = "{\"type\":\"long\",\"logicalType\":\"timestamp-nanos\",\"adjust-to-utc\":true}"; break;
        case IcebergType::kUuid:
            n = std::snprintf(out, cap, "{\"type\":\"fixed\",\"name\":\"uuid_fixed_%d\",\"size\":16,\"logicalType\":\"uuid\"}",
                              field_id);
            break;
        case IcebergType::kFixed:
            n = std::snprintf(out, cap, "{\"type\":\"fixed\",\"name\":\"fixed_%u_%d\",\"size\":%u}", t.fixed_len, field_id,
                              t.fixed_len);
            break;
        case IcebergType::kDecimal:
            n = std::snprintf(out, cap, "{\"type\":\"fixed\",\"name\":\"decimal_%u_%u_%d\",\"size\":%u,"
                              "\"logicalType\":\"decimal\",\"precision\":%u,\"scale\":%u}",
                              t.precision, t.scale, field_id, t.fixed_len, t.precision, t.scale);
            break;
        case IcebergType::kUnknown: return 0u;
    }
    if (simple != nullptr) n = std::snprintf(out, cap, "%s", simple);
    if (n <= 0 || static_cast<uint32_t>(n) >= cap) return 0u;
    return static_cast<uint32_t>(n);
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
