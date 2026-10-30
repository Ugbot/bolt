// bolt/lakehouse/iceberg_statistics.cpp — stat-based predicate prune.
//
// G2FEAT-125: a REAL manifest's bounds are Iceberg's BINARY single-value
// encoding — little-endian, exactly as wide as the column type — not the ASCII
// the JSON fixtures carry. `std::strtoll` on those bytes reads nothing (250.5
// is 00 00 00 00 00 50 6F 40, which starts with a NUL), so before this every
// bound from an Avro manifest decoded as "absent" and stat pruning was inert on
// exactly the tables it exists for. `DataFileRef::binary_bounds` selects the
// decoder; the schema's declared type gives the width, and a bound whose length
// does not match that width is refused rather than guessed at — a misread bound
// prunes away live rows, so this fails to "no information", never to a value.

#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/statistics.h"
#include "bolt/lakehouse/iceberg/partition_values.h"
#include <charconv>
#include <cmath>

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

bool parse_i64(const char* s, int64_t* out) noexcept {
    assert(s != nullptr && out != nullptr);
    const auto length=std::strlen(s);
    if(!length)return false;
    int64_t value=0;const auto result=std::from_chars(s,s+length,value);
    if(result.ec!=std::errc{} || result.ptr!=s+length)return false;
    *out=value;return true;
}

// A text bound (JSON manifests) of `n` bytes, not NUL-terminated.
bool parse_i64_n(const char* p, uint32_t n, int64_t* out) noexcept {
    assert(out != nullptr);
    char buf[32];
    if (p == nullptr || n == 0u || n >= sizeof(buf)) return false;
    std::memcpy(buf, p, n);
    buf[n] = '\0';
    return parse_i64(buf, out);
}

bool pred_as_i64(const PredicateValue& v, int64_t* out) noexcept {
    assert(out != nullptr);
    if (v.type == BoltType::Int64 || v.type == BoltType::Int32) {
        *out = v.i64; return true;
    }
    if (v.str_len > 0) {
        char buf[kLakeMaxValBytes]; std::memset(buf, 0, sizeof(buf));
        const uint32_t n = v.str_len < kLakeMaxValBytes - 1u
                              ? v.str_len : kLakeMaxValBytes - 1u;
        std::memcpy(buf, v.str, n); buf[n] = '\0';
        return parse_i64(buf, out);
    }
    return false;
}

const ColumnStatEntry* find_stat(const FileStats* s, int32_t fid) noexcept { return stats_find(s, fid); }

int32_t schema_field_id(const Schema* sch, const char* name) noexcept {
    assert(sch != nullptr && name != nullptr);
    for (uint32_t i = 0; i < sch->n_fields; ++i) {
        if (std::strcmp(sch->fields[i].name, name) == 0)
            return sch->fields[i].id;
    }
    return -1;
}

const SchemaField* schema_field(const Schema* sch, const char* name) noexcept {
    assert(sch != nullptr && name != nullptr);
    for (uint32_t i = 0; i < sch->n_fields; ++i) {   // bounded: n_fields
        if (std::strcmp(sch->fields[i].name, name) == 0) return &sch->fields[i];
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Iceberg binary single-value decoding, one column type at a time.
// ---------------------------------------------------------------------------

// Width, in bytes, of a bound for `type`, or 0 when it is not a fixed-width
// numeric we decode. `is_float` reports which of the two decoders applies.
uint32_t bound_width(const char* type, bool* is_float) noexcept {
    assert(type != nullptr && is_float != nullptr);
    *is_float = false;
    if (std::strcmp(type, "long") == 0)        return 8u;
    if (std::strcmp(type, "int") == 0)         return 4u;
    if (std::strcmp(type, "date") == 0)        return 4u;
    if (std::strcmp(type, "time") == 0)        return 8u;
    if (std::strcmp(type, "timestamp") == 0)   return 8u;
    if (std::strcmp(type, "timestamptz") == 0) return 8u;
    if (std::strcmp(type, "timestamp_ns") == 0) return 8u;
    if (std::strcmp(type, "timestamptz_ns") == 0) return 8u;
    if (std::strcmp(type, "boolean") == 0)     return 1u;
    if (std::strcmp(type, "double") == 0) { *is_float = true; return 8u; }
    if (std::strcmp(type, "float") == 0)  { *is_float = true; return 4u; }
    return 0u;   // string / binary / uuid / decimal / fixed — not decoded here
}

// Decode one bound to a double, covering both integral and floating columns so
// a single comparison lane serves every fixed-width type. False when the bytes
// are the wrong width for the declared type, which is the fail-closed case.
bool decode_bound(const char* bytes, uint32_t len, const char* type,
                  double* out) noexcept {
    assert(bytes != nullptr && type != nullptr && out != nullptr);
    IcebergTypeInfo info{};PartitionValue value{};
    if(!iceberg_type_parse(type,&info) || !partition_value_decode(info,
        reinterpret_cast<const uint8_t*>(bytes),len,&value))return false;
    if(value.is_float) {if(std::isnan(value.f64))return false;*out=value.f64;return true;}
    if(!value.is_int)return false;
    *out=static_cast<double>(value.i64);return true;
}

// The predicate's value as a double, mirroring `pred_as_i64`'s tolerance of a
// string-typed literal.
bool pred_as_f64(const PredicateValue& v, double* out) noexcept {
    assert(out != nullptr);
    if (v.type == BoltType::Float64 || v.type == BoltType::Float32) {
        *out = v.f64; return true;
    }
    int64_t i = 0;
    if (pred_as_i64(v, &i)) { *out = static_cast<double>(i); return true; }
    return false;
}

// Bounds for one predicate's column, as doubles. `*have_*` stay false when the
// bound is absent or undecodable, which leaves the file unpruned.
void bounds_of(const ColumnStatEntry* cs, const DataFileRef* f,
               const SchemaField* sf, double* lo, bool* have_lo,
               double* hi, bool* have_hi) noexcept {
    assert(cs != nullptr && f != nullptr && lo != nullptr && hi != nullptr);
    assert(have_lo != nullptr && have_hi != nullptr);
    *have_lo = false;
    *have_hi = false;
    if (f->binary_bounds) {
        if (sf == nullptr) return;
        if (cs->has_lower) {
            *have_lo = decode_bound(cs->lower, cs->lower_len, sf->type, lo);
        }
        if (cs->has_upper) {
            *have_hi = decode_bound(cs->upper, cs->upper_len, sf->type, hi);
        }
        return;
    }
    int64_t v = 0;
    if (cs->has_lower && parse_i64_n(cs->lower, cs->lower_len, &v)) {
        *lo = static_cast<double>(v); *have_lo = true;
    }
    if (cs->has_upper && parse_i64_n(cs->upper, cs->upper_len, &v)) {
        *hi = static_cast<double>(v); *have_hi = true;
    }
}

// Exact integer bounds for an integral column (G2ICE-244): a double holds no more than 53 bits, so two distinct
// nanosecond timestamps (or longs) can round to one double and a strict comparison on doubles would drop a file
// that holds a matching row. False when the column is not integral or a bound is not decoded.
bool bounds_of_i64(const ColumnStatEntry* cs, const DataFileRef* f, const SchemaField* sf, int64_t* lo, bool* have_lo,
                   int64_t* hi, bool* have_hi) noexcept {
    assert(cs != nullptr && f != nullptr);
    assert(lo != nullptr && hi != nullptr && have_lo != nullptr && have_hi != nullptr);
    *have_lo = false;
    *have_hi = false;
    if (!f->binary_bounds) {
        *have_lo = cs->has_lower && parse_i64_n(cs->lower, cs->lower_len, lo);
        *have_hi = cs->has_upper && parse_i64_n(cs->upper, cs->upper_len, hi);
        return true;
    }
    if (sf == nullptr) return false;
    bool is_float = false;
    const uint32_t w = bound_width(sf->type, &is_float);
    if (w == 0u || is_float) return false;
    IcebergTypeInfo info{};
    if(!iceberg_type_parse(sf->type,&info))return false;
    auto dec=[&info](const char* bytes,uint32_t length,int64_t* out) noexcept {
        PartitionValue value{};
        if(!partition_value_decode(info,reinterpret_cast<const uint8_t*>(bytes),length,&value) || !value.is_int)return false;
        *out=value.i64;return true;
    };
    *have_lo = cs->has_lower && dec(cs->lower, cs->lower_len, lo);
    *have_hi = cs->has_upper && dec(cs->upper, cs->upper_len, hi);
    return true;
}

// 0 = the predicate excludes [lo, hi]; 1 = it may match.
template <typename T>
bool range_may_match(PredicateOp op, T pv, T lo, bool have_lo, T hi, bool have_hi) noexcept {
    switch (op) {
        case PredicateOp::kEq: return !(have_lo && pv < lo) && !(have_hi && pv > hi);
        case PredicateOp::kLt: return !(have_lo && pv <= lo);
        case PredicateOp::kLe: return !(have_lo && pv < lo);
        case PredicateOp::kGt: return !(have_hi && pv >= hi);
        case PredicateOp::kGe: return !(have_hi && pv > hi);
        default:               return true;
    }
}

}  // namespace

bool stats_pass(const DataFileRef* f, const Schema* sch,
                const Predicate* preds, uint32_t n_preds) noexcept {
    assert(f != nullptr);
    if (sch == nullptr || n_preds == 0) return true;
    for (uint32_t p = 0; p < n_preds; ++p) {
        const Predicate& pr = preds[p];
        const int32_t fid = schema_field_id(sch, pr.column);
        if (fid < 0) continue;
        const ColumnStatEntry* cs = find_stat(&f->stats, fid);
        if (cs == nullptr) continue;
        if (pr.op == PredicateOp::kIsNull) {
            if (!cs->no_null_count && cs->null_count == 0) return false;   // unknown count: may hold nulls
            continue;
        }
        if (pr.op == PredicateOp::kIsNotNull) {
            if (!cs->no_null_count && cs->null_count > 0 && f->stats.record_count > 0 &&
                cs->null_count >= f->stats.record_count) return false;
            continue;
        }
        const SchemaField* sf = schema_field(sch, pr.column);
        int64_t iv = 0, ilo = 0, ihi = 0;
        bool have_lo = false, have_hi = false;
        if (pr.value.type == BoltType::Utf8 || pr.value.type == BoltType::Binary) continue;   // string bounds: not decoded here
        const bool int_pred = pr.value.type != BoltType::Float64 && pr.value.type != BoltType::Float32 &&
                              pred_as_i64(pr.value, &iv);
        if (int_pred && bounds_of_i64(cs, f, sf, &ilo, &have_lo, &ihi, &have_hi)) {
            if (!range_may_match(pr.op, iv, ilo, have_lo, ihi, have_hi)) return false;
            continue;
        }
        double pv = 0.0;
        if (!pred_as_f64(pr.value, &pv) || pv != pv) continue;
        double lo = 0.0, hi = 0.0;
        bounds_of(cs, f, sf, &lo, &have_lo, &hi, &have_hi);
        if (!range_may_match(pr.op, pv, lo, have_lo, hi, have_hi)) return false;
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
