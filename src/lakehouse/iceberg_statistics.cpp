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

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

bool parse_i64(const char* s, int64_t* out) noexcept {
    assert(s != nullptr && out != nullptr);
    if (s[0] == '\0') return false;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s) return false;
    *out = static_cast<int64_t>(v);
    return true;
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

const ColumnStatEntry* find_stat(const FileStats* s, int32_t fid) noexcept {
    assert(s != nullptr);
    for (uint32_t i = 0; i < s->n_cols; ++i) {
        if (s->cols[i].field_id == fid) return &s->cols[i];
    }
    return nullptr;
}

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
    bool is_float = false;
    const uint32_t w = bound_width(type, &is_float);
    if (w == 0u || len != w) return false;
    if (is_float) {
        if (w == 8u) { double d = 0.0; std::memcpy(&d, bytes, 8u); *out = d; }
        else         { float  f = 0.0f; std::memcpy(&f, bytes, 4u);
                       *out = static_cast<double>(f); }
        return true;
    }
    if (w == 1u) { *out = bytes[0] != 0 ? 1.0 : 0.0; return true; }
    if (w == 4u) {
        int32_t v = 0; std::memcpy(&v, bytes, 4u);
        *out = static_cast<double>(v);
        return true;
    }
    int64_t v = 0; std::memcpy(&v, bytes, 8u);
    *out = static_cast<double>(v);
    return true;
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
    if (cs->has_lower && parse_i64(cs->lower, &v)) {
        *lo = static_cast<double>(v); *have_lo = true;
    }
    if (cs->has_upper && parse_i64(cs->upper, &v)) {
        *hi = static_cast<double>(v); *have_hi = true;
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
            if (cs->null_count == 0) return false;
            continue;
        }
        if (pr.op == PredicateOp::kIsNotNull) {
            if (cs->null_count > 0 && f->stats.record_count > 0 &&
                cs->null_count >= f->stats.record_count) return false;
            continue;
        }
        double pv = 0.0;
        if (!pred_as_f64(pr.value, &pv)) continue;
        double lo = 0.0, hi = 0.0;
        bool have_lo = false, have_hi = false;
        bounds_of(cs, f, schema_field(sch, pr.column),
                  &lo, &have_lo, &hi, &have_hi);
        switch (pr.op) {
            case PredicateOp::kEq:
                if (have_lo && pv < lo) return false;
                if (have_hi && pv > hi) return false;
                break;
            case PredicateOp::kLt:
                if (have_lo && pv <= lo) return false;
                break;
            case PredicateOp::kLe:
                if (have_lo && pv < lo) return false;
                break;
            case PredicateOp::kGt:
                if (have_hi && pv >= hi) return false;
                break;
            case PredicateOp::kGe:
                if (have_hi && pv > hi) return false;
                break;
            default:
                break;
        }
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
