// bolt/lakehouse/iceberg_partition.cpp — partition pruning of data files (by their tuple) and of manifests (by the
// manifest list's partition summaries).
//
// A predicate on a source column is projected through each partition field's transform into an INCLUSIVE predicate
// on the partition value (spec: "Scan Planning": "an inclusive projection ... matches every partition that may hold
// matching rows"), using the projections Iceberg's reference implementations use (pyiceberg transforms.py,
// Java ProjectionUtil):
//   identity                       x op a          ->  p op a                (every op)
//   bucket[N]                      x = a           ->  p = bucket(a)         (no other op projects)
//   truncate[W] on int/long        x = a / x <= a / x >= a  ->  p = T(a) / p <= T(a) / p >= T(a)
//   year/month/day/hour            x < a           ->  p <= T(a - 1)        x > a  ->  p >= T(a + 1)
//   truncate[W] on string/binary   x < a, x <= a   ->  p <= T(a)            x > a, x >= a  ->  p >= T(a)
//   any non-void transform         x IS NULL       ->  p IS NULL            x IS NOT NULL  ->  p IS NOT NULL
// void, an unknown transform, decimal / uuid / fixed sources and a literal of the wrong kind project nothing: the
// file is kept (spec: "Partition fields that use an unknown transform can be read by ignoring the partition field for
// the purpose of filtering data files"). A file is skipped only when some conjunct's projection is false for it.
//
// G2FEAT-125: how a file's partition VALUE is matched to a spec FIELD depends on which parser produced the file. The
// JSON path reads a real "field-id" out of the document; the Avro path reports the ORDINAL within the partition
// struct (which Iceberg writes in spec order). `DataFileRef::partition_ordinal_ids` says which rule applies.

#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/partition_values.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

// A predicate on the partition value: `op` against `v` (a value of the field's result type).
struct Projected {
    PredicateOp    op;
    uint8_t        _pad[7];
    PartitionValue v;
};

bool literal_i64(const PredicateValue& v, int64_t* out) noexcept {
    assert(out != nullptr);
    if (is_integer(v.type) || v.type == BoltType::Bool || v.type == BoltType::Date32 || v.type == BoltType::Timestamp ||
        v.type == BoltType::Date64) {
        *out = v.i64;
        return true;
    }
    return false;
}

bool is_int_source(IcebergType t) noexcept {
    return t == IcebergType::kBoolean || t == IcebergType::kInt || t == IcebergType::kLong || t == IcebergType::kDate ||
           t == IcebergType::kTime || t == IcebergType::kTimestamp || t == IcebergType::kTimestampTz ||
           t == IcebergType::kTimestampNs || t == IcebergType::kTimestampTzNs;
}

TimeUnit unit_of(IcebergType t) noexcept {
    if (t == IcebergType::kDate) return TimeUnit::kDays;
    return (t == IcebergType::kTimestampNs || t == IcebergType::kTimestampTzNs) ? TimeUnit::kNanos : TimeUnit::kMicros;
}

// The literal as a value of the SOURCE type, false when its kind does not match (never coerced across kinds).
bool literal_value(const IcebergTypeInfo& src, const PredicateValue& lit, PartitionValue* out) noexcept {
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (is_int_source(src.type)) {
        if (!literal_i64(lit, &out->i64)) return false;
        if (src.type == IcebergType::kInt && (out->i64 < INT32_MIN || out->i64 > INT32_MAX)) return false;
        out->is_int = true;
        return true;
    }
    if (src.type == IcebergType::kFloat || src.type == IcebergType::kDouble) {
        int64_t i = 0;
        if (lit.type == BoltType::Float64 || lit.type == BoltType::Float32) out->f64 = lit.f64;
        else if (literal_i64(lit, &i)) out->f64 = static_cast<double>(i);
        else return false;
        out->is_float = true;
        return out->f64 == out->f64;                                      // a NaN literal projects nothing
    }
    if (src.type != IcebergType::kString && src.type != IcebergType::kBinary) return false;
    if ((lit.type != BoltType::Utf8 && lit.type != BoltType::Binary) || lit.str_len >= kLakeMaxValBytes) return false;
    out->is_str = true;
    out->str_len = static_cast<uint8_t>(lit.str_len);
    std::memcpy(out->str, lit.str, lit.str_len);
    return true;
}

// The partition value of source value `x` under `t` (an integer source for the time / int transforms).
bool transform_int(const Transform& t, IcebergType src, int64_t x, int64_t* out) noexcept {
    assert(out != nullptr);
    const TimeUnit u = unit_of(src);
    switch (t.kind) {
        case TransformKind::kTruncate:
            if (t.param <= 0) return false;
            if (src != IcebergType::kInt) return truncate_i64_checked(x, t.param, out);
            {
                int32_t r = 0;
                if (!truncate_i32_checked(static_cast<int32_t>(x), t.param, &r)) return false;   // no projection
                *out = r;
                return true;
            }
        case TransformKind::kYear:  *out = transform_year(x, u);  return true;
        case TransformKind::kMonth: *out = transform_month(x, u); return true;
        case TransformKind::kDay:   *out = transform_day(x, u);   return true;
        case TransformKind::kHour:  if (u == TimeUnit::kDays) return false; *out = transform_hour(x, u); return true;
        default:                    return false;
    }
}

// Monotone integer transforms (truncate on int/long, year/month/day/hour): x < a  ->  p <= T(a-1), etc.
bool project_monotone_int(const PartitionField& f, IcebergType src, PredicateOp op, int64_t a, Projected* out) noexcept {
    assert(out != nullptr);
    const int64_t lo = src == IcebergType::kInt ? INT32_MIN : INT64_MIN;
    const int64_t hi = src == IcebergType::kInt ? INT32_MAX : INT64_MAX;
    int64_t x = a;
    switch (op) {
        case PredicateOp::kEq: out->op = PredicateOp::kEq; break;
        case PredicateOp::kLe: out->op = PredicateOp::kLe; break;
        case PredicateOp::kGe: out->op = PredicateOp::kGe; break;
        case PredicateOp::kLt: if (a == lo) return false; x = a - 1; out->op = PredicateOp::kLe; break;
        case PredicateOp::kGt: if (a == hi) return false; x = a + 1; out->op = PredicateOp::kGe; break;
        default: return false;
    }
    int64_t p = 0;
    if (!transform_int(f.transform, src, x, &p)) return false;
    out->v.is_int = true;
    out->v.i64 = p;
    return true;
}

bool project_truncate_bytes(const PartitionField& f, PredicateOp op, const PartitionValue& a, Projected* out) noexcept {
    assert(out != nullptr && a.is_str);
    if (f.transform.param <= 0) return false;
    switch (op) {
        case PredicateOp::kEq: out->op = PredicateOp::kEq; break;
        case PredicateOp::kLt: case PredicateOp::kLe: out->op = PredicateOp::kLe; break;
        case PredicateOp::kGt: case PredicateOp::kGe: out->op = PredicateOp::kGe; break;
        default: return false;
    }
    out->v = a;
    out->v.str_len = static_cast<uint8_t>(truncate_utf8_len(a.str, a.str_len, f.transform.param));
    out->v.str[out->v.str_len] = '\0';
    return true;
}

// Project predicate `pr` (on the field's source column) to the partition value. False = nothing to prune on.
bool project(const PartitionField& f, const IcebergTypeInfo& src, const Predicate& pr, Projected* out) noexcept {
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    const TransformKind k = f.transform.kind;
    if (k == TransformKind::kVoid || k == TransformKind::kUnknown) return false;
    IcebergTypeInfo res{};
    if (!partition_result_type(f.transform, src, &res)) return false;
    if (pr.op == PredicateOp::kIsNull || pr.op == PredicateOp::kIsNotNull) { out->op = pr.op; return true; }
    PartitionValue a{};
    if (!literal_value(src, pr.value, &a)) return false;
    if (k == TransformKind::kIdentity) { out->op = pr.op; out->v = a; return true; }
    if (k == TransformKind::kBucket) {
        if (pr.op != PredicateOp::kEq || f.transform.param <= 0) return false;
        out->op = PredicateOp::kEq;
        out->v.is_int = true;
        if (a.is_str) out->v.i64 = bucket_bytes(a.str, a.str_len, f.transform.param);
        else if (unit_of(src.type) == TimeUnit::kNanos) out->v.i64 = bucket_i64(micros_from_nanos(a.i64), f.transform.param);
        else if (a.is_int) out->v.i64 = bucket_i64(a.i64, f.transform.param);
        else return false;
        return true;
    }
    if (a.is_int) return project_monotone_int(f, src.type, pr.op, a.i64, out);
    if (k == TransformKind::kTruncate && src.type == IcebergType::kString) return project_truncate_bytes(f, pr.op, a, out);
    if (k == TransformKind::kTruncate && src.type == IcebergType::kBinary) {
        if (!project_truncate_bytes(f, pr.op, a, out)) return false;
        out->v.str_len = static_cast<uint8_t>(truncate_binary_len(a.str_len, f.transform.param));
        return true;
    }
    return false;
}

// The projected predicate on one file's value of the field. 1 = may match, 0 = cannot.
int eval_value(const IcebergTypeInfo& res, const Projected& p, const PartitionValue& v) noexcept {
    if (v.is_partial) return 1;
    if (p.op == PredicateOp::kIsNull) return v.is_null ? 1 : 0;
    if (p.op == PredicateOp::kIsNotNull) return v.is_null ? 0 : 1;
    if (v.is_null) return 0;                                              // a NULL never satisfies a comparison
    if (v.is_float != p.v.is_float || v.is_str != p.v.is_str || (!v.is_str && !v.is_float && !v.is_int)) return 1;
    if (v.is_float && v.f64 != v.f64) return p.op == PredicateOp::kNe ? 1 : 0;   // NaN: only != holds
    const int c = partition_value_compare(res, v, p.v, false);
    switch (p.op) {
        case PredicateOp::kEq: return c == 0;
        case PredicateOp::kNe: return c != 0;
        case PredicateOp::kLt: return c < 0;
        case PredicateOp::kLe: return c <= 0;
        case PredicateOp::kGt: return c > 0;
        case PredicateOp::kGe: return c >= 0;
        default:               return 1;
    }
}

// The file's partition value for spec field `pi`, or null when absent.
const PartitionValue* value_for_spec_field(const DataFileRef* f, uint32_t pi, int32_t field_id) noexcept {
    assert(f != nullptr);
    if (f->partition_ordinal_ids) return pi < f->n_partition ? &f->partition[pi] : nullptr;
    for (uint32_t v = 0; v < f->n_partition; ++v)                         // bounded: n_partition
        if (f->partition[v].field_id == field_id) return &f->partition[v];
    return nullptr;
}

// A JSON fixture may carry an identity value of another kind than the column (a number for a string column, a string
// for a number): equality by rendered text, as before G2ICE-292. -1 = undecided.
int legacy_eq(const PartitionValue& pv, const PredicateValue& pred) noexcept {
    if (pv.is_null) return 0;
    char buf[32];
    const char* s = pv.str;
    uint32_t n = 0;
    if (pv.is_int) {
        if (pred.type == BoltType::Int64 || pred.type == BoltType::Int32) return pv.i64 == pred.i64 ? 1 : 0;
        const int w = std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(pv.i64));
        if (w <= 0) return -1;
        s = buf; n = static_cast<uint32_t>(w);
    } else if (pv.is_str) {
        n = partition_value_bytes_len(pv);
    } else {
        return -1;
    }
    if (pred.str_len == 0) return -1;
    return (n == pred.str_len && std::memcmp(s, pred.str, n) == 0) ? 1 : 0;
}

bool is_comparison(PredicateOp op) noexcept {
    return op != PredicateOp::kIsNull && op != PredicateOp::kIsNotNull;
}

const SchemaField* source_field(const Schema* sch, int32_t source_id) noexcept {
    assert(sch != nullptr);
    for (uint32_t s = 0; s < sch->n_fields; ++s)                         // bounded: n_fields
        if (sch->fields[s].id == source_id) return &sch->fields[s];
    return nullptr;
}

// One conjunct against one partition field of a file: false only when it provably excludes the file.
bool field_passes(const DataFileRef* f, const PartitionField& pf, uint32_t pi, const SchemaField& sf,
                  const Predicate& pr) noexcept {
    assert(f != nullptr);
    const PartitionValue* pv = value_for_spec_field(f, pi, pf.field_id);
    if (pv == nullptr) return true;
    // A null value under any known non-void transform means every source value is null: no comparison holds.
    const bool known = pf.transform.kind != TransformKind::kVoid && pf.transform.kind != TransformKind::kUnknown;
    if (known && pv->is_null && !pv->is_partial && is_comparison(pr.op)) return false;
    IcebergTypeInfo src{}, res{};
    const bool typed = iceberg_type_parse(sf.type, &src) && partition_result_type(pf.transform, src, &res);
    Projected p{};
    if (typed && project(pf, src, pr, &p)) {
        const bool kind_ok = pv->is_null || (p.op == PredicateOp::kIsNull || p.op == PredicateOp::kIsNotNull) ||
                             (pv->is_int == p.v.is_int && pv->is_str == p.v.is_str && pv->is_float == p.v.is_float);
        if (kind_ok) return eval_value(res, p, *pv) != 0;
    }
    if (pf.transform.kind == TransformKind::kIdentity && pr.op == PredicateOp::kEq) return legacy_eq(*pv, pr.value) != 0;
    return true;
}

}  // namespace

bool partition_passes(const DataFileRef* f, const PartitionSpec* spec, const Schema* schema, const Predicate* preds,
                      uint32_t n_preds) noexcept {
    assert(f != nullptr);
    assert(n_preds == 0 || preds != nullptr);
    if (spec == nullptr || schema == nullptr || n_preds == 0) return true;
    for (uint32_t p = 0; p < n_preds; ++p) {                              // bounded: kLakeMaxPredicates
        for (uint32_t pi = 0; pi < spec->n_fields; ++pi) {                // bounded: kIcebergMaxFieldsPerSpec
            const PartitionField& pf = spec->fields[pi];
            const SchemaField* sf = source_field(schema, pf.source_id);
            if (sf == nullptr || std::strcmp(sf->name, preds[p].column) != 0) continue;
            if (!field_passes(f, pf, pi, *sf, preds[p])) return false;
        }
    }
    return true;
}

namespace {

// The projected predicate over a manifest's summary of the field: false only when no value in [lower, upper] (and
// no null / NaN it records) can satisfy it.
bool summary_may_match(const PartitionFieldSummary& ps, const IcebergTypeInfo& res, const Projected& p) noexcept {
    if (p.op == PredicateOp::kIsNull) return ps.contains_null;
    if (p.op == PredicateOp::kIsNotNull)                                 // every value null: no bound, no NaN
        return !(ps.contains_null && !ps.has_lower && !ps.has_upper && !ps.bounds_dropped && ps.has_nan && !ps.contains_nan);
    if (!ps.has_lower || !ps.has_upper) return true;
    PartitionValue lo{}, hi{};
    if (!partition_value_decode(res, reinterpret_cast<const uint8_t*>(ps.lower), ps.lower_len, &lo) ||
        !partition_value_decode(res, reinterpret_cast<const uint8_t*>(ps.upper), ps.upper_len, &hi))
        return true;
    if (lo.is_int != p.v.is_int || lo.is_str != p.v.is_str || lo.is_float != p.v.is_float) return true;
    const int cl = partition_value_compare(res, lo, p.v, false);
    const int ch = partition_value_compare(res, hi, p.v, false);
    switch (p.op) {
        case PredicateOp::kEq: return cl <= 0 && ch >= 0;
        case PredicateOp::kNe: return !(cl == 0 && ch == 0) || (ps.has_nan && ps.contains_nan) || !ps.has_nan;
        case PredicateOp::kLt: return cl < 0;
        case PredicateOp::kLe: return cl <= 0;
        case PredicateOp::kGt: return ch > 0;
        case PredicateOp::kGe: return ch >= 0;
        default:               return true;
    }
}

}  // namespace

bool manifest_may_match(const ManifestListEntry* e, const PartitionSpec* spec, const Schema* schema,
                        const Predicate* preds, uint32_t n_preds) noexcept {
    assert(e != nullptr);
    assert(n_preds == 0 || preds != nullptr);
    if (spec == nullptr || schema == nullptr || n_preds == 0) return true;
    if (e->n_partitions == 0 || e->n_partitions > kIcebergMaxSummaries || e->n_partitions != spec->n_fields) return true;
    for (uint32_t p = 0; p < n_preds; ++p) {                              // bounded: kLakeMaxPredicates
        for (uint32_t i = 0; i < e->n_partitions; ++i) {                  // bounded: kIcebergMaxSummaries
            const PartitionField& pf = spec->fields[i];
            const SchemaField* sf = source_field(schema, pf.source_id);
            if (sf == nullptr || std::strcmp(sf->name, preds[p].column) != 0) continue;
            const PartitionFieldSummary& ps = e->partitions[i];
            const bool all_null = ps.contains_null && !ps.has_lower && !ps.has_upper && !ps.bounds_dropped &&
                                  ps.has_nan && !ps.contains_nan;
            const bool known = pf.transform.kind != TransformKind::kVoid && pf.transform.kind != TransformKind::kUnknown;
            if (known && all_null && is_comparison(preds[p].op)) return false;
            IcebergTypeInfo src{}, res{};
            if (!iceberg_type_parse(sf->type, &src) || !partition_result_type(pf.transform, src, &res)) continue;
            Projected pr{};
            if (!project(pf, src, preds[p], &pr)) continue;
            if (!summary_may_match(e->partitions[i], res, pr)) return false;
        }
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
