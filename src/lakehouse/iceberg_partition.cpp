// bolt/lakehouse/iceberg_partition.cpp — partition-value predicate matching.
// Identity-only equality for W4. Transform-aware pruning TODO(W5).
//
// G2FEAT-125: how a file's partition VALUE is matched to a spec FIELD depends
// on which parser produced the file. The JSON path reads a real "field-id" out
// of the document; the Avro path cannot (the flattened Avro field table does
// not model field-id annotations) and instead reports the ORDINAL within the
// partition struct. Iceberg writes that struct in spec order, so the ordinal
// join is exact — but joining an ordinal against a v2 spec's field-id (>=1000)
// matches NOTHING, which is why partition pruning did nothing at all for a
// real Avro manifest before this. `DataFileRef::partition_ordinal_ids` says
// which rule applies.

#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/metadata.h"

#include <cassert>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

// 1 = match, 0 = no, -1 = undecided.
int compare_eq(const PartitionValue& pv,
               const PredicateValue& pred) noexcept {
    if (pv.is_null) return 0;
    if (pv.is_int) {
        if (pred.type == BoltType::Int64 || pred.type == BoltType::Int32) {
            return pv.i64 == pred.i64 ? 1 : 0;
        }
        if (pred.str_len > 0) {
            char buf[32];
            const int n = std::snprintf(buf, sizeof(buf), "%lld",
                                        static_cast<long long>(pv.i64));
            if (n <= 0) return -1;
            return (static_cast<uint32_t>(n) == pred.str_len &&
                    std::memcmp(buf, pred.str, pred.str_len) == 0) ? 1 : 0;
        }
        return -1;
    }
    if (pv.is_str) {
        if (pred.str_len > 0) {
            const uint32_t n = static_cast<uint32_t>(std::strlen(pv.str));
            return (n == pred.str_len &&
                    std::memcmp(pv.str, pred.str, n) == 0) ? 1 : 0;
        }
        return -1;
    }
    return -1;
}

// The file's partition value for spec field `pi`, or null when absent.
const PartitionValue* value_for_spec_field(const DataFileRef* f,
                                           uint32_t pi,
                                           int32_t field_id) noexcept {
    assert(f != nullptr);
    if (f->partition_ordinal_ids) {
        // Ordinal join: partition[i] belongs to spec->fields[i], by Iceberg's
        // guarantee that the partition struct is written in spec order.
        if (pi >= f->n_partition) return nullptr;
        return &f->partition[pi];
    }
    for (uint32_t v = 0; v < f->n_partition; ++v) {   // bounded: n_partition
        if (f->partition[v].field_id == field_id) return &f->partition[v];
    }
    return nullptr;
}

}  // namespace

bool partition_passes(const DataFileRef* f, const PartitionSpec* spec,
                      const Schema* schema,
                      const Predicate* preds, uint32_t n_preds) noexcept {
    assert(f != nullptr);
    if (spec == nullptr || schema == nullptr) return true;
    if (n_preds == 0) return true;
    for (uint32_t p = 0; p < n_preds; ++p) {
        const Predicate& pr = preds[p];
        if (pr.op != PredicateOp::kEq) continue;
        for (uint32_t pi = 0; pi < spec->n_fields; ++pi) {
            const PartitionField& pf = spec->fields[pi];
            const char* src_name = nullptr;
            for (uint32_t s = 0; s < schema->n_fields; ++s) {
                if (schema->fields[s].id == pf.source_id) {
                    src_name = schema->fields[s].name; break;
                }
            }
            if (src_name == nullptr) continue;
            if (std::strcmp(src_name, pr.column) != 0) continue;
            if (pf.transform.kind != TransformKind::kIdentity) continue;
            const PartitionValue* pv =
                value_for_spec_field(f, pi, pf.field_id);
            if (pv == nullptr) continue;
            if (compare_eq(*pv, pr.value) == 0) return false;
        }
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

constexpr int64_t kDayUs  = INT64_C(86400000000);
constexpr int64_t kHourUs = INT64_C(3600000000);

int64_t floor_div(int64_t a, int64_t b) noexcept {
    assert(b > 0);
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

bool is_int_type(const char* t, bool* is_micros) noexcept {
    assert(t != nullptr && is_micros != nullptr);
    *is_micros = std::strcmp(t, "timestamp") == 0 ||
                 std::strcmp(t, "timestamptz") == 0;
    return *is_micros || std::strcmp(t, "int") == 0 ||
           std::strcmp(t, "long") == 0 || std::strcmp(t, "date") == 0;
}

bool decode_int_bound(const char* b, uint8_t len, int64_t* out) noexcept {
    assert(b != nullptr && out != nullptr);
    if (len == 4u) { int32_t v = 0; std::memcpy(&v, b, 4u); *out = v; return true; }
    if (len == 8u) { int64_t v = 0; std::memcpy(&v, b, 8u); *out = v; return true; }
    return false;
}

bool literal_i64(const PredicateValue& v, int64_t* out) noexcept {
    assert(out != nullptr);
    if (v.type == BoltType::Int64 || v.type == BoltType::Int32) {
        *out = v.i64;
        return true;
    }
    return false;
}

// The predicate's value mapped into the partition domain, and whether the
// mapping is a (monotone, non-strict) floor so Lt/Gt must widen to Le/Ge.
bool to_partition_domain(TransformKind k, bool src_micros, int64_t x,
                         int64_t* out, bool* floored) noexcept {
    assert(out != nullptr && floored != nullptr);
    *floored = false;
    if (k == TransformKind::kIdentity) { *out = x; return true; }
    if (!src_micros) return false;
    if (k == TransformKind::kDay)  { *out = floor_div(x, kDayUs);  *floored = true; return true; }
    if (k == TransformKind::kHour) { *out = floor_div(x, kHourUs); *floored = true; return true; }
    return false;
}

// False only when [lo, hi] provably holds no value satisfying `op x`.
bool range_may_match(PredicateOp op, int64_t lo, int64_t hi, int64_t x,
                     bool floored) noexcept {
    assert(lo <= hi);
    switch (op) {
        case PredicateOp::kEq: return x >= lo && x <= hi;
        case PredicateOp::kLt: return floored ? lo <= x : lo < x;
        case PredicateOp::kLe: return lo <= x;
        case PredicateOp::kGt: return floored ? hi >= x : hi > x;
        case PredicateOp::kGe: return hi >= x;
        default:               return true;
    }
}

const char* source_type(const Schema* sch, int32_t source_id,
                        const char** name) noexcept {
    assert(sch != nullptr && name != nullptr);
    for (uint32_t s = 0; s < sch->n_fields; ++s) {       // bounded: n_fields
        if (sch->fields[s].id == source_id) {
            *name = sch->fields[s].name;
            return sch->fields[s].type;
        }
    }
    *name = nullptr;
    return nullptr;
}

bool field_may_match(const PartitionFieldSummary& ps, const PartitionField& pf,
                     const char* type, const Predicate& pr) noexcept {
    assert(type != nullptr);
    bool micros = false;
    if (!is_int_type(type, &micros)) return true;
    if (!ps.has_lower || !ps.has_upper) return true;
    int64_t lo = 0, hi = 0, x = 0, px = 0;
    if (!decode_int_bound(ps.lower, ps.lower_len, &lo) ||
        !decode_int_bound(ps.upper, ps.upper_len, &hi) || lo > hi)
        return true;
    if (!literal_i64(pr.value, &x)) return true;
    bool floored = false;
    if (!to_partition_domain(pf.transform.kind, micros, x, &px, &floored))
        return true;
    return range_may_match(pr.op, lo, hi, px, floored);
}

}  // namespace

bool manifest_may_match(const ManifestListEntry* e, const PartitionSpec* spec,
                        const Schema* schema, const Predicate* preds,
                        uint32_t n_preds) noexcept {
    assert(e != nullptr);
    assert(n_preds == 0 || preds != nullptr);
    if (spec == nullptr || schema == nullptr || n_preds == 0) return true;
    if (e->n_partitions == 0 || e->n_partitions > kIcebergMaxSummaries ||
        e->n_partitions != spec->n_fields)
        return true;
    for (uint32_t p = 0; p < n_preds; ++p) {                  // bounded
        const Predicate& pr = preds[p];
        for (uint32_t i = 0; i < e->n_partitions; ++i) {      // bounded
            const PartitionField& pf = spec->fields[i];
            const char* name = nullptr;
            const char* type = source_type(schema, pf.source_id, &name);
            if (type == nullptr || std::strcmp(name, pr.column) != 0) continue;
            if (!field_may_match(e->partitions[i], pf, type, pr)) return false;
        }
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
