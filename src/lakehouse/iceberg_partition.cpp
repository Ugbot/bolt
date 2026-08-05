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
