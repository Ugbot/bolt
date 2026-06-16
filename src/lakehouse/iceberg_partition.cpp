// bolt/lakehouse/iceberg_partition.cpp — partition-value predicate matching.
// Identity-only equality for W4. Transform-aware pruning TODO(W5).

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
            for (uint32_t v = 0; v < f->n_partition; ++v) {
                if (f->partition[v].field_id != pf.field_id) continue;
                const int r = compare_eq(f->partition[v], pr.value);
                if (r == 0) return false;
                break;
            }
        }
    }
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
