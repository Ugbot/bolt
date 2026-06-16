// bolt/lakehouse/iceberg_statistics.cpp — stat-based predicate prune.

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
        int64_t pv = 0;
        if (!pred_as_i64(pr.value, &pv)) continue;
        int64_t lo = 0, hi = 0;
        bool have_lo = false, have_hi = false;
        if (cs->has_lower && parse_i64(cs->lower, &lo)) have_lo = true;
        if (cs->has_upper && parse_i64(cs->upper, &hi)) have_hi = true;
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
