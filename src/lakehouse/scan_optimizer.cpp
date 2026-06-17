// bolt/lakehouse/scan_optimizer.cpp — W8 selection-vector + plan helpers.

#include "bolt/lakehouse/scan_optimizer.h"

#include <cstring>

namespace bolt {
namespace lakehouse {

void sel_identity(uint32_t* sel, uint32_t n) noexcept {
    assert(sel != nullptr);
    assert(n <= kScanMorselMaxRows);
    if (sel == nullptr) return;
    for (uint32_t i = 0; i < n; ++i) sel[i] = i;
}

uint32_t sel_filter_passthrough(
    const uint32_t* sel_in, uint32_t sel_len,
    uint32_t* sel_out) noexcept {
    assert(sel_in != nullptr);
    assert(sel_out != nullptr);
    assert(sel_len <= kScanMorselMaxRows);
    if (sel_in == nullptr || sel_out == nullptr) return 0;
    if (sel_out != sel_in) {
        std::memcpy(sel_out, sel_in, sizeof(uint32_t) * sel_len);
    }
    return sel_len;
}

uint32_t sel_filter_mark(
    const uint32_t* sel_in, uint32_t sel_len,
    const uint8_t* mark, uint32_t* sel_out) noexcept {
    assert(sel_in != nullptr);
    assert(mark != nullptr);
    assert(sel_out != nullptr);
    assert(sel_len <= kScanMorselMaxRows);
    if (sel_in == nullptr || mark == nullptr || sel_out == nullptr) return 0;
    // Branchless: write sel_in[i] always, advance count by (mark==0).
    uint32_t count = 0;
    for (uint32_t i = 0; i < sel_len; ++i) {
        sel_out[count] = sel_in[i];
        count += (mark[i] == 0 ? 1u : 0u);
    }
    return count;
}

bool column_prune_plan(
    const ReadOptions* opts,
    const char* const* src_names, uint32_t n_src,
    ColumnPruneResult* out) noexcept {
    assert(opts != nullptr);
    assert(out != nullptr);
    assert(src_names != nullptr || n_src == 0);
    if (opts == nullptr || out == nullptr) return false;
    if (n_src > kLakeMaxProjection) return false;
    std::memset(out, 0, sizeof(*out));
    out->n_source = n_src;
    if (opts->n_projection == 0) {
        out->all_columns = true;
        out->n_keep = n_src;
        for (uint32_t i = 0; i < n_src; ++i) out->keep[i] = i;
        return true;
    }
    out->all_columns = false;
    // Walk source-schema order so output indices are ascending.
    for (uint32_t i = 0; i < n_src; ++i) {
        for (uint32_t j = 0; j < opts->n_projection; ++j) {
            if (std::strcmp(src_names[i], opts->projection[j]) == 0) {
                out->keep[out->n_keep++] = i;
                break;
            }
        }
        if (out->n_keep == kLakeMaxProjection) break;
    }
    return true;
}

namespace {

// Resolve `p`'s comparison value to int64 if possible. Returns true if
// usable for stats analysis at this layer.
bool predicate_as_i64(const Predicate* p, int64_t* out) noexcept {
    assert(p != nullptr);
    assert(out != nullptr);
    if (p->value.str_len > 0) return false;
    if (p->value.type == BoltType::Float64) return false;
    *out = p->value.i64;
    return true;
}

int col_lookup(const char* name, const char* const* names,
               uint32_t n) noexcept {
    assert(name != nullptr);
    assert(names != nullptr || n == 0);
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(names[i], name) == 0) return static_cast<int>(i);
    }
    return -1;
}

PushdownVerdict pushdown_one(
    const RowGroupStats* s, int slot, const Predicate* p) noexcept {
    assert(s != nullptr);
    assert(p != nullptr);
    assert(slot >= 0);
    if (p->op == PredicateOp::kIsNull) {
        return s->null_count[slot] != 0
            ? PushdownVerdict::kIndeterminate
            : PushdownVerdict::kSkip;
    }
    if (p->op == PredicateOp::kIsNotNull) {
        if (s->num_rows == 0) return PushdownVerdict::kSkip;
        return s->null_count[slot] < static_cast<int64_t>(s->num_rows)
            ? PushdownVerdict::kIndeterminate
            : PushdownVerdict::kSkip;
    }
    int64_t v = 0;
    if (!predicate_as_i64(p, &v)) return PushdownVerdict::kIndeterminate;
    const int64_t lo = s->min_i64[slot];
    const int64_t hi = s->max_i64[slot];
    switch (p->op) {
        case PredicateOp::kEq:
            if (v < lo || v > hi) return PushdownVerdict::kSkip;
            if (lo == hi && lo == v) return PushdownVerdict::kAllPass;
            return PushdownVerdict::kIndeterminate;
        case PredicateOp::kLt:
            if (lo >= v) return PushdownVerdict::kSkip;
            if (hi < v)  return PushdownVerdict::kAllPass;
            return PushdownVerdict::kIndeterminate;
        case PredicateOp::kLe:
            if (lo > v)  return PushdownVerdict::kSkip;
            if (hi <= v) return PushdownVerdict::kAllPass;
            return PushdownVerdict::kIndeterminate;
        case PredicateOp::kGt:
            if (hi <= v) return PushdownVerdict::kSkip;
            if (lo > v)  return PushdownVerdict::kAllPass;
            return PushdownVerdict::kIndeterminate;
        case PredicateOp::kGe:
            if (hi < v)  return PushdownVerdict::kSkip;
            if (lo >= v) return PushdownVerdict::kAllPass;
            return PushdownVerdict::kIndeterminate;
        case PredicateOp::kNe:
            if (lo == hi && lo == v) return PushdownVerdict::kSkip;
            return PushdownVerdict::kIndeterminate;
        default: return PushdownVerdict::kIndeterminate;
    }
}

}  // namespace

PushdownVerdict predicate_pushdown_stats(
    const RowGroupStats* stats,
    const char* const* col_names,
    const Predicate* preds, uint32_t n_preds) noexcept {
    assert(stats != nullptr);
    assert(col_names != nullptr || stats->n_cols == 0);
    assert(preds != nullptr || n_preds == 0);
    if (n_preds == 0) return PushdownVerdict::kAllPass;
    if (!stats->has_minmax) return PushdownVerdict::kIndeterminate;
    bool all_pass = true;
    for (uint32_t i = 0; i < n_preds; ++i) {
        const int slot = col_lookup(preds[i].column, col_names, stats->n_cols);
        if (slot < 0) { all_pass = false; continue; }
        const PushdownVerdict v = pushdown_one(stats, slot, &preds[i]);
        if (v == PushdownVerdict::kSkip) return PushdownVerdict::kSkip;
        if (v == PushdownVerdict::kIndeterminate) all_pass = false;
    }
    // Upstream backlog: page-index reader (Parquet OffsetIndex /
    // ColumnIndex) — needs bolt::ingest::parquet_read_page_index.
    // Upstream backlog: bloom-filter reader — needs
    // bolt::ingest::parquet_read_bloom_filter. Both fall back to
    // kIndeterminate today; the kSkip / kAllPass code paths above are
    // already complete for stats-level pushdown.
    return all_pass ? PushdownVerdict::kAllPass : PushdownVerdict::kIndeterminate;
}

namespace {

// One-predicate partition test. Eq-only today (matches the existing
// per-format implementations); other ops fall through as "indeterminate
// → keep".
bool partition_passes_one(const PartitionedFile* f,
                          const Predicate* p) noexcept {
    assert(f != nullptr);
    assert(p != nullptr);
    if (p->op != PredicateOp::kEq) return true;
    for (uint32_t i = 0; i < f->n_parts; ++i) {
        if (std::strcmp(f->parts[i].key, p->column) != 0) continue;
        if (p->value.str_len == 0) return true;
        if (f->parts[i].is_null) return false;
        return std::strcmp(f->parts[i].value, p->value.str) == 0;
    }
    return true;
}

}  // namespace

bool partition_prune(
    const PartitionedFile* files, uint32_t n_files,
    const Predicate* preds, uint32_t n_preds,
    uint32_t* out_keep_idx, uint32_t* out_n) noexcept {
    assert(files != nullptr || n_files == 0);
    assert(preds != nullptr || n_preds == 0);
    assert(out_keep_idx != nullptr);
    assert(out_n != nullptr);
    if (out_keep_idx == nullptr || out_n == nullptr) return false;
    const uint32_t cap = *out_n;
    uint32_t kept = 0;
    for (uint32_t i = 0; i < n_files; ++i) {
        bool ok = true;
        for (uint32_t j = 0; j < n_preds; ++j) {
            if (!partition_passes_one(&files[i], &preds[j])) {
                ok = false; break;
            }
        }
        if (!ok) continue;
        if (kept >= cap) return false;
        out_keep_idx[kept++] = i;
    }
    *out_n = kept;
    return true;
}

}  // namespace lakehouse
}  // namespace bolt
