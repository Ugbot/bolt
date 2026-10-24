// src/lakehouse/iceberg_dml_read.cpp — live rows, row positions and predicate
// matches for the Iceberg writer's delete / overwrite / compaction (G2ICE-280).
// Reads through the table scan; see iceberg_dml_read.h.

#include "iceberg_dml_read.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/kernels/bolt_filter_null.h"
#include "bolt/lakehouse/iceberg/scan.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {
namespace dml {

namespace {

namespace flt = ::bolt::kernels::filter;
using flt::CmpOp;

// The column's validity as LSB-first 64-bit words (the kernels' shape), or
// nullptr when it has no nulls.
bool validity_words(const BoltColumn& c, int64_t rows, Arena* a,
                    const uint64_t** out) noexcept {
    assert(a != nullptr && out != nullptr);
    assert(rows >= 0);
    *out = nullptr;
    if (c.validity == nullptr || c.stats.all_valid) return true;
    const size_t words = static_cast<size_t>(rows + 63) / 64u + 1u;
    auto* w = static_cast<uint64_t*>(a->allocate_zeroed(words * sizeof(uint64_t)));
    if (w == nullptr) return false;
    for (int64_t r = 0; r < rows; ++r) {                       // bounded by rows
        const int64_t bit = c.validity_offset + r;
        const uint64_t v = (c.validity[bit >> 3] >> (bit & 7)) & 1u;
        w[r >> 6] |= v << (r & 63);
    }
    *out = w;
    return true;
}

template <typename T>
size_t cmp_dispatch(PredicateOp op, uint32_t* sel, size_t n, const T* v, T lit,
                    const uint64_t* nulls) noexcept {
    assert(sel != nullptr && (v != nullptr || n == 0));
    assert(op != PredicateOp::kIsNull && op != PredicateOp::kIsNotNull);
    switch (op) {
        case PredicateOp::kEq: return flt::cmp_scalar<CmpOp::Eq, T>(sel, nullptr, 0, n, v, lit, nulls);
        case PredicateOp::kNe: return flt::cmp_scalar<CmpOp::Ne, T>(sel, nullptr, 0, n, v, lit, nulls);
        case PredicateOp::kLt: return flt::cmp_scalar<CmpOp::Lt, T>(sel, nullptr, 0, n, v, lit, nulls);
        case PredicateOp::kLe: return flt::cmp_scalar<CmpOp::Le, T>(sel, nullptr, 0, n, v, lit, nulls);
        case PredicateOp::kGt: return flt::cmp_scalar<CmpOp::Gt, T>(sel, nullptr, 0, n, v, lit, nulls);
        default:               return flt::cmp_scalar<CmpOp::Ge, T>(sel, nullptr, 0, n, v, lit, nulls);
    }
}

size_t utf8_dispatch(PredicateOp op, uint32_t* sel, size_t n, const StringView* v,
                     const uint8_t* ovf, const uint8_t* lit, uint32_t len,
                     const uint64_t* nulls) noexcept {
    assert(sel != nullptr && (v != nullptr || n == 0));
    assert(lit != nullptr || len == 0);
    switch (op) {
        case PredicateOp::kEq: return flt::cmp_scalar_utf8<CmpOp::Eq>(sel, n, v, ovf, lit, len, nulls);
        case PredicateOp::kNe: return flt::cmp_scalar_utf8<CmpOp::Ne>(sel, n, v, ovf, lit, len, nulls);
        case PredicateOp::kLt: return flt::cmp_scalar_utf8<CmpOp::Lt>(sel, n, v, ovf, lit, len, nulls);
        case PredicateOp::kLe: return flt::cmp_scalar_utf8<CmpOp::Le>(sel, n, v, ovf, lit, len, nulls);
        case PredicateOp::kGt: return flt::cmp_scalar_utf8<CmpOp::Gt>(sel, n, v, ovf, lit, len, nulls);
        default:               return flt::cmp_scalar_utf8<CmpOp::Ge>(sel, n, v, ovf, lit, len, nulls);
    }
}

bool literal_is_int(const PredicateValue& v) noexcept {
    return bolt::is_integer(v.type) || v.type == BoltType::Bool || v.type == BoltType::Date32 ||
           v.type == BoltType::Timestamp;
}

bool literal_is_float(const PredicateValue& v) noexcept {
    return v.type == BoltType::Float32 || v.type == BoltType::Float64;
}

// Integer-shaped column (any width) against an integer literal: widened to
// int64 so a literal outside the column's range compares exactly.
ReadStatus cmp_integer(const BoltColumn& c, int64_t rows, const Predicate* p,
                       const uint64_t* nulls, Arena* a, uint32_t* sel,
                       uint32_t* n) noexcept {
    assert(p != nullptr && sel != nullptr && n != nullptr);
    if (!literal_is_int(p->value)) return ReadStatus::kUnsupported;
    const int64_t* v = static_cast<const int64_t*>(c.data);
    if (c.type_size_bytes != 8u) {
        auto* w = a->allocate_array<int64_t>(static_cast<size_t>(rows) + 1u);
        if (w == nullptr) return ReadStatus::kFailed;
        for (int64_t r = 0; r < rows; ++r) {                   // bounded by rows
            switch (c.type_size_bytes) {
                case 1:  w[r] = static_cast<const int8_t*>(c.data)[r];  break;
                case 2:  w[r] = static_cast<const int16_t*>(c.data)[r]; break;
                case 4:  w[r] = static_cast<const int32_t*>(c.data)[r]; break;
                default: return ReadStatus::kUnsupported;
            }
        }
        v = w;
    }
    *n = static_cast<uint32_t>(cmp_dispatch<int64_t>(p->op, sel, static_cast<size_t>(rows), v, p->value.i64, nulls));
    return ReadStatus::kOk;
}

ReadStatus cmp_float(const BoltColumn& c, int64_t rows, const Predicate* p,
                     const uint64_t* nulls, Arena* a, uint32_t* sel,
                     uint32_t* n) noexcept {
    assert(p != nullptr && sel != nullptr && n != nullptr);
    const bool int_lit = literal_is_int(p->value);
    if (!int_lit && !literal_is_float(p->value)) return ReadStatus::kUnsupported;
    const double lit = int_lit ? static_cast<double>(p->value.i64) : p->value.f64;
    const double* v = static_cast<const double*>(c.data);
    if (c.type == BoltType::Float32) {
        auto* w = a->allocate_array<double>(static_cast<size_t>(rows) + 1u);
        if (w == nullptr) return ReadStatus::kFailed;
        const float* f = static_cast<const float*>(c.data);
        for (int64_t r = 0; r < rows; ++r) w[r] = static_cast<double>(f[r]);   // bounded by rows
        v = w;
    }
    *n = static_cast<uint32_t>(cmp_dispatch<double>(p->op, sel, static_cast<size_t>(rows), v, lit, nulls));
    return ReadStatus::kOk;
}

ReadStatus cmp_string(const BoltColumn& c, int64_t rows, const Predicate* p,
                      const uint64_t* nulls, uint32_t* sel, uint32_t* n) noexcept {
    assert(p != nullptr && sel != nullptr && n != nullptr);
    if (p->value.type != BoltType::Utf8 && p->value.type != BoltType::Binary) return ReadStatus::kUnsupported;
    if (p->value.str_len > kLakeMaxValBytes) return ReadStatus::kUnsupported;
    const auto* v = static_cast<const StringView*>(c.data);
    const auto* ovf = static_cast<const uint8_t*>(c.str_overflow_base);
    for (int64_t r = 0; r < rows && ovf == nullptr; ++r)                       // bounded by rows
        if (v[r].length > 12u && (nulls == nullptr || flt::bitmap_get(nulls, static_cast<size_t>(r))))
            return ReadStatus::kUnsupported;   // a spilled value with no buffer cannot be compared
    *n = static_cast<uint32_t>(utf8_dispatch(p->op, sel, static_cast<size_t>(rows), v, ovf,
        reinterpret_cast<const uint8_t*>(p->value.str), p->value.str_len, nulls));
    return ReadStatus::kOk;
}

int32_t find_column(const BoltBatch* b, uint32_t n_cols, const char* name) noexcept {
    assert(b != nullptr && name != nullptr);
    for (uint32_t i = 0; i < n_cols; ++i)                      // bounded by n_cols
        if (std::strcmp(b->schema.field(static_cast<int>(i)).name, name) == 0) return static_cast<int32_t>(i);
    return -1;
}

}  // namespace

ReadStatus eval_predicate(const BoltBatch* b, uint32_t n_cols, const Predicate* p,
                          Arena* a, uint32_t** out, uint32_t* n_out) noexcept {
    assert(b != nullptr && p != nullptr);
    assert(a != nullptr && out != nullptr && n_out != nullptr);
    *n_out = 0;
    const int32_t ci = find_column(b, n_cols, p->column);
    if (ci < 0) {
        std::fprintf(stderr, "[iceberg_writer] predicate column '%s' is not a column of the table\n", p->column);
        return ReadStatus::kFailed;
    }
    const int64_t rows = b->num_rows;
    auto* sel = a->allocate_array<uint32_t>(static_cast<size_t>(rows) + 1u);
    if (sel == nullptr) return ReadStatus::kFailed;
    *out = sel;
    if (rows == 0) return ReadStatus::kOk;
    const BoltColumn& c = b->columns[b->read_epoch][ci];
    if (c.format != ColumnFormat::Flat || c.data == nullptr || c.length < rows) return ReadStatus::kUnsupported;
    const uint64_t* nulls = nullptr;
    if (!validity_words(c, rows, a, &nulls)) return ReadStatus::kFailed;
    if (p->op == PredicateOp::kIsNull || p->op == PredicateOp::kIsNotNull) {
        *n_out = static_cast<uint32_t>(p->op == PredicateOp::kIsNull
            ? flt::cmp_is_null(sel, nullptr, 0, static_cast<size_t>(rows), nulls)
            : flt::cmp_is_not_null(sel, nullptr, 0, static_cast<size_t>(rows), nulls));
        return ReadStatus::kOk;
    }
    if (c.type == BoltType::Utf8 || c.type == BoltType::Binary) return cmp_string(c, rows, p, nulls, sel, n_out);
    if (c.type == BoltType::Float32 || c.type == BoltType::Float64) return cmp_float(c, rows, p, nulls, a, sel, n_out);
    if (bolt::is_integer(c.type) || c.type == BoltType::Bool || c.type == BoltType::Date32 ||
        c.type == BoltType::Timestamp)
        return cmp_integer(c, rows, p, nulls, a, sel, n_out);
    return ReadStatus::kUnsupported;
}

namespace {

struct ReadCtx {
    Arena*          arena;
    const Predicate* pred;
    RowVisitFn      visit;
    void*           ctx;
};

// One scan batch: strip `_pos`, evaluate the predicate, hand it on.
ReadStatus deliver(ReadCtx* rc, BoltBatch* b, const IcebergScanGroup& g) noexcept {
    assert(rc != nullptr && b != nullptr);
    assert(b->num_cols >= 1u);
    const uint32_t n_cols = b->num_cols - 1u;
    const BoltColumn& pc = b->columns[b->read_epoch][n_cols];
    if (pc.type != BoltType::Int64 || pc.format != ColumnFormat::Flat || pc.length != b->num_rows)
        return ReadStatus::kFailed;
    RowBatch rb{};
    rb.file_path = g.file_path;
    rb.batch = b;
    rb.pos = static_cast<const int64_t*>(pc.data);
    if (rc->pred != nullptr) {
        uint32_t* sel = nullptr;
        const ReadStatus st = eval_predicate(b, n_cols, rc->pred, rc->arena, &sel, &rb.n_match);
        if (st != ReadStatus::kOk) return st;
        rb.match = sel;
    }
    b->num_cols = n_cols;
    return rc->visit(rc->ctx, &rb) ? ReadStatus::kOk : ReadStatus::kFailed;
}

}  // namespace

ReadStatus read_live_rows(Arena* arena, const ObjectStore* store, const uint8_t* meta_json,
                          uint32_t meta_len, int64_t snapshot_id, const Predicate* pred,
                          FileKeepFn keep, void* keep_ctx, RowVisitFn visit, void* ctx) noexcept {
    assert(arena != nullptr && store != nullptr && meta_json != nullptr);
    assert(visit != nullptr && snapshot_id > 0);
    TableHandle* rh = nullptr;
    if (!iceberg_table_open_on_store(&rh, arena, store, "", meta_json, meta_len)) return ReadStatus::kFailed;
    auto* ro = arena->allocate_array<ReadOptions>(1);
    if (ro == nullptr) return ReadStatus::kFailed;
    read_options_init(ro);
    ro->snapshot_id = snapshot_id;
    ro->snapshot_selection = SnapshotSelection::Exact;
    ro->emit_row_position = true;
    if (pred != nullptr) { ro->predicates[0] = *pred; ro->n_predicates = 1u; }
    ScanHandle* sh = nullptr;
    if (!iceberg_scan_open(&sh, rh, ro)) return ReadStatus::kFailed;
    if (keep != nullptr) {
        const IcebergFileFilter f{keep, keep_ctx};
        iceberg_scan_set_file_filter(sh, &f);
    }
    ReadCtx rc{arena, pred, visit, ctx};
    ReadStatus st = ReadStatus::kOk;
    for (uint64_t guard = 0; guard < (UINT64_C(1) << 40); ++guard) {   // bounded: the scan ends
        auto* b = arena->allocate_array<BoltBatch>(1);
        if (b == nullptr) { st = ReadStatus::kFailed; break; }
        bool eof = false;
        if (!iceberg_scan_next_batch(sh, b, &eof)) { st = ReadStatus::kFailed; break; }
        if (eof) break;
        IcebergScanGroup g{};
        if (!iceberg_scan_last_group(sh, &g)) { st = ReadStatus::kFailed; break; }
        st = deliver(&rc, b, g);
        if (st != ReadStatus::kOk) break;
    }
    iceberg_scan_close(sh);
    iceberg_table_close(rh);
    return st;
}

bool position_delete_targets(Arena* a, const uint8_t* body, uint64_t len,
                             const char*** out, uint32_t* n_out) noexcept {
    namespace pq = ::bolt::ingest::parquet;
    assert(a != nullptr && body != nullptr);
    assert(out != nullptr && n_out != nullptr);
    *n_out = 0;
    auto* meta = a->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) return false;
    std::memset(meta, 0, sizeof(*meta));
    if (!pq::parquet_read_meta(body, len, a, meta)) return false;
    int32_t pc = -1;
    for (uint32_t c = 0; c < meta->n_columns; ++c)              // bounded by n_columns
        if ((meta->columns[c].has_field_id && meta->columns[c].field_id == 2147483546) ||
            std::strcmp(meta->columns[c].name, "file_path") == 0) pc = static_cast<int32_t>(c);
    if (pc < 0) return false;
    const char** paths = nullptr;
    uint32_t n = 0, cap = 0;
    auto* cols = a->allocate_array<BoltColumn>(meta->n_columns);
    if (cols == nullptr) return false;
    for (uint32_t rg = 0; rg < meta->n_row_groups; ++rg) {     // bounded by n_row_groups
        std::memset(cols, 0, sizeof(BoltColumn) * meta->n_columns);
        int64_t rows = 0;
        if (!pq::parquet_read_row_group(body, len, meta, rg, a, cols, &rows)) return false;
        for (int64_t r = 0; r < rows; ++r) {                    // bounded by rows
            const uint8_t* d = nullptr; int32_t dl = 0;
            if (cols[pc].is_null(r)) return false;               // a required column
            cols[pc].utf8_at(r, &d, &dl);
            if (dl <= 0) return false;
            if (n > 0 && std::strlen(paths[n - 1]) == static_cast<size_t>(dl) &&
                std::memcmp(paths[n - 1], d, static_cast<size_t>(dl)) == 0) continue;   // sorted: runs per file
            if (n == cap) {
                const uint32_t nc = cap == 0 ? 16u : cap * 2u;
                auto* g = a->allocate_array<const char*>(nc);
                if (g == nullptr) return false;
                if (n != 0) std::memcpy(g, paths, sizeof(const char*) * n);
                paths = g; cap = nc;
            }
            auto* s = a->allocate_array<char>(static_cast<size_t>(dl) + 1u);
            if (s == nullptr) return false;
            std::memcpy(s, d, static_cast<size_t>(dl));
            s[dl] = '\0';
            paths[n++] = s;
        }
    }
    *out = paths;
    *n_out = n;
    return true;
}

}  // namespace dml
}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
