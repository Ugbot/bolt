// bolt/lakehouse/iceberg_parquet_stats.cpp — Iceberg data-file stats from a
// parquet footer (G2ICE-43).
//
// The footer already holds, per row group, the null count and the min/max a
// spec-compliant writer computed over exactly the bytes in the file. Folding
// those is the whole job: no row is re-read. A bound this emits is only ever
// a bound some row group carried, re-encoded not at all — it is kept only
// where parquet's PLAIN stat encoding and Iceberg's single-value encoding are
// the same bytes. A wrong bound prunes live rows silently, so every doubt
// (legacy byte-array stats, millisecond timestamps, a row group without a
// stat, an unknown null count) degrades to "no bound", never to a value.

#include "bolt/lakehouse/iceberg/statistics.h"

#include "bolt/ingest/bolt_parquet_meta.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

namespace pq = ::bolt::ingest::parquet;

enum class BoundKind : uint8_t { kNone, kInt32, kInt64, kFloat, kDouble, kBool, kBytes };

BoundKind bound_kind(const pq::PqColumn& c) noexcept {
    const auto lg = static_cast<pq::PqLogical>(c.logical);
    const bool plain = lg == pq::PqLogical::None && c.converted < 0;
    const bool sint  = lg == pq::PqLogical::Int && c.int_signed != 0u;
    switch (c.physical) {
        case pq::PqType::Int32:
            if (plain || lg == pq::PqLogical::Date ||
                (sint && c.int_bits == 32u)) {
                return BoundKind::kInt32;
            }
            return BoundKind::kNone;
        case pq::PqType::Int64:
            // Millisecond timestamps are Iceberg micros only after scaling.
            if (plain || (sint && c.int_bits == 64u) ||
                (lg == pq::PqLogical::Timestamp &&
                 (c.time_unit == 2 || c.time_unit == 3))) {
                return BoundKind::kInt64;
            }
            return BoundKind::kNone;
        case pq::PqType::Float:   return plain ? BoundKind::kFloat : BoundKind::kNone;
        case pq::PqType::Double:  return plain ? BoundKind::kDouble : BoundKind::kNone;
        case pq::PqType::Boolean: return BoundKind::kBool;
        case pq::PqType::ByteArray:
            // ConvertedType UTF8 is 0.
            if (lg == pq::PqLogical::String || c.converted == 0) return BoundKind::kBytes;
            return BoundKind::kNone;
        default:
            return BoundKind::kNone;
    }
}

uint32_t fixed_width(BoundKind k) noexcept {
    switch (k) {
        case BoundKind::kInt32:
        case BoundKind::kFloat:  return 4u;
        case BoundKind::kInt64:
        case BoundKind::kDouble: return 8u;
        case BoundKind::kBool:   return 1u;
        default:                 return 0u;
    }
}

// <0 / 0 / >0 in the column's Iceberg sort order. Signed zeros compare by
// sign so a lower bound keeps -0.0 and an upper bound keeps +0.0.
int cmp_bound(BoundKind k, const uint8_t* a, uint32_t al, const uint8_t* b,
              uint32_t bl) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(k != BoundKind::kNone);
    switch (k) {
        case BoundKind::kInt32: {
            int32_t x, y; std::memcpy(&x, a, 4); std::memcpy(&y, b, 4);
            return (x > y) - (x < y);
        }
        case BoundKind::kInt64: {
            int64_t x, y; std::memcpy(&x, a, 8); std::memcpy(&y, b, 8);
            return (x > y) - (x < y);
        }
        case BoundKind::kFloat:
        case BoundKind::kDouble: {
            double x, y;
            if (k == BoundKind::kFloat) {
                float fx, fy; std::memcpy(&fx, a, 4); std::memcpy(&fy, b, 4);
                x = fx; y = fy;
            } else {
                std::memcpy(&x, a, 8); std::memcpy(&y, b, 8);
            }
            if (x != y) return x < y ? -1 : 1;
            const bool nx = std::signbit(x), ny = std::signbit(y);
            return nx == ny ? 0 : (nx ? -1 : 1);
        }
        case BoundKind::kBool: return (a[0] > b[0]) - (a[0] < b[0]);
        case BoundKind::kBytes: {
            const uint32_t n = al < bl ? al : bl;
            const int c = n == 0u ? 0 : std::memcmp(a, b, n);
            if (c != 0) return c;
            return (al > bl) - (al < bl);
        }
        default: return 0;
    }
}

// A chunk's own bound is usable only if present, the right width, trusted
// for its type, and NaN-free (a float stat of NaN orders nothing).
bool chunk_bound_ok(BoundKind k, const pq::PqChunk& ch, bool want_min) noexcept {
    const uint32_t len = want_min ? ch.min_len : ch.max_len;
    const uint8_t* b   = want_min ? ch.min_bytes : ch.max_bytes;
    if (len == 0u || len > kLakeMaxValBytes) return false;
    if (k == BoundKind::kBytes) {
        const uint32_t flag = want_min ? pq::kPqStatMinIsValueField
                                       : pq::kPqStatMaxIsValueField;
        return (ch.stats_flags & flag) != 0u;
    }
    if (len != fixed_width(k)) return false;
    if (k == BoundKind::kFloat) { float f; std::memcpy(&f, b, 4); return !std::isnan(f); }
    if (k == BoundKind::kDouble) { double d; std::memcpy(&d, b, 8); return !std::isnan(d); }
    return true;
}

struct ColFold {
    int64_t  nulls;
    bool     nulls_known;
    bool     lo_ok, hi_ok;     // still eligible for a bound
    bool     have;             // at least one non-all-null row group seen
    uint32_t lo_len, hi_len;
    uint8_t  lo[kLakeMaxValBytes];
    uint8_t  hi[kLakeMaxValBytes];
};

void fold_chunk(BoundKind k, const pq::PqChunk& ch, ColFold* f) noexcept {
    assert(f != nullptr);
    assert(ch.num_values >= 0);
    if (ch.null_count < 0) f->nulls_known = false;
    else                   f->nulls += ch.null_count;
    // An all-null chunk has no bound to offer and none to withhold.
    if (ch.null_count >= 0 && ch.null_count == ch.num_values) return;
    const bool lo_in = chunk_bound_ok(k, ch, true);
    const bool hi_in = chunk_bound_ok(k, ch, false);
    if (!lo_in) f->lo_ok = false;
    if (!hi_in) f->hi_ok = false;
    if (f->lo_ok && (!f->have ||
                     cmp_bound(k, ch.min_bytes, ch.min_len, f->lo, f->lo_len) < 0)) {
        std::memcpy(f->lo, ch.min_bytes, ch.min_len);
        f->lo_len = ch.min_len;
    }
    if (f->hi_ok && (!f->have ||
                     cmp_bound(k, ch.max_bytes, ch.max_len, f->hi, f->hi_len) > 0)) {
        std::memcpy(f->hi, ch.max_bytes, ch.max_len);
        f->hi_len = ch.max_len;
    }
    f->have = true;
}

void emit_col(BoundKind k, int32_t field_id, const ColFold& f,
              ColumnStatEntry* e) noexcept {
    assert(e != nullptr);
    assert(field_id > 0);
    std::memset(e, 0, sizeof(*e));
    e->field_id   = field_id;
    e->null_count = f.nulls;
    if (k == BoundKind::kNone || !f.have) return;
    if (f.lo_ok) {
        e->has_lower = true;
        e->lower_len = static_cast<uint8_t>(f.lo_len);
        std::memcpy(e->lower, f.lo, f.lo_len);
    }
    if (f.hi_ok) {
        e->has_upper = true;
        e->upper_len = static_cast<uint8_t>(f.hi_len);
        std::memcpy(e->upper, f.hi, f.hi_len);
    }
}

}  // namespace

bool file_stats_from_parquet_meta(const pq::PqMeta* meta,
                                  const int32_t* field_ids,
                                  uint32_t n_field_ids,
                                  FileStats* out) noexcept {
    assert(out != nullptr);
    if (meta == nullptr || field_ids == nullptr || out == nullptr) return false;
    if (meta->n_columns > pq::kPqMaxColumns) return false;
    if (meta->n_row_groups > pq::kPqMaxRowGroups) return false;
    if (meta->n_row_groups > 0u && meta->chunks == nullptr) return false;
    out->n_cols = 0;
    const uint32_t nc = meta->n_columns < n_field_ids ? meta->n_columns : n_field_ids;
    for (uint32_t c = 0; c < nc; ++c) {                     // bounded: kPqMaxColumns
        if (field_ids[c] <= 0) continue;
        if (out->n_cols >= kIcebergMaxStatCols) break;      // degrades pruning only
        const pq::PqColumn& col = meta->columns[c];
        if (col.max_rep != 0u) continue;                    // repeated: not a leaf value
        const BoundKind k = bound_kind(col);
        ColFold f{};
        f.nulls_known = true;
        f.lo_ok = f.hi_ok = (k != BoundKind::kNone);
        for (uint32_t g = 0; g < meta->n_row_groups; ++g) { // bounded: kPqMaxRowGroups
            const pq::PqRowGroup& rg = meta->row_groups[g];
            if (c >= rg.chunk_count || rg.chunk_off + c >= meta->n_chunks) {
                return false;                               // malformed footer
            }
            fold_chunk(k, meta->chunks[rg.chunk_off + c], &f);
        }
        if (!f.nulls_known) continue;
        emit_col(k, field_ids[c], f, &out->cols[out->n_cols]);
        ++out->n_cols;
    }
    assert(out->n_cols <= kIcebergMaxStatCols);
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
