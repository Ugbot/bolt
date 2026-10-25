// iceberg_stats_store.cpp — arena storage for a data file's metrics (G2ICE-336). Columns grow by doubling up to the
// iceberg_stat_columns budget; a bound is copied whole up to iceberg_bound_bytes. Past either: ResourceExhausted
// naming the knob, never a dropped or cut metric.

#include "bolt/lakehouse/iceberg/statistics.h"

#include <cassert>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_resource.h"
#include "lake_grow.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

constexpr char kStatColsKnob[] = "iceberg_stat_columns";
constexpr char kBoundKnob[] = "iceberg_bound_bytes";
constexpr uint32_t kStatsMagic = 0x31535449u;   // "ITS1"

bool grow_cols(FileStats* st, Arena* a, uint32_t need) noexcept {
    assert(st != nullptr && a != nullptr);
    const uint64_t cap = bolt_limits_value(bolt_limits_id::iceberg_stat_columns);
    if (need > cap) { set_resource_exhausted(kStatColsKnob, need, cap); return false; }
    uint32_t c = st->cap_cols;
    if (!lake_grow(a, nullptr, &st->cols, st->n_cols, &c, need)) return false;
    st->cap_cols = c;
    return true;
}

char* copy_bytes(Arena* a, const void* b, uint32_t n) noexcept {
    assert(a != nullptr);
    char* p = static_cast<char*>(a->allocate(n == 0u ? 1u : n, 1));
    if (p == nullptr) { set_resource_exhausted("arena_max_blocks", n, 0); return nullptr; }
    if (n > 0u) std::memcpy(p, b, n);
    return p;
}

}  // namespace

ColumnStatEntry* stats_add_col(FileStats* st, Arena* a, int32_t field_id) noexcept {
    assert(st != nullptr && a != nullptr);
    assert(st->n_cols <= st->cap_cols);
    const uint64_t budget = bolt_limits_value(bolt_limits_id::iceberg_stat_columns);
    if (uint64_t{st->n_cols} + 1u > budget) { set_resource_exhausted(kStatColsKnob, uint64_t{st->n_cols} + 1u, budget); return nullptr; }
    if (st->n_cols == st->cap_cols && !grow_cols(st, a, st->n_cols + 1u)) return nullptr;
    ColumnStatEntry* e = &st->cols[st->n_cols++];
    std::memset(e, 0, sizeof(*e));
    e->field_id = field_id;
    e->null_count = -1;
    return e;
}

ColumnStatEntry* stats_col(FileStats* st, Arena* a, int32_t field_id) noexcept {
    assert(st != nullptr && a != nullptr);
    for (uint32_t i = 0; i < st->n_cols; ++i) {                          // bounded: n_cols
        if (st->cols[i].field_id == field_id) return &st->cols[i];
    }
    return stats_add_col(st, a, field_id);
}

const ColumnStatEntry* stats_find(const FileStats* st, int32_t field_id) noexcept {
    assert(st != nullptr);
    for (uint32_t i = 0; i < st->n_cols; ++i) {                          // bounded: n_cols
        if (st->cols[i].field_id == field_id) return &st->cols[i];
    }
    return nullptr;
}

bool stats_set_bound(ColumnStatEntry* e, Arena* a, bool lower, const void* b, uint32_t n) noexcept {
    assert(e != nullptr && a != nullptr);
    assert(b != nullptr || n == 0u);
    const uint64_t cap = bolt_limits_value(bolt_limits_id::iceberg_bound_bytes);
    if (n > cap) { set_resource_exhausted(kBoundKnob, n, cap); return false; }
    char* p = copy_bytes(a, b, n);
    if (p == nullptr) return false;
    if (lower) { e->lower = p; e->lower_len = n; e->has_lower = true; }
    else       { e->upper = p; e->upper_len = n; e->has_upper = true; }
    return true;
}

bool stats_set_splits(FileStats* st, Arena* a, const int64_t* offs, uint32_t n) noexcept {
    assert(st != nullptr && a != nullptr);
    assert(offs != nullptr || n == 0u);
    st->n_split_offsets = 0;
    st->split_offsets = nullptr;
    if (n == 0u) return true;
    int64_t* p = a->allocate_array<int64_t>(n);
    if (p == nullptr) { set_resource_exhausted("arena_max_blocks", uint64_t{n} * 8u, 0); return false; }
    std::memcpy(p, offs, sizeof(int64_t) * n);
    st->split_offsets = p;
    st->n_split_offsets = n;
    return true;
}

bool stats_clone(const FileStats& src, Arena* a, FileStats* out) noexcept {
    assert(a != nullptr && out != nullptr);
    *out = src;
    out->cols = nullptr; out->n_cols = 0; out->cap_cols = 0;
    if (src.n_cols > 0u && !grow_cols(out, a, src.n_cols)) return false;
    for (uint32_t i = 0; i < src.n_cols; ++i) {                          // bounded: n_cols
        const ColumnStatEntry& s = src.cols[i];
        ColumnStatEntry& d = out->cols[i];
        d = s;
        d.lower = nullptr; d.upper = nullptr; d.has_lower = d.has_upper = false;
        if (s.has_lower && !stats_set_bound(&d, a, true, s.lower, s.lower_len)) return false;
        if (s.has_upper && !stats_set_bound(&d, a, false, s.upper, s.upper_len)) return false;
    }
    out->n_cols = src.n_cols;
    return stats_set_splits(out, a, src.split_offsets, src.n_split_offsets);
}

// Encoding: magic, the FileStats header, n_cols entries (pointers zeroed), the bound bytes in entry order, the splits.
uint64_t stats_encoded_size(const FileStats& st) noexcept {
    uint64_t n = 4u + sizeof(FileStats) + uint64_t{st.n_cols} * sizeof(ColumnStatEntry) + 8u * st.n_split_offsets;
    for (uint32_t i = 0; i < st.n_cols; ++i) {                           // bounded: n_cols
        if (st.cols[i].has_lower) n += st.cols[i].lower_len;
        if (st.cols[i].has_upper) n += st.cols[i].upper_len;
    }
    return n;
}

bool stats_encode(const FileStats& st, uint8_t* out, uint64_t cap, uint64_t* len) noexcept {
    assert(out != nullptr && len != nullptr);
    const uint64_t need = stats_encoded_size(st);
    if (need > cap) return false;
    uint8_t* p = out;
    std::memcpy(p, &kStatsMagic, 4); p += 4;
    FileStats h = st;
    h.cols = nullptr; h.split_offsets = nullptr; h.cap_cols = 0;
    std::memcpy(p, &h, sizeof(h)); p += sizeof(h);
    for (uint32_t i = 0; i < st.n_cols; ++i) {                           // bounded: n_cols
        ColumnStatEntry e = st.cols[i];
        e.lower = nullptr; e.upper = nullptr;
        std::memcpy(p, &e, sizeof(e)); p += sizeof(e);
    }
    for (uint32_t i = 0; i < st.n_cols; ++i) {                           // bounded: n_cols
        const ColumnStatEntry& e = st.cols[i];
        if (e.has_lower && e.lower_len > 0u) { std::memcpy(p, e.lower, e.lower_len); p += e.lower_len; }
        if (e.has_upper && e.upper_len > 0u) { std::memcpy(p, e.upper, e.upper_len); p += e.upper_len; }
    }
    if (st.n_split_offsets > 0u) { std::memcpy(p, st.split_offsets, 8u * st.n_split_offsets); p += 8u * st.n_split_offsets; }
    *len = static_cast<uint64_t>(p - out);
    assert(*len == need);
    return true;
}

bool stats_decode(const uint8_t* b, uint64_t len, Arena* a, FileStats* out) noexcept {
    assert(a != nullptr && out != nullptr);
    if (b == nullptr || len < 4u + sizeof(FileStats)) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, b, 4);
    if (magic != kStatsMagic) return false;
    FileStats h{};
    std::memcpy(&h, b + 4, sizeof(h));
    const uint64_t ents = uint64_t{h.n_cols} * sizeof(ColumnStatEntry);
    uint64_t at = 4u + sizeof(h);
    if (ents > len - at) return false;
    *out = h;
    out->cols = nullptr; out->n_cols = 0; out->cap_cols = 0; out->split_offsets = nullptr; out->n_split_offsets = 0;
    if (h.n_cols > 0u && !grow_cols(out, a, h.n_cols)) return false;
    if (ents > 0u) std::memcpy(out->cols, b + at, ents);
    at += ents;
    for (uint32_t i = 0; i < h.n_cols; ++i) {                            // bounded: n_cols
        ColumnStatEntry& e = out->cols[i];
        const uint64_t nl = e.has_lower ? e.lower_len : 0u, nu = e.has_upper ? e.upper_len : 0u;
        e.lower = e.upper = nullptr;
        if (nl + nu > len - at) return false;
        if (e.has_lower && !stats_set_bound(&e, a, true, b + at, e.lower_len)) return false;
        at += nl;
        if (e.has_upper && !stats_set_bound(&e, a, false, b + at, e.upper_len)) return false;
        at += nu;
    }
    out->n_cols = h.n_cols;
    if (8u * uint64_t{h.n_split_offsets} != len - at) return false;
    if (h.n_split_offsets == 0u) return true;
    int64_t* sp = a->allocate_array<int64_t>(h.n_split_offsets);
    if (sp == nullptr) return false;
    std::memcpy(sp, b + at, 8u * h.n_split_offsets);
    out->split_offsets = sp;
    out->n_split_offsets = h.n_split_offsets;
    return true;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
