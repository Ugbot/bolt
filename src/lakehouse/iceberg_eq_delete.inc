#include "bolt/lakehouse/iceberg/iceberg_limits.h"
// Equality deletes (spec: "Equality Delete Files") and the sequence-number / partition scoping that governs every delete file.
// Included by iceberg_scan.cpp inside the iceberg namespace, after ScanHandle and the position-delete code.
//
// A delete row is the VALUES of the equality columns (the delete file's columns are identified by field id). It removes every data row
// carrying the same values, but only from data files with a STRICTLY LOWER data sequence number, and only within the delete file's
// partition (a delete written under an unpartitioned spec is global). Key equality is by value, never by position.


uint64_t fnv64(const uint8_t* p, uint64_t n, uint64_t h = 1469598103934665603ull) noexcept {
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// Hash of a partition tuple, identical for a data file and a delete file in the same partition of the same spec.
uint64_t partition_hash(const DataFileRef& e) noexcept {
    uint64_t h = fnv64(reinterpret_cast<const uint8_t*>(&e.partition_spec_id), sizeof(e.partition_spec_id));
    for (uint32_t i = 0; i < e.n_partition; ++i) {
        const PartitionValue& p = e.partition[i];
        const uint8_t tag = p.is_null ? 0 : p.is_str ? 2 : 1;
        h = fnv64(&tag, 1, h);
        if (p.is_null) continue;
        if (p.is_str) { const uint32_t n = static_cast<uint32_t>(std::strlen(p.str)); h = fnv64(reinterpret_cast<const uint8_t*>(p.str), n, h); }
        else h = fnv64(reinterpret_cast<const uint8_t*>(&p.i64), sizeof(p.i64), h);
    }
    return h;
}

constexpr uint32_t kEqKeyMax = limits::kEqKeyMax;

bool put_u32(uint8_t* b, uint32_t cap, uint32_t* n, uint32_t v) noexcept {
    if (*n + 4 > cap) return false;
    std::memcpy(b + *n, &v, 4); *n += 4; return true;
}

// Canonical bytes of one row's equality columns. Every integer-like type compares as a signed 64-bit value, floats by value (-0.0 == 0.0),
// strings by bytes; a type with no defined equality here fails the scan by name rather than comparing wrongly.
bool encode_eq_key(const BoltColumn* const* cols, uint32_t n, int64_t r, uint8_t* buf, uint32_t* len) noexcept {
    uint32_t o = 0;
    for (uint32_t c = 0; c < n; ++c) {
        const BoltColumn& col = *cols[c];
        if (o + 9 > kEqKeyMax) return false;
        if (col.is_null(r)) { buf[o++] = 0; continue; }
        switch (col.type) {
            case BoltType::Bool: case BoltType::Int8: case BoltType::Int16: case BoltType::Int32: case BoltType::Int64:
            case BoltType::UInt8: case BoltType::UInt16: case BoltType::UInt32: case BoltType::UInt64:
            case BoltType::Date32: case BoltType::Date64: case BoltType::Timestamp: case BoltType::Duration: case BoltType::Decimal64: {
                int64_t v = 0;
                switch (col.type_size_bytes) {
                    case 1: v = static_cast<const int8_t*>(col.data)[r]; break;
                    case 2: v = static_cast<const int16_t*>(col.data)[r]; break;
                    case 4: v = static_cast<const int32_t*>(col.data)[r]; break;
                    default: v = static_cast<const int64_t*>(col.data)[r]; break;
                }
                buf[o++] = 1; std::memcpy(buf + o, &v, 8); o += 8; break;
            }
            case BoltType::Float32: case BoltType::Float64: {
                double d = col.type == BoltType::Float32 ? static_cast<double>(static_cast<const float*>(col.data)[r])
                                                          : static_cast<const double*>(col.data)[r];
                if (d == 0.0) d = 0.0;
                buf[o++] = 2; std::memcpy(buf + o, &d, 8); o += 8; break;
            }
            case BoltType::Utf8: case BoltType::Binary: {
                const uint8_t* d = nullptr; int32_t sl = 0;
                col.utf8_at(r, &d, &sl);
                if (sl < 0 || o + 5 + static_cast<uint32_t>(sl) > kEqKeyMax) return false;
                buf[o++] = 3; const uint32_t u = static_cast<uint32_t>(sl); std::memcpy(buf + o, &u, 4); o += 4;
                if (sl > 0) std::memcpy(buf + o, d, static_cast<size_t>(sl));
                o += static_cast<uint32_t>(sl); break;
            }
            default:
                std::fprintf(stderr, "[iceberg_scan] REFUSED: equality delete on a column of decoded type %d, which has no defined equality here\n", static_cast<int>(col.type));
                return false;
        }
    }
    *len = o;
    return true;
}

bool eq_set_probe(const EqSet& set, uint64_t h, const uint8_t* key, uint32_t len) noexcept {
    if (set.n_slots == 0) return false;
    uint32_t i = static_cast<uint32_t>(h) & (set.n_slots - 1);
    for (uint32_t probes = 0; probes < set.n_slots; ++probes, i = (i + 1) & (set.n_slots - 1)) {   // bounded by the table size
        const uint32_t slot = set.slots[i];
        if (slot == 0) return false;
        const uint32_t row = slot - 1;
        if (set.hashes[row] == h && set.key_len[row] == len && std::memcmp(set.key_bytes + set.key_off[row], key, len) == 0) return true;
    }
    return false;
}

// Grow the three per-row arrays together (they always share one capacity).
bool eq_grow_rows(ScanHandle* s, EqSet* set, uint32_t need) noexcept {
    if (need <= set->cap_rows) return true;
    uint32_t ncap = set->cap_rows == 0 ? 64u : set->cap_rows;
    while (ncap < need) ncap *= 2u;
    if (!s->budget->try_reserve(static_cast<uint64_t>(ncap) * (sizeof(uint64_t) + 2 * sizeof(uint32_t)))) return false;
    auto* h = s->scratch->allocate_array<uint64_t>(ncap);
    auto* o = s->scratch->allocate_array<uint32_t>(ncap);
    auto* l = s->scratch->allocate_array<uint32_t>(ncap);
    if (h == nullptr || o == nullptr || l == nullptr) return false;
    if (set->n_rows != 0) {
        std::memcpy(h, set->hashes, sizeof(uint64_t) * set->n_rows);
        std::memcpy(o, set->key_off, sizeof(uint32_t) * set->n_rows);
        std::memcpy(l, set->key_len, sizeof(uint32_t) * set->n_rows);
    }
    set->hashes = h; set->key_off = o; set->key_len = l; set->cap_rows = ncap;
    return true;
}

bool eq_set_add(ScanHandle* s, EqSet* set, const uint8_t* key, uint32_t len) noexcept {
    if (!eq_grow_rows(s, set, set->n_rows + 1u)) return false;
    if (set->key_used + len > set->key_cap) {
        uint64_t ncap = set->key_cap == 0 ? 4096 : set->key_cap * 2;
        while (ncap < set->key_used + len) ncap *= 2;
        if (!s->budget->try_reserve(ncap)) return false;
        auto* nb = s->scratch->allocate_array<uint8_t>(static_cast<size_t>(ncap));
        if (nb == nullptr) return false;
        if (set->key_used != 0) std::memcpy(nb, set->key_bytes, set->key_used);
        set->key_bytes = nb; set->key_cap = ncap;
    }
    std::memcpy(set->key_bytes + set->key_used, key, len);
    set->hashes[set->n_rows] = fnv64(key, len);
    set->key_off[set->n_rows] = static_cast<uint32_t>(set->key_used);
    set->key_len[set->n_rows] = len;
    set->key_used += len;
    ++set->n_rows;
    return true;
}

bool eq_set_build(ScanHandle* s, EqSet* set) noexcept {
    uint32_t slots = 16;
    while (slots < set->n_rows * 2u + 2u) slots *= 2;
    if (!s->budget->try_reserve(static_cast<uint64_t>(slots) * sizeof(uint32_t))) return false;
    set->slots = static_cast<uint32_t*>(s->scratch->allocate_zeroed(static_cast<size_t>(slots) * sizeof(uint32_t)));
    if (set->slots == nullptr) return false;
    set->n_slots = slots;
    for (uint32_t r = 0; r < set->n_rows; ++r) {
        uint32_t i = static_cast<uint32_t>(set->hashes[r]) & (slots - 1);
        while (set->slots[i] != 0) i = (i + 1) & (slots - 1);
        set->slots[i] = r + 1;
    }
    return true;
}

const Schema* emitted_schema(const ScanHandle* s) noexcept {
    // Resolved at scan open (before any data file binds the column map): the schema the emitted columns follow.
    return s->table->strict_pinned ? s->selected_schema : metadata_current_schema(&s->table->meta);
}

// Load one equality-delete file into a hashed key set. `df` has content == kEqualityDeletes and its equality field ids.
bool load_equality_delete_file(ScanHandle* s, const DataFileRef& df, int64_t seq, const PartitionSpec* spec) noexcept {
    assert(s != nullptr && df.content == FileContent::kEqualityDeletes);
    if (df.n_equality_ids == 0 || df.n_equality_ids > kMaxEqualityIds) return false;
    if (s->n_eq_sets >= kMaxEqualitySets) { std::fprintf(stderr, "[iceberg_scan] REFUSED: more than %u equality delete files\n", kMaxEqualitySets); return false; }
    const Schema* sch = emitted_schema(s);
    if (sch == nullptr) return false;
    if (!lake_grow(s->scratch, s->budget, &s->eq_sets, s->n_eq_sets, &s->eq_cap, s->n_eq_sets + 1u)) return false;
    EqSet* set = &s->eq_sets[s->n_eq_sets];
    std::memset(set, 0, sizeof(*set));
    set->seq = seq;
    set->spec_id = df.partition_spec_id;
    set->global = spec == nullptr || spec->n_fields == 0;
    set->part_hash = partition_hash(df);
    set->n_ids = df.n_equality_ids;
    for (uint32_t k = 0; k < set->n_ids; ++k) {
        set->ids[k] = df.equality_ids[k];
        set->schema_idx[k] = -1;
        for (uint32_t j = 0; j < sch->n_fields; ++j) if (sch->fields[j].id == set->ids[k]) set->schema_idx[k] = static_cast<int32_t>(j);
        if (set->schema_idx[k] < 0) {
            std::fprintf(stderr, "[iceberg_scan] REFUSED: an equality delete names field id %d, which is not in the table's current schema\n", set->ids[k]);
            return false;
        }
    }
    s->meta_arena->reset();
    pq::PqMeta* meta = s->meta_arena->allocate_array<pq::PqMeta>(1);
    if (meta == nullptr) return false;
    std::memset(meta, 0, sizeof(*meta));
    PinnedParquet pinned{};
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!open_parquet(s, df.file_path, s->meta_arena, meta, &pinned, &body, &blen)) return false;
    int32_t col_of[kMaxEqualityIds];
    for (uint32_t k = 0; k < set->n_ids; ++k) {
        col_of[k] = -1;
        for (uint32_t i = 0; i < meta->n_columns; ++i) if (meta->columns[i].has_field_id && meta->columns[i].field_id == set->ids[k]) col_of[k] = static_cast<int32_t>(i);
        if (col_of[k] < 0) { std::fprintf(stderr, "[iceberg_scan] REFUSED: an equality delete file lacks the column for field id %d\n", set->ids[k]); return false; }
    }
    BoltColumn* cols = s->scratch->allocate_array<BoltColumn>(meta->n_columns);
    if (cols == nullptr) return false;
    uint8_t key[kEqKeyMax];
    for (uint32_t rg = 0; rg < meta->n_row_groups && rg < kLakeMaxRowGroups; ++rg) {
        std::memset(cols, 0, sizeof(BoltColumn) * meta->n_columns);
        int64_t rows = 0;
        if (!read_parquet_group(s, &pinned, body, blen, meta, rg, cols, &rows, meta->n_columns)) return false;
        const BoltColumn* eq[kMaxEqualityIds];
        for (uint32_t k = 0; k < set->n_ids; ++k) eq[k] = &cols[col_of[k]];
        for (int64_t r = 0; r < rows; ++r) {
            uint32_t len = 0;
            if (!encode_eq_key(eq, set->n_ids, r, key, &len)) return false;
            if (!eq_set_add(s, set, key, len)) return false;
        }
    }
    if (!eq_set_build(s, set)) return false;
    ++s->n_eq_sets;
    return true;
}

// Which loaded sets apply to this data file: strictly newer sequence number, same partition (or a global delete).
void select_equality_sets(ScanHandle* s, const FileTask& f) noexcept {
    s->n_cur_eq = 0;
    for (uint32_t i = 0; i < s->n_eq_sets && s->n_cur_eq < kMaxEqualitySets; ++i) {
        const EqSet& e = s->eq_sets[i];
        if (e.seq <= f.seq) continue;
        if (!e.global && (e.spec_id != f.spec_id || e.part_hash != f.part_hash)) continue;
        s->cur_eq[s->n_cur_eq++] = i;
    }
}

// Remove rows matched by an applicable equality delete from the emitted columns (schema order). Rows are filtered in place.
bool apply_equality_deletes(ScanHandle* s, BoltColumn* cols, uint32_t n_cols, int64_t* rows) noexcept {
    if (s->n_cur_eq == 0 || *rows == 0) return true;
    auto* sel = s->scratch->allocate_array<uint32_t>(static_cast<size_t>(*rows));
    if (sel == nullptr) return false;
    uint32_t kept = 0;
    uint8_t key[kEqKeyMax];
    for (int64_t r = 0; r < *rows; ++r) {
        bool hit = false;
        for (uint32_t i = 0; i < s->n_cur_eq && !hit; ++i) {
            const EqSet& e = s->eq_sets[s->cur_eq[i]];
            const BoltColumn* eqcols[kMaxEqualityIds];
            for (uint32_t k = 0; k < e.n_ids; ++k) {
                if (static_cast<uint32_t>(e.schema_idx[k]) >= n_cols) return false;
                eqcols[k] = &cols[e.schema_idx[k]];
            }
            uint32_t len = 0;
            if (!encode_eq_key(eqcols, e.n_ids, r, key, &len)) return false;
            hit = eq_set_probe(e, fnv64(key, len), key, len);
        }
        if (!hit) sel[kept++] = static_cast<uint32_t>(r);
    }
    if (kept == static_cast<uint32_t>(*rows)) return true;
    return compact_selected(cols, n_cols, sel, kept, rows, s->scratch, "equality delete");
}
