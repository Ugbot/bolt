// bolt/lakehouse/iceberg_delete_eq_ops.inc — predicate and key equality deletes (G2ICE-283). Included inside
// iceberg_writer.cpp after iceberg_delete_eq.inc; not a standalone TU.
//
// A predicate delete in equality mode reads the live rows twice. The first pass collects the matched rows' keys, per
// delete file scope (the data file's (spec, tuple), or one global scope). The second pass refuses the delete if a row the
// predicate keeps carries one of those keys in the same scope: the equality delete would remove it too.

namespace {

// Keys collected by the first pass, each tagged with its group; open addressing built once the pass ends.
struct KeySet {
    uint64_t* hash;
    uint32_t* off;
    uint32_t* len;
    uint32_t* grp;
    uint8_t*  bytes;
    uint32_t* slots;
    uint32_t  n, cap, used, cap_bytes, n_slots, _pad;
};

uint64_t keyset_hash(const uint8_t* key, uint32_t len, uint32_t g) noexcept {
    assert(key != nullptr || len == 0u);
    return eqkey::fnv64(reinterpret_cast<const uint8_t*>(&g), sizeof(g), eqkey::fnv64(key, len));
}

bool keyset_add(Arena* a, KeySet* s, uint32_t g, const uint8_t* key, uint32_t len) noexcept {
    assert(a != nullptr && s != nullptr && key != nullptr);
    uint32_t c = s->cap, c2 = s->cap, c3 = s->cap, c4 = s->cap;
    if (!lake_grow(a, nullptr, &s->hash, s->n, &c, s->n + 1u) || !lake_grow(a, nullptr, &s->off, s->n, &c2, s->n + 1u) ||
        !lake_grow(a, nullptr, &s->len, s->n, &c3, s->n + 1u) || !lake_grow(a, nullptr, &s->grp, s->n, &c4, s->n + 1u))
        return false;
    s->cap = c;
    if (static_cast<uint64_t>(s->used) + len > UINT32_MAX - 1u) return false;
    if (!lake_grow(a, nullptr, &s->bytes, s->used, &s->cap_bytes, s->used + len + 1u)) return false;
    std::memcpy(s->bytes + s->used, key, len);
    s->hash[s->n] = keyset_hash(key, len, g);
    s->off[s->n] = s->used;
    s->len[s->n] = len;
    s->grp[s->n] = g;
    s->used += len;
    ++s->n;
    return true;
}

bool keyset_build(Arena* a, KeySet* s) noexcept {
    assert(a != nullptr && s != nullptr);
    uint32_t slots = 16u;
    while (slots < s->n * 2u + 2u) slots *= 2u;                           // bounded: doubles past 2n
    s->slots = static_cast<uint32_t*>(a->allocate_zeroed(static_cast<size_t>(slots) * sizeof(uint32_t)));
    if (s->slots == nullptr) return false;
    s->n_slots = slots;
    for (uint32_t r = 0; r < s->n; ++r) {                                 // bounded by n
        uint32_t i = static_cast<uint32_t>(s->hash[r]) & (slots - 1u);
        while (s->slots[i] != 0u) i = (i + 1u) & (slots - 1u);            // bounded: the table is at most half full
        s->slots[i] = r + 1u;
    }
    return true;
}

bool keyset_has(const KeySet* s, uint32_t g, const uint8_t* key, uint32_t len) noexcept {
    assert(s != nullptr && key != nullptr);
    if (s->n_slots == 0u) return false;
    const uint64_t h = keyset_hash(key, len, g);
    uint32_t i = static_cast<uint32_t>(h) & (s->n_slots - 1u);
    for (uint32_t p = 0; p < s->n_slots; ++p, i = (i + 1u) & (s->n_slots - 1u)) {   // bounded by the table size
        const uint32_t e = s->slots[i];
        if (e == 0u) return false;
        const uint32_t r = e - 1u;
        if (s->hash[r] == h && s->grp[r] == g && s->len[r] == len && std::memcmp(s->bytes + s->off[r], key, len) == 0)
            return true;
    }
    return false;
}

struct EqPass {
    TableHandle*     th;
    Arena*           ra;            // the pass's batches; the first pass's outlive the write
    const EqKeySpec* k;
    const LiveData*  live;
    bool             global;
    bool             conflict;
    int32_t          global_spec;
    EqGroup*         gs;
    uint32_t         ng, cap;
    KeySet           keys;
};

// The group of the rows of data file `path`; -2 = none yet and `insert` false, -1 = failure (an unknown file).
int64_t eq_pass_group(EqPass* p, const char* path, bool insert) noexcept {
    assert(p != nullptr && path != nullptr);
    if (p->global) return eq_group_at(p->th->arena, &p->gs, &p->ng, &p->cap, p->global_spec, nullptr, 0u, insert);
    const DataFileRef* like = live_data_find(p->live, path);
    if (like == nullptr) {
        std::fprintf(stderr, "[iceberg_writer] equality delete of rows of %s, which is not a live data file; nothing was written\n", path);
        return -1;
    }
    return eq_group_at(p->th->arena, &p->gs, &p->ng, &p->cap, like->partition_spec_id, like->partition, like->n_partition, insert);
}

bool eq_collect(void* c, const dml::RowBatch* rb) noexcept {
    auto* p = static_cast<EqPass*>(c);
    assert(p != nullptr && rb != nullptr && rb->file_path != nullptr);
    if (rb->n_match == 0u) return true;
    const int64_t g = eq_pass_group(p, rb->file_path, true);
    if (g < 0) return false;
    Arena* a = p->th->arena;
    void* mem = a->allocate(sizeof(BoltBatch), alignof(BoltBatch));
    if (mem == nullptr) return false;
    BoltBatch* kb = new (mem) BoltBatch();
    const Schema* sch = metadata_current_schema(&p->th->meta);
    if (!key_rows(a, rb->batch, sch, p->k, p->k->col, rb->match, rb->n_match, kb)) return false;
    if (!eq_group_add(a, &p->gs[g], kb)) return false;
    const BoltColumn* cols[limits::kMaxEqualityIds];
    for (uint32_t i = 0; i < p->k->n; ++i) cols[i] = &kb->columns[kb->read_epoch][i];   // bounded: kMaxEqualityIds
    uint8_t key[limits::kEqKeyMax];
    for (int64_t r = 0; r < kb->num_rows; ++r) {                          // bounded by the batch
        uint32_t len = 0;
        if (!eqkey::encode(cols, p->k->n, r, key, &len)) return false;
        if (!keyset_add(a, &p->keys, static_cast<uint32_t>(g), key, len)) return false;
    }
    return true;
}

// Key columns of a scan batch, flat (a dictionary / RLE column materialized into the pass arena).
bool scan_key_cols(EqPass* p, const BoltBatch* b, const BoltColumn** cols) noexcept {
    assert(p != nullptr && b != nullptr && cols != nullptr);
    const BoltColumn* bc = b->columns[b->read_epoch];
    for (uint32_t i = 0; i < p->k->n; ++i) {                              // bounded: kMaxEqualityIds
        const BoltColumn& c = bc[p->k->col[i]];
        if (c.format == ColumnFormat::Flat || c.format == ColumnFormat::View || c.format == ColumnFormat::VarBinary) {
            cols[i] = &c;
            continue;
        }
        auto* m = p->ra->allocate_array<BoltColumn>(1);
        if (m == nullptr) return false;
        *m = c.materialize(p->ra);
        if (m->format != ColumnFormat::Flat) return false;
        cols[i] = m;
    }
    return true;
}

bool eq_check(void* c, const dml::RowBatch* rb) noexcept {
    auto* p = static_cast<EqPass*>(c);
    assert(p != nullptr && rb != nullptr && rb->file_path != nullptr);
    const int64_t g = eq_pass_group(p, rb->file_path, false);
    if (g == -2) return true;                                             // no key of this scope is deleted
    if (g < 0) return false;
    const int64_t nr = rb->batch->num_rows;
    auto* hit = static_cast<uint8_t*>(p->ra->allocate_zeroed(static_cast<size_t>(nr) + 1u));
    const BoltColumn* cols[limits::kMaxEqualityIds];
    if (hit == nullptr || !scan_key_cols(p, rb->batch, cols)) return false;
    for (uint32_t i = 0; i < rb->n_match; ++i) hit[rb->match[i]] = 1u;   // bounded by n_match
    uint8_t key[limits::kEqKeyMax];
    for (int64_t r = 0; r < nr; ++r) {                                    // bounded by the batch
        if (hit[r] != 0u) continue;
        uint32_t len = 0;
        if (!eqkey::encode(cols, p->k->n, r, key, &len)) return false;
        if (keyset_has(&p->keys, static_cast<uint32_t>(g), key, len)) { p->conflict = true; return false; }
    }
    return true;
}

bool live_has_tuples(const LiveData* l) noexcept {
    assert(l != nullptr);
    for (uint32_t i = 0; i < l->n; ++i) if (l->refs[i].n_partition != 0u) return true;   // bounded by n
    return false;
}

// Both passes; false with p->conflict set when a kept row shares a deleted key.
bool eq_passes(TableHandle* th, const Predicate* pred, EqPass* p) noexcept {
    assert(th != nullptr && pred != nullptr && p != nullptr);
    if (th->meta.current_snapshot_id <= 0) return true;
    if (!dml_read(th, p->ra, pred, nullptr, nullptr, eq_collect, p)) return false;
    if (p->keys.n == 0u) return true;
    if (!keyset_build(th->arena, &p->keys)) return false;
    Arena* first = p->ra;
    Arena* second = scratch_arena(th);
    if (second == nullptr) return false;
    p->ra = second;
    const bool ok = dml_read(th, second, pred, nullptr, nullptr, eq_check, p);
    second->~Arena();
    p->ra = first;
    return ok;
}

bool eq_conflict(TableHandle* th) noexcept {
    assert(th != nullptr);
    std::fprintf(stderr, "[iceberg_writer] equality delete refused: a row the predicate keeps has the key of a row it "
                 "deletes, and an equality delete removes both (use position deletes); nothing was written\n");
    th->last_error = CommitError::kFailed;
    return false;
}

bool dml_change_eq(TableHandle* th, const Predicate* pred, const BoltBatch* add, SnapshotOp op,
                   const DeleteOptions* o) noexcept {
    assert(th != nullptr && o != nullptr);
    th->last_error = CommitError::kFailed;
    if (pred == nullptr) return eq_refuse(th, "a row delete needs a predicate");
    EqKeySpec k{};
    LiveData live{};
    if (!pred_column_known(th, pred) || !handle_is_current(th) || !eq_key_resolve(th, o, &k)) return false;
    if (!live_data_build(th, &live)) return false;
    EqPass p{};
    p.th = th; p.k = &k; p.live = &live;
    p.global = o->global || !live_has_tuples(&live);
    const uint32_t prev_dirty = th->dirty_meta;
    bool added = false;
    if (p.global && !unpartitioned_spec(th, &p.global_spec, &added)) return false;
    p.ra = scratch_arena(th);
    bool ok = p.ra != nullptr && eq_passes(th, pred, &p);
    if (ok && p.ng == 0u && (add == nullptr || add->num_rows == 0)) {
        th->last_error = CommitError::kNone;                              // nothing matched: nothing to commit
    } else if (ok) {
        ok = eq_commit(th, &k, p.gs, p.ng, add, op);
    } else if (p.conflict) {
        eq_conflict(th);
    }
    const bool committed = ok && (p.ng != 0u || (add != nullptr && add->num_rows > 0));
    if (!committed && added) { --th->meta.n_specs; th->dirty_meta = prev_dirty; }
    if (p.ra != nullptr) p.ra->~Arena();
    return ok;
}

// table_delete_keys: the groups the keys fall in, one per tuple of the default spec, or one global group.
bool keys_groups(TableHandle* th, BoltBatch* kb, int32_t global_spec, bool global, EqGroup** gs, uint32_t* ng) noexcept {
    assert(th != nullptr && kb != nullptr && gs != nullptr && ng != nullptr);
    uint32_t cap = 0;
    *ng = 0;
    const PartitionSpec* spec = metadata_spec(&th->meta, th->meta.current_spec_id);
    if (!global && spec != nullptr && spec->n_fields != 0u) {
        PartSplit* sp = th->arena->allocate_array<PartSplit>(1);
        if (sp == nullptr || !split_init(th, spec, sp) || !split_batch(th, sp, kb)) return false;
        for (uint32_t g = 0; g < sp->n_groups; ++g) {                     // bounded by the tuples
            const int64_t at = eq_group_at(th->arena, gs, ng, &cap, spec->spec_id, sp->groups[g].tuple, spec->n_fields, true);
            if (at < 0) return false;
            for (uint32_t b = 0; b < sp->groups[g].n_batches; ++b)        // bounded by the group's batches
                if (!eq_group_add(th->arena, &(*gs)[at], sp->groups[g].batches[b])) return false;
        }
        return true;
    }
    const int64_t at = eq_group_at(th->arena, gs, ng, &cap, global_spec, nullptr, 0u, true);
    return at >= 0 && eq_group_add(th->arena, &(*gs)[at], kb);
}

// Whether keys can be scoped to the default spec's tuples (else the delete is global).
bool keys_partition_scoped(TableHandle* th, const LiveData* live, const BoltBatch* keys) noexcept {
    assert(th != nullptr && live != nullptr && keys != nullptr);
    const PartitionSpec* spec = metadata_spec(&th->meta, th->meta.current_spec_id);
    if (spec == nullptr || spec->n_fields == 0u || !live_has_tuples(live)) return false;
    for (uint32_t i = 0; i < live->n; ++i)                                // bounded by n
        if (live->refs[i].partition_spec_id != spec->spec_id) return false;
    PartSplit sp{};
    sp.spec = spec;
    int32_t col[kIcebergMaxFieldsPerSpec];
    for (uint32_t i = 0; i < spec->n_fields; ++i)                         // bounded: kIcebergMaxFieldsPerSpec
        if (spec->fields[i].transform.kind == TransformKind::kVoid) return false;
    return source_columns(th, &sp, keys, col);
}

}  // namespace

bool table_delete_with(TableHandle* th, const Predicate* pred, const DeleteOptions* opts) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    if (opts == nullptr || opts->kind == DeleteFileKind::kPosition) return dml_change(th, pred, nullptr, SnapshotOp::kDelete, opts);
    return dml_change_eq(th, pred, nullptr, SnapshotOp::kDelete, opts);
}

bool table_overwrite_with(TableHandle* th, const Predicate* pred, const BoltBatch* batch,
                          const DeleteOptions* opts) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    if (batch == nullptr) { th->last_error = CommitError::kFailed; return false; }
    const auto* schema=metadata_current_schema(&th->meta);BoltBatch* staged=nullptr;
    if(!schema || write_batch_conform(*schema,*batch,th->arena,&staged)!=JsonValueStatus::Ok) {
        th->last_error=CommitError::kFailed;return false;
    }
    if (opts == nullptr || opts->kind == DeleteFileKind::kPosition) return dml_change(th, pred, staged, SnapshotOp::kOverwrite, opts);
    return dml_change_eq(th, pred, staged, SnapshotOp::kOverwrite, opts);
}

bool table_delete_keys(TableHandle* th, const BoltBatch* keys, const DeleteOptions* opts) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    th->last_error = CommitError::kFailed;
    if (keys == nullptr || opts == nullptr || opts->kind != DeleteFileKind::kEquality)
        return eq_refuse(th, "table_delete_keys needs keys and DeleteFileKind::kEquality");
    EqKeySpec k{};
    if (!handle_is_current(th) || !eq_key_resolve(th, opts, &k)) return false;
    if (keys->num_cols != k.n) return eq_refuse(th, "the key batch must hold exactly the key columns");
    const Schema* sch = metadata_current_schema(&th->meta);
    int32_t in_col[limits::kMaxEqualityIds];
    for (uint32_t i = 0; i < k.n; ++i) {                                  // bounded: kMaxEqualityIds
        in_col[i] = keys->schema.find_field(sch->fields[k.col[i]].name);
        if (in_col[i] < 0) return eq_refuse(th, "a key column the key batch does not name");
    }
    if (keys->num_rows <= 0) { th->last_error = CommitError::kNone; return true; }
    if (keys->num_rows > static_cast<int64_t>(UINT32_MAX)) return eq_refuse(th, "a key batch of 2^32 rows");
    const uint32_t n = static_cast<uint32_t>(keys->num_rows);
    auto* rows = th->arena->allocate_array<uint32_t>(n);
    void* mem = th->arena->allocate(sizeof(BoltBatch), alignof(BoltBatch));
    if (rows == nullptr || mem == nullptr) return false;
    for (uint32_t i = 0; i < n; ++i) rows[i] = i;                         // bounded by n
    BoltBatch* kb = new (mem) BoltBatch();
    LiveData live{};
    if (!key_rows(th->arena, keys, sch, &k, in_col, rows, n, kb) || !live_data_build(th, &live)) return false;
    const bool global = opts->global || !keys_partition_scoped(th, &live, kb);
    const uint32_t prev_dirty = th->dirty_meta;
    int32_t gspec = 0;
    bool added = false;
    if (global && !unpartitioned_spec(th, &gspec, &added)) return false;
    EqGroup* gs = nullptr;
    uint32_t ng = 0;
    const bool ok = keys_groups(th, kb, gspec, global, &gs, &ng) &&
                    eq_commit(th, &k, gs, ng, nullptr, SnapshotOp::kDelete);
    if (!ok && added) { --th->meta.n_specs; th->dirty_meta = prev_dirty; }
    return ok;
}
