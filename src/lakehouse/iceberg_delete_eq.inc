// bolt/lakehouse/iceberg_delete_eq.inc — equality-delete files (G2ICE-283). Included inside iceberg_writer.cpp at
// namespace scope after iceberg_dml_ops.inc; not a standalone TU. API: writer.h table_delete_with / table_delete_keys.
//
// An equality delete file holds key values (spec: "Equality Delete Files"); a reader drops every row of a data file
// with a STRICTLY LOWER data sequence number, in the delete file's partition (or every partition when its spec is
// unpartitioned), whose key equals one of them. The file's columns are the key fields with their field ids; the
// manifest entry carries equality_ids.

namespace {

struct EqKeySpec {
    uint32_t n;
    uint32_t _pad;
    int32_t  ids[limits::kMaxEqualityIds];
    int32_t  col[limits::kMaxEqualityIds];   // index of each key field in the current schema
};

// The rows of one delete file: its spec and tuple, and the key batches.
struct EqGroup {
    int32_t               spec_id;
    uint32_t              n_part;
    const PartitionValue* tuple;
    const BoltBatch**     batches;
    uint32_t              n, cap;
    int64_t               rows;
};

bool eq_refuse(TableHandle* th, const char* what) noexcept {
    assert(th != nullptr && what != nullptr);
    std::fprintf(stderr, "[iceberg_writer] equality delete refused: %s; nothing was written\n", what);
    th->last_error = CommitError::kFailed;
    return false;
}

// A key column must compare the same in every reader: integers, dates, times, timestamps, strings and binaries do. Float
// and double (-0.0 / NaN), decimal, uuid and nested types are not identifier types (spec: "Identifier Field IDs").
bool eq_type_keyable(const char* t) noexcept {
    assert(t != nullptr);
    IcebergTypeInfo info{};
    if (!iceberg_type_parse(t, &info)) return false;
    switch (info.type) {
        case IcebergType::kBoolean: case IcebergType::kInt: case IcebergType::kLong: case IcebergType::kDate:
        case IcebergType::kTime: case IcebergType::kTimestamp: case IcebergType::kTimestampTz:
        case IcebergType::kString: case IcebergType::kBinary: case IcebergType::kFixed: return true;
        default: return false;
    }
}

bool eq_key_resolve(TableHandle* th, const DeleteOptions* o, EqKeySpec* k) noexcept {
    assert(th != nullptr && o != nullptr && k != nullptr);
    const Schema* sch = metadata_current_schema(&th->meta);
    if (sch == nullptr) return eq_refuse(th, "the table has no current schema");
    *k = EqKeySpec{};
    const bool own = o->n_equality_ids != 0u;
    const uint32_t n = own ? o->n_equality_ids : sch->n_identifier;
    if (n == 0u) return eq_refuse(th, "no key (name equality_ids, or set the schema's identifier-field-ids)");
    if (n > limits::kMaxEqualityIds) return eq_refuse(th, "a key of more than limits::kMaxEqualityIds columns");
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded: kMaxEqualityIds
        const int32_t id = own ? o->equality_ids[i] : sch->identifier_ids[i];
        int32_t at = -1;
        for (uint32_t j = 0; j < sch->n_fields && at < 0; ++j)            // bounded by n_fields
            if (sch->fields[j].id == id) at = static_cast<int32_t>(j);
        if (at < 0) return eq_refuse(th, "a key field id that is not a top-level column of the current schema");
        for (uint32_t j = 0; j < i; ++j)                                  // bounded by i
            if (k->ids[j] == id) return eq_refuse(th, "a key that names one field twice");
        if (!eq_type_keyable(sch->fields[at].type))
            return eq_refuse(th, "a key column of a type without exact equality (float, double, decimal, uuid or nested)");
        k->ids[i] = id;
        k->col[i] = at;
    }
    k->n = n;
    return true;
}

// Rows `rows[0..n)` of `in`'s columns in_col[0..k->n) as a batch of the key fields, over `a` (spilled bytes shared).
bool key_rows(Arena* a, const BoltBatch* in, const Schema* sch, const EqKeySpec* k, const int32_t* in_col,
              const uint32_t* rows, uint32_t n, BoltBatch* out) noexcept {
    assert(a != nullptr && in != nullptr && sch != nullptr && k != nullptr && out != nullptr);
    assert(in_col != nullptr && (rows != nullptr || n == 0u));
    BoltBatch view{};
    BoltBatch::init_empty(&view);
    view.arena = a;
    view.num_rows = in->num_rows;
    view.num_cols = k->n;
    if (!BoltBatch::alloc_columns(&view, a, k->n)) return false;
    const BoltColumn* ic = in->columns[in->read_epoch];
    BoltColumn* vc = view.columns[view.read_epoch];
    for (uint32_t i = 0; i < k->n; ++i) {                                 // bounded: kMaxEqualityIds
        const SchemaField& f = sch->fields[k->col[i]];
        view.schema.add_field(f.name, ic[in_col[i]].type, !f.required);
        vc[i] = ic[in_col[i]];
    }
    return gather_batch(a, &view, rows, n, out);
}

// Index of the group for (spec, tuple); with `insert` false, -2 when there is none. -1 = allocation failure.
int64_t eq_group_at(Arena* a, EqGroup** gs, uint32_t* ng, uint32_t* cap, int32_t spec, const PartitionValue* t,
                    uint32_t np, bool insert) noexcept {
    assert(a != nullptr && gs != nullptr && ng != nullptr && cap != nullptr);
    for (uint32_t g = 0; g < *ng; ++g) {                                  // bounded: one group per tuple written
        const EqGroup& e = (*gs)[g];
        if (e.spec_id == spec && e.n_part == np && (np == 0u || tuple_equal(e.tuple, t, np))) return g;
    }
    if (!insert) return -2;
    if (!lake_grow(a, nullptr, gs, *ng, cap, *ng + 1u)) return -1;
    (*gs)[*ng] = EqGroup{};
    (*gs)[*ng].spec_id = spec;
    (*gs)[*ng].n_part = np;
    (*gs)[*ng].tuple = t;
    return static_cast<int64_t>((*ng)++);
}

bool eq_group_add(Arena* a, EqGroup* g, const BoltBatch* b) noexcept {
    assert(a != nullptr && g != nullptr && b != nullptr);
    if (!lake_grow(a, nullptr, &g->batches, g->n, &g->cap, g->n + 1u)) return false;
    g->batches[g->n++] = b;
    g->rows += b->num_rows;
    return true;
}

// The id of an unpartitioned spec, added to the table's specs (not as the default) when it has none.
bool unpartitioned_spec(TableHandle* th, int32_t* id, bool* added) noexcept {
    assert(th != nullptr && id != nullptr && added != nullptr);
    *added = false;
    PartitionSpec none{};
    none.spec_id = -1;
    PartitionSpec next{};
    int32_t existing = -1;
    if (!normalize_spec(&th->meta, metadata_current_schema(&th->meta), &none, &next, &existing)) return false;
    *id = existing >= 0 ? existing : next.spec_id;
    if (existing >= 0) return true;
    if (th->meta.n_specs >= kIcebergMaxSpecs) return eq_refuse(th, "a global delete needs an unpartitioned spec and the table holds kIcebergMaxSpecs specs");
    th->meta.specs[th->meta.n_specs++] = next;
    th->dirty_meta |= kDirtySpec;
    *added = true;
    return true;
}

// One equality-delete parquet file of group `g`; `df` describes it for the manifest.
bool write_eq_file(TableHandle* th, int64_t snap_id, const EqKeySpec* k, const EqGroup& g, char (&rel)[128],
                   DataFileRef* df) noexcept {
    assert(th != nullptr && k != nullptr && df != nullptr);
    assert(g.n > 0u && g.rows > 0);
    const int rn = std::snprintf(rel, sizeof(rel), "data/%lld-%016llx-eq-deletes.parquet",
                                 static_cast<long long>(snap_id), static_cast<unsigned long long>(unique_u64()));
    if (rn <= 0 || static_cast<uint32_t>(rn) >= sizeof(rel)) return false;
    const Schema* sch = metadata_current_schema(&th->meta);
    Schema ks{};
    if (!schema_reserve(&ks, th->arena, nullptr, k->n)) return false;
    for (uint32_t i = 0; i < k->n; ++i) ks.fields[i] = sch->fields[k->col[i]];   // bounded: kMaxEqualityIds
    ks.n_fields = k->n;
    ingest::parquet::ParquetWriteOpts* po = th->arena->allocate_array<ingest::parquet::ParquetWriteOpts>(1);
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (po == nullptr || !make_pq_opts(&ks, nullptr, po, &wide)) return false;
    auto* w = data_file_begin(th, rel, snap_id, po);
    if (w == nullptr) { th->last_error = CommitError::kFailed; return false; }
    for (uint32_t b = 0; b < g.n; ++b)                                    // bounded by the group's batches
        if (!ingest::parquet::parquet_write_row_group(w, g.batches[b])) { data_file_abandon(th, w); return false; }
    int64_t size = 0;
    if (!data_file_finish(th, rel, w, &size)) return false;
    *df = DataFileRef{};
    df->status = ManifestStatus::kAdded;
    df->content = FileContent::kEqualityDeletes;
    df->partition_spec_id = g.spec_id;
    df->snapshot_id = snap_id;
    df->n_partition = g.n_part;
    if (g.n_part != 0u) std::memcpy(df->partition, g.tuple, sizeof(PartitionValue) * g.n_part);
    df->n_equality_ids = k->n;
    std::memcpy(df->equality_ids, k->ids, sizeof(int32_t) * k->n);
    df->stats.record_count = g.rows;
    df->stats.file_size_in_bytes = size;
    if (!abs_location(th->root, rel, df->file_path, sizeof(df->file_path))) { data_file_complete(th, rel, false); return false; }
    return true;
}

// Every group's file, plus the added rows' data files (an overwrite); written files are completed by the caller.
bool eq_write_outputs(TableHandle* th, int64_t snap_id, const EqKeySpec* k, const EqGroup* gs, uint32_t ng,
                      const BoltBatch* add, DmlOutputs* o) noexcept {
    assert(th != nullptr && k != nullptr && o != nullptr);
    assert(gs != nullptr || ng == 0u);
    *o = DmlOutputs{};
    o->dels = th->arena->allocate_array<DataFileRef>(ng + 1u);
    o->del_rels = th->arena->allocate_array<char[128]>(ng + 1u);
    bool ok = o->dels != nullptr && o->del_rels != nullptr;
    for (uint32_t g = 0; ok && g < ng; ++g) {                             // bounded by ng
        ok = write_eq_file(th, snap_id, k, gs[g], o->del_rels[g], &o->dels[g]);
        if (ok) ++o->n_dels;
    }
    ok = ok && (add == nullptr || add->num_rows <= 0 || dml_add_files(th, snap_id, add, o));
    if (!ok) dml_complete(th, o, false);
    return ok;
}

// Write and commit the groups (and an overwrite's rows) as one snapshot.
bool eq_commit(TableHandle* th, const EqKeySpec* k, const EqGroup* gs, uint32_t ng, const BoltBatch* add,
               SnapshotOp op) noexcept {
    assert(th != nullptr && k != nullptr);
    const uint32_t files = ng + (add != nullptr && add->num_rows > 0 ? 1u : 0u);
    bool ok = th->file_io == nullptr || files <= 1u ||
              refuse_named(th, "an equality delete writing more than one file through FileWriteIO (one staged file per commit)");
    const int64_t snap_id = ok ? mint_snapshot_id(th) : 0;
    DmlOutputs o{};
    ok = ok && eq_write_outputs(th, snap_id, k, gs, ng, add, &o);
    bool committed = false;
    if (ok) {
        ChangeSet cs{};
        cs.op = op;
        cs.snap_id = snap_id;
        cs.add_dels = o.dels;
        cs.n_add_dels = o.n_dels;
        cs.add_data = o.dats;
        cs.n_add_data = o.n_dats;
        committed = commit_change(th, &cs);
        dml_complete(th, &o, committed);
    }
    return committed;
}

}  // namespace
