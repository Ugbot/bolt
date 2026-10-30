// bolt/lakehouse/iceberg_partitioned.inc — partitioned writes (G2ICE-292). Included inside iceberg_writer.cpp's
// anonymous namespace after write_data_file / mint_snapshot_id, before iceberg_change.inc; not a standalone TU.
//
// An append (and an overwrite's added rows) to a partitioned table computes each row's partition tuple under the
// table's DEFAULT spec (partition_values.h), groups the rows by tuple and writes one data file per tuple. The file's
// manifest entry records the tuple and the spec id it was written under; a later spec change leaves those files
// with their own spec (spec: "Partition Evolution"). Rows are moved into per-tuple batches by a gather over arena
// storage; nothing on this path allocates outside the table / commit arenas.

bool publish_snapshot(TableHandle* th, const DataFileRef* files, uint32_t nf, SnapshotOp op, int64_t snap_id) noexcept;

// The source and result types of every field of `spec`. A source column is looked up in the current schema, then in
// older schemas (a dropped column keeps its type for the specs that name it).
bool spec_types_for(const TableHandle* th, const PartitionSpec* spec, IcebergTypeInfo* src,
                    IcebergTypeInfo* res) noexcept {
    assert(th != nullptr && spec != nullptr);
    assert(src != nullptr && res != nullptr);
    const Schema* cur = metadata_current_schema(&th->meta);
    for (uint32_t i = 0; i < spec->n_fields; ++i) {                       // bounded: kIcebergMaxFieldsPerSpec
        const int32_t sid = spec->fields[i].source_id;
        src[i] = IcebergTypeInfo{};
        bool found = false;
        for (uint32_t k = 0; cur != nullptr && k < cur->n_fields && !found; ++k)       // bounded by n_fields
            if (cur->fields[k].id == sid) found = iceberg_type_parse(cur->fields[k].type, &src[i]) || true;
        for (uint32_t s = th->meta.n_schemas; s-- > 0u && !found;) {      // bounded by n_schemas
            const Schema& sch = th->meta.schemas[s];
            for (uint32_t k = 0; k < sch.n_fields && !found; ++k)         // bounded by n_fields
                if (sch.fields[k].id == sid) found = iceberg_type_parse(sch.fields[k].type, &src[i]) || true;
        }
        if (!partition_result_type(spec->fields[i].transform, src[i], &res[i])) return false;
    }
    return true;
}

// ---- row gather ------------------------------------------------------------------------------------------------

bool gather_validity(Arena* a, const BoltColumn& c, const uint32_t* rows, uint32_t n, BoltColumn* out) noexcept {
    assert(a != nullptr && out != nullptr);
    assert(rows != nullptr || n == 0u);
    out->validity = nullptr;
    out->validity_offset = 0;
    out->stats.all_valid = true;
    if (c.validity == nullptr) return true;
    auto* v = static_cast<uint8_t*>(a->allocate_zeroed(n / 8u + 1u));
    if (v == nullptr) return false;
    bool all = true;
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        const bool ok = !c.is_null(rows[i]);
        if (ok) v[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u));
        all = all && ok;
    }
    out->validity = v;
    out->stats.all_valid = all;
    return true;
}

bool gather_var_binary(Arena* a, const BoltColumn& c, const uint32_t* rows, uint32_t n, BoltColumn* out) noexcept {
    assert(a != nullptr && out != nullptr);
    assert(c.format == ColumnFormat::VarBinary);
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        const uint8_t* d = nullptr; int32_t len = 0;
        c.utf8_at(rows[i], &d, &len);
        total += static_cast<uint64_t>(len);
    }
    if (total > static_cast<uint64_t>(INT32_MAX)) return false;
    auto* bytes = a->allocate_array<uint8_t>(static_cast<size_t>(total) + 1u);
    auto* offs = a->allocate_array<int32_t>(static_cast<size_t>(n) + 1u);
    if (bytes == nullptr || offs == nullptr) return false;
    int32_t o = 0;
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        const uint8_t* d = nullptr; int32_t len = 0;
        c.utf8_at(rows[i], &d, &len);
        offs[i] = o;
        if (len > 0) std::memcpy(bytes + o, d, static_cast<size_t>(len));
        o += len;
    }
    offs[n] = o;
    const BoltColumn vb = BoltColumn::make_var_binary(bytes, nullptr, offs, n, c.type, a);
    if (vb.dict_child == nullptr) return false;
    out->data = vb.data;
    out->dict_child = vb.dict_child;
    return true;
}

// Rows `rows[0..n)` of `in` as a new column over arena storage (values copied; spilled string bytes stay shared).
bool gather_column(Arena* a, const BoltColumn& in, const uint32_t* rows, uint32_t n, BoltColumn* out) noexcept {
    assert(a != nullptr && out != nullptr);
    assert(rows != nullptr || n == 0u);
    BoltColumn src = in;
    if (in.format != ColumnFormat::Flat && in.format != ColumnFormat::View && in.format != ColumnFormat::VarBinary) {
        src = in.materialize(a);
        if (src.format != ColumnFormat::Flat || (src.data == nullptr && in.length > 0)) return false;
    }
    *out = src;
    out->length = n;
    out->format = src.format == ColumnFormat::View ? ColumnFormat::Flat : src.format;
    if (!gather_validity(a, src, rows, n, out)) return false;
    if (src.format == ColumnFormat::VarBinary) return gather_var_binary(a, src, rows, n, out);
    const bool view = src.type == BoltType::Utf8 || src.type == BoltType::Binary;
    const size_t w = view ? sizeof(StringView)
                          : (src.type_size_bytes != 0u ? src.type_size_bytes : type_size(src.type));
    if (w == 0u || (src.data == nullptr && n != 0u)) return false;
    auto* d = static_cast<uint8_t*>(a->allocate(static_cast<size_t>(n) * w + 1u, 16u));
    if (d == nullptr) return false;
    const auto* s = static_cast<const uint8_t*>(src.data);
    for (uint32_t i = 0; i < n; ++i) std::memcpy(d + i * w, s + static_cast<size_t>(rows[i]) * w, w);   // bounded by n
    out->data = d;
    return true;
}

bool gather_batch(Arena* a, const BoltBatch* in, const uint32_t* rows, uint32_t n, BoltBatch* out) noexcept {
    assert(a != nullptr && in != nullptr && out != nullptr);
    assert(rows != nullptr || n == 0u);
    BoltBatch::init_empty(out);
    out->arena = a;
    out->num_rows = n;
    if (!BoltBatch::alloc_columns(out, a, in->num_cols)) return false;
    out->schema = in->schema;                                             // field descriptors are shared
    const BoltColumn* ic = in->columns[in->read_epoch];
    BoltColumn* oc = out->columns[out->read_epoch];
    for (uint32_t c = 0; c < in->num_cols; ++c)                           // bounded by num_cols
        if (!gather_column(a, ic[c], rows, n, &oc[c])) return false;
    return true;
}

// ---- grouping rows by tuple -------------------------------------------------------------------------------------

uint64_t tuple_hash(const PartitionValue* t, uint32_t n, int32_t spec_id) noexcept {
    assert(t != nullptr || n == 0u);
    uint64_t h = 0xcbf29ce484222325ull ^ static_cast<uint64_t>(static_cast<uint32_t>(spec_id));
    auto mix = [&h](const void* p, size_t len) noexcept {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < len; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }   // bounded by len
    };
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        const PartitionValue& v = t[i];
        const uint8_t tag = v.is_null ? 0u : v.is_float ? 3u : v.is_str ? 2u : 1u;
        mix(&tag, 1u);
        if (v.is_null) continue;
        if (v.is_float) mix(&v.f64, sizeof(double));
        else if (v.is_str) mix(v.str, partition_value_bytes_len(v));
        else mix(&v.i64, sizeof(int64_t));
    }
    return h;
}

bool tuple_equal(const PartitionValue* a, const PartitionValue* b, uint32_t n) noexcept {
    assert(a != nullptr && b != nullptr);
    for (uint32_t i = 0; i < n; ++i)                                      // bounded by n
        if (!partition_value_equal(a[i], b[i])) return false;
    return true;
}

// One partition tuple of an append and the per-batch row selections that hold it.
struct PartGroup {
    PartitionValue*   tuple;
    const BoltBatch** batches;
    uint32_t          n_batches, cap_batches;
    int64_t           rows;
};

struct PartSplit {
    const PartitionSpec* spec;
    IcebergTypeInfo      src[kIcebergMaxFieldsPerSpec];
    PartGroup*           groups;
    uint32_t             n_groups, cap_groups;
    SwissTableGrowable   index;                                           // tuple hash -> group
};

// The group of `t`, created when new. Hash collisions probe forward to the next key.
int64_t split_group(TableHandle* th, PartSplit* sp, const PartitionValue* t) noexcept {
    assert(th != nullptr && sp != nullptr && t != nullptr);
    const uint32_t nf = sp->spec->n_fields;
    uint64_t key = tuple_hash(t, nf, sp->spec->spec_id);
    for (uint32_t probe = 0; probe <= sp->n_groups; ++probe, ++key) {    // bounded: one probe per existing group
        const int32_t g = sp->index.find(key);
        if (g < 0) break;
        if (tuple_equal(sp->groups[g].tuple, t, nf)) return g;
    }
    if (!lake_grow(th->arena, nullptr, &sp->groups, sp->n_groups, &sp->cap_groups, sp->n_groups + 1u)) return -1;
    PartGroup& pg = sp->groups[sp->n_groups];
    pg = PartGroup{};
    pg.tuple = th->arena->allocate_array<PartitionValue>(nf + 1u);
    if (pg.tuple == nullptr || !sp->index.insert(key, sp->n_groups)) return -1;
    std::memcpy(pg.tuple, t, sizeof(PartitionValue) * nf);
    return static_cast<int64_t>(sp->n_groups++);
}

// For each partition field, the batch column holding its source (by the current schema's name), -1 for a void field.
bool source_columns(const TableHandle* th, const PartSplit* sp, const BoltBatch* b, int32_t* col) noexcept {
    assert(th != nullptr && sp != nullptr && b != nullptr && col != nullptr);
    const Schema* cur = metadata_current_schema(&th->meta);
    for (uint32_t i = 0; i < sp->spec->n_fields; ++i) {                   // bounded: kIcebergMaxFieldsPerSpec
        const PartitionField& f = sp->spec->fields[i];
        col[i] = -1;
        if (f.transform.kind == TransformKind::kVoid) continue;
        const char* name = nullptr;
        for (uint32_t k = 0; cur != nullptr && k < cur->n_fields; ++k)   // bounded by n_fields
            if (cur->fields[k].id == f.source_id) name = cur->fields[k].name;
        col[i] = name == nullptr ? -1 : b->schema.find_field(name);
        if (col[i] < 0 || static_cast<uint32_t>(col[i]) >= b->num_cols) return false;
    }
    return true;
}

// Gather the rows of `b` into one batch per group they fell in (`gid[r]` = row r's group), appended to that group.
bool split_gather(TableHandle* th, PartSplit* sp, const BoltBatch* b, const uint32_t* gid) noexcept {
    assert(th != nullptr && sp != nullptr && b != nullptr && gid != nullptr);
    Arena* a = th->arena;
    const uint32_t ng = sp->n_groups, nr = static_cast<uint32_t>(b->num_rows);
    auto* start = static_cast<uint32_t*>(a->allocate_zeroed((static_cast<size_t>(ng) + 1u) * sizeof(uint32_t)));
    auto* order = a->allocate_array<uint32_t>(static_cast<size_t>(nr) + 1u);
    if (start == nullptr || order == nullptr) return false;
    for (uint32_t r = 0; r < nr; ++r) ++start[gid[r] + 1u];              // counting sort by group: bounded by nr
    for (uint32_t g = 0; g < ng; ++g) start[g + 1u] += start[g];        // bounded by ng
    auto* fill = a->allocate_array<uint32_t>(static_cast<size_t>(ng) + 1u);
    if (fill == nullptr) return false;
    std::memcpy(fill, start, sizeof(uint32_t) * ng);
    for (uint32_t r = 0; r < nr; ++r) order[fill[gid[r]]++] = r;         // rows stay in batch order per group
    for (uint32_t g = 0; g < ng; ++g) {                                   // bounded by ng
        const uint32_t n = start[g + 1u] - start[g];
        if (n == 0u) continue;
        void* mem = a->allocate(sizeof(BoltBatch), alignof(BoltBatch));
        if (mem == nullptr) return false;
        BoltBatch* gb = new (mem) BoltBatch();
        if (!gather_batch(a, b, order + start[g], n, gb)) return false;
        PartGroup& pg = sp->groups[g];
        if (!lake_grow(a, nullptr, &pg.batches, pg.n_batches, &pg.cap_batches, pg.n_batches + 1u)) return false;
        pg.batches[pg.n_batches++] = gb;
        pg.rows += n;
    }
    return true;
}

// Assign every row of `b` to its tuple's group; gather each group's rows into a batch of that group.
bool split_batch(TableHandle* th, PartSplit* sp, const BoltBatch* b) noexcept {
    assert(th != nullptr && sp != nullptr && b != nullptr);
    Arena* a = th->arena;
    const uint32_t nf = sp->spec->n_fields;
    const int64_t nr = b->num_rows;
    int32_t col[kIcebergMaxFieldsPerSpec];
    if (nr <= 0) return true;
    if (nr > static_cast<int64_t>(UINT32_MAX) || !source_columns(th, sp, b, col))
        return refuse_named(th, "an append whose batch lacks a partition source column (or holds 2^32 rows)");
    auto* gid = a->allocate_array<uint32_t>(static_cast<size_t>(nr));
    auto* t = a->allocate_array<PartitionValue>(nf + 1u);
    if (gid == nullptr || t == nullptr) return false;
    const BoltColumn* cols = b->columns[b->read_epoch];
    for (int64_t r = 0; r < nr; ++r) {                                    // bounded by num_rows
        for (uint32_t i = 0; i < nf; ++i) {                               // bounded by n_fields
            const PartitionField& f = sp->spec->fields[i];
            if (col[i] < 0) { std::memset(&t[i], 0, sizeof(t[i])); t[i].field_id = f.field_id; t[i].is_null = true; continue; }
            if (!partition_value_of(f, sp->src[i], cols[col[i]], r, &t[i]))
                return refuse_named(th, "a partition value of this column type, or wider than 63 bytes");
        }
        const int64_t g = split_group(th, sp, t);
        if (g < 0) return false;
        gid[r] = static_cast<uint32_t>(g);
    }
    return split_gather(th, sp, b, gid);
}

bool split_init(TableHandle* th, const PartitionSpec* spec, PartSplit* sp) noexcept {
    assert(th != nullptr && spec != nullptr && sp != nullptr);
    assert(spec->n_fields > 0u);
    std::memset(static_cast<void*>(sp), 0, sizeof(*sp));
    sp->spec = spec;
    IcebergTypeInfo res[kIcebergMaxFieldsPerSpec];
    if (!spec_types_for(th, spec, sp->src, res))
        return refuse_named(th, "a partition spec with an unknown transform, or one not defined for its source type");
    return SwissTableGrowable::create(&sp->index, 16u, th->arena);
}

// The data files of a partitioned append of `batches`: one per tuple, each stamped with the spec and its tuple.
// `rels[i]` is file i's store key. Written files are left to the caller to complete (published or abandoned).
bool write_partitioned_files(TableHandle* th, const PartitionSpec* spec, const BoltBatch* const* batches,
                             uint32_t n_batches, int64_t snap_id, DataFileRef** out, char (**rels)[128],
                             uint32_t* n_out) noexcept {
    assert(th != nullptr && spec != nullptr && out != nullptr && rels != nullptr && n_out != nullptr);
    *n_out = 0;
    PartSplit* sp = th->arena->allocate_array<PartSplit>(1);
    if (sp == nullptr || !split_init(th, spec, sp)) return false;
    for (uint32_t i = 0; i < n_batches; ++i)                              // bounded by n_batches
        if (!split_batch(th, sp, batches[i])) return false;
    if (sp->n_groups > 1u && th->file_io != nullptr)
        return refuse_named(th, "a partitioned write spanning more than one partition through FileWriteIO (one staged file per commit)");
    *out = th->arena->allocate_array<DataFileRef>(sp->n_groups + 1u);
    *rels = th->arena->allocate_array<char[128]>(sp->n_groups + 1u);
    if (*out == nullptr || *rels == nullptr) return false;
    for (uint32_t g = 0; g < sp->n_groups; ++g) {                         // bounded by n_groups
        const PartGroup& pg = sp->groups[g];
        int64_t size = 0, rows = 0;
        if (!write_data_file(th, pg.batches, pg.n_batches, snap_id, (*rels)[g], 128u, &size, &rows)) return false;
        DataFileRef& df = (*out)[(*n_out)++];
        df = DataFileRef{};
        df.status = ManifestStatus::kAdded;
        df.content = FileContent::kData;
        df.partition_spec_id = spec->spec_id;
        df.snapshot_id = snap_id;
        df.n_partition = spec->n_fields;
        std::memcpy(df.partition, pg.tuple, sizeof(PartitionValue) * spec->n_fields);
        if (!abs_location(th->root, (*rels)[g], df.file_path, sizeof(df.file_path))) return false;
        df.stats.record_count = rows;
        df.stats.file_size_in_bytes = size;
        if (!compute_file_stats(th, pg.batches, pg.n_batches, &df.stats)) return false;
    }
    return true;
}

// append_commit_ex on a partitioned table: one data file per tuple under the default spec, one snapshot.
// `out_path` reports the first file written; `out_rows` every row.
bool append_partitioned(AppendHandle* ah, const PartitionSpec* spec, int64_t snap_id, char* out_path,
                        uint32_t out_path_cap, int64_t* out_rows) noexcept {
    assert(ah != nullptr && ah->th != nullptr && spec != nullptr);
    assert(snap_id > 0);
    TableHandle* th = ah->th;
    DataFileRef* files = nullptr;
    char (*rels)[128] = nullptr;
    uint32_t n = 0;
    bool ok = write_partitioned_files(th, spec, ah->pending, ah->n_pending, snap_id, &files, &rels, &n);
    if (ok && n == 0u) { ah->n_pending = 0; th->last_error = CommitError::kNone; return true; }   // no rows: no snapshot
    ok = ok && publish_snapshot(th, files, n, SnapshotOp::kAppend, snap_id);
    for (uint32_t i = 0; i < n; ++i) data_file_complete(th, rels[i], ok);   // bounded by n
    if (!ok) {
        if (th->last_error == CommitError::kNone) th->last_error = CommitError::kFailed;
        return false;
    }
    int64_t rows = 0;
    for (uint32_t i = 0; i < n; ++i) rows += files[i].stats.record_count;   // bounded by n
    ah->n_pending = 0;
    if (out_path != nullptr && out_path_cap != 0u) {
        std::strncpy(out_path, n != 0u ? files[0].file_path : "", out_path_cap - 1u);
        out_path[out_path_cap - 1u] = '\0';
    }
    if (out_rows != nullptr) *out_rows = rows;
    return true;
}

// Reader-to-writer lane preparation is shared through write_batch_conform.
