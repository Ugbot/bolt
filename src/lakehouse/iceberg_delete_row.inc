// bolt/lakehouse/iceberg_delete_row.inc — position deletes that carry the deleted row (G2ICE-283). Included inside
// iceberg_writer.cpp after iceberg_dml_partition.inc and before iceberg_dml_ops.inc; not a standalone TU.
//
// Spec "Position Delete Files": an optional third column `row` (field id 2147483544) is a struct of the table's columns
// holding each deleted row as it was in its data file. Rows stay sorted by file_path then pos.

namespace {

constexpr int32_t kPosDeleteRowId = 2147483544;

// The `row` struct is written for primitive top-level columns (what this writer writes into data files).
bool row_schema_ok(TableHandle* th) noexcept {
    assert(th != nullptr);
    const Schema* sch = metadata_current_schema(&th->meta);
    if (sch == nullptr || sch->n_fields == 0u || sch->n_fields > 0xFFFEu) return false;
    for (uint32_t i = 0; i < sch->n_fields; ++i) {                        // bounded by n_fields
        IcebergTypeInfo info{};
        if (std::strcmp(sch->fields[i].type,"unknown") && !iceberg_type_parse(sch->fields[i].type, &info)) return false;
    }
    return true;
}

void row_node_of(const ingest::parquet::ParquetWriteColumn& c, ingest::parquet::ParquetNestedNode* n) noexcept {
    assert(n != nullptr);
    *n = ingest::parquet::ParquetNestedNode{};
    std::memcpy(n->name, c.name, sizeof(n->name));
    n->type = c.type;
    n->nullable = c.nullable;
    n->has_field_id = c.has_field_id;
    n->field_id = c.field_id;
    n->precision = c.precision;
    n->scale = c.scale;
    n->logical = c.logical;
    n->ts_local = c.ts_local;
    n->timestamp_unit = c.timestamp_unit;
    n->time_micros = c.time_micros;
    n->integer_bits = c.integer_bits;
    n->integer_signed = c.integer_signed;
    n->fixed_len = c.fixed_len;
}

// Parquet options: file_path, pos, and `row` as a struct of the current schema's columns with their field ids.
bool row_delete_opts(TableHandle* th, ingest::parquet::ParquetWriteOpts* po,
                     std::vector<ingest::parquet::ParquetWriteColumn>* wide) noexcept {
    assert(th != nullptr && po != nullptr && wide != nullptr);
    Schema dsch{};
    if (!position_delete_schema(th->arena, &dsch) || !make_pq_opts(&dsch, nullptr, po, wide,th->arena)) return false;
    const Schema* cur = metadata_current_schema(&th->meta);
    auto* tmp = th->arena->allocate_array<ingest::parquet::ParquetWriteOpts>(1);
    std::vector<ingest::parquet::ParquetWriteColumn> wide_row;
    if (tmp == nullptr || cur == nullptr || !make_pq_opts(cur, nullptr, tmp, &wide_row,th->arena)) return false;
    const uint32_t n = tmp->n_columns;
    auto* nodes = th->arena->allocate_array<ingest::parquet::ParquetNestedNode>(n + 1u);
    if (nodes == nullptr) return false;
    nodes[0] = ingest::parquet::ParquetNestedNode{};
    std::strncpy(nodes[0].name, "row", sizeof(nodes[0].name) - 1u);
    nodes[0].type = BoltType::Struct;
    nodes[0].nullable = true;
    nodes[0].has_field_id = true;
    nodes[0].field_id = kPosDeleteRowId;
    nodes[0].first_child = 1u;
    nodes[0].n_children = static_cast<uint16_t>(n);
    for (uint32_t i = 0; i < n; ++i) row_node_of(ingest::parquet::pw_opts_column(*tmp, i), &nodes[i + 1u]);   // bounded by n
    ingest::parquet::ParquetWriteColumn& rc = po->columns[2];
    rc = ingest::parquet::ParquetWriteColumn{};
    std::strncpy(rc.name, "row", sizeof(rc.name) - 1u);
    rc.type = BoltType::Struct;
    rc.nullable = true;
    rc.nested = nodes;
    rc.n_nested = n + 1u;
    po->n_columns = 3u;
    return true;
}

// One row group: `path` on every row, pos[0..n), and `rows` (n rows of the current schema) as the `row` struct.
bool row_delete_batch(Arena* a, const char* path, const int64_t* pos, const BoltBatch* rows, BoltBatch* out) noexcept {
    assert(a != nullptr && path != nullptr && pos != nullptr && rows != nullptr && out != nullptr);
    const uint32_t n = static_cast<uint32_t>(rows->num_rows);
    auto* paths = a->allocate_array<const char*>(n + 1u);
    if (paths == nullptr || n == 0u) return false;
    for (uint32_t i = 0; i < n; ++i) paths[i] = path;                     // bounded by n
    BoltBatch two{};
    if (!pos_delete_batch(a, paths, pos, n, &two)) return false;
    BoltBatch::init_empty(out);
    out->arena = a;
    out->num_rows = n;
    out->num_cols = 3;
    if (!BoltBatch::alloc_columns(out, a, 3)) return false;
    out->schema.add_field("file_path", BoltType::Utf8, false);
    out->schema.add_field("pos", BoltType::Int64, false);
    out->schema.add_field("row", BoltType::Struct, true);
    BoltColumn* oc = out->columns[out->read_epoch];
    oc[0] = two.columns[two.read_epoch][0];
    oc[1] = two.columns[two.read_epoch][1];
    oc[2] = BoltColumn::make_struct(rows->columns[rows->read_epoch], rows->num_cols, n, nullptr, a);
    return oc[2].data != nullptr;
}

// The partly deleted files of one (spec, tuple), in file_path order.
struct RowGroup {
    const DataFileRef* like;
    uint32_t*          files;
    uint32_t           n, cap;
};

bool write_row_delete_file(TableHandle* th, int64_t snap_id, const DmlPlan* p, const RowGroup& g, char (&rel)[128],
                           DataFileRef* df) noexcept {
    assert(th != nullptr && p != nullptr && df != nullptr);
    assert(g.n > 0u);
    const int rn = std::snprintf(rel, sizeof(rel), "data/%lld-%016llx-deletes.parquet",
                                 static_cast<long long>(snap_id), static_cast<unsigned long long>(unique_u64()));
    if (rn <= 0 || static_cast<uint32_t>(rn) >= sizeof(rel)) return false;
    auto* po = th->arena->allocate_array<ingest::parquet::ParquetWriteOpts>(1);
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (po == nullptr || !row_delete_opts(th, po, &wide)) return false;
    auto* w = data_file_begin(th, rel, snap_id, po);
    if (w == nullptr) { th->last_error = CommitError::kFailed; return false; }
    int64_t total = 0;
    for (uint32_t k = 0; k < g.n; ++k) {                                  // bounded by the group's files
        const DmlFile& f = p->files[g.files[k]];
        uint32_t off = 0;
        for (uint32_t b = 0; b < f.n_rows; ++b) {                         // bounded by the file's matched batches
            BoltBatch rb{},physical{};
            const auto* schema=metadata_current_schema(&th->meta);
            if (off + f.rows[b]->num_rows > f.n_pos || !schema || !physical_batch(*schema,*f.rows[b],th->arena,&physical) ||
                !row_delete_batch(th->arena, f.path, f.pos + off, &physical, &rb) ||
                !ingest::parquet::parquet_write_row_group(w, &rb)) { data_file_abandon(th, w); return false; }
            off += static_cast<uint32_t>(f.rows[b]->num_rows);
        }
        if (off != f.n_pos) { data_file_abandon(th, w); return false; }
        total += off;
    }
    int64_t size = 0;
    if (!data_file_finish(th, rel, w, &size)) return false;
    *df = DataFileRef{};
    df->status = ManifestStatus::kAdded;
    df->content = FileContent::kPositionDeletes;
    df->partition_spec_id = g.like != nullptr ? g.like->partition_spec_id : th->meta.current_spec_id;
    df->snapshot_id = snap_id;
    if (g.like != nullptr) {
        df->n_partition = g.like->n_partition;
        std::memcpy(df->partition, g.like->partition, sizeof(PartitionValue) * df->n_partition);
    }
    df->stats.record_count = total;
    df->stats.file_size_in_bytes = size;
    if (!abs_location(th->root, rel, df->file_path, sizeof(df->file_path))) { data_file_complete(th, rel, false); return false; }
    return true;
}

bool same_tuple(const DataFileRef* x, const DataFileRef* y) noexcept {
    if (x == y) return true;
    if (x == nullptr || y == nullptr) return false;
    return x->partition_spec_id == y->partition_spec_id && x->n_partition == y->n_partition &&
           tuple_equal(x->partition, y->partition, x->n_partition);
}

// The partly deleted files grouped by their data file's (spec, tuple); file_path order is kept within a group.
bool row_groups(TableHandle* th, const DmlPlan* p, const LiveData* live, RowGroup** gs, uint32_t* ng) noexcept {
    assert(th != nullptr && p != nullptr && live != nullptr && gs != nullptr && ng != nullptr);
    const char** rm = nullptr;
    uint32_t n_rm = 0, n_part = 0, cap = 0;
    uint32_t* order = nullptr;
    uint64_t rows = 0;
    if (!dml_partition_files(th, p, &rm, &n_rm, &order, &n_part, &rows)) return false;
    const PartitionSpec* cur = metadata_spec(&th->meta, th->meta.current_spec_id);
    *ng = 0;
    for (uint32_t k = 0; k < n_part; ++k) {                               // bounded by the partly deleted files
        const DataFileRef* like = live_data_find(live, p->files[order[k]].path);
        if (like == nullptr && cur != nullptr && cur->n_fields != 0u) return false;
        uint32_t g = 0;
        while (g < *ng && !same_tuple((*gs)[g].like, like)) ++g;        // bounded by the groups
        if (g == *ng) {
            if (!lake_grow(th->arena, nullptr, gs, *ng, &cap, *ng + 1u)) return false;
            (*gs)[(*ng)++] = RowGroup{like, nullptr, 0, 0};
        }
        RowGroup& rg = (*gs)[g];
        if (!lake_grow(th->arena, nullptr, &rg.files, rg.n, &rg.cap, rg.n + 1u)) return false;
        rg.files[rg.n++] = order[k];
    }
    return true;
}

// One position-delete file with rows per (spec, tuple) of the partly deleted files.
bool write_row_deletes(TableHandle* th, int64_t snap_id, const DmlPlan* p, const LiveData* live, DataFileRef** out,
                       char (**rels)[128], uint32_t* n_out) noexcept {
    assert(th != nullptr && p != nullptr && live != nullptr);
    assert(out != nullptr && rels != nullptr && n_out != nullptr);
    *n_out = 0;
    RowGroup* gs = nullptr;
    uint32_t ng = 0;
    if (!row_groups(th, p, live, &gs, &ng)) return false;
    if (ng > 1u && th->file_io != nullptr)
        return refuse_named(th, "position deletes spanning more than one partition through FileWriteIO (one staged file per commit)");
    *out = th->arena->allocate_array<DataFileRef>(ng + 1u);
    *rels = th->arena->allocate_array<char[128]>(ng + 1u);
    if (*out == nullptr || *rels == nullptr) return false;
    for (uint32_t g = 0; g < ng; ++g) {                                   // bounded by ng
        if (!write_row_delete_file(th, snap_id, p, gs[g], (*rels)[g], &(*out)[g])) return false;
        ++*n_out;
    }
    return true;
}

}  // namespace
