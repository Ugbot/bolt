// bolt/lakehouse/iceberg_dml.inc — predicate delete and overwrite (G2ICE-280). Included inside iceberg_writer.cpp
// at namespace scope, after position_delete_schema; not a standalone TU.
//
// table_delete(pred) reads the current snapshot's LIVE rows (every delete already applied) through the table scan,
// with each row's file position, and evaluates `pred` on them (dml::eval_predicate, SQL three-valued: a NULL never
// matches a comparison). Per live data file:
//   * every live row matches   -> the file is removed (its manifest entry becomes DELETED);
//   * some live rows match     -> one v2 position-delete file names exactly those (file_path, pos) pairs, sorted by
//                                 file_path then pos (spec: "Position Delete Files");
//   * no row matches           -> untouched.
// All of it is ONE snapshot with operation "delete" (spec: "Data files were removed and their contents logically
// deleted and/or delete files were added to delete rows"). A predicate that matches nothing commits nothing and
// returns true, as pyiceberg does (Java would commit an empty "delete" snapshot; the spec defines neither).
// table_overwrite(pred, batch) is the same delete plus an append of `batch`, in one "overwrite" snapshot, committed
// even when the predicate matches nothing.

namespace {

// One position-delete parquet file from a (file_path, pos) batch; `df` describes it for the manifest.
bool write_pos_delete_batch(TableHandle* th, int64_t snap_id, const BoltBatch* b, char (&rel)[128],
                            DataFileRef* df) noexcept {
    assert(th != nullptr && b != nullptr && df != nullptr);
    assert(b->num_cols == 2u && b->num_rows > 0);
    const int rn = std::snprintf(rel, sizeof(rel), "data/%lld-%016llx-deletes.parquet",
                                 static_cast<long long>(snap_id), static_cast<unsigned long long>(unique_u64()));
    if (rn <= 0 || static_cast<uint32_t>(rn) >= sizeof(rel)) return false;
    Schema dsch{};
    if (!position_delete_schema(th->arena, &dsch)) return false;
    ingest::parquet::ParquetWriteOpts po{};
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (!make_pq_opts(&dsch, nullptr, &po, &wide)) return false;
    // G2ICE-117: the spec-RESERVED field ids in the physical footer ([0]=file_path, [1]=pos).
    assert(po.n_columns == 2u && "position_delete_schema shape changed");
    po.columns[0].has_field_id = true;
    po.columns[0].field_id     = dsch.fields[0].id;   // 2147483546 file_path
    po.columns[1].has_field_id = true;
    po.columns[1].field_id     = dsch.fields[1].id;   // 2147483545 pos
    auto* w = data_file_begin(th, rel, snap_id, &po);
    if (w == nullptr) { th->last_error = CommitError::kFailed; return false; }
    if (!ingest::parquet::parquet_write_row_group(w, b)) {
        data_file_abandon(th, w);
        th->last_error = CommitError::kFailed;
        return false;
    }
    int64_t size = 0;
    if (!data_file_finish(th, rel, w, &size)) return false;
    *df = DataFileRef{};
    df->status            = ManifestStatus::kAdded;
    df->content           = FileContent::kPositionDeletes;
    df->partition_spec_id = th->meta.current_spec_id;
    df->snapshot_id       = snap_id;
    if (!abs_location(th->root, rel, df->file_path, sizeof(df->file_path))) {
        data_file_complete(th, rel, false);
        return false;
    }
    df->stats.record_count       = b->num_rows;
    df->stats.file_size_in_bytes = size;
    return true;
}

// A (file_path Utf8, pos Int64) batch over arena storage; paths[i] is row i's file (pointers may repeat).
bool pos_delete_batch(Arena* a, const char* const* paths, const int64_t* pos, uint32_t n, BoltBatch* b) noexcept {
    assert(a != nullptr && b != nullptr);
    assert(paths != nullptr && pos != nullptr && n > 0);
    uint64_t spill_bytes = 0;
    for (uint32_t i = 0; i < n; ++i)                            // bounded by n
        if (i == 0 || paths[i] != paths[i - 1]) spill_bytes += std::strlen(paths[i]);
    if (spill_bytes > UINT32_MAX - 1u) return false;
    auto* svs = a->allocate_array<StringView>(n);
    auto* spill = a->allocate_array<uint8_t>(spill_bytes + 1u);
    auto* pv = a->allocate_array<int64_t>(n);
    if (svs == nullptr || spill == nullptr || pv == nullptr) return false;
    uint32_t off = 0, run_off = 0;
    for (uint32_t i = 0; i < n; ++i) {                          // bounded by n
        const uint32_t len = static_cast<uint32_t>(std::strlen(paths[i]));
        if (len == 0 || pos[i] < 0) return false;
        if (i == 0 || paths[i] != paths[i - 1]) { run_off = off; std::memcpy(spill + off, paths[i], len); off += len; }
        svs[i] = StringView::from_cstr(paths[i]);
        if (len > 12u) { svs[i].ref.buf_idx = 0; svs[i].ref.offset = run_off; }
        pv[i] = pos[i];
    }
    BoltBatch::init_empty(b);
    b->arena = a;
    b->num_rows = static_cast<int64_t>(n);
    b->num_cols = 2;
    if (!BoltBatch::alloc_columns(b, a, 2)) return false;
    b->schema.add_field("file_path", BoltType::Utf8, false);
    b->schema.add_field("pos", BoltType::Int64, false);
    BoltColumn* cols = b->columns[b->read_epoch];
    cols[0].type = BoltType::Utf8;
    cols[0].length = static_cast<int64_t>(n);
    cols[0].data = svs;
    cols[0].str_overflow_base = spill;
    cols[1].type = BoltType::Int64;
    cols[1].length = static_cast<int64_t>(n);
    cols[1].data = pv;
    return true;
}

// One data file the predicate pass read: its live rows and the positions that matched.
struct DmlFile {
    const char* path;
    int64_t     live_rows;
    int64_t*    pos;
    uint32_t    n_pos;
    uint32_t    cap_pos;
    BoltBatch** rows;        // with_rows: the matched rows, aligned with pos (G2ICE-283 position `row` column)
    uint32_t    n_rows, cap_rows;
    uint8_t*    live;        // dv: bitmap of the positions read live (G2ICE-283 deletion vectors)
    uint32_t    n_live;      // bytes of `live`
    uint32_t    _pad;
};

struct DmlPlan {
    TableHandle* th;
    DmlFile*     files;
    uint32_t     n_files;
    uint32_t     cap_files;
    bool         with_rows;
    bool         dv;          // v3: deletes are deletion vectors (iceberg_delete_dv.inc)
};

bool dml_mark_live(Arena* a, DmlFile* f, int64_t pos) noexcept;

// with_rows: the matched rows of `rb`, gathered in match order (as their positions are), appended to `f`.
bool dml_keep_rows(Arena* a, DmlFile* f, const dml::RowBatch* rb) noexcept {
    assert(a != nullptr && f != nullptr && rb != nullptr);
    void* mem = a->allocate(sizeof(BoltBatch), alignof(BoltBatch));
    if (mem == nullptr) return false;
    BoltBatch* b = new (mem) BoltBatch();
    if (!gather_batch(a, rb->batch, rb->match, rb->n_match, b)) return false;
    if (!lake_grow(a, nullptr, &f->rows, f->n_rows, &f->cap_rows, f->n_rows + 1u)) return false;
    f->rows[f->n_rows++] = b;
    return true;
}

bool dml_visit(void* c, const dml::RowBatch* rb) noexcept {
    auto* p = static_cast<DmlPlan*>(c);
    assert(p != nullptr && rb != nullptr && rb->file_path != nullptr);
    Arena* a = p->th->arena;
    if (p->n_files == 0 || std::strcmp(p->files[p->n_files - 1].path, rb->file_path) != 0) {
        if (!lake_grow(a, nullptr, &p->files, p->n_files, &p->cap_files, p->n_files + 1u)) return false;
        const size_t len = std::strlen(rb->file_path);
        char* path = a->allocate_array<char>(len + 1u);
        if (path == nullptr) return false;
        std::memcpy(path, rb->file_path, len + 1u);
        p->files[p->n_files++] = DmlFile{path, 0, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0};
    }
    DmlFile& f = p->files[p->n_files - 1];
    f.live_rows += rb->batch->num_rows;
    for (int64_t r = 0; p->dv && r < rb->batch->num_rows; ++r)          // bounded by the batch
        if (!dml_mark_live(a, &f, rb->pos[r])) return false;
    if (rb->n_match == 0) return true;
    if (static_cast<uint64_t>(f.n_pos) + rb->n_match > UINT32_MAX) return false;
    if (!lake_grow(a, nullptr, &f.pos, f.n_pos, &f.cap_pos, f.n_pos + rb->n_match)) return false;
    for (uint32_t i = 0; i < rb->n_match; ++i) f.pos[f.n_pos++] = rb->pos[rb->match[i]];   // bounded by n_match
    return !p->with_rows || dml_keep_rows(a, &f, rb);
}

// The table's metadata document at the handle's version, into `a`.
bool current_metadata_bytes(TableHandle* th, Arena* a, const uint8_t** body, uint32_t* len) noexcept {
    assert(th != nullptr && a != nullptr);
    assert(body != nullptr && len != nullptr);
    char rel[64];
    version_key(th->meta_version, rel);
    uint64_t l = 0;
    if (os_get(th->os, rel, a, body, &l) != kOsOk || l == 0 || l > UINT32_MAX) return false;
    *len = static_cast<uint32_t>(l);
    return true;
}

Arena* scratch_arena(TableHandle* th) noexcept {
    assert(th != nullptr && th->arena != nullptr);
    void* p = th->arena->allocate(sizeof(Arena), alignof(Arena));
    return p == nullptr ? nullptr : new (p) Arena();
}

// Read the live rows of the current snapshot into `visit`. kUnsupported / kFailed set last_error.
bool dml_read(TableHandle* th, Arena* ra, const Predicate* pred, dml::FileKeepFn keep, void* keep_ctx,
              dml::RowVisitFn visit, void* ctx) noexcept {
    assert(th != nullptr && ra != nullptr && visit != nullptr);
    const uint8_t* meta = nullptr; uint32_t ml = 0;
    if (!current_metadata_bytes(th, ra, &meta, &ml)) { th->last_error = CommitError::kFailed; return false; }
    const dml::ReadStatus st = dml::read_live_rows(ra, th->os, meta, ml, th->meta.current_snapshot_id, pred,
                                                   keep, keep_ctx, visit, ctx);
    if (st == dml::ReadStatus::kOk) return true;
    if (st == dml::ReadStatus::kUnsupported) return refuse_named(th, "a predicate of this column / literal type");
    th->last_error = CommitError::kFailed;
    return false;
}

// Whole files to remove (`rm`), and the partly deleted files (`order_out`, by file_path) with their deleted row count.
bool dml_partition_files(TableHandle* th, const DmlPlan* p, const char*** rm, uint32_t* n_rm, uint32_t** order_out,
                         uint32_t* n_part_out, uint64_t* rows_out) noexcept {
    assert(th != nullptr && p != nullptr);
    assert(rm != nullptr && n_rm != nullptr && order_out != nullptr && n_part_out != nullptr && rows_out != nullptr);
    Arena* a = th->arena;
    *n_rm = 0;
    uint64_t rows = 0;
    auto* order = a->allocate_array<uint32_t>(p->n_files + 1u);
    *rm = a->allocate_array<const char*>(p->n_files + 1u);
    if (order == nullptr || *rm == nullptr) return false;
    uint32_t n_part = 0;
    for (uint32_t i = 0; i < p->n_files; ++i) {                 // bounded by n_files
        const DmlFile& f = p->files[i];
        if (f.n_pos == 0) continue;
        if (static_cast<int64_t>(f.n_pos) == f.live_rows) { (*rm)[(*n_rm)++] = f.path; continue; }
        order[n_part++] = i;
        rows += f.n_pos;
    }
    for (uint32_t gap = n_part / 2u; gap > 0u; gap /= 2u)       // shell sort by path: bounded, no recursion
        for (uint32_t i = gap; i < n_part; ++i) {
            const uint32_t x = order[i];
            uint32_t j = i;
            while (j >= gap && std::strcmp(p->files[order[j - gap]].path, p->files[x].path) > 0) { order[j] = order[j - gap]; j -= gap; }
            order[j] = x;
        }
    *order_out = order;
    *n_part_out = n_part;
    *rows_out = rows;
    return true;
}

// Whole files to remove and the (path, pos) rows of the partial ones, in file_path then pos order.
bool dml_split(TableHandle* th, const DmlPlan* p, const char*** rm, uint32_t* n_rm, const char*** del_paths,
               int64_t** del_pos, uint32_t* n_del) noexcept {
    assert(th != nullptr && p != nullptr);
    assert(rm != nullptr && n_rm != nullptr && n_del != nullptr);
    Arena* a = th->arena;
    *n_del = 0;
    uint32_t* order = nullptr;
    uint32_t n_part = 0;
    uint64_t rows = 0;
    if (!dml_partition_files(th, p, rm, n_rm, &order, &n_part, &rows)) return false;
    if (rows > UINT32_MAX) return false;
    *del_paths = a->allocate_array<const char*>(rows + 1u);
    *del_pos = a->allocate_array<int64_t>(rows + 1u);
    if (*del_paths == nullptr || *del_pos == nullptr) return false;
    for (uint32_t k = 0; k < n_part; ++k) {                     // bounded by n_part; positions ascend per file
        const DmlFile& f = p->files[order[k]];
        for (uint32_t i = 0; i < f.n_pos; ++i) { (*del_paths)[*n_del] = f.path; (*del_pos)[(*n_del)++] = f.pos[i]; }
    }
    return true;
}

// The data file for an overwrite's new rows, with per-column stats from the batch itself.
bool dml_data_file(TableHandle* th, int64_t snap_id, const BoltBatch* batch, char (&rel)[128],
                   DataFileRef* df) noexcept {
    assert(th != nullptr && batch != nullptr && df != nullptr);
    int64_t size = 0, rows = 0;
    const BoltBatch* one[1] = {batch};
    if (!write_data_file(th, one, 1u, snap_id, rel, sizeof(rel), &size, &rows)) return false;
    *df = DataFileRef{};
    df->status = ManifestStatus::kAdded;
    df->content = FileContent::kData;
    df->partition_spec_id = th->meta.current_spec_id;
    df->snapshot_id = snap_id;
    if (!abs_location(th->root, rel, df->file_path, sizeof(df->file_path))) { data_file_complete(th, rel, false); return false; }
    df->stats.record_count = rows;
    df->stats.file_size_in_bytes = size;
    if (!compute_file_stats(th, one, 1u, &df->stats)) { data_file_complete(th, rel, false); return false; }
    return true;
}

}  // namespace
