// bolt/lakehouse/iceberg_compact.inc — table_rewrite / table_compact (G2ICE-280). Included inside
// iceberg_writer.cpp at namespace scope after iceberg_dml_ops.inc; not a standalone TU.
//
// Bin-pack compaction. The candidates are the live data files smaller than target/2. They are rewritten when there
// are at least two of them, or when one has position deletes against it. Their LIVE rows (every delete applied, read
// through the table scan) are written in scan order into new data files, starting a new output whenever the
// candidates' input bytes would pass the target. One snapshot with operation "replace" (spec: "Data and delete files
// were added and removed without changing table data") removes the candidates and every position-delete file whose
// targets are all candidates (or files no longer live), since those deletes are now applied, and adds the outputs.
// Equality-delete files are kept: they may still apply to other files, and the outputs' higher sequence number keeps
// them from applying again. Row groups of the outputs are the scan's row groups (not re-chunked).

namespace {

struct LiveFile {
    const char* path;
    int64_t     size;
    int64_t     rows;
};

struct CompactPlan {
    LiveFile*    data;
    uint32_t     n_data, cap_data;
    LiveFile*    pos;
    uint32_t     n_pos, cap_pos;
    const char** cand;
    int64_t*     cand_size;
    uint8_t*     cand_dels;
    uint32_t     n_cand;
    uint32_t     n_cand_dels;
    RmIndex      cand_ix;
    RmIndex      live_ix;
    const char** rm_dels;
    uint32_t     n_rm_dels;
    uint32_t     _pad;
};

bool push_live(Arena* a, LiveFile** arr, uint32_t* n, uint32_t* cap, const DataFileRef& e) noexcept {
    assert(a != nullptr && arr != nullptr && n != nullptr);
    if (!lake_grow(a, nullptr, arr, *n, cap, *n + 1u)) return false;
    (*arr)[(*n)++] = LiveFile{e.file_path, e.stats.file_size_in_bytes, e.stats.record_count};
    return true;
}

// Every live data and position-delete file of the current snapshot. Equality deletes are left where they are.
bool compact_collect(TableHandle* th, CompactPlan* p) noexcept {
    assert(th != nullptr && p != nullptr);
    ManifestListEntry* ml = nullptr; uint32_t n = 0;
    if (!parent_list(th, th->meta.current_snapshot_id, &ml, &n)) return false;
    for (uint32_t m = 0; m < n; ++m) {                          // bounded by n
        DataFileRef* e = nullptr; uint32_t ne = 0;
        if (!read_manifest_entries(th, ml[m], &e, &ne)) return false;
        for (uint32_t i = 0; i < ne; ++i) {                     // bounded by ne
            if (e[i].status == ManifestStatus::kDeleted) continue;
            if (e[i].is_dv) return refuse_named(th, "compacting a table with deletion vectors");
            const bool ok = e[i].content == FileContent::kData ? push_live(th->arena, &p->data, &p->n_data, &p->cap_data, e[i])
                          : e[i].content == FileContent::kPositionDeletes ? push_live(th->arena, &p->pos, &p->n_pos, &p->cap_pos, e[i])
                          : true;
            if (!ok) return false;
        }
    }
    return true;
}

bool compact_candidates(TableHandle* th, CompactPlan* p, int64_t target) noexcept {
    assert(th != nullptr && p != nullptr && target > 0);
    Arena* a = th->arena;
    p->cand = a->allocate_array<const char*>(p->n_data + 1u);
    p->cand_size = a->allocate_array<int64_t>(p->n_data + 1u);
    auto* live = a->allocate_array<const char*>(p->n_data + 1u);
    if (p->cand == nullptr || p->cand_size == nullptr || live == nullptr) return false;
    for (uint32_t i = 0; i < p->n_data; ++i) {                  // bounded by n_data
        live[i] = p->data[i].path;
        if (p->data[i].size >= target / 2) continue;
        p->cand_size[p->n_cand] = p->data[i].size;
        p->cand[p->n_cand++] = p->data[i].path;
    }
    p->cand_dels = static_cast<uint8_t*>(a->allocate_zeroed(p->n_cand + 1u));
    return p->cand_dels != nullptr && rm_index_build(a, p->cand, p->n_cand, &p->cand_ix) &&
           rm_index_build(a, live, p->n_data, &p->live_ix);
}

// Which candidates have position deletes, and which position-delete files the rewrite fully applies.
bool compact_deletes(TableHandle* th, Arena* ra, CompactPlan* p) noexcept {
    assert(th != nullptr && ra != nullptr && p != nullptr);
    p->rm_dels = th->arena->allocate_array<const char*>(p->n_pos + 1u);
    if (p->rm_dels == nullptr) return false;
    for (uint32_t d = 0; d < p->n_pos; ++d) {                   // bounded by n_pos
        const uint8_t* body = nullptr; uint64_t len = 0;
        if (os_get(th->os, store_key_for(th->root, p->pos[d].path), ra, &body, &len) != kOsOk) return false;
        const char** t = nullptr; uint32_t nt = 0;
        if (!dml::position_delete_targets(ra, body, len, &t, &nt)) return false;
        bool applied = true, reaches = false;
        for (uint32_t k = 0; k < nt; ++k) {                     // bounded by nt
            const int64_t c = rm_index_find(&p->cand_ix, t[k]);
            if (c >= 0) { reaches = true; if (p->cand_dels[c] == 0) { p->cand_dels[c] = 1; ++p->n_cand_dels; } continue; }
            if (rm_index_find(&p->live_ix, t[k]) >= 0) applied = false;   // still deletes rows of a file we keep
        }
        if (applied && reaches) p->rm_dels[p->n_rm_dels++] = p->pos[d].path;
    }
    return true;
}

// The rewrite's output files, written as the scan delivers the candidates' live rows.
struct CompactOut {
    TableHandle*        th;
    const CompactPlan*  plan;
    int64_t             snap_id;
    int64_t             target;
    const char*         cur_path;
    int64_t             in_bytes;
    ingest::parquet::ParquetWriter* w;
    char                rel[128];
    const BoltBatch**   batches;
    uint32_t            n_batches, cap_batches;
    int64_t             rows;
    DataFileRef*        outs;
    uint32_t            n_outs, cap_outs;
    char                (*rels)[128];
    uint32_t            cap_rels;
    bool                file_open;
};

bool compact_keep(void* c, const DataFileRef* e) noexcept {
    const auto* p = static_cast<const CompactPlan*>(c);
    assert(p != nullptr && e != nullptr);
    return rm_index_find(&p->cand_ix, e->file_path) >= 0;
}

bool compact_close(CompactOut* o) noexcept {
    assert(o != nullptr && o->th != nullptr);
    if (!o->file_open) return true;
    TableHandle* th = o->th;
    o->file_open = false;
    int64_t size = 0;
    if (!data_file_finish(th, o->rel, o->w, &size)) return false;
    if (!lake_grow(th->arena, nullptr, &o->outs, o->n_outs, &o->cap_outs, o->n_outs + 1u) ||
        !lake_grow(th->arena, nullptr, &o->rels, o->n_outs, &o->cap_rels, o->n_outs + 1u)) {
        data_file_complete(th, o->rel, false);
        return false;
    }
    std::memcpy(o->rels[o->n_outs], o->rel, sizeof(o->rel));
    DataFileRef& df = o->outs[o->n_outs++];
    df = DataFileRef{};
    df.status = ManifestStatus::kAdded;
    df.content = FileContent::kData;
    df.partition_spec_id = th->meta.current_spec_id;
    df.snapshot_id = o->snap_id;
    if (!abs_location(th->root, o->rel, df.file_path, sizeof(df.file_path))) return false;
    df.stats.record_count = o->rows;
    df.stats.file_size_in_bytes = size;
    if (const Schema* cur = metadata_current_schema(&th->meta)) compute_file_stats(o->batches, o->n_batches, cur, &df.stats);
    o->n_batches = 0;
    o->rows = 0;
    return true;
}

bool compact_open(CompactOut* o) noexcept {
    assert(o != nullptr && !o->file_open);
    TableHandle* th = o->th;
    const int n = std::snprintf(o->rel, sizeof(o->rel), "data/%lld-%016llx.parquet",
                                static_cast<long long>(o->snap_id), static_cast<unsigned long long>(unique_u64()));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(o->rel)) return false;
    const Schema* sch = metadata_current_schema(&th->meta);
    if (sch == nullptr) return false;
    ingest::parquet::ParquetWriteOpts po{};
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (!make_pq_opts(sch, nullptr, &po, &wide)) return false;
    o->w = data_file_begin(th, o->rel, o->snap_id, &po);
    o->file_open = o->w != nullptr;
    return o->file_open;
}

bool compact_visit(void* c, const dml::RowBatch* rb) noexcept {
    auto* o = static_cast<CompactOut*>(c);
    assert(o != nullptr && rb != nullptr);
    if (o->cur_path == nullptr || std::strcmp(o->cur_path, rb->file_path) != 0) {   // the next input file
        const int64_t k = rm_index_find(&o->plan->cand_ix, rb->file_path);
        if (k < 0) return false;                                // the filter admitted only candidates
        const int64_t size = o->plan->cand_size[k];
        if (o->in_bytes > 0 && o->in_bytes + size > o->target) {
            if (!compact_close(o)) return false;
            o->in_bytes = 0;
        }
        o->in_bytes += size;
        o->cur_path = o->plan->cand[k];
    }
    if (rb->batch->num_rows == 0) return true;
    if (!o->file_open && !compact_open(o)) return false;
    if (!ingest::parquet::parquet_write_row_group(o->w, rb->batch)) return false;
    if (!lake_grow(o->th->arena, nullptr, &o->batches, o->n_batches, &o->cap_batches, o->n_batches + 1u)) return false;
    o->batches[o->n_batches++] = rb->batch;
    o->rows += rb->batch->num_rows;
    return true;
}

void compact_discard(CompactOut* o) noexcept {
    assert(o != nullptr && o->th != nullptr);
    if (o->file_open) { data_file_abandon(o->th, o->w); data_file_complete(o->th, o->rel, false); o->file_open = false; }
    for (uint32_t i = 0; i < o->n_outs; ++i) data_file_complete(o->th, o->rels[i], false);   // bounded by n_outs
}

// Read and write the candidates' live rows, then commit the replace. `ra` holds every batch until the commit.
bool compact_rewrite(TableHandle* th, Arena* ra, CompactPlan* p, int64_t target) noexcept {
    assert(th != nullptr && ra != nullptr && p != nullptr);
    CompactOut o{};
    o.th = th;
    o.plan = p;
    o.snap_id = mint_snapshot_id(th);
    o.target = target;
    bool ok = dml_read(th, ra, nullptr, compact_keep, p, compact_visit, &o) && compact_close(&o);
    if (!ok) {
        if (th->last_error == CommitError::kNone) th->last_error = CommitError::kFailed;
        compact_discard(&o);
        return false;
    }
    ChangeSet cs{};
    cs.op = SnapshotOp::kReplace;
    cs.snap_id = o.snap_id;
    cs.add_data = o.outs;
    cs.n_add_data = o.n_outs;
    cs.rm_data = p->cand;
    cs.n_rm_data = p->n_cand;
    cs.rm_dels = p->rm_dels;
    cs.n_rm_dels = p->n_rm_dels;
    const bool committed = commit_change(th, &cs);
    for (uint32_t i = 0; i < o.n_outs; ++i) data_file_complete(th, o.rels[i], committed);   // bounded by n_outs
    return committed;
}

bool compact_table(TableHandle* th, int64_t target) noexcept {
    assert(th != nullptr && th->arena != nullptr);
    th->last_error = CommitError::kFailed;
    if (target <= 1) return false;
    if (!handle_is_current(th)) return false;
    const PartitionSpec* spec = metadata_spec(&th->meta, th->meta.current_spec_id);
    if (spec != nullptr && spec->n_fields != 0) return refuse_named(th, "compacting a partitioned table");
    if (th->meta.current_snapshot_id <= 0) { th->last_error = CommitError::kNone; return true; }
    CompactPlan p{};
    if (!compact_collect(th, &p) || !compact_candidates(th, &p, target)) return false;
    Arena* ra = scratch_arena(th);
    if (ra == nullptr) return false;
    bool ok = compact_deletes(th, ra, &p);
    int64_t in_bytes = 0;
    for (uint32_t i = 0; i < p.n_cand; ++i) in_bytes += p.cand_size[i];   // bounded by n_cand
    const bool worth = p.n_cand >= 2u || p.n_cand_dels != 0;
    if (ok && worth && th->file_io != nullptr && in_bytes > target) {
        ra->~Arena();
        return refuse_named(th, "a compaction into more than one file through FileWriteIO (one staged file per commit)");
    }
    if (ok && !worth) { ra->~Arena(); th->last_error = CommitError::kNone; return true; }
    ok = ok && compact_rewrite(th, ra, &p, target);
    ra->~Arena();
    return ok;
}

}  // namespace

bool table_rewrite(TableHandle* th, const OptimizeOptions* opts) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    OptimizeOptions o;
    optimize_options_init(&o);
    if (opts != nullptr) o = *opts;
    if (!o.run_bin_pack) { th->last_error = CommitError::kNone; return true; }   // no strategy asked for
    if (o.target_file_size_mb <= 0) { th->last_error = CommitError::kFailed; return false; }
    return compact_table(th, static_cast<int64_t>(o.target_file_size_mb) << 20);
}

bool table_compact(TableHandle* th, int64_t target_file_size_bytes) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    return compact_table(th, target_file_size_bytes);
}
