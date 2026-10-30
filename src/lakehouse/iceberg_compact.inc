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
//
// Format versions (G2ICE-346): v1 tables are compacted as v1 (no delete files exist there). At v3 a deletion vector names one
// data file; a candidate with a DV is rewritten like one with position deletes, and the commit drops the DVs of the removed files.
// Rewritten rows keep their row lineage: each output file writes `_row_id` and `_last_updated_sequence_number` (field ids
// 2147483540 / 2147483539) as the values the rows read before (spec "Row Lineage": a row copied by a compaction keeps them).
//
// Partitioned tables (G2ICE-292): files are only merged within one (spec, partition tuple). Candidates are grouped by
// it; the two-files-or-a-delete rule applies per group, each group is read and written on its own, and its outputs
// carry the group's spec id and tuple (a file never moves to another spec by being compacted).

namespace {

// Test-only negative control (G2ICE-346 drill): BOLT_TEST_COMPACT_DROP_LINEAGE=1 writes v3 outputs without the lineage columns,
// so readers inherit fresh row ids. Never set in production.
bool lineage_test_drop() noexcept {
    const char* v = std::getenv("BOLT_TEST_COMPACT_DROP_LINEAGE");
    return v != nullptr && v[0] == '1';
}

struct LiveFile {
    const char*        path;
    int64_t            size;
    int64_t            rows;
    const DataFileRef* ref;          // its manifest entry (spec id and partition tuple), in the table arena
};

// The data files one position-delete file names.
struct DelTargets {
    const char** t;
    uint32_t     n;
    uint32_t     _pad;
};

struct CompactPlan {
    LiveFile*    data;
    uint32_t     n_data, cap_data;
    LiveFile*    pos;
    uint32_t     n_pos, cap_pos;
    LiveFile*    dvs;                // v3 deletion vectors; `path` is the data file each one names
    uint32_t     n_dvs, cap_dvs;
    const char** cand;
    int64_t*     cand_size;
    uint8_t*     cand_dels;
    uint32_t*    cand_group;         // (spec, tuple) group of each candidate
    const DataFileRef** cand_ref;
    uint32_t     n_cand;
    uint32_t     n_cand_dels;
    uint32_t     n_groups;
    uint32_t     cur_group;          // the group being rewritten (compact_keep reads only its files)
    const DataFileRef** group_like;
    RmIndex      cand_ix;
    RmIndex      live_ix;
    DelTargets*  targets;            // per position-delete file
    const char** rm_dels;
    uint32_t     n_rm_dels;
    uint32_t     _pad;
};

bool push_live(Arena* a, LiveFile** arr, uint32_t* n, uint32_t* cap, const DataFileRef& e) noexcept {
    assert(a != nullptr && arr != nullptr && n != nullptr);
    if (!lake_grow(a, nullptr, arr, *n, cap, *n + 1u)) return false;
    (*arr)[(*n)++] = LiveFile{e.file_path, e.stats.file_size_in_bytes, e.stats.record_count, &e};
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
            if (e[i].is_dv) {
                if (th->meta.format_version < 3) return refuse_named(th, "a deletion vector in a table below format-version 3");
                if (!push_live(th->arena, &p->dvs, &p->n_dvs, &p->cap_dvs, e[i])) return false;
                p->dvs[p->n_dvs - 1u].path = e[i].referenced_data_file;
                continue;
            }
            const bool ok = e[i].content == FileContent::kData ? push_live(th->arena, &p->data, &p->n_data, &p->cap_data, e[i])
                          : e[i].content == FileContent::kPositionDeletes ? push_live(th->arena, &p->pos, &p->n_pos, &p->cap_pos, e[i])
                          : true;
            if (!ok) return false;
        }
    }
    return true;
}

// Group of a candidate: same spec id and the same tuple.
int64_t cand_group_of(TableHandle* th, CompactPlan* p, SwissTableGrowable* ix, const DataFileRef* e) noexcept {
    assert(th != nullptr && p != nullptr && ix != nullptr && e != nullptr);
    uint64_t key = tuple_hash(e->partition, e->n_partition, e->partition_spec_id);
    for (uint32_t probe = 0; probe <= p->n_groups; ++probe, ++key) {      // bounded: one probe per existing group
        const int32_t g = ix->find(key);
        if (g < 0) break;
        const DataFileRef* o = p->group_like[g];
        if (o->partition_spec_id == e->partition_spec_id && o->n_partition == e->n_partition &&
            tuple_equal(o->partition, e->partition, e->n_partition)) return g;
    }
    if (!ix->insert(key, p->n_groups)) return -1;
    p->group_like[p->n_groups] = e;
    return static_cast<int64_t>(p->n_groups++);
}

bool compact_candidates(TableHandle* th, CompactPlan* p, int64_t target) noexcept {
    assert(th != nullptr && p != nullptr && target > 0);
    Arena* a = th->arena;
    p->cand = a->allocate_array<const char*>(p->n_data + 1u);
    p->cand_size = a->allocate_array<int64_t>(p->n_data + 1u);
    p->cand_group = a->allocate_array<uint32_t>(p->n_data + 1u);
    p->cand_ref = a->allocate_array<const DataFileRef*>(p->n_data + 1u);
    p->group_like = a->allocate_array<const DataFileRef*>(p->n_data + 1u);
    auto* live = a->allocate_array<const char*>(p->n_data + 1u);
    SwissTableGrowable ix{};
    if (p->cand == nullptr || p->cand_size == nullptr || p->cand_group == nullptr || p->cand_ref == nullptr ||
        p->group_like == nullptr || live == nullptr || !SwissTableGrowable::create(&ix, 16u, a)) return false;
    for (uint32_t i = 0; i < p->n_data; ++i) {                  // bounded by n_data
        live[i] = p->data[i].path;
        if (p->data[i].size >= target / 2) continue;
        const int64_t g = cand_group_of(th, p, &ix, p->data[i].ref);
        if (g < 0) return false;
        p->cand_group[p->n_cand] = static_cast<uint32_t>(g);
        p->cand_ref[p->n_cand] = p->data[i].ref;
        p->cand_size[p->n_cand] = p->data[i].size;
        p->cand[p->n_cand++] = p->data[i].path;
    }
    p->cand_dels = static_cast<uint8_t*>(a->allocate_zeroed(p->n_cand + 1u));
    return p->cand_dels != nullptr && rm_index_build(a, p->cand, p->n_cand, &p->cand_ix) &&
           rm_index_build(a, live, p->n_data, &p->live_ix);
}

// Read every position-delete file's targets and mark the candidates with deletes.
bool compact_deletes(TableHandle* th, Arena* ra, CompactPlan* p) noexcept {
    assert(th != nullptr && ra != nullptr && p != nullptr);
    p->targets = th->arena->allocate_array<DelTargets>(p->n_pos + 1u);
    if (p->targets == nullptr) return false;
    for (uint32_t d = 0; d < p->n_pos; ++d) {                   // bounded by n_pos
        const uint8_t* body = nullptr; uint64_t len = 0;
        if (os_get(th->os, store_key_for(th->root, p->pos[d].path), ra, &body, &len) != kOsOk) return false;
        DelTargets& dt = p->targets[d];
        dt = DelTargets{};
        if (!dml::position_delete_targets(ra, body, len, &dt.t, &dt.n)) return false;
        for (uint32_t k = 0; k < dt.n; ++k) {                   // bounded by n
            const int64_t c = rm_index_find(&p->cand_ix, dt.t[k]);
            if (c >= 0 && p->cand_dels[c] == 0) { p->cand_dels[c] = 1; ++p->n_cand_dels; }
        }
    }
    for (uint32_t d = 0; d < p->n_dvs; ++d) {                   // bounded by n_dvs: a DV names exactly one data file
        const int64_t c = rm_index_find(&p->cand_ix, p->dvs[d].path);
        if (c >= 0 && p->cand_dels[c] == 0) { p->cand_dels[c] = 1; ++p->n_cand_dels; }
    }
    return true;
}

// Keep only the candidates of groups worth rewriting (two or more files, or one with position deletes); returns the
// number of such groups.
bool compact_select(TableHandle* th, CompactPlan* p, uint32_t* n_worth) noexcept {
    assert(th != nullptr && p != nullptr && n_worth != nullptr);
    auto* files = static_cast<uint32_t*>(th->arena->allocate_zeroed((p->n_groups + 1u) * sizeof(uint32_t)));
    auto* dels = static_cast<uint8_t*>(th->arena->allocate_zeroed(p->n_groups + 1u));
    if (files == nullptr || dels == nullptr) return false;
    for (uint32_t i = 0; i < p->n_cand; ++i) { ++files[p->cand_group[i]]; dels[p->cand_group[i]] |= p->cand_dels[i]; }
    *n_worth = 0;
    for (uint32_t g = 0; g < p->n_groups; ++g) *n_worth += (files[g] >= 2u || dels[g] != 0u) ? 1u : 0u;   // bounded
    uint32_t w = 0;
    p->n_cand_dels = 0;
    for (uint32_t i = 0; i < p->n_cand; ++i) {                  // bounded by n_cand
        const uint32_t g = p->cand_group[i];
        if (files[g] < 2u && dels[g] == 0u) continue;
        p->cand[w] = p->cand[i]; p->cand_size[w] = p->cand_size[i]; p->cand_group[w] = g;
        p->cand_ref[w] = p->cand_ref[i]; p->cand_dels[w] = p->cand_dels[i];
        p->n_cand_dels += p->cand_dels[i];
        ++w;
    }
    p->n_cand = w;
    return rm_index_build(th->arena, p->cand, p->n_cand, &p->cand_ix);
}

// The position-delete files the rewrite fully applies: every target is rewritten (or no longer live), and one is.
bool compact_applied(TableHandle* th, CompactPlan* p) noexcept {
    assert(th != nullptr && p != nullptr);
    p->rm_dels = th->arena->allocate_array<const char*>(p->n_pos + 1u);
    if (p->rm_dels == nullptr) return false;
    for (uint32_t d = 0; d < p->n_pos; ++d) {                   // bounded by n_pos
        bool applied = true, reaches = false;
        for (uint32_t k = 0; k < p->targets[d].n; ++k) {        // bounded by n
            const char* t = p->targets[d].t[k];
            if (rm_index_find(&p->cand_ix, t) >= 0) { reaches = true; continue; }
            if (rm_index_find(&p->live_ix, t) >= 0) applied = false;   // still deletes rows of a file we keep
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
    bool                lineage;          // v3: outputs write _row_id / _last_updated_sequence_number
};

// The current schema plus the two row-lineage fields, for a v3 output file (arena copy; the table's schema is untouched).
const Schema* lineage_schema(TableHandle* th, const Schema* sch) noexcept {
    assert(th != nullptr && sch != nullptr);
    auto* f = th->arena->allocate_array<SchemaField>(sch->n_fields + 2u);
    auto* out = th->arena->allocate_array<Schema>(1);
    if (f == nullptr || out == nullptr) return nullptr;
    std::memcpy(f, sch->fields, sizeof(SchemaField) * sch->n_fields);
    const struct { int32_t id; const char* name; } extra[2] = {{2147483540, "_row_id"}, {2147483539, "_last_updated_sequence_number"}};
    for (uint32_t k = 0; k < 2; ++k) {                          // bounded: two reserved fields
        SchemaField& x = f[sch->n_fields + k];
        std::memset(&x, 0, sizeof(x));
        x.id = extra[k].id;
        x.required = false;
        std::strncpy(x.name, extra[k].name, sizeof(x.name) - 1u);
        std::strncpy(x.type, "long", sizeof(x.type) - 1u);
    }
    *out = *sch;
    out->fields = f;
    out->n_fields = sch->n_fields + 2u;
    out->cap_fields = out->n_fields;
    return out;
}

// The table columns of `b` followed by the row's lineage columns, as one batch for the writer.
BoltBatch* with_lineage(Arena* a, const BoltBatch* b, const BoltColumn* lineage) noexcept {
    assert(a != nullptr && b != nullptr && lineage != nullptr);
    auto* w = a->allocate_array<BoltBatch>(1);
    if (w == nullptr) return nullptr;
    BoltBatch::init_empty(w);
    w->arena = a;
    if (!BoltBatch::alloc_columns(w, a, b->num_cols + 2u)) return nullptr;
    const BoltColumn* src = b->columns[b->read_epoch];
    BoltColumn* dst = w->columns[w->read_epoch];
    for (uint32_t c = 0; c < b->num_cols; ++c) {                // bounded by num_cols
        dst[c] = src[c];
        if (!w->schema.add_field(b->schema.field(static_cast<int>(c)).name, src[c].type, true)) return nullptr;
    }
    dst[b->num_cols] = lineage[0];
    dst[b->num_cols + 1u] = lineage[1];
    if (!w->schema.add_field("_row_id", BoltType::Int64, true) ||
        !w->schema.add_field("_last_updated_sequence_number", BoltType::Int64, true)) return nullptr;
    w->num_cols = b->num_cols + 2u;
    w->num_rows = b->num_rows;
    return w;
}

bool compact_keep(void* c, const DataFileRef* e) noexcept {
    const auto* p = static_cast<const CompactPlan*>(c);
    assert(p != nullptr && e != nullptr);
    const int64_t k = rm_index_find(&p->cand_ix, e->file_path);
    return k >= 0 && p->cand_group[k] == p->cur_group;
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
    const DataFileRef* like = o->plan->group_like[o->plan->cur_group];   // the group's spec and tuple
    df.partition_spec_id = like->partition_spec_id;
    df.n_partition = like->n_partition;
    std::memcpy(df.partition, like->partition, sizeof(PartitionValue) * like->n_partition);
    df.snapshot_id = o->snap_id;
    if (!abs_location(th->root, o->rel, df.file_path, sizeof(df.file_path))) return false;
    df.stats.record_count = o->rows;
    df.stats.file_size_in_bytes = size;
    if (!compute_file_stats(th, o->batches, o->n_batches, &df.stats)) return false;
    o->n_batches = 0;
    o->rows = 0;
    return true;
}

bool compact_open(CompactOut* o) noexcept {
    assert(o != nullptr && !o->file_open);
    TableHandle* th = o->th;
    // G2ICE-270: FileWriteIO holds one staged receipt. The previous output is published (its remote copy is the data), so its
    // local source is released for this one to stage; compact_discard removes every published output if the commit fails.
    // The last output keeps its source until the commit resolves, exactly as a one-file compaction always did.
    if (th->file_io != nullptr && o->n_outs > 0) data_file_complete(th, o->rels[o->n_outs - 1u], true);
    const int n = std::snprintf(o->rel, sizeof(o->rel), "data/%lld-%016llx.parquet",
                                static_cast<long long>(o->snap_id), static_cast<unsigned long long>(unique_u64()));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(o->rel)) return false;
    const Schema* sch = metadata_current_schema(&th->meta);
    if (sch == nullptr) return false;
    if (o->lineage && (sch = lineage_schema(th, sch)) == nullptr) return false;
    ingest::parquet::ParquetWriteOpts po{};
    std::vector<ingest::parquet::ParquetWriteColumn> wide;
    if (!make_pq_opts(sch, nullptr, &po, &wide,th->arena)) return false;
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
    const auto* schema=metadata_current_schema(&o->th->meta);BoltBatch* prepared=nullptr;
    if(!schema || write_batch_conform(*schema,*rb->batch,o->th->arena,&prepared)!=JsonValueStatus::Ok)return false;
    const BoltBatch* out=prepared;
    if(o->lineage) {
        if(!rb->lineage || !(out=with_lineage(o->th->arena,prepared,rb->lineage)))return false;
        schema=lineage_schema(o->th,schema);
    }
    if(!schema || !physical_row_group(o->w,*schema,*out,o->th->arena))return false;
    if (!lake_grow(o->th->arena, nullptr, &o->batches, o->n_batches, &o->cap_batches, o->n_batches + 1u)) return false;
    o->batches[o->n_batches++] = prepared;
    o->rows += rb->batch->num_rows;
    return true;
}

void compact_discard(CompactOut* o) noexcept {
    assert(o != nullptr && o->th != nullptr);
    TableHandle* th = o->th;
    const bool was_open = o->file_open;                          // then every closed output's source was released (compact_open)
    if (o->file_open) { data_file_abandon(th, o->w); data_file_complete(th, o->rel, false); o->file_open = false; }
    for (uint32_t i = 0; i < o->n_outs; ++i) {                   // bounded by n_outs
        const bool released = th->file_io != nullptr && (was_open || i + 1u < o->n_outs);
        if (!released) { data_file_complete(th, o->rels[i], false); continue; }
        if (th->write_uncertain) continue;                       // an uncertain commit may reference it: keep it
        const int rc = os_delete(th->os, o->rels[i]);
        (void)rc;
    }
}

// Read and write the candidates' live rows, then commit the replace. `ra` holds every batch until the commit.
bool compact_rewrite(TableHandle* th, Arena* ra, CompactPlan* p, int64_t target) noexcept {
    assert(th != nullptr && ra != nullptr && p != nullptr);
    CompactOut o{};
    o.th = th;
    o.plan = p;
    o.snap_id = mint_snapshot_id(th);
    o.target = target;
    o.lineage = th->meta.format_version >= 3 && !lineage_test_drop();   // the drop is the negative control only
    bool ok = true;
    auto* worth = static_cast<uint8_t*>(th->arena->allocate_zeroed(p->n_groups + 1u));
    if (worth == nullptr) return false;
    for (uint32_t i = 0; i < p->n_cand; ++i) worth[p->cand_group[i]] = 1u;   // bounded by n_cand
    for (uint32_t g = 0; g < p->n_groups && ok; ++g) {          // one read per group: outputs never mix partitions
        if (worth[g] == 0u) continue;
        p->cur_group = g;
        o.cur_path = nullptr;
        o.in_bytes = 0;
        ok = dml_read(th, ra, nullptr, compact_keep, p, compact_visit, &o, o.lineage) && compact_close(&o);
    }
    if (!ok) {
        if (th->last_error == CommitError::kNone) th->last_error = CommitError::kFailed;
        compact_discard(&o);
        return false;
    }
    ChangeSet cs{};
    cs.op = SnapshotOp::kReplace;
    cs.merge_manifests = 1u;
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
    if (th->meta.current_snapshot_id <= 0) { th->last_error = CommitError::kNone; return true; }
    CompactPlan p{};
    if (!compact_collect(th, &p) || !compact_candidates(th, &p, target)) return false;
    Arena* ra = scratch_arena(th);
    if (ra == nullptr) return false;
    uint32_t n_worth = 0;
    bool ok = compact_deletes(th, ra, &p) && compact_select(th, &p, &n_worth) && compact_applied(th, &p);
    const bool worth = n_worth != 0u;
    if (ok && !worth) { ra->~Arena(); th->last_error = CommitError::kNone; return true; }
    ok = ok && compact_rewrite(th, ra, &p, target);
    ra->~Arena();
    return ok;
}

}  // namespace

bool table_rewrite(TableHandle* th, const OptimizeOptions* opts) noexcept {
    assert(th != nullptr);
    OptimizeOptions o;
    optimize_options_init(&o);
    if (opts != nullptr) o = *opts;
    if (!o.run_bin_pack) { th->last_error = CommitError::kNone; return write_ready(th); }   // no strategy asked for
    if (o.target_file_size_mb <= 0) { th->last_error = CommitError::kFailed; return false; }
    return table_compact(th, static_cast<int64_t>(o.target_file_size_mb) << 20);
}

bool table_compact(TableHandle* th, int64_t target_file_size_bytes) noexcept {
    assert(th != nullptr);
    const uint8_t was = th->v1_maintenance;
    th->v1_maintenance = 1u;                                     // a v1 table is compacted as v1 (write_ready)
    const bool ok = write_ready(th) && compact_table(th, target_file_size_bytes);
    th->v1_maintenance = was;
    return ok;
}
