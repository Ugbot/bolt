// bolt/lakehouse/iceberg_delete_dv_ops.inc — table_delete_positions on a format-version 3 table (G2ICE-283). Included
// inside iceberg_writer.cpp after iceberg_dml_ops.inc; not a standalone TU.
//
// v3 deletes rows with deletion vectors, never position-delete files. The named files are read (their live rows, every
// delete applied) so each DV holds the file's earlier deletes too; a file whose every live row is named is removed instead.

namespace {

struct PosPick {
    DmlPlan*            plan;
    const char* const*  paths;   // sorted by (path, pos)
    const int64_t*      pos;
    uint32_t            n;
    RmIndex             files;   // the distinct named files
};

bool pos_pick_keep(void* c, const DataFileRef* e) noexcept {
    auto* p = static_cast<PosPick*>(c);
    assert(p != nullptr && e != nullptr);
    return rm_index_find(&p->files, e->file_path) >= 0;
}

// Whether (path, pos) is named: a binary search over the sorted request.
bool pos_named(const PosPick* p, const char* path, int64_t pos) noexcept {
    assert(p != nullptr && path != nullptr);
    uint32_t lo = 0, hi = p->n;
    while (lo < hi) {                                                    // bounded: log2(n) steps
        const uint32_t mid = lo + (hi - lo) / 2u;
        const int c = std::strcmp(p->paths[mid], path);
        if (c < 0 || (c == 0 && p->pos[mid] < pos)) lo = mid + 1u; else hi = mid;
    }
    return lo < p->n && std::strcmp(p->paths[lo], path) == 0 && p->pos[lo] == pos;
}

bool pos_pick_visit(void* c, const dml::RowBatch* rb) noexcept {
    auto* p = static_cast<PosPick*>(c);
    assert(p != nullptr && rb != nullptr && rb->file_path != nullptr);
    const int64_t nr = rb->batch->num_rows;
    auto* match = p->plan->th->arena->allocate_array<uint32_t>(static_cast<size_t>(nr) + 1u);
    if (match == nullptr) return false;
    uint32_t nm = 0;
    for (int64_t r = 0; r < nr; ++r)                                     // bounded by the batch
        if (pos_named(p, rb->file_path, rb->pos[r])) match[nm++] = static_cast<uint32_t>(r);
    dml::RowBatch picked = *rb;
    picked.match = match;
    picked.n_match = nm;
    return dml_visit(p->plan, &picked);
}

bool delete_positions_v3(TableHandle* th, const char* const* paths, const int64_t* pos, uint32_t n) noexcept {
    assert(th != nullptr && paths != nullptr && pos != nullptr);
    DmlPlan plan{th, nullptr, 0, 0, false, true};
    PosPick pick{&plan, paths, pos, n, RmIndex{}};
    auto* distinct = th->arena->allocate_array<const char*>(n + 1u);
    if (distinct == nullptr) return false;
    uint32_t nd = 0;
    for (uint32_t i = 0; i < n; ++i) if (i == 0u || std::strcmp(paths[i], paths[i - 1u]) != 0) distinct[nd++] = paths[i];   // bounded by n
    if (!rm_index_build(th->arena, distinct, nd, &pick.files)) return false;
    Arena* ra = scratch_arena(th);
    if (ra == nullptr) return false;
    bool ok = th->meta.current_snapshot_id > 0 && dml_read(th, ra, nullptr, pos_pick_keep, &pick, pos_pick_visit, &pick);
    const char** rm = nullptr; const char** dp = nullptr; int64_t* dpos = nullptr;
    uint32_t n_rm = 0, n_del = 0;
    ok = ok && dml_split(th, &plan, &rm, &n_rm, &dp, &dpos, &n_del);
    LiveData live{};
    ok = ok && live_data_build(th, &live);
    if (ok && n_rm == 0u && n_del == 0u) { ra->~Arena(); th->last_error = CommitError::kNone; return true; }   // nothing live named
    const int64_t snap_id = ok ? mint_snapshot_id(th) : 0;
    DmlOutputs o{};
    ok = ok && dml_write_outputs(th, snap_id, dp, dpos, n_del, nullptr, &live, &plan, &o);
    bool committed = false;
    if (ok) {
        ChangeSet cs{};
        cs.op = SnapshotOp::kDelete;
        cs.snap_id = snap_id;
        cs.rm_data = rm;
        cs.n_rm_data = n_rm;
        cs.add_dels = o.dels;
        cs.n_add_dels = o.n_dels;
        cs.rm_dvs = o.replaced_dvs;
        cs.n_rm_dvs = o.n_replaced_dvs;
        committed = commit_change(th, &cs);
        dml_complete(th, &o, committed);
    }
    ra->~Arena();
    return committed;
}

}  // namespace
