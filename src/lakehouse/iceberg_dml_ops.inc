// bolt/lakehouse/iceberg_dml_ops.inc — table_delete / table_overwrite (G2ICE-280). Included inside
// iceberg_writer.cpp at namespace scope after iceberg_dml.inc; not a standalone TU. Semantics: iceberg_dml.inc.

namespace {

struct DmlOutputs {
    DataFileRef del;
    DataFileRef dat;
    char        del_rel[128];
    char        dat_rel[128];
    bool        has_del;
    bool        has_dat;
};

void dml_complete(TableHandle* th, const DmlOutputs* o, bool committed) noexcept {
    assert(th != nullptr && o != nullptr);
    if (o->has_del) data_file_complete(th, o->del_rel, committed);
    if (o->has_dat) data_file_complete(th, o->dat_rel, committed);
}

// Write the position-delete file and the new data file this change needs.
bool dml_write_outputs(TableHandle* th, int64_t snap_id, const char* const* del_paths, const int64_t* del_pos,
                       uint32_t n_del, const BoltBatch* add, DmlOutputs* o) noexcept {
    assert(th != nullptr && o != nullptr);
    assert(snap_id > 0);
    *o = DmlOutputs{};
    if (n_del != 0) {
        BoltBatch b{};
        if (!pos_delete_batch(th->arena, del_paths, del_pos, n_del, &b)) return false;
        if (!write_pos_delete_batch(th, snap_id, &b, o->del_rel, &o->del)) return false;
        o->has_del = true;
    }
    if (add != nullptr && add->num_rows > 0) {
        if (!dml_data_file(th, snap_id, add, o->dat_rel, &o->dat)) { dml_complete(th, o, false); return false; }
        o->has_dat = true;
    }
    return true;
}

// The predicate names a column of the current schema (checked even when the table has no rows to read).
bool pred_column_known(TableHandle* th, const Predicate* pred) noexcept {
    assert(th != nullptr && pred != nullptr);
    const Schema* sch = metadata_current_schema(&th->meta);
    for (uint32_t i = 0; sch != nullptr && i < sch->n_fields; ++i)   // bounded by n_fields
        if (std::strncmp(sch->fields[i].name, pred->column, sizeof(pred->column)) == 0) return true;
    std::fprintf(stderr, "[iceberg_writer] predicate column '%.*s' is not a column of the table; nothing was written\n",
                 static_cast<int>(sizeof(pred->column)), pred->column);
    return false;
}

bool dml_change(TableHandle* th, const Predicate* pred, const BoltBatch* add, SnapshotOp op) noexcept {
    assert(th != nullptr && th->arena != nullptr);
    th->last_error = CommitError::kFailed;
    if (pred == nullptr) {
        std::fprintf(stderr, "[iceberg_writer] a row delete needs a predicate; nothing was written\n");
        return false;
    }
    if (!pred_column_known(th, pred)) return false;
    if (!handle_is_current(th)) return false;
    const PartitionSpec* spec = metadata_spec(&th->meta, th->meta.current_spec_id);
    if (spec != nullptr && spec->n_fields != 0) return refuse_named(th, "a delete or overwrite of a partitioned table");
    DmlPlan plan{th, nullptr, 0, 0};
    Arena* ra = scratch_arena(th);
    if (ra == nullptr) return false;
    bool ok = th->meta.current_snapshot_id <= 0 || dml_read(th, ra, pred, nullptr, nullptr, dml_visit, &plan);
    const char** rm = nullptr; const char** del_paths = nullptr; int64_t* del_pos = nullptr;
    uint32_t n_rm = 0, n_del = 0;
    ok = ok && dml_split(th, &plan, &rm, &n_rm, &del_paths, &del_pos, &n_del);
    ra->~Arena();
    if (!ok) return false;
    if ((add == nullptr || add->num_rows == 0) && n_rm == 0 && n_del == 0) { th->last_error = CommitError::kNone; return true; }
    if (th->file_io != nullptr && n_del != 0 && add != nullptr && add->num_rows > 0)
        return refuse_named(th, "an overwrite that also writes position deletes through FileWriteIO (one staged file per commit)");
    const int64_t snap_id = mint_snapshot_id(th);
    DmlOutputs o{};
    if (!dml_write_outputs(th, snap_id, del_paths, del_pos, n_del, add, &o)) return false;
    ChangeSet cs{};
    cs.op = op;
    cs.snap_id = snap_id;
    cs.rm_data = rm;
    cs.n_rm_data = n_rm;
    cs.add_dels = o.has_del ? &o.del : nullptr;
    cs.n_add_dels = o.has_del ? 1u : 0u;
    cs.add_data = o.has_dat ? &o.dat : nullptr;
    cs.n_add_data = o.has_dat ? 1u : 0u;
    const bool committed = commit_change(th, &cs);
    dml_complete(th, &o, committed);
    return committed;
}

}  // namespace

bool table_overwrite(TableHandle* th, const Predicate* pred, const BoltBatch* batch) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    if (batch == nullptr) { th->last_error = CommitError::kFailed; return false; }
    return dml_change(th, pred, batch, SnapshotOp::kOverwrite);
}

bool table_delete(TableHandle* th, const Predicate* pred) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    return dml_change(th, pred, nullptr, SnapshotOp::kDelete);
}
