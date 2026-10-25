// bolt/lakehouse/iceberg_dml_ops.inc — table_delete / table_overwrite (G2ICE-280). Included inside
// iceberg_writer.cpp at namespace scope after iceberg_dml.inc; not a standalone TU. Semantics: iceberg_dml.inc.

namespace {

struct DmlOutputs {
    DataFileRef* dels;
    char         (*del_rels)[128];
    uint32_t     n_dels;
    uint32_t     n_dats;
    DataFileRef* dats;
    char         (*dat_rels)[128];
    const char** replaced_dvs;      // v3: the data files whose DV the new ones replace
    uint32_t     n_replaced_dvs;
};

void dml_complete(TableHandle* th, const DmlOutputs* o, bool committed) noexcept {
    assert(th != nullptr && o != nullptr);
    for (uint32_t i = 0; i < o->n_dels; ++i) data_file_complete(th, o->del_rels[i], committed);   // bounded by n_dels
    for (uint32_t i = 0; i < o->n_dats; ++i) data_file_complete(th, o->dat_rels[i], committed);   // bounded by n_dats
}

// The added rows' data files: one, or one per tuple of the default spec when the table is partitioned (G2ICE-292).
bool dml_add_files(TableHandle* th, int64_t snap_id, const BoltBatch* add, DmlOutputs* o) noexcept {
    assert(th != nullptr && add != nullptr && o != nullptr);
    const PartitionSpec* spec = metadata_spec(&th->meta, th->meta.current_spec_id);
    const BoltBatch* one[1] = {add};
    if (spec != nullptr && spec->n_fields != 0u)
        return write_partitioned_files(th, spec, one, 1u, snap_id, &o->dats, &o->dat_rels, &o->n_dats);
    o->dats = th->arena->allocate_array<DataFileRef>(1);
    o->dat_rels = th->arena->allocate_array<char[128]>(1);
    if (o->dats == nullptr || o->dat_rels == nullptr) return false;
    if (!dml_data_file(th, snap_id, add, o->dat_rels[0], &o->dats[0])) return false;
    o->n_dats = 1;
    return true;
}

// Write the position-delete files and the new data files this change needs.
// `rows`: the plan whose matched rows go in each delete's `row` column (G2ICE-283); null writes (file_path, pos) only.
bool dml_write_outputs(TableHandle* th, int64_t snap_id, const char* const* del_paths, const int64_t* del_pos,
                       uint32_t n_del, const BoltBatch* add, const LiveData* live, const DmlPlan* rows,
                       DmlOutputs* o) noexcept {
    assert(th != nullptr && o != nullptr && live != nullptr);
    assert(snap_id > 0);
    *o = DmlOutputs{};
    bool ok = n_del == 0 ||
              (rows != nullptr && rows->dv ? write_dvs(th, snap_id, rows, live, &o->dels, &o->del_rels, &o->n_dels, &o->replaced_dvs)
               : rows != nullptr ? write_row_deletes(th, snap_id, rows, live, &o->dels, &o->del_rels, &o->n_dels)
                                 : write_grouped_deletes(th, snap_id, del_paths, del_pos, n_del, live, &o->dels,
                                                         &o->del_rels, &o->n_dels));
    if (rows != nullptr && rows->dv) o->n_replaced_dvs = o->n_dels;
    ok = ok && (add == nullptr || add->num_rows <= 0 || dml_add_files(th, snap_id, add, o));
    if (!ok) dml_complete(th, o, false);
    return ok;
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

// The read, split and commit of dml_change; `ra` holds the read's batches (the kept rows share its bytes) until it returns.
bool dml_change_in(TableHandle* th, const Predicate* pred, const BoltBatch* add, SnapshotOp op, DmlPlan* plan,
                   Arena* ra) noexcept {
    assert(th != nullptr && plan != nullptr && ra != nullptr);
    bool ok = th->meta.current_snapshot_id <= 0 || dml_read(th, ra, pred, nullptr, nullptr, dml_visit, plan);
    const char** rm = nullptr; const char** del_paths = nullptr; int64_t* del_pos = nullptr;
    uint32_t n_rm = 0, n_del = 0;
    ok = ok && dml_split(th, plan, &rm, &n_rm, &del_paths, &del_pos, &n_del);
    if (!ok) return false;
    if ((add == nullptr || add->num_rows == 0) && n_rm == 0 && n_del == 0) { th->last_error = CommitError::kNone; return true; }
    if (th->file_io != nullptr && n_del != 0 && add != nullptr && add->num_rows > 0)
        return refuse_named(th, "an overwrite that also writes position deletes through FileWriteIO (one staged file per commit)");
    LiveData live{};
    if (n_del != 0 && !live_data_build(th, &live)) return false;
    const int64_t snap_id = mint_snapshot_id(th);
    DmlOutputs o{};
    if (!dml_write_outputs(th, snap_id, del_paths, del_pos, n_del, add, &live, plan->with_rows || plan->dv ? plan : nullptr, &o))
        return false;
    ChangeSet cs{};
    cs.op = op;
    cs.snap_id = snap_id;
    cs.rm_data = rm;
    cs.n_rm_data = n_rm;
    cs.add_dels = o.dels;
    cs.n_add_dels = o.n_dels;
    cs.add_data = o.dats;
    cs.n_add_data = o.n_dats;
    cs.rm_dvs = o.replaced_dvs;
    cs.n_rm_dvs = o.n_replaced_dvs;
    const bool committed = commit_change(th, &cs);
    dml_complete(th, &o, committed);
    return committed;
}

bool dml_change(TableHandle* th, const Predicate* pred, const BoltBatch* add, SnapshotOp op,
                const DeleteOptions* opts) noexcept {
    assert(th != nullptr && th->arena != nullptr);
    th->last_error = CommitError::kFailed;
    if (pred == nullptr) {
        std::fprintf(stderr, "[iceberg_writer] a row delete needs a predicate; nothing was written\n");
        return false;
    }
    if (!pred_column_known(th, pred)) return false;
    if (!handle_is_current(th)) return false;
    DmlPlan plan{th, nullptr, 0, 0, opts != nullptr && opts->position_row, th->meta.format_version >= 3};
    if (plan.with_rows && plan.dv)
        return refuse_named(th, "a position delete `row` column on a format-version 3 table (its deletes are deletion vectors)");
    if (plan.with_rows && !row_schema_ok(th))
        return refuse_named(th, "a position delete `row` column for a schema with a nested column");
    Arena* ra = scratch_arena(th);
    if (ra == nullptr) return false;
    const bool ok = dml_change_in(th, pred, add, op, &plan, ra);
    ra->~Arena();
    return ok;
}

}  // namespace

bool table_overwrite(TableHandle* th, const Predicate* pred, const BoltBatch* batch) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    if (batch == nullptr) { th->last_error = CommitError::kFailed; return false; }
    return dml_change(th, pred, batch, SnapshotOp::kOverwrite, nullptr);
}

bool table_delete(TableHandle* th, const Predicate* pred) noexcept {
    assert(th != nullptr);
    if (!write_ready(th)) return false;
    return dml_change(th, pred, nullptr, SnapshotOp::kDelete, nullptr);
}
