// bolt/lakehouse/iceberg_change_commit.inc — commit_change (G2ICE-280). Included inside iceberg_writer.cpp's
// anonymous namespace after iceberg_change.inc and iceberg_summary.inc; not a standalone TU.

// Record the partition a changed file belongs to (spec: summary "changed-partition-count").
bool note_partition(TableHandle* th, ChangeState* st, const DataFileRef& d) noexcept {
    assert(th != nullptr && st != nullptr);
    assert(d.n_partition <= kIcebergMaxPartitionValues);
    if (!lake_grow(th->arena, nullptr, &st->part_keys, st->n_part_keys, &st->cap_part_keys, st->n_part_keys + 1u))
        return false;
    st->part_keys[st->n_part_keys++] = tuple_hash(d.partition, d.n_partition, d.partition_spec_id);
    return true;
}

int64_t distinct_partitions(ChangeState* st) noexcept {
    assert(st != nullptr);
    uint64_t* k = st->part_keys;
    const uint32_t n = st->n_part_keys;
    for (uint32_t gap = n / 2u; gap > 0u; gap /= 2u)                      // shell sort: bounded, no recursion
        for (uint32_t i = gap; i < n; ++i) {
            const uint64_t x = k[i];
            uint32_t j = i;
            while (j >= gap && k[j - gap] > x) { k[j] = k[j - gap]; j -= gap; }
            k[j] = x;
        }
    int64_t d = 0;
    for (uint32_t i = 0; i < n; ++i) d += (i == 0u || k[i] != k[i - 1u]) ? 1 : 0;   // bounded by n
    assert(d <= static_cast<int64_t>(n));
    return d;
}

// Carry `mle` forward, or rewrite it when it holds a file the change removes.
bool filter_manifest(TableHandle* th, ChangeState* st, const ManifestListEntry& mle,
                     ManifestListEntry* out) noexcept {
    assert(th != nullptr && st != nullptr && out != nullptr);
    const bool del = mle.content == ManifestContent::kDeleteManifest;
    RmIndex* rm = del ? &st->rm_dels : &st->rm_data;
    *out = mle;
    if (rm->n == 0) return true;
    DataFileRef* e = nullptr; uint32_t n = 0;
    if (!read_manifest_entries(th, mle, &e, &n)) { th->last_error = CommitError::kFailed; return false; }
    bool hit = false;
    for (uint32_t i = 0; i < n && !hit; ++i)                    // bounded by n
        hit = e[i].status != ManifestStatus::kDeleted && rm_index_find(rm, e[i].file_path) >= 0;
    if (!hit) return true;
    uint32_t w = 0;
    for (uint32_t i = 0; i < n; ++i) {                          // bounded by n
        if (e[i].status == ManifestStatus::kDeleted) continue;
        if (del && (e[i].content == FileContent::kEqualityDeletes || e[i].is_dv || e[i].n_equality_ids != 0))
            return refuse_named(th, "rewriting a delete manifest that holds equality deletes or deletion vectors");
        const int64_t k = rm_index_find(rm, e[i].file_path);
        if (k >= 0) {
            if (rm->found[k] != 0) { th->last_error = CommitError::kFailed; return false; }   // listed live twice
            rm->found[k] = 1;
            count_file(&st->sum, e[i], false);
            if (!note_partition(th, st, e[i])) return false;
        }
        carry_entry(&e[i], mle, k >= 0, st->cs->snap_id);
        e[w++] = e[i];
    }
    char tag[24];
    std::snprintf(tag, sizeof(tag), "-r%u", st->n_rewritten++);
    return write_manifest_entry(th, st, e, w, mle.partition_spec_id, tag, out);
}

bool all_found(TableHandle* th, const RmIndex* r, const char* what) noexcept {
    assert(th != nullptr && r != nullptr);
    for (uint32_t i = 0; i < r->n; ++i) {                       // bounded by n
        if (r->found[i] != 0) continue;
        th->last_error = CommitError::kConflict;
        std::fprintf(stderr, "[iceberg_writer] %s %s is no longer live in the table; nothing was committed\n",
                     what, r->paths[i]);
        return false;
    }
    return true;
}

// The parent's manifest list (empty for the first snapshot). Fails rather than drop a manifest it cannot read.
bool parent_list(TableHandle* th, int64_t parent, ManifestListEntry** ml, uint32_t* n) noexcept {
    assert(th != nullptr && ml != nullptr && n != nullptr);
    *ml = nullptr;
    *n = 0;
    if (parent <= 0) return true;
    const Snapshot* ps = nullptr;
    for (uint32_t i = 0; i < th->meta.n_snapshots; ++i)         // bounded
        if (th->meta.snapshots[i].snapshot_id == parent) { ps = &th->meta.snapshots[i]; break; }
    if (ps == nullptr) return false;
    if (ps->manifest_list[0] == '\0') return true;
    const uint8_t* pb = nullptr; uint64_t pl = 0;
    if (os_get(th->os, store_key_for(th->root, ps->manifest_list), th->arena, &pb, &pl) != kOsOk) return false;
    return manifest_list_parse_avro_grow(pb, pl, th->arena, 1u, ml, n);
}

// Write the manifest list, record the snapshot and its summary, publish metadata.json.
bool publish_change(TableHandle* th, ChangeState* st, const ManifestListEntry* ml, uint32_t n) noexcept {
    assert(th != nullptr && st != nullptr);
    const int64_t snap_id = st->cs->snap_id;
    const int64_t parent = th->meta.current_snapshot_id;
    const uint8_t* mlb = nullptr; uint64_t mll = 0;
    if (!manifest_list_write_avro(ml, n, snap_id, st->seq, th->arena, &mlb, &mll)) return false;
    char mlrel[128];
    std::snprintf(mlrel, sizeof(mlrel), "metadata/snap-%lld.avro", static_cast<long long>(snap_id));
    char mlabs[kIcebergMaxManifestPath];
    if (!abs_location(th->root, mlrel, mlabs, sizeof(mlabs))) return false;
    if (!commit_object(th, mlrel, mlb, mll, CommitObjectKind::kManifestList, snap_id)) return false;
    if (!track_key(th, st, mlrel)) return false;
    Snapshot s{};
    s.snapshot_id = snap_id;
    s.parent_snapshot_id = parent;
    s.timestamp_ms = now_ms();
    s.sequence_number = ++th->meta.last_sequence_number;
    assert(s.sequence_number == st->seq);
    s.op = st->cs->op;
    s.schema_id = th->meta.current_schema_id;   // G2ICE-50: the schema the snapshot is committed under
    std::strncpy(s.manifest_list, mlabs, sizeof(s.manifest_list) - 1u);
    const uint32_t n_log = th->meta.n_snapshot_log, n_snap = th->meta.n_snapshots, n_sums = th->n_sums;
    bool pushed = metadata_push_snapshot(&th->meta, &s) && metadata_snapshot_log_push(&th->meta, s.timestamp_ms, snap_id) &&
                  lake_grow(th->arena, nullptr, &th->sums, th->n_sums, &th->cap_sums, th->n_sums + 1u);
    if (pushed) {
        th->sums[th->n_sums].snapshot_id = snap_id;
        th->sums[th->n_sums].s = st->sum;
        ++th->n_sums;
        th->meta.current_snapshot_id = snap_id;
        if (persist_metadata(th)) { apply_retention_policy(th); return true; }
    } else {
        th->last_error = CommitError::kFailed;
    }
    th->meta.n_snapshot_log = n_log;
    th->meta.n_snapshots = n_snap;
    th->n_sums = n_sums;
    --th->meta.last_sequence_number;
    th->meta.current_snapshot_id = parent;
    return false;
}

// One manifest per spec id among `files` (a manifest is homogeneous by spec): new files are written under the
// default spec, but a position delete carries its data file's spec (G2ICE-292).
bool add_by_spec(TableHandle* th, ChangeState* st, const DataFileRef* files, uint32_t nf, ManifestListEntry* ml,
                 uint32_t* n) noexcept {
    assert(th != nullptr && st != nullptr && ml != nullptr && n != nullptr);
    assert(files != nullptr || nf == 0u);
    if (nf == 0u) return true;
    auto* done = static_cast<uint8_t*>(th->arena->allocate_zeroed(nf));
    auto* grp = th->arena->allocate_array<DataFileRef>(nf);
    if (done == nullptr || grp == nullptr) return false;
    for (uint32_t i = 0; i < nf; ++i) {                                   // bounded by nf
        if (done[i] != 0u) continue;
        const int32_t spec = files[i].partition_spec_id;
        uint32_t g = 0;
        for (uint32_t j = i; j < nf; ++j)                                 // bounded by nf
            if (done[j] == 0u && files[j].partition_spec_id == spec) { done[j] = 1u; grp[g++] = files[j]; }
        char tag[24];
        const uint32_t k = st->n_manifests_added++;
        tag[0] = '\0';
        if (k != 0u) std::snprintf(tag, sizeof(tag), "-a%u", k);
        if (!write_manifest_entry(th, st, grp, g, spec, tag, &ml[(*n)++])) return false;
    }
    return true;
}

bool change_add(TableHandle* th, ChangeState* st, ManifestListEntry* ml, uint32_t* n) noexcept {
    assert(th != nullptr && st != nullptr && ml != nullptr && n != nullptr);
    const ChangeSet* cs = st->cs;
    for (uint32_t i = 0; i < cs->n_add_data; ++i) {                       // bounded
        count_file(&st->sum, cs->add_data[i], true);
        if (!note_partition(th, st, cs->add_data[i])) return false;
    }
    for (uint32_t i = 0; i < cs->n_add_dels; ++i) {                       // bounded
        count_file(&st->sum, cs->add_dels[i], true);
        if (!note_partition(th, st, cs->add_dels[i])) return false;
    }
    return add_by_spec(th, st, cs->add_data, cs->n_add_data, ml, n) &&
           add_by_spec(th, st, cs->add_dels, cs->n_add_dels, ml, n);
}

bool change_prepare(TableHandle* th, const ChangeSet* cs, ChangeState* st) noexcept {
    assert(th != nullptr && cs != nullptr && st != nullptr);
    *st = ChangeState{};
    st->cs = cs;
    st->seq = th->meta.last_sequence_number + 1;
    const Schema* sch = metadata_current_schema(&th->meta);
    const uint8_t* sj = nullptr; uint64_t sl = 0;
    if (sch == nullptr || !emit_schema_json(th->arena, sch, &sj, &sl) || sl == 0 || sl > UINT32_MAX) return false;
    st->schema_json = sj;
    st->schema_len = static_cast<uint32_t>(sl);
    return rm_index_build(th->arena, cs->rm_data, cs->n_rm_data, &st->rm_data) &&
           rm_index_build(th->arena, cs->rm_dels, cs->n_rm_dels, &st->rm_dels);
}

// The one commit path. Nothing it wrote survives a refused commit (unless uncertain: then it is retained).
bool commit_change(TableHandle* th, const ChangeSet* cs) noexcept {
    assert(th != nullptr && cs != nullptr && cs->snap_id > 0);
    assert(cs->add_data != nullptr || cs->n_add_data == 0);
    if (!handle_is_current(th)) return false;
    ChangeState st{};
    th->last_error = CommitError::kFailed;
    if (!change_prepare(th, cs, &st)) return false;
    const int64_t parent = th->meta.current_snapshot_id;
    ManifestListEntry* pml = nullptr; uint32_t pn = 0;
    if (!parent_list(th, parent, &pml, &pn)) return false;
    // One manifest per (content, spec) of the added files, at most one each per file.
    auto* ml = th->arena->allocate_array<ManifestListEntry>(pn + cs->n_add_data + cs->n_add_dels + 2u);
    if (ml == nullptr) return false;
    uint32_t n = 0;
    bool ok = true;
    for (uint32_t i = 0; i < pn && ok; ++i) ok = filter_manifest(th, &st, pml[i], &ml[n++]);   // bounded by pn
    ok = ok && all_found(th, &st.rm_data, "data file") && all_found(th, &st.rm_dels, "delete file");
    ok = ok && change_add(th, &st, ml, &n);
    st.sum.changed_partitions = ok ? distinct_partitions(&st) : 0;
    ok = ok && summary_totals(th, parent, ml, n, &st.sum);
    ok = ok && publish_change(th, &st, ml, n);
    if (!ok) {
        if (th->last_error == CommitError::kNone) th->last_error = CommitError::kFailed;
        drop_written(th, &st);
    }
    return ok;
}
