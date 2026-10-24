// bolt/lakehouse/iceberg_change.inc — one snapshot that adds and removes files (G2ICE-280). Included inside
// iceberg_writer.cpp's anonymous namespace after write_data_file / mint_snapshot_id; not a standalone TU.
//
// Every commit goes through commit_change: append (adds data files), the position-delete commit (adds a delete
// file), table_delete / table_overwrite (remove data files, add position deletes, add data) and compaction (replace
// data and delete files). The new snapshot's manifest list is the parent's, with
//   * every manifest that holds a removed file REWRITTEN: removed entries become DELETED (snapshot id = this
//     snapshot), every other live entry EXISTING with the sequence numbers it was first committed with, entries
//     already DELETED dropped (spec: "Manifests", "Sequence Number Inheritance");
//   * every other manifest carried forward unchanged;
//   * one new manifest for the added data files and one for the added delete files.
// A file to remove that is not live in the parent snapshot fails the commit with kConflict: the commit was planned
// against a table state that no longer holds (the version CAS catches the same race one step later).

struct ChangeSet {
    SnapshotOp         op;
    uint8_t            _pad[7];
    int64_t            snap_id;
    const DataFileRef* add_data;
    uint32_t           n_add_data;
    uint32_t           n_add_dels;
    const DataFileRef* add_dels;
    const char* const* rm_data;     // locations exactly as the parent's manifests record them
    const char* const* rm_dels;
    uint32_t           n_rm_data;
    uint32_t           n_rm_dels;
};

// Sorted (hash, index) pairs over one removal list, with a found flag per path.
struct RmIndex {
    const char* const* paths;
    uint64_t*          hash;
    uint32_t*          idx;
    uint8_t*           found;
    uint32_t           n;
    uint32_t           _pad;
};

uint64_t rm_path_hash(const char* p) noexcept {
    assert(p != nullptr);
    uint64_t h = 0xcbf29ce484222325ull;
    for (const char* c = p; *c != '\0'; ++c) { h ^= static_cast<uint8_t>(*c); h *= 0x100000001b3ull; }   // bounded by NUL
    return h;
}

bool rm_index_build(Arena* a, const char* const* paths, uint32_t n, RmIndex* out) noexcept {
    assert(a != nullptr && out != nullptr);
    assert(paths != nullptr || n == 0);
    *out = RmIndex{};
    out->paths = paths;
    out->n = n;
    if (n == 0) return true;
    out->hash = a->allocate_array<uint64_t>(n);
    out->idx = a->allocate_array<uint32_t>(n);
    out->found = static_cast<uint8_t*>(a->allocate_zeroed(n));
    if (out->hash == nullptr || out->idx == nullptr || out->found == nullptr) return false;
    for (uint32_t i = 0; i < n; ++i) { out->idx[i] = i; out->hash[i] = rm_path_hash(paths[i]); }
    for (uint32_t gap = n / 2u; gap > 0u; gap /= 2u) {           // shell sort: bounded, no recursion
        for (uint32_t i = gap; i < n; ++i) {
            const uint64_t h = out->hash[i]; const uint32_t x = out->idx[i];
            uint32_t j = i;
            while (j >= gap && out->hash[j - gap] > h) { out->hash[j] = out->hash[j - gap]; out->idx[j] = out->idx[j - gap]; j -= gap; }
            out->hash[j] = h; out->idx[j] = x;
        }
    }
    return true;
}

// Index into the removal list of `path`, or -1.
int64_t rm_index_find(const RmIndex* r, const char* path) noexcept {
    assert(r != nullptr && path != nullptr);
    if (r->n == 0) return -1;
    const uint64_t h = rm_path_hash(path);
    uint32_t lo = 0, hi = r->n;
    while (lo < hi) { const uint32_t mid = lo + (hi - lo) / 2u; if (r->hash[mid] < h) lo = mid + 1u; else hi = mid; }
    for (uint32_t i = lo; i < r->n && r->hash[i] == h; ++i)      // bounded by the run
        if (std::strcmp(r->paths[r->idx[i]], path) == 0) return static_cast<int64_t>(r->idx[i]);
    return -1;
}

// Per-commit bookkeeping: the objects written (removed again on a refused commit) and the summary deltas.
struct ChangeState {
    const ChangeSet* cs;
    int64_t          seq;
    const uint8_t*   schema_json;
    uint32_t         schema_len;
    uint32_t         n_keys;
    char             (*keys)[128];
    uint32_t         cap_keys;
    uint32_t         n_rewritten;
    SnapSummary      sum;
    RmIndex          rm_data;
    RmIndex          rm_dels;
    uint64_t*        part_keys;     // (spec, tuple) hash of every file added or removed: changed-partition-count
    uint32_t         n_part_keys, cap_part_keys;
    uint32_t         n_manifests_added;
};

bool track_key(TableHandle* th, ChangeState* st, const char* key) noexcept {
    assert(th != nullptr && st != nullptr && key != nullptr);
    if (std::strlen(key) >= 128u) return false;
    if (!lake_grow(th->arena, nullptr, &st->keys, st->n_keys, &st->cap_keys, st->n_keys + 1u)) return false;
    std::strncpy(st->keys[st->n_keys], key, 127u);
    st->keys[st->n_keys][127] = '\0';
    ++st->n_keys;
    return true;
}

void drop_written(TableHandle* th, ChangeState* st) noexcept {
    assert(th != nullptr && st != nullptr);
    if (th->write_uncertain) return;                     // a candidate may be visible: keep its objects
    for (uint32_t i = 0; i < st->n_keys; ++i) (void)os_delete(th->os, st->keys[i]);   // bounded by n_keys
}

void count_file(SnapSummary* s, const DataFileRef& d, bool added) noexcept {
    assert(s != nullptr);
    const int64_t rc = d.stats.record_count;
    s->v[added ? kSumAddedFilesSize : kSumRemovedFilesSize] += d.stats.file_size_in_bytes;
    if (d.content == FileContent::kData) {
        s->v[added ? kSumAddedDataFiles : kSumDeletedDataFiles] += 1;
        s->v[added ? kSumAddedRecords : kSumDeletedRecords] += rc;
        return;
    }
    s->v[added ? kSumAddedDeleteFiles : kSumRemovedDeleteFiles] += 1;
    if (d.content == FileContent::kEqualityDeletes) {
        s->v[added ? kSumAddedEqDeleteFiles : kSumRemovedEqDeleteFiles] += 1;
        s->v[added ? kSumAddedEqDeletes : kSumRemovedEqDeletes] += rc;
    } else {
        s->v[added ? kSumAddedPosDeleteFiles : kSumRemovedPosDeleteFiles] += 1;
        s->v[added ? kSumAddedPosDeletes : kSumRemovedPosDeletes] += rc;
    }
}

// An entry carried into a rewritten manifest: its inherited fields made explicit (spec: "When writing an existing
// file to a new manifest or marking an existing file as deleted, the data and file sequence numbers must be non-null
// and set to the original values").
void carry_entry(DataFileRef* e, const ManifestListEntry& mle, bool removed, int64_t snap_id) noexcept {
    assert(e != nullptr);
    assert(snap_id > 0);
    if (!e->has_sequence_number) { e->sequence_number = mle.sequence_number; e->has_sequence_number = true; }
    if (!e->has_file_sequence_number) {
        e->file_sequence_number = e->status == ManifestStatus::kAdded ? mle.sequence_number : e->sequence_number;
        e->has_file_sequence_number = true;
    }
    if (e->snapshot_id == 0) e->snapshot_id = mle.added_snapshot_id;
    e->status = removed ? ManifestStatus::kDeleted : ManifestStatus::kExisting;
    if (removed) e->snapshot_id = snap_id;
}

bool refuse_named(TableHandle* th, const char* what) noexcept;
bool spec_types_for(const TableHandle* th, const PartitionSpec* spec, IcebergTypeInfo* src,
                    IcebergTypeInfo* res) noexcept;

// Write one manifest of `files` (added, existing or deleted, per entry status) under spec `spec_id` and its list
// entry, with the entries' partition tuples and the list's partition summaries (G2ICE-292).
bool write_manifest_entry(TableHandle* th, ChangeState* st, const DataFileRef* files, uint32_t nf, int32_t spec_id,
                          const char* tag, ManifestListEntry* out) noexcept {
    assert(th != nullptr && st != nullptr && out != nullptr);
    assert(files != nullptr && nf > 0 && tag != nullptr);
    const PartitionSpec* spec = metadata_spec(&th->meta, spec_id);
    if (spec == nullptr) { th->last_error = CommitError::kFailed; return false; }
    IcebergTypeInfo src[kIcebergMaxFieldsPerSpec], res[kIcebergMaxFieldsPerSpec];
    if (spec->n_fields != 0u && !spec_types_for(th, spec, src, res))
        return refuse_named(th, "a manifest of a partition spec with an unknown transform, or one not defined for its source");
    const uint8_t* body = nullptr; uint64_t len = 0;
    if (!manifest_write_avro_typed(files, nf, st->cs->snap_id, st->seq, reinterpret_cast<const char*>(st->schema_json),
                                   st->schema_len, spec, res, th->arena, &body, &len)) { th->last_error = CommitError::kFailed; return false; }
    char rel[128];
    const int rn = std::snprintf(rel, sizeof(rel), "metadata/manifest-%lld%s.avro",
                                 static_cast<long long>(st->cs->snap_id), tag);
    if (rn <= 0 || static_cast<size_t>(rn) >= sizeof(rel)) return false;
    if (!commit_object(th, rel, body, len, CommitObjectKind::kManifest, st->cs->snap_id)) return false;
    if (!track_key(th, st, rel)) return false;
    ManifestListEntry m{};
    m.partition_spec_id = spec_id;
    m.content = files[0].content != FileContent::kData ? ManifestContent::kDeleteManifest : ManifestContent::kDataManifest;
    m.manifest_length = static_cast<int64_t>(len);
    m.added_snapshot_id = st->cs->snap_id;
    m.sequence_number = st->seq;
    m.min_sequence_number = st->seq;
    for (uint32_t i = 0; i < nf; ++i) {                     // bounded by nf
        const DataFileRef& d = files[i];
        const bool carried = d.status != ManifestStatus::kAdded;
        if (carried && d.status != ManifestStatus::kDeleted && d.sequence_number < m.min_sequence_number)
            m.min_sequence_number = d.sequence_number;
        int64_t* fc = d.status == ManifestStatus::kAdded ? &m.added_files_count
                    : d.status == ManifestStatus::kExisting ? &m.existing_files_count : &m.deleted_files_count;
        int64_t* rc = d.status == ManifestStatus::kAdded ? &m.added_rows_count
                    : d.status == ManifestStatus::kExisting ? &m.existing_rows_count : &m.deleted_rows_count;
        *fc += 1;
        *rc += d.stats.record_count;
    }
    if (!abs_location(th->root, rel, m.manifest_path, sizeof(m.manifest_path))) return false;
    manifest_partition_summaries(files, nf, spec, res, &m);
    *out = m;
    return true;
}

bool read_manifest_entries(TableHandle* th, const ManifestListEntry& mle, DataFileRef** out, uint32_t* n) noexcept {
    assert(th != nullptr && out != nullptr && n != nullptr);
    const uint8_t* body = nullptr; uint64_t len = 0;
    if (os_get(th->os, store_key_for(th->root, mle.manifest_path), th->arena, &body, &len) != kOsOk) return false;
    return manifest_parse_avro_grow(body, len, th->arena, mle.partition_spec_id, 16u, out, n);
}

bool refuse_named(TableHandle* th, const char* what) noexcept {
    assert(th != nullptr && what != nullptr);
    th->last_error = CommitError::kUnimplemented;
    std::fprintf(stderr, "[iceberg_writer] refused (kUnimplemented): %s; nothing was written\n", what);
    return false;
}
