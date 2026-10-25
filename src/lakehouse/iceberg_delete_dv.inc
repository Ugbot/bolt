// bolt/lakehouse/iceberg_delete_dv.inc — v3 deletes as deletion vectors (G2ICE-283). Included inside iceberg_writer.cpp
// after iceberg_delete_row.inc and before iceberg_dml_ops.inc; not a standalone TU.
//
// Spec "Deletion Vectors": a v3 writer deletes rows with a DV, at most one per data file, and merges the file's existing
// deletes into it. The DML read already applies every delete, so a file's deleted positions are exactly those the read did
// not return live, plus the rows the predicate matched. One Puffin file per commit holds every DV; each manifest entry names
// its blob (content_offset, content_size_in_bytes) and its data file (referenced_data_file), carries that file's spec and
// tuple, and the DV it replaces is removed (ChangeSet::rm_dvs).

namespace {

// with dv: row `pos` of the file was read live.
bool dml_mark_live(Arena* a, DmlFile* f, int64_t pos) noexcept {
    assert(a != nullptr && f != nullptr);
    if (pos < 0 || pos >= static_cast<int64_t>(UINT32_MAX)) return false;   // a DV holds 32-bit positions here
    const uint32_t byte = static_cast<uint32_t>(pos >> 3);
    if (byte >= f->n_live) {
        uint32_t cap = f->n_live;
        if (!lake_grow(a, nullptr, &f->live, f->n_live, &cap, byte + 1u)) return false;
        std::memset(f->live + f->n_live, 0, cap - f->n_live);
        f->n_live = cap;
    }
    f->live[byte] = static_cast<uint8_t>(f->live[byte] | (1u << (pos & 7)));
    return true;
}

bool dml_is_live(const DmlFile& f, int64_t pos) noexcept {
    const uint64_t byte = static_cast<uint64_t>(pos) >> 3;
    return byte < f.n_live && (f.live[byte] & (1u << (pos & 7))) != 0u;
}

// The file's deleted positions after this change, ascending: every position not read live, and the matched ones.
bool dv_positions(Arena* a, const DmlFile& f, int64_t records, uint32_t** out, uint64_t* n) noexcept {
    assert(a != nullptr && out != nullptr && n != nullptr);
    if (records <= 0 || records > static_cast<int64_t>(UINT32_MAX)) return false;
    auto* matched = static_cast<uint8_t*>(a->allocate_zeroed(static_cast<size_t>(records) / 8u + 1u));
    *out = a->allocate_array<uint32_t>(static_cast<size_t>(records));
    if (matched == nullptr || *out == nullptr) return false;
    for (uint32_t i = 0; i < f.n_pos; ++i)                                // bounded by the matched rows
        if (f.pos[i] >= 0 && f.pos[i] < records) matched[f.pos[i] >> 3] = static_cast<uint8_t>(matched[f.pos[i] >> 3] | (1u << (f.pos[i] & 7)));
    *n = 0;
    for (int64_t p = 0; p < records; ++p)                                 // bounded by the file's records
        if (!dml_is_live(f, p) || (matched[p >> 3] & (1u << (p & 7))) != 0u) (*out)[(*n)++] = static_cast<uint32_t>(p);
    return true;
}

// The DV entries of the partly deleted files (one Puffin), and the data files whose old DV they replace.
bool write_dvs(TableHandle* th, int64_t snap_id, const DmlPlan* p, const LiveData* live, DataFileRef** out, char (**rels)[128],
               uint32_t* n_out, const char*** replaced) noexcept {
    assert(th != nullptr && p != nullptr && live != nullptr);
    assert(out != nullptr && rels != nullptr && n_out != nullptr && replaced != nullptr);
    *n_out = 0;
    if (th->file_io != nullptr) return refuse_named(th, "deletion vectors through FileWriteIO (one staged Parquet file per commit)");
    const char** rm = nullptr;
    uint32_t n_rm = 0, n_part = 0;
    uint32_t* order = nullptr;
    uint64_t rows = 0;
    if (!dml_partition_files(th, p, &rm, &n_rm, &order, &n_part, &rows)) return false;
    auto* blobs = th->arena->allocate_array<DvBlobIn>(n_part + 1u);
    auto* placed = th->arena->allocate_array<DvBlobPlacement>(n_part + 1u);
    auto* likes = th->arena->allocate_array<const DataFileRef*>(n_part + 1u);
    *replaced = th->arena->allocate_array<const char*>(n_part + 1u);
    if (blobs == nullptr || placed == nullptr || likes == nullptr || *replaced == nullptr) return false;
    for (uint32_t k = 0; k < n_part; ++k) {                               // bounded by the partly deleted files
        const DmlFile& f = p->files[order[k]];
        likes[k] = live_data_find(live, f.path);
        uint32_t* pos = nullptr; uint64_t np = 0;
        if (likes[k] == nullptr || !dv_positions(th->arena, f, likes[k]->stats.record_count, &pos, &np))
            return refuse_named(th, "a deletion vector of a data file this read does not list live, or of 2^32 rows");
        blobs[k] = DvBlobIn{pos, np, likes[k]->file_path};
        (*replaced)[k] = likes[k]->file_path;
    }
    const uint8_t* puffin = nullptr; uint64_t len = 0;
    if (!puffin_write_deletion_vectors(blobs, n_part, "bolt", th->arena, &puffin, &len, placed)) return false;
    *rels = th->arena->allocate_array<char[128]>(n_part + 1u);   // every entry names the one Puffin (completed per entry)
    *out = th->arena->allocate_array<DataFileRef>(n_part + 1u);
    if (*rels == nullptr || *out == nullptr) return false;
    const int rn = std::snprintf((*rels)[0], 128u, "data/%lld-%016llx-dvs.puffin", static_cast<long long>(snap_id),
                                 static_cast<unsigned long long>(unique_u64()));
    if (rn <= 0 || rn >= 128 || os_put(th->os, (*rels)[0], puffin, len) != kOsOk) { th->last_error = CommitError::kFailed; return false; }
    for (uint32_t k = 1; k < n_part; ++k) std::memcpy((*rels)[k], (*rels)[0], 128u);   // bounded by n_part
    for (uint32_t k = 0; k < n_part; ++k) {                               // bounded by the partly deleted files
        DataFileRef& df = (*out)[k];
        df = DataFileRef{};
        df.status = ManifestStatus::kAdded;
        df.content = FileContent::kPositionDeletes;
        df.is_dv = true;
        df.content_offset = placed[k].offset;
        df.content_size = placed[k].length;
        df.partition_spec_id = likes[k]->partition_spec_id;
        df.n_partition = likes[k]->n_partition;
        std::memcpy(df.partition, likes[k]->partition, sizeof(PartitionValue) * df.n_partition);
        df.snapshot_id = snap_id;
        df.stats.record_count = static_cast<int64_t>(blobs[k].n);
        df.stats.file_size_in_bytes = static_cast<int64_t>(len);
        std::strncpy(df.referenced_data_file, likes[k]->file_path, sizeof(df.referenced_data_file) - 1u);
        if (!abs_location(th->root, (*rels)[0], df.file_path, sizeof(df.file_path))) { data_file_complete(th, (*rels)[0], false); return false; }
    }
    *n_out = n_part;
    return true;
}

}  // namespace
