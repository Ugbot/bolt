// bolt/lakehouse/iceberg_commit.inc — the Iceberg writer's commit protocol
// (G2ICE-163). Included INSIDE iceberg_writer.cpp's anonymous namespace; not
// a standalone TU. Relies on TableHandle, now_ms, abs_location and metadata_json_emit from
// the including file.
//
// The table version is the compare-and-swap token: metadata/v<N>.metadata.json
// is created with the store's atomic put_if_absent, so of two writers building
// on version N-1 exactly one publishes v<N> and the other sees kConflict.

// Uncertainty is sticky until external reconciliation and a fresh handle.
bool write_ready(TableHandle* th) noexcept {
    assert(th != nullptr);
    assert(th->os != nullptr);
    th->file_io_configurable = false;
    // G2ICE-279: only the v2 writer exists. An opened v1 or v3 table is refused by name, never rewritten as v2.
    if (th->prior_meta != nullptr && th->meta.format_version != 2) {
        th->last_error = CommitError::kUnimplemented;
        std::fprintf(stderr, "[iceberg_writer] format-version %d tables are not written yet (G2ICE-276); nothing was written\n",
                     th->meta.format_version);
        return false;
    }
    if (!th->write_uncertain) return true;
    th->last_error = CommitError::kUncertain;
    return false;
}

// Create `key` iff absent. kOsExists is a conflict: someone else holds the
// name (another writer's table version, or a colliding object name).
bool commit_object(TableHandle* th, const char* key, const uint8_t* data,
                   uint64_t len, CommitObjectKind kind, int64_t snapshot_id) noexcept {
    assert(th != nullptr && th->os != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    if (!write_ready(th)) return false;
    // Complete identity must fit BEFORE sending a conditional write. Never
    // expose a truncated receipt for a possibly accepted candidate.
    CommitCandidate& c = th->candidate;
    c = CommitCandidate{};
    const size_t key_len = std::strlen(key);
    const size_t uuid_len = std::strlen(th->meta.table_uuid);
    if (key_len >= sizeof(c.object_key) || uuid_len >= sizeof(c.table_uuid) ||
        !abs_location(th->root, key, c.object_location, sizeof(c.object_location))) {
        th->last_error = CommitError::kFailed;
        return false;
    }
    std::memcpy(c.object_key, key, key_len + 1u);
    std::memcpy(c.table_uuid, th->meta.table_uuid, uuid_len + 1u);
    c.kind = kind;
    c.length = len;
    c.base_metadata_version = th->meta_version;
    c.candidate_metadata_version = th->meta_version + 1;
    c.snapshot_id = snapshot_id;
    const int rc = os_put_if_absent(th->os, key, data, len);
    if (rc == kOsOk) return true;
    if (rc == kOsWriteUncertain) {
        // The caller still owns these exact bytes. Hash only the uncertain
        // path; successful local commits need no additional full-body pass.
        crypto::sha256_hex(data, len, c.sha256);
        th->write_uncertain = true;
        th->last_error = CommitError::kUncertain;
        return false;
    }
    th->last_error = (rc == kOsExists) ? CommitError::kConflict
                                       : CommitError::kFailed;
    return false;
}

void version_key(int64_t v, char (&out)[64]) noexcept {
    assert(v >= 0);
    const int n = std::snprintf(out, sizeof(out), "metadata/v%lld.metadata.json",
                                static_cast<long long>(v));
    assert(n > 0 && static_cast<size_t>(n) < sizeof(out));
    (void)n;
}

bool version_exists(ObjectStore* os, int64_t v) noexcept {
    assert(os != nullptr && v >= 0);
    char key[64];
    version_key(v, key);
    ObjectMeta m{};
    return os_head(os, key, &m) == kOsOk && m.exists;
}

// Highest committed version at or above `from`. Versions are dense (each is
// created only on top of its predecessor), so probe forward until a gap.
constexpr int64_t kMaxVersionProbe = 1 << 20;
int64_t probe_latest(ObjectStore* os, int64_t from) noexcept {
    assert(os != nullptr && from >= 0);
    int64_t v = from;
    for (int64_t i = 0; i < kMaxVersionProbe; ++i) {   // bounded
        if (!version_exists(os, v + 1)) break;
        ++v;
    }
    assert(v >= from);
    return v;
}

// The hint is advisory and plain-overwritten, so writers can interleave and
// leave an older number last. Re-probing after the write and rewriting until
// stable makes the final hint >= every version committed before this writer
// finished: the last writer to write the hint also probes after writing it.
bool write_version_hint(ObjectStore* os, int64_t v) noexcept {
    assert(os != nullptr && v >= 1);
    for (uint32_t round = 0; round < 64u; ++round) {    // bounded
        char body[32];
        const int n = std::snprintf(body, sizeof(body), "%lld",
                                    static_cast<long long>(v));
        if (n <= 0) return false;
        if (os_put(os, "metadata/version-hint.text",
                   reinterpret_cast<const uint8_t*>(body),
                   static_cast<uint64_t>(n)) != kOsOk) return false;
        const int64_t latest = probe_latest(os, v);
        if (latest == v) return true;
        v = latest;
    }
    return true;   // hint is advisory; readers probe forward regardless
}

bool persist_metadata(TableHandle* th) noexcept {
    assert(th != nullptr && th->arena != nullptr && th->os != nullptr);
    assert(th->meta_version >= 0);
    if (!write_ready(th)) return false;
    const int64_t next = th->meta_version + 1;
    // The version this commit supersedes goes on the metadata-log, stamped
    // with that version's own last-updated-ms. Staged, not pushed, so a lost
    // CAS leaves the handle's log untouched.
    MetadataLogEntry prev{};
    const bool has_prev = th->meta_version >= 1;
    if (has_prev) {
        char prel[64];
        version_key(th->meta_version, prel);
        if (!abs_location(th->root, prel, prev.metadata_file,
                          sizeof(prev.metadata_file))) {
            th->last_error = CommitError::kFailed;
            return false;
        }
        prev.timestamp_ms = th->meta.last_updated_ms;
    }
    const int64_t prev_updated_ms = th->meta.last_updated_ms;
    th->meta.last_updated_ms = now_ms();
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!metadata_json_emit(&th->meta, th->refs, th->n_refs, th->is_view,
                            th->view_dialect, th->view_sql,
                            th->view_version_id, has_prev ? &prev : nullptr,
                            th->arena, &body, &blen)) {
        th->meta.last_updated_ms = prev_updated_ms;
        th->last_error = CommitError::kFailed;
        return false;
    }
    if (th->prior_meta != nullptr && !th->is_view) {         // G2ICE-279: keep every member the commit does not own
        const uint8_t* merged = nullptr; uint64_t mlen = 0;
        if (!metadata_merge(th->prior_meta, th->prior_meta_len, body, blen, th->dirty_meta, th->dropped_refs, th->n_dropped_refs,
                            th->arena, &merged, &mlen)) {
            std::fprintf(stderr, "[iceberg_writer] the table's metadata document could not be merged (malformed or over %u members); nothing was written\n",
                         kMergeMaxMembers);
            th->meta.last_updated_ms = prev_updated_ms;
            th->last_error = CommitError::kFailed;
            return false;
        }
        body = merged;
        blen = mlen;
    }
    char rel[64];
    version_key(next, rel);
    // The CAS: v<next> is created only if no other writer created it first.
    if (!commit_object(th, rel, body, blen, CommitObjectKind::kMetadata,
                       th->meta.current_snapshot_id)) {
        th->meta.last_updated_ms = prev_updated_ms;
        return false;
    }
    if (has_prev) metadata_metadata_log_push(&th->meta, &prev);
    if (th->prior_meta != nullptr) { th->prior_meta = body; th->prior_meta_len = blen; }
    th->dirty_meta = 0u;
    th->n_dropped_refs = 0u;
    th->meta_version = next;
    th->last_error = CommitError::kNone;
    // Committed. A failed hint write does not un-commit it (readers probe).
    (void)write_version_hint(th->os, next);
    return true;
}

// Cheap early check before writing data: has someone already moved past the
// version this handle was opened on? The atomic check is still persist's.
bool handle_is_current(TableHandle* th) noexcept {
    assert(th != nullptr && th->os != nullptr);
    if (!write_ready(th)) return false;
    if (!version_exists(th->os, th->meta_version + 1)) return true;
    th->last_error = CommitError::kConflict;
    return false;
}
