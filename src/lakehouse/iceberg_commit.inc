// bolt/lakehouse/iceberg_commit.inc — the Iceberg writer's commit protocol
// (G2ICE-163). Included INSIDE iceberg_writer.cpp's anonymous namespace; not
// a standalone TU. Relies on TableHandle, now_ms and metadata_json_emit from
// the including file.
//
// The table version is the compare-and-swap token: metadata/v<N>.metadata.json
// is created with the store's atomic put_if_absent, so of two writers building
// on version N-1 exactly one publishes v<N> and the other sees kConflict.

// Create `key` iff absent. kOsExists is a conflict: someone else holds the
// name (another writer's table version, or a colliding object name).
bool commit_object(TableHandle* th, const char* key, const uint8_t* data,
                   uint64_t len) noexcept {
    assert(th != nullptr && th->os != nullptr && key != nullptr);
    assert(data != nullptr || len == 0);
    const int rc = os_put_if_absent(th->os, key, data, len);
    if (rc == kOsOk) return true;
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
    const int64_t next = th->meta_version + 1;
    th->meta.last_updated_ms = now_ms();
    const uint8_t* body = nullptr; uint64_t blen = 0;
    if (!metadata_json_emit(&th->meta, th->refs, th->n_refs, th->is_view,
                            th->view_dialect, th->view_sql,
                            th->view_version_id, th->arena, &body, &blen)) {
        th->last_error = CommitError::kFailed;
        return false;
    }
    char rel[64];
    version_key(next, rel);
    // The CAS: v<next> is created only if no other writer created it first.
    if (!commit_object(th, rel, body, blen)) return false;
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
    if (!version_exists(th->os, th->meta_version + 1)) return true;
    th->last_error = CommitError::kConflict;
    return false;
}
