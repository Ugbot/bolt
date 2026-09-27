// bolt/lakehouse/iceberg_expire.inc — snapshot expiry and the per-table
// retention policy. Included by iceberg_writer.cpp (needs TableHandle and
// persist_metadata); not a standalone TU.

// ---------------------------------------------------------------------------
// Snapshot expiry + orphan removal.
// ---------------------------------------------------------------------------

namespace {

// Java's rule: drop every snapshot-log entry up to and including the newest
// one whose snapshot is gone, so the log never names an expired snapshot and
// never jumps over a hole.
void prune_snapshot_log(Metadata* m) noexcept {
    assert(m != nullptr);
    assert(m->n_snapshot_log <= m->cap_snapshot_log);
    uint32_t cut = 0;
    for (uint32_t i = 0; i < m->n_snapshot_log; ++i) {        // bounded
        if (snapshot_by_id(m, m->snapshot_log[i].snapshot_id) == nullptr) {
            cut = i + 1u;
        }
    }
    if (cut == 0) return;
    const uint32_t keep = m->n_snapshot_log - cut;
    std::memmove(&m->snapshot_log[0], &m->snapshot_log[cut],
                 sizeof(SnapshotLogEntry) * keep);
    m->n_snapshot_log = keep;
}

int64_t index_of_snapshot(const Metadata* m, int64_t id) noexcept {
    assert(m != nullptr);
    for (uint32_t i = 0; i < m->n_snapshots; ++i) {           // bounded
        if (m->snapshots[i].snapshot_id == id) return static_cast<int64_t>(i);
    }
    return -1;
}

uint64_t path_hash(const char* p) noexcept {
    assert(p != nullptr);
    uint64_t h = 1469598103934665603ull;
    for (; *p != '\0'; ++p) {                                  // bounded by NUL
        h ^= static_cast<uint8_t>(*p);
        h *= 1099511628211ull;
    }
    return h;
}

void sort_u64(uint64_t* v, uint32_t n) noexcept {
    assert(v != nullptr || n == 0u);
    // Shell sort: bounded, no recursion, no allocation.
    for (uint32_t gap = n / 2u; gap > 0u; gap /= 2u) {
        for (uint32_t i = gap; i < n; ++i) {
            const uint64_t x = v[i];
            uint32_t j = i;
            while (j >= gap && v[j - gap] > x) { v[j] = v[j - gap]; j -= gap; }
            v[j] = x;
        }
    }
}

bool has_u64(const uint64_t* v, uint32_t n, uint64_t x) noexcept {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {                                          // O(log n)
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (v[mid] < x) lo = mid + 1u; else hi = mid;
    }
    return lo < n && v[lo] == x;
}

bool read_manifest_list(TableHandle* th, const Snapshot& s,
                        ManifestListEntry** out, uint32_t* n) noexcept {
    assert(th != nullptr && out != nullptr && n != nullptr);
    *out = nullptr;
    *n = 0;
    if (s.manifest_list[0] == '\0') return true;
    const uint8_t* b = nullptr; uint64_t l = 0;
    if (os_get(th->os, store_key_for(th->root, s.manifest_list), th->arena,
               &b, &l) != kOsOk) return false;
    return manifest_list_parse_avro_grow(b, l, th->arena, 1u, out, n);
}

// Delete the manifest lists of `gone` and every manifest no retained snapshot
// reaches. Data files are left to table_remove_orphans. A retained list that
// cannot be read aborts the cleanup (nothing deleted): reachability unknown.
uint32_t delete_expired_files(TableHandle* th, const Snapshot* gone,
                              uint32_t n_gone) noexcept {
    assert(th != nullptr && (gone != nullptr || n_gone == 0u));
    uint32_t cap = 0, n_live = 0;
    uint64_t* live = nullptr;
    const Metadata& m = th->meta;
    for (uint32_t i = 0; i < m.n_snapshots; ++i) {            // bounded
        ManifestListEntry* ml = nullptr; uint32_t nm = 0;
        if (!read_manifest_list(th, m.snapshots[i], &ml, &nm)) return 0;
        if (!lake_grow(th->arena, nullptr, &live, n_live, &cap, n_live + nm))
            return 0;
        for (uint32_t j = 0; j < nm; ++j)
            live[n_live++] = path_hash(ml[j].manifest_path);
    }
    sort_u64(live, n_live);
    uint32_t deleted = 0;
    for (uint32_t g = 0; g < n_gone; ++g) {                   // bounded
        ManifestListEntry* ml = nullptr; uint32_t nm = 0;
        if (!read_manifest_list(th, gone[g], &ml, &nm)) continue;
        for (uint32_t j = 0; j < nm; ++j) {
            if (has_u64(live, n_live, path_hash(ml[j].manifest_path))) continue;
            if (os_delete(th->os, store_key_for(th->root, ml[j].manifest_path)) ==
                kOsOk) ++deleted;
        }
        if (gone[g].manifest_list[0] != '\0' &&
            os_delete(th->os, store_key_for(th->root, gone[g].manifest_list)) ==
                kOsOk) ++deleted;
    }
    return deleted;
}

}  // namespace

bool table_expire_snapshots_ex(TableHandle* th, const ExpireOptions* o,
                               ExpireResult* r) noexcept {
    assert(th != nullptr && o != nullptr);
    ExpireResult res{};
    Metadata& m = th->meta;
    const uint32_t n = m.n_snapshots;
    if (n == 0) { if (r != nullptr) *r = res; return true; }
    bool* keep = th->arena->allocate_array<bool>(n);
    Snapshot* before = th->arena->allocate_array<Snapshot>(n);
    if (keep == nullptr || before == nullptr) return false;
    std::memset(keep, 0, n);
    std::memcpy(before, m.snapshots, sizeof(Snapshot) * n);
    // The current snapshot and its newest `retain_last` ancestors (>= 1).
    const uint32_t retain = o->retain_last > 0 ? o->retain_last : 1u;
    int64_t id = m.current_snapshot_id;
    for (uint32_t k = 0; k < retain && k < n && id > 0; ++k) {  // bounded
        const int64_t at = index_of_snapshot(&m, id);
        if (at < 0) break;
        keep[at] = true;
        id = m.snapshots[at].parent_snapshot_id;
    }
    for (uint32_t i = 0; i < th->n_refs; ++i) {               // bounded
        const int64_t at = index_of_snapshot(&m, th->refs[i].snapshot_id);
        if (at >= 0) keep[at] = true;
    }
    uint32_t nk = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (m.snapshots[i].timestamp_ms >= o->older_than_ms) keep[i] = true;
        if (keep[i]) m.snapshots[nk++] = m.snapshots[i];
    }
    res.expired = n - nk;
    if (res.expired == 0) { if (r != nullptr) *r = res; return true; }
    Snapshot* gone = th->arena->allocate_array<Snapshot>(res.expired);
    if (gone == nullptr) { std::memcpy(m.snapshots, before, sizeof(Snapshot) * n); return false; }
    uint32_t ng = 0;
    for (uint32_t i = 0; i < n; ++i) if (!keep[i]) gone[ng++] = before[i];
    assert(ng == res.expired);
    m.n_snapshots = nk;
    const uint32_t n_log = m.n_snapshot_log;
    SnapshotLogEntry* log_before =
        th->arena->allocate_array<SnapshotLogEntry>(n_log + 1u);
    if (log_before == nullptr) {
        std::memcpy(m.snapshots, before, sizeof(Snapshot) * n);
        m.n_snapshots = n;
        return false;
    }
    std::memcpy(log_before, m.snapshot_log, sizeof(SnapshotLogEntry) * n_log);
    prune_snapshot_log(&m);
    assert(snapshot_by_id(&m, m.current_snapshot_id) != nullptr ||
           m.current_snapshot_id < 0);
    if (!persist_metadata(th)) {
        std::memcpy(m.snapshots, before, sizeof(Snapshot) * n);
        m.n_snapshots = n;
        std::memcpy(m.snapshot_log, log_before, sizeof(SnapshotLogEntry) * n_log);
        m.n_snapshot_log = n_log;
        return false;
    }
    if (o->delete_files) res.files_deleted = delete_expired_files(th, gone, ng);
    if (r != nullptr) *r = res;
    return true;
}

bool table_expire_snapshots(TableHandle* th, uint64_t older_than_ms,
                            int32_t retain_last_n) noexcept {
    assert(th != nullptr);
    ExpireOptions o{};
    o.older_than_ms = older_than_ms > static_cast<uint64_t>(INT64_MAX)
                          ? INT64_MAX : static_cast<int64_t>(older_than_ms);
    o.retain_last = retain_last_n > 0 ? static_cast<uint32_t>(retain_last_n) : 1u;
    o.delete_files = true;
    return table_expire_snapshots_ex(th, &o, nullptr);
}

bool table_set_retention(TableHandle* th, int64_t max_snapshot_age_ms,
                         int32_t min_snapshots_to_keep) noexcept {
    assert(th != nullptr);
    if (max_snapshot_age_ms < 0 || min_snapshots_to_keep < 0) return false;
    const int64_t pa = th->meta.retention_max_age_ms;
    const int32_t pk = th->meta.retention_min_snapshots;
    th->meta.retention_max_age_ms = max_snapshot_age_ms;
    th->meta.retention_min_snapshots = min_snapshots_to_keep;
    if (persist_metadata(th)) { apply_retention_policy(th); return true; }
    th->meta.retention_max_age_ms = pa;
    th->meta.retention_min_snapshots = pk;
    return false;
}

namespace {

// The per-table policy (Iceberg's history.expire.* properties) run after each
// commit. A failed expiry leaves the commit committed; the next one retries.
void apply_retention_policy(TableHandle* th) noexcept {
    assert(th != nullptr);
    const Metadata& m = th->meta;
    if (m.retention_max_age_ms <= 0 && m.retention_min_snapshots <= 0) return;
    ExpireOptions o{};
    o.older_than_ms = m.retention_max_age_ms > 0
                          ? now_ms() - m.retention_max_age_ms : INT64_MAX;
    o.retain_last = m.retention_min_snapshots > 0
                        ? static_cast<uint32_t>(m.retention_min_snapshots) : 1u;
    o.delete_files = true;
    if (!table_expire_snapshots_ex(th, &o, nullptr)) {
        std::fprintf(stderr, "[iceberg] retention expiry failed on %s; will "
                     "retry on the next commit\n", th->root);
    }
}

}  // namespace

