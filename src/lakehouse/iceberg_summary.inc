// bolt/lakehouse/iceberg_summary.inc — snapshot summary totals (G2ICE-280). Included inside iceberg_writer.cpp's
// anonymous namespace after iceberg_change.inc; not a standalone TU.
//
// Totals follow Iceberg-Java: the parent snapshot's totals plus this commit's deltas. When the parent carries no
// totals (a snapshot an older writer committed) they are recounted from every live entry of the new snapshot's
// manifests, once; from then on the chain carries them.

// Signed decimal inside a JSON string value ("\"123\""); false on anything else.
bool json_string_i64(const char* v, uint32_t n, int64_t* out) noexcept {
    assert(v != nullptr && out != nullptr);
    if (n < 3u || v[0] != '"' || v[n - 1u] != '"') return false;
    int64_t x = 0;
    bool neg = false;
    uint32_t i = 1;
    if (v[i] == '-') { neg = true; ++i; }
    if (i >= n - 1u) return false;
    for (; i < n - 1u; ++i) {                                 // bounded by n
        if (v[i] < '0' || v[i] > '9' || x > (INT64_MAX - 9) / 10) return false;
        x = x * 10 + (v[i] - '0');
    }
    *out = neg ? -x : x;
    return true;
}

// The totals another commit wrote on snapshot `id`, read from the document this handle last read or wrote.
bool prior_summary_totals(TableHandle* th, int64_t id, SnapSummary* out) noexcept {
    assert(th != nullptr && out != nullptr);
    if (th->prior_meta == nullptr) return false;
    auto* mem = th->arena->allocate_array<JsonMember>(kMergeMaxMembers * 2u);
    if (mem == nullptr) return false;
    const char* doc = reinterpret_cast<const char*>(th->prior_meta);
    const char* end = doc + th->prior_meta_len;
    const char* arr = nullptr; uint32_t arr_len = 0;
    if (!object_value(doc, end, "snapshots", mem, &arr, &arr_len)) return false;
    char idt[24];
    const int il = std::snprintf(idt, sizeof(idt), "%lld", static_cast<long long>(id));
    if (il <= 0) return false;
    const char* oe = nullptr;
    const char* obj = prior_snapshot(arr, arr + arr_len, idt, static_cast<uint32_t>(il), mem, &oe);
    const char* sum = nullptr; uint32_t sum_len = 0;
    if (obj == nullptr || !object_value(obj, oe, "summary", mem, &sum, &sum_len)) return false;
    JsonMember* sm = mem + kMergeMaxMembers;
    uint32_t ns = 0;
    if (!json_object_members(sum, sum + sum_len, sm, &ns)) return false;
    for (uint32_t k = kSumFirstTotal; k < kSumKeys; ++k) {     // bounded
        bool got = false;
        for (uint32_t i = 0; i < ns && !got; ++i)               // bounded by ns
            if (member_is(sm[i], kSumNames[k])) got = json_string_i64(sm[i].val, sm[i].val_len, &out->v[k]);
        if (!got) return false;
    }
    out->has_totals = true;
    return true;
}

bool parent_totals(TableHandle* th, int64_t parent, SnapSummary* out) noexcept {
    assert(th != nullptr && out != nullptr);
    *out = SnapSummary{};
    const SnapSummary* mine = find_summary(th->sums, th->n_sums, parent);
    if (mine != nullptr && mine->has_totals) { *out = *mine; return true; }
    return prior_summary_totals(th, parent, out);
}

// Every live entry of the snapshot's manifests, counted.
bool recount_totals(TableHandle* th, const ManifestListEntry* ml, uint32_t n, SnapSummary* s) noexcept {
    assert(th != nullptr && s != nullptr);
    assert(ml != nullptr || n == 0);
    for (uint32_t k = kSumFirstTotal; k < kSumKeys; ++k) s->v[k] = 0;
    for (uint32_t m = 0; m < n; ++m) {                          // bounded by n
        DataFileRef* e = nullptr; uint32_t ne = 0;
        if (!read_manifest_entries(th, ml[m], &e, &ne)) return false;
        for (uint32_t i = 0; i < ne; ++i) {                     // bounded by ne
            const DataFileRef& d = e[i];
            if (d.status == ManifestStatus::kDeleted) continue;
            s->v[kSumTotalFilesSize] += d.stats.file_size_in_bytes;
            if (d.content == FileContent::kData) {
                s->v[kSumTotalDataFiles] += 1;
                s->v[kSumTotalRecords] += d.stats.record_count;
                continue;
            }
            s->v[kSumTotalDeleteFiles] += 1;
            s->v[d.content == FileContent::kEqualityDeletes ? kSumTotalEqDeletes : kSumTotalPosDeletes] +=
                d.stats.record_count;
        }
    }
    s->has_totals = true;
    return true;
}

// Fill `s`'s totals (its deltas already hold this commit's changes).
bool summary_totals(TableHandle* th, int64_t parent, const ManifestListEntry* ml, uint32_t n,
                    SnapSummary* s) noexcept {
    assert(th != nullptr && s != nullptr);
    SnapSummary p{};
    if (parent > 0 && !parent_totals(th, parent, &p)) return recount_totals(th, ml, n, s);
    const int64_t* d = s->v;
    s->v[kSumTotalRecords]     = p.v[kSumTotalRecords] + d[kSumAddedRecords] - d[kSumDeletedRecords];
    s->v[kSumTotalFilesSize]   = p.v[kSumTotalFilesSize] + d[kSumAddedFilesSize] - d[kSumRemovedFilesSize];
    s->v[kSumTotalDataFiles]   = p.v[kSumTotalDataFiles] + d[kSumAddedDataFiles] - d[kSumDeletedDataFiles];
    s->v[kSumTotalDeleteFiles] = p.v[kSumTotalDeleteFiles] + d[kSumAddedDeleteFiles] - d[kSumRemovedDeleteFiles];
    s->v[kSumTotalPosDeletes]  = p.v[kSumTotalPosDeletes] + d[kSumAddedPosDeletes] - d[kSumRemovedPosDeletes];
    s->v[kSumTotalEqDeletes]   = p.v[kSumTotalEqDeletes] + d[kSumAddedEqDeletes] - d[kSumRemovedEqDeletes];
    s->has_totals = true;
    assert(s->v[kSumTotalDataFiles] >= 0);
    return true;
}
