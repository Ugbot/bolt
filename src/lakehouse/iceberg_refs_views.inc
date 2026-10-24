// bolt/lakehouse/iceberg_refs_views.inc — branches, tags and views. Included inside iceberg_writer.cpp at
// namespace scope; not a standalone TU.

// ---------------------------------------------------------------------------
// Branches + tags.
// ---------------------------------------------------------------------------

bool table_create_branch(TableHandle* th, const char* name,
                         int64_t base_snapshot_id) noexcept {
    assert(th != nullptr && name != nullptr);
    if (!write_ready(th)) return false;
    if (th->n_refs >= kIcebergMaxRefs) return false;
    NamedRef& r = th->refs[th->n_refs];
    std::memset(&r, 0, sizeof(r));
    std::strncpy(r.name, name, sizeof(r.name) - 1u);
    r.snapshot_id = base_snapshot_id;
    r.kind = 0;
    th->n_refs++;
    th->dirty_meta |= kDirtyRefs;
    return persist_metadata(th);
}

bool table_create_tag(TableHandle* th, const char* name,
                       int64_t snapshot_id) noexcept {
    assert(th != nullptr && name != nullptr);
    if (!write_ready(th)) return false;
    if (th->n_refs >= kIcebergMaxRefs) return false;
    NamedRef& r = th->refs[th->n_refs];
    std::memset(&r, 0, sizeof(r));
    std::strncpy(r.name, name, sizeof(r.name) - 1u);
    r.snapshot_id = snapshot_id;
    r.kind = 1;
    th->n_refs++;
    th->dirty_meta |= kDirtyRefs;
    return persist_metadata(th);
}

bool table_drop_ref(TableHandle* th, const char* name) noexcept {
    assert(th != nullptr && name != nullptr);
    if (!write_ready(th)) return false;
    uint32_t out = 0;
    bool dropped = false;
    for (uint32_t i = 0; i < th->n_refs; ++i) {
        if (std::strcmp(th->refs[i].name, name) == 0) { dropped = true; continue; }
        if (out != i) th->refs[out] = th->refs[i];
        ++out;
    }
    if (!dropped && th->prior_meta == nullptr) return false;
    th->n_refs = out;
    if (th->n_dropped_refs >= 16u) return false;
    std::strncpy(th->dropped_refs[th->n_dropped_refs++], name, 63u);   // a foreign ref (one this handle never loaded) is dropped from the document
    th->dirty_meta |= kDirtyRefs;
    return persist_metadata(th);
}

// ---------------------------------------------------------------------------
// View support.
// ---------------------------------------------------------------------------

bool view_create(TableHandle** out, Arena* arena, ObjectStore* os,
                 const char* path, const char* sql_dialect, const char* sql,
                 const Schema* schema) noexcept {
    assert(out != nullptr && arena != nullptr && os != nullptr);
    assert(path != nullptr && sql_dialect != nullptr && sql != nullptr);
    assert(schema != nullptr);
    if (std::strlen(sql_dialect) + 1u > kIcebergMaxSqlDialect) return false;
    if (std::strlen(sql)         + 1u > kIcebergMaxSql)        return false;
    WriteOptions wo; write_options_init(&wo);
    if (!table_create(out, arena, os, path, schema, nullptr, nullptr, &wo))
        return false;
    TableHandle* h = *out;
    h->is_view = true;
    std::strncpy(h->view_dialect, sql_dialect, sizeof(h->view_dialect) - 1u);
    std::strncpy(h->view_sql,     sql,         sizeof(h->view_sql)     - 1u);
    h->view_version_id = 1;
    return persist_metadata(h);
}

bool view_replace(TableHandle* th, const char* sql_dialect,
                  const char* sql) noexcept {
    assert(th != nullptr && sql_dialect != nullptr && sql != nullptr);
    if (!write_ready(th)) return false;
    if (!th->is_view) return false;
    if (std::strlen(sql_dialect) + 1u > kIcebergMaxSqlDialect) return false;
    if (std::strlen(sql)         + 1u > kIcebergMaxSql)        return false;
    std::memset(th->view_dialect, 0, sizeof(th->view_dialect));
    std::memset(th->view_sql,     0, sizeof(th->view_sql));
    std::strncpy(th->view_dialect, sql_dialect, sizeof(th->view_dialect) - 1u);
    std::strncpy(th->view_sql,     sql,         sizeof(th->view_sql)     - 1u);
    th->view_version_id++;
    return persist_metadata(th);
}

