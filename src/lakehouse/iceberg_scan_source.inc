// Private bounded source adapter. Included after ScanHandle, inside namespace.
bool scan_open_admit(ScanHandle** out, TableHandle* table, const ReadOptions* opts,
    const IcebergScanIO* io) noexcept {
    assert(out != nullptr && table != nullptr);
    assert(table->arena != nullptr);
    if (out == nullptr || table == nullptr || table->arena == nullptr ||
        (io != nullptr && io->workspace_bytes > SIZE_MAX)) return false;
    using detail::ObjectSpan;
    const ObjectSpan inputs[]{{io, io == nullptr ? 0u : sizeof(*io)},
        {io == nullptr ? nullptr : io->workspace,
         io == nullptr ? 0u : static_cast<size_t>(io->workspace_bytes)},
        {opts,opts == nullptr ? 0u : sizeof(*opts)}};
    const ObjectSpan outputs[]{{out, sizeof(*out)}};
    const ObjectSpan protected_spans[]{{table,sizeof(*table)}, {table->arena,sizeof(Arena)},
        {table->os.vt,table->os.vt == nullptr ? 0u : sizeof(ObjectStoreVT)},
        {table->os.impl,table->os.impl == nullptr ? 0u : 1u}};
    Arena* scratch = io == nullptr ? nullptr : io->transport_scratch;
    if (table->strict_pinned && !pinned_scan_io_admit(table,inputs,3,outputs[0],scratch)) return false;
    // Reject ALL proposed Arena object aliases before inspecting its backing.
    if (scratch!=nullptr) {
        const ObjectSpan object{scratch,sizeof(Arena)};
        if (!detail::object_span_valid(object)) return false;
        for (const auto span : protected_spans)
            if (detail::object_spans_overlap(object,span)) return false;
        for (const auto span : inputs)
            if (detail::object_spans_overlap(object,span)) return false;
        if (detail::object_spans_overlap(object,outputs[0])) return false;
    }
    for (const auto span : protected_spans) {
        if (!detail::object_span_valid(span)) return false;
        if (scratch != nullptr && (detail::object_spans_overlap(span,{scratch,sizeof(Arena)}) ||
            scratch->overlaps_backing(span.data,span.size))) return false;
    }
    return detail::object_io_admit(inputs, 3, outputs, 1,scratch,protected_spans,4);
}

bool scan_outputs_admit(ScanHandle* s, const detail::ObjectSpan* outputs,
    uint32_t count) noexcept {
    assert(s != nullptr && outputs != nullptr);
    assert(count > 0 && count <= 2);
    if (s->table->strict_pinned)
        for (uint32_t i=0; i<count; ++i)
            if (!pinned_scan_io_admit(s->table,nullptr,0,outputs[i],nullptr)) return false;
    const detail::ObjectSpan inputs[]{{s->meta_arena,s->meta_arena == nullptr ? 0u : sizeof(Arena)},
        {s->man_arena,s->man_arena == nullptr ? 0u : sizeof(Arena)}};
    const detail::ObjectSpan protected_spans[]{{s,sizeof(*s)}, {s->table,sizeof(*s->table)},
        {s->scratch,sizeof(Arena)}, {s->table->os.vt,sizeof(ObjectStoreVT)},
        {s->table->os.impl,s->table->os.impl == nullptr ? 0u : 1u},
        {s->source_workspace.bytes,static_cast<size_t>(s->source_workspace.capacity)},
        {s->source_arena,s->source_arena == nullptr ? 0u : sizeof(Arena)}};
    if (!detail::object_io_admit(inputs,2,outputs,count,s->transport_arena,
                                 protected_spans,7)) return false;
    for (uint32_t i = 0; i < count; ++i) {
        if ((s->meta_arena != nullptr && s->meta_arena->overlaps_backing(outputs[i].data,outputs[i].size)) ||
            (s->man_arena != nullptr && s->man_arena->overlaps_backing(outputs[i].data,outputs[i].size))) return false;
    }
    return true;
}

pq::ParquetSourceStatus pinned_parquet_read(void* context, uint64_t offset,
    uint32_t length, uint8_t* dst) noexcept {
    assert(context != nullptr && dst != nullptr);
    assert(length > 0 && length <= pq::kParquetSourceMaxReadBytes);
    auto* file = static_cast<PinnedParquet*>(context);
    file->io->reset();
    uint32_t got = 0;
    const int rc = os_get_range_if_match(file->store, file->key, &file->version,
                                        offset, length, file->io, dst, &got);
    if (rc == kOsPreconditionFailed) return pq::ParquetSourceStatus::Changed;
    if (rc == kOsBufferSmall) return pq::ParquetSourceStatus::Capacity;
    return rc == kOsOk && got == length ? pq::ParquetSourceStatus::Ok
                                       : pq::ParquetSourceStatus::IoError;
}

Arena* scan_fixed_arena(Arena* owner, uint32_t bytes) noexcept {
    assert(owner != nullptr);
    assert(bytes > 0);
    void* p = owner->allocate(sizeof(Arena), alignof(Arena));
    if (p == nullptr) return nullptr;
    auto* arena = new (p) Arena({bytes, bytes, 64, false});
    arena->set_growth_enabled(false);
    if (arena->allocate(bytes, 1) == nullptr) { arena->~Arena(); return nullptr; }
    arena->reset();
    return arena;
}

bool borrowed_scan_io_valid(ScanHandle* s, const IcebergScanIO& io) noexcept {
    assert(s != nullptr && s->scratch != nullptr);
    assert(s->table != nullptr);
    if (io.workspace_bytes == 0 || io.workspace_bytes > SIZE_MAX ||
        io.workspace == nullptr || io.transport_scratch == nullptr ||
        io.request_bytes == 0 || io.request_bytes > pq::kParquetSourceMaxReadBytes ||
        io.transport_scratch == s->scratch) return false;
    using detail::ObjectSpan;
    const ObjectSpan inputs[]{{io.workspace, static_cast<size_t>(io.workspace_bytes)},
        {io.transport_scratch, sizeof(Arena)}};
    if (!detail::object_io_admit(inputs, 2, nullptr, 0, s->scratch, nullptr, 0) ||
        detail::object_spans_overlap(inputs[0], inputs[1]) ||
        io.transport_scratch->overlaps_backing(io.workspace,
                                               static_cast<size_t>(io.workspace_bytes))) return false;
    return true;
}

bool setup_scan_source(ScanHandle* s, const IcebergScanIO* supplied) noexcept {
    assert(s != nullptr && s->table != nullptr);
    assert(s->scratch != nullptr);
    const auto* vt = s->table->os.vt;
    if (vt == nullptr) return false;
    const bool head = vt->head_versioned != nullptr;
    const bool ranges = vt->get_range_if_match != nullptr;
    if (head != ranges || (supplied != nullptr && !head)) return false;
    s->ranged = head;
    if (!head) return true;
    for (uint32_t i = 0; i < kMaxColumns; ++i) s->source_indices[i] = static_cast<uint16_t>(i);
    if (supplied != nullptr) {
        const IcebergScanIO io = *supplied;
        if (s->table->strict_pinned && io.request_bytes>s->table->request_bytes) return false;
        if (!borrowed_scan_io_valid(s, io)) return false;
        io.transport_scratch->set_growth_enabled(false);
        io.transport_scratch->reset();
        const uint32_t reserve = io.request_bytes > (64u << 10) ? io.request_bytes : (64u << 10);
        if (io.transport_scratch->allocate(reserve, 1) == nullptr) return false;
        io.transport_scratch->reset();
        s->source_workspace = {io.workspace, io.workspace_bytes, io.request_bytes};
        s->transport_arena = io.transport_scratch;
        return true;
    }
    s->source_arena = scan_fixed_arena(s->scratch, kIcebergScanWorkspaceBytes);
    s->transport_arena = scan_fixed_arena(s->scratch, pq::kParquetSourceMaxReadBytes);
    s->owns_transport = true;
    if (s->source_arena == nullptr || s->transport_arena == nullptr) return false;
    auto* bytes = static_cast<uint8_t*>(s->source_arena->allocate(kIcebergScanWorkspaceBytes, 1));
    if (bytes == nullptr) return false;
    s->source_workspace = {bytes, kIcebergScanWorkspaceBytes, pq::kParquetSourceMaxReadBytes};
    if (s->table->strict_pinned) s->source_workspace.request_bytes=s->table->request_bytes;
    return true;
}

// Pick ONE key before I/O. A located file never falls through to another key
// after a missing object, transport failure or changed version.
bool scan_source_key(ScanHandle* s, const char* path, char* out) noexcept {
    assert(s != nullptr && path != nullptr);
    assert(out != nullptr);
    const auto* table = s->table;
    if (table->strict_pinned)
        return s3_uri_object_key(table->fs_root,path,out,kCatMaxPath)==S3UriStatus::Ok;
    if (key_from_location(table->meta.location, table->table_rel, path, out, kCatMaxPath))
        return true;
    if (std::strstr(path, "://") != nullptr || path[0] == '/' || path[0] == '\\') return false;
    uint32_t prefix = 0;
    if (path_prefix_len(path, table->table_rel, &prefix)) {
        detail::ObjectSpan span{};
        if (!detail::object_key_span(path, kCatMaxPath, &span)) return false;
        std::memcpy(out, path, span.size);
        return true;
    }
    return path_join(table->table_rel, path, out, kCatMaxPath);
}

bool open_parquet(ScanHandle* s, const char* path, Arena* metadata,
    pq::PqMeta* meta, PinnedParquet* file, const uint8_t** body,
    uint64_t* length) noexcept {
    assert(s != nullptr && file != nullptr && meta != nullptr);
    assert(metadata != nullptr && body != nullptr && length != nullptr);
    if (!s->ranged) {
        return read_ref(&s->table->os, s->table->meta.location, s->table->fs_root,
                        s->table->table_rel, path, s->scratch, body, length) &&
               pq::parquet_read_meta(*body, *length, metadata, meta);
    }
    *file = PinnedParquet{};
    if (!scan_source_key(s, path, file->key)) return false;
    file->store = &s->table->os;
    file->io = s->transport_arena;
    file->io->reset();
    const int rc = os_head_versioned(file->store, file->key, file->io, &file->version);
    if (rc != kOsOk || !os_object_version_valid(&file->version) || file->version.size < 12) return false;
    file->source = {file->version.size, file, pinned_parquet_read};
    if (s->table->strict_pinned)
        return pq::parquet_read_meta_source_flat_strict(&file->source,&s->source_workspace,
            metadata,meta)==pq::ParquetSourceStatus::Ok;
    return pq::parquet_read_meta_source(&file->source, &s->source_workspace,
                                        metadata, meta) == pq::ParquetSourceStatus::Ok;
}

bool read_parquet_group(ScanHandle* s, PinnedParquet* file, const uint8_t* body,
    uint64_t length, const pq::PqMeta* meta, uint32_t row_group,
    BoltColumn* columns, int64_t* rows, uint32_t selected=0) noexcept {
    assert(s != nullptr && meta != nullptr && columns != nullptr);
    assert(rows != nullptr);
    if (!s->ranged) return pq::parquet_read_row_group(body, length, meta, row_group,
                                                    s->scratch, columns, rows);
    if (meta->n_columns == 0 || meta->n_columns > kMaxColumns) return false;
    const uint32_t count = selected==0 ? meta->n_columns : selected;
    if (count>kMaxColumns) return false;
    return pq::parquet_read_row_group_cols_source(&file->source, &s->source_workspace,
        meta, row_group, s->source_indices, count, s->scratch,
        columns, rows) == pq::ParquetSourceStatus::Ok;
}
