// iceberg_statistics_parse.inc — table metadata `statistics` /
// `partition-statistics` and Puffin footer blobs (G2ICE-450). Included by
// iceberg_metadata.cpp inside its anonymous namespace (uses its JSON helpers).

// `fields`: an array of field ids.
bool parse_blob_fields(const bj::StructuralIndex* idx, bj::Iterator* it, Arena* a, Budget* b,
                       BlobMetadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) { skip_value(it); return false; }
    bj::iter_advance(it);
    uint32_t cap = 0, g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) != bj::TokenType::EndArray && g++ < guard) {
        int64_t v = 0;
        if (!read_int64(it, &v) || v < INT32_MIN || v > INT32_MAX) return false;
        if (!lake_grow(a, b, &out->fields, out->n_fields, &cap, out->n_fields + 1u)) return false;
        out->fields[out->n_fields++] = static_cast<int32_t>(v);
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    return true;
}

// `properties`: string values; only `ndv` is kept.
bool parse_blob_properties(const bj::StructuralIndex* idx, bj::Iterator* it, BlobMetadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    uint32_t g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < guard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (!tok_eq(idx, key, "ndv")) { skip_value(it); continue; }
        char v[32];
        int64_t x = 0;
        if (read_str(idx, it, v, sizeof(v)) && parse_i64_str(v, &x)) out->ndv = x;
    }
    if (bj::iter_peek(it) != bj::TokenType::EndObject) return false;
    bj::iter_advance(it);
    return true;
}

// One blob-metadata (table metadata) or blob (Puffin footer) object.
bool parse_blob(const bj::StructuralIndex* idx, bj::Iterator* it, Arena* a, Budget* b,
                BlobMetadata* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->offset = out->length = out->ndv = -1;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) { skip_value(it); return false; }
    bj::iter_advance(it);
    bool have_type = false, ok = true;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "type")) have_type = read_str(idx, it, out->type, sizeof(out->type));
        else if (tok_eq(idx, key, "snapshot-id")) ok = read_int64(it, &out->snapshot_id) && ok;
        else if (tok_eq(idx, key, "sequence-number")) ok = read_int64(it, &out->sequence_number) && ok;
        else if (tok_eq(idx, key, "offset")) ok = read_int64(it, &out->offset) && ok;
        else if (tok_eq(idx, key, "length")) ok = read_int64(it, &out->length) && ok;
        else if (tok_eq(idx, key, "compression-codec")) (void)read_str(idx, it, out->codec, sizeof(out->codec));
        else if (tok_eq(idx, key, "fields")) ok = parse_blob_fields(idx, it, a, b, out) && ok;
        else if (tok_eq(idx, key, "properties")) ok = parse_blob_properties(idx, it, out) && ok;
        else skip_value(it);
    }
    if (bj::iter_peek(it) != bj::TokenType::EndObject) return false;
    bj::iter_advance(it);
    return ok && have_type;
}

bool parse_blob_array(const bj::StructuralIndex* idx, bj::Iterator* it, Arena* a, Budget* b,
                      BlobMetadata** blobs, uint32_t* n, uint32_t* cap) noexcept {
    assert(idx != nullptr && it != nullptr && blobs != nullptr && n != nullptr && cap != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) { skip_value(it); return false; }
    bj::iter_advance(it);
    uint32_t g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) == bj::TokenType::BeginObject && g++ < guard) {
        if (!lake_grow(a, b, blobs, *n, cap, *n + 1u)) return false;
        if (!parse_blob(idx, it, a, b, &(*blobs)[*n])) return false;
        ++*n;
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    return true;
}

bool parse_statistics_file(const bj::StructuralIndex* idx, bj::Iterator* it, Metadata* m,
                           StatisticsFile* out) noexcept {
    assert(idx != nullptr && it != nullptr && m != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->snapshot_id = out->file_size = out->footer_size = -1;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) { skip_value(it); return false; }
    bj::iter_advance(it);
    bool ok = true, have_path = false;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "snapshot-id")) ok = read_int64(it, &out->snapshot_id) && ok;
        else if (tok_eq(idx, key, "statistics-path")) have_path = read_str(idx, it, out->path, sizeof(out->path));
        else if (tok_eq(idx, key, "file-size-in-bytes")) ok = read_int64(it, &out->file_size) && ok;
        else if (tok_eq(idx, key, "file-footer-size-in-bytes")) ok = read_int64(it, &out->footer_size) && ok;
        else if (tok_eq(idx, key, "blob-metadata"))
            ok = parse_blob_array(idx, it, m->arena, m->budget, &out->blobs, &out->n_blobs, &out->cap_blobs) && ok;
        else skip_value(it);
    }
    if (bj::iter_peek(it) != bj::TokenType::EndObject) return false;
    bj::iter_advance(it);
    return ok && have_path;
}

bool parse_partition_statistics_file(const bj::StructuralIndex* idx, bj::Iterator* it,
                                     PartitionStatisticsFile* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->snapshot_id = out->file_size = -1;
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) { skip_value(it); return false; }
    bj::iter_advance(it);
    bool ok = true, have_path = false;
    uint32_t g = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g++ < kIterGuard) {
        const int32_t key = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key, "snapshot-id")) ok = read_int64(it, &out->snapshot_id) && ok;
        else if (tok_eq(idx, key, "statistics-path")) have_path = read_str(idx, it, out->path, sizeof(out->path));
        else if (tok_eq(idx, key, "file-size-in-bytes")) ok = read_int64(it, &out->file_size) && ok;
        else skip_value(it);
    }
    if (bj::iter_peek(it) != bj::TokenType::EndObject) return false;
    bj::iter_advance(it);
    return ok && have_path;
}

// `statistics` (partition == false) or `partition-statistics`: an array of files.
bool parse_statistics_list(const bj::StructuralIndex* idx, bj::Iterator* it, Metadata* m,
                           bool partition) noexcept {
    assert(idx != nullptr && it != nullptr && m != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginArray) return skip_value(it);
    bj::iter_advance(it);
    uint32_t g = 0;
    const uint32_t guard = array_guard(idx);
    while (bj::iter_peek(it) == bj::TokenType::BeginObject && g++ < guard) {
        if (partition) {
            if (!lake_grow(m->arena, m->budget, &m->partition_statistics, m->n_partition_statistics,
                           &m->cap_partition_statistics, m->n_partition_statistics + 1u)) return false;
            if (!parse_partition_statistics_file(idx, it, &m->partition_statistics[m->n_partition_statistics]))
                return false;
            ++m->n_partition_statistics;
        } else {
            if (!lake_grow(m->arena, m->budget, &m->statistics, m->n_statistics, &m->cap_statistics,
                           m->n_statistics + 1u)) return false;
            if (!parse_statistics_file(idx, it, m, &m->statistics[m->n_statistics])) return false;
            ++m->n_statistics;
        }
    }
    if (bj::iter_peek(it) != bj::TokenType::EndArray) return false;
    bj::iter_advance(it);
    return true;
}
