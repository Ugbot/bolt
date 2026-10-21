// Included inside iceberg_writer.cpp's anonymous namespace, after path helpers.
constexpr uint32_t kStageHashBytes = 64u << 10;
bool write_ready(TableHandle* th) noexcept;

bool stage_text(const char* text, uint32_t cap, size_t* length) noexcept {
    assert(text != nullptr && length != nullptr);
    assert(cap > 0);
    for (uint32_t i = 0; i < cap; ++i) {
        if (text[i] == 0) { *length = i; return true; }
    }
    return false;
}
bool file_io_valid(const FileWriteIO* config) noexcept {
    assert(config != nullptr);
    size_t length = 0;
    if (!stage_text(config->staging_root, kIcebergStagedPathCap, &length) || length == 0 ||
        config->publish_if_absent == nullptr || config->max_staged_file_bytes < 4u ||
        config->max_staged_file_bytes > INT64_MAX || config->max_staged_file_bytes > SIZE_MAX ||
        config->max_staged_file_bytes > crypto::kSha256MaxInputBytes) return false;
    const std::filesystem::path root(config->staging_root);
    std::error_code ec;
    const bool directory = root.is_absolute() && std::filesystem::is_directory(root, ec);
    assert(length < kIcebergStagedPathCap);
    return directory && !ec;
}
void stage_hex(const uint8_t digest[32], char out[65]) noexcept {
    assert(digest != nullptr && out != nullptr);
    constexpr char digits[] = "0123456789abcdef";
    for (uint32_t i = 0; i < 32; ++i) {
        out[2 * i] = digits[digest[i] >> 4u];
        out[2 * i + 1] = digits[digest[i] & 15u];
    }
    out[64] = 0;
    assert(std::strlen(out) == 64);
}
bool stage_hash_number(crypto::Sha256State* hash, uint64_t value) noexcept {
    assert(hash != nullptr);
    uint8_t bytes[8]{};
    for (uint32_t i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (i * 8u));
    assert(sizeof(bytes) == 8);
    return crypto::sha256_stream_update(hash, bytes, sizeof(bytes));
}
bool stage_hash_text(crypto::Sha256State* hash, const char* text) noexcept {
    assert(hash != nullptr && text != nullptr);
    const size_t length = std::strlen(text);
    assert(length < kIcebergStagedPathCap);
    return stage_hash_number(hash, length) && crypto::sha256_stream_update(hash,
        reinterpret_cast<const uint8_t*>(text), length);
}
bool stage_operation_id(StagedDataFile* receipt) noexcept {
    assert(receipt != nullptr);
    assert(receipt->candidate.kind == CommitObjectKind::kDataFile);
    const CommitCandidate& c = receipt->candidate;
    crypto::Sha256State hash{};
    uint8_t digest[32]{};
    if (!crypto::sha256_stream_init(&hash) || !stage_hash_text(&hash, "bolt.iceberg.data.v1") ||
        !stage_hash_text(&hash, c.object_key) || !stage_hash_text(&hash, c.object_location) ||
        !stage_hash_text(&hash, c.table_uuid) || !stage_hash_text(&hash, c.sha256) ||
        !stage_hash_number(&hash, c.length) || !stage_hash_number(&hash, static_cast<uint64_t>(c.base_metadata_version)) ||
        !stage_hash_number(&hash, static_cast<uint64_t>(c.candidate_metadata_version)) ||
        !stage_hash_number(&hash, static_cast<uint64_t>(c.snapshot_id)) ||
        !crypto::sha256_stream_final(&hash, digest)) return false;
    stage_hex(digest, receipt->operation_id);
    return true;
}
bool stage_hash_closed(FileWriteState* state) noexcept {
    assert(state != nullptr && state->hash_workspace != nullptr);
    assert(state->receipt.source_retained);
    io::ReadFile file{}; file.handle = -1;
    if (io::file_open_read(state->receipt.local_path, false, &file) != io::FileIoStatus::kOk) return false;
    crypto::Sha256State hash{};
    bool ok = file.size >= 12u && file.size <= state->config.max_staged_file_bytes &&
        file.size <= INT64_MAX && crypto::sha256_stream_init(&hash);
    uint64_t offset = 0;
    const uint64_t steps = file.size / kStageHashBytes + 1u;
    for (uint64_t step = 0; ok && offset < file.size && step < steps; ++step) {
        const uint64_t want = file.size - offset < kStageHashBytes ? file.size - offset : kStageHashBytes;
        uint64_t got = 0;
        ok = io::file_pread(file, offset, state->hash_workspace, want, &got) == io::FileIoStatus::kOk &&
            got == want && crypto::sha256_stream_update(&hash, state->hash_workspace, want);
        if (ok) offset += want;
    }
    uint8_t digest[32]{};
    ok = ok && offset == file.size && crypto::sha256_stream_final(&hash, digest);
    if (ok) {
        state->receipt.candidate.length = file.size;
        stage_hex(digest, state->receipt.candidate.sha256);
    }
    io::file_close(&file);
    return ok && stage_operation_id(&state->receipt);
}
bool stage_directory_cleanable(const char* path) noexcept {
    assert(path != nullptr);
    const std::filesystem::path source(path);
    std::error_code ec;
    std::filesystem::directory_iterator entry(source.parent_path(), ec), end;
    if (ec) return false;
    for (uint32_t i = 0; entry != end && i < 2; ++i) {
        if (entry->path().filename() != source.filename()) return false;
        entry.increment(ec);
        if (ec) return false;
    }
    assert(path[0] != 0);
    return entry == end;
}
void stage_cleanup(FileWriteState* state) noexcept {
    assert(state != nullptr);
    assert(state->has_receipt);
    auto& receipt = state->receipt;
    if (!receipt.source_retained) return;
    if (!stage_directory_cleanable(receipt.local_path)) { receipt.cleanup_status = kOsIoError; return; }
    std::error_code ec;
    (void)std::filesystem::remove(receipt.local_path, ec);
    if (!ec) {
        receipt.source_retained = false;
        (void)std::filesystem::remove(std::filesystem::path(receipt.local_path).parent_path(), ec);
    }
    receipt.cleanup_status = ec ? kOsIoError : kOsOk;
}
bool stage_prepare(TableHandle* th, const char* key, int64_t snapshot) noexcept {
    assert(th != nullptr && th->file_io != nullptr);
    assert(key != nullptr && snapshot > 0);
    FileWriteState& state = *th->file_io;
    if (state.has_receipt && state.receipt.source_retained) return false;
    StagedDataFile next{};
    auto& c = next.candidate;
    size_t key_length = 0, uuid_length = 0;
    if (th->meta_version == INT64_MAX || !stage_text(key, sizeof(c.object_key), &key_length) ||
        !stage_text(th->meta.table_uuid, sizeof(c.table_uuid), &uuid_length) || uuid_length == 0 ||
        !abs_location(th->root, key, c.object_location, sizeof(c.object_location))) return false;
    std::memcpy(c.object_key, key, key_length + 1u);
    std::memcpy(c.table_uuid, th->meta.table_uuid, uuid_length + 1u);
    c.kind = CommitObjectKind::kDataFile;
    c.base_metadata_version = th->meta_version;
    c.candidate_metadata_version = th->meta_version + 1;
    c.snapshot_id = snapshot;
    char directory[kIcebergStagedPathCap]{};
    if (std::strncmp(key, "data/", 5) != 0 || std::strchr(key + 5, '/') != nullptr ||
        std::strchr(key + 5, '\\') != nullptr ||
        !path_join(state.config.staging_root, key + 5, directory, sizeof(directory)) ||
        !path_join(directory, "source.parquet", next.local_path, sizeof(next.local_path))) return false;
    std::error_code ec;
    if (!std::filesystem::create_directory(directory, ec) || ec) return false;
    next.source_retained = true;
    state.receipt = next;
    state.has_receipt = true;
    return true;
}
ingest::parquet::ParquetWriter* data_file_begin(TableHandle* th, const char* key,
    int64_t snapshot, const ingest::parquet::ParquetWriteOpts* options) noexcept {
    assert(th != nullptr && key != nullptr);
    assert(options != nullptr);
    char location[sizeof(DataFileRef::file_path)]{};
    if (!abs_location(th->root, key, location, sizeof(location))) return nullptr;
    if (th->file_io != nullptr) {
        if (!stage_prepare(th, key, snapshot)) return nullptr;
        auto* writer = ingest::parquet::parquet_write_open_bounded(th->file_io->receipt.local_path,
            options, th->file_io->config.max_staged_file_bytes);
        if (writer == nullptr) stage_cleanup(th->file_io);
        return writer;
    }
    char full[kMaxPathLen]{};
    if (!path_join(th->root, key, full, sizeof(full))) return nullptr;
    std::error_code ec;
    (void)std::filesystem::create_directories(std::filesystem::path(full).parent_path(), ec);
    return ec ? nullptr : ingest::parquet::parquet_write_open(full, options);
}
void data_file_abandon(TableHandle* th, ingest::parquet::ParquetWriter* writer) noexcept {
    assert(th != nullptr && writer != nullptr);
    const bool closed = ingest::parquet::parquet_write_close(writer);
    (void)closed; // primary serialization failure already determines refusal
    if (th->file_io != nullptr) stage_cleanup(th->file_io);
    assert(!th->write_uncertain);
}
bool data_file_finish(TableHandle* th, const char* key,
    ingest::parquet::ParquetWriter* writer, int64_t* size) noexcept {
    assert(th != nullptr && key != nullptr);
    assert(writer != nullptr && size != nullptr);
    if (!ingest::parquet::parquet_write_close(writer)) {
        if (th->file_io != nullptr) stage_cleanup(th->file_io);
        th->last_error = CommitError::kFailed;
        return false;
    }
    if (th->file_io == nullptr) {
        ObjectMeta meta{};
        if (os_head(th->os, key, &meta) != kOsOk || !meta.exists || meta.size > INT64_MAX) {
            th->last_error = CommitError::kFailed;
            return false;
        }
        *size = static_cast<int64_t>(meta.size);
        return true;
    }
    FileWriteState& state = *th->file_io;
    if (!stage_hash_closed(&state)) { stage_cleanup(&state); th->last_error = CommitError::kFailed; return false; }
    const int rc = state.config.publish_if_absent(state.config.context, &state.receipt);
    if (rc == kOsOk) {
        state.receipt.published = true;
        *size = static_cast<int64_t>(state.receipt.candidate.length);
        return true;
    }
    if (rc == kOsWriteUncertain) {
        th->candidate = state.receipt.candidate;
        th->write_uncertain = true;
        th->last_error = CommitError::kUncertain;
    } else {
        th->last_error = rc == kOsExists ? CommitError::kConflict : CommitError::kFailed;
        stage_cleanup(&state);
    }
    return false;
}
void data_file_complete(TableHandle* th, const char* key, bool committed) noexcept {
    assert(th != nullptr && key != nullptr);
    assert(th->os != nullptr);
    if (th->write_uncertain) return;
    if (th->file_io == nullptr) {
        if (!committed) { const int cleanup = os_delete(th->os, key); (void)cleanup; }
        return;
    }
    FileWriteState& state = *th->file_io;
    if (!committed && state.receipt.published) {
        const int rc = os_delete(th->os, key);
        state.receipt.remote_cleanup_status = rc;
        if (rc != kOsOk && rc != kOsNotFound) return; // preserve source evidence on failed cleanup
    }
    stage_cleanup(&state);
}
