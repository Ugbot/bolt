// Included in the iceberg namespace after TableHandle and private helpers.
CommitError table_last_commit_error(const TableHandle* th) noexcept {
    assert(th != nullptr);
    return th->last_error;
}
bool table_uncertain_commit(const TableHandle* th, CommitCandidate* out) noexcept {
    assert(th != nullptr);
    assert(out != nullptr);
    *out = th->write_uncertain ? th->candidate : CommitCandidate{};
    return th->write_uncertain;
}
bool table_set_file_write_io(TableHandle* th, const FileWriteIO* config) noexcept {
    assert(th != nullptr && config != nullptr);
    assert(th->arena != nullptr);
    if (!th->file_io_configurable || th->file_io != nullptr || th->write_uncertain ||
        !file_io_valid(config)) return false;
    const FileWriteIO copied = *config;
    auto* state = th->arena->allocate_array<FileWriteState>(1);
    if (state == nullptr) return false;
    state->hash_workspace = th->arena->allocate_array<uint8_t>(kStageHashBytes);
    if (state->hash_workspace == nullptr) return false;
    state->config = copied;
    state->receipt = {};
    state->has_receipt = false;
    th->file_io = state;
    th->file_io_configurable = false;
    return true;
}
bool table_staged_data_file(const TableHandle* th, StagedDataFile* out) noexcept {
    assert(th != nullptr && out != nullptr);
    assert(th->arena != nullptr);
    assert(th->os != nullptr && th->os->vt != nullptr);
    const detail::ObjectSpan output{out, sizeof(*out)};
    const detail::ObjectSpan protected_spans[]{{th, sizeof(*th)},
        {th->file_io, th->file_io == nullptr ? 0u : sizeof(FileWriteState)},
        {th->arena, sizeof(Arena)},
        {th->file_io == nullptr ? nullptr : th->file_io->hash_workspace, th->file_io == nullptr ? 0u : kStageHashBytes},
        {th->os, sizeof(ObjectStore)}, {th->os->vt, sizeof(ObjectStoreVT)},
        {th->os->impl, th->os->impl == nullptr ? 0u : 1u}};
    if (!detail::object_io_admit(nullptr, 0, &output, 1, nullptr, protected_spans, 7)) return false;
    const bool found = th->file_io != nullptr && th->file_io->has_receipt;
    *out = found ? th->file_io->receipt : StagedDataFile{};
    return found;
}
bool table_uncertain_data_file(const TableHandle* th, StagedDataFile* out) noexcept {
    assert(th != nullptr && out != nullptr);
    assert(th->arena != nullptr);
    assert(th->os != nullptr && th->os->vt != nullptr);
    if (th->write_uncertain) return table_staged_data_file(th, out);
    const detail::ObjectSpan output{out, sizeof(*out)};
    const detail::ObjectSpan protected_spans[]{{th, sizeof(*th)},
        {th->file_io, th->file_io == nullptr ? 0u : sizeof(FileWriteState)},
        {th->arena, sizeof(Arena)},
        {th->file_io == nullptr ? nullptr : th->file_io->hash_workspace, th->file_io == nullptr ? 0u : kStageHashBytes},
        {th->os, sizeof(ObjectStore)}, {th->os->vt, sizeof(ObjectStoreVT)},
        {th->os->impl, th->os->impl == nullptr ? 0u : 1u}};
    if (!detail::object_io_admit(nullptr, 0, &output, 1, nullptr, protected_spans, 7)) return false;
    *out = {};
    return false;
}
const Metadata* table_metadata(const TableHandle* th) noexcept {
    assert(th != nullptr);
    return &th->meta;
}
