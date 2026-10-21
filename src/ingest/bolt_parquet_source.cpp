#include "bolt/ingest/bolt_parquet_source.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace bolt::ingest::parquet {
namespace {
using Status = ParquetSourceStatus;

struct Span { const void* data; size_t bytes; };
bool valid_span(Span span) noexcept {
    if (span.bytes == 0) return true;
    return span.data != nullptr &&
        span.bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(span.data);
}
bool overlaps(Span a, Span b) noexcept {
    assert(valid_span(a));
    assert(valid_span(b));
    if (a.bytes == 0 || b.bytes == 0) return false;
    const auto first = reinterpret_cast<uintptr_t>(a.data);
    const auto second = reinterpret_cast<uintptr_t>(b.data);
    return first <= second ? second - first < a.bytes : first - second < b.bytes;
}
bool disjoint(Span output, const Span* inputs, uint32_t count) noexcept {
    assert(inputs != nullptr);
    assert(count <= 10u);
    if (!valid_span(output)) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (!valid_span(inputs[i]) || overlaps(output, inputs[i])) return false;
    return true;
}
bool span_bytes(uint64_t count, size_t element, size_t* out) noexcept {
    assert(out != nullptr);
    assert(element > 0);
    if (count > SIZE_MAX / element) return false;
    *out = static_cast<size_t>(count) * element;
    return true;
}

bool metadata_spans(const ParquetByteSource* source, const ParquetSourceWorkspace* w,
                     const Arena* arena, const PqMeta* out) noexcept {
    assert(source != nullptr && w != nullptr);
    assert(out != nullptr);
    Span inputs[]{{source, sizeof(*source)}, {w, sizeof(*w)}, {arena, sizeof(*arena)},
                  {w->bytes, static_cast<size_t>(w->capacity)}};
    return disjoint(inputs[3], inputs, 3) && disjoint({out, sizeof(*out)}, inputs, 4);
}

bool projection_spans(const ParquetByteSource* source, const ParquetSourceWorkspace* w,
    const PqMeta* meta, const uint16_t* indices, uint32_t count,
    const Arena* arena, const BoltColumn* out, const int64_t* rows) noexcept {
    assert(source != nullptr && w != nullptr && meta != nullptr);
    assert(out != nullptr && rows != nullptr);
    if (!valid_span({meta, sizeof(*meta)}) || count > kMaxColumns ||
        meta->n_columns > kMaxColumns || meta->n_columns > meta->columns_cap ||
        meta->n_chunks > meta->chunks_cap) return false;
    size_t chunk_bytes = 0;
    if (!span_bytes(meta->n_chunks, sizeof(PqChunk), &chunk_bytes)) return false;
    Span inputs[]{{source, sizeof(*source)}, {w, sizeof(*w)}, {meta, sizeof(*meta)},
        {indices, sizeof(*indices) * count}, {meta->columns, sizeof(PqColumn) * meta->n_columns},
        {meta->chunks, chunk_bytes}, {arena, sizeof(*arena)},
        {w->bytes, static_cast<size_t>(w->capacity)}, {out, sizeof(*out) * count}};
    return disjoint(inputs[7], inputs, 7) && disjoint(inputs[8], inputs, 8) &&
        disjoint({rows, sizeof(*rows)}, inputs, 9);
}

bool admitted(const ParquetByteSource* source, const ParquetSourceWorkspace* w) noexcept {
    assert(source != nullptr);
    assert(w != nullptr);
    if (!valid_span({source, sizeof(*source)}) || !valid_span({w, sizeof(*w)})) return false;
    return source->read_at != nullptr && w->bytes != nullptr && w->capacity > 0 &&
        w->capacity <= SIZE_MAX && w->request_bytes > 0 &&
        w->request_bytes <= kParquetSourceMaxReadBytes &&
        valid_span({w->bytes, static_cast<size_t>(w->capacity)});
}

Status read_exact(const ParquetByteSource& source, const ParquetSourceWorkspace& w,
                  uint64_t offset, uint64_t length, uint8_t* dst) noexcept {
    assert(source.read_at != nullptr && dst != nullptr);
    assert(w.request_bytes > 0 && w.request_bytes <= kParquetSourceMaxReadBytes);
    if (offset > source.size || length > source.size - offset || length > SIZE_MAX)
        return Status::BadArg;
    // Each iteration advances by a positive request; the caller bounds length
    // by its finite workspace (or the 4/8-byte magic reads).
    for (uint64_t done = 0; done < length; ) {
        const auto take = static_cast<uint32_t>(std::min<uint64_t>(length - done, w.request_bytes));
        const Status status = source.read_at(source.context, offset + done, take,
                                             dst + static_cast<size_t>(done));
        if (status != Status::Ok) return status;
        done += take;
    }
    return Status::Ok;
}

bool chunk_extent(const PqChunk& chunk, uint64_t limit, uint64_t* offset,
                   uint64_t* length) noexcept {
    assert(offset != nullptr);
    assert(length != nullptr);
    const int64_t start = chunk.dictionary_page_offset > 0
        ? chunk.dictionary_page_offset : chunk.data_page_offset;
    if (start < 4 || chunk.total_compressed_size <= 0 || chunk.data_page_offset < start)
        return false;
    const auto first = static_cast<uint64_t>(start);
    const auto bytes = static_cast<uint64_t>(chunk.total_compressed_size);
    if (first > limit || bytes > limit - first ||
        static_cast<uint64_t>(chunk.data_page_offset) - first >= bytes) return false;
    *offset = first;
    *length = bytes;
    return true;
}

bool metadata_extents(const PqMeta& meta, uint64_t footer) noexcept {
    assert(meta.columns != nullptr);
    assert(meta.chunks != nullptr);
    if (meta.n_columns == 0 || meta.n_columns > kMaxColumns ||
        meta.n_row_groups > kPqMaxRowGroups || meta.n_chunks > meta.chunks_cap ||
        meta.n_columns > meta.columns_cap) return false;
    for (uint32_t i = 0; i < meta.n_chunks; ++i) {
        const auto& chunk = meta.chunks[i];
        if (chunk.num_values < 0 || chunk.dictionary_page_offset < 0) return false;
        if (chunk.num_values == 0 && chunk.total_compressed_size == 0) continue;
        uint64_t offset = 0, length = 0;
        if (!chunk_extent(chunk, footer, &offset, &length)) return false;
    }
    for (uint32_t i = 0; i < meta.n_row_groups; ++i) {
        const auto& group = meta.row_groups[i];
        if (group.num_rows < 0 || group.chunk_off > meta.n_chunks ||
            meta.n_columns > meta.n_chunks - group.chunk_off) return false;
    }
    return true;
}

Status projection_extents(const ParquetByteSource& source, const ParquetSourceWorkspace& w,
    const PqMeta& meta, uint32_t row_group, const uint16_t* indices, uint32_t count) noexcept {
    assert(indices != nullptr);
    assert(count > 0);
    if (source.size < 12 || meta.columns == nullptr || meta.chunks == nullptr ||
        meta.n_columns > kMaxColumns || meta.n_columns > meta.columns_cap ||
        meta.n_chunks > meta.chunks_cap || meta.n_row_groups > kPqMaxRowGroups ||
        row_group >= meta.n_row_groups || count > meta.n_columns) return Status::BadArg;
    const auto& group = meta.row_groups[row_group];
    if (group.num_rows < 0 || group.chunk_off > meta.n_chunks ||
        meta.n_columns > meta.n_chunks - group.chunk_off) return Status::DecodeFailed;
    for (uint32_t i = 0; i < count; ++i) {
        if (indices[i] >= meta.n_columns) return Status::BadArg;
        const auto& chunk = meta.chunks[group.chunk_off + indices[i]];
        if (chunk.num_values != group.num_rows) return Status::DecodeFailed;
        if (group.num_rows == 0 && chunk.total_compressed_size == 0) continue;
        uint64_t offset = 0, length = 0;
        if (!chunk_extent(chunk, source.size - 8u, &offset, &length)) return Status::DecodeFailed;
        if (length > w.capacity) return Status::Capacity;
    }
    return Status::Ok;
}
}  // namespace

template<bool FlatStrict>
static ParquetSourceStatus read_meta_source(const ParquetByteSource* source,
    ParquetSourceWorkspace* w, Arena* arena, PqMeta* out) noexcept {
    assert(source != nullptr && w != nullptr);
    assert(arena != nullptr && out != nullptr);
    if (!admitted(source, w) || !metadata_spans(source, w, arena, out)) return Status::BadArg;
    if (arena->overlaps_backing(w->bytes, static_cast<size_t>(w->capacity))) return Status::BadArg;
    if (source->size < 12) return Status::DecodeFailed;
    uint8_t magic[4]{}, tail[8]{};
    Status status = read_exact(*source, *w, 0, sizeof(magic), magic);
    if (status != Status::Ok) return status;
    status = read_exact(*source, *w, source->size - sizeof(tail), sizeof(tail), tail);
    if (status != Status::Ok) return status;
    if (std::memcmp(magic, "PAR1", 4) != 0 || std::memcmp(tail + 4, "PAR1", 4) != 0)
        return Status::DecodeFailed;
    const uint32_t length = static_cast<uint32_t>(tail[0]) |
        (static_cast<uint32_t>(tail[1]) << 8u) | (static_cast<uint32_t>(tail[2]) << 16u) |
        (static_cast<uint32_t>(tail[3]) << 24u);
    if (length == 0 || length > source->size - 12u) return Status::DecodeFailed;
    if (length > w->capacity) return Status::Capacity;
    const uint64_t footer = source->size - 8u - length;
    status = read_exact(*source, *w, footer, length, w->bytes);
    if (status != Status::Ok) return status;
    auto* staged = arena->allocate_array<PqMeta>(1);
    if (staged == nullptr) return Status::Capacity;
    *staged = PqMeta{};
    status = parquet_read_meta_slice(w->bytes, length, arena, staged);
    if (status != Status::Ok) return status;
    if constexpr (FlatStrict) {
        const auto parsed=pq_parse_file_meta_strict(w->bytes,length,staged,kPqMaxRowGroups);
        if (parsed==PqMetaParseStatus::Capacity) return Status::Capacity;
        if (parsed!=PqMetaParseStatus::Ok) return Status::DecodeFailed;
    }
    if (!metadata_extents(*staged, footer)) return Status::DecodeFailed;
    *out = *staged;
    return Status::Ok;
}

ParquetSourceStatus parquet_read_meta_source(const ParquetByteSource* source,
    ParquetSourceWorkspace* workspace, Arena* metadata, PqMeta* out) noexcept {
    return read_meta_source<false>(source,workspace,metadata,out);
}

ParquetSourceStatus parquet_read_meta_source_flat_strict(const ParquetByteSource* source,
    ParquetSourceWorkspace* workspace, Arena* metadata, PqMeta* out) noexcept {
    return read_meta_source<true>(source,workspace,metadata,out);
}

ParquetSourceStatus parquet_read_row_group_cols_source(const ParquetByteSource* source,
    ParquetSourceWorkspace* w, const PqMeta* meta, uint32_t row_group,
    const uint16_t* indices, uint32_t count, Arena* arena,
    BoltColumn* out, int64_t* rows) noexcept {
    assert(source != nullptr && w != nullptr && meta != nullptr);
    assert(arena != nullptr && out != nullptr && rows != nullptr);
    if (!admitted(source, w) || indices == nullptr || count == 0) return Status::BadArg;
    if (!projection_spans(source, w, meta, indices, count, arena, out, rows)) return Status::BadArg;
    if (arena->overlaps_backing(w->bytes, static_cast<size_t>(w->capacity))) return Status::BadArg;
    Status status = projection_extents(*source, *w, *meta, row_group, indices, count);
    if (status == Status::BadArg) return status;
    *rows = 0;
    if (status != Status::Ok) return status;
    auto* staged = arena->allocate_array<BoltColumn>(count);
    if (staged == nullptr) return Status::Capacity;
    const auto& group = meta->row_groups[row_group];
    for (uint32_t i = 0; i < count; ++i) {
        const auto& chunk = meta->chunks[group.chunk_off + indices[i]];
        uint64_t offset = 0, length = 0;
        if (group.num_rows != 0 || chunk.total_compressed_size != 0) {
            if (!chunk_extent(chunk, source->size - 8u, &offset, &length)) return Status::DecodeFailed;
            status = read_exact(*source, *w, offset, length, w->bytes);
            if (status != Status::Ok) return status;
        }
        int64_t decoded = 0;
        if (!parquet_read_col_region(w->bytes, length, offset, meta, row_group, indices[i],
                                     arena, &staged[i], &decoded) || decoded != group.num_rows)
            return Status::DecodeFailed;
    }
    std::memcpy(out, staged, sizeof(BoltColumn) * count);
    *rows = group.num_rows;
    return Status::Ok;
}
}  // namespace bolt::ingest::parquet
