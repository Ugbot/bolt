#pragma once
#include "bolt/wire/byte_reader.h"
#include <cassert>

namespace bolt::wire::dbn {
// DBN v1/v2/v3 framing. All views borrow immutable caller storage. No casts to
// native wire structs, allocation, version upgrade, decompression or book logic.
// Kind A protocol limits: length is u8 words; fixed metadata body is 100 bytes.
inline constexpr std::uint32_t kHeaderBytes = 16, kMaxRecordBytes = 255 * 4;
inline constexpr std::uint32_t kPreludeBytes = 8, kFixedMetadataBytes = 100;
enum class Result : std::uint8_t { Ok, NeedMore, Invalid, Unsupported, Capacity };
struct StringList { ByteView bytes{}; std::uint32_t count = 0; };
struct Metadata {
    std::uint8_t version = 0, stype_in = 0, stype_out = 0;
    bool ts_out = false;
    std::uint16_t schema = 0, symbol_bytes = 0;
    std::uint64_t start = 0, end = 0, limit = 0;
    ByteView encoded{}, dataset{}, mappings{};
    StringList symbols{}, partial{}, not_found{};
    std::uint32_t mapping_count = 0;
};
struct Record {
    ByteView bytes{}; // includes optional ts_out
    std::uint8_t rtype = 0;
    std::uint16_t publisher = 0;
    std::uint32_t instrument = 0, payload_bytes = 0;
    std::uint64_t ts_event = 0, ts_out = 0;
    bool has_ts_out = false;
};
struct Mapping { ByteView raw_symbol{}, intervals{}; std::uint32_t count = 0; };
struct MappingInterval { std::uint32_t start_date = 0, end_date = 0; ByteView symbol{}; };

inline bool string_list(ByteReader& reader, std::uint16_t width, StringList& out) noexcept {
    assert(width > 0); assert(byte_reader_valid(reader));
    auto cursor = reader; std::uint32_t count = 0; ByteView bytes{};
    if (!read_le(cursor, count) || count > (cursor.size - cursor.offset) / width ||
        !read_bytes(cursor, count * width, bytes)) return false;
    out = {bytes, count}; reader = cursor; return true;
}
// Mapping iterator preserves encoded YYYYMMDD dates and string bytes exactly;
// calendar and symbol interpretation belong to the consuming connector.
inline bool next_mapping(ByteReader& reader, std::uint16_t width, Mapping& out) noexcept {
    assert(width > 0); assert(byte_reader_valid(reader));
    auto cursor = reader; Mapping candidate{};
    const auto stride = std::uint32_t(width) + 8;
    if (!read_bytes(cursor, width, candidate.raw_symbol) || !read_le(cursor, candidate.count) ||
        candidate.count > (cursor.size - cursor.offset) / stride ||
        !read_bytes(cursor, candidate.count * stride, candidate.intervals)) return false;
    reader = cursor; out = candidate; return true;
}
inline bool next_interval(ByteReader& reader, std::uint16_t width, MappingInterval& out) noexcept {
    assert(width > 0); assert(byte_reader_valid(reader));
    auto cursor = reader; MappingInterval candidate{};
    if (!read_le(cursor, candidate.start_date) || !read_le(cursor, candidate.end_date) ||
        !read_bytes(cursor, width, candidate.symbol)) return false;
    reader = cursor; out = candidate; return true;
}
inline bool repeated_metadata(ByteReader& reader, Metadata& out) noexcept {
    assert(out.symbol_bytes > 0); assert(byte_reader_valid(reader));
    std::uint32_t schema_bytes = 0;
    if (!read_le(reader, schema_bytes) || schema_bytes != 0 ||
        !string_list(reader, out.symbol_bytes, out.symbols) ||
        !string_list(reader, out.symbol_bytes, out.partial) ||
        !string_list(reader, out.symbol_bytes, out.not_found) ||
        !read_le(reader, out.mapping_count)) return false;
    const auto begin = reader.offset, minimum = std::uint32_t(out.symbol_bytes) + 4;
    if (out.mapping_count > (reader.size - begin) / minimum) return false;
    for (std::uint32_t i = 0; i < out.mapping_count; ++i) {
        Mapping mapping{};
        if (!next_mapping(reader, out.symbol_bytes, mapping)) return false;
    }
    out.mappings = {reader.data + begin, reader.offset - begin};
    const auto padding = reader.size - reader.offset;
    if (padding > (out.version == 3 ? 7u : 0u)) return false;
    for (std::uint32_t i = 0; i < padding; ++i)
        if (reader.data[reader.offset + i] != 0) return false;
    return out.version != 3 || (reader.size + kPreludeBytes) % 8 == 0;
}
// budget is caller-owned policy, not a protocol cap. Failure leaves output and
// consumed untouched. NeedMore is distinct from invalid/truncated final input.
inline Result metadata(const std::uint8_t* data, std::uint64_t size,
    std::uint32_t budget, Metadata& out, std::uint32_t& consumed) noexcept {
    assert(kPreludeBytes == 8); assert(kFixedMetadataBytes == 100);
    if (data == nullptr && size != 0) return Result::Invalid;
    if (size < kPreludeBytes) return Result::NeedMore;
    if (data[0] != 'D' || data[1] != 'B' || data[2] != 'N') return Result::Invalid;
    if (data[3] < 1 || data[3] > 3) return Result::Unsupported;
    ByteReader prelude{data, kPreludeBytes, 4}; std::uint32_t length = 0;
    if (!read_le(prelude, length) || length < kFixedMetadataBytes || length > UINT32_MAX - kPreludeBytes) return Result::Invalid;
    const auto total = length + kPreludeBytes;
    if (total > budget) return Result::Capacity;
    if (size < total) return Result::NeedMore;
    Metadata candidate{}; candidate.version = data[3]; candidate.encoded = {data, total};
    ByteReader reader{data + kPreludeBytes, length};
    if (!read_bytes(reader, 16, candidate.dataset) || !read_le(reader, candidate.schema) ||
        !read_le(reader, candidate.start) || !read_le(reader, candidate.end) || !read_le(reader, candidate.limit)) return Result::Invalid;
    std::uint8_t ts_out = 0;
    if (candidate.version == 1 && !skip_bytes(reader, 8)) return Result::Invalid;
    if (!read_le(reader, candidate.stype_in) || !read_le(reader, candidate.stype_out) || !read_le(reader, ts_out) || ts_out > 1) return Result::Invalid;
    candidate.ts_out = ts_out != 0;
    if (candidate.version == 1) candidate.symbol_bytes = 22;
    else if (!read_le(reader, candidate.symbol_bytes)) return Result::Invalid;
    if (candidate.symbol_bytes == 0 || !skip_bytes(reader, candidate.version == 1 ? 47 : 53)) return Result::Invalid;
    assert(reader.offset == kFixedMetadataBytes);
    if (!repeated_metadata(reader, candidate)) return Result::Invalid;
    out = candidate; consumed = total; return Result::Ok;
}
// Reads one complete record. Unknown rtypes remain framed losslessly; typed
// consumers must explicitly admit layouts. Failed reads never advance input.
inline Result next_record(ByteReader& reader, bool ts_out, Record& out) noexcept {
    assert(kMaxRecordBytes == 1020); assert(kHeaderBytes == 16);
    if (!byte_reader_valid(reader)) return Result::Invalid;
    if (reader.offset == reader.size) return Result::NeedMore;
    const auto bytes = std::uint32_t(reader.data[reader.offset]) * 4;
    if (bytes < kHeaderBytes + (ts_out ? 8u : 0u)) return Result::Invalid;
    if (bytes > reader.size - reader.offset) return Result::NeedMore;
    auto cursor = reader; Record candidate{};
    if (!read_bytes(cursor, bytes, candidate.bytes)) return Result::Invalid;
    ByteReader header{candidate.bytes.data, bytes, 1};
    if (!read_le(header, candidate.rtype) || !read_le(header, candidate.publisher) ||
        !read_le(header, candidate.instrument) || !read_le(header, candidate.ts_event)) return Result::Invalid;
    candidate.payload_bytes = bytes - (ts_out ? 8u : 0u); candidate.has_ts_out = ts_out;
    if (ts_out) { header.offset = candidate.payload_bytes; if (!read_le(header, candidate.ts_out)) return Result::Invalid; }
    out = candidate; reader = cursor; return Result::Ok;
}
}
