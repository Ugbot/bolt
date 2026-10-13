#pragma once
#include "bolt/bolt_port.h"
#include <bit>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace bolt::wire {
// Borrowed immutable bytes. Caller supplies actual storage of at least size
// bytes. Failed operations preserve both the cursor and destination.
struct ByteReader {
    const std::uint8_t* data = nullptr;
    std::uint32_t size = 0;
    std::uint32_t offset = 0;
};
struct ByteView {
    const std::uint8_t* data = nullptr;
    std::uint32_t size = 0;
};

BOLT_FORCE_INLINE bool byte_reader_valid(const ByteReader& reader) noexcept {
    return reader.offset <= reader.size && (reader.data != nullptr || reader.size == 0);
}

BOLT_FORCE_INLINE bool read_bytes(ByteReader& reader, std::uint32_t count, ByteView& out) noexcept {
    if (!byte_reader_valid(reader) || count > reader.size - reader.offset) return false;
    out = {reader.data == nullptr ? nullptr : reader.data + reader.offset, count};
    reader.offset += count;
    assert(reader.offset <= reader.size);
    assert(out.data != nullptr || out.size == 0);
    return true;
}

BOLT_FORCE_INLINE bool skip_bytes(ByteReader& reader, std::uint32_t count) noexcept {
    ByteView ignored{};
    if (!read_bytes(reader, count, ignored)) return false;
    assert(reader.offset <= reader.size);
    assert(ignored.size == count);
    return true;
}

template <typename T> BOLT_FORCE_INLINE bool read_le(ByteReader& reader, T& out) noexcept {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    constexpr auto width = static_cast<std::uint32_t>(sizeof(T));
    if (!byte_reader_valid(reader) || width > reader.size - reader.offset) return false;
    assert(reader.data != nullptr);
    std::uint64_t bits = 0;
    for (std::uint32_t i = 0; i < width; ++i)
        bits |= std::uint64_t(reader.data[reader.offset + i]) << (i * 8);
    using Unsigned = std::make_unsigned_t<T>;
    out = std::bit_cast<T>(static_cast<Unsigned>(bits));
    reader.offset += width;
    assert(reader.offset <= reader.size);
    return true;
}
}
