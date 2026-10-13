#pragma once
#include "bolt/bolt_port.h"
#include "byte_reader.h"
#include <limits>

namespace bolt::wire::sbe {
// SBE is schema driven: these are bounded wire primitives, not an XML parser
// or a template registry. A transport must supply exactly one bounded message.
enum class ByteOrder : std::uint8_t { Little, Big };
enum class Status : std::uint8_t {
    Ok, End, Absent, Null, Truncated, InvalidInput, LimitExceeded,
    UnexpectedSchema, UnexpectedTemplate, UnsupportedVersion
};
struct Header {
    std::uint16_t block_length = 0;
    std::uint16_t template_id = 0;
    std::uint16_t schema_id = 0;
    std::uint16_t version = 0;
};
struct Expected {
    std::uint16_t schema_id = 0;
    std::uint16_t template_id = 0;
    std::uint16_t minimum_block_length = 0;
    std::uint16_t minimum_version = 0;
    std::uint16_t maximum_version = 0;
};
struct Block {
    ByteView bytes{};
    std::uint16_t acting_version = 0;
};
struct Message {
    Header header{};
    Block root{};
    ByteReader tail{};
};

template <typename T> inline constexpr bool is_scalar =
    (std::is_integral_v<T> && !std::is_same_v<T, bool>) ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

// Compile-time byte order; no unaligned typed dereference or aliasing. The
// common LE integral path reuses Bolt's transactional bounded byte reader.
template <ByteOrder Order, typename T>
BOLT_FORCE_INLINE bool read_scalar(ByteReader& reader, T& out) noexcept {
    static_assert(is_scalar<T>);
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    static_assert(std::is_integral_v<T> || std::numeric_limits<T>::is_iec559);
    if constexpr (Order == ByteOrder::Little && std::is_integral_v<T>) {
        return read_le(reader, out);
    } else {
        constexpr auto width = static_cast<std::uint32_t>(sizeof(T));
        if (!byte_reader_valid(reader) || width > reader.size - reader.offset) return false;
        assert(reader.data != nullptr);
        std::uint64_t bits = 0;
        for (std::uint32_t i = 0; i < width; ++i) {
            constexpr bool little = Order == ByteOrder::Little;
            const auto shift = (little ? i : width - i - 1) * 8;
            bits |= std::uint64_t(reader.data[reader.offset + i]) << shift;
        }
        if constexpr (sizeof(T) == 1) out = std::bit_cast<T>(static_cast<std::uint8_t>(bits));
        if constexpr (sizeof(T) == 2) out = std::bit_cast<T>(static_cast<std::uint16_t>(bits));
        if constexpr (sizeof(T) == 4) out = std::bit_cast<T>(static_cast<std::uint32_t>(bits));
        if constexpr (sizeof(T) == 8) out = std::bit_cast<T>(bits);
        reader.offset += width;
        assert(reader.offset <= reader.size);
        return true;
    }
}

// Common four-uint16 SBE header only. Other header composites are decoded by
// the schema's generated adapter using read_scalar. No CME size prefix here.
template <ByteOrder Order>
BOLT_FORCE_INLINE Status read_header(ByteReader& reader, Header& out) noexcept {
    auto next = reader;
    Header header{};
    if (!read_scalar<Order>(next, header.block_length) ||
        !read_scalar<Order>(next, header.template_id) ||
        !read_scalar<Order>(next, header.schema_id) ||
        !read_scalar<Order>(next, header.version)) return Status::Truncated;
    assert(next.offset <= next.size);
    assert(next.offset - reader.offset == 8);
    reader = next;
    out = header;
    return Status::Ok;
}

template <ByteOrder Order, std::uint32_t MaxBytes = 16 * 1024 * 1024>
BOLT_FORCE_INLINE Status open_message(ByteView bytes, const Expected& expected, Message& out) noexcept {
    static_assert(MaxBytes >= 8);
    if ((bytes.data == nullptr && bytes.size != 0) ||
        expected.minimum_version > expected.maximum_version) return Status::InvalidInput;
    if (bytes.size > MaxBytes) return Status::LimitExceeded;
    Message next{};
    next.tail = {bytes.data, bytes.size, 0};
    const auto status = read_header<Order>(next.tail, next.header);
    if (status != Status::Ok) return status;
    if (next.header.schema_id != expected.schema_id) return Status::UnexpectedSchema;
    if (next.header.template_id != expected.template_id) return Status::UnexpectedTemplate;
    if (next.header.version < expected.minimum_version ||
        next.header.version > expected.maximum_version) return Status::UnsupportedVersion;
    if (next.header.block_length < expected.minimum_block_length) return Status::InvalidInput;
    if (!read_bytes(next.tail, next.header.block_length, next.root.bytes)) return Status::Truncated;
    next.root.acting_version = next.header.version;
    assert(next.root.bytes.size == next.header.block_length);
    assert(next.tail.offset <= next.tail.size);
    out = next;
    return Status::Ok;
}

// Call only after the schema-specific decoder consumed every declared group
// and data field. An unknown tail is not silently treated as decoded data.
BOLT_FORCE_INLINE Status finish_message(const ByteReader& tail) noexcept {
    if (!byte_reader_valid(tail)) return Status::InvalidInput;
    assert(tail.offset <= tail.size);
    assert(tail.data != nullptr || tail.size == 0);
    return tail.offset == tail.size ? Status::Ok : Status::InvalidInput;
}
static_assert(std::is_trivially_copyable_v<Message>);
} // namespace bolt::wire::sbe
