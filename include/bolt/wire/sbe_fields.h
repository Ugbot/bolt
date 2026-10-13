#pragma once
#include "sbe.h"
#include <cmath>

namespace bolt::wire::sbe {
// Caller supplies schema-derived offset/sinceVersion. A field absent in an
// older acting version is distinct from a present null and from truncation.
template <ByteOrder Order, typename T>
BOLT_FORCE_INLINE Status read_field(const Block& block, std::uint32_t offset,
                         std::uint16_t since_version, T& out) noexcept {
    static_assert(is_scalar<T>);
    if (block.bytes.data == nullptr && block.bytes.size != 0) return Status::InvalidInput;
    if (block.acting_version < since_version) return Status::Absent;
    ByteReader reader{block.bytes.data, block.bytes.size, offset};
    if (!read_scalar<Order>(reader, out)) return Status::Truncated;
    assert(reader.offset <= block.bytes.size);
    assert(reader.offset - offset == sizeof(T));
    return Status::Ok;
}

template <ByteOrder Order, typename T>
BOLT_FORCE_INLINE Status read_optional(const Block& block, std::uint32_t offset,
                            std::uint16_t since_version, T null_value, T& out) noexcept {
    T next{};
    const auto status = read_field<Order>(block, offset, since_version, next);
    if (status != Status::Ok) return status;
    bool is_null = next == null_value;
    if constexpr (std::is_floating_point_v<T>)
        is_null = std::isnan(null_value) ? std::isnan(next) : is_null;
    if (is_null) return Status::Null;
    assert(block.acting_version >= since_version);
    assert(offset <= block.bytes.size);
    out = next;
    return Status::Ok;
}

// Character arrays/composites remain borrowed views, including embedded NUL.
BOLT_FORCE_INLINE Status read_fixed_bytes(const Block& block, std::uint32_t offset,
                               std::uint32_t length, std::uint16_t since_version,
                               ByteView& out) noexcept {
    if (block.bytes.data == nullptr && block.bytes.size != 0) return Status::InvalidInput;
    if (block.acting_version < since_version) return Status::Absent;
    ByteReader reader{block.bytes.data, block.bytes.size, offset};
    if (!read_bytes(reader, length, out)) return Status::Truncated;
    assert(out.size == length);
    assert(reader.offset <= block.bytes.size);
    return Status::Ok;
}

// Length type is schema-defined. Caps are caller-selected at compile time.
// Length prefixes and data are committed together; any failure preserves both
// cursor and output. No implicit UTF-8 validation or string allocation.
template <ByteOrder Order, typename Length, std::uint32_t MaxBytes>
BOLT_FORCE_INLINE Status read_data(ByteReader& reader, ByteView& out) noexcept {
    static_assert(std::is_unsigned_v<Length> && !std::is_same_v<Length, bool>);
    auto next = reader;
    Length length{};
    if (!read_scalar<Order>(next, length)) return Status::Truncated;
    if (length > MaxBytes) return Status::LimitExceeded;
    ByteView value{};
    if (!read_bytes(next, static_cast<std::uint32_t>(length), value)) return Status::Truncated;
    assert(value.size <= MaxBytes);
    assert(next.offset <= next.size);
    reader = next;
    out = value;
    return Status::Ok;
}
} // namespace bolt::wire::sbe
