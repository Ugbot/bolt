#pragma once
#include "sbe.h"
#include <cstddef>

namespace bolt::wire::sbe {
// Schema-neutral append cursor. Storage is caller owned and remains fixed.
struct Writer { std::uint8_t* data = nullptr; std::uint32_t size = 0, offset = 0; };
struct WriteBlock {
    std::uint8_t* data = nullptr;
    std::uint32_t size = 0;
    std::uint16_t acting_version = 0;
};
namespace writer_detail {
inline bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) noexcept {
    assert(!an || a); assert(!bn || b);
    if (!an || !bn) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
inline bool valid(const Writer& w) noexcept {
    return (w.data || !w.size) && w.offset <= w.size &&
        !overlap(&w, sizeof(w), w.data, w.size);
}
inline bool valid(const WriteBlock& b) noexcept {
    return (b.data || !b.size) && !overlap(&b, sizeof(b), b.data, b.size);
}
template<typename T, typename V>
inline Status convert(V value, T& out) noexcept {
    static_assert(is_scalar<T> && is_scalar<V>);
    static_assert(sizeof(T) <= 8 && sizeof(V) <= 8);
    if constexpr (std::is_integral_v<T> && std::is_integral_v<V>) {
        if constexpr (std::is_signed_v<V>) {
            if (value < 0) {
                if constexpr (!std::is_signed_v<T>) return Status::LimitExceeded;
                else if (static_cast<std::int64_t>(value) < std::numeric_limits<T>::min()) return Status::LimitExceeded;
                out = static_cast<T>(value); return Status::Ok;
            }
        }
        if (static_cast<std::uint64_t>(value) > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
            return Status::LimitExceeded;
        out = static_cast<T>(value);
    } else if constexpr (std::is_same_v<T, V>) out = value;
    else return Status::InvalidInput; // Floating conversions require an explicit caller decision.
    assert(sizeof(out) == sizeof(T)); assert(is_scalar<T>);
    return Status::Ok;
}
// Public shape arguments use deduced input types too. bool is a control,
// never an encoded integer length/version/count; floating inputs are refused.
template<typename T, typename V>
inline Status shape(V value, T& out) noexcept {
    static_assert(std::is_unsigned_v<T> && !std::is_same_v<T, bool>);
    if constexpr (std::is_integral_v<V> && !std::is_same_v<V, bool>) {
        const auto status = convert(value, out);
        if (status != Status::Ok) return status;
        assert(static_cast<std::uint64_t>(out) == static_cast<std::uint64_t>(value));
        assert(static_cast<std::uint64_t>(out) <= std::numeric_limits<T>::max());
        return Status::Ok;
    } else return Status::InvalidInput;
}
template<ByteOrder Order, typename T>
inline void store(std::uint8_t* out, T value) noexcept {
    static_assert(is_scalar<T>);
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
    static_assert(std::is_integral_v<T> || std::numeric_limits<T>::is_iec559);
    static_assert(Order == ByteOrder::Little || Order == ByteOrder::Big);
    assert(out != nullptr); assert(sizeof(T) <= 8);
    std::uint64_t bits = 0;
    if constexpr (sizeof(T) == 1) bits = std::bit_cast<std::uint8_t>(value);
    if constexpr (sizeof(T) == 2) bits = std::bit_cast<std::uint16_t>(value);
    if constexpr (sizeof(T) == 4) bits = std::bit_cast<std::uint32_t>(value);
    if constexpr (sizeof(T) == 8) bits = std::bit_cast<std::uint64_t>(value);
    for (std::uint32_t i = 0; i < sizeof(T); ++i) {
        const auto shift = (Order == ByteOrder::Little ? i : sizeof(T) - i - 1) * 8;
        out[i] = static_cast<std::uint8_t>(bits >> shift);
    }
}
inline Status block_range(const WriteBlock& block, std::uint32_t offset,
    std::uint32_t width, std::uint16_t since) noexcept {
    if (!valid(block)) return Status::InvalidInput;
    if (block.acting_version < since) return Status::Absent;
    if (offset > block.size || width > block.size - offset) return Status::Truncated;
    assert(offset <= block.size); assert(width <= block.size - offset);
    return Status::Ok;
}
} // namespace writer_detail
// An explicit WireT with a wider Value is checked before conversion; no scalar
// narrows at the function boundary. All non-Ok returns leave bytes/cursor intact.
template<ByteOrder Order, typename WireT, typename Value = WireT>
BOLT_FORCE_INLINE Status write_scalar(Writer& writer, Value value) noexcept {
    WireT converted{};
    const auto status = writer_detail::convert(value, converted);
    if (status != Status::Ok) return status;
    if (!writer_detail::valid(writer)) return Status::InvalidInput;
    if (sizeof(WireT) > writer.size - writer.offset) return Status::Truncated;
    assert(writer.data != nullptr); assert(writer.offset <= writer.size);
    writer_detail::store<Order>(writer.data + writer.offset, converted);
    writer.offset += sizeof(WireT);
    return Status::Ok;
}
template<ByteOrder Order>
BOLT_FORCE_INLINE Status write_header(Writer& writer, const Header& header) noexcept {
    if (!writer_detail::valid(writer) || writer_detail::overlap(&header, sizeof(header), writer.data, writer.size))
        return Status::InvalidInput;
    if (writer.size - writer.offset < 8) return Status::Truncated;
    auto* p = writer.data + writer.offset;
    assert(p != nullptr); assert(writer.size - writer.offset >= 8);
    writer_detail::store<Order>(p, header.block_length); writer_detail::store<Order>(p + 2, header.template_id);
    writer_detail::store<Order>(p + 4, header.schema_id); writer_detail::store<Order>(p + 6, header.version);
    writer.offset += 8;
    return Status::Ok;
}
// Reserve and zero the complete fixed block (including padding), then expose a
// bounded view. Result/cursor descriptors must not live inside writable storage.
template<typename LengthValue, typename VersionValue>
BOLT_FORCE_INLINE Status open_write_block(Writer& writer, LengthValue length_value,
    VersionValue version_value, WriteBlock& out) noexcept {
    std::uint32_t length = 0; std::uint16_t version = 0;
    const auto length_status = writer_detail::shape(length_value, length);
    if (length_status != Status::Ok) return length_status;
    const auto version_status = writer_detail::shape(version_value, version);
    if (version_status != Status::Ok) return version_status;
    if (!writer_detail::valid(writer) ||
        writer_detail::overlap(&out, sizeof(out), writer.data, writer.size) ||
        writer_detail::overlap(&out, sizeof(out), &writer, sizeof(writer))) return Status::InvalidInput;
    if (length > writer.size - writer.offset) return Status::Truncated;
    auto* p = writer.data ? writer.data + writer.offset : nullptr;
    assert(writer.offset <= writer.size); assert(length <= writer.size - writer.offset);
    for (std::uint32_t i = 0; i < length; ++i) p[i] = 0;
    out = {p, length, version}; writer.offset += length;
    return Status::Ok;
}
template<ByteOrder Order, typename WireT, typename Value = WireT,
    typename OffsetValue = std::uint32_t, typename SinceValue = std::uint16_t>
BOLT_FORCE_INLINE Status write_field(WriteBlock& block, OffsetValue offset_value,
    SinceValue since_value, Value value) noexcept {
    std::uint32_t offset = 0; std::uint16_t since_version = 0;
    const auto offset_status = writer_detail::shape(offset_value, offset);
    if (offset_status != Status::Ok) return offset_status;
    const auto since_status = writer_detail::shape(since_value, since_version);
    if (since_status != Status::Ok) return since_status;
    const auto range = writer_detail::block_range(block, offset, sizeof(WireT), since_version);
    if (range != Status::Ok) return range;
    WireT converted{};
    const auto status = writer_detail::convert(value, converted);
    if (status != Status::Ok) return status;
    assert(block.data != nullptr); assert(offset <= block.size);
    writer_detail::store<Order>(block.data + offset, converted);
    return Status::Ok;
}
// A successfully encoded null returns Ok. A present value colliding with the
// null sentinel is refused; floating NaN nulls match any NaN payload.
template<ByteOrder Order, typename WireT, typename Value = WireT,
    typename OffsetValue = std::uint32_t, typename SinceValue = std::uint16_t>
BOLT_FORCE_INLINE Status write_optional(WriteBlock& block, OffsetValue offset_value,
    SinceValue since_value, bool present, Value value, WireT null_value) noexcept {
    std::uint32_t offset = 0; std::uint16_t since_version = 0;
    const auto offset_status = writer_detail::shape(offset_value, offset);
    if (offset_status != Status::Ok) return offset_status;
    const auto since_status = writer_detail::shape(since_value, since_version);
    if (since_status != Status::Ok) return since_status;
    const auto range = writer_detail::block_range(block, offset, sizeof(WireT), since_version);
    if (range != Status::Ok) return range;
    WireT converted = null_value;
    if (present) {
        const auto status = writer_detail::convert(value, converted);
        if (status != Status::Ok) return status;
        bool null = converted == null_value;
        if constexpr (std::is_floating_point_v<WireT>)
            null = null || (converted != converted && null_value != null_value);
        if (null) return Status::InvalidInput;
    }
    assert(block.data != nullptr); assert(offset <= block.size);
    writer_detail::store<Order>(block.data + offset, converted);
    return Status::Ok;
}
template<typename OffsetValue, typename SinceValue>
BOLT_FORCE_INLINE Status write_fixed_bytes(WriteBlock& block, OffsetValue offset_value,
    ByteView input, SinceValue since_value) noexcept {
    std::uint32_t offset = 0; std::uint16_t since_version = 0;
    const auto offset_status = writer_detail::shape(offset_value, offset);
    if (offset_status != Status::Ok) return offset_status;
    const auto since_status = writer_detail::shape(since_value, since_version);
    if (since_status != Status::Ok) return since_status;
    if (!input.data && input.size) return Status::InvalidInput;
    const auto range = writer_detail::block_range(block, offset, input.size, since_version);
    if (range != Status::Ok) return range;
    if (writer_detail::overlap(input.data, input.size, block.data, block.size)) return Status::InvalidInput;
    assert(offset <= block.size); assert(input.size <= block.size - offset);
    for (std::uint32_t i = 0; i < input.size; ++i) block.data[offset + i] = input.data[i];
    return Status::Ok;
}
// Packed dimensions only. Generated adapters supply schema scalar types. Fixed
// entries must fit in remaining storage; nested data is admitted separately.
template<ByteOrder Order, typename BlockLength, typename Count,
    std::uint32_t MaxCount, std::uint32_t MaxDepth = 8,
    typename BlockValue = std::uint64_t, typename CountValue = std::uint64_t,
    typename DepthValue = std::uint32_t>
BOLT_FORCE_INLINE Status write_group_dimensions(Writer& writer, BlockValue block_length,
    CountValue count, DepthValue depth_value) noexcept {
    static_assert(std::is_unsigned_v<BlockLength> && std::is_unsigned_v<Count>);
    static_assert(!std::is_same_v<BlockLength, bool> && !std::is_same_v<Count, bool>);
    static_assert(sizeof(BlockLength) <= 4 && sizeof(Count) <= 4 && MaxDepth > 0);
    if (!writer_detail::valid(writer)) return Status::InvalidInput;
    std::uint32_t depth = 0;
    const auto depth_status = writer_detail::shape(depth_value, depth);
    if (depth_status != Status::Ok) return depth_status;
    if (!depth || depth > MaxDepth) return Status::LimitExceeded;
    BlockLength length{}; Count entries{};
    const auto length_status = writer_detail::shape(block_length, length);
    if (length_status != Status::Ok) return length_status;
    const auto count_status = writer_detail::shape(count, entries);
    if (count_status != Status::Ok) return count_status;
    if (entries > MaxCount) return Status::LimitExceeded;
    constexpr auto width = sizeof(BlockLength) + sizeof(Count);
    if (width > writer.size - writer.offset) return Status::Truncated;
    const auto remaining = writer.size - writer.offset - width;
    if (length && entries > remaining / length) return Status::Truncated;
    assert(writer.data != nullptr); assert(writer.offset <= writer.size);
    writer_detail::store<Order>(writer.data + writer.offset, length);
    writer_detail::store<Order>(writer.data + writer.offset + sizeof(BlockLength), entries);
    writer.offset += width;
    return Status::Ok;
}
template<ByteOrder Order, typename Length, std::uint32_t MaxBytes>
BOLT_FORCE_INLINE Status write_data(Writer& writer, ByteView input) noexcept {
    static_assert(std::is_unsigned_v<Length> && !std::is_same_v<Length, bool>);
    static_assert(sizeof(Length) <= 8);
    if (!writer_detail::valid(writer) || (!input.data && input.size)) return Status::InvalidInput;
    if (input.size > MaxBytes) return Status::LimitExceeded;
    Length length{};
    if (writer_detail::convert(input.size, length) != Status::Ok) return Status::LimitExceeded;
    if (writer_detail::overlap(input.data, input.size, writer.data, writer.size)) return Status::InvalidInput;
    if (sizeof(Length) > writer.size - writer.offset ||
        input.size > writer.size - writer.offset - sizeof(Length)) return Status::Truncated;
    assert(writer.data != nullptr); assert(writer.offset <= writer.size);
    writer_detail::store<Order>(writer.data + writer.offset, length);
    auto* p = writer.data + writer.offset + sizeof(Length);
    for (std::uint32_t i = 0; i < input.size; ++i) p[i] = input.data[i];
    writer.offset += static_cast<std::uint32_t>(sizeof(Length)) + input.size;
    return Status::Ok;
}
static_assert(std::is_trivially_copyable_v<Writer> && std::is_trivially_copyable_v<WriteBlock>);
} // namespace bolt::wire::sbe
