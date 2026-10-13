#pragma once
#include "sbe.h"

namespace bolt::wire::sbe {
struct Group {
    std::uint32_t block_length = 0;
    std::uint32_t remaining = 0;
    std::uint16_t acting_version = 0;
};

// Packed block-length/count dimensions. The schema selects their types (CME
// often uses uint16/uint8, other schemas uint16/uint16). Non-packed/custom
// composites use read_scalar and schema-derived offsets instead.
// Depth is schema traversal depth, starting at one; generated callers must
// pass the real depth. This API has no recursion, hidden stack or allocation.
template <ByteOrder Order, typename BlockLength, typename Count,
          std::uint32_t MaxCount, std::uint32_t MaxDepth = 8>
BOLT_FORCE_INLINE Status open_group(ByteReader& reader, std::uint32_t minimum_block_length,
                         std::uint16_t acting_version, std::uint32_t depth,
                         Group& out) noexcept {
    static_assert(std::is_unsigned_v<BlockLength> && std::is_unsigned_v<Count>);
    static_assert(!std::is_same_v<BlockLength, bool> && !std::is_same_v<Count, bool>);
    static_assert(sizeof(BlockLength) <= 4 && sizeof(Count) <= 4);
    static_assert(MaxDepth > 0);
    if (depth == 0 || depth > MaxDepth) return Status::LimitExceeded;
    auto next = reader;
    BlockLength block_length{};
    Count count{};
    if (!read_scalar<Order>(next, block_length) ||
        !read_scalar<Order>(next, count)) return Status::Truncated;
    if (count > MaxCount) return Status::LimitExceeded;
    if (block_length < minimum_block_length) return Status::InvalidInput;
    // Fixed bytes alone must fit; nested groups/data are checked on traversal.
    if (std::uint64_t(block_length) * count > next.size - next.offset) return Status::Truncated;
    assert(next.offset <= next.size);
    assert(count <= MaxCount);
    reader = next;
    out = {static_cast<std::uint32_t>(block_length), static_cast<std::uint32_t>(count), acting_version};
    return Status::Ok;
}

// Advances over one FIXED block. The schema-specific caller must consume this
// entry's nested groups and variable data before requesting the next entry.
// Multiplying block_length by count is NOT a valid skip for nested groups.
BOLT_FORCE_INLINE Status next_entry(ByteReader& reader, Group& group, Block& out) noexcept {
    if (!byte_reader_valid(reader)) return Status::InvalidInput;
    if (group.remaining == 0) return Status::End;
    auto next = reader;
    ByteView fixed{};
    if (!read_bytes(next, group.block_length, fixed)) return Status::Truncated;
    assert(next.offset <= next.size);
    assert(fixed.size == group.block_length);
    reader = next;
    --group.remaining;
    out = {fixed, group.acting_version};
    return Status::Ok;
}
static_assert(std::is_trivially_copyable_v<Group>);
} // namespace bolt::wire::sbe
