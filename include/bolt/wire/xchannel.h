#pragma once
#include "xchannel_format.h"

namespace bolt::wire::xchannel {
struct Region { ByteView bytes{}; std::uint64_t file_offset = 0; };
struct Cursor {
    std::uint64_t offset = kPrefixBytes, record_index = 0;
    Instance instance{};
};
struct Roll {
    std::uint64_t next_sequence = 0, next_base_record_index = 0;
    Instance next_instance{};
};
struct Record {
    ByteView payload{};
    std::uint64_t offset = 0, next_offset = 0, record_index = 0, user_meta_u64 = 0;
    std::uint16_t message_type = 0;
    Kind kind = Kind::User;
    Roll roll{};
};
// File must come from successful admission and remain unchanged. Cursor begins
// only here; arbitrary saved offsets cannot be admitted without replay proving a
// record boundary/index. A retained file with base>0 starts there explicitly.
inline Cursor begin(const File& f) noexcept {
    assert(f.prefix.data != nullptr);
    assert(f.prefix.size == kPrefixBytes);
    return {kPrefixBytes, f.base_record_index, f.instance};
}
inline std::uint64_t required_region_offset(const File& f, const Cursor& c) noexcept {
    assert(f.region_size >= kPrefixBytes + kRollBytes);
    assert(c.offset >= kPrefixBytes);
    return c.offset - c.offset % f.region_size;
}
namespace detail {
inline bool valid_file(const File& f) noexcept {
    return f.prefix.data != nullptr && f.prefix.size == kPrefixBytes && aligned(f.prefix.data) &&
        f.region_size >= kPrefixBytes + kRollBytes && f.region_size % 8 == 0;
}
inline Status slot(const File& f, Region region, const Cursor& c, std::uint32_t& local) noexcept {
    if (!valid_file(f) || region.bytes.data == nullptr || !aligned(region.bytes.data) ||
        !equal(f.instance, c.instance) || c.offset < kPrefixBytes || c.offset % 8 != 0 ||
        c.record_index < f.base_record_index || region.file_offset % f.region_size != 0 ||
        region.bytes.size > f.region_size) return Status::Invalid;
    const auto offset = static_cast<std::uint32_t>(c.offset % f.region_size);
    if (f.region_size - offset < kRollBytes) return Status::Corrupt;
    if (region.file_offset != c.offset - offset) return Status::NeedRegion;
    if (offset >= region.bytes.size) return Status::Truncated;
    assert(offset < f.region_size);
    assert(region.bytes.size <= f.region_size);
    local = offset;
    return Status::Ok;
}
inline Status decode_roll(const File& f, const Cursor& c, Record& r) noexcept {
    assert(r.payload.size == 40);
    assert(r.payload.data != nullptr);
    const auto* p = r.payload.data;
    if (r.user_meta_u64 != 0 || !zero(p + 36, 4) ||
        le<std::uint32_t>(p + 32) != io::crc32c(p, 32) || zero(p + 8, 16))
        return Status::Corrupt;
    if (f.sequence == UINT64_MAX) return Status::Overflow;
    r.roll.next_sequence = le<std::uint64_t>(p);
    r.roll.next_base_record_index = le<std::uint64_t>(p + 24);
    std::memcpy(r.roll.next_instance.bytes, p + 8, 16);
    if (r.roll.next_sequence != f.sequence + 1) return Status::Corrupt;
    // This count is final only AFTER the Roll's committed byte was acquired.
    const auto count = acquire_count(f);
    if (count > UINT64_MAX - f.base_record_index) return Status::Overflow;
    if (r.roll.next_base_record_index != f.base_record_index + count ||
        r.roll.next_base_record_index != c.record_index) return Status::Gap;
    return Status::Ok;
}
inline bool same_record(const Record& a, const Record& b) noexcept {
    return a.payload.data == b.payload.data && a.payload.size == b.payload.size &&
        a.offset == b.offset && a.next_offset == b.next_offset &&
        a.record_index == b.record_index && a.user_meta_u64 == b.user_meta_u64 &&
        a.message_type == b.message_type && a.kind == b.kind &&
        a.roll.next_sequence == b.roll.next_sequence &&
        a.roll.next_base_record_index == b.roll.next_base_record_index &&
        equal(a.roll.next_instance, b.roll.next_instance);
}
} // namespace detail

// One bounded poll. Do not inspect payload until Ok. Idle reads ONLY commitment;
// other header bytes may still be concurrently written. A returned payload is
// borrowed until caller unpins/replaces the region, including after consume.
// Do not enqueue a raw view into asynchronous work without an owned pin/copy.
// All non-Ok results preserve out; every call preserves c. Mapping boundaries
// return NeedRegion; required_region_offset gives the caller the exact region.
inline Status peek(const File& f, Region region, const Cursor& c, Record& out) noexcept {
    std::uint32_t local = 0;
    auto status = detail::slot(f, region, c, local);
    if (status != Status::Ok) return status;
    const auto* p = region.bytes.data + local;
    const auto committed = detail::acquire_byte(p);
    if (committed == 0) return Status::Idle;
    if (committed != 1) return Status::Corrupt;
    if (region.bytes.size - local < kHeaderBytes) return Status::Truncated;
    const auto length = detail::le<std::uint32_t>(p + 4);
    const auto remaining = f.region_size - local;
    if (p[1] < 1 || p[1] > 3 || length > remaining - kHeaderBytes) return Status::Corrupt;
    const auto total = (std::uint64_t(kHeaderBytes) + length + 7u) & ~std::uint64_t(7u);
    if (total > remaining) return Status::Corrupt;
    if (total > UINT64_MAX - c.offset) return Status::Overflow;
    Record r{};
    r.kind = static_cast<Kind>(p[1]);
    r.offset = c.offset;
    r.next_offset = c.offset + total;
    r.record_index = c.record_index;
    r.message_type = detail::le<std::uint16_t>(p + 2);
    r.user_meta_u64 = detail::le<std::uint64_t>(p + 8);
    if (r.kind == Kind::User) {
        if (remaining - total < kRollBytes) return Status::Corrupt;
        if (f.mtu != 0 && length > f.mtu) return Status::Corrupt;
        if (length > f.payload_cap) return Status::Capacity;
        if (c.record_index == UINT64_MAX) return Status::Overflow;
    } else if (r.kind == Kind::Skip) {
        if (std::uint64_t(kHeaderBytes) + length != remaining) return Status::Corrupt;
        // Skip padding is opaque; the reader need not touch/map all of it.
        r.payload = {};
        out = r;
        return Status::Ok;
    } else if (length != 40) return Status::Corrupt;
    if (total > region.bytes.size - local) return Status::Truncated;
    r.payload = {p + kHeaderBytes, length};
    if (r.kind == Kind::Roll) {
        status = detail::decode_roll(f, c, r);
        if (status != Status::Ok) return status;
    }
    assert(r.next_offset > c.offset);
    assert(r.payload.size == length);
    out = r;
    return Status::Ok;
}
// Token must be an unmodified successful peek from this exact mapping/cursor.
// Refusing output simply means not calling consume. No hot allocation/IO/lock.
inline Status consume(const File& f, Region region, Cursor& c, const Record& token) noexcept {
    Record current{};
    const auto status = peek(f, region, c, current);
    if (status != Status::Ok) return status;
    if (!detail::same_record(current, token) || current.kind == Kind::Roll) return Status::Invalid;
    Cursor next = c;
    next.offset = current.next_offset;
    if (current.kind == Kind::User) ++next.record_index;
    assert(next.offset > c.offset);
    assert(next.record_index >= c.record_index);
    c = next;
    return Status::Ok;
}
// Successor File must already be successfully admitted (Prepared is never a
// File). Caller pins both prefixes/old region until this check finishes, then
// replaces its active File/Cursor together. The old cursor never changes.
inline Status follow_successor(const File& f, Region region, const Cursor& c,
    const Record& token, const File& successor, Cursor& out) noexcept {
    Record current{};
    const auto status = peek(f, region, c, current);
    if (status != Status::Ok) return status;
    if (!detail::same_record(current, token) || current.kind != Kind::Roll ||
        !detail::valid_file(successor)) return Status::Invalid;
    if (successor.generation != f.generation) return Status::Recreated;
    if (successor.sequence != current.roll.next_sequence ||
        !detail::equal(successor.instance, current.roll.next_instance) ||
        !detail::equal(successor.predecessor, f.instance) ||
        successor.predecessor_sequence != f.sequence) return Status::Corrupt;
    if (successor.base_record_index != current.roll.next_base_record_index) return Status::Gap;
    if (successor.region_size != f.region_size || successor.mtu != f.mtu ||
        std::memcmp(successor.name, f.name, 48) != 0) return Status::Corrupt;
    assert(successor.sequence == f.sequence + 1);
    assert(successor.base_record_index == c.record_index);
    out = begin(successor);
    return Status::Ok;
}
} // namespace bolt::wire::xchannel
