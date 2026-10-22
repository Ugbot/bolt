#pragma once
#include "byte_reader.h"
#include "bolt/io/bolt_crc32c.h"
#include <atomic>
#include <cstring>

// xchannel 7.0.0 / FORMAT4, upstream 49063b06aa31cba3ddd659a19808f1aed2e6fc26.
// Caller owns mapping/discovery/retention. This header never maps or writes.
namespace bolt::wire::xchannel {
inline constexpr std::uint32_t kHeaderBytes = 16, kPrefixBytes = 208;
inline constexpr std::uint32_t kRollBytes = 56;
enum class Status : std::uint8_t {
    Ok, Idle, Prepared, NeedRegion, Truncated, Capacity, Gap, Recreated,
    Unsupported, Corrupt, Overflow, Invalid
};
enum class Kind : std::uint8_t { Channel = 0, User = 1, Skip = 2, Roll = 3 };
struct Limits {
    std::uint32_t page_size = 0, max_region_bytes = 0, max_payload_bytes = 0;
};
struct Expected {
    std::uint64_t path_sequence = 0, generation = 0, base_record_index = 0;
    bool check_base = false;
};
struct Instance { std::uint8_t bytes[16]{}; };
struct File {
    // Prefix MUST remain pinned until all operations using this File finish.
    ByteView prefix{};
    Instance instance{}, predecessor{};
    std::uint64_t sequence = 0, generation = 0, base_record_index = 0;
    std::uint64_t predecessor_sequence = UINT64_MAX;
    std::uint32_t region_size = 0, mtu = 0, payload_cap = 0;
    std::uint8_t name[48]{};
};
namespace detail {
inline bool aligned(const void* p) noexcept {
    return (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0;
}
inline std::uint8_t acquire_byte(const std::uint8_t* p) noexcept {
    assert(p != nullptr);
    static_assert(std::atomic_ref<std::uint8_t>::required_alignment == 1);
    // atomic_ref is used for loads only; const_cast never authorizes a store.
    const auto value = std::atomic_ref<std::uint8_t>(
        *const_cast<std::uint8_t*>(p)).load(std::memory_order_acquire);
    assert(value <= UINT8_MAX);
    return value;
}
inline std::uint64_t acquire_count(const File& f) noexcept {
    assert(f.prefix.size == kPrefixBytes);
    assert(aligned(f.prefix.data));
    static_assert(std::atomic_ref<std::uint64_t>::required_alignment <= 8);
    auto* p = reinterpret_cast<std::uint64_t*>(
        const_cast<std::uint8_t*>(f.prefix.data + 24));
    return std::atomic_ref<std::uint64_t>(*p).load(std::memory_order_acquire);
}
template <typename T> inline T le(const std::uint8_t* p) noexcept {
    static_assert(std::is_unsigned_v<T>);
    assert(p != nullptr);
    T value{};
    ByteReader r{p, static_cast<std::uint32_t>(sizeof(T))};
    const bool read = read_le(r, value);
    assert(read);
    assert(r.offset == sizeof(T));
    (void)read;
    return value;
}
inline bool equal(const Instance& a, const Instance& b) noexcept {
    return std::memcmp(a.bytes, b.bytes, 16) == 0; // sv-memcmp-ok: fixed inline instance bytes, no StringView offsets.
}
inline bool zero(const std::uint8_t* p, std::uint32_t size) noexcept {
    assert(p != nullptr);
    assert(size <= 48);
    std::uint8_t bits = 0;
    for (std::uint32_t i = 0; i < size; ++i) bits |= p[i];
    return bits == 0;
}
inline Status identity(const std::uint8_t* p, File& f) noexcept {
    assert(p != nullptr);
    assert(aligned(p));
    if (zero(p + 144, 16) || !zero(p + 184, 8) ||
        !zero(p + 193, 3) || !zero(p + 200, 8)) return Status::Corrupt;
    if (le<std::uint32_t>(p + 196) != io::crc32c(p + 144, 40)) return Status::Corrupt;
    std::memcpy(f.instance.bytes, p + 144, 16);
    std::memcpy(f.predecessor.bytes, p + 160, 16);
    f.predecessor_sequence = le<std::uint64_t>(p + 176);
    if (zero(f.predecessor.bytes, 16) != (f.predecessor_sequence == UINT64_MAX))
        return Status::Corrupt;
    return Status::Ok;
}
inline Status geometry(const std::uint8_t* p, Limits limits, File& f) noexcept {
    assert(p != nullptr);
    assert(limits.page_size != 0);
    f.region_size = le<std::uint32_t>(p + 48);
    f.mtu = le<std::uint32_t>(p + 52);
    if (f.region_size < kPrefixBytes + kRollBytes || f.region_size % 8 != 0 ||
        f.region_size % limits.page_size != 0) return Status::Corrupt;
    if (f.region_size > limits.max_region_bytes) return Status::Capacity;
    f.payload_cap = limits.max_payload_bytes;
    return Status::Ok;
}
} // namespace detail

// Startup only. Prefix starts at file offset zero, at least 8-byte aligned.
// Single external writer; mapped byte/u64 atomics must be lock-free across
// processes on this host. Caller guarantees object storage and mapping lifetime;
// no truncation/reuse while pinned. C++ alone does not define interprocess mmap
// synchronization: the host ABI must implement these as shared hardware atomics.
// Only LE framing/host is supported. Acquire gives visibility, NOT durability.
// Expected generation is explicit (including zero, which cannot detect reuse).
// All failure paths preserve out, including Prepared (base may still change).
inline Status admit_file(ByteView prefix, Expected expected, Limits limits, File& out) noexcept {
    if (prefix.data == nullptr || !detail::aligned(prefix.data) || limits.page_size == 0 ||
        limits.max_region_bytes < kPrefixBytes + kRollBytes) return Status::Invalid;
    if constexpr (std::endian::native != std::endian::little ||
        !std::atomic_ref<std::uint8_t>::is_always_lock_free ||
        !std::atomic_ref<std::uint64_t>::is_always_lock_free) return Status::Unsupported;
    if (prefix.size == 0) return Status::Truncated;
    const auto committed = detail::acquire_byte(prefix.data);
    if (committed == 0) return Status::Idle;
    if (committed != 1) return Status::Corrupt;
    // Version must be inspected before accessing the v4 extension. A real v3
    // prefix is 144 bytes, not 208; it must report Unsupported, not Truncated.
    if (prefix.size < 58) return Status::Truncated;
    if (detail::le<std::uint16_t>(prefix.data + 56) != 4) return Status::Unsupported;
    if (prefix.size < kPrefixBytes) return Status::Truncated;
    const auto* p = prefix.data;
    if (p[1] != 0 || detail::le<std::uint32_t>(p + 4) != 192) return Status::Corrupt;
    if (p[58] != 1 || p[59] != 8 || p[64] != 8 ||
        detail::le<std::uint32_t>(p + 60) != 0) return Status::Unsupported;
    if ((detail::acquire_byte(p + 113) & 0xfeu) != 0 || !detail::zero(p + 114, 2))
        return Status::Corrupt;
    File f{};
    auto status = detail::geometry(p, limits, f);
    if (status != Status::Ok) return status;
    status = detail::identity(p, f);
    if (status != Status::Ok) return status;
    const auto publication = detail::acquire_byte(p + 192);
    if (publication == 0) return Status::Prepared;
    if (publication != 1) return Status::Corrupt;
    f.sequence = detail::le<std::uint64_t>(p + 40);
    f.generation = detail::le<std::uint64_t>(p + 136);
    f.base_record_index = detail::le<std::uint64_t>(p + 32);
    if (f.generation != expected.generation) return Status::Recreated;
    if (f.sequence != expected.path_sequence) return Status::Gap;
    if (expected.check_base && f.base_record_index != expected.base_record_index) return Status::Gap;
    f.prefix = {p, kPrefixBytes};
    std::memcpy(f.name, p + 65, 48);
    assert(f.prefix.size == kPrefixBytes);
    assert(f.region_size >= kPrefixBytes + kRollBytes);
    out = f;
    return Status::Ok;
}
} // namespace bolt::wire::xchannel
