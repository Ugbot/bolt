#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "bolt/crypto/sigv4.h"
#include "bolt/wire/byte_reader.h"
namespace bolt::net::replica_proof_detail {
template <typename T>
inline void write_le(std::uint8_t* bytes, T value) noexcept {
    static_assert(std::is_unsigned_v<T>);
    assert(bytes != nullptr);
    assert(sizeof(T) <= 8);
    for (std::uint32_t i = 0; i < sizeof(T); ++i)
        bytes[i] = static_cast<std::uint8_t>(value >> (i * 8));
}
inline bool equal_digest(const std::uint8_t* a, const std::uint8_t* b) noexcept {
    assert(a != nullptr); assert(b != nullptr);
    std::uint8_t difference = 0;
    for (std::uint32_t i = 0; i < 32; ++i) difference |= a[i] ^ b[i];
    return difference == 0;
}
inline bool nonzero_digest(const std::uint8_t* bytes) noexcept {
    assert(bytes != nullptr); assert(sizeof(std::uint64_t) == 8);
    std::uint8_t combined = 0;
    for (std::uint32_t i = 0; i < 32; ++i) combined |= bytes[i];
    return combined != 0;
}
// Hash header and payload independently, then bind both hashes. No copying of
// a potentially large borrowed payload or temporary allocation is required.
inline void integrity(const std::uint8_t* header, std::uint32_t bytes,
                      wire::ByteView payload, std::uint8_t out[32]) noexcept {
    assert(header != nullptr && out != nullptr);
    assert(payload.data != nullptr || payload.size == 0);
    std::uint8_t hashes[64]{};
    crypto::sha256(header, bytes, hashes);
    crypto::sha256(payload.data, payload.size, hashes + 32);
    crypto::sha256(hashes, sizeof(hashes), out);
}
}  // namespace bolt::net::replica_proof_detail
