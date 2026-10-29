#pragma once
#include <cassert>
#include <cstdint>

namespace bolt {
// Exact lowercase hexadecimal; capacity includes the terminating NUL. Never clips.
inline bool hex_text(const std::uint8_t* bytes, std::uint32_t len,
                     char* out, std::uint32_t cap) noexcept {
    assert(bytes != nullptr || len == 0u);
    assert(out != nullptr);
    if (cap == 0u || len > (cap - 1u) / 2u) return false;
    constexpr char digits[] = "0123456789abcdef";
    for (std::uint32_t i = 0; i < len; ++i) {
        out[2u * i] = digits[bytes[i] >> 4u];
        out[2u * i + 1u] = digits[bytes[i] & 15u];
    }
    out[2u * len] = '\0';
    return true;
}

// UUID bytes are in canonical network order (including Parquet's UUID lane).
inline bool uuid_text(const std::uint8_t* bytes, char* out, std::uint32_t cap) noexcept {
    assert(bytes != nullptr);
    assert(out != nullptr);
    if (cap < 37u) return false;
    char hex[33];
    if (!hex_text(bytes, 16u, hex, sizeof(hex))) return false;
    std::uint32_t w = 0;
    for (std::uint32_t i = 0; i < 32u; ++i) {
        if (i == 8u || i == 12u || i == 16u || i == 20u) out[w++] = '-';
        out[w++] = hex[i];
    }
    out[w] = '\0';
    assert(w == 36u);
    return true;
}
}  // namespace bolt
