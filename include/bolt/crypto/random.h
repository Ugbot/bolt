#pragma once

#include <cstdint>

namespace bolt::crypto {

// The OS getentropy contract bounds each request to 256 bytes.
inline constexpr uint32_t kRandomBytesCap = 256;

// Cryptographic OS entropy; no allocation, TLS dependency, or fallback PRNG.
// Null, zero, or over-cap requests return false without touching output.
// Other failures clear all requested output bytes before returning false.
bool random_bytes(uint8_t* output, uint32_t size) noexcept;

}  // namespace bolt::crypto
