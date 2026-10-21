// Incremental owned SHA-256; caller-owned state and no allocation.
#pragma once

#include <cstdint>

namespace bolt::crypto {
inline constexpr uint64_t kSha256MaxInputBytes = UINT64_MAX / 8u;

// Implementation fields are public solely to keep the state fixed-size POD.
// Initialize through sha256_stream_init; never edit fields of a live state.
// State copies fork a hash at the copied position. Calls are thread-confined.
struct Sha256State {
    uint32_t h[8]{};
    uint64_t bitlen = 0;
    uint8_t buf[64]{};
    uint32_t buflen = 0;
    uint8_t phase = 0;
    uint8_t _pad[3]{};
};
static_assert(sizeof(Sha256State) == 112);
static_assert(alignof(Sha256State) == alignof(uint64_t));

// Init starts or explicitly restarts a state. Null returns false.
bool sha256_stream_init(Sha256State* state) noexcept;
// Active state only. A zero-length update accepts null data. Input must not
// overlap state, and its address extent and cumulative bit length must fit.
// Invalid arguments/phase/overflow return false before reading input or
// changing state. Actual input storage remains the caller's responsibility.
bool sha256_stream_update(Sha256State* state, const uint8_t* data,
                          uint64_t length) noexcept;
// Writes exactly32 digest bytes and consumes the active state. Output must
// not overlap state. Invalid argument/phase leaves state and output unchanged.
// Init is required before reuse after success; update/final cannot repeat it.
bool sha256_stream_final(Sha256State* state, uint8_t digest[32]) noexcept;
}  // namespace bolt::crypto
