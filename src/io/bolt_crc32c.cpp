// bolt_crc32c.cpp — Software slicing-by-8 CRC32C (Castagnoli).
//
// This is the portable fallback that lives behind `bolt::io::crc32c` on
// targets without SSE4.2 / ARM CRC. Also exposed as `crc32c_software`
// so tests can force-exercise the software path on any host.
//
// Algorithm: Intel / Koopman "slicing-by-8". Eight 256-entry tables index
// 8 consecutive input bytes in parallel; the CRC state is updated once
// per 8 bytes by XORing 8 table lookups. ~4× faster than a bytewise LUT
// on scalar hardware and still deterministic across platforms.
//
// Poly: 0x1EDC6F41 (Castagnoli) in reversed form 0x82F63B78.
//
// Tiger Style: noexcept, no exceptions, no heap. Tables are `constexpr`
// and therefore materialised at compile time — zero runtime init, no
// branch on hot path, no thread-safety question. The 8 KB of `.rodata`
// links once per binary (the whole reason this file is a TU).

#include "bolt/io/bolt_crc32c.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace io {

namespace {

// Castagnoli polynomial in reflected form.
constexpr uint32_t kPolyReflected = 0x82F63B78u;

// ---------------------------------------------------------------------------
// Slicing-by-8 tables — computed at compile time.
//
// Row 0: standard bytewise CRC32C LUT. Row k (1..7): row (k-1) advanced
// by one more byte of implicit zero, i.e. LUT for a byte k positions
// further down the stream. Koopman's construction.
// ---------------------------------------------------------------------------

struct SliceTables {
    uint32_t t[8][256];
};

constexpr SliceTables make_slice_tables() noexcept {
    SliceTables s{};
    // Row 0 — classical bytewise CRC32C.
    for (unsigned i = 0; i < 256u; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j) {
            c = (c & 1u) ? ((c >> 1) ^ kPolyReflected) : (c >> 1);
        }
        s.t[0][i] = c;
    }
    // Rows 1..7 — each advances the prior row by one zero byte.
    for (unsigned i = 0; i < 256u; ++i) {
        uint32_t c = s.t[0][i];
        for (int k = 1; k < 8; ++k) {
            c = s.t[0][c & 0xFFu] ^ (c >> 8);
            s.t[k][i] = c;
        }
    }
    return s;
}

// `constexpr` guarantees compile-time evaluation when used in a constant
// context; giving it external linkage via an anonymous-namespace `inline
// constexpr` ensures one copy, statically initialised in `.rodata`.
inline constexpr SliceTables kSlice = make_slice_tables();

}  // namespace

uint32_t crc32c_software(const void* BOLT_RESTRICT data,
                          size_t len,
                          uint32_t seed) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));  // sanity upper bound

    uint32_t crc = ~seed;
    const uint8_t* p = static_cast<const uint8_t*>(data);

    // Align to 8 bytes so the inner loop can pull aligned u64 loads.
    while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 7u) != 0u) {
        crc = kSlice.t[0][(crc ^ *p) & 0xFFu] ^ (crc >> 8);
        ++p;
        --len;
    }

    // Slicing-by-8 hot loop: consume 8 bytes per iter with 8 XOR-of-LUT.
    while (len >= 8u) {
        uint64_t word;
        std::memcpy(&word, p, sizeof(word));
        crc ^= static_cast<uint32_t>(word);
        const uint32_t hi = static_cast<uint32_t>(word >> 32);
        crc = kSlice.t[7][ crc        & 0xFFu]
            ^ kSlice.t[6][(crc >>  8) & 0xFFu]
            ^ kSlice.t[5][(crc >> 16) & 0xFFu]
            ^ kSlice.t[4][(crc >> 24) & 0xFFu]
            ^ kSlice.t[3][ hi         & 0xFFu]
            ^ kSlice.t[2][(hi  >>  8) & 0xFFu]
            ^ kSlice.t[1][(hi  >> 16) & 0xFFu]
            ^ kSlice.t[0][(hi  >> 24) & 0xFFu];
        p   += 8;
        len -= 8u;
    }

    // Byte-at-a-time tail.
    while (len > 0u) {
        crc = kSlice.t[0][(crc ^ *p) & 0xFFu] ^ (crc >> 8);
        ++p;
        --len;
    }
    assert(p >= static_cast<const uint8_t*>(data) || data == nullptr);
    return ~crc;
}

}  // namespace io
}  // namespace bolt
