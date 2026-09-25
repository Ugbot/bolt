// bolt_crc64nvme.cpp — software slicing-by-8 CRC-64/NVME.

#include "bolt/io/bolt_crc64nvme.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace io {

namespace {

constexpr uint64_t kPolyReflected = 0x9A6C9329AC4BC9B5ull;

struct Tables {
    uint64_t t[8][256];
};

constexpr Tables make_tables() {
    Tables tb{};
    for (uint32_t b = 0; b < 256; ++b) {
        uint64_t c = b;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (kPolyReflected ^ (c >> 1)) : (c >> 1);
        }
        tb.t[0][b] = c;
    }
    for (int s = 1; s < 8; ++s) {
        for (uint32_t b = 0; b < 256; ++b) {
            const uint64_t prev = tb.t[s - 1][b];
            tb.t[s][b] = (prev >> 8) ^ tb.t[0][prev & 0xFFu];
        }
    }
    return tb;
}

constexpr Tables kTables = make_tables();

static_assert(kTables.t[0][0] == 0 && kTables.t[0][0x80] == kPolyReflected,
              "slicing table generation");

inline uint64_t load_le64(const uint8_t* p) noexcept {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

}  // namespace

uint64_t crc64nvme(const void* data, size_t len, uint64_t seed) noexcept {
    assert(data != nullptr || len == 0);
    assert(len <= (static_cast<size_t>(1) << 48));
    const auto& T = kTables.t;
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t crc = ~seed;
    while (len >= 8u) {
        crc ^= load_le64(p);
        crc = T[7][crc & 0xFFu] ^ T[6][(crc >> 8) & 0xFFu] ^
              T[5][(crc >> 16) & 0xFFu] ^ T[4][(crc >> 24) & 0xFFu] ^
              T[3][(crc >> 32) & 0xFFu] ^ T[2][(crc >> 40) & 0xFFu] ^
              T[1][(crc >> 48) & 0xFFu] ^ T[0][crc >> 56];
        p += 8;
        len -= 8u;
    }
    while (len > 0u) {
        crc = T[0][(crc ^ *p) & 0xFFu] ^ (crc >> 8);
        ++p;
        --len;
    }
    return ~crc;
}

}  // namespace io
}  // namespace bolt
