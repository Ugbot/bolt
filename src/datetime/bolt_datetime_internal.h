// Shared helpers for bolt_datetime_fmt.cpp / bolt_datetime_parse.cpp.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::dtfmt::detail {

extern const char* const kMonthLong[12];
extern const char* const kDowLong[7];

std::uint8_t letter_max(char c) noexcept;

// Bounded output cursor: every append checks the capacity.
struct Buf {
    char*         p;
    std::uint32_t cap;
    std::uint32_t len;

    bool ch(char c) noexcept {
        if (len >= cap) return false;
        p[len++] = c;
        return true;
    }
    bool str(const char* s, std::uint32_t n) noexcept {
        if (n > cap - len) return false;
        if (n != 0) std::memcpy(p + len, s, n);
        len += n;
        return true;
    }
    // Decimal of a non-negative value, zero-padded to `width`.
    bool num(std::int64_t v, std::uint32_t width) noexcept {
        assert(v >= 0);
        char tmp[20];
        std::uint32_t n = 0;
        do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v != 0 && n < 20);
        while (n < width && n < 20) tmp[n++] = '0';
        if (n > cap - len) return false;
        for (std::uint32_t i = 0; i < n; ++i) p[len + i] = tmp[n - 1 - i];
        len += n;
        return true;
    }
};

}  // namespace bolt::dtfmt::detail
