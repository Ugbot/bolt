// bolt/ingest/bolt_pb_cursor.h — a bounded, fail-closed protobuf wire reader.
//
// No protobuf runtime: varint, fixed32, fixed64 and length-delimited fields are
// read by hand, anything else is skipped by wire type, and a truncated or
// over-long field fails the read. Used by Prometheus remote write and OTLP.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace bolt::ingest::pb {

// Protobuf wire types.
inline constexpr std::uint8_t kWireVarint = 0;
inline constexpr std::uint8_t kWireF64    = 1;
inline constexpr std::uint8_t kWireLen    = 2;
inline constexpr std::uint8_t kWireF32    = 5;

// A cursor over an immutable byte span. All reads are bounds-checked.
struct Cursor {
    const std::uint8_t* p;
    const std::uint8_t* end;
    bool empty() const noexcept { return p >= end; }
    std::size_t remaining() const noexcept {
        return static_cast<std::size_t>(end - p);
    }
};

// Read a base-128 varint (<=10 bytes). False on truncation / overlong.
inline bool read_varint(Cursor* c, std::uint64_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    std::uint64_t v = 0;
    for (std::uint32_t i = 0; i < 10; ++i) {
        if (c->p >= c->end) return false;
        const std::uint8_t b = *c->p++;
        v |= static_cast<std::uint64_t>(b & 0x7Fu) << (7u * i);
        if ((b & 0x80u) == 0) { *out = v; return true; }
    }
    return false;  // > 10 bytes: malformed
}

// Read a tag: field number + wire type. False on truncation.
inline bool read_tag(Cursor* c, std::uint32_t* field, std::uint8_t* wire) noexcept {
    assert(c != nullptr && field != nullptr && wire != nullptr);
    std::uint64_t key = 0;
    if (!read_varint(c, &key)) return false;
    *wire  = static_cast<std::uint8_t>(key & 0x7u);
    *field = static_cast<std::uint32_t>(key >> 3);
    return *field != 0;  // field 0 is illegal
}

// Read a length-delimited field into a sub-cursor. False on truncation.
inline bool read_len_delimited(Cursor* c, Cursor* sub) noexcept {
    assert(c != nullptr && sub != nullptr);
    std::uint64_t n = 0;
    if (!read_varint(c, &n)) return false;
    if (n > c->remaining()) return false;
    sub->p   = c->p;
    sub->end = c->p + n;
    c->p    += n;
    return true;
}

inline bool read_fixed64(Cursor* c, std::uint64_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    if (c->remaining() < 8) return false;
    std::uint64_t v = 0;
    std::memcpy(&v, c->p, 8);   // little-endian wire == host on x86/ARM64
    c->p += 8;
    *out = v;
    return true;
}

// Advance past a field of unmodeled wire type. False on truncation.
inline bool skip_field(Cursor* c, std::uint8_t wire) noexcept {
    assert(c != nullptr);
    switch (wire) {
        case kWireVarint: { std::uint64_t t; return read_varint(c, &t); }
        case kWireF64:    { if (c->remaining() < 8) return false; c->p += 8; return true; }
        case kWireF32:    { if (c->remaining() < 4) return false; c->p += 4; return true; }
        case kWireLen:    { Cursor sub; return read_len_delimited(c, &sub); }
        default:          return false;   // groups (3/4) unsupported
    }
}

inline bool read_fixed32(Cursor* c, std::uint32_t* out) noexcept {
    assert(c != nullptr && out != nullptr);
    if (c->remaining() < 4) return false;
    std::memcpy(out, c->p, 4);
    c->p += 4;
    return true;
}

inline std::string_view view_of(const Cursor& c) noexcept {
    return std::string_view(reinterpret_cast<const char*>(c.p), c.remaining());
}

}  // namespace bolt::ingest::pb
