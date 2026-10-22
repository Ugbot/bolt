#pragma once
#include "bolt/wire/mold_udp64_limits.h"
#include "sbe.h"

namespace bolt::wire::mold_udp64 {
inline constexpr std::uint32_t kHeaderBytes = 20;
inline constexpr std::uint32_t kSessionBytes = 10;
enum class Status : std::uint8_t { Ok, End, Invalid, Truncated, Capacity, Overflow };
enum class Kind : std::uint8_t { Data, Heartbeat, EndSession };
struct Limits { std::uint32_t datagram_bytes = 0, messages = 0; };
struct Packet {
    ByteView bytes{}, session{};
    std::uint64_t first_sequence = 0;
    std::uint16_t message_count = 0;
    Kind kind = Kind::Data;
};
struct Cursor { std::uint32_t offset = kHeaderBytes, ordinal = 0; };
struct Message { ByteView bytes{}; std::uint64_t sequence = 0; std::uint32_t ordinal = 0; };
// Exact whole UDP payload, including zero-length application messages. Unknown
// application bytes are opaque. Failure preserves output, no socket ownership.
inline Status frame(ByteView bytes, Limits limits, Packet& out) noexcept {
    if (bytes.data == nullptr || limits.datagram_bytes < kHeaderBytes ||
        limits.datagram_bytes > kUdpPayloadMax || limits.messages == 0) return Status::Invalid;
    if (bytes.size > limits.datagram_bytes) return Status::Capacity;
    if (bytes.size < kHeaderBytes) return Status::Truncated;
    ByteReader r{bytes.data, bytes.size};
    Packet p{};
    p.bytes = bytes;
    if (!read_bytes(r, kSessionBytes, p.session) ||
        !sbe::read_scalar<sbe::ByteOrder::Big>(r, p.first_sequence) ||
        !sbe::read_scalar<sbe::ByteOrder::Big>(r, p.message_count)) return Status::Truncated;
    if (p.message_count == 0 || p.message_count == UINT16_MAX) {
        if (bytes.size != kHeaderBytes) return Status::Invalid;
        p.kind = p.message_count == 0 ? Kind::Heartbeat : Kind::EndSession;
    } else {
        if (p.message_count > limits.messages) return Status::Capacity;
        if (std::uint64_t(p.message_count - 1) > UINT64_MAX - p.first_sequence) return Status::Overflow;
        for (std::uint32_t i = 0; i < p.message_count; ++i) {
            std::uint16_t length{};
            if (!sbe::read_scalar<sbe::ByteOrder::Big>(r, length) || !skip_bytes(r, length))
                return Status::Truncated;
        }
        if (r.offset != bytes.size) return Status::Invalid;
    }
    assert(p.session.size == kSessionBytes);
    assert(r.offset == bytes.size);
    out = p;
    return Status::Ok;
}
// Cursor begins {}; packet and cursor must refer to the same immutable frame.
inline Status next(const Packet& p, Cursor& c, Message& out) noexcept {
    if (p.bytes.data == nullptr || p.bytes.size < kHeaderBytes ||
        p.bytes.size > kUdpPayloadMax || c.offset < kHeaderBytes || c.offset > p.bytes.size)
        return Status::Invalid;
    if (p.kind != Kind::Data) return c.ordinal == 0 && c.offset == kHeaderBytes ? Status::End : Status::Invalid;
    if (c.ordinal > p.message_count) return Status::Invalid;
    if (c.ordinal == p.message_count) return c.offset == p.bytes.size ? Status::End : Status::Invalid;
    ByteReader r{p.bytes.data, p.bytes.size, c.offset};
    std::uint16_t length{};
    Message m{};
    if (!sbe::read_scalar<sbe::ByteOrder::Big>(r, length) || !read_bytes(r, length, m.bytes))
        return Status::Truncated;
    if (c.ordinal > UINT64_MAX - p.first_sequence) return Status::Overflow;
    m.sequence = p.first_sequence + c.ordinal;
    m.ordinal = c.ordinal;
    c.offset = r.offset;
    ++c.ordinal;
    assert(c.offset <= p.bytes.size);
    assert(m.bytes.size == length);
    out = m;
    return Status::Ok;
}
} // namespace bolt::wire::mold_udp64
