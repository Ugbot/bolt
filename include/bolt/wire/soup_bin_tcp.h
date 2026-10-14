#pragma once
#include "sbe.h"

namespace bolt::wire::soup_bin_tcp {
inline constexpr std::uint32_t kLengthBytes = 2;
inline constexpr std::uint32_t kFrameBytesMax = UINT16_MAX + kLengthBytes;
inline constexpr std::uint32_t kSequenceDigits = 20;
enum class Status : std::uint8_t { Ok, NeedMore, Invalid, Capacity, Unsupported, Overflow };
enum class Direction : std::uint8_t { Server, Client };
struct Packet { ByteView bytes{}, payload{}; std::uint8_t type = 0; };
// Borrow exactly one logical frame from a possibly partial/coalesced TCP chunk.
// The limit is caller-admitted storage, not a protocol message-size policy.
// NeedMore/failure preserves output and consumed. No TCP accumulator/socket.
inline Status frame(ByteView input, std::uint32_t max_frame_bytes, Direction direction,
    Packet& out, std::uint32_t& consumed) noexcept {
    if ((input.data == nullptr && input.size != 0) || max_frame_bytes < 3 ||
        max_frame_bytes > kFrameBytesMax ||
        (direction != Direction::Server && direction != Direction::Client)) return Status::Invalid;
    ByteReader r{input.data, input.size};
    std::uint16_t length{};
    if (!sbe::read_scalar<sbe::ByteOrder::Big>(r, length)) return Status::NeedMore;
    if (length == 0) return Status::Invalid;
    const std::uint32_t total = std::uint32_t(length) + kLengthBytes;
    if (total > max_frame_bytes) return Status::Capacity;
    if (total > input.size) return Status::NeedMore;
    Packet p{{input.data, total}, {input.data + 3, total - 3}, input.data[2]};
    switch (p.type) {
    case '+': break;
    case 'A': if (direction != Direction::Server || length != 31) return Status::Invalid; break;
    case 'J':
        if (direction != Direction::Server || length != 2 ||
            (p.payload.data[0] != 'A' && p.payload.data[0] != 'S')) return Status::Invalid;
        break;
    case 'S': if (direction != Direction::Server) return Status::Invalid; break;
    case 'H': case 'Z': if (direction != Direction::Server || length != 1) return Status::Invalid; break;
    case 'L': if (direction != Direction::Client || length != 47) return Status::Invalid; break;
    case 'U': if (direction != Direction::Client) return Status::Invalid; break;
    case 'R': case 'O': if (direction != Direction::Client || length != 1) return Status::Invalid; break;
    default: return Status::Unsupported;
    }
    assert(p.bytes.size <= max_frame_bytes);
    assert(p.payload.size + 3 == total);
    consumed = total;
    out = p;
    return Status::Ok;
}
// Soup sequence numbers are ASCII decimal, LEFT space padded, not binary.
// Also used by Nasdaq GLIMPSE's 20-byte end-of-snapshot sequence field.
inline Status sequence_number(ByteView text, std::uint64_t& out) noexcept {
    if (text.data == nullptr || text.size != kSequenceDigits) return Status::Invalid;
    std::uint64_t value = 0;
    bool digits = false;
    for (std::uint32_t i = 0; i < kSequenceDigits; ++i) {
        const auto c = text.data[i];
        if (c == ' ' && !digits) continue;
        if (c < '0' || c > '9') return Status::Invalid;
        digits = true;
        const auto digit = std::uint64_t(c - '0');
        if (value > (UINT64_MAX - digit) / 10) return Status::Overflow;
        value = value * 10 + digit;
    }
    if (!digits) return Status::Invalid;
    assert(text.size == kSequenceDigits);
    assert(digits);
    out = value;
    return Status::Ok;
}
} // namespace bolt::wire::soup_bin_tcp
