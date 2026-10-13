// Paired bounded SBE core versus an independent bounds-checked typed decoder.
// This microbenchmark is NOT generated-code or all-format parity evidence.
#include "bolt/bolt_port.h"
#include "bolt/wire/sbe_fields.h"
#include "bolt/wire/sbe_groups.h"
#include "../tests/data/sbe_fixtures.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace bolt::wire;
using namespace bolt::wire::sbe;
constexpr auto LE = ByteOrder::Little;
constexpr std::uint32_t kMessages = 1024, kPairs = 400, kRounds = 128;
constexpr std::uint32_t kMessageBytes = sizeof(sbe_fixtures::limits50);
struct Row { std::uint8_t bytes[kMessageBytes]; };
alignas(64) static Row rows[kMessages];
static double reference_ns[kPairs]{}, bounded_ns[kPairs]{};
using Decode = bool (*)(ByteView, std::uint64_t&) noexcept;

BOLT_NOINLINE bool bounded_decode(ByteView bytes, std::uint64_t& out) noexcept {
    Message message{};
    if (open_message<LE>(bytes, {1, 50, 11, 9, 9}, message) != Status::Ok) return false;
    std::uint64_t time = 0;
    std::uint8_t indicator = 0;
    Group group{};
    Block entry{};
    if (read_field<LE>(message.root, 0, 0, time) != Status::Ok ||
        read_field<LE>(message.root, 8, 0, indicator) != Status::Ok ||
        open_group<LE, std::uint16_t, std::uint8_t, 255>(
            message.tail, 32, 9, 1, group) != Status::Ok || group.remaining != 1 ||
        next_entry(message.tail, group, entry) != Status::Ok) return false;
    std::int64_t high = 0, low = 0, band = 0;
    std::uint32_t sequence = 0;
    std::int32_t security = 0;
    const auto high_status = read_optional<LE>(entry, 0, 0, INT64_MAX, high);
    if ((high_status != Status::Ok && high_status != Status::Null) ||
        read_field<LE>(entry, 8, 0, low) != Status::Ok ||
        read_field<LE>(entry, 16, 0, band) != Status::Ok ||
        read_field<LE>(entry, 24, 0, security) != Status::Ok ||
        read_field<LE>(entry, 28, 0, sequence) != Status::Ok ||
        finish_message(message.tail) != Status::Ok) return false;
    out = time + indicator + static_cast<std::uint64_t>(high) +
        static_cast<std::uint64_t>(low) + static_cast<std::uint64_t>(band) +
        static_cast<std::uint64_t>(security) + sequence + (high_status == Status::Null);
    return true;
}

// Direct memcpy reads are independent of Bolt's reader, with explicit endian
// conversion. All dynamic bounds and identity checks match bounded_decode.
template <typename T> T reference_le(const std::uint8_t* data) noexcept {
    T value{};
    if constexpr (std::endian::native == std::endian::little) {
        std::memcpy(&value, data, sizeof(T));
    } else {
        std::uint8_t reversed[sizeof(T)]{};
        for (std::uint32_t i = 0; i < sizeof(T); ++i) reversed[i] = data[sizeof(T) - i - 1];
        std::memcpy(&value, reversed, sizeof(T));
    }
    return value;
}
BOLT_NOINLINE bool reference_decode(ByteView bytes, std::uint64_t& out) noexcept {
    if (bytes.data == nullptr || bytes.size < 8 || bytes.size > 16 * 1024 * 1024) return false;
    const auto* p = bytes.data;
    const auto root = reference_le<std::uint16_t>(p);
    if (reference_le<std::uint16_t>(p + 2) != 50 || reference_le<std::uint16_t>(p + 4) != 1 ||
        reference_le<std::uint16_t>(p + 6) != 9 || root < 11 || root > bytes.size - 8) return false;
    const auto dimensions = std::uint32_t(8 + root);
    if (bytes.size - dimensions < 3) return false;
    const auto group = reference_le<std::uint16_t>(p + dimensions);
    const auto count = p[dimensions + 2];
    if (count != 1 || group < 32 || group != bytes.size - dimensions - 3) return false;
    const auto* entry = p + dimensions + 3;
    const auto high = reference_le<std::int64_t>(entry);
    out = reference_le<std::uint64_t>(p + 8) + p[16] +
        (high == INT64_MAX ? 0 : static_cast<std::uint64_t>(high)) +
        static_cast<std::uint64_t>(reference_le<std::int64_t>(entry + 8)) +
        static_cast<std::uint64_t>(reference_le<std::int64_t>(entry + 16)) +
        static_cast<std::uint64_t>(reference_le<std::int32_t>(entry + 24)) +
        reference_le<std::uint32_t>(entry + 28) + (high == INT64_MAX);
    return true;
}

BOLT_NOINLINE double sample(Decode decode, std::uint64_t& checksum) noexcept {
    std::uint64_t sum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t round = 0; round < kRounds; ++round) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        for (std::uint32_t row = 0; row < kMessages; ++row) {
            std::uint64_t value = 0;
            if (!decode({rows[row].bytes, kMessageBytes}, value)) std::abort();
            sum += value;
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    checksum += sum;
    return std::chrono::duration<double, std::nano>(finish - start).count() / (kMessages * kRounds);
}

void initialize_and_check() noexcept {
    for (std::uint32_t row = 0; row < kMessages; ++row) {
        std::memcpy(rows[row].bytes, sbe_fixtures::limits50, kMessageBytes);
        rows[row].bytes[50] = static_cast<std::uint8_t>(row);
        rows[row].bytes[51] = static_cast<std::uint8_t>(row >> 8);
        std::uint64_t a = 0, b = 0;
        if (!reference_decode({rows[row].bytes, kMessageBytes}, a) ||
            !bounded_decode({rows[row].bytes, kMessageBytes}, b) || a != b) std::abort();
    }
    // Compare failure behavior on every truncated prefix of the actual fixture.
    for (std::uint32_t size = 0; size < kMessageBytes; ++size) {
        std::uint64_t a = 999, b = 999;
        if (reference_decode({rows[0].bytes, size}, a) || bounded_decode({rows[0].bytes, size}, b) ||
            a != 999 || b != 999) std::abort();
    }
}

int main() {
    initialize_and_check();
    std::uint64_t reference_sum = 0, bounded_sum = 0;
    for (std::uint32_t warm = 0; warm < 16; ++warm) {
        sample(reference_decode, reference_sum);
        sample(bounded_decode, bounded_sum);
    }
    for (std::uint32_t pair = 0; pair < kPairs; ++pair) {
        if ((pair & 1) == 0) {
            reference_ns[pair] = sample(reference_decode, reference_sum);
            bounded_ns[pair] = sample(bounded_decode, bounded_sum);
        } else {
            bounded_ns[pair] = sample(bounded_decode, bounded_sum);
            reference_ns[pair] = sample(reference_decode, reference_sum);
        }
    }
    if (reference_sum != bounded_sum) return 1;
    std::puts("pair,first,reference_ns_per_message,bounded_ns_per_message");
    for (std::uint32_t pair = 0; pair < kPairs; ++pair)
        std::printf("%u,%s,%.6f,%.6f\n", pair, (pair & 1) ? "bounded" : "reference", reference_ns[pair], bounded_ns[pair]);
    std::sort(reference_ns, reference_ns + kPairs);
    std::sort(bounded_ns, bounded_ns + kPairs);
    std::fprintf(stderr, "messages=%u rounds=%u pairs=%u checksum=%" PRIu64
        " reference_p50=%.6f reference_p99=%.6f bounded_p50=%.6f bounded_p99=%.6f ratio_p50=%.6f ratio_p99=%.6f\n",
        kMessages, kRounds, kPairs, bounded_sum, reference_ns[kPairs / 2], reference_ns[kPairs * 99 / 100],
        bounded_ns[kPairs / 2], bounded_ns[kPairs * 99 / 100],
        bounded_ns[kPairs / 2] / reference_ns[kPairs / 2], bounded_ns[kPairs * 99 / 100] / reference_ns[kPairs * 99 / 100]);
    return 0;
}
