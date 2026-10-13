#include "bolt/wire/byte_reader.h"
#include <cstdio>
#include <cstdlib>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); } } while (false)
using namespace bolt::wire;
namespace {
void exact_values() {
    const std::uint8_t bytes[]{0xff, 0x78, 0x56, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x90};
    ByteReader reader{bytes, sizeof(bytes), 1}; // Deliberately unaligned.
    std::uint64_t value = 0;
    CHECK(read_le(reader, value) && value == 0x90abcdef12345678ull);
    CHECK(reader.offset == sizeof(bytes));
    reader.offset = 1;
    std::uint16_t half = 0;
    std::uint32_t word = 0;
    CHECK(read_le(reader, half) && half == 0x5678);
    CHECK(read_le(reader, word) && word == 0xcdef1234u);
    reader.offset = 0;
    std::int8_t negative = 0;
    CHECK(read_le(reader, negative) && negative == -1);
    const std::uint8_t minimum[]{0, 0, 0, 0, 0, 0, 0, 0x80};
    reader = {minimum, sizeof(minimum), 0};
    std::int64_t signed_value = 0;
    CHECK(read_le(reader, signed_value) && signed_value == INT64_MIN);
    const std::uint8_t narrow[]{0xff, 0, 0x80, 0xff, 0xff, 0xff, 0xff};
    reader = {narrow, sizeof(narrow), 0};
    std::uint8_t byte = 0;
    std::int16_t signed_half = 0;
    std::int32_t signed_word = 0;
    CHECK(read_le(reader, byte) && byte == 255);
    CHECK(read_le(reader, signed_half) && signed_half == INT16_MIN);
    CHECK(read_le(reader, signed_word) && signed_word == -1);
}
void truncation_preserves_outputs() {
    const std::uint8_t bytes[8]{};
    for (std::uint32_t length = 0; length < sizeof(bytes); ++length) {
        ByteReader reader{bytes, length, 0};
        std::uint64_t value = 99;
        CHECK(!read_le(reader, value) && value == 99 && reader.offset == 0);
        ByteView view{bytes, 7};
        CHECK(!read_bytes(reader, 8, view) && view.data == bytes && view.size == 7);
        CHECK(!skip_bytes(reader, UINT32_MAX) && reader.offset == 0);
    }
}
void spans_and_bad_cursors() {
    const std::uint8_t bytes[]{1, 2, 3};
    ByteReader reader{bytes, sizeof(bytes), 0};
    ByteView view{};
    CHECK(read_bytes(reader, 2, view) && view.data == bytes && view.size == 2);
    CHECK(skip_bytes(reader, 1) && reader.offset == 3);
    CHECK(read_bytes(reader, 0, view) && view.data == bytes + 3 && view.size == 0);
    reader = {};
    CHECK(read_bytes(reader, 0, view) && view.data == nullptr);
    std::uint8_t value = 99;
    CHECK(!read_le(reader, value) && value == 99);
    reader = {nullptr, 8, 0};
    CHECK(!skip_bytes(reader, 0) && !read_le(reader, value));
    view = {bytes, 2};
    CHECK(!read_bytes(reader, 1, view) && view.data == bytes && view.size == 2);
    CHECK(reader.offset == 0 && value == 99);
    reader = {bytes, sizeof(bytes), UINT32_MAX};
    CHECK(!read_bytes(reader, 0, view) && reader.offset == UINT32_MAX);
    CHECK(view.data == bytes && view.size == 2);
}
}
int main() {
    exact_values(); truncation_preserves_outputs(); spans_and_bad_cursors();
    std::puts("Bolt byte reader: exact LE values, bounds and transactional failures passed");
    return 0;
}
