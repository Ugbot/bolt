#include "bolt/wire/sbe_fields.h"
#include "bolt/wire/sbe_groups.h"
#include "data/sbe_fixtures.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); } } while (false)
using namespace bolt::wire;
using namespace bolt::wire::sbe;
constexpr auto LE = ByteOrder::Little;
constexpr auto BE = ByteOrder::Big;
namespace {
bool decode_limits(ByteView bytes, std::uint64_t& result) {
    Message message{};
    if (open_message<LE>(bytes, {1, 50, 11, 9, 9}, message) != Status::Ok) return false;
    std::uint64_t time = 0;
    std::uint8_t indicator = 1;
    if (read_field<LE>(message.root, 0, 0, time) != Status::Ok ||
        read_field<LE>(message.root, 8, 0, indicator) != Status::Ok) return false;
    Group group{};
    if (open_group<LE, std::uint16_t, std::uint8_t, 255>(
        message.tail, 32, 9, 1, group) != Status::Ok || group.remaining != 1) return false;
    Block entry{};
    if (next_entry(message.tail, group, entry) != Status::Ok) return false;
    std::int64_t high = 123, low = 0, band = 0;
    std::int32_t security = 0;
    std::uint32_t sequence = 0;
    if (read_optional<LE>(entry, 0, 0, INT64_MAX, high) != Status::Null || high != 123 ||
        read_field<LE>(entry, 8, 0, low) != Status::Ok ||
        read_field<LE>(entry, 16, 0, band) != Status::Ok ||
        read_field<LE>(entry, 24, 0, security) != Status::Ok ||
        read_field<LE>(entry, 28, 0, sequence) != Status::Ok) return false;
    if (next_entry(message.tail, group, entry) != Status::End ||
        finish_message(message.tail) != Status::Ok) return false;
    CHECK(time == 1633099253937623627ull && indicator == 0);
    CHECK(low == 9000000000000ll && band == 10000000000ll);
    CHECK(security == 5620 && sequence == 1869);
    result = time + static_cast<std::uint64_t>(low + band) + sequence + static_cast<std::uint64_t>(security);
    return true;
}

void official_fixture_and_every_truncation() {
    constexpr auto size = static_cast<std::uint32_t>(sizeof(sbe_fixtures::limits50));
    std::uint8_t unaligned[size + 1]{};
    std::memcpy(unaligned + 1, sbe_fixtures::limits50, size);
    std::uint64_t result = 99;
    CHECK(decode_limits({unaligned + 1, size}, result) && result != 99);
    for (std::uint32_t length = 0; length < size; ++length) {
        result = 99;
        CHECK(!decode_limits({unaligned + 1, length}, result) && result == 99);
    }
}

bool decode_nested(ByteView bytes, std::int32_t& out) {
    Message message{};
    if (open_message<LE>(bytes, {700, 7, 4, 2, 2}, message) != Status::Ok) return false;
    std::uint16_t id = 0;
    std::uint8_t side = 0, flags = 0;
    if (read_field<LE>(message.root, 0, 0, id) != Status::Ok ||
        read_field<LE>(message.root, 2, 0, side) != Status::Ok ||
        read_field<LE>(message.root, 3, 2, flags) != Status::Ok) return false;
    CHECK(id == 42 && side == 1 && flags == 5);
    Group rows{};
    if (open_group<LE, std::uint16_t, std::uint16_t, 2>(
        message.tail, 2, 2, 1, rows) != Status::Ok || rows.remaining != 2) return false;
    std::int32_t sum = 0;
    for (std::uint32_t row = 0; row < 2; ++row) {
        Block entry{};
        std::int16_t value = 0;
        Group samples{};
        if (next_entry(message.tail, rows, entry) != Status::Ok ||
            read_field<LE>(entry, 0, 0, value) != Status::Ok ||
            open_group<LE, std::uint16_t, std::uint16_t, 1>(
                message.tail, 4, 2, 2, samples) != Status::Ok) return false;
        CHECK(samples.remaining == (row == 0 ? 1u : 0u));
        if (row == 0) {
            float sample = 0;
            if (next_entry(message.tail, samples, entry) != Status::Ok ||
                read_field<LE>(entry, 0, 0, sample) != Status::Ok) return false;
            CHECK(sample == 1.5f);
        }
        ByteView label{};
        if (read_data<LE, std::uint8_t, 3>(message.tail, label) != Status::Ok) return false;
        CHECK(label.size == (row == 0 ? 3u : 0u));
        if (row == 0) CHECK(std::memcmp(label.data, "abc", 3) == 0);
        sum += value;
    }
    ByteView name{};
    if (read_data<LE, std::uint16_t, 3>(message.tail, name) != Status::Ok ||
        finish_message(message.tail) != Status::Ok) return false;
    CHECK(name.size == 3 && std::memcmp(name.data, "xyz", 3) == 0);
    out = sum;
    return true;
}

void nested_groups_and_variable_data() {
    constexpr auto size = static_cast<std::uint32_t>(sizeof(sbe_fixtures::nested));
    std::int32_t result = 99;
    CHECK(decode_nested({sbe_fixtures::nested, size}, result) && result == 7);
    for (std::uint32_t length = 0; length < size; ++length) {
        result = 99;
        CHECK(!decode_nested({sbe_fixtures::nested, length}, result) && result == 99);
    }
}

void endian_fields_absence_and_null() {
    const std::uint8_t bytes[]{0xff, 0xfe, 0x12, 0x34, 0x56, 0x78,
        0x3f, 0xc0, 0, 0, 0xc0, 0x04, 0, 0, 0, 0, 0, 0, 0x7f, 0xc0, 0, 0};
    ByteReader reader{bytes, sizeof(bytes), 0};
    std::int16_t negative = 0;
    std::uint32_t number = 0;
    float real = 0;
    double wide = 0;
    CHECK(read_scalar<BE>(reader, negative) && negative == -2);
    CHECK(read_scalar<BE>(reader, number) && number == 0x12345678);
    CHECK(read_scalar<BE>(reader, real) && real == 1.5f);
    CHECK(read_scalar<BE>(reader, wide) && wide == -2.5);
    Block block{{bytes, sizeof(bytes)}, 1};
    CHECK(read_optional<BE>(block, 18, 0, std::numeric_limits<float>::quiet_NaN(), real) == Status::Null);
    CHECK(real == 1.5f);
    CHECK(read_field<BE>(block, UINT32_MAX, 2, number) == Status::Absent && number == 0x12345678);
    CHECK(read_field<BE>(block, UINT32_MAX, 1, number) == Status::Truncated && number == 0x12345678);
    ByteView raw{};
    CHECK(read_fixed_bytes(block, 10, 8, 1, raw) == Status::Ok && raw.size == 8);
    CHECK(read_fixed_bytes(block, 10, UINT32_MAX, 1, raw) == Status::Truncated && raw.size == 8);
    CHECK(read_fixed_bytes(block, 10, 8, 2, raw) == Status::Absent && raw.size == 8);
}

void identity_versions_and_transactionality() {
    const ByteView bytes{sbe_fixtures::limits50, sizeof(sbe_fixtures::limits50)};
    Message message{};
    message.header.template_id = 999;
    CHECK(open_message<LE>(bytes, {2, 50, 11, 9, 9}, message) == Status::UnexpectedSchema);
    CHECK(open_message<LE>(bytes, {1, 51, 11, 9, 9}, message) == Status::UnexpectedTemplate);
    CHECK(open_message<LE>(bytes, {1, 50, 11, 10, 11}, message) == Status::UnsupportedVersion);
    CHECK(open_message<LE>(bytes, {1, 50, 11, 0, 8}, message) == Status::UnsupportedVersion);
    CHECK(open_message<LE>(bytes, {1, 50, 12, 9, 9}, message) == Status::InvalidInput);
    CHECK(open_message<LE>(bytes, {1, 50, 11, 10, 9}, message) == Status::InvalidInput);
    CHECK((open_message<LE, 8>(bytes, {1, 50, 11, 9, 9}, message) == Status::LimitExceeded));
    CHECK(open_message<LE>({nullptr, 54}, {1, 50, 11, 9, 9}, message) == Status::InvalidInput);
    CHECK(message.header.template_id == 999 && message.tail.offset == 0);
    ByteReader reader{sbe_fixtures::limits50, 7, 0};
    Header header{};
    header.schema_id = 333;
    CHECK(read_header<LE>(reader, header) == Status::Truncated);
    CHECK(reader.offset == 0 && header.schema_id == 333);
    CHECK(finish_message({nullptr, 4, 0}) == Status::InvalidInput);
    CHECK(finish_message({sbe_fixtures::limits50, 54, 53}) == Status::InvalidInput);
    const std::uint8_t big[]{0, 4, 0, 7, 2, 0xbc, 0, 2, 0, 42, 1, 5};
    CHECK(open_message<BE>({big, sizeof(big)}, {700, 7, 4, 2, 2}, message) == Status::Ok);
    CHECK(message.header.schema_id == 700 && finish_message(message.tail) == Status::Ok);
}

void group_and_data_limits() {
    const std::uint8_t bytes[]{4, 0, 2, 0, 1, 2, 3, 4};
    ByteReader reader{bytes, sizeof(bytes), 0};
    Group group{77, 99, 1};
    CHECK((open_group<LE, std::uint16_t, std::uint16_t, 1>(reader, 4, 1, 1, group) == Status::LimitExceeded));
    CHECK((open_group<LE, std::uint16_t, std::uint16_t, 2>(reader, 4, 1, 1, group) == Status::Truncated));
    CHECK((open_group<LE, std::uint16_t, std::uint16_t, 2>(reader, 5, 1, 1, group) == Status::InvalidInput));
    CHECK((open_group<LE, std::uint16_t, std::uint16_t, 2>(reader, 4, 1, 9, group) == Status::LimitExceeded));
    CHECK(reader.offset == 0 && group.remaining == 99 && group.block_length == 77);
    Block entry{{bytes, 3}, 1};
    CHECK(next_entry(reader, group, entry) == Status::Truncated);
    CHECK(reader.offset == 0 && group.remaining == 99 && entry.bytes.size == 3);
    ByteView out{bytes, 3};
    CHECK((read_data<LE, std::uint16_t, 3>(reader, out) == Status::LimitExceeded));
    reader.size = 5;
    CHECK((read_data<LE, std::uint16_t, 4>(reader, out) == Status::Truncated));
    CHECK(reader.offset == 0 && out.size == 3);
    const std::uint8_t huge[]{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    reader = {huge, sizeof(huge), 0};
    CHECK((read_data<LE, std::uint64_t, UINT32_MAX>(reader, out) == Status::LimitExceeded));
    CHECK(reader.offset == 0 && out.size == 3);
    const std::uint8_t empty[]{0, 0, 0, 0};
    reader = {empty, sizeof(empty), 0};
    CHECK((open_group<LE, std::uint16_t, std::uint16_t, 2>(reader, 0, 1, 1, group) == Status::Ok));
    CHECK(next_entry(reader, group, entry) == Status::End && reader.offset == 4);
    CHECK(entry.bytes.size == 3);
}
} // namespace
int main() {
    official_fixture_and_every_truncation();
    nested_groups_and_variable_data();
    endian_fields_absence_and_null();
    identity_versions_and_transactionality();
    group_and_data_limits();
    std::puts("Bolt SBE: official oracle, nested groups/data, endian/version/null and bounded failures passed");
    return 0;
}
