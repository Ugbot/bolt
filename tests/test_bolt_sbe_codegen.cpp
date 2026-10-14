#include "data/sbe-codegen/features.h"
#include "data/sbe-codegen/features-big.h"
#include "data/sbe-codegen/historical-limits.h"
#include "data/sbe-codegen/nested.h"
#include "data/sbe_fixtures.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); } } while (false)
using bolt::wire::ByteView;
using bolt::wire::sbe::Status;
namespace {
// Independent manual LE oracle: body offsets 0,2,8,16,17,18,24,32.
constexpr std::uint8_t features[]{
    40, 0, 8, 0, 0xbd, 2, 2, 0,
    42, 0, 1, 0, 2, 0, 3, 0, 0x15, 0xcd, 0x5b, 7, 0, 0, 0, 0,
    2, 5, 'A', 'B', 0, 0, 0, 0, 0, 0, 0xc0, 0x7f, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 4, 0xc0
};
constexpr std::uint8_t features_big[]{
    0, 40, 0, 8, 2, 0xbd, 0, 2,
    0, 42, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 7, 0x5b, 0xcd, 0x15,
    2, 5, 'A', 'B', 0, 0, 0, 0, 0x7f, 0xc0, 0, 0, 0, 0, 0, 0,
    0xc0, 4, 0, 0, 0, 0, 0, 0
};
static_assert(sizeof(features) == 48 && sizeof(features_big) == 48);

struct FeaturesVisitor {
    std::uint32_t calls = 0;
    template <typename View> Status on_message_Features(View view) noexcept {
        assert(view.block.bytes.size >= 28);
        assert(view.block.acting_version <= 2);
        ++calls;
        std::uint16_t id = 0, number = 0;
        CHECK(view.get_id(id) == Status::Ok && id == 42);
        for (std::uint32_t i = 0; i < 3; ++i) {
            CHECK(view.get_numbers(i, number) == Status::Ok && number == i + 1);
        }
        CHECK(view.get_numbers(3, number) == Status::InvalidInput && number == 3);
        std::int64_t mantissa = 0;
        std::int8_t exponent = 0;
        CHECK(view.get_price_mantissa(mantissa) == Status::Ok && mantissa == 123456789);
        CHECK(view.get_price_exponent(exponent) == Status::Ok && exponent == -4);
        // Derive the generated enum type from the getter's owning namespace.
        bool indicative = false;
        CHECK(view.has_flags_Indicative(indicative) == Status::Ok && indicative);
        char symbol = 'x';
        CHECK(view.get_symbol(2, symbol) == Status::Ok && symbol == '\0');
        float real = 9.0f;
        CHECK(view.get_real(real) == Status::Null && real == 9.0f);
        double added = 9.0;
        std::int32_t constant = 99;
        if (view.block.acting_version == 2) {
            CHECK(view.get_added(added) == Status::Ok && added == -2.5);
            CHECK(view.get_constant(constant) == Status::Ok && constant == -7);
        } else {
            CHECK(view.get_added(added) == Status::Absent && added == 9.0);
            CHECK(view.get_constant(constant) == Status::Absent && constant == 99);
        }
        return Status::Ok;
    }
};

struct LimitsVisitor {
    std::uint32_t entries = 0;
    Status on_message_Limits(generated_historical_limits::message_Limits_view view) noexcept {
        assert(view.block.bytes.size == 11);
        assert(view.block.acting_version == 9);
        std::uint64_t time = 0;
        std::uint8_t indicator = 1;
        CHECK(view.get_TransactTime(time) == Status::Ok && time == 1633099253937623627ULL);
        CHECK(view.get_MatchEventIndicator(indicator) == Status::Ok && indicator == 0);
        return Status::Ok;
    }
    Status on_message_Limits_Entries(generated_historical_limits::message_Limits_Entries_view view) noexcept {
        assert(view.block.bytes.size == 32);
        assert(view.block.acting_version == 9);
        std::int64_t high = 123, low = 0, band = 0;
        std::int32_t security = 0;
        std::uint32_t sequence = 0;
        CHECK(view.get_HighLimitPrice(high) == Status::Null && high == 123);
        CHECK(view.get_LowLimitPrice(low) == Status::Ok && low == 9000000000000LL);
        CHECK(view.get_MaxPriceVariation(band) == Status::Ok && band == 10000000000LL);
        CHECK(view.get_SecurityID(security) == Status::Ok && security == 5620);
        CHECK(view.get_RptSeq(sequence) == Status::Ok && sequence == 1869);
        ++entries;
        return Status::Ok;
    }
};

struct NestedVisitor {
    std::int32_t sum = 0;
    std::uint32_t rows = 0, samples = 0, labels = 0, names = 0;
    bool reject_sample = false;
    Status on_message_Nested(generated_nested::message_Nested_view view) noexcept {
        assert(view.block.bytes.size >= 3);
        assert(view.block.acting_version <= 2);
        std::uint16_t id = 0;
        generated_nested::message_Nested_side_enum side{};
        CHECK(view.get_id(id) == Status::Ok && id == 42);
        const auto status = view.get_side(side);
        if (status != Status::Ok) return status;
        CHECK(side == generated_nested::message_Nested_side_enum::value_Buy);
        std::uint8_t flags = 99;
        if (view.block.acting_version == 2) CHECK(view.get_flags(flags) == Status::Ok && flags == 5);
        else CHECK(view.get_flags(flags) == Status::Absent && flags == 99);
        return Status::Ok;
    }
    Status on_message_Nested_rows(generated_nested::message_Nested_rows_view view) noexcept {
        assert(view.block.bytes.size >= 2);
        assert(rows < 2);
        std::int16_t value = 0;
        CHECK(view.get_value(value) == Status::Ok);
        CHECK(value == (rows == 0 ? -2 : 9));
        sum += value;
        ++rows;
        return Status::Ok;
    }
    Status on_message_Nested_rows_samples(generated_nested::message_Nested_rows_samples_view view) noexcept {
        assert(view.block.bytes.size >= 4);
        assert(samples == 0);
        float sample = 0;
        CHECK(view.get_sample(sample) == Status::Ok && sample == 1.5f);
        ++samples;
        return reject_sample ? Status::InvalidInput : Status::Ok;
    }
    Status on_message_Nested_rows_label(ByteView value) noexcept {
        assert(labels < 2);
        assert(value.data != nullptr || value.size == 0);
        CHECK(value.size == (labels == 0 ? 3u : 0u));
        if (labels == 0) CHECK(std::memcmp(value.data, "abc", 3) == 0);
        ++labels;
        return Status::Ok;
    }
    Status on_message_Nested_name(ByteView value) noexcept {
        assert(names == 0);
        assert(value.data != nullptr || value.size == 0);
        CHECK(value.size == 3 && std::memcmp(value.data, "xyz", 3) == 0);
        ++names;
        return Status::Ok;
    }
};

void generated_features_endian_and_versions() {
    FeaturesVisitor visitor{};
    CHECK(generated_features::decode_Features({features, sizeof(features)}, visitor) == Status::Ok);
    CHECK(generated_features_big::decode_Features({features_big, sizeof(features_big)}, visitor) == Status::Ok);
    CHECK(visitor.calls == 2);
    std::uint8_t old[36]{};
    std::memcpy(old, features, sizeof(old));
    old[0] = 28; old[6] = 1;
    CHECK(generated_features::decode_Features({old, sizeof(old)}, visitor) == Status::Ok);
    CHECK(visitor.calls == 3);
    for (std::uint32_t length = 0; length < sizeof(features); ++length) {
        FeaturesVisitor next{};
        CHECK(generated_features::decode_Features({features, length}, next) != Status::Ok);
    }
    CHECK(generated_features::decode_Features({nullptr, 1}, visitor) == Status::InvalidInput);
    generated_features::message_Features_view view{{{features + 8, 40}, 2}};
    generated_features::message_Features_side_enum side{};
    CHECK(view.get_side(side) == Status::Ok && side == generated_features::message_Features_side_enum::value_Sell);
    std::uint8_t changed[48]{};
    std::memcpy(changed, features, sizeof(changed));
    changed[24] = 3;
    view.block = {{changed + 8, 40}, 1};
    CHECK(view.get_side(side) == Status::InvalidInput && side == generated_features::message_Features_side_enum::value_Sell);
    changed[24] = 99; view.block.acting_version = 2;
    CHECK(view.get_side(side) == Status::InvalidInput);
}

void generated_historical_and_nested_oracles() {
    LimitsVisitor limits{};
    CHECK(generated_historical_limits::decode_Limits({sbe_fixtures::limits50, sizeof(sbe_fixtures::limits50)}, limits) == Status::Ok);
    CHECK(limits.entries == 1);
    for (std::uint32_t length = 0; length < sizeof(sbe_fixtures::limits50); ++length) {
        LimitsVisitor next{};
        CHECK(generated_historical_limits::decode_Limits({sbe_fixtures::limits50, length}, next) != Status::Ok);
    }
    NestedVisitor nested{};
    CHECK(generated_nested::decode_Nested({sbe_fixtures::nested, sizeof(sbe_fixtures::nested)}, nested) == Status::Ok);
    CHECK(nested.rows == 2 && nested.samples == 1 && nested.labels == 2 && nested.names == 1 && nested.sum == 7);
    for (std::uint32_t length = 0; length < sizeof(sbe_fixtures::nested); ++length) {
        NestedVisitor next{};
        CHECK(generated_nested::decode_Nested({sbe_fixtures::nested, length}, next) != Status::Ok);
    }
    NestedVisitor reject{};
    reject.reject_sample = true;
    CHECK(generated_nested::decode_Nested({sbe_fixtures::nested, sizeof(sbe_fixtures::nested)}, reject) == Status::InvalidInput);
    CHECK(reject.rows == 1 && reject.samples == 1 && reject.labels == 0 && reject.names == 0);
}

void generated_malformed_and_older_nested() {
    std::uint8_t changed[43]{};
    std::memcpy(changed, sbe_fixtures::nested, sizeof(sbe_fixtures::nested));
    NestedVisitor unknown{};
    changed[10] = 99;
    CHECK(generated_nested::decode_Nested({changed, 42}, unknown) == Status::InvalidInput);
    changed[10] = 1; changed[14] = 0xff; changed[15] = 1;
    NestedVisitor count{};
    CHECK(generated_nested::decode_Nested({changed, 42}, count) == Status::LimitExceeded);
    std::memcpy(changed, sbe_fixtures::nested, sizeof(sbe_fixtures::nested));
    NestedVisitor tail{};
    CHECK(generated_nested::decode_Nested({changed, 43}, tail) == Status::InvalidInput);
    std::memmove(changed + 11, changed + 12, 30);
    changed[0] = 3; changed[6] = 1;
    NestedVisitor older{};
    CHECK(generated_nested::decode_Nested({changed, 41}, older) == Status::Ok);
    CHECK(older.sum == 7 && older.names == 1);
    changed[6] = 3;
    NestedVisitor future{};
    CHECK(generated_nested::decode_Nested({changed, 41}, future) == Status::UnsupportedVersion);
}

void scalar_constant_enum_optional_and_referenced_composite() {
    const std::uint8_t bytes[]{0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x7f,0xff,0,0x34,0x12};
    generated_features::message_Other_view view{{{bytes, sizeof(bytes)}, 2}};
    std::int64_t price = 99;
    generated_features::message_Other_direction_enum direction{};
    auto optional = generated_features::message_Other_optionalDirection_enum::value_Sell;
    std::uint16_t code = 0;
    CHECK(view.get_price(price) == Status::Null && price == 99);
    CHECK(view.get_direction(direction) == Status::Ok && direction == generated_features::message_Other_direction_enum::value_Sell);
    CHECK(view.get_optionalDirection(optional) == Status::Null);
    CHECK(optional == generated_features::message_Other_optionalDirection_enum::value_Sell);
    CHECK(view.get_code(code) == Status::Ok && code == 0x1234);
    const std::uint8_t referenced[]{0x15,0xcd,0x5b,7,0,0,0,0,'U','S','D'};
    generated_features::message_Referenced_view composite{{{referenced, sizeof(referenced)}, 2}};
    std::int8_t exponent = 0;
    char currency = 0;
    CHECK(composite.get_price_amount_mantissa(price) == Status::Ok && price == 123456789);
    CHECK(composite.get_price_amount_exponent(exponent) == Status::Ok && exponent == -4);
    CHECK(composite.get_price_currency(2, currency) == Status::Ok && currency == 'D');
}
} // namespace
int main() {
    generated_features_endian_and_versions();
    generated_historical_and_nested_oracles();
    generated_malformed_and_older_nested();
    scalar_constant_enum_optional_and_referenced_composite();
    std::puts("Bolt generated SBE: independent bytes, historical fields, nested/version/null/enum/error oracles passed");
    return 0;
}
