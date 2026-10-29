#include "bolt/bolt_nested_column.h"
#include "bolt/bolt_nested_type_wire.h"
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
namespace {
namespace nr = bolt::nested_row;
using bolt::BoltColumn; using bolt::BoltType; using bolt::Arena;

BoltColumn fixture(Arena& arena) {
    const int64_t input[]{11,22,33}; const int32_t spans[]{0,2,2,2,3};
    auto* values = arena.allocate_array<int64_t>(3); std::memcpy(values,input,sizeof(input));
    auto* offsets = arena.allocate_array<int32_t>(5); std::memcpy(offsets,spans,sizeof(spans));
    auto* valid = static_cast<uint8_t*>(arena.allocate(1,1)); *valid = 0x0D;
    auto leaf = BoltColumn::make_flat(values,nullptr,3,BoltType::Int64);
    return BoltColumn::make_list(&leaf,offsets,4,valid,&arena);
}
nr::NamedType named(const BoltColumn& source) {
    nr::NamedType type{};
    EXPECT_TRUE(nr::tree_from_column(source,&type.values));
    for (uint32_t i = 0; i < type.values.n; ++i) {
        type.values.nodes[i].nullable = 1;
        type.fields[i].field_id = static_cast<int32_t>(i+1u);
        EXPECT_TRUE(nr::set_field_name(&type,i,i ? "element" : "list",i ? 7 : 4));
    }
    return type;
}
TEST(NestedColumn, TypedCarrierOwnsBytesAndPreservesNullAndEmpty) {
    Arena input, encoded, decoded;
    const auto source = fixture(input); auto type = named(source); BoltColumn bytes{}, back{};
    ASSERT_TRUE(nr::encode_column(type,source,&encoded,{},&bytes));
    EXPECT_EQ(bytes.type,BoltType::List); EXPECT_EQ(bytes.format,bolt::ColumnFormat::VarBinary);
    const auto* offsets = static_cast<const int32_t*>(bytes.dict_child->data);
    EXPECT_GT(offsets[1],offsets[0]); EXPECT_EQ(offsets[1],offsets[2]); EXPECT_GT(offsets[3],offsets[2]);
    input.reset();
    std::memset(input.allocate(4096,1),0xDA,4096);
    ASSERT_TRUE(nr::decode_column(type,bytes,&decoded,{},&back));
    ASSERT_EQ(back.length,4); EXPECT_FALSE(back.is_null(0)); EXPECT_TRUE(back.is_null(1)); EXPECT_FALSE(back.is_null(2));
    EXPECT_EQ(back.list_offsets()[2],back.list_offsets()[3]);
    EXPECT_EQ(static_cast<const int64_t*>(back.list_element()->data)[2],33);
    // Decode an independent legacy Binary carrier: no durable-row format fork.
    bytes.type = BoltType::Binary; BoltColumn legacy{};
    ASSERT_TRUE(nr::decode_column(type.values,bytes,&decoded,{},&legacy));
    EXPECT_EQ(legacy.list_offsets()[4],3);
}
TEST(NestedColumn, EncodingMatchesEveryExistingDurableRowByte) {
    Arena input, output; const auto source = fixture(input); const auto type = named(source); BoltColumn bytes{};
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&bytes));
    const auto* offsets = static_cast<const int32_t*>(bytes.dict_child->data);
    // Literal legacy bytes, independent of either current encode entry point.
    const uint8_t golden[]{
        1,2,0,0,0,1,11,0,0,0,0,0,0,0,1,22,0,0,0,0,0,0,0,
        1,0,0,0,0,
        1,1,0,0,0,1,33,0,0,0,0,0,0,0};
    static_assert(sizeof(golden) == 42);
    const int32_t golden_offsets[]{0,23,23,28,42};
    EXPECT_EQ(std::memcmp(offsets,golden_offsets,sizeof(golden_offsets)),0);
    EXPECT_EQ(std::memcmp(bytes.data,golden,sizeof(golden)),0);
    for (int64_t row = 0; row < source.length; ++row) {
        if (source.is_null(row)) { EXPECT_EQ(offsets[row],offsets[row+1]); continue; }
        const auto size = nr::encode_row(type.values,source,row,nullptr,0);
        std::vector<uint8_t> expected(size);
        ASSERT_EQ(nr::encode_row(type.values,source,row,expected.data(),size),size);
        ASSERT_EQ(offsets[row+1]-offsets[row],size);
        EXPECT_EQ(std::memcmp(static_cast<const uint8_t*>(bytes.data)+offsets[row],expected.data(),size),0);
    }
}
TEST(NestedColumn, BoundsAndMalformedValuesNeverPublish) {
    Arena input, output; const auto source = fixture(input); auto type = named(source);
    BoltColumn bytes{}, unchanged{}; unchanged.length = 713;
    auto limit = nr::ColumnLimits{}; limit.encoded_bytes = 1;
    EXPECT_FALSE(nr::encode_column(type,source,&output,limit,&unchanged)); EXPECT_EQ(unchanged.length,713);
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&bytes));
    limit = {}; limit.values = 1;
    EXPECT_FALSE(nr::decode_column(type,bytes,&output,limit,&unchanged)); EXPECT_EQ(unchanged.length,713);
    auto* offsets = static_cast<int32_t*>(bytes.dict_child->data);
    const auto saved = offsets[4]; offsets[4] = offsets[3]-1;
    EXPECT_FALSE(nr::decode_column(type,bytes,&output,{},&unchanged)); EXPECT_EQ(unchanged.length,713);
    offsets[4] = saved + 1; static_cast<uint8_t*>(bytes.data)[saved] = 0;
    EXPECT_FALSE(nr::decode_column(type,bytes,&output,{},&unchanged)); EXPECT_EQ(unchanged.length,713);
    offsets[4] = saved;
    auto* data = static_cast<uint8_t*>(bytes.data); const auto flag = data[0]; data[0] = 2;
    EXPECT_FALSE(nr::decode_column(type,bytes,&output,{},&unchanged)); EXPECT_EQ(unchanged.length,713);
    data[0] = flag;
    // A list count cannot force a billion-iteration decode from a tiny row.
    uint32_t huge = 0x3FFFFFFFu; std::memcpy(data+1,&huge,4);
    EXPECT_FALSE(nr::decode_column(type,bytes,&output,{},&unchanged)); EXPECT_EQ(unchanged.length,713);
}
TEST(NestedColumn, SchemaRefusesCyclesSharedNodesOrphansAndAmbiguousNames) {
    Arena input; const auto source = fixture(input); auto type = named(source);
    ASSERT_TRUE(nr::validate_named_type(type));
    type.values.nodes[0].first_child = 0; EXPECT_FALSE(nr::validate_named_type(type));
    type = named(source); ++type.values.n; EXPECT_FALSE(nr::validate_named_type(type));
    type = named(source); type.fields[1].field_id = type.fields[0].field_id;
    EXPECT_FALSE(nr::validate_named_type(type));
    BoltColumn children[]{source,source}; const auto structure = BoltColumn::make_struct(children,2,4,nullptr,&input);
    nr::NamedType object{}; ASSERT_TRUE(nr::tree_from_column(structure,&object.values));
    ASSERT_TRUE(nr::set_field_name(&object,1,"duplicate",9)); ASSERT_TRUE(nr::set_field_name(&object,2,"duplicate",9));
    EXPECT_FALSE(nr::validate_named_type(object));
    for (uint32_t i = 0; i < object.values.n; ++i) object.values.nodes[i].nullable = 1;
    ASSERT_TRUE(nr::set_field_name(&object,2,"other",5));
    EXPECT_TRUE(nr::validate_named_type(object));
    object.values.nodes[0].nullable = 0; EXPECT_FALSE(nr::validate_named_type(object));
    EXPECT_FALSE(nr::set_field_name(&object,1,"a\0b",3));
}
TEST(NestedColumn, DecimalScaleSurvivesExistingAndTypedCodecs) {
    Arena input, output; int64_t values[]{12345,-12345};
    auto decimal = BoltColumn::make_flat(values,nullptr,2,BoltType::Decimal64); decimal.decimal_scale = 4;
    const auto source = BoltColumn::make_struct(&decimal,1,2,nullptr,&input);
    auto type = named(source); std::strcpy(type.fields[1].name,"amount"); type.fields[1].precision = 18;
    ASSERT_EQ(type.values.nodes[1].scale,4); BoltColumn encoded{}, back{};
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&encoded));
    ASSERT_TRUE(nr::decode_column(type,encoded,&output,{},&back));
    ASSERT_EQ(back.child_at(0)->decimal_scale,4);
    EXPECT_EQ(static_cast<const int64_t*>(back.child_at(0)->data)[1],-12345);
}
TEST(NestedColumn, EmptyByteLeavesAndMalformedInputDescriptors) {
    Arena input, output;
    int32_t offsets[]{0,0,0};
    auto empty = BoltColumn::make_var_binary(nullptr,nullptr,offsets,2,BoltType::Utf8,&input);
    const auto source = BoltColumn::make_struct(&empty,1,2,nullptr,&input);
    auto type = named(source); BoltColumn carrier{}, decoded{};
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&carrier));
    ASSERT_TRUE(nr::decode_column(type,carrier,&output,{},&decoded));
    ASSERT_TRUE(nr::encode_column(type,decoded,&output,{},&carrier));
    EXPECT_EQ(decoded.child_at(0)->length,2);
    auto invalid = source; invalid.data = nullptr; decoded.length = 777;
    EXPECT_FALSE(nr::encode_column(type,invalid,&output,{},&decoded)); EXPECT_EQ(decoded.length,777);
    invalid = source; invalid.validity_offset = -1;
    EXPECT_FALSE(nr::encode_column(type,invalid,&output,{},&decoded)); EXPECT_EQ(decoded.length,777);
}

TEST(NestedColumn, ArenaExhaustionDoesNotPublish) {
    Arena input; const auto source = fixture(input); const auto type = named(source);
    Arena tiny{{128,128,64,false}}; tiny.set_growth_enabled(false);
    ASSERT_NE(tiny.allocate(128,1),nullptr); tiny.reset();
    BoltColumn out{}; out.length = 777;
    EXPECT_FALSE(nr::encode_column(type,source,&tiny,{},&out)); EXPECT_EQ(out.length,777);
    Arena encoded; BoltColumn carrier{};
    ASSERT_TRUE(nr::encode_column(type,source,&encoded,{},&carrier));
    tiny.reset();
    EXPECT_FALSE(nr::decode_column(type,carrier,&tiny,{},&out)); EXPECT_EQ(out.length,777);
}

TEST(NestedColumn, NamedSchemaWireIsCanonicalAndRefusesTruncationOrUnknownVersion) {
    Arena input; auto type = named(fixture(input));
    ASSERT_TRUE(nr::set_field_name(&type,1,"a\"雪",5));
    type.fields[1].temporal = bolt::schema::Temporal::Timestamp;
    type.fields[1].unit = bolt::schema::Unit::Nanosecond;
    uint32_t size = 0; ASSERT_TRUE(nr::named_type_wire_size(type,&size));
    std::vector<uint8_t> wire(size,0xDA); uint32_t written = 777;
    EXPECT_FALSE(nr::named_type_wire_encode(type,wire.data(),size-1,&written)); EXPECT_EQ(written,777u);
    EXPECT_EQ(wire[0],0xDA);
    ASSERT_TRUE(nr::named_type_wire_encode(type,wire.data(),size,&written)); EXPECT_EQ(written,size);
    EXPECT_EQ(wire[4],2); EXPECT_EQ(wire[5],0); // LE node count, independent of native padding.
    nr::NamedType decoded{}; ASSERT_TRUE(nr::named_type_wire_decode(wire.data(),size,&decoded));
    EXPECT_STREQ(decoded.fields[1].name,"a\"雪"); EXPECT_EQ(decoded.fields[1].field_id,2);
    EXPECT_EQ(decoded.fields[1].unit,bolt::schema::Unit::Nanosecond);
    std::vector<uint8_t> again(size); ASSERT_TRUE(nr::named_type_wire_encode(decoded,again.data(),size,&written));
    EXPECT_EQ(again,wire);
    for (uint32_t cut = 0; cut < size; ++cut) {
        decoded.values.n = 777; EXPECT_FALSE(nr::named_type_wire_decode(wire.data(),cut,&decoded));
        EXPECT_EQ(decoded.values.n,777u);
    }
    wire[3] = '2'; EXPECT_FALSE(nr::named_type_wire_decode(wire.data(),size,&decoded));
    EXPECT_EQ(decoded.values.n,777u);
    wire[3] = '1'; wire.push_back(0);
    EXPECT_FALSE(nr::named_type_wire_decode(wire.data(),size+1,&decoded)); EXPECT_EQ(decoded.values.n,777u);
}

TEST(NestedColumn, NamedMapsRequirePresentEntriesAndKeys) {
    Arena input, output; char text[]{'k'}; int32_t spans[]{0,1}; int64_t number[]{42}; uint8_t key_valid = 1;
    BoltColumn fields[]{BoltColumn::make_var_binary(text,&key_valid,spans,1,BoltType::Utf8,&input),
                        BoltColumn::make_flat(number,nullptr,1,BoltType::Int64)};
    auto entry = BoltColumn::make_struct(fields,2,1,nullptr,&input);
    auto source = BoltColumn::make_list(&entry,spans,1,nullptr,&input); source.type = BoltType::Map;
    nr::NamedType type{}; ASSERT_TRUE(nr::tree_from_column(source,&type.values));
    const char* names[]{"map","entries","key","value"};
    for (uint32_t i = 0; i < type.values.n; ++i) {
        type.values.nodes[i].nullable = 1;
        ASSERT_TRUE(nr::set_field_name(&type,i,names[i],static_cast<uint32_t>(std::strlen(names[i]))));
    }
    EXPECT_FALSE(nr::validate_named_type(type));
    type.fields[1].nullable = 0; type.fields[2].nullable = 0;
    ASSERT_TRUE(nr::validate_named_type(type)); BoltColumn carrier{}, back{};
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&carrier));
    ASSERT_TRUE(nr::decode_column(type,carrier,&output,{},&back));
    EXPECT_EQ(static_cast<const int64_t*>(back.list_element()->child_at(1)->data)[0],42);
    key_valid = 0; back.length = 777;
    EXPECT_FALSE(nr::encode_column(type,source,&output,{},&back)); EXPECT_EQ(back.length,777);
}

TEST(NestedColumn, UtcAndLocalTimestampRemainDistinctAcrossSchemaWire) {
    Arena arena; auto local = named(fixture(arena));
    local.fields[1].temporal = bolt::schema::Temporal::Timestamp;
    local.fields[1].unit = bolt::schema::Unit::Nanosecond;
    auto utc = local; utc.fields[1].temporal = bolt::schema::Temporal::TimestampUtc;
    uint8_t first[nr::kMaxNamedTypeBytes], second[nr::kMaxNamedTypeBytes]; uint32_t n = 0, m = 0;
    ASSERT_TRUE(nr::named_type_wire_encode(local,first,sizeof(first),&n));
    ASSERT_TRUE(nr::named_type_wire_encode(utc,second,sizeof(second),&m));
    ASSERT_EQ(n,m); EXPECT_NE(std::memcmp(first,second,n),0);
    nr::NamedType decoded{}; ASSERT_TRUE(nr::named_type_wire_decode(second,m,&decoded));
    EXPECT_EQ(decoded.fields[1].temporal,bolt::schema::Temporal::TimestampUtc);
    EXPECT_EQ(decoded.fields[1].unit,bolt::schema::Unit::Nanosecond);
    utc.fields[1].temporal = static_cast<bolt::schema::Temporal>(6);
    EXPECT_FALSE(nr::validate_named_type(utc));
}

TEST(NestedColumn, RequirednessAndFixedWidthAreValidated) {
    Arena input, output; auto source = fixture(input); auto type = named(source); BoltColumn encoded{}, back{};
    ASSERT_TRUE(nr::encode_column(type,source,&output,{},&encoded));
    type.fields[0].nullable = 0;
    EXPECT_FALSE(nr::decode_column(type,encoded,&output,{},&back));
    type = named(source); type.fields[1].fixed_length = 8;
    EXPECT_FALSE(nr::validate_named_type(type));
    type = named(source); type.fields[1].temporal = bolt::schema::Temporal::Timestamp;
    EXPECT_FALSE(nr::validate_named_type(type));
    type.fields[1].unit = bolt::schema::Unit::Nanosecond;
    EXPECT_TRUE(nr::validate_named_type(type));
}
}  // namespace
