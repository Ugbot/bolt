#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/bolt_arena.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace {
namespace pq=bolt::ingest::parquet;
using S=pq::PqMetaParseStatus;
enum class Shape { Plain,Dot,Timestamp,LocalTime,LegacyTimestamp,LegacyUnsigned,
    MissingUtc,MissingUnit,DuplicateUtc,UnknownLogical,PhysicalMismatch,InvalidPhysical,Contradictory,
    MissingRepetition,LongName,EmbeddedNul,Nested,DuplicateName,IntMissingSign,BadIntWidth,
    Decimal,DecimalMissingScale,DecimalBadPrecision,DecimalConflict };
// Independent compact-Thrift encoder: long field headers avoid depending on
// the production writer or its parser's delta-header implementation.
struct Bytes {
    std::vector<std::uint8_t> bytes;
    void var(std::uint64_t value) {
        for(unsigned i=0;i<10;++i){const auto b=static_cast<std::uint8_t>(value&127u);value>>=7;
            bytes.push_back(static_cast<std::uint8_t>(b | (value?128u:0u)));if(!value)return;}
        std::abort();
    }
    void zig(std::int64_t value){var((static_cast<std::uint64_t>(value)<<1)^std::uint64_t(value>>63));}
    void field(std::uint16_t id,std::uint8_t type){bytes.push_back(type);zig(id);}
    void number(std::uint16_t id,std::int64_t value,std::uint8_t type=pq::kTcI32){field(id,type);zig(value);}
    void name(const std::string& value){field(4,pq::kTcBinary);var(value.size());bytes.insert(bytes.end(),value.begin(),value.end());}
    void stop(){bytes.push_back(0);}
    void list(std::uint32_t count){bytes.push_back(0xfc);var(count);}
    void overflow(std::uint8_t low){bytes.push_back(static_cast<std::uint8_t>(low|128u));
        for(unsigned i=0;i<8;++i)bytes.push_back(0x80);bytes.push_back(0x02);}
};
void logical(Bytes& b,Shape shape) {
    if(shape==Shape::Plain || shape==Shape::Dot || shape==Shape::LongName || shape==Shape::EmbeddedNul ||
        shape==Shape::Nested || shape==Shape::DuplicateName || shape==Shape::MissingRepetition ||
        shape==Shape::InvalidPhysical || shape==Shape::LegacyTimestamp || shape==Shape::LegacyUnsigned)return;
    b.field(10,pq::kTcStruct);
    const bool decimal=shape==Shape::Decimal || shape==Shape::DecimalMissingScale || shape==Shape::DecimalBadPrecision || shape==Shape::DecimalConflict;
    const bool integer=shape==Shape::IntMissingSign || shape==Shape::BadIntWidth;
    b.field(decimal?5:integer?10:shape==Shape::UnknownLogical?99:shape==Shape::LocalTime?7:8,pq::kTcStruct);
    if(decimal) {
        if(shape!=Shape::DecimalMissingScale)b.number(1,2);
        b.number(2,shape==Shape::DecimalBadPrecision?0:18);
    } else if(integer) {
        b.field(1,pq::kTcByte);b.bytes.push_back(shape==Shape::BadIntWidth?7:64);
        if(shape!=Shape::IntMissingSign)b.field(2,pq::kTcTrue);
    } else {
        if(shape!=Shape::MissingUtc)b.field(1,shape==Shape::LocalTime?pq::kTcFalse:pq::kTcTrue);
        if(shape==Shape::DuplicateUtc)b.field(1,pq::kTcFalse);
        if(shape!=Shape::MissingUnit){b.field(2,pq::kTcStruct);b.field(3,pq::kTcStruct);b.stop();b.stop();}
    }
    b.stop();b.stop();
}
void leaf(Bytes& b,Shape shape) {
    const bool decimal=shape==Shape::Decimal || shape==Shape::DecimalMissingScale || shape==Shape::DecimalBadPrecision || shape==Shape::DecimalConflict;
    b.number(1,shape==Shape::InvalidPhysical?99:shape==Shape::PhysicalMismatch?1:decimal?7:2);
    if(decimal)b.number(2,16);
    if(shape!=Shape::MissingRepetition)b.number(3,0);
    b.name(shape==Shape::Dot?"Price.Ticks":shape==Shape::LongName?std::string(64,'X'):
        shape==Shape::EmbeddedNul?std::string("A\0B",3):"MiXeD");
    if(shape==Shape::LegacyTimestamp)b.number(6,10);
    if(shape==Shape::Contradictory)b.number(6,0);
    if(shape==Shape::LegacyUnsigned)b.number(6,14);
    logical(b,shape);
    if(shape==Shape::DecimalConflict){b.number(6,5);b.number(7,3);b.number(8,18);}
    b.stop();
}
Bytes metadata(Shape shape,bool explicit_root=false) {
    Bytes b;b.number(1,1);b.field(2,pq::kTcList);
    b.list(shape==Shape::Nested || shape==Shape::DuplicateName?3:2);
    if(explicit_root)b.number(3,0);
    b.name("schema");b.number(5,shape==Shape::DuplicateName?2:1);b.stop();
    if(shape==Shape::Nested){b.number(3,0);b.name("group");b.number(5,1);b.stop();}
    leaf(b,shape);if(shape==Shape::DuplicateName)leaf(b,shape);
    b.number(3,0,pq::kTcI64);b.field(4,pq::kTcList);b.list(0);b.stop();return b;
}
struct Output {
    pq::PqMeta* meta=new pq::PqMeta{};pq::PqColumn columns[4]{};pq::PqChunk chunks[4]{};
    Output(){meta->columns=columns;meta->columns_cap=4;meta->chunks=chunks;meta->chunks_cap=4;}
    ~Output(){delete meta;}
};
S parse(const Bytes& b,Output& out){return pq::pq_parse_file_meta_strict(b.bytes.data(),
    static_cast<std::uint32_t>(b.bytes.size()),out.meta,4);}
bool parse_legacy(const Bytes& b,Output& out){return pq::pq_parse_file_meta(b.bytes.data(),
    static_cast<std::uint32_t>(b.bytes.size()),out.meta);}
Bytes field_id_metadata(const Bytes& encoded_id,bool nested=false) {
    Bytes b;b.number(1,1);b.field(2,pq::kTcList);b.list(nested?3:2);
    b.name("schema");b.number(5,1);b.stop();
    if(nested){b.number(3,0);b.name("group");b.number(5,1);b.number(9,23);b.stop();}
    b.number(1,2);b.number(3,0);b.name("MiXeD");
    b.bytes.insert(b.bytes.end(),encoded_id.bytes.begin(),encoded_id.bytes.end());b.stop();
    b.number(3,0,pq::kTcI64);b.field(4,pq::kTcList);b.list(0);b.stop();return b;
}

TEST(ParquetMetaStrict, SignedFieldIdsAndAbsenceSurviveBothParsers) {
    for(const std::int64_t id:{std::int64_t{INT32_MIN},std::int64_t{-1},std::int64_t{0},
                              std::int64_t{17},std::int64_t{INT32_MAX}}) {
        Bytes encoded;encoded.number(9,id);const auto bytes=field_id_metadata(encoded);
        Output exact;ASSERT_EQ(parse(bytes,exact),S::Ok);ASSERT_EQ(exact.meta->n_columns,1u);
        EXPECT_EQ(exact.columns[0].has_field_id,1);EXPECT_EQ(exact.columns[0].field_id,id);
        Output legacy;ASSERT_TRUE(parse_legacy(bytes,legacy));ASSERT_EQ(legacy.meta->n_columns,1u);
        EXPECT_EQ(legacy.columns[0].has_field_id,1);EXPECT_EQ(legacy.columns[0].field_id,id);
    }
    const auto absent=field_id_metadata(Bytes{});Output exact,legacy;
    ASSERT_EQ(parse(absent,exact),S::Ok);ASSERT_TRUE(parse_legacy(absent,legacy));
    EXPECT_EQ(exact.columns[0].has_field_id,0);EXPECT_EQ(exact.columns[0].field_id,0);
    EXPECT_EQ(legacy.columns[0].has_field_id,0);EXPECT_EQ(legacy.columns[0].field_id,0);
}
TEST(ParquetMetaStrict, LegacyNestedLeafKeepsItsOwnFieldId) {
    for(const bool present:{false,true}) {
        Bytes encoded;if(present)encoded.number(9,-1);
        const auto bytes=field_id_metadata(encoded,true);Output legacy;
        ASSERT_TRUE(parse_legacy(bytes,legacy));ASSERT_EQ(legacy.meta->n_columns,1u);
        EXPECT_STREQ(legacy.columns[0].name,"group.MiXeD");
        EXPECT_EQ(legacy.columns[0].has_field_id,present?1:0);
        EXPECT_EQ(legacy.columns[0].field_id,present?-1:0);
        Output exact;EXPECT_EQ(parse(bytes,exact),S::Unsupported);
    }
}
TEST(ParquetMetaStrict, MalformedFieldIdsNeverTruncateOrUseLastValue) {
    std::vector<Bytes> malformed;
    for(const std::int64_t id:{std::int64_t{INT32_MIN}-1,std::int64_t{INT32_MAX}+1}) {
        Bytes encoded;encoded.number(9,id);malformed.push_back(encoded);
    }
    Bytes wrapped;wrapped.field(9,pq::kTcI32);wrapped.overflow(0);malformed.push_back(wrapped);
    Bytes overlong;overlong.field(9,pq::kTcI32);
    for(unsigned i=0;i<5;++i)overlong.bytes.push_back(0x80);
    overlong.bytes.push_back(0);malformed.push_back(overlong);
    Bytes duplicate;duplicate.number(9,17);duplicate.number(9,18);malformed.push_back(duplicate);
    for(const auto type:{pq::kTcI16,pq::kTcI64,pq::kTcBinary,pq::kTcTrue}) {
        Bytes encoded;encoded.field(9,type);
        if(type!=pq::kTcTrue)encoded.zig(0);
        malformed.push_back(encoded);
    }
    for(const auto& encoded:malformed) {
        const auto bytes=field_id_metadata(encoded);Output exact,legacy;
        EXPECT_EQ(parse(bytes,exact),S::Invalid);EXPECT_FALSE(parse_legacy(bytes,legacy));
    }
}

TEST(ParquetMetaStrict, ExactFlatNamesAndLogicalMembers) {
    for(const auto shape:{Shape::Plain,Shape::Dot,Shape::Timestamp,Shape::LocalTime,Shape::LegacyUnsigned,Shape::Decimal}) {
        Output out;ASSERT_EQ(parse(metadata(shape),out),S::Ok);
        EXPECT_EQ(out.meta->num_rows,0);EXPECT_EQ(out.meta->n_row_groups,0u);ASSERT_EQ(out.meta->n_columns,1u);
        const auto& c=out.columns[0];EXPECT_EQ(c.max_def,0);EXPECT_EQ(c.max_rep,0);
        if(shape==Shape::Dot)EXPECT_STREQ(c.name,"Price.Ticks");else EXPECT_STREQ(c.name,"MiXeD");
        if(shape==Shape::Timestamp){EXPECT_EQ(c.ts_utc,1);EXPECT_EQ(c.time_unit,3);}
        if(shape==Shape::LocalTime){EXPECT_EQ(c.ts_utc,0);EXPECT_EQ(c.time_unit,3);}
        if(shape==Shape::LegacyUnsigned){EXPECT_EQ(c.int_signed,0);EXPECT_EQ(c.int_bits,64);}
        if(shape==Shape::Decimal){EXPECT_EQ(c.precision,18);EXPECT_EQ(c.scale,2);}
    }
}
TEST(ParquetMetaStrict, MalformedAndUnsupportedNeverBecomePhysicalFallback) {
    for(const auto shape:{Shape::MissingUtc,Shape::MissingUnit,Shape::DuplicateUtc,Shape::PhysicalMismatch,
        Shape::InvalidPhysical,Shape::Contradictory,Shape::MissingRepetition,Shape::EmbeddedNul,Shape::IntMissingSign,
        Shape::BadIntWidth,Shape::DecimalMissingScale,Shape::DecimalBadPrecision,Shape::DecimalConflict}) {
        Output out;EXPECT_EQ(parse(metadata(shape),out),S::Invalid)<<static_cast<int>(shape);
    }
    for(const auto shape:{Shape::UnknownLogical,Shape::LongName,Shape::Nested,Shape::DuplicateName,Shape::LegacyTimestamp}) {
        Output out;EXPECT_EQ(parse(metadata(shape),out),S::Unsupported)<<static_cast<int>(shape);
    }
}
TEST(ParquetMetaStrict, TruncationCapacityAndLegacyRemainExplicit) {
    const auto valid=metadata(Shape::Timestamp);
    for(std::size_t length=0;length<valid.bytes.size();++length) {
        Output out;EXPECT_NE(pq::pq_parse_file_meta_strict(valid.bytes.data(),static_cast<std::uint32_t>(length),out.meta,4),S::Ok);
    }
    Output narrow;narrow.meta->columns_cap=1;EXPECT_EQ(parse(metadata(Shape::DuplicateName),narrow),S::Capacity);
    const auto incomplete=metadata(Shape::MissingUtc);Output legacy;
    ASSERT_TRUE(pq::pq_parse_file_meta(incomplete.bytes.data(),static_cast<std::uint32_t>(incomplete.bytes.size()),legacy.meta));
    EXPECT_EQ(parse(incomplete,legacy),S::Invalid); // additive strictness only
}
enum class ChunkShape { Plain,WrongPath,MissingValues,MissingData,WrongPhysical,External,RowMismatch,Dictionary,DuplicateDictionary };
void row_group(Bytes& b,ChunkShape shape) {
    b.field(4,pq::kTcList);b.list(1);b.field(1,pq::kTcList);b.list(1);
    if(shape==ChunkShape::External){b.field(1,pq::kTcBinary);b.var(1);b.bytes.push_back('x');}
    b.number(2,4,pq::kTcI64);b.field(3,pq::kTcStruct);
    b.number(1,shape==ChunkShape::WrongPhysical?1:2);
    b.field(2,pq::kTcList);b.bytes.push_back(0x15);b.zig(0);
    b.field(3,pq::kTcList);b.bytes.push_back(0x18);
    const std::string name=shape==ChunkShape::WrongPath?"other":"MiXeD";
    b.var(name.size());b.bytes.insert(b.bytes.end(),name.begin(),name.end());b.number(4,0);
    if(shape!=ChunkShape::MissingValues)b.number(5,shape==ChunkShape::RowMismatch?2:1,pq::kTcI64);
    b.number(6,8,pq::kTcI64);b.number(7,8,pq::kTcI64);
    if(shape!=ChunkShape::MissingData)b.number(9,4,pq::kTcI64);
    if(shape==ChunkShape::DuplicateDictionary)b.number(11,999999,pq::kTcI64);
    if(shape==ChunkShape::Dictionary || shape==ChunkShape::DuplicateDictionary)b.number(11,4,pq::kTcI64);
    b.stop();b.stop();b.number(2,8,pq::kTcI64);b.number(3,1,pq::kTcI64);b.stop();
}
Bytes chunk_metadata(ChunkShape shape,bool groups_first=false) {
    Bytes b;b.number(1,1);
    if(groups_first)row_group(b,shape);
    b.field(2,pq::kTcList);b.list(2);b.name("schema");b.number(5,1);b.stop();leaf(b,Shape::Plain);
    b.number(3,1,pq::kTcI64);if(!groups_first)row_group(b,shape);b.stop();return b;
}
TEST(ParquetMetaStrict, RowGroupsRequireExactFlatPathsAndCompleteChunkMetadata) {
    for(bool groups_first:{false,true}) {
        Output out;ASSERT_EQ(parse(chunk_metadata(ChunkShape::Plain,groups_first),out),S::Ok);
        EXPECT_EQ(out.meta->num_rows,1);EXPECT_EQ(out.meta->n_row_groups,1u);
    }
    for(const auto shape:{ChunkShape::WrongPath,ChunkShape::MissingValues,ChunkShape::MissingData,
                         ChunkShape::WrongPhysical,ChunkShape::RowMismatch}) {
        Output out;EXPECT_EQ(parse(chunk_metadata(shape),out),S::Invalid);
    }
    Output out;EXPECT_EQ(parse(chunk_metadata(ChunkShape::External),out),S::Unsupported);
    const auto bytes=chunk_metadata(ChunkShape::Plain);out.meta->chunks_cap=1;
    EXPECT_EQ(pq::pq_parse_file_meta_strict(bytes.bytes.data(),static_cast<std::uint32_t>(bytes.bytes.size()),out.meta,1),S::Ok);
}
TEST(ParquetMetaStrict, DuplicateDictionaryOffsetNeverUsesLastValue) {
    Output valid;EXPECT_EQ(parse(chunk_metadata(ChunkShape::Dictionary),valid),S::Ok);
    const auto bytes=chunk_metadata(ChunkShape::DuplicateDictionary);Output legacy;
    ASSERT_TRUE(pq::pq_parse_file_meta(bytes.bytes.data(),static_cast<std::uint32_t>(bytes.bytes.size()),legacy.meta));
    EXPECT_EQ(legacy.chunks[0].dictionary_page_offset,4);
    Output exact;EXPECT_EQ(parse(bytes,exact),S::Invalid);
}
Bytes wrapped_prefix(std::initializer_list<std::uint8_t> pattern,std::size_t offset) {
    auto b=metadata(Shape::Plain);const auto found=std::search(b.bytes.begin(),b.bytes.end(),pattern.begin(),pattern.end());
    if(found==b.bytes.end())std::abort();
    const auto index=static_cast<std::size_t>(found-b.bytes.begin())+offset;
    Bytes overflow;overflow.overflow(b.bytes[index]);
    b.bytes.erase(b.bytes.begin()+static_cast<std::ptrdiff_t>(index));
    b.bytes.insert(b.bytes.begin()+static_cast<std::ptrdiff_t>(index),overflow.bytes.begin(),overflow.bytes.end());return b;
}
TEST(ParquetMetaStrict, OverflowedLengthsAndSkippedVarintsNeverWrap) {
    // Each ten-byte value used to wrap to the small original value: the final
    // 0x02 payload exceeds ULEB128's one-bit tenth-byte allowance.
    std::vector<Bytes> malformed;
    malformed.push_back(wrapped_prefix({8,8,5,'M','i'},2)); // schema name
    malformed.push_back(wrapped_prefix({9,4,0xfc,2},3)); // schema list
    for(const auto type:{pq::kTcBinary,pq::kTcList,pq::kTcMap,pq::kTcI64}) {
        auto b=metadata(Shape::Plain);b.bytes.pop_back();b.field(15,type);
        if(type==pq::kTcList)b.bytes.push_back(0xf3);
        b.overflow(type==pq::kTcBinary || type==pq::kTcList?5:type==pq::kTcMap?1:0);
        if(type==pq::kTcMap){b.bytes.push_back(0x33);b.bytes.push_back(0);b.bytes.push_back(0);}
        if(type==pq::kTcBinary || type==pq::kTcList)for(unsigned i=0;i<5;++i)b.bytes.push_back('x');
        b.stop();malformed.push_back(b);
    }
    for(const auto& bytes:malformed) {
        Output legacy;ASSERT_TRUE(pq::pq_parse_file_meta(bytes.bytes.data(),static_cast<std::uint32_t>(bytes.bytes.size()),legacy.meta));
        Output exact;EXPECT_EQ(parse(bytes,exact),S::Invalid);
    }
}
} // namespace

TEST(ParquetMetaStrict, RetainedTreeAdmitsNestedMembersWithoutRelaxingFlatContract) {
    for(const auto shape:{Shape::Plain,Shape::Dot,Shape::Timestamp,Shape::LocalTime,Shape::Nested}) {
        const auto bytes=metadata(shape);bolt::Arena arena;pq::PqSchemaTree tree{};
        ASSERT_TRUE(pq::parquet_schema_tree_slice(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),&arena,&tree));
        Output nested;
        EXPECT_EQ(pq::pq_parse_file_meta_tree_strict(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),
            nested.meta,4,&tree,&arena),S::Ok);
        Output flat;EXPECT_EQ(parse(bytes,flat),shape==Shape::Nested?S::Unsupported:S::Ok);
        ++tree.nodes[1].field_id;tree.nodes[1].has_field_id=1;
        EXPECT_NE(pq::pq_parse_file_meta_tree_strict(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),
            nested.meta,4,&tree,&arena),S::Ok);
    }
}
TEST(ParquetMetaStrict, RetainedTreeStillRejectsMalformedRawAnnotationsAndSiblingNames) {
    for(const auto shape:{Shape::MissingUtc,Shape::MissingUnit,Shape::DuplicateUtc,Shape::Contradictory,
        Shape::MissingRepetition,Shape::DecimalMissingScale,Shape::DecimalConflict,Shape::DuplicateName}) {
        const auto bytes=metadata(shape);bolt::Arena arena;pq::PqSchemaTree tree{};
        if(!pq::parquet_schema_tree_slice(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),&arena,&tree))continue;
        Output out;EXPECT_NE(pq::pq_parse_file_meta_tree_strict(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),
            out.meta,4,&tree,&arena),S::Ok)<<static_cast<int>(shape);
    }
}

TEST(ParquetMetaStrict, RetainedTreeAdmitsExplicitRequiredRootFromIndependentWriters) {
    const auto bytes=metadata(Shape::Nested,true);bolt::Arena arena;pq::PqSchemaTree tree{};
    ASSERT_TRUE(pq::parquet_schema_tree_slice(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),&arena,&tree));
    Output out;EXPECT_EQ(pq::pq_parse_file_meta_tree_strict(bytes.bytes.data(),static_cast<uint32_t>(bytes.bytes.size()),
        out.meta,4,&tree,&arena),S::Ok);
    Output flat;EXPECT_NE(parse(bytes,flat),S::Ok);
}
