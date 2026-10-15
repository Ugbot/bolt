#include "bolt/bolt_arrow_import.h"
#include "bolt/bolt_column.h"
#include <gtest/gtest.h>
#include <cstring>
#include <limits>
namespace {
namespace ar=bolt::arrow;
void release_schema(ArrowSchema* s) { s->release=nullptr; }
void release_array(ArrowArray* a) { a->release=nullptr; }
struct Fixture {
    ArrowSchema schema{},leaf{};ArrowSchema* schemas[1]{&leaf};
    ArrowArray array{},column{};ArrowArray* arrays[1]{&column};
    const void* root_buffers[1]{};const void* buffers[3]{};
    explicit Fixture(const char* format,const void* data,std::int64_t rows) {
        schema.format="+s";schema.n_children=1;schema.children=schemas;schema.release=release_schema;
        leaf.format=format;leaf.name="value";leaf.flags=2;leaf.release=release_schema;
        array.length=rows;array.n_buffers=1;array.buffers=root_buffers;array.n_children=1;array.children=arrays;array.release=release_array;
        column.length=rows;column.n_buffers=2;column.buffers=buffers;column.release=release_array;buffers[1]=data;
    }
};
template<class T> void integer(const char* code,bolt::BoltType type) {
    const T values[]{std::numeric_limits<T>::min(),std::numeric_limits<T>::max(),static_cast<T>(1)};
    Fixture fixture{code,values,3};bolt::Arena scratch,destination;ar::ImportState state{};
    ASSERT_EQ(ar::import_begin(&fixture.schema,{},&scratch,state),ar::ImportError::Ok);
    ASSERT_EQ(ar::import_append(state,&fixture.array),ar::ImportError::Ok);
    bolt::BoltBatch batch{};ASSERT_EQ(ar::import_finish(state,&destination,batch),ar::ImportError::Ok);
    ASSERT_EQ(batch.col(0).type,type);EXPECT_EQ(std::memcmp(batch.col(0).data,values,sizeof(values)),0);
    EXPECT_NE(fixture.array.release,nullptr);EXPECT_NE(fixture.schema.release,nullptr); // generic API borrows
    EXPECT_EQ(ar::import_append(state,&fixture.array),ar::ImportError::Invalid);
}
TEST(ArrowImport, ExactIntegerWidthsAndUnsignedExtrema) {
    integer<std::int8_t>("c",bolt::BoltType::Int8);integer<std::int16_t>("s",bolt::BoltType::Int16);
    integer<std::int32_t>("i",bolt::BoltType::Int32);integer<std::int64_t>("l",bolt::BoltType::Int64);
    integer<std::uint8_t>("C",bolt::BoltType::UInt8);integer<std::uint16_t>("S",bolt::BoltType::UInt16);
    integer<std::uint32_t>("I",bolt::BoltType::UInt32);integer<std::uint64_t>("L",bolt::BoltType::UInt64);
}
TEST(ArrowImport, ParentChildOffsetsNullBitsAndCopiedMultibatchLifetime) {
    std::uint64_t values[]{0,1,2,UINT64_MAX,4,5,6,7,8,9};std::uint8_t valid[]{239,254};
    Fixture f{"L",values,6};f.array.offset=1;f.column.offset=2;f.column.length=8;
    f.column.null_count=-1;f.buffers[0]=valid;
    bolt::Arena scratch,destination;ar::ImportState state{};
    ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
    ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);
    values[3]=123; // Native chunk must retain UINT64_MAX after producer mutation.
    f.array.offset=0;f.array.length=3;f.column.offset=0;f.column.length=3;f.column.null_count=0;f.buffers[0]=nullptr;
    ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);
    bolt::BoltBatch out{};ASSERT_EQ(ar::import_finish(state,&destination,out),ar::ImportError::Ok);
    EXPECT_EQ(out.num_rows,9);const auto* got=static_cast<const std::uint64_t*>(out.col(0).data);
    EXPECT_EQ(got[0],UINT64_MAX);EXPECT_TRUE(out.col(0).is_null(1));EXPECT_TRUE(out.col(0).is_null(5));
    EXPECT_EQ(got[2],5u);EXPECT_EQ(got[6],0u);EXPECT_EQ(got[8],2u);EXPECT_FALSE(out.col(0).is_null(8));
}
TEST(ArrowImport, BinaryLargeOffsetsFlatUtf8AndEmptyTypedStream) {
    const std::uint8_t data[]{9,0,255,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'};
    std::int64_t offsets[]{0,1,3,18};Fixture f{"Z",nullptr,2};f.column.offset=1;
    f.column.n_buffers=3;f.buffers[1]=offsets;f.buffers[2]=data;
    for (const char* format:{"Z","U"}) {
        f.leaf.format=format;bolt::Arena scratch,destination;ar::ImportState state{};
        ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
        ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);
        bolt::BoltBatch batch{};ASSERT_EQ(ar::import_finish(state,&destination,batch),ar::ImportError::Ok);
        EXPECT_EQ(batch.col(0).format,format[0]=='U'?bolt::ColumnFormat::Flat:bolt::ColumnFormat::VarBinary);
        EXPECT_EQ(batch.col(0).stats.total_string_bytes,17u);
        const std::uint8_t* bytes=nullptr;std::int32_t n=0;batch.col(0).utf8_at(0,&bytes,&n);
        ASSERT_EQ(n,2);EXPECT_EQ(bytes[0],0);EXPECT_EQ(bytes[1],255);
        batch.col(0).utf8_at(1,&bytes,&n);ASSERT_EQ(n,15);EXPECT_EQ(std::memcmp(bytes,data+3,15),0);
    }
    for (const char* format:{"u","z","L","b"}) {
        f.leaf.format=format;bolt::Arena scratch,destination;ar::ImportState state{};
        ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
        bolt::BoltBatch batch{};ASSERT_EQ(ar::import_finish(state,&destination,batch),ar::ImportError::Ok);
        EXPECT_EQ(batch.num_rows,0);EXPECT_EQ(batch.num_cols,1u);
    }
}
TEST(ArrowImport, BooleanFloatDecimalAndTemporalMetadata) {
    std::uint8_t bits[]{177,1};Fixture f{"b",bits,3};f.column.offset=4;
    bolt::Arena scratch,destination;ar::ImportState state{};
    ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
    ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);
    bolt::BoltBatch out{};ASSERT_EQ(ar::import_finish(state,&destination,out),ar::ImportError::Ok);
    const std::uint8_t expected[]{1,1,0};EXPECT_EQ(std::memcmp(out.col(0).data,expected,3),0);EXPECT_EQ(out.col(0).type,bolt::BoltType::Bool);
    std::uint64_t decimal[]{UINT64_MAX,UINT64_MAX,123,0};
    for (const char* format:{"d:38,7","tss:","tsm:","tsu:","tsn:","tDs","tDm","tDu","tDn","tdD","tdm","f","g"}) {
        Fixture item{format,decimal,2};bolt::Arena a,b;ar::ImportState next{};
        ASSERT_EQ(ar::import_begin(&item.schema,{},&a,next),ar::ImportError::Ok);
        ASSERT_EQ(ar::import_append(next,&item.array),ar::ImportError::Ok);
        bolt::BoltBatch result{};ASSERT_EQ(ar::import_finish(next,&b,result),ar::ImportError::Ok);
        if (format[0]=='d') { EXPECT_EQ(next.fields[0].precision,38);EXPECT_EQ(result.col(0).decimal_scale,7);EXPECT_EQ(std::memcmp(result.col(0).data,decimal,32),0); }
        if (format[0]=='f' || format[0]=='g') EXPECT_EQ(std::memcmp(result.col(0).data,decimal,format[0]=='f'?8:16),0);
        if (format[0]=='t' && (format[1]=='s' || format[1]=='D')) {
            const auto unit=format[2]=='s'?1:format[2]=='m'?2:format[2]=='u'?3:4;
            EXPECT_EQ(next.fields[0].unit,unit);EXPECT_EQ(next.fields[0].temporal_kind,format[1]=='s'?1:2);
            if (unit!=3) EXPECT_EQ(result.col(0).type,bolt::BoltType::Int64);
        }
    }
}
TEST(ArrowImport, UnsupportedAndMalformedShapesPreserveOutput) {
    std::int64_t value=1;Fixture f{"l",&value,1};
    for (const char* format:{"+l","tsu:UTC","d:0,0","d:39,0","d:3,-1","d:3,4","d:3,1,256","vu","w:16"}) {
        f.leaf.format=format;bolt::Arena scratch;ar::ImportState state{};
        EXPECT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Unsupported);
    }
    f.leaf.format="l";f.leaf.dictionary=&f.leaf;bolt::Arena scratch,destination;ar::ImportState state{};
    EXPECT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Unsupported);f.leaf.dictionary=nullptr;
    ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
    f.column.offset=INT64_MAX;EXPECT_EQ(ar::import_append(state,&f.array),ar::ImportError::Invalid);
    bolt::BoltBatch out{};out.num_rows=99;EXPECT_EQ(ar::import_finish(state,&destination,out),ar::ImportError::Invalid);EXPECT_EQ(out.num_rows,99);
}
TEST(ArrowImport, BoundsAndCombinedAllocationBudget) {
    std::uint64_t values[]{1,2};Fixture f{"L",values,2};ar::ImportLimits limits{};limits.max_rows=1;
    bolt::Arena scratch;ar::ImportState state{};
    ASSERT_EQ(ar::import_begin(&f.schema,limits,&scratch,state),ar::ImportError::Ok);
    EXPECT_EQ(ar::import_append(state,&f.array),ar::ImportError::Limit);
    limits={};limits.max_bytes=1;EXPECT_EQ(ar::import_begin(&f.schema,limits,&scratch,state),ar::ImportError::Limit);
    limits={};limits.max_batches=1;ASSERT_EQ(ar::import_begin(&f.schema,limits,&scratch,state),ar::ImportError::Ok);
    ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);EXPECT_EQ(ar::import_append(state,&f.array),ar::ImportError::Limit);
    limits={};ASSERT_EQ(ar::import_begin(&f.schema,limits,&scratch,state),ar::ImportError::Ok);
    f.buffers[1]=reinterpret_cast<void*>(UINTPTR_MAX-3);
    EXPECT_EQ(ar::import_append(state,&f.array),ar::ImportError::Invalid);
    f.buffers[1]=values;f.column.dictionary=&f.column;
    ASSERT_EQ(ar::import_begin(&f.schema,limits,&scratch,state),ar::ImportError::Ok);
    EXPECT_EQ(ar::import_append(state,&f.array),ar::ImportError::Unsupported);
    const std::int32_t descending[]{2,1};Fixture binary{"z",nullptr,1};
    binary.column.n_buffers=3;binary.buffers[1]=descending;binary.buffers[2]=values;
    ASSERT_EQ(ar::import_begin(&binary.schema,limits,&scratch,state),ar::ImportError::Ok);
    EXPECT_EQ(ar::import_append(state,&binary.array),ar::ImportError::Invalid);
}
TEST(ArrowImport, RootMetadataRefusesAndZeroBatchFinalBuffersAreBudgeted) {
    Fixture f{"u",nullptr,0};f.schema.metadata="unsupported";
    bolt::Arena scratch;ar::ImportState state{};
    EXPECT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Unsupported);
    f.schema.metadata=nullptr;
    ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
    const auto bound=state.charged_bytes;
    ar::ImportLimits limits{};limits.max_bytes=bound-1;bolt::Arena small;
    EXPECT_EQ(ar::import_begin(&f.schema,limits,&small,state),ar::ImportError::Limit);
    limits.max_bytes=bound;bolt::Arena exact,destination;
    ASSERT_EQ(ar::import_begin(&f.schema,limits,&exact,state),ar::ImportError::Ok);
    bolt::BoltBatch out{};ASSERT_EQ(ar::import_finish(state,&destination,out),ar::ImportError::Ok);
    EXPECT_EQ(out.num_rows,0);EXPECT_LE(exact.total_allocated()+destination.total_allocated(),bound);
}
TEST(ArrowImport, UnsupportedArenaAlignmentRefusesBeforeBudgetedAllocations) {
    const std::int64_t value=42;Fixture f{"l",&value,1};
    bolt::ArenaConfig config{};config.alignment=128;
    bolt::Arena wide{config},scratch;ar::ImportState state{};
    EXPECT_EQ(ar::import_begin(&f.schema,{},&wide,state),ar::ImportError::Unsupported);
    EXPECT_EQ(wide.total_allocated(),0u);EXPECT_EQ(state.fields,nullptr);
    ASSERT_EQ(ar::import_begin(&f.schema,{},&scratch,state),ar::ImportError::Ok);
    ASSERT_EQ(ar::import_append(state,&f.array),ar::ImportError::Ok);
    bolt::BoltBatch out{};out.num_rows=777;
    EXPECT_EQ(ar::import_finish(state,&wide,out),ar::ImportError::Unsupported);
    EXPECT_EQ(out.num_rows,777);EXPECT_EQ(wide.total_allocated(),0u);
    EXPECT_FALSE(state.finished);
}
} // namespace
