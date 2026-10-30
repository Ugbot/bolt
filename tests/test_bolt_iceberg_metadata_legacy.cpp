#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/bolt_arena.h"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
namespace b = bolt::lakehouse::iceberg;
namespace {
const std::string legacy = R"("schema":{"type":"struct","schema-id":2,"fields":[{"id":7,"name":"now","required":true,"type":"long"}]})";
const std::string modern = R"("schemas":[{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"old","required":true,"type":"int"}]},{"type":"struct","schema-id":2,"fields":[{"id":7,"name":"now","required":true,"type":"long"}]}])";
b::Metadata* parse(bolt::Arena& arena, const std::string& input) {
    auto* out = arena.allocate_array<b::Metadata>(1u); EXPECT_NE(out,nullptr); if (!out) return nullptr;
    b::metadata_init(out,&arena,nullptr); return b::metadata_parse(reinterpret_cast<const uint8_t*>(input.data()),static_cast<uint32_t>(input.size()),&arena,out) ? out : nullptr;
}
TEST(IcebergLegacyMetadata, ModernHistoryWinsWithoutDuplicatingLegacyCurrentSchemaInEitherOrder) {
    bolt::Arena arena;
    for (const bool legacy_first : {false,true}) {
        const std::string input = "{\"format-version\":1,\"current-schema-id\":2," + (legacy_first ? legacy + "," + modern : modern + "," + legacy) + "}";
        const auto* out = parse(arena,input); ASSERT_NE(out,nullptr); ASSERT_EQ(out->n_schemas,2u);
        EXPECT_EQ(out->schemas[0].schema_id,0); EXPECT_EQ(out->schemas[1].schema_id,2);
        const auto* current = b::metadata_current_schema(out); ASSERT_NE(current,nullptr); EXPECT_EQ(current->schema_id,2); EXPECT_EQ(current->fields[0].id,7);
    }
}
TEST(IcebergLegacyMetadata, LegacyOnlyAndModernOnlyKeepTheirExistingSelection) {
    bolt::Arena arena; const auto* old = parse(arena,"{\"format-version\":1," + legacy + "}"); ASSERT_NE(old,nullptr); EXPECT_EQ(old->n_schemas,1u); EXPECT_EQ(old->current_schema_id,2);
    const auto* now = parse(arena,"{\"format-version\":2,\"current-schema-id\":2," + modern + "}"); ASSERT_NE(now,nullptr); EXPECT_EQ(now->n_schemas,2u); EXPECT_EQ(now->current_schema_id,2);
}
TEST(IcebergLegacyMetadata, MixedRepresentationsNeedAnExplicitCurrentId) {
    bolt::Arena arena; EXPECT_EQ(parse(arena,"{\"format-version\":1," + legacy + "," + modern + "}"),nullptr);
}
TEST(IcebergLegacyMetadata, DuplicateModernHistoriesDoNotAppendAnotherHistory) {
    bolt::Arena arena; EXPECT_EQ(parse(arena,"{\"format-version\":1,\"current-schema-id\":2," + modern + "," + modern + "}"),nullptr);
}
TEST(IcebergLegacyMetadata, MixedCurrentIdMustBelongToAuthoritativeHistory) {
    bolt::Arena arena; EXPECT_EQ(parse(arena,"{\"format-version\":1,\"current-schema-id\":9," + legacy + "," + modern + "}"),nullptr);
}
} // namespace
