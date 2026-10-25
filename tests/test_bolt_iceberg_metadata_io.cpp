#include "bolt/lakehouse/iceberg/metadata_io.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/parse/bolt_json.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <cstdlib>
#include <string>

namespace {
namespace bl = bolt::lakehouse;
namespace ice = bl::iceberg;
using Status = ice::MetadataLoadStatus;
constexpr char kUuid[] = "12345678-1234-1234-1234-123456789abc";
constexpr char kTable[] = "s3://data-bucket/independent table";
constexpr char kJson[] = R"({"format-version":2,"table-uuid":"12345678-1234-1234-1234-123456789abc","location":"s3://data-bucket/independent table","last-updated-ms":999,"last-sequence-number":17,"current-snapshot-id":19,"current-schema-id":9,"schemas":[{"type":"struct","schema-id":9,"fields":[{"id":7,"name":"value","required":true,"type":"long"}]}],"default-spec-id":4,"partition-specs":[{"spec-id":4,"fields":[]}],"default-sort-order-id":3,"sort-orders":[{"order-id":3,"fields":[]}],"snapshots":[{"snapshot-id":19,"sequence-number":17,"timestamp-ms":999,"schema-id":9,"manifest-list":"s3://metadata-bucket/root/snap.avro"}]})";

struct Origin {
    std::string bytes = kJson;
    uint32_t calls = 0, discovery = 0, changed_at = UINT32_MAX, short_at = UINT32_MAX;
    uint32_t largest = 0;
    uint32_t scratch_bytes = 0;
    uint64_t next = 0;
    int failure = bl::kOsOk;
    static Origin& self(void* p) { return *static_cast<Origin*>(p); }
    static int get(void* p, const char*, bolt::Arena*, const uint8_t**, uint64_t*) noexcept {
        ++self(p).discovery; return bl::kOsIoError;
    }
    static int head(void* p, const char*, bl::ObjectMeta*) noexcept {
        ++self(p).discovery; return bl::kOsIoError;
    }
    static int list(void* p, const char*, bl::ObjectEntry*, uint32_t, uint32_t*) noexcept {
        ++self(p).discovery; return bl::kOsIoError;
    }
    static int range(void* p, const char* key, const bl::ObjectVersion* expected,
        uint64_t offset, uint32_t length, bolt::Arena* io, uint8_t* dst, uint32_t* got) noexcept {
        auto& origin = self(p); ++origin.calls; *got = 0;
        if (std::strcmp(key, "metadata/target%.json") != 0 || offset != origin.next ||
            length == 0 || length > ice::kMetadataLoadMaxRequestBytes ||
            expected->size != origin.bytes.size() || std::strcmp(expected->etag, "\"revision-A\"") != 0)
            return bl::kOsBadArg;
        if (origin.calls == origin.changed_at) return bl::kOsPreconditionFailed;
        if (origin.failure != bl::kOsOk) return origin.failure;
        if (origin.scratch_bytes != 0 && io->allocate(origin.scratch_bytes) == nullptr)
            return bl::kOsBufferSmall;
        if (offset > origin.bytes.size() || length > origin.bytes.size() - offset) return bl::kOsIoError;
        const uint32_t copied = origin.calls == origin.short_at ? length - 1u : length;
        std::memcpy(dst, origin.bytes.data() + offset, copied);
        origin.next += length; origin.largest = origin.largest < length ? length : origin.largest;
        *got = copied; return bl::kOsOk;
    }
};

class IcebergMetadataIO : public ::testing::Test {
protected:
    Origin origin;
    bl::ObjectStoreVT vt{Origin::get, nullptr, Origin::list, nullptr, Origin::head,
        nullptr, nullptr, nullptr, Origin::range};
    bl::ObjectStore store{&vt, &origin};
    ice::MetadataObjectIdentity id{};
    ice::MetadataLoadLimits limits{1u << 20, 64};
    bolt::Arena metadata{{4u << 20, 4u << 20, 64, false}};
    bolt::Arena io{{1u << 20, 1u << 20, 64, false}};
    ice::Metadata out{};
    ice::MetadataLoadResult result{};
    std::array<uint8_t, sizeof(out)> out_before{};
    std::array<uint8_t, sizeof(result)> result_before{};
    void SetUp() override {
        std::strcpy(id.store_location, "s3://metadata-bucket/root space");
        std::strcpy(id.table_location, kTable); std::strcpy(id.table_uuid, kUuid);
        std::strcpy(id.metadata_uri, "s3://metadata-bucket/root space/metadata/target%.json");
        std::strcpy(id.object_key, "metadata/target%.json");
        std::strcpy(id.version.etag, "\"revision-A\""); id.version.size = origin.bytes.size();
        metadata.set_growth_enabled(false); io.set_growth_enabled(false);
        out.format_version = -123; std::strcpy(result.sha256, "unchanged");
        std::memcpy(out_before.data(), &out, sizeof(out));
        std::memcpy(result_before.data(), &result, sizeof(result));
    }
    Status load() { return ice::metadata_load_exact(&store, &id, &limits, &metadata, &io, &out, &result); }
    void unchanged() {
        EXPECT_EQ(std::memcmp(&out, out_before.data(), sizeof(out)), 0);
        EXPECT_EQ(std::memcmp(&result, result_before.data(), sizeof(result)), 0);
    }
    void no_discovery() { EXPECT_EQ(origin.discovery, 0u); }
};

TEST_F(IcebergMetadataIO, ExactPinnedObjectOwnsSchemaCountersAndCompleteHash) {
    ASSERT_EQ(load(), Status::Ok);
    EXPECT_GT(origin.calls, 1u); EXPECT_LE(origin.largest, 64u); no_discovery();
    EXPECT_EQ(std::memcmp(&result.object, &id, sizeof(id)), 0);
    char expected[65]{};
    bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(), expected);
    EXPECT_STREQ(result.sha256, expected); EXPECT_STREQ(out.location, kTable); EXPECT_STREQ(out.table_uuid, kUuid);
    EXPECT_EQ(out.last_sequence_number, 17); EXPECT_EQ(out.last_updated_ms, 999);
    EXPECT_EQ(out.current_snapshot_id, 19); EXPECT_EQ(out.current_schema_id, 9);
    EXPECT_EQ(out.current_spec_id, 4); EXPECT_EQ(out.default_sort_order_id, 3);
    ASSERT_EQ(out.n_schemas, 1u); ASSERT_EQ(out.schemas[0].n_fields, 1u);
    EXPECT_EQ(out.schemas[0].fields[0].id, 7); EXPECT_STREQ(out.schemas[0].fields[0].name, "value");
    ASSERT_EQ(out.n_snapshots, 1u); EXPECT_EQ(out.snapshots[0].schema_id, 9);
    EXPECT_EQ(out.arena, &metadata);
    origin.bytes.assign(origin.bytes.size(), 'x'); io.reset();
    EXPECT_STREQ(out.schemas[0].fields[0].name, "value"); EXPECT_STREQ(out.table_uuid, kUuid);
}

TEST_F(IcebergMetadataIO, BucketOnlyStoreRootDoesNotRestrictIndependentTableLocation) {
    std::strcpy(id.store_location, "s3://metadata-bucket");
    std::strcpy(id.metadata_uri, "s3://metadata-bucket/metadata/target%.json");
    ASSERT_EQ(load(), Status::Ok); EXPECT_STREQ(out.location, kTable); no_discovery();
}

TEST_F(IcebergMetadataIO, ChangedOnLaterRangeDoesNotRepinRetryOrPublish) {
    origin.changed_at = 3;
    EXPECT_EQ(load(), Status::Changed); EXPECT_EQ(origin.calls, 3u);
    unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, PerRequestProviderScratchIsResetWithoutLosingRetainedBytes) {
    limits.request_bytes = 17; origin.scratch_bytes = 128u << 10;
    ASSERT_EQ(load(), Status::Ok);
    EXPECT_GT(origin.calls, 8u); EXPECT_EQ(io.total_allocated(), origin.scratch_bytes);
    ASSERT_EQ(out.n_schemas, 1u); EXPECT_STREQ(out.schemas[0].fields[0].name, "value");
    EXPECT_EQ(out.last_sequence_number, 17); no_discovery();
    char expected[65]{};
    bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(), expected);
    EXPECT_STREQ(result.sha256, expected);
}

TEST_F(IcebergMetadataIO, ShortRangeIsAnErrorWithoutFallbackOrPartialOutput) {
    origin.short_at = 2;
    EXPECT_EQ(load(), Status::IoError); EXPECT_EQ(origin.calls, 2u);
    unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, MissingAndRangeFailureRemainDistinctWithoutFallback) {
    for (const int failure : {bl::kOsNotFound, bl::kOsIoError, bl::kOsNotImplemented}) {
        origin.failure = failure; origin.calls = 0;
        EXPECT_EQ(load(), failure == bl::kOsNotFound ? Status::NotFound :
            failure == bl::kOsNotImplemented ? Status::Unsupported : Status::IoError);
        EXPECT_EQ(origin.calls, 1u); unchanged(); no_discovery();
    }
}

TEST_F(IcebergMetadataIO, WeakValidatorAndPartialVtableRefuseBeforeIo) {
    std::strcpy(id.version.etag, "W/\"revision-A\"");
    EXPECT_EQ(load(), Status::BadArg);
    std::strcpy(id.version.etag, "\"revision-A\""); vt.get_range_if_match = nullptr;
    EXPECT_EQ(load(), Status::Unsupported); EXPECT_EQ(origin.calls, 0u);
    vt.get_range_if_match = Origin::range; id.table_uuid[0] = 0;
    EXPECT_EQ(load(), Status::BadArg); EXPECT_EQ(origin.calls, 0u);
    unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, ExtentRequestAndParserRepresentabilityBudgetsRefuseBeforeIo) {
    limits.max_bytes = id.version.size - 1u;
    EXPECT_EQ(load(), Status::Capacity); limits.max_bytes = 1u << 20;
    id.version.size = UINT64_MAX; EXPECT_EQ(load(), Status::Capacity);
    id.version.size = origin.bytes.size(); limits.request_bytes = 0;
    EXPECT_EQ(load(), Status::BadArg); limits.request_bytes = (1u << 20) + 1u;
    EXPECT_EQ(load(), Status::BadArg); limits.request_bytes = 64; limits.max_bytes = 0;
    EXPECT_EQ(load(), Status::BadArg); limits.max_bytes = 1u << 30;
    EXPECT_EQ(load(), Status::BadArg); EXPECT_EQ(origin.calls, 0u); unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, UnsupportedSchemesAndNoncanonicalMappingsNeverRead) {
    const auto original = id;
    for (const char* root : {"https://metadata-bucket/root", "file:///root", "s3://metadata-bucket/root/",
             "s3://metadata-bucket/root?space", "s3://metadata-bucket/root#space",
             "s3://metadata-bucket/root\\x", "s3://metadata-bucket/../root", "s3://metadata-bucket//root"}) {
        id = original; std::strcpy(id.store_location, root);
        const Status status = load();
        EXPECT_TRUE(status == Status::BadArg || status == Status::Unsupported);
        unchanged();
    }
    id = original; std::strcpy(id.metadata_uri, "s3://other-bucket/root/metadata/target%.json");
    EXPECT_EQ(load(), Status::IdentityMismatch); unchanged();
    id = original; std::strcpy(id.object_key, "metadata/../target.json");
    EXPECT_EQ(load(), Status::BadArg); id = original; std::memset(id.object_key, 'x', sizeof(id.object_key));
    EXPECT_EQ(load(), Status::BadArg); EXPECT_EQ(origin.calls, 0u); no_discovery();
}

TEST_F(IcebergMetadataIO, ParsedUuidAndTableLocationMustMatchUntruncatedIdentity) {
    const auto original = id;
    std::strcpy(id.table_uuid, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
    EXPECT_EQ(load(), Status::IdentityMismatch); unchanged();
    id = original; origin.next = 0;
    std::strcpy(id.table_location, "s3://data-bucket/another-table");
    EXPECT_EQ(load(), Status::IdentityMismatch); unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, MalformedMissingAndDuplicateIdentityJsonCannotPublish) {
    for (const auto& body : {std::string("{malformed}"), std::string("{}"),
         std::string(kJson).insert(1, "\"location\":\"s3://data-bucket/independent table\","),
         std::string(kJson).insert(1, "\"table-uuid\":12,")}) {
        origin.bytes = body; id.version.size = body.size(); origin.next = 0;
        EXPECT_EQ(load(), Status::DecodeFailed); unchanged(); no_discovery();
    }
}

TEST_F(IcebergMetadataIO, EscapedRootKeysCannotHideDuplicateIdentityOrUnknownAliases) {
    for (const char* property : {R"(,"\u006cocation":"s3://other-bucket/table")",
             R"(,"table-\u0075uid":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")",
             R"(,"un\u006bnown":0)"}) {
        origin.bytes = kJson;
        origin.bytes.insert(origin.bytes.size() - 1u, property);
        id.version.size = origin.bytes.size(); origin.next = 0; origin.calls = 0;
        EXPECT_EQ(load(), Status::DecodeFailed);
        EXPECT_EQ(origin.next, id.version.size); EXPECT_GT(origin.calls, 1u);
        unchanged(); no_discovery();
    }
}

TEST_F(IcebergMetadataIO, EscapedOnlyRootIdentitySpellingsCannotPublish) {
    for (const auto& spelling : {std::array<const char*, 2>{"location", R"(\u006cocation)"},
             std::array<const char*, 2>{"table-uuid", R"(table-\u0075uid)"}}) {
        origin.bytes = kJson;
        const auto position = origin.bytes.find(spelling[0]);
        ASSERT_NE(position, std::string::npos);
        origin.bytes.replace(position, std::strlen(spelling[0]), spelling[1]);
        id.version.size = origin.bytes.size(); origin.next = 0;
        EXPECT_EQ(load(), Status::DecodeFailed); unchanged(); no_discovery();
    }
}

TEST_F(IcebergMetadataIO, EscapedIdentityValuesAreExplicitlyRefusedEvenWhenRawUuidMatches) {
    for (const auto& value : {std::array<const char*, 2>{kTable, R"(s3:\/\/data-bucket/independent table)"},
             std::array<const char*, 2>{kUuid, R"(\u00312345678-1234-1234-1234-123456789abc)"}}) {
        origin.bytes = kJson;
        const auto position = origin.bytes.find(value[0]);
        ASSERT_NE(position, std::string::npos);
        origin.bytes.replace(position, std::strlen(value[0]), value[1]);
        id.version.size = origin.bytes.size(); origin.next = 0;
        EXPECT_EQ(load(), Status::DecodeFailed); unchanged(); no_discovery();
    }
    // Raw UUID comparison alone would accept this caller-pinned escaped text.
    std::strcpy(id.table_uuid, R"(\u00312345678-1234-1234-1234-123456789abc)");
    origin.next = 0;
    EXPECT_EQ(load(), Status::DecodeFailed); unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, NoGrowingArenaOrBodyAllocationConvenienceFallback) {
    metadata.set_growth_enabled(true); EXPECT_EQ(load(), Status::BadArg);
    metadata.set_growth_enabled(false); io.set_growth_enabled(true); EXPECT_EQ(load(), Status::BadArg);
    io.set_growth_enabled(false);
    bolt::Arena tiny{{64, 64, 64, false}}; tiny.set_growth_enabled(false);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &tiny, &io, &out, &result), Status::Capacity);
    EXPECT_EQ(origin.calls, 0u); unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, DescriptorStoreAndArenaAliasesLeaveEverythingUnchanged) {
    const auto saved = id;
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io,
        &out, reinterpret_cast<ice::MetadataLoadResult*>(&id)), Status::BadArg);
    EXPECT_EQ(std::memcmp(&id, &saved, sizeof(id)), 0);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io,
        reinterpret_cast<ice::Metadata*>(&result), &result), Status::BadArg);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io,
        &out, reinterpret_cast<ice::MetadataLoadResult*>(&store)), Status::BadArg);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io,
        &out, reinterpret_cast<ice::MetadataLoadResult*>(&vt)), Status::BadArg);
    EXPECT_EQ(store.vt, &vt); EXPECT_EQ(store.impl, &origin); EXPECT_EQ(vt.get_range_if_match, Origin::range);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io,
        reinterpret_cast<ice::Metadata*>(&metadata), &result), Status::BadArg);
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &metadata, &out, &result), Status::BadArg);
    auto* inside = metadata.allocate_array<ice::MetadataObjectIdentity>(1); ASSERT_NE(inside, nullptr); *inside = id;
    const auto allocated = metadata.total_allocated();
    EXPECT_EQ(ice::metadata_load_exact(&store, inside, &limits, &metadata, &io, &out, &result), Status::BadArg);
    EXPECT_EQ(metadata.total_allocated(), allocated);
    auto* retained_output = io.allocate_array<ice::MetadataLoadResult>(1);
    ASSERT_NE(retained_output, nullptr); *retained_output = result;
    const auto io_allocated = io.total_allocated();
    EXPECT_EQ(ice::metadata_load_exact(&store, &id, &limits, &metadata, &io, &out, retained_output), Status::BadArg);
    EXPECT_EQ(io.total_allocated(), io_allocated);
    EXPECT_EQ(std::memcmp(retained_output, result_before.data(), sizeof(result)), 0);
    EXPECT_EQ(origin.calls, 0u); unchanged(); no_discovery();
}

TEST_F(IcebergMetadataIO, LargerThanSixteenMiBMetadataUsesOnlyPinnedOneMiBRanges) {
    origin.bytes.append(17u << 20, ' '); id.version.size = origin.bytes.size();
    limits.max_bytes = id.version.size; limits.request_bytes = 1u << 20;
    // Existing fionn reserves one12-byte token per source byte. Admit that
    // exact parser workspace at startup rather than changing a body limit.
    constexpr size_t slack = 4u << 20;
    const size_t capacity = origin.bytes.size() + (origin.bytes.size() + 2u) * sizeof(bolt::parse::json::Token) + slack;
    bolt::Arena large{{capacity, capacity, 64, false}}; large.set_growth_enabled(false);
    ASSERT_EQ(ice::metadata_load_exact(&store, &id, &limits, &large, &io, &out, &result), Status::Ok);
    EXPECT_GT(result.object.version.size, 16u << 20); EXPECT_GT(origin.calls, 16u);
    EXPECT_LE(origin.largest, 1u << 20); EXPECT_EQ(origin.next, id.version.size); no_discovery();
    char expected[65]{};
    bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(), expected);
    EXPECT_STREQ(result.sha256, expected); EXPECT_EQ(out.last_sequence_number, 17);
    ASSERT_EQ(out.n_schemas, 1u); EXPECT_EQ(out.schemas[0].fields[0].id, 7);
}
} // namespace

TEST_F(IcebergMetadataIO, CurrentSnapshotPresenceSeparatesSignedMinusOneFromAbsence) {
    const auto original=origin.bytes;
    for(const auto value:std::array<const char*,3>{"-1","0","-2"}) {
        origin.bytes=original;
        auto at=origin.bytes.find("\"current-snapshot-id\":19"); ASSERT_NE(at,std::string::npos);
        origin.bytes.replace(at+22,2,value);
        at=origin.bytes.find("\"snapshot-id\":19"); ASSERT_NE(at,std::string::npos);
        origin.bytes.replace(at+14,2,value);
        id.version.size=origin.bytes.size(); origin.next=0; metadata.reset();
        ASSERT_EQ(load(),Status::Ok); EXPECT_TRUE(out.current_snapshot_present);
        EXPECT_EQ(out.current_snapshot_id,std::strtoll(value,nullptr,10));
    }
    for(const bool omit:std::array<bool,2>{false,true}) {
        origin.bytes=original; auto at=origin.bytes.find("\"current-snapshot-id\":19"); ASSERT_NE(at,std::string::npos);
        if(omit) origin.bytes.erase(at,25); else origin.bytes.replace(at+22,2,"null");
        id.version.size=origin.bytes.size(); origin.next=0; metadata.reset();
        ASSERT_EQ(load(),Status::Ok); EXPECT_FALSE(out.current_snapshot_present); EXPECT_EQ(out.current_snapshot_id,-1);
    }
}

TEST_F(IcebergMetadataIO, MalformedCurrentSnapshotCannotBecomeAbsent) {
    const auto original=origin.bytes;
    for(const auto value:std::array<const char*,7>{"true","\"19\"","[]","{}","19.5","9223372036854775808","-9223372036854775809"}) {
        origin.bytes=original;
        const auto at=origin.bytes.find("\"current-snapshot-id\":19"); ASSERT_NE(at,std::string::npos);
        origin.bytes.replace(at+22,2,value); id.version.size=origin.bytes.size();
        origin.next=0; metadata.reset();
        EXPECT_EQ(load(),Status::DecodeFailed); unchanged();
    }
}

TEST_F(IcebergMetadataIO, RetainedExactBytesUseSameIdentitySchemaAndDigestWithoutTransport) {
    origin.failure = bl::kOsIoError;
    ASSERT_EQ(ice::metadata_validate_exact(&id, &limits,
        reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(),
        &metadata, &out, &result), Status::Ok);
    EXPECT_EQ(origin.calls, 0u); no_discovery();
    EXPECT_EQ(out.current_schema_id, 9); ASSERT_EQ(out.n_schemas, 1u);
    EXPECT_EQ(out.schemas[0].fields[0].id, 7); EXPECT_TRUE(out.current_snapshot_present);
    EXPECT_EQ(std::memcmp(&result.object, &id, sizeof(id)), 0);
    char expected[65]{};
    bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(), expected);
    EXPECT_STREQ(result.sha256, expected);
    origin.bytes.assign(origin.bytes.size(), 'x');
    EXPECT_STREQ(out.schemas[0].fields[0].name, "value"); EXPECT_STREQ(out.table_uuid, kUuid);
}

TEST_F(IcebergMetadataIO, RetainedExtentIdentityAndCapacityFailurePublishNothing) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(origin.bytes.data());
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits, bytes, origin.bytes.size() - 1,
        &metadata, &out, &result), Status::IdentityMismatch); unchanged();
    limits.max_bytes = origin.bytes.size() - 1;
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits, bytes, origin.bytes.size(),
        &metadata, &out, &result), Status::Capacity); unchanged();
    limits.max_bytes = 1u << 20; id.table_uuid[0] = 'X';
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits, bytes, origin.bytes.size(),
        &metadata, &out, &result), Status::IdentityMismatch); unchanged();
    EXPECT_EQ(origin.calls, 0u); no_discovery();
}

TEST_F(IcebergMetadataIO, RetainedMalformedIdentityNeverPassesTheSharedValidator) {
    const auto at = origin.bytes.find("\"table-uuid\""); ASSERT_NE(at, std::string::npos);
    origin.bytes.replace(at, 12, "\"table-\\u0075uid\"");
    id.version.size = origin.bytes.size();
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits,
        reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(),
        &metadata, &out, &result), Status::DecodeFailed);
    unchanged(); EXPECT_EQ(origin.calls, 0u); no_discovery();
}

TEST_F(IcebergMetadataIO, RetainedAliasesRefuseBeforeArenaOrOutputMutation) {
    auto* input = static_cast<uint8_t*>(metadata.allocate(origin.bytes.size(), 1)); ASSERT_NE(input, nullptr);
    std::memcpy(input, origin.bytes.data(), origin.bytes.size());
    const auto used = metadata.total_allocated();
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits, input, origin.bytes.size(),
        &metadata, &out, &result), Status::BadArg);
    EXPECT_EQ(metadata.total_allocated(), used); unchanged();
    EXPECT_EQ(std::memcmp(input, origin.bytes.data(), origin.bytes.size()), 0);
    auto* alias = static_cast<ice::Metadata*>(metadata.allocate(sizeof(ice::Metadata), alignof(ice::Metadata)));
    ASSERT_NE(alias, nullptr); alias->format_version = -77;
    EXPECT_EQ(ice::metadata_validate_exact(&id, &limits,
        reinterpret_cast<const uint8_t*>(origin.bytes.data()), origin.bytes.size(),
        &metadata, alias, &result), Status::BadArg);
    EXPECT_EQ(alias->format_version, -77); EXPECT_EQ(origin.calls, 0u); no_discovery();
}
