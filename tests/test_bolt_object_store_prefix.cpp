#include <gtest/gtest.h>
#include "bolt/lakehouse/object_store/prefix.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
namespace bl = bolt::lakehouse;
struct Origin {
    uint32_t calls = 0;
    int status = bl::kOsOk;
    char last[bl::kOsMaxKey]{};
    const uint8_t* payload = nullptr;
    uint64_t bytes = 0, offset = 0;
    const bl::ObjectVersion* expected = nullptr;
    const char* listed[2]{"root/table/metadata/v1.json", "root/table/metadata/v2.json"};
    uint32_t listed_count = 2;
    bool invalid_count = false;
};
Origin* record(void* context, const char* key) {
    auto* state = static_cast<Origin*>(context);
    ++state->calls;
    std::snprintf(state->last, sizeof(state->last), "%s", key);
    return state;
}
int get(void* c, const char* key, bolt::Arena* arena, const uint8_t** data, uint64_t* n) noexcept {
    auto* state = record(c, key);
    auto* bytes = arena->allocate_array<uint8_t>(1);
    if (bytes == nullptr) return bl::kOsBufferSmall;
    *bytes = 42;
    *data = bytes;
    *n = 1;
    return state->status;
}
int put(void* c, const char* key, const uint8_t* data, uint64_t n) noexcept {
    auto* state = record(c, key);
    state->payload = data;
    state->bytes = n;
    return state->status;
}
int erase(void* c, const char* key) noexcept { return record(c, key)->status; }
int head(void* c, const char* key, bl::ObjectMeta* out) noexcept {
    auto* state = record(c, key);
    *out = {4, 0, true, {}};
    return state->status;
}
int list(void* c, const char* key, bl::ObjectEntry* out, uint32_t cap, uint32_t* n) noexcept {
    auto* state = record(c, key);
    *n = state->invalid_count ? cap + 1u : std::min(state->listed_count, cap);
    if (state->invalid_count) return bl::kOsOk;
    for (uint32_t i = 0; i < *n; ++i) {
        std::snprintf(out[i].key, sizeof(out[i].key), "%s", state->listed[i]);
        out[i].size = i + 1u;
    }
    return state->status == bl::kOsOk && state->listed_count > cap
        ? bl::kOsBufferSmall : state->status;
}
int range(void* c, const char* key, uint64_t off, uint64_t n,
          uint8_t* dst, uint64_t* got) noexcept {
    auto* state = record(c, key);
    state->offset = off;
    state->bytes = n;
    if (state->status != bl::kOsOk) { *got = 0; return state->status; }
    std::memset(dst, 42, static_cast<size_t>(n));
    *got = n;
    return state->status;
}
int version(void* c, const char* key, bolt::Arena*, bl::ObjectVersion* out) noexcept {
    auto* state = record(c, key);
    *out = {4, "\"origin-v1\""};
    return state->status;
}
int pinned(void* c, const char* key, const bl::ObjectVersion* expected, uint64_t off,
           uint32_t n, bolt::Arena*, uint8_t* dst, uint32_t* got) noexcept {
    auto* state = record(c, key);
    state->expected = expected;
    uint64_t count = 0;
    // Record exactly one origin operation; reuse only the tiny byte-copy body.
    state->offset = off;
    state->bytes = n;
    if (state->status == bl::kOsOk) {
        std::memset(dst, 42, n);
        count = n;
    }
    *got = static_cast<uint32_t>(count);
    return state->status;
}
const bl::ObjectStoreVT kVT{get, put, list, erase, head, put, range, version, pinned};
struct Fixture {
    Origin origin;
    bl::ObjectStore underlying{&kVT, &origin};
    bl::PrefixObjectStore state;
    bl::ObjectStore store{};
    Fixture() { EXPECT_TRUE(bl::prefix_object_store_init(&state, &underlying, "root/table", &store)); }
};
}  // namespace

TEST(ObjectStorePrefix, MapsAllOrdinaryOperationsWithoutChangingBuffers) {
    Fixture f;
    bolt::Arena arena;
    const uint8_t body[]{3, 8};
    ASSERT_EQ(bl::os_put(&f.store, "data/a.parquet", body, sizeof(body)), bl::kOsOk);
    EXPECT_STREQ(f.origin.last, "root/table/data/a.parquet");
    EXPECT_EQ(f.origin.payload, body);
    EXPECT_EQ(f.origin.bytes, sizeof(body));
    const uint8_t* got = nullptr;
    uint64_t size = 0;
    ASSERT_EQ(bl::os_get(&f.store, "data/a.parquet", &arena, &got, &size), bl::kOsOk);
    ASSERT_EQ(size, 1u);
    EXPECT_EQ(got[0], 42);
    bl::ObjectMeta meta{};
    ASSERT_EQ(bl::os_head(&f.store, "data/a.parquet", &meta), bl::kOsOk);
    EXPECT_EQ(meta.size, 4u);
    std::array<uint8_t, 2> range_bytes{};
    ASSERT_EQ(bl::os_get_range(&f.store, "data/a.parquet", 1, 2, range_bytes.data(), &size), bl::kOsOk);
    EXPECT_EQ(range_bytes, (std::array<uint8_t, 2>{42, 42}));
    EXPECT_EQ(f.origin.offset, 1u);
    ASSERT_EQ(bl::os_delete(&f.store, "data/a.parquet"), bl::kOsOk);
    EXPECT_EQ(f.origin.calls, 5u);
}

TEST(ObjectStorePrefix, PreservesConditionalOutcomesAndNeverRetries) {
    Fixture f;
    const uint8_t body[]{9};
    for (const int status : {bl::kOsOk, bl::kOsExists, bl::kOsWriteUncertain, bl::kOsIoError}) {
        f.origin.status = status;
        const uint32_t before = f.origin.calls;
        EXPECT_EQ(bl::os_put_if_absent(&f.store, "metadata/v1.json", body, 1), status);
        EXPECT_EQ(f.origin.calls, before + 1u);
        EXPECT_STREQ(f.origin.last, "root/table/metadata/v1.json");
    }
}

TEST(ObjectStorePrefix, PreservesStrongValidatorAndPreconditionFailure) {
    Fixture f;
    bolt::Arena arena;
    bl::ObjectVersion version{};
    ASSERT_EQ(bl::os_head_versioned(&f.store, "data/a", &arena, &version), bl::kOsOk);
    EXPECT_STREQ(version.etag, "\"origin-v1\"");
    std::array<uint8_t, 2> bytes{};
    uint32_t got = 0;
    ASSERT_EQ(bl::os_get_range_if_match(&f.store, "data/a", &version, 1, 2, &arena, bytes.data(), &got), bl::kOsOk);
    EXPECT_EQ(f.origin.expected, &version);
    EXPECT_EQ(got, 2u);
    f.origin.status = bl::kOsPreconditionFailed;
    ASSERT_EQ(bl::os_get_range_if_match(&f.store, "data/a", &version, 1, 2, &arena, bytes.data(), &got), bl::kOsPreconditionFailed);
    EXPECT_EQ(got, 0u);
    EXPECT_EQ(f.origin.calls, 3u);
}

TEST(ObjectStorePrefix, ListingStripsOnlyRootAndPreservesCapacityOutcome) {
    Fixture f;
    std::array<bl::ObjectEntry, 2> entries{};
    uint32_t count = 0;
    ASSERT_EQ(bl::os_list(&f.store, "metadata/", entries.data(), 2, &count), bl::kOsOk);
    EXPECT_EQ(count, 2u);
    EXPECT_STREQ(entries[0].key, "metadata/v1.json");
    EXPECT_STREQ(entries[1].key, "metadata/v2.json");
    EXPECT_EQ(entries[1].size, 2u);
    ASSERT_EQ(bl::os_list(&f.store, "", entries.data(), 1, &count), bl::kOsBufferSmall);
    EXPECT_EQ(count, 1u);
    EXPECT_STREQ(entries[0].key, "metadata/v1.json");
    EXPECT_STREQ(f.origin.last, "root/table/");
    EXPECT_EQ(bl::os_list(&f.store, "", nullptr, 0, &count), bl::kOsBufferSmall);
    EXPECT_EQ(count, 0u);
}

TEST(ObjectStorePrefix, RefusesOutOfRequestedPrefixOrMalformedOriginListing) {
    Fixture f;
    std::array<bl::ObjectEntry, 2> entries{};
    uint32_t count = 9;
    for (const char* bad : {"other/table/metadata/x", "root/table/data/x", "root/table/../x"}) {
        f.origin.listed[1] = bad;
        EXPECT_EQ(bl::os_list(&f.store, "metadata/", entries.data(), 2, &count), bl::kOsIoError);
        EXPECT_EQ(count, 0u);
    }
    f.origin.invalid_count = true;
    EXPECT_EQ(bl::os_list(&f.store, "", entries.data(), 2, &count), bl::kOsIoError);
    EXPECT_EQ(count, 0u);
}

TEST(ObjectStorePrefix, RejectsInvalidKeysBeforeOriginEffects) {
    Fixture f;
    const uint8_t body[]{1};
    for (const char* key : {"/escape", "../escape", "data/../escape", "data/./x",
                            "data//x", "data\\x", "data/\nx"}) {
        EXPECT_EQ(bl::os_put(&f.store, key, body, 1), bl::kOsBadArg) << key;
    }
    EXPECT_EQ(f.origin.calls, 0u);
    const std::string overflow(bl::kOsMaxKey, 'a');
    EXPECT_EQ(bl::os_put(&f.store, overflow.c_str(), body, 1), bl::kOsBadArg);
    EXPECT_EQ(f.origin.calls, 0u);
}

TEST(ObjectStorePrefix, InvalidInitPreservesOutputAndActiveState) {
    Fixture f;
    const auto* vt = f.store.vt;
    for (const char* prefix : {"", "/root", "../root", "root//table"}) {
        EXPECT_FALSE(bl::prefix_object_store_init(&f.state, &f.underlying, prefix, &f.store));
        EXPECT_EQ(f.store.vt, vt);
        EXPECT_STREQ(f.state.prefix, "root/table/");
    }
    EXPECT_FALSE(bl::prefix_object_store_init(&f.state, &f.store, "again", &f.store));
    EXPECT_FALSE(bl::prefix_object_store_init(&f.state, &f.underlying, "again", &f.state.origin));
    const uint8_t body[]{1};
    EXPECT_EQ(bl::os_put(&f.store, "still-valid", body, 1), bl::kOsOk);
    EXPECT_STREQ(f.origin.last, "root/table/still-valid");
}

TEST(ObjectStorePrefix, RootMarkerListingRemainsAddressableInsideThePrefix) {
    Fixture f;
    f.origin.listed[0] = "root/table/";
    f.origin.listed_count = 1;
    bl::ObjectEntry entry{};
    uint32_t count = 0;
    ASSERT_EQ(bl::os_list(&f.store, "", &entry, 1, &count), bl::kOsOk);
    ASSERT_EQ(count, 1u);
    EXPECT_STREQ(entry.key, "");
    bolt::Arena arena;
    const uint8_t* bytes = nullptr;
    uint64_t length = 0;
    ASSERT_EQ(bl::os_get(&f.store, entry.key, &arena, &bytes, &length), bl::kOsOk);
    EXPECT_STREQ(f.origin.last, "root/table/");
    EXPECT_EQ(length, 1u);
    EXPECT_EQ(bytes[0], 42);
}

TEST(ObjectStorePrefix, DirectVersionedCallbacksRejectAliasesBeforeOutputOrOriginEffects) {
    Fixture f;
    bolt::Arena arena;
    bl::ObjectVersion identity{4, "\"v1\""};
    std::array<char, 64> key{};
    std::memcpy(key.data(), "data/a", 7);
    const auto saved = key;
    uint32_t got = 17;
    EXPECT_EQ(f.store.vt->get_range_if_match(f.store.impl, key.data(), &identity, 0, 4,
        &arena, reinterpret_cast<uint8_t*>(key.data()), &got), bl::kOsBadArg);
    EXPECT_EQ(key, saved);
    EXPECT_EQ(got, 17u);
    std::array<uint8_t, 4> output{1, 2, 3, 4};
    EXPECT_EQ(f.store.vt->get_range_if_match(f.store.impl, key.data(), &identity, 0, 4,
        &arena, output.data(), reinterpret_cast<uint32_t*>(&identity.size)), bl::kOsBadArg);
    EXPECT_EQ(identity.size, 4u);
    EXPECT_EQ(output, (std::array<uint8_t, 4>{1, 2, 3, 4}));
    std::memcpy(identity.etag, "data/a", 7);
    const auto original = identity;
    EXPECT_EQ(f.store.vt->head_versioned(f.store.impl, identity.etag, &arena, &identity), bl::kOsBadArg);
    EXPECT_EQ(std::memcmp(&identity, &original, sizeof(identity)), 0);
    EXPECT_EQ(f.store.vt->head_versioned(f.store.impl, "data/a", &arena,
        reinterpret_cast<bl::ObjectVersion*>(&f.state)), bl::kOsBadArg);
    EXPECT_EQ(f.origin.calls, 0u);
    EXPECT_STREQ(f.state.prefix, "root/table/");
}

TEST(ObjectStorePrefix, DisjointRefusedKeysPublishNoRangeOrVersionResult) {
    Fixture f;
    bolt::Arena arena;
    std::array<uint8_t, 2> bytes{7, 8};
    bl::ObjectVersion expected{4, "\"v1\""}, got_version = expected;
    uint64_t legacy_length = 17;
    uint32_t versioned_length = 17;
    EXPECT_EQ(bl::os_get_range(&f.store, "../escape", 0, 2, bytes.data(), &legacy_length), bl::kOsBadArg);
    EXPECT_EQ(legacy_length, 0u);
    EXPECT_EQ(bl::os_get_range_if_match(&f.store, "../escape", &expected, 0, 2, &arena,
        bytes.data(), &versioned_length), bl::kOsBadArg);
    EXPECT_EQ(versioned_length, 0u);
    EXPECT_EQ(bl::os_head_versioned(&f.store, "../escape", &arena, &got_version), bl::kOsBadArg);
    EXPECT_EQ(got_version.size, 0u);
    EXPECT_EQ(got_version.etag[0], 0);
    EXPECT_EQ(bytes, (std::array<uint8_t, 2>{7, 8}));
    EXPECT_EQ(f.origin.calls, 0u);
}

TEST(ObjectStorePrefix, OptionalOriginOperationsRemainUnavailable) {
    Fixture f;
    bl::ObjectStoreVT minimal = kVT;
    minimal.put_if_absent = nullptr;
    minimal.get_range = nullptr;
    minimal.head_versioned = nullptr;
    minimal.get_range_if_match = nullptr;
    f.underlying.vt = &minimal;
    ASSERT_TRUE(bl::prefix_object_store_init(&f.state, &f.underlying, "root", &f.store));
    EXPECT_EQ(f.store.vt->put_if_absent, nullptr);
    EXPECT_EQ(f.store.vt->get_range, nullptr);
    EXPECT_EQ(f.store.vt->head_versioned, nullptr);
    EXPECT_EQ(f.store.vt->get_range_if_match, nullptr);
    const uint8_t body[]{1};
    EXPECT_EQ(bl::os_put_if_absent(&f.store, "x", body, 1), bl::kOsNotImplemented);
    EXPECT_EQ(f.origin.calls, 0u);
}

TEST(ObjectStorePrefix, FlattensNestedRootsAndBoundsJoinedKeys) {
    Fixture f;
    bl::PrefixObjectStore nested;
    bl::ObjectStore view{};
    ASSERT_TRUE(bl::prefix_object_store_init(&nested, &f.store, "child/", &view));
    EXPECT_EQ(nested.origin.impl, &f.origin);
    EXPECT_EQ(nested.origin.vt, &kVT);
    const uint8_t body[]{1};
    ASSERT_EQ(bl::os_put(&view, "x", body, 1), bl::kOsOk);
    EXPECT_STREQ(f.origin.last, "root/table/child/x");
    const std::string prefix(bl::kOsMaxKey - 3u, 'a');
    ASSERT_TRUE(bl::prefix_object_store_init(&nested, &f.underlying, prefix.c_str(), &view));
    EXPECT_EQ(bl::os_put(&view, "x", body, 1), bl::kOsOk);
    EXPECT_EQ(std::strlen(f.origin.last), bl::kOsMaxKey - 1u);
    const uint32_t before = f.origin.calls;
    EXPECT_EQ(bl::os_put(&view, "xy", body, 1), bl::kOsBadArg);
    EXPECT_EQ(f.origin.calls, before);
}
