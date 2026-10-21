#include "bolt/lakehouse/object_store/range_cache.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace bolt::lakehouse;
constexpr uint32_t kPage = kObjectRangePageBytes;
uint8_t pattern(uint64_t offset, uint64_t generation) {
    return static_cast<uint8_t>((offset * 131u) ^ (offset >> 11u) ^ generation);
}
struct MemoryOrigin {
    std::array<std::vector<uint8_t>, 2> bytes;
    uint64_t generation[2]{1, 1};
    bool exists[2]{true, true};
    uint32_t heads = 0, ranges = 0, writes = 0, deletes = 0;
    uint64_t range_bytes = 0;
    int write_status = kOsOk, head_status = kOsOk, range_status = kOsOk;
    bool short_range = false, weak_head = false, change_on_range = false;
    ObjectRangeCache* cache = nullptr;
    uint64_t invalidation_before_write = 0;
    ObjectStore store{};
    static uint32_t index(const char* key) { return std::strcmp(key, "a") == 0 ? 0u : 1u; }
    void replace(uint32_t i, size_t length) {
        ++generation[i]; bytes[i].resize(length); exists[i] = true;
        for (size_t off = 0; off < length; ++off) bytes[i][off] = pattern(off, generation[i]);
    }
    static MemoryOrigin& self(void* context) { return *static_cast<MemoryOrigin*>(context); }
    static int version(void* context, const char* key, bolt::Arena*, ObjectVersion* out) noexcept {
        auto& s = self(context); ++s.heads; *out = {};
        if (s.head_status != kOsOk) return s.head_status;
        const uint32_t i = index(key);
        if (!s.exists[i]) return kOsNotFound;
        out->size = s.bytes[i].size();
        std::snprintf(out->etag, sizeof(out->etag), "%s\"opaque-%llu\"", s.weak_head ? "W/" : "",
                      static_cast<unsigned long long>(s.generation[i]));
        return kOsOk;
    }
    static int range(void* context, const char* key, const ObjectVersion* expected, uint64_t off,
                      uint32_t length, bolt::Arena*, uint8_t* dst, uint32_t* out) noexcept {
        auto& s = self(context); ++s.ranges; *out = 0;
        const uint32_t i = index(key);
        if (s.change_on_range) { s.change_on_range = false; s.replace(i, s.bytes[i].size()); }
        if (s.range_status != kOsOk) return s.range_status;
        char etag[64]{};
        std::snprintf(etag, sizeof(etag), "\"opaque-%llu\"", static_cast<unsigned long long>(s.generation[i]));
        if (expected->size != s.bytes[i].size() || std::strcmp(etag, expected->etag) != 0) return kOsPreconditionFailed;
        if (!s.exists[i]) return kOsNotFound;
        if (off > s.bytes[i].size() || length > s.bytes[i].size() - off) return kOsBadArg;
        const uint32_t actual = s.short_range && length != 0 ? length - 1u : length;
        if (actual != 0) std::memcpy(dst, s.bytes[i].data() + off, actual);
        *out = actual; s.range_bytes += actual;
        return kOsOk;
    }
    void observe_invalidation() {
        if (cache == nullptr) return;
        ObjectRangeCacheStats stats{};
        object_range_cache_stats(cache, &stats);
        invalidation_before_write = stats.invalidated_pages;
    }
    static int put(void* context, const char* key, const uint8_t* data, uint64_t length) noexcept {
        auto& s = self(context); ++s.writes; s.observe_invalidation();
        if (s.write_status == kOsOk || s.write_status == kOsWriteUncertain) {
            const uint32_t i = index(key); ++s.generation[i]; s.exists[i] = true;
            if (length == 0) s.bytes[i].clear();
            else s.bytes[i].assign(data, data + length);
        }
        return s.write_status;
    }
    static int create(void* context, const char* key, const uint8_t* data, uint64_t length) noexcept {
        auto& s = self(context);
        s.observe_invalidation();
        if (s.exists[index(key)] && s.write_status == kOsOk) { ++s.writes; return kOsExists; }
        return put(context, key, data, length);
    }
    static int remove(void* context, const char* key) noexcept {
        auto& s = self(context); ++s.deletes; s.observe_invalidation();
        if (s.write_status == kOsOk) s.exists[index(key)] = false;
        return s.write_status;
    }
    static int get(void* context, const char* key, bolt::Arena* arena, const uint8_t** data, uint64_t* length) noexcept {
        auto& s = self(context); const uint32_t i = index(key); *data = nullptr; *length = 0;
        if (!s.exists[i]) return kOsNotFound;
        auto* buffer = static_cast<uint8_t*>(arena->copy_into(s.bytes[i].data(), s.bytes[i].size(), 1));
        if (buffer == nullptr && !s.bytes[i].empty()) return kOsBufferSmall;
        *data = buffer; *length = s.bytes[i].size(); return kOsOk;
    }
    static int list(void*, const char*, ObjectEntry*, uint32_t, uint32_t* count) noexcept { *count = 0; return kOsOk; }
    static int head(void* context, const char* key, ObjectMeta* out) noexcept {
        auto& s = self(context); const uint32_t i = index(key); *out = {};
        if (!s.exists[i]) return kOsNotFound;
        out->size = s.bytes[i].size(); out->exists = true; return kOsOk;
    }
    void initialize() {
        static const ObjectStoreVT vt{get, put, list, remove, head, create, nullptr, version, range};
        store = ObjectStore{&vt, this};
        replace(0, 3u * kPage + 13u); replace(1, 2u * kPage + 7u);
    }
};
class ObjectRangeCacheTest : public ::testing::Test {
protected:
    MemoryOrigin origin;
    bolt::Arena backing{{128u << 10, 64u << 20, 64, false}};
    bolt::Arena io{{kPage, kPage, 64, false}};
    ObjectRangeCache* cache = nullptr;
    ObjectStore adapter{};
    int initialize(uint32_t pages) {
        backing.reset(); io.reset();
        const int rc = object_range_cache_init(&cache, &origin.store, &backing, &io, pages, &adapter);
        origin.cache = cache; return rc;
    }
    void SetUp() override { origin.initialize(); ASSERT_EQ(initialize(2), kOsOk); }
    int read(uint64_t offset, uint64_t length, std::vector<uint8_t>* out,
              const ObjectVersion* expected = nullptr, const char* key = "a") {
        out->assign(static_cast<size_t>(length), 0);
        uint64_t got = UINT64_MAX;
        const int rc = object_range_cache_read(cache, key, expected, offset, length, out->data(), &got);
        EXPECT_EQ(got, rc == kOsOk ? length : 0u);
        if (rc == kOsOk) {
            const auto& data = origin.bytes[MemoryOrigin::index(key)];
            EXPECT_TRUE(std::equal(out->begin(), out->end(), data.begin() + static_cast<std::ptrdiff_t>(offset)));
        }
        return rc;
    }
    ObjectRangeCacheStats stats() { ObjectRangeCacheStats s{}; object_range_cache_stats(cache, &s); return s; }
};

TEST_F(ObjectRangeCacheTest, RepeatedReadsValidateHeadAndReduceActualOriginBytes) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(17, 100, &output), kOsOk);
    ASSERT_EQ(read(17, 100, &output), kOsOk);
    const auto s = stats();
    EXPECT_EQ(origin.heads, 2u); EXPECT_EQ(origin.ranges, 1u);
    EXPECT_EQ(origin.range_bytes, kPage);
    EXPECT_EQ(s.head_calls, 2u); EXPECT_EQ(s.range_calls, 1u);
    EXPECT_EQ(s.origin_bytes, origin.range_bytes); EXPECT_EQ(s.requested_bytes, 200u);
    EXPECT_EQ(s.page_hits, 1u); EXPECT_EQ(s.page_misses, 1u);
}
TEST_F(ObjectRangeCacheTest, DisabledCacheUsesExactValidatedOriginRanges) {
    ASSERT_EQ(initialize(0), kOsOk);
    std::vector<uint8_t> output;
    ASSERT_EQ(read(17, 100, &output), kOsOk); ASSERT_EQ(read(17, 100, &output), kOsOk);
    EXPECT_EQ(origin.heads, 2u); EXPECT_EQ(origin.ranges, 2u); EXPECT_EQ(origin.range_bytes, 200u);
    EXPECT_EQ(stats().fills, 0u); EXPECT_EQ(stats().page_hits, 0u);
}
TEST_F(ObjectRangeCacheTest, FullRepeatedPageHasMeasurableCacheOnOffControl) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(0, kPage, &output), kOsOk); ASSERT_EQ(read(0, kPage, &output), kOsOk);
    EXPECT_EQ(origin.range_bytes, kPage); EXPECT_EQ(origin.ranges, 1u);
    const uint64_t before = origin.range_bytes;
    ASSERT_EQ(initialize(0), kOsOk);
    ASSERT_EQ(read(0, kPage, &output), kOsOk); ASSERT_EQ(read(0, kPage, &output), kOsOk);
    EXPECT_EQ(origin.range_bytes - before, 2u * kPage);
    EXPECT_EQ(stats().range_calls, 2u);
}
TEST_F(ObjectRangeCacheTest, CrossPageLastPageAndLruStayExactAndBounded) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(kPage - 3u, 10, &output), kOsOk);
    ASSERT_EQ(read(3u * kPage, 13, &output), kOsOk);
    EXPECT_EQ(stats().evictions, 1u);
    ASSERT_EQ(read(0, 1, &output), kOsOk);
    EXPECT_EQ(stats().evictions, 2u); EXPECT_EQ(stats().fills, 4u);
    uint64_t got = 1;
    EXPECT_EQ(os_get_range(&adapter, "a", 4u * kPage, 1, output.data(), &got), kOsOk);
    EXPECT_EQ(got, 0u);
}
TEST_F(ObjectRangeCacheTest, SameSizeExternalReplacementNeverUsesOldBytes) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(0, 100, &output), kOsOk);
    const auto old = output;
    origin.replace(0, origin.bytes[0].size());
    ASSERT_EQ(read(0, 100, &output), kOsOk);
    EXPECT_NE(output, old); EXPECT_EQ(origin.ranges, 2u); EXPECT_EQ(stats().invalidated_pages, 1u);
}
TEST_F(ObjectRangeCacheTest, PinnedVersionAndChangedBetweenHeadAndRangeFailExplicitly) {
    ObjectVersion version{};
    ASSERT_EQ(object_range_cache_head(cache, "a", &version), kOsOk);
    origin.replace(0, origin.bytes[0].size());
    std::vector<uint8_t> output;
    EXPECT_EQ(read(0, 1, &output, &version), kOsPreconditionFailed);
    EXPECT_EQ(origin.ranges, 0u);
    origin.change_on_range = true;
    EXPECT_EQ(read(0, 1, &output), kOsPreconditionFailed);
    EXPECT_EQ(stats().fills, 0u); EXPECT_EQ(stats().preconditions, 2u);
}
TEST_F(ObjectRangeCacheTest, KeysAndExtentsArePartOfPageIdentity) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(0, 16, &output, nullptr, "a"), kOsOk);
    ASSERT_EQ(read(0, 16, &output, nullptr, "b"), kOsOk);
    EXPECT_EQ(origin.ranges, 2u);
    origin.bytes[0].resize(17); // Even a broken provider reusing ETag cannot alias a different extent.
    ASSERT_EQ(read(0, 17, &output, nullptr, "a"), kOsOk);
    EXPECT_EQ(origin.ranges, 3u);
}
TEST_F(ObjectRangeCacheTest, ErrorsWeakVersionsAndShortRangesCannotPopulate) {
    std::vector<uint8_t> output;
    origin.weak_head = true;
    EXPECT_EQ(read(0, 1, &output), kOsIoError);
    origin.weak_head = false; origin.short_range = true;
    EXPECT_EQ(read(0, 1, &output), kOsIoError);
    origin.short_range = false; origin.head_status = kOsIoError;
    EXPECT_EQ(read(0, 1, &output), kOsIoError);
    EXPECT_EQ(stats().fills, 0u);
    origin.head_status = kOsOk;
    ASSERT_EQ(read(0, 1, &output), kOsOk);
    EXPECT_EQ(stats().fills, 1u);
}
TEST_F(ObjectRangeCacheTest, WritesInvalidateBeforeOriginAndAccountAckUncertaintyFailureBytes) {
    std::vector<uint8_t> output;
    ASSERT_EQ(read(0, 1, &output), kOsOk);
    const uint8_t body[]{9, 8, 7};
    EXPECT_EQ(os_put_if_absent(&adapter, "a", body, sizeof(body)), kOsExists);
    EXPECT_EQ(origin.invalidation_before_write, 1u);
    ASSERT_EQ(read(0, 1, &output), kOsOk);
    origin.write_status = kOsWriteUncertain;
    EXPECT_EQ(os_put(&adapter, "a", body, sizeof(body)), kOsWriteUncertain);
    EXPECT_EQ(origin.invalidation_before_write, 2u);
    ASSERT_EQ(read(0, sizeof(body), &output), kOsOk);
    origin.write_status = kOsIoError;
    EXPECT_EQ(os_put(&adapter, "a", body, sizeof(body)), kOsIoError);
    EXPECT_EQ(origin.invalidation_before_write, 3u);
    origin.write_status = kOsOk;
    EXPECT_EQ(os_put(&adapter, "a", body, sizeof(body)), kOsOk);
    auto s = stats();
    EXPECT_EQ(s.write_attempts, 4u); EXPECT_EQ(s.write_attempted_bytes, 12u);
    EXPECT_EQ(s.write_ack_bytes, 3u); EXPECT_EQ(s.write_uncertain, 1u); EXPECT_EQ(s.write_failures, 2u);
    EXPECT_EQ(origin.writes, 4u); // No hidden retries.
    ASSERT_EQ(read(0, 1, &output), kOsOk);
    EXPECT_EQ(os_delete(&adapter, "a"), kOsOk);
    EXPECT_GE(origin.invalidation_before_write, 4u);
    EXPECT_EQ(read(0, 1, &output), kOsNotFound);
    EXPECT_EQ(stats().delete_attempts, 1u);
}
TEST_F(ObjectRangeCacheTest, StartupAndReadBudgetsRefuseBeforeOriginEffects) {
    ObjectRangeCache* rejected = nullptr;
    ObjectStore adapter_out{};
    EXPECT_EQ(object_range_cache_init(&rejected, &origin.store, &backing, &io,
        kObjectRangeMaxPages + 1u, &adapter_out), kOsBadArg);
    EXPECT_EQ(rejected, nullptr);
    uint8_t byte = 0; uint64_t got = 7;
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, kObjectRangeMaxReadBytes + 1,
                                      &byte, &got), kOsBufferSmall);
    EXPECT_EQ(got, 0u); EXPECT_EQ(origin.heads, 0u);
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, UINT64_MAX, 1, &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 0u); EXPECT_EQ(origin.ranges, 0u);
}
}  // namespace

namespace {
TEST_F(ObjectRangeCacheTest, IoArenaAliasesAreRefusedBeforeResetOrOriginRequest) {
    auto* bytes = static_cast<uint8_t*>(io.allocate(128, 1));
    ASSERT_NE(bytes, nullptr);
    uint64_t got = 1;
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 1, bytes, &got), kOsBadArg);
    EXPECT_EQ(got, 1u); EXPECT_EQ(origin.heads, 0u);
    const char key[]{'a', 0};
    std::memcpy(bytes, key, sizeof(key));
    uint8_t destination = 0;
    EXPECT_EQ(object_range_cache_read(cache, reinterpret_cast<const char*>(bytes), nullptr,
                                      0, 1, &destination, &got), kOsBadArg);
    EXPECT_EQ(origin.heads, 0u);
}
TEST_F(ObjectRangeCacheTest, UnsupportedLegacyOriginIsNeverSilentlyEmulated) {
    ObjectStoreVT legacy = *origin.store.vt;
    legacy.head_versioned = nullptr; legacy.get_range_if_match = nullptr;
    ObjectStore store{&legacy, &origin};
    ObjectRangeCache* absent = nullptr; ObjectStore unused{};
    EXPECT_EQ(object_range_cache_init(&absent, &store, &backing, &io, 0, &unused), kOsNotImplemented);
    EXPECT_EQ(absent, nullptr); EXPECT_EQ(origin.heads, 0u);
}
}  // namespace

namespace {
TEST_F(ObjectRangeCacheTest, MaximumLogicalReadAcrossSixtyFivePagesStaysWithinSixtyFourSlots) {
    origin.replace(0, static_cast<size_t>(kObjectRangeMaxReadBytes + 1u));
    ASSERT_EQ(initialize(kObjectRangeMaxPages), kOsOk);
    std::vector<uint8_t> output;
    ASSERT_EQ(read(1, kObjectRangeMaxReadBytes, &output), kOsOk);
    EXPECT_EQ(origin.ranges, kObjectRangeMaxPages + 1u);
    EXPECT_EQ(stats().fills, kObjectRangeMaxPages + 1u);
    EXPECT_EQ(stats().evictions, 1u);
    EXPECT_EQ(stats().origin_bytes, kObjectRangeMaxReadBytes + 1u);
}
}  // namespace

namespace {
TEST_F(ObjectRangeCacheTest, KeyAndDestinationAliasesLeaveIdentityAndCountersUnchanged) {
    std::array<uint8_t, 64> key{}; key[0] = 'a';
    const auto before = key; uint64_t got = 91;
    EXPECT_EQ(object_range_cache_read(cache, reinterpret_cast<const char*>(key.data()), nullptr,
        0, key.size(), key.data(), &got), kOsBadArg);
    EXPECT_EQ(key, before); EXPECT_EQ(got, 91u);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
}
TEST_F(ObjectRangeCacheTest, OutputLengthCannotAliasKeyExpectedOrDestination) {
    alignas(uint64_t) std::array<char, 32> key{}; key[0] = 'a';
    const auto saved_key = key;
    ObjectVersion expected{}; expected.size = origin.bytes[0].size(); std::strcpy(expected.etag, "\"opaque-2\"");
    const auto saved_expected = expected;
    alignas(uint64_t) std::array<uint8_t, 32> dst{}; dst.fill(0xa5); const auto saved_dst = dst;
    EXPECT_EQ(object_range_cache_read(cache, key.data(), nullptr, 0, 1, dst.data(),
        reinterpret_cast<uint64_t*>(key.data())), kOsBadArg);
    EXPECT_EQ(object_range_cache_read(cache, "a", &expected, 0, 1, dst.data(), &expected.size), kOsBadArg);
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 1, dst.data(),
        reinterpret_cast<uint64_t*>(dst.data())), kOsBadArg);
    EXPECT_EQ(key, saved_key); EXPECT_EQ(std::memcmp(&expected, &saved_expected, sizeof(expected)), 0);
    EXPECT_EQ(dst, saved_dst); EXPECT_EQ(origin.heads, 0u);
}
TEST_F(ObjectRangeCacheTest, ExpectedDescriptorCannotAliasDestinationOrResetWorkspace) {
    ObjectVersion version{}; version.size = origin.bytes[0].size(); std::strcpy(version.etag, "\"opaque-2\"");
    const auto saved = version; uint64_t got = 19;
    EXPECT_EQ(object_range_cache_read(cache, "a", &version, 0, 1,
        reinterpret_cast<uint8_t*>(&version), &got), kOsBadArg);
    EXPECT_EQ(std::memcmp(&saved, &version, sizeof(version)), 0); EXPECT_EQ(got, 19u);
    auto* borrowed = io.allocate_array<ObjectVersion>(1); ASSERT_NE(borrowed, nullptr); *borrowed = version;
    uint8_t byte = 0;
    EXPECT_EQ(object_range_cache_read(cache, "a", borrowed, 0, 1, &byte, &got), kOsBadArg);
    EXPECT_EQ(std::memcmp(&saved, borrowed, sizeof(version)), 0); EXPECT_EQ(got, 19u);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
}
TEST_F(ObjectRangeCacheTest, PrivateMetadataAndArenaObjectsCannotBeOutputs) {
    uint8_t byte = 0; uint64_t got = 71;
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 1,
        reinterpret_cast<uint8_t*>(cache), &got), kOsBadArg);
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 1, &byte,
        reinterpret_cast<uint64_t*>(cache)), kOsBadArg);
    EXPECT_EQ(object_range_cache_head(cache, "a", reinterpret_cast<ObjectVersion*>(cache)), kOsBadArg);
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 1,
        reinterpret_cast<uint8_t*>(&backing), &got), kOsBadArg);
    EXPECT_EQ(got, 71u); EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
    std::vector<uint8_t> valid; EXPECT_EQ(read(0, 1, &valid), kOsOk);
}
TEST_F(ObjectRangeCacheTest, WrappersAndDirectAdaptersPreserveRejectedIoOutputs) {
    auto* got64 = io.allocate_array<uint64_t>(1); ASSERT_NE(got64, nullptr); *got64 = 77;
    auto* got32 = io.allocate_array<uint32_t>(1); ASSERT_NE(got32, nullptr); *got32 = 33;
    auto* version = io.allocate_array<ObjectVersion>(1); ASSERT_NE(version, nullptr); version->size = 55;
    bolt::Arena unrelated;
    ObjectVersion expected{}; expected.size = origin.bytes[0].size(); std::strcpy(expected.etag, "\"opaque-2\"");
    uint8_t byte = 0;
    EXPECT_EQ(os_get_range(&adapter, "a", 0, 1, &byte, got64), kOsBadArg);
    EXPECT_EQ(os_get_range_if_match(&adapter, "a", &expected, 0, 1, &unrelated, &byte, got32), kOsBadArg);
    EXPECT_EQ(os_head_versioned(&adapter, "a", &unrelated, version), kOsBadArg);
    EXPECT_EQ(adapter.vt->get_range_if_match(adapter.impl, "a", &expected, 0, 1, &unrelated, &byte, got32), kOsBadArg);
    EXPECT_EQ(*got64, 77u); EXPECT_EQ(*got32, 33u); EXPECT_EQ(version->size, 55u);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
}
TEST_F(ObjectRangeCacheTest, GenericWrappersRejectAliasesBeforeProviderDispatch) {
    alignas(ObjectVersion) std::array<char, sizeof(ObjectVersion)> storage{}; storage[0] = 'a';
    const auto saved = storage; uint8_t byte = 0;
    EXPECT_EQ(os_head_versioned(&origin.store, storage.data(), &io,
        reinterpret_cast<ObjectVersion*>(storage.data())), kOsBadArg);
    EXPECT_EQ(os_get_range(&adapter, storage.data(), 0, 1, &byte,
        reinterpret_cast<uint64_t*>(storage.data())), kOsBadArg);
    ObjectVersion expected{}; expected.size = 32; std::strcpy(expected.etag, "\"opaque\"");
    const auto original = expected;
    EXPECT_EQ(os_get_range_if_match(&origin.store, "a", &expected, 0, 1, &io, &byte,
        reinterpret_cast<uint32_t*>(&expected.size)), kOsBadArg);
    EXPECT_EQ(storage, saved); EXPECT_EQ(std::memcmp(&expected, &original, sizeof(expected)), 0);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(origin.ranges, 0u);
}
TEST_F(ObjectRangeCacheTest, OverflowingAddressesRefuseWithoutDereferenceOrOutputMutation) {
    uint8_t byte = 0; uint64_t got = 81;
    const auto* bad_key = reinterpret_cast<const char*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(object_range_cache_read(cache, bad_key, nullptr, 0, 1, &byte, &got), kOsBadArg);
    auto* bad_dst = reinterpret_cast<uint8_t*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(object_range_cache_read(cache, "a", nullptr, 0, 8, bad_dst, &got), kOsBadArg);
    const auto* bad_expected = reinterpret_cast<const ObjectVersion*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(object_range_cache_read(cache, "a", bad_expected, 0, 1, &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 81u); EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
}
TEST_F(ObjectRangeCacheTest, ConditionalAdapterRetainsOneMiBAndDescriptorAdmission) {
    uint8_t byte = 0; uint32_t got = 81;
    ObjectVersion expected{}; expected.size = origin.bytes[0].size(); std::strcpy(expected.etag, "\"opaque-2\"");
    EXPECT_EQ(os_get_range_if_match(&adapter, "a", &expected, 0, kPage + 1u, &io, &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 0u); EXPECT_EQ(origin.heads, 0u);
    EXPECT_EQ(adapter.vt->get_range_if_match(adapter.impl, "a", nullptr, 0, 1, &io, &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 0u); EXPECT_EQ(origin.ranges, 0u);
}
}  // namespace

namespace {
TEST_F(ObjectRangeCacheTest, StartupAliasesPreserveOriginAndResetWorkspaceDescriptors) {
    const auto saved = origin.store;
    ObjectRangeCache* unused = cache;
    EXPECT_EQ(object_range_cache_init(&unused, &origin.store, &backing, &io, 0, &origin.store), kOsBadArg);
    EXPECT_EQ(origin.store.vt, saved.vt); EXPECT_EQ(origin.store.impl, saved.impl); EXPECT_EQ(unused, cache);
    auto* output = io.allocate_array<ObjectStore>(1); ASSERT_NE(output, nullptr); *output = saved;
    EXPECT_EQ(object_range_cache_init(&unused, &origin.store, &backing, &io, 0, output), kOsBadArg);
    EXPECT_EQ(output->vt, saved.vt); EXPECT_EQ(output->impl, saved.impl); EXPECT_EQ(unused, cache);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(stats().reads, 0u);
}
}  // namespace

namespace {
TEST_F(ObjectRangeCacheTest, StartupOutputsCannotMutateBorrowedOriginVtable) {
    ObjectStoreVT borrowed = *origin.store.vt;
    origin.store.vt = &borrowed;
    std::array<uint8_t, sizeof(ObjectStoreVT)> saved{};
    std::memcpy(saved.data(), &borrowed, sizeof(borrowed));
    ObjectRangeCache* untouched = cache;
    auto* bad_adapter = reinterpret_cast<ObjectStore*>(&borrowed);
    EXPECT_EQ(object_range_cache_init(&untouched, &origin.store, &backing, &io, 0, bad_adapter), kOsBadArg);
    EXPECT_EQ(untouched, cache); EXPECT_EQ(std::memcmp(saved.data(), &borrowed, sizeof(borrowed)), 0);
    ObjectStore adapter_out{};
    auto** bad_handle = reinterpret_cast<ObjectRangeCache**>(&borrowed);
    EXPECT_EQ(object_range_cache_init(bad_handle, &origin.store, &backing, &io, 0, &adapter_out), kOsBadArg);
    EXPECT_EQ(adapter_out.vt, nullptr); EXPECT_EQ(adapter_out.impl, nullptr);
    EXPECT_EQ(std::memcmp(saved.data(), &borrowed, sizeof(borrowed)), 0);
    EXPECT_EQ(origin.heads, 0u); EXPECT_EQ(origin.ranges, 0u);
}
}  // namespace
