#include "bolt/lakehouse/object_store/pinned.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <utility>

namespace {
using namespace bolt::lakehouse;
constexpr uint32_t kRequest = 64u;
constexpr uint64_t kBodyCap = 32u << 20;

struct PinnedMemoryOrigin {
    bolt::Arena storage{{1024, 64u << 20, 64, false}};
    uint8_t* body = nullptr;
    ObjectVersion current{};
    ObjectStore store{};
    uint32_t ranges = 0, forbidden = 0, change_on_request = 0;
    uint32_t largest_request = 0, workspace_bytes = 0;
    uint64_t bytes_read = 0;
    int failure = kOsOk;
    bool short_success = false;

    void resize(uint64_t bytes) {
        ASSERT_LE(bytes, kBodyCap);
        storage.reset();
        body = bytes == 0 ? nullptr : static_cast<uint8_t*>(storage.allocate(static_cast<size_t>(bytes), 1));
        ASSERT_TRUE(body != nullptr || bytes == 0);
        current.size = bytes;
        std::strcpy(current.etag, "\"generation-1\"");
        for (uint64_t i = 0; i < bytes; ++i) body[i] = static_cast<uint8_t>(i * 131u);
    }
    static PinnedMemoryOrigin& self(void* p) { return *static_cast<PinnedMemoryOrigin*>(p); }
    static int get(void* p, const char*, bolt::Arena*, const uint8_t**, uint64_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int put(void* p, const char*, const uint8_t*, uint64_t) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int list(void* p, const char*, ObjectEntry*, uint32_t, uint32_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int remove(void* p, const char*) noexcept { ++self(p).forbidden; return kOsIoError; }
    static int head(void* p, const char*, ObjectMeta*) noexcept { ++self(p).forbidden; return kOsIoError; }
    static int version(void* p, const char*, bolt::Arena*, ObjectVersion*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int range(void* p, const char*, uint64_t, uint64_t, uint8_t*, uint64_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int conditional(void* p, const char* key, const ObjectVersion* pin, uint64_t offset,
                           uint32_t length, bolt::Arena* scratch, uint8_t* dst, uint32_t* got) noexcept {
        auto& s = self(p); ++s.ranges; *got = 0;
        if (s.ranges == s.change_on_request) std::strcpy(s.current.etag, "\"generation-2\"");
        if (s.failure != kOsOk) return s.failure;
        if (std::strcmp(key, "a") != 0 || pin->size != s.current.size ||
            std::strcmp(pin->etag, s.current.etag) != 0) return kOsPreconditionFailed;
        if (offset > s.current.size || length > s.current.size - offset) return kOsBadArg;
        const uint32_t need = s.workspace_bytes == 0 ? length : s.workspace_bytes;
        auto* response = static_cast<uint8_t*>(scratch->allocate(need, 1));
        if (response == nullptr) return kOsBufferSmall;
        const uint32_t actual = s.short_success ? length - 1u : length;
        std::memcpy(response, s.body + static_cast<size_t>(offset), actual);
        std::memcpy(dst, response, actual);
        if (length > s.largest_request) s.largest_request = length;
        s.bytes_read += actual; *got = actual;
        return kOsOk;
    }
    void initialize() {
        static const ObjectStoreVT vt{get, put, list, remove, head, put, range, version, conditional};
        store = {&vt, this};
        resize(2u * kRequest + 7u);
    }
};

class PinnedObjectStoreTest : public ::testing::Test {
protected:
    PinnedMemoryOrigin origin;
    bolt::Arena io{{kRequest, kRequest, 64, false}};
    bolt::Arena result{{1024, 64u << 20, 64, false}};
    bolt::Arena scratch{{1024, 1024, 64, false}};
    std::array<PinnedObject, 2> pins{};
    PinnedObjectStoreLimits limits{2, kBodyCap, kRequest};
    PinnedObjectStore state{};
    ObjectStore adapter{};
    int initialize(uint32_t count = 2) {
        io.reset();
        return pinned_object_store_init(&state, &origin.store, "s3://logical-bucket/table",
                                         pins.data(), count, &limits, &io, &adapter);
    }
    void SetUp() override {
        origin.initialize();
        std::strcpy(pins[0].key, "a"); pins[0].version = origin.current;
        std::strcpy(pins[1].key, "empty"); pins[1].version = origin.current; pins[1].version.size = 0;
        io.set_growth_enabled(false);
        ASSERT_EQ(initialize(), kOsOk);
    }
    void TearDown() override { EXPECT_EQ(origin.forbidden, 0u); }
    int conditional(ObjectVersion* pin, uint32_t length, uint8_t* dst, uint32_t* got,
                    uint64_t offset = 0) {
        return os_get_range_if_match(&adapter, "a", pin, offset, length, &scratch, dst, got);
    }
};

TEST_F(PinnedObjectStoreTest, HeadsCopyDirectoryWithoutOriginAndMissingIsNotFound) {
    ObjectMeta meta{}; ObjectVersion version{};
    ASSERT_EQ(os_head(&adapter, "a", &meta), kOsOk);
    EXPECT_TRUE(meta.exists); EXPECT_EQ(meta.size, pins[0].version.size);
    ASSERT_EQ(os_head_versioned(&adapter, "a", &scratch, &version), kOsOk);
    EXPECT_EQ(std::memcmp(&version, &pins[0].version, sizeof(version)), 0);
    EXPECT_EQ(os_head(&adapter, "missing", &meta), kOsNotFound); EXPECT_FALSE(meta.exists);
    EXPECT_EQ(os_head_versioned(&adapter, "missing", &scratch, &version), kOsNotFound);
    EXPECT_EQ(version.size, 0u); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, FullGetUsesExactConditionalChunksAndSeparateResettableIo) {
    const uint8_t* data = nullptr; uint64_t bytes = 0;
    auto* retained = static_cast<uint8_t*>(result.allocate(1, 1)); ASSERT_NE(retained, nullptr); *retained = 91;
    auto* scratch_byte = static_cast<uint8_t*>(scratch.allocate(1, 1)); ASSERT_NE(scratch_byte, nullptr); *scratch_byte = 77;
    ASSERT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsOk);
    ASSERT_EQ(bytes, origin.current.size); EXPECT_EQ(std::memcmp(data, origin.body, static_cast<size_t>(bytes)), 0);
    EXPECT_EQ(origin.ranges, 3u); EXPECT_EQ(origin.bytes_read, bytes); EXPECT_EQ(origin.largest_request, kRequest);
    EXPECT_EQ(*retained, 91); EXPECT_EQ(*scratch_byte, 77); EXPECT_EQ(io.total_reserved(), kRequest);
}

TEST_F(PinnedObjectStoreTest, EmptyObjectAndEmptyRangesSucceedWithoutOriginIo) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(uintptr_t{1}); uint64_t bytes = 91;
    EXPECT_EQ(os_get(&adapter, "empty", &result, &data, &bytes), kOsOk);
    EXPECT_EQ(data, nullptr); EXPECT_EQ(bytes, 0u);
    uint64_t got = 81;
    EXPECT_EQ(os_get_range(&adapter, "empty", 0, 0, nullptr, &got), kOsOk); EXPECT_EQ(got, 0u);
    EXPECT_EQ(os_get_range(&adapter, "empty", 1, 0, nullptr, &got), kOsBadArg);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, SameKeyDifferentTagAndExtentRefuseBeforeOriginIo) {
    std::array<uint8_t, kRequest> dst{}; uint32_t got = 81;
    ObjectVersion wrong = pins[0].version; ++wrong.size;
    EXPECT_EQ(conditional(&wrong, 1, dst.data(), &got), kOsPreconditionFailed); EXPECT_EQ(got, 0u);
    wrong = pins[0].version; std::strcpy(wrong.etag, "\"another-version\"");
    EXPECT_EQ(conditional(&wrong, 1, dst.data(), &got), kOsPreconditionFailed);
    std::strcpy(wrong.etag, "W/\"weak\"");
    EXPECT_EQ(conditional(&wrong, 1, dst.data(), &got), kOsBadArg);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, ChangeBeforeOrDuringReadNeverRepinsOrRetries) {
    std::strcpy(origin.current.etag, "\"changed-before-read\"");
    const uint8_t* data = origin.body; uint64_t bytes = 91;
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsPreconditionFailed);
    EXPECT_EQ(data, origin.body); EXPECT_EQ(bytes, 91u); EXPECT_EQ(origin.ranges, 1u);
    origin.current = pins[0].version; origin.change_on_request = 3;
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsPreconditionFailed);
    EXPECT_EQ(origin.ranges, 3u); EXPECT_EQ(bytes, 91u);
    ObjectVersion local{};
    ASSERT_EQ(os_head_versioned(&adapter, "a", &scratch, &local), kOsOk);
    EXPECT_STREQ(local.etag, pins[0].version.etag);
}

TEST_F(PinnedObjectStoreTest, OriginSizeChangesErrorsAndShortSuccessStayExplicit) {
    std::array<uint8_t, kRequest> dst{}; uint32_t got = 91;
    ++origin.current.size;
    EXPECT_EQ(conditional(&pins[0].version, 1, dst.data(), &got), kOsPreconditionFailed);
    origin.current = pins[0].version;
    for (const int error : {kOsNotFound, kOsCancelled, kOsIoError, kOsBufferSmall}) {
        origin.failure = error; EXPECT_EQ(conditional(&pins[0].version, 1, dst.data(), &got), error);
        EXPECT_EQ(got, 0u);
    }
    origin.failure = kOsOk; origin.short_success = true;
    EXPECT_EQ(conditional(&pins[0].version, kRequest, dst.data(), &got), kOsIoError); EXPECT_EQ(got, 0u);
    const uint8_t* data = origin.body; uint64_t bytes = 71;
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsIoError);
    EXPECT_EQ(data, origin.body); EXPECT_EQ(bytes, 71u);
}

TEST_F(PinnedObjectStoreTest, LargeMetadataBodyExceedsSixteenMiBWithoutLegacyGetCeiling) {
    constexpr uint32_t size = (17u << 20) + 17u;
    origin.resize(size); std::memset(origin.body, 'x', size);
    constexpr char prefix[] = "{\"padding\":\"";
    std::memcpy(origin.body, prefix, sizeof(prefix) - 1); std::memcpy(origin.body + size - 2u, "\"}", 2);
    pins[0].version = origin.current;
    bolt::Arena large_io{{1u << 20, 1u << 20, 64, false}};
    large_io.set_growth_enabled(false); limits.request_bytes = 1u << 20;
    ASSERT_EQ(pinned_object_store_init(&state, &origin.store, "logical", pins.data(), 2,
                                     &limits, &large_io, &adapter), kOsOk);
    const uint8_t* data = nullptr; uint64_t bytes = 0;
    ASSERT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsOk);
    EXPECT_EQ(bytes, size); EXPECT_EQ(std::memcmp(data, origin.body, size), 0);
    EXPECT_EQ(origin.ranges, 18u); EXPECT_EQ(origin.largest_request, 1u << 20);
}

TEST_F(PinnedObjectStoreTest, ExtentAndBudgetRefusalsNeverContactOrigin) {
    std::array<uint8_t, kRequest + 1u> dst{}; uint64_t got = 91;
    EXPECT_EQ(os_get_range(&adapter, "a", 0, dst.size(), dst.data(), &got), kOsBufferSmall);
    EXPECT_EQ(got, 0u);
    EXPECT_EQ(os_get_range(&adapter, "a", UINT64_MAX, 1, dst.data(), &got), kOsBadArg);
    EXPECT_EQ(os_get_range(&adapter, "a", pins[0].version.size, 1, dst.data(), &got), kOsBadArg);
    EXPECT_EQ(os_get_range(&adapter, "missing", 0, 1, dst.data(), &got), kOsNotFound);
    limits.max_get_bytes = 1; ASSERT_EQ(initialize(), kOsOk);
    const uint8_t* data = origin.body; uint64_t bytes = 77;
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsBufferSmall);
    EXPECT_EQ(bytes, 77u); EXPECT_EQ(data, origin.body); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, FrozenResultAndTransportExhaustionAreBufferSmall) {
    bolt::Arena tiny{{1, 1, 64, false}}; tiny.set_growth_enabled(false);
    const uint8_t* data = origin.body; uint64_t bytes = 77;
    EXPECT_EQ(os_get(&adapter, "a", &tiny, &data, &bytes), kOsBufferSmall);
    EXPECT_EQ(origin.ranges, 0u); EXPECT_EQ(bytes, 77u);
    origin.workspace_bytes = kRequest + 1u;
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &bytes), kOsBufferSmall);
    EXPECT_EQ(origin.ranges, 1u); EXPECT_EQ(io.total_reserved(), kRequest); EXPECT_EQ(bytes, 77u);
}

TEST_F(PinnedObjectStoreTest, MutationsAndListingAreExplicitlyUnsupported) {
    uint32_t count = 77;
    EXPECT_EQ(os_put(&adapter, "a", origin.body, 1), kOsNotImplemented);
    EXPECT_EQ(os_put_if_absent(&adapter, "new", origin.body, 1), kOsNotImplemented);
    EXPECT_EQ(os_delete(&adapter, "a"), kOsNotImplemented);
    EXPECT_EQ(os_list(&adapter, "", nullptr, 0, &count), kOsNotImplemented);
    EXPECT_EQ(count, 77u); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, InitRejectsUnsortedDuplicateNoncanonicalAndWeakPins) {
    const auto saved = pins;
    std::swap(pins[0], pins[1]); EXPECT_EQ(initialize(), kOsBadArg); pins = saved;
    pins[1] = pins[0]; EXPECT_EQ(initialize(), kOsBadArg); pins = saved;
    for (const char* key : {"", "/a", "a/", "a//b", "a/./b", "a/../b", "a\\b", "a\nb"}) {
        std::strcpy(pins[0].key, key); EXPECT_EQ(initialize(), kOsBadArg);
    }
    pins = saved; std::strcpy(pins[0].version.etag, "W/\"weak\""); EXPECT_EQ(initialize(), kOsBadArg);
    pins = saved; std::memset(pins[0].key, 'a', kOsMaxKey); EXPECT_EQ(initialize(), kOsBadArg);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, InitFailureLeavesStateAndViewUnchanged) {
    const auto saved_state = state; const auto saved_adapter = adapter;
    for (const uint32_t request : {0u, kOsPinnedMaxRequestBytes + 1u}) {
        limits.request_bytes = request; EXPECT_EQ(initialize(), kOsBadArg);
        EXPECT_EQ(std::memcmp(&state, &saved_state, sizeof(state)), 0);
        EXPECT_EQ(adapter.vt, saved_adapter.vt); EXPECT_EQ(adapter.impl, saved_adapter.impl);
    }
    limits.request_bytes = kRequest; limits.max_entries = 1; EXPECT_EQ(initialize(), kOsBadArg);
    limits.max_entries = 2; limits.max_get_bytes = 0; EXPECT_EQ(initialize(), kOsBadArg);
    limits.max_get_bytes = kOsPinnedMaxGetBytes + 1; EXPECT_EQ(initialize(), kOsBadArg);
    limits.max_get_bytes = kBodyCap; limits.request_bytes = kRequest + 1;
    EXPECT_EQ(initialize(), kOsBufferSmall);
    EXPECT_EQ(std::memcmp(&state, &saved_state, sizeof(state)), 0);
    EXPECT_EQ(adapter.vt, saved_adapter.vt); EXPECT_EQ(adapter.impl, saved_adapter.impl);
}

TEST_F(PinnedObjectStoreTest, InitRequiresFrozenEmptyExclusiveIoAndConditionalInterface) {
    io.set_growth_enabled(true); EXPECT_EQ(initialize(), kOsBadArg); io.set_growth_enabled(false);
    io.set_concurrent(true); EXPECT_EQ(initialize(), kOsBadArg); io.set_concurrent(false);
    ASSERT_NE(io.allocate(1, 1), nullptr);
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, "logical", pins.data(), 2,
                                     &limits, &io, &adapter), kOsBadArg);
    io.reset(); ObjectStoreVT legacy = *origin.store.vt; legacy.get_range_if_match = nullptr;
    ObjectStore unsupported{&legacy, &origin};
    EXPECT_EQ(pinned_object_store_init(&state, &unsupported, "logical", pins.data(), 2,
                                     &limits, &io, &adapter), kOsNotImplemented);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, DirectoryAndStateCannotBeWritableReadOutputs) {
    const auto saved_state = state; const auto saved_pins = pins;
    uint64_t got = 91;
    EXPECT_EQ(os_get_range(&adapter, "a", 0, 1, reinterpret_cast<uint8_t*>(&state), &got), kOsBadArg);
    EXPECT_EQ(os_get_range(&adapter, "a", 0, 1, reinterpret_cast<uint8_t*>(pins.data()), &got), kOsBadArg);
    EXPECT_EQ(os_head(&adapter, "a", reinterpret_cast<ObjectMeta*>(&state)), kOsBadArg);
    EXPECT_EQ(os_head_versioned(&adapter, "a", &scratch, &pins[0].version), kOsBadArg);
    EXPECT_EQ(got, 91u); EXPECT_EQ(std::memcmp(&state, &saved_state, sizeof(state)), 0);
    EXPECT_EQ(std::memcmp(pins.data(), saved_pins.data(), sizeof(pins)), 0); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, InputOutputAndIoAliasesPreserveBytesAndDoNotReset) {
    alignas(uint64_t) std::array<char, 32> key{}; std::strcpy(key.data(), "a"); const auto saved_key = key;
    uint8_t dst = 0; uint64_t got = 91;
    EXPECT_EQ(os_get_range(&adapter, key.data(), 0, 1, reinterpret_cast<uint8_t*>(key.data()), &got), kOsBadArg);
    EXPECT_EQ(os_get_range(&adapter, key.data(), 0, 1, &dst, reinterpret_cast<uint64_t*>(key.data())), kOsBadArg);
    EXPECT_EQ(key, saved_key); EXPECT_EQ(got, 91u);
    auto* bytes = static_cast<uint8_t*>(io.allocate(32, 1)); ASSERT_NE(bytes, nullptr);
    std::memset(bytes, 0xa5, 32); const auto used = io.total_allocated();
    EXPECT_EQ(os_get_range(&adapter, "a", 0, 1, bytes, &got), kOsBadArg);
    EXPECT_EQ(bytes[0], 0xa5); EXPECT_EQ(io.total_allocated(), used);
    std::memcpy(bytes, "a", 2);
    EXPECT_EQ(os_get_range(&adapter, reinterpret_cast<const char*>(bytes), 0, 1, &dst, &got), kOsBadArg);
    EXPECT_EQ(bytes[0], 'a'); EXPECT_EQ(io.total_allocated(), used); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, ExpectedAndLengthAliasesAreRejectedWithoutPublication) {
    ObjectVersion expected = pins[0].version; const auto saved = expected; uint32_t got = 91;
    EXPECT_EQ(conditional(&expected, 1, reinterpret_cast<uint8_t*>(&expected), &got), kOsBadArg);
    uint8_t byte = 0;
    EXPECT_EQ(conditional(&expected, 1, &byte, reinterpret_cast<uint32_t*>(&expected.size)), kOsBadArg);
    EXPECT_EQ(std::memcmp(&expected, &saved, sizeof(expected)), 0); EXPECT_EQ(got, 91u);
    auto* count = io.allocate_array<uint32_t>(1); ASSERT_NE(count, nullptr); *count = 77;
    EXPECT_EQ(conditional(&expected, 1, &byte, count), kOsBadArg); EXPECT_EQ(*count, 77u);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, FullGetRejectsResultStateIoAndOutputAliasesBeforeAllocation) {
    const uint8_t* data = origin.body; uint64_t bytes = 91;
    EXPECT_EQ(os_get(&adapter, "a", &io, &data, &bytes), kOsBadArg);
    auto* output = result.allocate_array<const uint8_t*>(1); ASSERT_NE(output, nullptr); *output = origin.body;
    const auto used = result.total_allocated();
    EXPECT_EQ(os_get(&adapter, "a", &result, output, &bytes), kOsBadArg);
    EXPECT_EQ(*output, origin.body); EXPECT_EQ(bytes, 91u); EXPECT_EQ(result.total_allocated(), used);
    EXPECT_EQ(os_get(&adapter, "a", &result, &data, &state.limits.max_get_bytes), kOsBadArg);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, StartupAndViewAliasesPreserveInputsAndOutputs) {
    const auto saved = origin.store; const auto saved_pins = pins; const auto saved_state = state;
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, "logical", pins.data(), 2,
                                     &limits, &io, &origin.store), kOsBadArg);
    EXPECT_EQ(origin.store.vt, saved.vt); EXPECT_EQ(origin.store.impl, saved.impl);
    EXPECT_EQ(pinned_object_store_init(reinterpret_cast<PinnedObjectStore*>(pins.data()), &origin.store,
                 "logical", pins.data(), 2, &limits, &io, &adapter), kOsBadArg);
    EXPECT_EQ(std::memcmp(pins.data(), saved_pins.data(), sizeof(pins)), 0);
    EXPECT_EQ(pinned_object_store_view(&state, reinterpret_cast<ObjectStore*>(&state)), kOsBadArg);
    EXPECT_EQ(std::memcmp(&state, &saved_state, sizeof(state)), 0);
    ObjectStore view{}; ASSERT_EQ(pinned_object_store_view(&state, &view), kOsOk);
    EXPECT_EQ(view.vt, adapter.vt); EXPECT_EQ(view.impl, adapter.impl); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, ReadOnlyInputsMayBorrowTheImmutableDirectory) {
    uint8_t byte = 0; uint32_t got = 0;
    ASSERT_EQ(os_get_range_if_match(&adapter, pins[0].key, &pins[0].version, 0, 1,
                                   &scratch, &byte, &got), kOsOk);
    EXPECT_EQ(got, 1u); EXPECT_EQ(byte, origin.body[0]);
}

TEST_F(PinnedObjectStoreTest, EmptyDirectoryAndBoundedNamespaceAreAdmittedWithoutIo) {
    ASSERT_EQ(pinned_object_store_init(&state, &origin.store, "logical", nullptr, 0,
                                     &limits, &io, &adapter), kOsOk);
    ObjectMeta meta{};
    EXPECT_EQ(os_head(&adapter, "a", &meta), kOsNotFound);
    std::array<char, kOsMaxKey> unterminated{}; unterminated.fill('x');
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, unterminated.data(), nullptr, 0,
                                     &limits, &io, &adapter), kOsBadArg);
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, "", nullptr, 0,
                                     &limits, &io, &adapter), kOsBadArg);
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, "logical", pins.data(),
             kOsPinnedMaxEntries + 1u, &limits, &io, &adapter), kOsBadArg);
    EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, ArenaObjectAndOriginVtableAliasesAreRejectedBeforeUse) {
    const auto saved = state; const auto vt = *origin.store.vt;
    EXPECT_EQ(pinned_object_store_init(&state, &origin.store, "logical", pins.data(), 2,
                 &limits, reinterpret_cast<bolt::Arena*>(&state), &adapter), kOsBadArg);
    EXPECT_EQ(std::memcmp(&state, &saved, sizeof(state)), 0);
    ObjectStoreVT writable_vt = vt; ObjectStore borrowed{&writable_vt, &origin};
    EXPECT_EQ(pinned_object_store_init(&state, &borrowed, "logical", pins.data(), 2,
                 &limits, &io, reinterpret_cast<ObjectStore*>(&writable_vt)), kOsBadArg);
    EXPECT_EQ(std::memcmp(&writable_vt, &vt, sizeof(vt)), 0);
    uint8_t byte = 0; uint32_t got = 91;
    EXPECT_EQ(adapter.vt->get_range_if_match(adapter.impl, "a", &pins[0].version, 0, 1,
                 reinterpret_cast<bolt::Arena*>(&state), &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 91u); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, OverflowingSpansRefuseWithoutWritesOrOriginIo) {
    uint8_t byte = 0; uint64_t got = 91;
    const auto* bad_key = reinterpret_cast<const char*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(os_get_range(&adapter, bad_key, 0, 1, &byte, &got), kOsBadArg);
    auto* bad_dst = reinterpret_cast<uint8_t*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(os_get_range(&adapter, "a", 0, 8, bad_dst, &got), kOsBadArg);
    uint32_t got32 = 71;
    const auto* bad_version = reinterpret_cast<const ObjectVersion*>(UINTPTR_MAX - 3u);
    EXPECT_EQ(os_get_range_if_match(&adapter, "a", bad_version, 0, 1, &scratch, &byte, &got32), kOsBadArg);
    EXPECT_EQ(got, 91u); EXPECT_EQ(got32, 71u); EXPECT_EQ(origin.ranges, 0u);
}

TEST_F(PinnedObjectStoreTest, WorkspaceObjectAliasesRefuseBeforeAnyProposedArenaInspection) {
    // External copies keep earlier input/output admission from masking the
    // workspace check. Directory/vtable/backing storage is not an Arena.
    const char key[] = "a";
    const ObjectVersion expected = pins[0].version;
    const auto saved_state = state; const auto saved_pins = pins;
    const auto saved_origin_vt = *origin.store.vt; const auto saved_adapter_vt = *adapter.vt;
    auto* backing = static_cast<uint8_t*>(io.allocate(kRequest, 1)); ASSERT_NE(backing, nullptr);
    std::memset(backing, 0xa5, kRequest);
    const auto used = io.total_allocated(); const auto reserved = io.total_reserved();
    const std::array<bolt::Arena*, 5> aliases{{
        reinterpret_cast<bolt::Arena*>(pins.data()),
        reinterpret_cast<bolt::Arena*>(const_cast<ObjectStoreVT*>(origin.store.vt)),
        reinterpret_cast<bolt::Arena*>(const_cast<ObjectStoreVT*>(adapter.vt)),
        reinterpret_cast<bolt::Arena*>(backing),
        reinterpret_cast<bolt::Arena*>(reinterpret_cast<uint8_t*>(&io) + 1u)}};
    for (auto* alias : aliases) {
        const uint8_t* data = origin.body; uint64_t bytes = 91;
        EXPECT_EQ(adapter.vt->get_object(adapter.impl, key, alias, &data, &bytes), kOsBadArg);
        EXPECT_EQ(data, origin.body); EXPECT_EQ(bytes, 91u);
        uint8_t dst = 0x5a; uint32_t got = 71;
        EXPECT_EQ(adapter.vt->get_range_if_match(adapter.impl, key, &expected, 0, 1,
                                                alias, &dst, &got), kOsBadArg);
        EXPECT_EQ(dst, 0x5a); EXPECT_EQ(got, 71u);
        ObjectVersion out = expected;
        EXPECT_EQ(adapter.vt->head_versioned(adapter.impl, key, alias, &out), kOsBadArg);
        EXPECT_EQ(std::memcmp(&out, &expected, sizeof(out)), 0);
        EXPECT_EQ(io.total_allocated(), used); EXPECT_EQ(io.total_reserved(), reserved);
    }
    const uint8_t* data = origin.body; uint64_t bytes = 81;
    EXPECT_EQ(os_get(&adapter, key, aliases[0], &data, &bytes), kOsBadArg);
    EXPECT_EQ(data, origin.body); EXPECT_EQ(bytes, 81u);
    EXPECT_EQ(std::memcmp(&state, &saved_state, sizeof(state)), 0);
    EXPECT_EQ(std::memcmp(pins.data(), saved_pins.data(), sizeof(pins)), 0);
    EXPECT_EQ(std::memcmp(origin.store.vt, &saved_origin_vt, sizeof(saved_origin_vt)), 0);
    EXPECT_EQ(std::memcmp(adapter.vt, &saved_adapter_vt, sizeof(saved_adapter_vt)), 0);
    for (uint32_t i = 0; i < kRequest; ++i) EXPECT_EQ(backing[i], 0xa5);
    EXPECT_EQ(origin.ranges, 0u);
}
}  // namespace
