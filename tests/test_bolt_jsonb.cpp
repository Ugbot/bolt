// test_bolt_jsonb.cpp — Layer 1.4 of plan `this-was-a-freach-hashed-crab.md`.
//
// Covers: scalar round-trips, flat objects, sorted-key invariant, nested
// objects, arrays, missing-key path lookup, malformed-buffer rejection,
// stride-offset path, binary-search correctness across {1, 2, 32, 1024}.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/doc/bolt_jsonb.h"

using namespace bolt::doc;

namespace {

bolt::Arena make_arena() {
    bolt::ArenaConfig cfg{};
    cfg.initial_block_size = 256u * 1024u;
    cfg.max_block_size     = 4u * 1024u * 1024u;
    cfg.alignment          = 64u;
    return bolt::Arena(cfg);
}

bool lookup_key(JsonbView v, const char* key, JsonbValue* out) {
    const int32_t klen = static_cast<int32_t>(std::strlen(key));
    uint8_t kind = 0;
    const char* str = key;
    int32_t idx = 0;
    return path_lookup(v, &kind, &str, &klen, &idx, 1, out);
}

bool lookup_index(JsonbView v, int32_t i, JsonbValue* out) {
    uint8_t kind = 1;
    int32_t idx = i;
    int32_t klen = 0;
    const char* str = nullptr;
    return path_lookup(v, &kind, &str, &klen, &idx, 1, out);
}

}  // namespace

TEST(BoltJsonb, EncodeDecodeRoundTripScalars) {
    bolt::Arena arena = make_arena();
    {
        JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64));
        ASSERT_TRUE(jsonb_emit_string(&b, "hello", 5));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));
        JsonbValue out{};
        ASSERT_TRUE(path_lookup(v, nullptr, nullptr, nullptr, nullptr, 0, &out));
        // Depth=0 returns the root container; root is a scalar wrapper.
        // Use array_lookup at index 0.
        ASSERT_TRUE(lookup_index(v, 0, &out));
        EXPECT_EQ(out.tag, JsonbTag::String);
        ASSERT_EQ(out.len, 5);
        EXPECT_EQ(0, std::memcmp(out.bytes, "hello", 5));
    }
    {
        JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64));
        ASSERT_TRUE(jsonb_emit_int64(&b, -42));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));
        JsonbValue out{};
        ASSERT_TRUE(lookup_index(v, 0, &out));
        EXPECT_EQ(out.tag, JsonbTag::Int64);
        EXPECT_EQ(out.i64, -42);
    }
    {
        JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64));
        ASSERT_TRUE(jsonb_emit_float64(&b, 3.14159));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));
        JsonbValue out{};
        ASSERT_TRUE(lookup_index(v, 0, &out));
        EXPECT_EQ(out.tag, JsonbTag::Float64);
        EXPECT_DOUBLE_EQ(out.f64, 3.14159);
    }
    for (bool bv : {true, false}) {
        JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64));
        ASSERT_TRUE(jsonb_emit_bool(&b, bv));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));
        JsonbValue out{};
        ASSERT_TRUE(lookup_index(v, 0, &out));
        EXPECT_EQ(out.tag, bv ? JsonbTag::BoolTrue : JsonbTag::BoolFalse);
    }
    {
        JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64));
        ASSERT_TRUE(jsonb_emit_null(&b));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));
        JsonbValue out{};
        ASSERT_TRUE(lookup_index(v, 0, &out));
        EXPECT_EQ(out.tag, JsonbTag::Null);
    }
}

TEST(BoltJsonb, EncodeDecodeFlatObject) {
    bolt::Arena arena = make_arena();
    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 256));
    ASSERT_TRUE(jsonb_begin_object(&b));
    // Insert deliberately out of canonical order.
    ASSERT_TRUE(jsonb_emit_key(&b, "gamma", 5));
    ASSERT_TRUE(jsonb_emit_float64(&b, 3.14));
    ASSERT_TRUE(jsonb_emit_key(&b, "alpha", 5));
    ASSERT_TRUE(jsonb_emit_int64(&b, 1));
    ASSERT_TRUE(jsonb_emit_key(&b, "beta", 4));
    ASSERT_TRUE(jsonb_emit_string(&b, "two", 3));
    ASSERT_TRUE(jsonb_end_object(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    JsonbValue out{};
    ASSERT_TRUE(lookup_key(v, "alpha", &out));
    EXPECT_EQ(out.tag, JsonbTag::Int64);
    EXPECT_EQ(out.i64, 1);

    ASSERT_TRUE(lookup_key(v, "beta", &out));
    EXPECT_EQ(out.tag, JsonbTag::String);
    ASSERT_EQ(out.len, 3);
    EXPECT_EQ(0, std::memcmp(out.bytes, "two", 3));

    ASSERT_TRUE(lookup_key(v, "gamma", &out));
    EXPECT_EQ(out.tag, JsonbTag::Float64);
    EXPECT_DOUBLE_EQ(out.f64, 3.14);
}

TEST(BoltJsonb, SortedKeyInvariant) {
    bolt::Arena arena = make_arena();
    std::vector<std::string> keys = {"zebra", "ant", "monkey", "fox", "bear",
                                     "owl", "kangaroo", "ibis", "cat", "dog"};
    {
        std::mt19937 rng(0xC0DEF00D);
        std::shuffle(keys.begin(), keys.end(), rng);
    }

    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 1024));
    ASSERT_TRUE(jsonb_begin_object(&b));
    for (size_t i = 0; i < keys.size(); ++i) {
        ASSERT_TRUE(jsonb_emit_key(&b, keys[i].data(),
                                   static_cast<int32_t>(keys[i].size())));
        ASSERT_TRUE(jsonb_emit_int64(&b, static_cast<int64_t>(i)));
    }
    ASSERT_TRUE(jsonb_end_object(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    // Decode header, walk keys in stored order, assert canonical sort.
    ASSERT_GE(v.bytes, 12);
    uint32_t hdr;
    std::memcpy(&hdr, v.data, 4);
    ASSERT_TRUE(hdr_is_object(hdr));
    const int32_t count = static_cast<int32_t>(hdr_count(hdr));
    ASSERT_EQ(count, static_cast<int32_t>(keys.size()));

    std::vector<std::string> expected = keys;
    std::sort(expected.begin(), expected.end(),
              [](const std::string& a, const std::string& b) {
                  if (a.size() != b.size()) return a.size() < b.size();
                  return a < b;
              });

    // Walk: keys' Jentry table starts at offset 4; payloads start after both
    // tables.  Use array_lookup-style resolve via path_lookup(key=...).
    for (size_t i = 0; i < expected.size(); ++i) {
        JsonbValue out{};
        ASSERT_TRUE(lookup_key(v, expected[i].c_str(), &out));
        EXPECT_EQ(out.tag, JsonbTag::Int64);
    }
    // Also verify the keys are physically stored in sorted order by walking
    // keys via the in-place buffer.
    const uint8_t* keys_payload = v.data + 12 + count * 8;
    int32_t walk_off = 0;
    for (int32_t i = 0; i < count; ++i) {
        uint32_t je;
        std::memcpy(&je, v.data + 12 + i * 4, 4);
        ASSERT_FALSE(je_is_offset(je));  // none of these reach stride at small N
        const int32_t klen = static_cast<int32_t>(je_value(je));
        const std::string got(reinterpret_cast<const char*>(keys_payload + walk_off),
                              static_cast<size_t>(klen));
        EXPECT_EQ(got, expected[static_cast<size_t>(i)]);
        walk_off += klen;
    }
}

TEST(BoltJsonb, EncodeDecodeNestedObject) {
    bolt::Arena arena = make_arena();
    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 512));
    // { "a" : { "b" : { "c" : 99 } } }
    ASSERT_TRUE(jsonb_begin_object(&b));
    ASSERT_TRUE(jsonb_emit_key(&b, "a", 1));
    ASSERT_TRUE(jsonb_begin_object(&b));
    ASSERT_TRUE(jsonb_emit_key(&b, "b", 1));
    ASSERT_TRUE(jsonb_begin_object(&b));
    ASSERT_TRUE(jsonb_emit_key(&b, "c", 1));
    ASSERT_TRUE(jsonb_emit_int64(&b, 99));
    ASSERT_TRUE(jsonb_end_object(&b));
    ASSERT_TRUE(jsonb_end_object(&b));
    ASSERT_TRUE(jsonb_end_object(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    const uint8_t kinds[3] = {0, 0, 0};
    const char* strs[3] = {"a", "b", "c"};
    const int32_t lens[3] = {1, 1, 1};
    const int32_t idxs[3] = {0, 0, 0};
    JsonbValue out{};
    ASSERT_TRUE(path_lookup(v, kinds, strs, lens, idxs, 3, &out));
    EXPECT_EQ(out.tag, JsonbTag::Int64);
    EXPECT_EQ(out.i64, 99);
}

TEST(BoltJsonb, EncodeDecodeArray) {
    bolt::Arena arena = make_arena();
    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 256));
    ASSERT_TRUE(jsonb_begin_array(&b));
    ASSERT_TRUE(jsonb_emit_int64(&b, 1));
    ASSERT_TRUE(jsonb_emit_string(&b, "two", 3));
    ASSERT_TRUE(jsonb_emit_float64(&b, 3.0));
    ASSERT_TRUE(jsonb_emit_null(&b));
    ASSERT_TRUE(jsonb_end_array(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    JsonbValue out{};
    ASSERT_TRUE(lookup_index(v, 0, &out));
    EXPECT_EQ(out.tag, JsonbTag::Int64); EXPECT_EQ(out.i64, 1);
    ASSERT_TRUE(lookup_index(v, 1, &out));
    EXPECT_EQ(out.tag, JsonbTag::String); ASSERT_EQ(out.len, 3);
    EXPECT_EQ(0, std::memcmp(out.bytes, "two", 3));
    ASSERT_TRUE(lookup_index(v, 2, &out));
    EXPECT_EQ(out.tag, JsonbTag::Float64); EXPECT_DOUBLE_EQ(out.f64, 3.0);
    ASSERT_TRUE(lookup_index(v, 3, &out));
    EXPECT_EQ(out.tag, JsonbTag::Null);
}

TEST(BoltJsonb, PathLookupMissingKeyReturnsNull) {
    bolt::Arena arena = make_arena();
    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 128));
    ASSERT_TRUE(jsonb_begin_object(&b));
    ASSERT_TRUE(jsonb_emit_key(&b, "exists", 6));
    ASSERT_TRUE(jsonb_emit_int64(&b, 1));
    ASSERT_TRUE(jsonb_end_object(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    JsonbValue out{};
    ASSERT_TRUE(lookup_key(v, "missing", &out));
    EXPECT_EQ(out.tag, JsonbTag::Null);
}

TEST(BoltJsonb, MalformedBufferRejected) {
    // Bad header — declares 1000 children but buffer is tiny.
    uint8_t buf[16] = {0};
    uint32_t hdr = 1000u | kHdrFlagObject;
    uint32_t kb = 9999;
    uint32_t vb = 9999;
    std::memcpy(buf + 0, &hdr, 4);
    std::memcpy(buf + 4, &kb, 4);
    std::memcpy(buf + 8, &vb, 4);
    JsonbView v{buf, 16};
    JsonbValue out{};
    EXPECT_FALSE(lookup_key(v, "anything", &out));

    // Zero-length buffer.
    JsonbView empty{nullptr, 0};
    EXPECT_FALSE(lookup_key(empty, "k", &out));

    // Sub-header buffer.
    uint8_t tiny[8] = {0};
    JsonbView t{tiny, 8};
    EXPECT_FALSE(lookup_key(t, "k", &out));
}

TEST(BoltJsonb, StrideOffsetsHandledCorrectly) {
    bolt::Arena arena = make_arena();
    JsonbBuilder b{}; ASSERT_TRUE(jsonb_builder_init(&b, &arena, 8192));
    ASSERT_TRUE(jsonb_begin_object(&b));
    char keybuf[32];
    for (int i = 0; i < 100; ++i) {
        const int n = std::snprintf(keybuf, sizeof(keybuf), "key_%03d", i);
        ASSERT_TRUE(jsonb_emit_key(&b, keybuf, n));
        ASSERT_TRUE(jsonb_emit_int64(&b, i));
    }
    ASSERT_TRUE(jsonb_end_object(&b));
    JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

    // Lookup mid-range key after sort.
    JsonbValue out{};
    ASSERT_TRUE(lookup_key(v, "key_050", &out));
    EXPECT_EQ(out.tag, JsonbTag::Int64);
    EXPECT_EQ(out.i64, 50);
    ASSERT_TRUE(lookup_key(v, "key_000", &out));
    EXPECT_EQ(out.i64, 0);
    ASSERT_TRUE(lookup_key(v, "key_099", &out));
    EXPECT_EQ(out.i64, 99);
}

TEST(BoltJsonb, BinarySearchCorrectnessOnObjectsOf1_2_32_1024Keys) {
    bolt::Arena arena = make_arena();
    for (int n : {1, 2, 32, 1024}) {
        JsonbBuilder b{};
        ASSERT_TRUE(jsonb_builder_init(&b, &arena, 64 * 1024));
        ASSERT_TRUE(jsonb_begin_object(&b));
        std::vector<std::string> keys;
        keys.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            char kb[32];
            const int kl = std::snprintf(kb, sizeof(kb), "k%05d", i);
            keys.emplace_back(kb, static_cast<size_t>(kl));
        }
        std::mt19937 rng(static_cast<uint32_t>(n) * 31u);
        std::vector<int> order(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        for (int i : order) {
            ASSERT_TRUE(jsonb_emit_key(&b, keys[i].data(),
                                       static_cast<int32_t>(keys[i].size())));
            ASSERT_TRUE(jsonb_emit_int64(&b, i));
        }
        ASSERT_TRUE(jsonb_end_object(&b));
        JsonbView v{}; ASSERT_TRUE(jsonb_finish(&b, &v));

        JsonbValue out{};
        ASSERT_TRUE(lookup_key(v, keys.front().c_str(), &out));
        EXPECT_EQ(out.i64, 0);
        ASSERT_TRUE(lookup_key(v, keys[static_cast<size_t>(n / 2)].c_str(), &out));
        EXPECT_EQ(out.i64, n / 2);
        ASSERT_TRUE(lookup_key(v, keys.back().c_str(), &out));
        EXPECT_EQ(out.i64, n - 1);
        // Missing.
        ASSERT_TRUE(lookup_key(v, "definitely_not_there", &out));
        EXPECT_EQ(out.tag, JsonbTag::Null);
    }
}
