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

// ---------------------------------------------------------------------------
// Sub-wave 3.B.1 — nested-sibling regression coverage.
//
// Pre-fix, JsonbBuilder shared one global `children[]` array across all open
// frames. When two siblings of an outer frame straddled a nested container —
// e.g. {"items":[...], "meta":{...}} — the inner frame's children polluted
// the global array, so the outer frame's child run was no longer contiguous
// in `children[first..first+count)`. encode_frame walked the wrong slots and
// only one of the two top-level keys round-tripped.
//
// The fix truncates `children_size` back to `first` after encode_frame
// consumes a frame's children, restoring the contiguous-walk invariant for
// the outer frame.
// ---------------------------------------------------------------------------

TEST(BoltJsonb, NestedSiblingsRoundTrip) {
    // Pattern: {"items":[{"id":7}], "meta":{"v":1}}
    // Two top-level keys with a nested container between them.
    bolt::Arena arena = make_arena();
    JsonbBuilder b{};
    ASSERT_TRUE(jsonb_builder_init(&b, &arena, 4096));

    ASSERT_TRUE(jsonb_begin_object(&b));
        ASSERT_TRUE(jsonb_emit_key(&b, "items", 5));
        ASSERT_TRUE(jsonb_begin_array(&b));
            ASSERT_TRUE(jsonb_begin_object(&b));
                ASSERT_TRUE(jsonb_emit_key(&b, "id", 2));
                ASSERT_TRUE(jsonb_emit_int64(&b, 7));
            ASSERT_TRUE(jsonb_end_object(&b));
        ASSERT_TRUE(jsonb_end_array(&b));
        ASSERT_TRUE(jsonb_emit_key(&b, "meta", 4));
        ASSERT_TRUE(jsonb_begin_object(&b));
            ASSERT_TRUE(jsonb_emit_key(&b, "v", 1));
            ASSERT_TRUE(jsonb_emit_int64(&b, 1));
        ASSERT_TRUE(jsonb_end_object(&b));
    ASSERT_TRUE(jsonb_end_object(&b));

    JsonbView view{};
    ASSERT_TRUE(jsonb_finish(&b, &view));

    // Both top-level keys must resolve.
    {
        JsonbValue v{};
        ASSERT_TRUE(lookup_key(view, "items", &v));
        EXPECT_EQ(v.tag, JsonbTag::Array);
    }
    {
        JsonbValue v{};
        ASSERT_TRUE(lookup_key(view, "meta", &v));
        EXPECT_EQ(v.tag, JsonbTag::Object);  // pre-fix: this fails
    }

    // Deep path through the array element should also resolve.
    {
        const uint8_t kinds[3] = { 0, 1 /* array idx */, 0 };
        const char*   strs[3]  = { "items", nullptr, "id" };
        const int32_t lens[3]  = { 5, 0, 2 };
        const int32_t idxs[3]  = { 0, 0, 0 };
        JsonbValue v{};
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 3, &v));
        EXPECT_EQ(v.tag, JsonbTag::Int64);
        EXPECT_EQ(v.i64, 7);
    }

    // Deep path through the meta object should also resolve.
    {
        const uint8_t kinds[2] = { 0, 0 };
        const char*   strs[2]  = { "meta", "v" };
        const int32_t lens[2]  = { 4, 1 };
        const int32_t idxs[2]  = { 0, 0 };
        JsonbValue v{};
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 2, &v));
        EXPECT_EQ(v.tag, JsonbTag::Int64);
        EXPECT_EQ(v.i64, 1);
    }
}

TEST(BoltJsonb, MultipleSiblingsAcrossSiblingNests) {
    // Three top-level keys of different shapes interleaved with nested
    // containers. Stresses the children-array contiguity invariant from
    // multiple angles: scalar between two nested siblings, then array of
    // objects, then a deep object.
    bolt::Arena arena = make_arena();
    JsonbBuilder b{};
    ASSERT_TRUE(jsonb_builder_init(&b, &arena, 4096));

    ASSERT_TRUE(jsonb_begin_object(&b));
        // Key 1: scalar
        ASSERT_TRUE(jsonb_emit_key(&b, "alpha", 5));
        ASSERT_TRUE(jsonb_emit_int64(&b, 42));

        // Key 2: array of objects (nested 2 deep)
        ASSERT_TRUE(jsonb_emit_key(&b, "beta", 4));
        ASSERT_TRUE(jsonb_begin_array(&b));
            ASSERT_TRUE(jsonb_begin_object(&b));
                ASSERT_TRUE(jsonb_emit_key(&b, "x", 1));
                ASSERT_TRUE(jsonb_emit_int64(&b, 1));
            ASSERT_TRUE(jsonb_end_object(&b));
            ASSERT_TRUE(jsonb_begin_object(&b));
                ASSERT_TRUE(jsonb_emit_key(&b, "x", 1));
                ASSERT_TRUE(jsonb_emit_int64(&b, 2));
            ASSERT_TRUE(jsonb_end_object(&b));
        ASSERT_TRUE(jsonb_end_array(&b));

        // Key 3: another scalar — sandwiched by nested siblings.
        ASSERT_TRUE(jsonb_emit_key(&b, "gamma", 5));
        ASSERT_TRUE(jsonb_emit_string(&b, "hello", 5));

        // Key 4: object with nested array
        ASSERT_TRUE(jsonb_emit_key(&b, "delta", 5));
        ASSERT_TRUE(jsonb_begin_object(&b));
            ASSERT_TRUE(jsonb_emit_key(&b, "list", 4));
            ASSERT_TRUE(jsonb_begin_array(&b));
                ASSERT_TRUE(jsonb_emit_int64(&b, 100));
                ASSERT_TRUE(jsonb_emit_int64(&b, 200));
                ASSERT_TRUE(jsonb_emit_int64(&b, 300));
            ASSERT_TRUE(jsonb_end_array(&b));
        ASSERT_TRUE(jsonb_end_object(&b));

        // Key 5: trailing scalar after the last nested sibling.
        ASSERT_TRUE(jsonb_emit_key(&b, "epsilon", 7));
        ASSERT_TRUE(jsonb_emit_bool(&b, true));
    ASSERT_TRUE(jsonb_end_object(&b));

    JsonbView view{};
    ASSERT_TRUE(jsonb_finish(&b, &view));

    JsonbValue v{};

    ASSERT_TRUE(lookup_key(view, "alpha", &v));
    EXPECT_EQ(v.tag, JsonbTag::Int64);
    EXPECT_EQ(v.i64, 42);

    ASSERT_TRUE(lookup_key(view, "beta", &v));
    EXPECT_EQ(v.tag, JsonbTag::Array);

    ASSERT_TRUE(lookup_key(view, "gamma", &v));
    EXPECT_EQ(v.tag, JsonbTag::String);
    EXPECT_EQ(v.len, 5);
    EXPECT_EQ(0, std::memcmp(v.bytes, "hello", 5));

    ASSERT_TRUE(lookup_key(view, "delta", &v));
    EXPECT_EQ(v.tag, JsonbTag::Object);

    ASSERT_TRUE(lookup_key(view, "epsilon", &v));
    EXPECT_EQ(v.tag, JsonbTag::BoolTrue);

    // Drill into beta[1].x — must be 2.
    {
        const uint8_t kinds[3] = { 0, 1, 0 };
        const char*   strs[3]  = { "beta", nullptr, "x" };
        const int32_t lens[3]  = { 4, 0, 1 };
        const int32_t idxs[3]  = { 0, 1, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 3, &v));
        EXPECT_EQ(v.tag, JsonbTag::Int64);
        EXPECT_EQ(v.i64, 2);
    }

    // Drill into delta.list[2] — must be 300.
    {
        const uint8_t kinds[3] = { 0, 0, 1 };
        const char*   strs[3]  = { "delta", "list", nullptr };
        const int32_t lens[3]  = { 5, 4, 0 };
        const int32_t idxs[3]  = { 0, 0, 2 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 3, &v));
        EXPECT_EQ(v.tag, JsonbTag::Int64);
        EXPECT_EQ(v.i64, 300);
    }
}

TEST(BoltJsonb, DeeplyNestedSiblings) {
    // Five-level nesting, with siblings at every level. Each level is an
    // object with two keys: a scalar "n" carrying the depth, and a
    // "child" that opens the next level. The final level closes with a
    // pair of scalar siblings to confirm contiguity at the leaf depth too.
    bolt::Arena arena = make_arena();
    JsonbBuilder b{};
    ASSERT_TRUE(jsonb_builder_init(&b, &arena, 4096));

    ASSERT_TRUE(jsonb_begin_object(&b));
        ASSERT_TRUE(jsonb_emit_key(&b, "n", 1));
        ASSERT_TRUE(jsonb_emit_int64(&b, 0));
        ASSERT_TRUE(jsonb_emit_key(&b, "child", 5));
        ASSERT_TRUE(jsonb_begin_object(&b));
            ASSERT_TRUE(jsonb_emit_key(&b, "n", 1));
            ASSERT_TRUE(jsonb_emit_int64(&b, 1));
            ASSERT_TRUE(jsonb_emit_key(&b, "child", 5));
            ASSERT_TRUE(jsonb_begin_object(&b));
                ASSERT_TRUE(jsonb_emit_key(&b, "n", 1));
                ASSERT_TRUE(jsonb_emit_int64(&b, 2));
                ASSERT_TRUE(jsonb_emit_key(&b, "child", 5));
                ASSERT_TRUE(jsonb_begin_object(&b));
                    ASSERT_TRUE(jsonb_emit_key(&b, "n", 1));
                    ASSERT_TRUE(jsonb_emit_int64(&b, 3));
                    ASSERT_TRUE(jsonb_emit_key(&b, "child", 5));
                    ASSERT_TRUE(jsonb_begin_object(&b));
                        ASSERT_TRUE(jsonb_emit_key(&b, "n", 1));
                        ASSERT_TRUE(jsonb_emit_int64(&b, 4));
                        ASSERT_TRUE(jsonb_emit_key(&b, "leaf_a", 6));
                        ASSERT_TRUE(jsonb_emit_int64(&b, 1000));
                        ASSERT_TRUE(jsonb_emit_key(&b, "leaf_b", 6));
                        ASSERT_TRUE(jsonb_emit_int64(&b, 2000));
                    ASSERT_TRUE(jsonb_end_object(&b));
                ASSERT_TRUE(jsonb_end_object(&b));
            ASSERT_TRUE(jsonb_end_object(&b));
        ASSERT_TRUE(jsonb_end_object(&b));
    ASSERT_TRUE(jsonb_end_object(&b));

    JsonbView view{};
    ASSERT_TRUE(jsonb_finish(&b, &view));

    JsonbValue v{};

    // n at root.
    ASSERT_TRUE(lookup_key(view, "n", &v));
    EXPECT_EQ(v.tag, JsonbTag::Int64);
    EXPECT_EQ(v.i64, 0);

    // Walk down the chain and verify every "n" along the way.
    {
        const uint8_t kinds[2] = { 0, 0 };
        const char*   strs[2]  = { "child", "n" };
        const int32_t lens[2]  = { 5, 1 };
        const int32_t idxs[2]  = { 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 2, &v));
        EXPECT_EQ(v.i64, 1);
    }
    {
        const uint8_t kinds[3] = { 0, 0, 0 };
        const char*   strs[3]  = { "child", "child", "n" };
        const int32_t lens[3]  = { 5, 5, 1 };
        const int32_t idxs[3]  = { 0, 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 3, &v));
        EXPECT_EQ(v.i64, 2);
    }
    {
        const uint8_t kinds[4] = { 0, 0, 0, 0 };
        const char*   strs[4]  = { "child", "child", "child", "n" };
        const int32_t lens[4]  = { 5, 5, 5, 1 };
        const int32_t idxs[4]  = { 0, 0, 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 4, &v));
        EXPECT_EQ(v.i64, 3);
    }
    {
        const uint8_t kinds[5] = { 0, 0, 0, 0, 0 };
        const char*   strs[5]  = { "child", "child", "child", "child", "n" };
        const int32_t lens[5]  = { 5, 5, 5, 5, 1 };
        const int32_t idxs[5]  = { 0, 0, 0, 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 5, &v));
        EXPECT_EQ(v.i64, 4);
    }
    // Both leaf siblings at the deepest level.
    {
        const uint8_t kinds[5] = { 0, 0, 0, 0, 0 };
        const char*   strs[5]  = { "child", "child", "child", "child", "leaf_a" };
        const int32_t lens[5]  = { 5, 5, 5, 5, 6 };
        const int32_t idxs[5]  = { 0, 0, 0, 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 5, &v));
        EXPECT_EQ(v.i64, 1000);
    }
    {
        const uint8_t kinds[5] = { 0, 0, 0, 0, 0 };
        const char*   strs[5]  = { "child", "child", "child", "child", "leaf_b" };
        const int32_t lens[5]  = { 5, 5, 5, 5, 6 };
        const int32_t idxs[5]  = { 0, 0, 0, 0, 0 };
        ASSERT_TRUE(path_lookup(view, kinds, strs, lens, idxs, 5, &v));
        EXPECT_EQ(v.i64, 2000);
    }
}
