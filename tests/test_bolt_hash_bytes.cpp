// test_bolt_hash_bytes.cpp — coverage for bolt::hash_bytes (byte/string hash).
//
// hash_bytes is the canonical byte-range hash used by the RDF term interners
// (they intern via bolt::SwissTable). These tests pin down its contract:
//   - empty (len 0) is well-defined and stable,
//   - same bytes => same hash, determinism across repeated calls,
//   - one-bit flip => different hash (avalanche sanity),
//   - real RDF-term-like strings hash distinctly,
//   - composes with swiss_mix (the SwissTable finalizer).
//
// Test code may use the stdlib freely; the kernel under test does not.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "bolt/bolt_hash.h"
#include "bolt/kernels/bolt_hash_sv.h"

using bolt::hash_bytes;

namespace {

uint64_t h(const std::string& s) {
    return hash_bytes(s.data(), s.size());
}

}  // namespace

// Empty input is well-defined: nullptr+0 and ""+0 both hash, and identically.
TEST(BoltHashBytes, EmptyIsWellDefined) {
    const uint64_t a = hash_bytes(nullptr, 0);
    const uint64_t b = hash_bytes("", 0);
    EXPECT_EQ(a, b);
    // Stable across two calls.
    EXPECT_EQ(hash_bytes(nullptr, 0), a);
}

// Same bytes always produce the same hash (determinism, repeated calls).
TEST(BoltHashBytes, Deterministic) {
    const std::string s = "<http://example.org/resource/12345>";
    const uint64_t a = h(s);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(h(s), a);
    }
    // Independent buffer with identical content hashes identically.
    std::vector<char> buf(s.begin(), s.end());
    EXPECT_EQ(hash_bytes(buf.data(), buf.size()), a);
}

// Length is part of the digest: "ab" != "ab\0" even though the prefix matches.
TEST(BoltHashBytes, LengthSensitive) {
    const uint64_t a = hash_bytes("ab", 2);
    const uint64_t b = hash_bytes("ab\0", 3);
    EXPECT_NE(a, b);
}

// One-bit flip changes the hash — avalanche sanity across every bit position
// of a 24-byte input (block path + tail path both exercised).
TEST(BoltHashBytes, OneBitFlipChangesHash) {
    uint8_t base[24];
    for (size_t i = 0; i < sizeof(base); ++i) base[i] = static_cast<uint8_t>(i * 7 + 1);
    const uint64_t ref = hash_bytes(base, sizeof(base));

    for (size_t byte = 0; byte < sizeof(base); ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            uint8_t flipped[24];
            std::memcpy(flipped, base, sizeof(base));
            flipped[byte] ^= static_cast<uint8_t>(1u << bit);
            EXPECT_NE(hash_bytes(flipped, sizeof(flipped)), ref)
                << "byte=" << byte << " bit=" << bit;
        }
    }
}

// Known vectors are stable across two calls (regression pin — values are
// whatever the algorithm produces; the point is they don't drift).
TEST(BoltHashBytes, KnownVectorsStable) {
    const char* vectors[] = {
        "a", "ab", "abc", "abcdefgh", "abcdefghi",
        "the quick brown fox jumps over the lazy dog",
    };
    for (const char* v : vectors) {
        const size_t n = std::strlen(v);
        EXPECT_EQ(hash_bytes(v, n), hash_bytes(v, n));
    }
}

// Real RDF-term-like strings (IRIs and literals) hash distinctly — the
// interner relies on low collision rate for these shapes.
TEST(BoltHashBytes, RdfTermsHashDistinctly) {
    const std::vector<std::string> terms = {
        "<http://ex/foo>",
        "<http://ex/bar>",
        "<http://example.org/resource/1>",
        "<http://example.org/resource/2>",
        "\"literal\"",
        "\"literal\"@en",
        "\"literal\"^^<http://www.w3.org/2001/XMLSchema#string>",
        "_:b0",
        "_:b1",
        "",
    };
    std::set<uint64_t> seen;
    for (const auto& t : terms) {
        const uint64_t hv = hash_bytes(t.data(), t.size());
        EXPECT_TRUE(seen.insert(hv).second) << "collision on: " << t;
    }
}

// Composes with swiss_mix: hash_bytes already runs its accumulator through
// swiss_mix, so feeding the output into swiss_mix again (as SwissTable would
// on a raw key) still yields a stable, well-distributed value. Sanity-check
// that distinct inputs stay distinct after a second mix.
TEST(BoltHashBytes, ComposesWithSwissMix) {
    const uint64_t hx = hash_bytes("<http://ex/foo>", 15);
    const uint64_t hy = hash_bytes("<http://ex/bar>", 15);
    EXPECT_NE(bolt::swiss_mix(hx), bolt::swiss_mix(hy));
    // Idempotent across repeated mixing calls.
    EXPECT_EQ(bolt::swiss_mix(hx), bolt::swiss_mix(hx));
}

// Independently generated integer reference vectors: byte i = (17*i+3)%256,
// blocks/tails packed little-endian. Pin the fold state and every historical
// finalizer, rather than obtaining expected values from production helpers.
TEST(BoltHashBytes, ExplicitV1GoldensAndGenericTierArePinned) {
    struct Vector { uint32_t len; uint64_t state, wyhash3, xxh3, murmur3; };
    constexpr Vector vectors[] = {
        {0u, 0x9e3779b97f4a7c15ull, 0xacfb261e473d9c3aull, 0x1309a4ec498bddf4ull, 0x9ca066f1a4ab2eeaull},
        {1u, 0x1d0fd20836cbd034ull, 0x7d4696075e71df53ull, 0xb868b9ebfb14d86full, 0xf61851252e4f5bd7ull},
        {3u, 0x47ab1fa4e44a71aeull, 0x41d7ace54bc51e6bull, 0xa91294703e56f8cbull, 0xcb905d931007442aull},
        {7u, 0xbf94d85836f7edafull, 0x1bb52558424ce715ull, 0xc5d8a1a1b2d543a4ull, 0xb345a96798deea5eull},
        {8u, 0x22bbf74041e78950ull, 0xbb8d0a7eac9f5dceull, 0x006bc270cf7ae1eaull, 0x7b2a3ca4dac6e31bull},
        {9u, 0x9586d4c4f436c7e2ull, 0x4057c50e0b01948cull, 0xf1325efde16b91d9ull, 0x56b64daba71fb237ull},
        {12u, 0x984097e078bf0255ull, 0x3e325093f37f0944ull, 0x0be1cddb1dc79e41ull, 0xc83f4888b72f6545ull},
        {16u, 0x78d8c249807d510dull, 0xaaa8c71b0adf5c37ull, 0xa13adee6c4b67ac1ull, 0xffe38051470fefc6ull},
        {24u, 0x168e9b95f4f1188dull, 0x85294ecb5959ab43ull, 0xf55679f45e49da3dull, 0x01c6d3ac106a7b40ull},
        {33u, 0x518cfcdf74184d03ull, 0x8706091677a88e22ull, 0x12f5ef69f9631ad4ull, 0x5a2d8d609eb4ceb6ull},
        {39u, 0x0594e3ff54f9c148ull, 0xa6bccff9d8c98574ull, 0x241259bc2bb2f023ull, 0xb204ec91620bd7b1ull},
    };
    uint8_t bytes[48];
    for (uint32_t i = 0; i < 48; ++i) bytes[i] = static_cast<uint8_t>(17u * i + 3u);
    for (const auto& v : vectors) {
        SCOPED_TRACE(v.len);
        EXPECT_EQ(bolt::hash_bytes_accumulate_v1(bytes, v.len), v.state);
        EXPECT_EQ(bolt::hash_bytes_wyhash3_v1(bytes, v.len), v.wyhash3);
        EXPECT_EQ(bolt::swiss_mix_wyhash3(v.state), v.wyhash3);
        EXPECT_EQ(bolt::swiss_mix_xxh3(v.state), v.xxh3);
        EXPECT_EQ(bolt::swiss_mix_murmur3(v.state), v.murmur3);
#if defined(BOLT_HASH_TIER_XXH3) && BOLT_HASH_TIER_XXH3
        EXPECT_EQ(hash_bytes(bytes, v.len), v.xxh3);
#elif defined(BOLT_HASH_TIER_MURMUR3) && BOLT_HASH_TIER_MURMUR3
        EXPECT_EQ(hash_bytes(bytes, v.len), v.murmur3);
#else
        EXPECT_EQ(hash_bytes(bytes, v.len), v.wyhash3);
#endif
    }
    EXPECT_EQ(bolt::hash_bytes_wyhash3_v1(nullptr, 0), vectors[0].wyhash3);
    EXPECT_EQ(bolt::hash_bytes_wyhash3_v1("mseg-bloom-golden", 17), 0x3983ea84822c3d7dull);
    EXPECT_EQ(bolt::hash_bytes_wyhash3_v1("ab\0", 3), 0x53024485f99fec79ull);
}

// hash_bytes_sv is hash_bytes over each view's bytes, bit for bit: every
// length 0..40, garbage past the length in the inline bytes, lengths that
// change row to row (the seed cache), spilled rows, both entry points.
TEST(BoltHashBytes, StringViewRowsMatchHashBytes) {
    uint64_t x = 0x243F6A8885A308D3ULL;
    auto next = [&x] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<uint8_t> over(1 << 16);
    for (auto& b : over) b = static_cast<uint8_t>(next());
    std::vector<bolt::StringView> rows;
    std::vector<std::string> bytes;
    for (int i = 0; i < 4000; ++i) {
        const uint32_t len = i < 41 * 4 ? static_cast<uint32_t>(i / 4) : static_cast<uint32_t>(next() % 41);
        bolt::StringView v;
        std::memset(&v, 0xA5 ^ i, sizeof(v));   // junk past the length must not count
        v.length = len;
        const size_t off = static_cast<size_t>(next() % (over.size() - 64));
        std::string s(reinterpret_cast<const char*>(over.data() + off), len);
        if (len <= 12) std::memcpy(v.prefix, s.data(), len);
        else { std::memcpy(v.prefix, s.data(), 4); v.ref.buf_idx = 0; v.ref.offset = static_cast<uint32_t>(off); }
        rows.push_back(v);
        bytes.push_back(s);
    }
    std::vector<uint64_t> out(rows.size()), fixed(rows.size());
    bolt::hash_bytes_sv(rows.data(), rows.size(), over.data(), out.data());
    bolt::hash_bytes_sv_wyhash3_v1(rows.data(), rows.size(), over.data(), fixed.data());
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t want = hash_bytes(bytes[i].data(), bytes[i].size());
        ASSERT_EQ(out[i], want) << "row " << i << " len " << bytes[i].size();
        ASSERT_EQ(bolt::hash_bytes_sv(rows[i], over.data()), want) << "row " << i;
        const uint64_t persisted = bolt::hash_bytes_wyhash3_v1(bytes[i].data(), bytes[i].size());
        ASSERT_EQ(fixed[i], persisted) << "fixed row " << i;
        ASSERT_EQ(bolt::hash_bytes_sv_wyhash3_v1(rows[i], over.data()), persisted) << "fixed row " << i;
    }
}
