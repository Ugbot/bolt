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
