// Roaring bitmap deserialiser — hand-built portable-format vectors covering
// the Array, Bitset, and Run container types + the offset-header path.

#include "bolt/ingest/bolt_roaring.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using bolt::Arena;
using bolt::ingest::RoaringBitmap;
using bolt::ingest::roaring_cardinality;
using bolt::ingest::roaring_contains;
using bolt::ingest::roaring_deserialize;

// Little-endian helpers for building portable vectors.
void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>((x >> (8*i)) & 0xFF));
}

// Single Array container {1, 2, 100} in key 0. NO_RUNCONTAINER cookie, n=1.
TEST(BoltRoaring, ArrayContainer) {
    std::vector<uint8_t> data;
    put32(data, 0x00003B4Du);     // cookie (no run)
    put32(data, 1u);              // n_containers
    put16(data, 0u);              // key 0
    put16(data, 2u);              // cardinality - 1 (3 values)
    // n < 4 ⇒ no offset header. Array payload (sorted u16):
    put16(data, 1u);
    put16(data, 2u);
    put16(data, 100u);

    Arena arena;
    RoaringBitmap bm;
    ASSERT_TRUE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), 3u);
    EXPECT_TRUE(roaring_contains(&bm, 1u));
    EXPECT_TRUE(roaring_contains(&bm, 2u));
    EXPECT_TRUE(roaring_contains(&bm, 100u));
    EXPECT_FALSE(roaring_contains(&bm, 0u));
    EXPECT_FALSE(roaring_contains(&bm, 3u));
    EXPECT_FALSE(roaring_contains(&bm, 65536u));   // different high bits
}

// Two containers (key 0 array, key 1 array) ⇒ exercises the offset header
// only when n >= 4; here n=2 so still no offset header.
TEST(BoltRoaring, TwoArrayContainers) {
    std::vector<uint8_t> data;
    put32(data, 0x00003B4Du);
    put32(data, 2u);             // n_containers
    put16(data, 0u); put16(data, 0u);   // key 0, card-1=0 (1 value)
    put16(data, 1u); put16(data, 1u);   // key 1, card-1=1 (2 values)
    put16(data, 42u);                   // key 0 payload: {42}
    put16(data, 7u); put16(data, 9u);   // key 1 payload: {7,9} -> 0x10007,0x10009

    Arena arena;
    RoaringBitmap bm;
    ASSERT_TRUE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), 3u);
    EXPECT_TRUE(roaring_contains(&bm, 42u));
    EXPECT_TRUE(roaring_contains(&bm, (1u << 16) | 7u));
    EXPECT_TRUE(roaring_contains(&bm, (1u << 16) | 9u));
    EXPECT_FALSE(roaring_contains(&bm, (1u << 16) | 8u));
}

// Run container {10..14} via the run-flag cookie. One run: start=10, len-1=4.
TEST(BoltRoaring, RunContainer) {
    std::vector<uint8_t> data;
    // cookie = SERIAL_COOKIE (0x373B) | ((n-1) << 16); n=1 ⇒ 0x0000373B.
    put32(data, 0x0000373Bu);
    // run-flag bitset: ceil(1/8)=1 byte, bit 0 set (container 0 is a run).
    data.push_back(0x01u);
    put16(data, 0u);             // key 0
    put16(data, 4u);             // cardinality - 1 (5 values: 10..14)
    // n=1 < threshold ⇒ no offset header (run cookie never has one anyway).
    put16(data, 1u);             // n_runs
    put16(data, 10u);            // run start
    put16(data, 4u);             // run length - 1

    Arena arena;
    RoaringBitmap bm;
    ASSERT_TRUE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), 5u);
    for (uint32_t v = 10; v <= 14; ++v) EXPECT_TRUE(roaring_contains(&bm, v));
    EXPECT_FALSE(roaring_contains(&bm, 9u));
    EXPECT_FALSE(roaring_contains(&bm, 15u));
}

// Bitset container: cardinality > 4096 ⇒ 8192-byte word slab. Set bits 0..4999.
TEST(BoltRoaring, BitsetContainer) {
    std::vector<uint8_t> data;
    put32(data, 0x00003B4Du);
    put32(data, 1u);
    put16(data, 0u);              // key 0
    put16(data, 4999u);           // cardinality - 1 (5000 values: 0..4999)
    // n=1 < 4 ⇒ no offset header. 1024 u64 words.
    std::vector<uint64_t> words(1024, 0);
    for (uint32_t i = 0; i < 5000; ++i) words[i >> 6] |= (1ull << (i & 63));
    for (uint64_t w : words) {
        for (int i = 0; i < 8; ++i) data.push_back(static_cast<uint8_t>((w >> (8*i)) & 0xFF));
    }

    Arena arena;
    RoaringBitmap bm;
    ASSERT_TRUE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), 5000u);
    EXPECT_TRUE(roaring_contains(&bm, 0u));
    EXPECT_TRUE(roaring_contains(&bm, 4999u));
    EXPECT_FALSE(roaring_contains(&bm, 5000u));
}

// Truncated input must fail cleanly, never crash.
TEST(BoltRoaring, TruncatedNeverCrashes) {
    std::vector<uint8_t> data;
    put32(data, 0x00003B4Du);
    put32(data, 1u);
    put16(data, 0u); put16(data, 2u);
    put16(data, 1u);   // only 1 of 3 array values
    Arena arena;
    RoaringBitmap bm;
    EXPECT_FALSE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    // Empty / tiny inputs.
    EXPECT_FALSE(roaring_deserialize(data.data(), 0, &arena, &bm));
    EXPECT_FALSE(roaring_deserialize(data.data(), 2, &arena, &bm));
}

}  // namespace
