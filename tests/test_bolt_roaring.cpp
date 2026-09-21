// Roaring bitmap (de)serialiser — hand-built portable-format vectors
// covering the Array, Bitset, and Run container types + the offset-header
// path, PLUS a round-trip through the real serialiser + an independent
// cross-check against real CRoaring output (G2ICE-80).
//
// G2ICE-80 correction: this file's original hand-built fixtures used cookie
// 0x00003B4D/0x0000373B and assumed the NO_RUNCONTAINER offset table is only
// present when n>=4 — both WRONG (the real values are 12346/0x303A and
// 12347/0x303B; the offset table is ALWAYS present for NO_RUNCONTAINER,
// verified against a real `pyroaring.BitMap.serialize()` byte dump — see
// bolt_roaring.h's banner). The fixtures below were re-derived against the
// real format so this file stops self-certifying the bug it was written to
// pin down.

#include "bolt/ingest/bolt_roaring.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using bolt::Arena;
using bolt::ingest::RoaringBitmap;
using bolt::ingest::RoaringBitmap64;
using bolt::ingest::roaring_cardinality;
using bolt::ingest::roaring_cardinality64;
using bolt::ingest::roaring_contains;
using bolt::ingest::roaring_contains64;
using bolt::ingest::roaring_deserialize;
using bolt::ingest::roaring_deserialize_r64;
using bolt::ingest::roaring_serialize;
using bolt::ingest::roaring_serialize_bound;
using bolt::ingest::roaring_serialize_r64_bound;
using bolt::ingest::roaring_serialize_r64_single_bucket;
using bolt::ingest::roaring_to_sorted_array;

// Little-endian helpers for building portable vectors.
void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>((x >> (8*i)) & 0xFF));
}

// Single Array container {1, 2, 100} in key 0. NO_RUNCONTAINER cookie, n=1.
// Real layout: cookie(4) n(4) keyscards(4) offsets(4, ALWAYS present for
// this cookie) payload(6) = 22 bytes; offset[0]=16 (start of payload).
TEST(BoltRoaring, ArrayContainer) {
    std::vector<uint8_t> data;
    put32(data, 12346u);          // SERIAL_COOKIE_NO_RUNCONTAINER
    put32(data, 1u);              // n_containers
    put16(data, 0u);              // key 0
    put16(data, 2u);              // cardinality - 1 (3 values)
    put32(data, 16u);             // offset[0]: payload starts at byte 16
    put16(data, 1u);
    put16(data, 2u);
    put16(data, 100u);
    ASSERT_EQ(data.size(), 22u);

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

// Two containers (key 0 array {42}, key 1 array {7,9}). Real layout:
// cookie(4) n(4) keyscards(2*4=8) offsets(2*4=8) payloads(2+4=6) = 30 bytes;
// payload region starts at 24; offset[0]=24, offset[1]=24+2=26.
TEST(BoltRoaring, TwoArrayContainers) {
    std::vector<uint8_t> data;
    put32(data, 12346u);
    put32(data, 2u);             // n_containers
    put16(data, 0u); put16(data, 0u);   // key 0, card-1=0 (1 value)
    put16(data, 1u); put16(data, 1u);   // key 1, card-1=1 (2 values)
    put32(data, 24u);            // offset[0]
    put32(data, 26u);            // offset[1]
    put16(data, 42u);                   // key 0 payload: {42}
    put16(data, 7u); put16(data, 9u);   // key 1 payload: {7,9} -> 0x10007,0x10009
    ASSERT_EQ(data.size(), 30u);

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
    // cookie = SERIAL_COOKIE (12347/0x303B) | ((n-1) << 16); n=1 ⇒ 0x0000303B.
    put32(data, 12347u);
    // run-flag bitset: ceil(1/8)=1 byte, bit 0 set (container 0 is a run).
    data.push_back(0x01u);
    put16(data, 0u);             // key 0
    put16(data, 4u);             // cardinality - 1 (5 values: 10..14)
    // n=1 < threshold(4) ⇒ no offset header for the RUN-cookie form.
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

// Bitset container: cardinality > 4096 ⇒ 8192-byte word slab. Set bits
// 0..4999. Real layout adds the ALWAYS-present offset[0]=16 before payload.
TEST(BoltRoaring, BitsetContainer) {
    std::vector<uint8_t> data;
    put32(data, 12346u);
    put32(data, 1u);
    put16(data, 0u);              // key 0
    put16(data, 4999u);           // cardinality - 1 (5000 values: 0..4999)
    put32(data, 16u);             // offset[0]
    std::vector<uint64_t> words(1024, 0);
    for (uint32_t i = 0; i < 5000; ++i) words[i >> 6] |= (1ull << (i & 63));
    for (uint64_t w : words) {
        for (int i = 0; i < 8; ++i) data.push_back(static_cast<uint8_t>((w >> (8*i)) & 0xFF));
    }
    ASSERT_EQ(data.size(), 16u + 8192u);

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
    put32(data, 12346u);
    put32(data, 1u);
    put16(data, 0u); put16(data, 2u);
    put16(data, 1u);   // missing the offset table AND the rest of the payload
    Arena arena;
    RoaringBitmap bm;
    EXPECT_FALSE(roaring_deserialize(data.data(), data.size(), &arena, &bm));
    // Empty / tiny inputs.
    EXPECT_FALSE(roaring_deserialize(data.data(), 0, &arena, &bm));
    EXPECT_FALSE(roaring_deserialize(data.data(), 2, &arena, &bm));
}

// ---------------------------------------------------------------------------
// G2ICE-80: serialiser round-trip + a byte-exact cross-check against a REAL
// CRoaring (`pyroaring.BitMap([1,2,3,100000,100001,5000000]).serialize()`)
// output, captured once and pinned here so a regression in the writer is
// caught without needing Python at test time.
// ---------------------------------------------------------------------------

TEST(BoltRoaring, SerializeMatchesRealCRoaringBytes) {
    const uint32_t vals[] = {1u, 2u, 3u, 100000u, 100001u, 5000000u};
    const uint8_t expected[] = {
        0x3a, 0x30, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x4c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
        0x26, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
        0x03, 0x00, 0xa0, 0x86, 0xa1, 0x86, 0x40, 0x4b,
    };
    Arena arena;
    std::vector<uint8_t> buf(roaring_serialize_bound(6));
    uint64_t len = 0;
    ASSERT_TRUE(roaring_serialize(vals, 6, &arena, buf.data(), buf.size(), &len));
    ASSERT_EQ(len, sizeof(expected));
    EXPECT_EQ(0, std::memcmp(buf.data(), expected, sizeof(expected)));
}

TEST(BoltRoaring, SerializeDeserializeRoundTrip) {
    const uint32_t vals[] = {0u, 1u, 65535u, 65536u, 70000u, 5000000u, 0xFFFFFFFEu};
    const uint64_t n = 7;
    Arena arena;
    std::vector<uint8_t> buf(roaring_serialize_bound(n));
    uint64_t len = 0;
    ASSERT_TRUE(roaring_serialize(vals, n, &arena, buf.data(), buf.size(), &len));

    Arena arena2;
    RoaringBitmap bm{};
    ASSERT_TRUE(roaring_deserialize(buf.data(), len, &arena2, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), n);
    for (uint32_t v : vals) EXPECT_TRUE(roaring_contains(&bm, v));
    EXPECT_FALSE(roaring_contains(&bm, 42u));

    std::vector<uint32_t> out(n);
    uint64_t got = 0;
    ASSERT_TRUE(roaring_to_sorted_array(&bm, out.data(), out.size(), &got));
    ASSERT_EQ(got, n);
    for (uint64_t i = 0; i < n; ++i) EXPECT_EQ(out[i], vals[i]);
}

// A dense single-key run forces the Bitset container path in the writer.
TEST(BoltRoaring, SerializeDenseBitsetRoundTrip) {
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 5000; ++i) vals.push_back(i * 2u);   // 5000, key 0
    Arena arena;
    std::vector<uint8_t> buf(roaring_serialize_bound(vals.size()));
    uint64_t len = 0;
    ASSERT_TRUE(roaring_serialize(vals.data(), vals.size(), &arena, buf.data(),
                                  buf.size(), &len));
    Arena arena2;
    RoaringBitmap bm{};
    ASSERT_TRUE(roaring_deserialize(buf.data(), len, &arena2, &bm));
    EXPECT_EQ(roaring_cardinality(&bm), 5000u);
    EXPECT_TRUE(roaring_contains(&bm, 0u));
    EXPECT_TRUE(roaring_contains(&bm, 9998u));
    EXPECT_FALSE(roaring_contains(&bm, 1u));
    EXPECT_FALSE(roaring_contains(&bm, 10000u));
}

TEST(BoltRoaring, SerializeRejectsUnsortedOrDuplicate) {
    Arena arena;
    std::vector<uint8_t> buf(64);
    uint64_t len = 0;
    const uint32_t dup[] = {1u, 1u, 2u};
    EXPECT_FALSE(roaring_serialize(dup, 3, &arena, buf.data(), buf.size(), &len));
    const uint32_t unsorted[] = {2u, 1u, 3u};
    EXPECT_FALSE(roaring_serialize(unsorted, 3, &arena, buf.data(), buf.size(), &len));
}

TEST(BoltRoaring, SerializeR64SingleBucketRoundTrip) {
    const uint32_t vals[] = {3u, 7u, 100u};
    Arena arena;
    std::vector<uint8_t> buf(roaring_serialize_r64_bound(3));
    uint64_t len = 0;
    ASSERT_TRUE(roaring_serialize_r64_single_bucket(vals, 3, &arena, buf.data(),
                                                    buf.size(), &len));
    Arena arena2;
    RoaringBitmap64 bm{};
    ASSERT_TRUE(roaring_deserialize_r64(buf.data(), len, &arena2, &bm));
    EXPECT_EQ(roaring_cardinality64(&bm), 3u);
    EXPECT_TRUE(roaring_contains64(&bm, 3u));
    EXPECT_TRUE(roaring_contains64(&bm, 7u));
    EXPECT_TRUE(roaring_contains64(&bm, 100u));
    EXPECT_FALSE(roaring_contains64(&bm, 4u));
}

TEST(BoltRoaring, SerializeR64EmptyBitmap) {
    Arena arena;
    std::vector<uint8_t> buf(roaring_serialize_r64_bound(0));
    uint64_t len = 0;
    ASSERT_TRUE(roaring_serialize_r64_single_bucket(nullptr, 0, &arena, buf.data(),
                                                    buf.size(), &len));
    Arena arena2;
    RoaringBitmap64 bm{};
    ASSERT_TRUE(roaring_deserialize_r64(buf.data(), len, &arena2, &bm));
    EXPECT_EQ(roaring_cardinality64(&bm), 0u);
    EXPECT_FALSE(roaring_contains64(&bm, 0u));
}

}  // namespace
