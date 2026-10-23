// Iceberg v3 deletion-vector blobs: round trip, corruption refusals, and the multi-bucket (positions >= 2^32) layout.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_deflate.h"
#include "bolt/lakehouse/iceberg/deletion_vector.h"

using namespace bolt::lakehouse::iceberg;

TEST(IcebergDeletionVector, RoundTripsPositionsAndRefusesCorruption) {
    bolt::Arena a;
    std::vector<uint32_t> pos = {0, 1, 7, 100, 65535, 65536, 70000, 1000000};
    const uint8_t* blob = nullptr; uint64_t len = 0;
    ASSERT_TRUE(iceberg_dv_encode(pos.data(), pos.size(), &a, &blob, &len));
    bolt::ingest::RoaringBitmap64 dv{};
    ASSERT_TRUE(iceberg_dv_decode(blob, len, &a, &dv));
    EXPECT_EQ(dv.cardinality, pos.size());
    for (uint32_t p : pos) EXPECT_TRUE(iceberg_dv_contains(&dv, p));
    for (uint32_t p : {2u, 6u, 99u, 65534u, 999999u}) EXPECT_FALSE(iceberg_dv_contains(&dv, p));

    std::vector<uint8_t> bad(blob, blob + len);
    bad[len / 2] ^= 0x01;  // payload bit flip: CRC must catch it
    bolt::ingest::RoaringBitmap64 x{};
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
    bad.assign(blob, blob + len); bad[4] = 0;  // wrong magic
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
    EXPECT_FALSE(iceberg_dv_decode(blob, len - 1, &a, &x));  // truncated
    bad.assign(blob, blob + len); bad[3] ^= 0x04;  // length field disagrees with the blob
    EXPECT_FALSE(iceberg_dv_decode(bad.data(), bad.size(), &a, &x));
}

TEST(IcebergDeletionVector, DecodesAMultiBucketBitmapFromAnotherWriter) {
    // Hand-built per the spec: two 32-bit buckets (high keys 0 and 1) so positions 5 and 2^32+9 are set.
    auto portable = [](uint16_t value) {  // one array container holding `value`
        std::vector<uint8_t> v = {0x3A, 0x30, 0, 0, 1, 0, 0, 0,  // cookie 12346, 1 container
                                  0, 0, 0, 0,                    // key 0, card-1 0
                                  0x10, 0, 0, 0,                 // offset header (16)
                                  static_cast<uint8_t>(value & 0xFF), static_cast<uint8_t>(value >> 8)};
        return v;
    };
    std::vector<uint8_t> bm;
    auto u64 = [&](uint64_t x) { for (int i = 0; i < 8; ++i) bm.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    auto u32 = [&](uint32_t x) { for (int i = 0; i < 4; ++i) bm.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    u64(2);
    u32(0); { auto p = portable(5); bm.insert(bm.end(), p.begin(), p.end()); }
    u32(1); { auto p = portable(9); bm.insert(bm.end(), p.begin(), p.end()); }
    std::vector<uint8_t> body = {0xD1, 0xD3, 0x39, 0x64};
    body.insert(body.end(), bm.begin(), bm.end());
    std::vector<uint8_t> blob(4);
    for (int i = 0; i < 4; ++i) blob[i] = static_cast<uint8_t>(body.size() >> (8 * (3 - i)));
    blob.insert(blob.end(), body.begin(), body.end());
    const uint32_t crc = bolt::ingest::crc32_ieee(body.data(), body.size(), 0u);
    for (int i = 0; i < 4; ++i) blob.push_back(static_cast<uint8_t>(crc >> (8 * (3 - i))));

    bolt::Arena a;
    bolt::ingest::RoaringBitmap64 dv{};
    ASSERT_TRUE(iceberg_dv_decode(blob.data(), blob.size(), &a, &dv));
    EXPECT_EQ(dv.cardinality, 2u);
    EXPECT_TRUE(iceberg_dv_contains(&dv, 5));
    EXPECT_TRUE(iceberg_dv_contains(&dv, (1ull << 32) + 9));
    EXPECT_FALSE(iceberg_dv_contains(&dv, 9));
    EXPECT_FALSE(iceberg_dv_contains(&dv, (1ull << 32) + 5));
}
