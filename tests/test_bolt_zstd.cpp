// zstd codec roundtrip — skips when libzstd is not compiled in.

#include "bolt/ingest/bolt_zstd.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

TEST(BoltZstd, Roundtrip) {
    if (!zstd_available()) GTEST_SKIP() << "libzstd not compiled in";
    std::string src;
    for (int i = 0; i < 1000; ++i) src += "the quick brown fox ";
    const auto* s = reinterpret_cast<const uint8_t*>(src.data());

    const uint64_t bound = zstd_max_compressed_len(src.size());
    ASSERT_GT(bound, 0u);
    std::vector<uint8_t> comp(static_cast<size_t>(bound));
    uint64_t clen = bound;
    ASSERT_EQ(zstd_compress(s, src.size(), comp.data(), &clen, 3), kCodecOk);
    ASSERT_GT(clen, 0u);
    EXPECT_LT(clen, src.size());

    std::vector<uint8_t> back(src.size());
    ASSERT_EQ(zstd_decompress(comp.data(), clen, back.data(), back.size()),
              kCodecOk);
    EXPECT_EQ(0, std::memcmp(back.data(), s, src.size()));
}

TEST(BoltZstd, UnavailableReturnsCode) {
    if (zstd_available()) GTEST_SKIP() << "zstd present; nothing to assert";
    uint8_t dst[16]; uint64_t n = sizeof(dst);
    const uint8_t in[4] = {1, 2, 3, 4};
    EXPECT_EQ(zstd_compress(in, 4, dst, &n, 1), kCodecUnavailable);
}

}  // namespace
