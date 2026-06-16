// gzip/zlib codec roundtrip — skips when zlib is not compiled in.

#include "bolt/ingest/bolt_gzip.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

TEST(BoltGzip, Roundtrip) {
    if (!gzip_available()) GTEST_SKIP() << "zlib not compiled in";
    std::string src;
    for (int i = 0; i < 1000; ++i) src += "lakehouse deflate payload ";
    const auto* s = reinterpret_cast<const uint8_t*>(src.data());

    const uint64_t bound = gzip_max_compressed_len(src.size());
    ASSERT_GT(bound, 0u);
    std::vector<uint8_t> comp(static_cast<size_t>(bound));
    uint64_t clen = bound;
    ASSERT_EQ(gzip_compress(s, src.size(), comp.data(), &clen, 6), kCodecOk);
    ASSERT_GT(clen, 0u);

    std::vector<uint8_t> back(src.size());
    ASSERT_EQ(gzip_decompress(comp.data(), clen, back.data(), back.size()),
              kCodecOk);
    EXPECT_EQ(0, std::memcmp(back.data(), s, src.size()));
}

TEST(BoltGzip, UnavailableReturnsCode) {
    if (gzip_available()) GTEST_SKIP() << "zlib present; nothing to assert";
    uint8_t dst[16]; uint64_t n = sizeof(dst);
    const uint8_t in[4] = {1, 2, 3, 4};
    EXPECT_EQ(gzip_compress(in, 4, dst, &n, 1), kCodecUnavailable);
}

}  // namespace
