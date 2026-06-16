// lz4 codec roundtrip — skips when liblz4 is not compiled in.

#include "bolt/ingest/bolt_lz4.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

TEST(BoltLz4, Roundtrip) {
    if (!lz4_available()) GTEST_SKIP() << "liblz4 not compiled in";
    std::string src;
    for (int i = 0; i < 1000; ++i) src += "lz4 block compressible run ";
    const auto* s = reinterpret_cast<const uint8_t*>(src.data());

    const uint64_t bound = lz4_max_compressed_len(src.size());
    ASSERT_GT(bound, 0u);
    std::vector<uint8_t> comp(static_cast<size_t>(bound));
    uint64_t clen = bound;
    ASSERT_EQ(lz4_compress(s, src.size(), comp.data(), &clen, 0), kCodecOk);
    ASSERT_GT(clen, 0u);

    std::vector<uint8_t> back(src.size());
    ASSERT_EQ(lz4_decompress(comp.data(), clen, back.data(), back.size()),
              kCodecOk);
    EXPECT_EQ(0, std::memcmp(back.data(), s, src.size()));
}

TEST(BoltLz4, UnavailableReturnsCode) {
    if (lz4_available()) GTEST_SKIP() << "lz4 present; nothing to assert";
    uint8_t dst[16]; uint64_t n = sizeof(dst);
    const uint8_t in[4] = {1, 2, 3, 4};
    EXPECT_EQ(lz4_compress(in, 4, dst, &n, 0), kCodecUnavailable);
}

}  // namespace
