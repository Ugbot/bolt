// test_bolt_md5.cpp — RFC 1321 §A.5 vectors plus padding-boundary lengths
// whose expected digests come from Python hashlib.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "bolt/kernels/bolt_md5.h"

namespace km = bolt::kernels::md5;

namespace {

std::string hex_of(const void* p, uint64_t n) {
    char out[km::k_hex_chars];
    km::md5_hex(static_cast<const uint8_t*>(p), n, out);
    return std::string(out, km::k_hex_chars);
}

std::string hex_of(const char* s) { return hex_of(s, std::strlen(s)); }

}  // namespace

TEST(Md5, Rfc1321Vectors) {
    EXPECT_EQ(hex_of(nullptr, 0), "d41d8cd98f00b204e9800998ecf8427e");
    EXPECT_EQ(hex_of("a"), "0cc175b9c0f1b6a831c399e269772661");
    EXPECT_EQ(hex_of("abc"), "900150983cd24fb0d6963f7d28e17f72");
    EXPECT_EQ(hex_of("message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
    EXPECT_EQ(hex_of("abcdefghijklmnopqrstuvwxyz"),
              "c3fcd3d76192e4007dfb496cca67e13b");
    EXPECT_EQ(hex_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
              "d174ab98d277d9f5a5611c2c9f419d9f");
    EXPECT_EQ(hex_of("1234567890123456789012345678901234567890"
                     "1234567890123456789012345678901234567890"),
              "57edf4a22be3c955ac49da2e2107b67a");
}

TEST(Md5, PaddingBoundaries) {
    struct Case { uint32_t n; const char* hex; };
    const Case cases[] = {
        {55, "c9e512626618c9980ef21a96597af94c"},
        {56, "ecde7caa08e9f5657c863df107cac60a"},
        {57, "b17b1a018dd6a4d1edda8aca15f17846"},
        {63, "2f0301069e1c40af7f6c8f843b1b13f2"},
        {64, "b6bf87c24b1bc334e2541387a92b981b"},
        {65, "f168246f08b6134d66bd2a10343fa9f1"},
        {119, "d5dc3d8264de3aa24dee105910ee27fe"},
        {120, "bcc139b3848923904860d1eefd6e5923"},
        {128, "3e85b70ffc8df5c735ecf2a8f14f1bee"},
        {1000, "2b1e78d5765de9e10495a01412a1cf22"},
    };
    uint8_t buf[1000];
    for (uint32_t i = 0; i < sizeof(buf); ++i) {
        buf[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xffu);
    }
    for (const Case& c : cases) {
        EXPECT_EQ(hex_of(buf, c.n), c.hex) << "n=" << c.n;
    }
}

TEST(Md5, RawDigestMatchesHex) {
    uint8_t d[km::k_digest_bytes];
    km::md5(reinterpret_cast<const uint8_t*>("abc"), 3, d);
    EXPECT_EQ(d[0], 0x90u);
    EXPECT_EQ(d[15], 0x72u);
}
