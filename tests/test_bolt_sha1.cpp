// test_bolt_sha1.cpp — FIPS 180 SHA-1 vectors, across the padding edges.

#include "bolt/crypto/sha1.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace {

std::string hex(const uint8_t* d) {
    static const char* k = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 20; ++i) {
        s += k[d[i] >> 4];
        s += k[d[i] & 0xF];
    }
    return s;
}

std::string sha1_of(const std::string& m) {
    uint8_t d[20];
    bolt::crypto::sha1(reinterpret_cast<const uint8_t*>(m.data()), m.size(), d);
    return hex(d);
}

TEST(BoltSha1, Vectors) {
    EXPECT_EQ(sha1_of(""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(sha1_of("abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(sha1_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    EXPECT_EQ(sha1_of(std::string(1000000, 'a')),
              "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}

TEST(BoltSha1, PaddingBoundaries) {
    // 55/56/63/64 bytes straddle the one- vs two-block padding split.
    EXPECT_EQ(sha1_of(std::string(55, 'x')), "cef734ba81a024479e09eb5a75b6ddae62e6abf1");
    EXPECT_EQ(sha1_of(std::string(56, 'x')), "901305367c259952f4e7af8323f480d59f81335b");
    EXPECT_EQ(sha1_of(std::string(63, 'x')), "0ddc4e0cccd9a12850deb5abb0853a4425559fec");
    EXPECT_EQ(sha1_of(std::string(64, 'x')), "bb2fa3ee7afb9f54c6dfb5d021f14b1ffe40c163");
}

}  // namespace
