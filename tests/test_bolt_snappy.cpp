// W-PQ — snappy block decompressor vs cramjam-generated vectors
// (tests/data/snappy_vectors.json: [[b64 compressed, b64 raw], ...])
// plus handcrafted overlap-copy cases and corrupt-input fuzzing.

#include "bolt/ingest/bolt_snappy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using bolt::ingest::snappy_decompress;
using bolt::ingest::snappy_uncompressed_len;

std::string slurp(const char* path) {
    std::string v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

int b64v(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::vector<uint8_t> b64decode(const std::string& s) {
    std::vector<uint8_t> out;
    int acc = 0, bits = 0;
    for (char c : s) {
        const int v = b64v(c);
        if (v < 0) continue;   // '=' padding / whitespace
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// Minimal extraction of [["b64","b64"],...] pairs (the vectors file is
// machine-generated, flat, and contains no escapes).
std::vector<std::pair<std::string, std::string>> load_vectors() {
    const std::string j =
        slurp((std::string(BOLT_TEST_DATA_DIR) + "/snappy_vectors.json").c_str());
    std::vector<std::pair<std::string, std::string>> out;
    std::vector<std::string> strs;
    size_t i = 0;
    while (i < j.size()) {
        if (j[i] == '"') {
            const size_t e = j.find('"', i + 1);
            if (e == std::string::npos) break;
            strs.push_back(j.substr(i + 1, e - i - 1));
            i = e + 1;
        } else {
            ++i;
        }
    }
    for (size_t k = 0; k + 1 < strs.size(); k += 2) {
        out.emplace_back(strs[k], strs[k + 1]);
    }
    return out;
}

TEST(BoltSnappy, CramjamVectors) {
    const auto vecs = load_vectors();
    ASSERT_EQ(vecs.size(), 5u);
    for (size_t i = 0; i < vecs.size(); ++i) {
        const auto comp = b64decode(vecs[i].first);
        const auto raw  = b64decode(vecs[i].second);
        uint64_t ulen = 0;
        ASSERT_GT(snappy_uncompressed_len(comp.data(), comp.size(), &ulen), 0u)
            << "case " << i;
        ASSERT_EQ(ulen, raw.size()) << "case " << i;
        std::vector<uint8_t> dst(raw.size() + 8, 0xCD);
        ASSERT_TRUE(snappy_decompress(comp.data(), comp.size(), dst.data(),
                                      raw.size())) << "case " << i;
        EXPECT_EQ(0, std::memcmp(dst.data(), raw.data(), raw.size()))
            << "case " << i;
        // Guard bytes untouched (no overflow past dst_len).
        for (size_t g = raw.size(); g < dst.size(); ++g) {
            ASSERT_EQ(dst[g], 0xCD);
        }
    }
}

TEST(BoltSnappy, HandcraftedOverlapCopy) {
    // "ab" then copy(offset=2, len=8) => "ab" * 5 — the overlapping
    // forward-copy (RLE) semantics. Encoded by hand:
    //   preamble 10; literal tag len=2 (tag=(2-1)<<2=0x04) 'a' 'b';
    //   copy-1byte: len=8 -> ((8-4)&7)<<2 | 1 = 0x11, offset=2 -> hi 0, lo 2.
    const uint8_t comp[] = {10, 0x04, 'a', 'b', 0x11, 0x02};
    uint8_t dst[10];
    ASSERT_TRUE(snappy_decompress(comp, sizeof(comp), dst, sizeof(dst)));
    EXPECT_EQ(0, std::memcmp(dst, "ababababab", 10));
}

TEST(BoltSnappy, CorruptInputsNeverCrash) {
    const auto vecs = load_vectors();
    ASSERT_FALSE(vecs.empty());
    const auto comp = b64decode(vecs[3].first);   // the 5000-byte case
    const auto raw  = b64decode(vecs[3].second);
    std::vector<uint8_t> dst(raw.size());
    // Truncations.
    for (size_t cut = 1; cut <= comp.size(); cut += 7) {
        (void)snappy_decompress(comp.data(), comp.size() - cut, dst.data(),
                                dst.size());
    }
    // Byte flips.
    std::vector<uint8_t> mut(comp);
    for (size_t i = 0; i < mut.size(); i += 3) {
        mut[i] ^= 0xFF;
        (void)snappy_decompress(mut.data(), mut.size(), dst.data(), dst.size());
        mut[i] ^= 0xFF;
    }
    // Wrong dst_len must fail cleanly.
    EXPECT_FALSE(snappy_decompress(comp.data(), comp.size(), dst.data(),
                                   dst.size() - 1));
    SUCCEED();
}

}  // namespace
