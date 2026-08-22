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

// ---------------------------------------------------------------------------
// Differential gate for the wide-store rewrite of the copy loops.
//
// The committed cramjam vectors are real-world streams, but a general-purpose
// compressor emits whatever it likes -- it does not aim at the branches this
// decoder added. These tests synthesise tag streams directly so the awkward
// shapes are hit ON PURPOSE: offsets below 8 (the pattern-growth path),
// off == 1 runs, lengths at the 16 and 64 boundaries, and matches landing in
// the last few bytes where the overcopying fast lane must decline and the
// exact tail path must take over.
//
// The oracle is written straight from the format description, byte at a time,
// deliberately NOT as a copy of the previous implementation -- a shared
// misreading of the spec must not be able to make both sides agree.
// ---------------------------------------------------------------------------

bool naive_decompress(const uint8_t* src, uint64_t src_len, uint8_t* dst,
                      uint64_t dst_len) {
    uint64_t hdr = 0;
    const uint32_t consumed = snappy_uncompressed_len(src, src_len, &hdr);
    if (consumed == 0 || hdr != dst_len) return false;
    uint64_t ip = consumed, op = 0;
    while (ip < src_len) {
        const uint8_t tag = src[ip++];
        if ((tag & 3u) == 0) {
            uint64_t len = (tag >> 2) + 1u;
            if (len > 60u) {
                const uint32_t nb = static_cast<uint32_t>(len - 60u);
                if (ip + nb > src_len) return false;
                uint64_t v = 0;
                for (uint32_t k = 0; k < nb; ++k)
                    v |= static_cast<uint64_t>(src[ip + k]) << (8u * k);
                ip += nb;
                len = v + 1u;
            }
            if (ip + len > src_len || op + len > dst_len) return false;
            for (uint64_t k = 0; k < len; ++k) dst[op + k] = src[ip + k];
            ip += len;
            op += len;
            continue;
        }
        uint64_t len = 0, off = 0;
        if ((tag & 3u) == 1) {
            if (ip + 1 > src_len) return false;
            len = ((tag >> 2) & 7u) + 4u;
            off = (static_cast<uint64_t>(tag >> 5) << 8) | src[ip];
            ip += 1;
        } else if ((tag & 3u) == 2) {
            if (ip + 2 > src_len) return false;
            len = (tag >> 2) + 1u;
            off = static_cast<uint64_t>(src[ip]) |
                  (static_cast<uint64_t>(src[ip + 1]) << 8);
            ip += 2;
        } else {
            if (ip + 4 > src_len) return false;
            len = (tag >> 2) + 1u;
            uint32_t o32;
            std::memcpy(&o32, src + ip, 4);
            off = o32;
            ip += 4;
        }
        if (off == 0 || off > op) return false;
        if (op + len > dst_len) return false;
        for (uint64_t k = 0; k < len; ++k) dst[op + k] = dst[op + k - off];
        op += len;
    }
    return op == dst_len;
}

void put_varint(std::vector<uint8_t>& s, uint64_t v) {
    while (v >= 0x80u) { s.push_back(static_cast<uint8_t>(v) | 0x80u); v >>= 7; }
    s.push_back(static_cast<uint8_t>(v));
}

void emit_literal(std::vector<uint8_t>& s, const uint8_t* p, uint64_t len) {
    if (len <= 60) {
        s.push_back(static_cast<uint8_t>((len - 1) << 2));
    } else {
        const uint64_t n = len - 1;
        const uint32_t nb = (n < 256) ? 1u : (n < 65536 ? 2u : 4u);
        s.push_back(static_cast<uint8_t>((60 + nb) << 2));
        for (uint32_t k = 0; k < nb; ++k)
            s.push_back(static_cast<uint8_t>((n >> (8 * k)) & 0xFF));
    }
    s.insert(s.end(), p, p + len);
}

void emit_copy(std::vector<uint8_t>& s, uint64_t off, uint64_t len, int form) {
    if (form == 1 && len >= 4 && len <= 11 && off <= 2047) {
        s.push_back(static_cast<uint8_t>(((off >> 8) << 5) |
                                         ((len - 4) << 2) | 1u));
        s.push_back(static_cast<uint8_t>(off & 0xFF));
    } else if (form == 3) {
        s.push_back(static_cast<uint8_t>(((len - 1) << 2) | 3u));
        for (int k = 0; k < 4; ++k)
            s.push_back(static_cast<uint8_t>((off >> (8 * k)) & 0xFF));
    } else {
        s.push_back(static_cast<uint8_t>(((len - 1) << 2) | 2u));
        s.push_back(static_cast<uint8_t>(off & 0xFF));
        s.push_back(static_cast<uint8_t>((off >> 8) & 0xFF));
    }
}

struct SynthStream {
    std::vector<uint8_t> comp;
    std::vector<uint8_t> raw;
};

// Decode with both, compare byte for byte. Guard bytes past dst_len catch an
// overcopy that happens to land on values the oracle also produced.
void expect_matches_oracle(const SynthStream& s, const char* what) {
    std::vector<uint8_t> a(s.raw.size() + 16, 0xCD);
    std::vector<uint8_t> b(s.raw.size());
    const bool ok_new = snappy_decompress(s.comp.data(), s.comp.size(),
                                          a.data(), s.raw.size());
    const bool ok_ref = naive_decompress(s.comp.data(), s.comp.size(),
                                         b.data(), b.size());
    ASSERT_EQ(ok_new, ok_ref) << what;
    if (!ok_ref) return;
    ASSERT_EQ(b, s.raw) << "oracle disagrees with the generator: " << what;
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), b.size())) << what;
    for (size_t g = s.raw.size(); g < a.size(); ++g)
        ASSERT_EQ(a[g], 0xCD) << "wrote past dst_len: " << what;
}

// A 64-bit LCG rather than <random>: reproducible across stdlib versions, so
// a failure a year from now replays exactly.
struct Rng {
    uint64_t s;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull;
                      return s >> 17; }
};

SynthStream generate(Rng& rng, uint64_t target, bool small_offsets) {
    std::vector<uint8_t> body, raw;
    while (raw.size() < target) {
        const uint64_t left = target - raw.size();
        if (raw.empty() || (rng.next() % 3) == 0) {
            uint64_t len = 1 + rng.next() % 70;
            if (len > left) len = left;
            std::vector<uint8_t> lit(len);
            for (auto& x : lit) x = static_cast<uint8_t>(rng.next() % 4);
            emit_literal(body, lit.data(), len);
            raw.insert(raw.end(), lit.begin(), lit.end());
            continue;
        }
        uint64_t maxoff = raw.size() > 60000 ? 60000 : raw.size();
        const uint64_t cap = small_offsets ? (maxoff < 7 ? maxoff : 7) : maxoff;
        const uint64_t off = 1 + rng.next() % cap;
        uint64_t len = 4 + rng.next() % 61;               // 4..64, per format
        if (len > left) len = left;
        if (len < 4) {                                    // tail too short
            std::vector<uint8_t> lit(left);
            for (auto& x : lit) x = static_cast<uint8_t>(rng.next() % 4);
            emit_literal(body, lit.data(), left);
            raw.insert(raw.end(), lit.begin(), lit.end());
            continue;
        }
        emit_copy(body, off, len, static_cast<int>(rng.next() % 3) + 1);
        for (uint64_t k = 0; k < len; ++k) raw.push_back(raw[raw.size() - off]);
    }
    SynthStream s;
    put_varint(s.comp, raw.size());
    s.comp.insert(s.comp.end(), body.begin(), body.end());
    s.raw = std::move(raw);
    return s;
}

// Sizes straddle the fast-lane guards so matches land in the final bytes.
const uint64_t kSizes[] = {1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
                           63, 64, 65, 127, 200, 1000, 4096, 40000};

TEST(BoltSnappy, DifferentialVsOracleMixedOffsets) {
    Rng rng{0xC0FFEEull};
    for (int i = 0; i < 2000; ++i) {
        const SynthStream s =
            generate(rng, kSizes[i % (sizeof(kSizes) / sizeof(kSizes[0]))],
                     false);
        ASSERT_NO_FATAL_FAILURE(expect_matches_oracle(s, "mixed"))
            << "iteration " << i;
    }
}

TEST(BoltSnappy, DifferentialVsOracleSmallOffsets) {
    // Offsets 1..7 force snappy_copy_match's pattern-growth path on nearly
    // every token; a compressor almost never produces a stream this dense in
    // sub-8 offsets, which is why it is generated rather than sampled.
    Rng rng{0xB1A5ull};
    for (int i = 0; i < 2000; ++i) {
        const SynthStream s =
            generate(rng, kSizes[i % (sizeof(kSizes) / sizeof(kSizes[0]))],
                     true);
        ASSERT_NO_FATAL_FAILURE(expect_matches_oracle(s, "small offsets"))
            << "iteration " << i;
    }
}

TEST(BoltSnappy, AdversarialCopyShapes) {
    struct Case { uint64_t off, len; const char* name; };
    const Case cases[] = {
        {1,  4,  "off=1 len=4"},        {1,  64, "off=1 len=64 max run"},
        {1,  9,  "off=1 crossing 8"},   {2,  64, "off=2 growth"},
        {3,  10, "off=3 non-power-of-2"}, {5, 64, "off=5 growth to 10"},
        {7,  8,  "off=7 growth to 14"}, {8,  64, "off=8 exactly"},
        {9,  9,  "off=9 == len"},       {16, 16, "off=16 len=16"},
        {17, 4,  "off>len short"},      {64, 64, "off=64 len=64"},
    };
    for (const Case& c : cases) {
        // `tail` bytes after the match: below 8 the fast lane must decline.
        for (int tail = 0; tail < 12; ++tail) {
            std::vector<uint8_t> raw, body;
            for (uint64_t i = 0; i < c.off; ++i)
                raw.push_back(static_cast<uint8_t>('a' + (i % 26)));
            emit_literal(body, raw.data(), raw.size());
            for (uint64_t k = 0; k < c.len; ++k)
                raw.push_back(raw[raw.size() - c.off]);
            emit_copy(body, c.off, c.len,
                      (c.off <= 2047 && c.len <= 11) ? 1 : 2);
            if (tail > 0) {
                const std::vector<uint8_t> t(static_cast<size_t>(tail), 0x7E);
                emit_literal(body, t.data(), t.size());
                raw.insert(raw.end(), t.begin(), t.end());
            }
            SynthStream s;
            put_varint(s.comp, raw.size());
            s.comp.insert(s.comp.end(), body.begin(), body.end());
            s.raw = raw;
            ASSERT_NO_FATAL_FAILURE(expect_matches_oracle(s, c.name))
                << " tail=" << tail;
        }
    }
}

TEST(BoltSnappy, TruncatedSynthStreamsAgreeWithOracle) {
    // Every prefix must produce the same verdict as the oracle. A fast lane
    // that reads ahead of its bounds check shows up here.
    Rng rng{7};
    for (int i = 0; i < 24; ++i) {
        const SynthStream s = generate(rng, 200 + (i * 37) % 900, i % 2 == 0);
        for (size_t cut = 1; cut < s.comp.size(); ++cut) {
            std::vector<uint8_t> a(s.raw.size(), 0xAA), b(s.raw.size(), 0xAA);
            const bool ok_new =
                snappy_decompress(s.comp.data(), cut, a.data(), a.size());
            const bool ok_ref =
                naive_decompress(s.comp.data(), cut, b.data(), b.size());
            ASSERT_EQ(ok_new, ok_ref) << "cut " << cut;
            if (ok_new) ASSERT_EQ(a, b) << "cut " << cut;
        }
    }
}

}  // namespace
