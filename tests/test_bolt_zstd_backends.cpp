// Both zstd implementations behind bolt/ingest/bolt_zstd.h, in one build: every encoder against
// every decoder, hostile inputs on both decoders, and the public interface against its selection.

#include "bolt/ingest/bolt_zstd.h"
#include "bolt_zstd_backends.h"   // private: the subject of this suite

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;
namespace zb = bolt::ingest::zstd_backend;

struct Enc { const char* name; uint64_t (*bound)(uint64_t) noexcept;
             int (*compress)(const uint8_t*, uint64_t, uint8_t*, uint64_t*, int) noexcept; };
struct Dec { const char* name; uint64_t (*scratch)() noexcept;
             int (*decode)(const uint8_t*, uint64_t, uint8_t*, uint64_t, uint64_t*, void*, uint64_t) noexcept; };

std::vector<Enc> encoders() {
    std::vector<Enc> v{{"bolt", zb::self_bound, zb::self_compress}};
    if (zb::ref_built()) v.push_back({"reference", zb::ref_bound, zb::ref_compress});
    return v;
}
std::vector<Dec> decoders() {
    std::vector<Dec> v{{"bolt", zb::self_scratch_size, zb::self_decode}};
    if (zb::ref_built()) v.push_back({"reference", zb::ref_scratch_size, zb::ref_decode});
    return v;
}

uint64_t rng(uint64_t* s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

std::vector<std::vector<uint8_t>> corpora() {
    std::vector<std::vector<uint8_t>> c;
    c.push_back({});                                             // empty
    c.push_back({42});                                           // one byte
    c.push_back(std::vector<uint8_t>(70000, 7));                 // one run
    { std::string t; for (int i = 0; i < 9000; ++i) t += "the quick brown fox " + std::to_string(i % 97);
      c.emplace_back(t.begin(), t.end()); }                      // text
    { std::vector<uint8_t> v(8 * 20000); uint64_t x = 1700000000000ull;
      for (size_t i = 0; i < 20000; ++i) { x += 15000 + (i % 3); std::memcpy(&v[8 * i], &x, 8); } c.push_back(v); }  // sorted int64
    { std::vector<uint8_t> v(300000); uint64_t s = 0x9E3779B97F4A7C15ull;
      for (auto& b : v) b = static_cast<uint8_t>(rng(&s)); c.push_back(v); }                                       // noise, > 1 block
    { std::vector<uint8_t> v(200000); uint64_t s = 99;
      for (auto& b : v) b = static_cast<uint8_t>(rng(&s) % 5); c.push_back(v); }                                    // low entropy
    return c;
}

struct Scratch { std::vector<uint64_t> w; void* p; uint64_t n;
                 explicit Scratch(uint64_t bytes) : w(bytes / 8 + 2), p(w.data()), n(bytes) {} };

TEST(BoltZstdBackends, EveryEncoderAgainstEveryDecoder) {
    int pairs = 0;
    for (const Enc& e : encoders()) for (const Dec& d : decoders()) {
        Scratch sc(d.scratch());
        for (const auto& raw : corpora()) for (int level : {1, 3, 9}) {
            const uint64_t bound = e.bound(raw.size());
            ASSERT_GT(bound, 0u);
            std::vector<uint8_t> comp(bound);
            uint64_t clen = bound;
            ASSERT_EQ(e.compress(raw.data(), raw.size(), comp.data(), &clen, level), kCodecOk) << e.name;
            ASSERT_LE(clen, bound);
            std::vector<uint8_t> back(raw.size() + 1, 0xEE);
            uint64_t got = 0;
            ASSERT_EQ(d.decode(comp.data(), clen, back.data(), raw.size(), &got, sc.p, sc.n), kZstdOk)
                << e.name << " -> " << d.name << " n=" << raw.size() << " level=" << level;
            ASSERT_EQ(got, raw.size());
            ASSERT_EQ(0, std::memcmp(back.data(), raw.data(), raw.size()));
            ASSERT_EQ(back[raw.size()], 0xEE) << "wrote past the output";
        }
        ++pairs;
    }
    EXPECT_GE(pairs, 1);
    if (zb::ref_built()) EXPECT_EQ(pairs, 4);
}

TEST(BoltZstdBackends, StatusCodesAgree) {
    const auto raw = corpora()[3];
    for (const Enc& e : encoders()) for (const Dec& d : decoders()) {
        Scratch sc(d.scratch());
        std::vector<uint8_t> comp(e.bound(raw.size()));
        uint64_t clen = comp.size();
        ASSERT_EQ(e.compress(raw.data(), raw.size(), comp.data(), &clen, 3), kCodecOk);
        std::vector<uint8_t> out(raw.size());
        uint64_t got = 0;
        EXPECT_EQ(d.decode(comp.data(), clen, out.data(), raw.size() / 2, &got, sc.p, sc.n), kZstdNeedMore) << d.name;
        EXPECT_EQ(d.decode(comp.data(), clen, out.data(), out.size(), &got, sc.p, 16), kZstdScratchSmall) << d.name;
        EXPECT_NE(d.decode(comp.data(), clen / 2, out.data(), out.size(), &got, sc.p, sc.n), kZstdOk) << d.name;
        // long enough to hold a frame header: shorter junk is "truncated" to the reference decoder
        uint8_t junk[64];
        for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = static_cast<uint8_t>(i + 1);
        EXPECT_EQ(d.decode(junk, sizeof(junk), out.data(), out.size(), &got, sc.p, sc.n), kZstdBadInput) << d.name;
    }
}

// Truncations and single-byte flips: a decoder errors or returns a size within the buffer; never
// writes outside it (ASan / the guard byte).
TEST(BoltZstdBackends, HostileInputsBothDecoders) {
    const auto raws = corpora();
    uint64_t seed = 0xC0FFEE, cases = 0;
    for (const Enc& e : encoders()) for (const Dec& d : decoders()) {
        Scratch sc(d.scratch());
        for (size_t ci = 2; ci < raws.size(); ++ci) {
            const auto& raw = raws[ci];
            std::vector<uint8_t> comp(e.bound(raw.size()));
            uint64_t clen = comp.size();
            ASSERT_EQ(e.compress(raw.data(), raw.size(), comp.data(), &clen, 3), kCodecOk);
            std::vector<uint8_t> out(raw.size() + 1, 0xEE);
            for (int it = 0; it < 400; ++it) {                    // bounded
                std::vector<uint8_t> bad(comp.begin(), comp.begin() + static_cast<long>(clen));
                if (it % 2 == 0) bad.resize(rng(&seed) % (clen + 1));
                else bad[rng(&seed) % clen] ^= static_cast<uint8_t>(1u << (rng(&seed) % 8));
                uint64_t got = 0;
                const int rc = d.decode(bad.data(), bad.size(), out.data(), raw.size(), &got, sc.p, sc.n);
                if (rc == kZstdOk) ASSERT_LE(got, raw.size());
                ASSERT_EQ(out[raw.size()], 0xEE) << d.name;
                ++cases;
            }
        }
    }
    EXPECT_GE(cases, 2000u);
}

TEST(BoltZstdBackends, InterfaceIsItsSelection) {
    const std::string enc = zstd_encoder_name(), dec = zstd_decoder_name();
    EXPECT_TRUE(enc == "reference" || enc == "bolt");
    EXPECT_TRUE(dec == "reference" || dec == "bolt");
    EXPECT_STREQ(zstd_stream_decoder_name(), "bolt");
    if (!zb::ref_built()) { EXPECT_EQ(enc, "bolt"); EXPECT_EQ(dec, "bolt"); }
    EXPECT_TRUE(zstd_available());
    EXPECT_EQ(zstd_decode_scratch_size(), dec == "reference" ? zb::ref_scratch_size() : zb::self_scratch_size());
    const auto raw = corpora()[4];
    EXPECT_EQ(zstd_max_compressed_len(raw.size()), enc == "reference" ? zb::ref_bound(raw.size()) : zb::self_bound(raw.size()));
    // the interface's bytes are its selected encoder's bytes
    std::vector<uint8_t> a(zstd_max_compressed_len(raw.size())), b(a.size());
    uint64_t an = a.size(), bn = b.size();
    ASSERT_EQ(zstd_compress(raw.data(), raw.size(), a.data(), &an, 3), kCodecOk);
    ASSERT_EQ((enc == "reference" ? zb::ref_compress : zb::self_compress)(raw.data(), raw.size(), b.data(), &bn, 3), kCodecOk);
    ASSERT_EQ(an, bn);
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), an));
    std::vector<uint8_t> back(raw.size());
    ASSERT_EQ(zstd_decompress(a.data(), an, back.data(), back.size()), kCodecOk);
    EXPECT_EQ(0, std::memcmp(back.data(), raw.data(), raw.size()));
    EXPECT_EQ(zstd_decompress(a.data(), an, back.data(), back.size() - 1), kCodecCorrupt);
}

}  // namespace
