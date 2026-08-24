// Self-contained DEFLATE compressor + GZIP writer.
//
// bolt already INFLATED without a dependency, which is what let the parquet
// reader open a GZIP file on a default build. The other direction had
// nothing: bolt_gzip.h is behind find_package(ZLIB), so bolt could read GZIP
// parquet and never write it.
//
// Round-tripping through bolt's own inflate is necessary and NOT sufficient
// -- a compressor and decompressor that share a misreading of the bit order
// agree perfectly. DEFLATE is especially exposed to that: bits go into bytes
// LSB-first, but a Huffman CODE is transmitted MSB-first, and an encoder that
// gets both wrong in the same direction as its decoder produces a stream only
// it can read. So the real gate is scripts/deflate_zlib_check.py, which hands
// bolt's output to python's zlib.
//
// The other thing a round-trip cannot see is RATIO. A compressor that emits
// every byte as a literal round-trips flawlessly and is useless, and a
// subtler version of that shipped here during development: positions in the
// match finder were stored as uint16, so every hash chain silently pointed at
// garbage once the input passed 65535 bytes. 100 KB of one repeated byte
// compressed to 34 KB instead of 634. Nothing about correctness caught it,
// which is why the ratio assertions below are as specific as they are.

#include "bolt/ingest/bolt_deflate.h"
#include "bolt/ingest/bolt_inflate.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<std::uint32_t>(s >> 32);
    }
};

std::vector<std::uint8_t> make_input(int kind, std::size_t n,
                                     std::uint64_t seed) {
    Rng r{seed ? seed : 1};
    std::vector<std::uint8_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        switch (kind) {
            case 0: v[i] = static_cast<std::uint8_t>(r.next()); break;
            case 1: v[i] = 0x5A; break;
            case 2: v[i] = static_cast<std::uint8_t>(i % 3); break;
            case 3: v[i] = static_cast<std::uint8_t>((i / 64) % 7); break;
            case 4: v[i] = static_cast<std::uint8_t>('a' + (r.next() % 6)); break;
            default:
                v[i] = static_cast<std::uint8_t>((r.next() % 16 == 0)
                                                 ? r.next() : 0xC3);
                break;
        }
    }
    return v;
}

// Compress, inflate with bolt's own decoder, compare. Returns the compressed
// size, or SIZE_MAX on any failure.
std::size_t roundtrip(const std::vector<std::uint8_t>& in, std::string* why) {
    static DeflateState st;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(
        deflate_bound(in.size())));
    std::uint64_t n = 0;
    if (!deflate_raw_compress(in.data(), in.size(), out.data(), out.size(), &n,
                              &st)) {
        *why = "compress failed";
        return static_cast<std::size_t>(-1);
    }
    std::vector<std::uint8_t> back(in.size() ? in.size() : 1);
    std::uint64_t got = 0;
    const int rc = inflate_raw(out.data(), n, back.data(), back.size(), &got);
    if (rc != kInflateOk) { *why = "inflate rejected the stream"; return static_cast<std::size_t>(-1); }
    if (got != in.size()) { *why = "inflate produced the wrong length"; return static_cast<std::size_t>(-1); }
    if (!in.empty() && std::memcmp(back.data(), in.data(), in.size()) != 0) {
        *why = "round trip changed the bytes";
        return static_cast<std::size_t>(-1);
    }
    return static_cast<std::size_t>(n);
}

// ---- correctness ----------------------------------------------------------

TEST(BoltDeflate, RoundTripsAcrossShapesAndSizes) {
    // Sizes straddle the 65535-byte STORED block limit and the 32 KiB window,
    // which are the two places the block/back-reference logic changes.
    const std::size_t sizes[] = {0, 1, 2, 3, 4, 7, 63, 255, 256, 1023,
                                 32767, 32768, 32769, 65534, 65535, 65536,
                                 100000, 200000};
    for (int kind = 0; kind < 6; ++kind) {
        for (std::size_t n : sizes) {
            const auto in = make_input(kind, n, 0x9E3779B9ull ^ n ^
                                       static_cast<std::uint64_t>(kind));
            std::string why;
            SCOPED_TRACE(testing::Message() << "kind=" << kind << " n=" << n);
            EXPECT_NE(roundtrip(in, &why), static_cast<std::size_t>(-1)) << why;
        }
    }
}

TEST(BoltDeflate, MatchesReachBackAcrossTheWindowLateInTheInput) {
    // Pins the uint16-position bug directly. Two copies of a 10 KB
    // incompressible block, both placed PAST 65535 bytes into the input (the
    // point where a 16-bit position wraps) and 24000 apart (inside DEFLATE's
    // 32 KiB window, so the match is representable). With truncated positions
    // the hash chain pointed at garbage there and the second copy was
    // re-emitted verbatim.
    //
    // Note what this test is NOT: a match cannot reach further back than
    // 32768 bytes, so two copies separated by more than the window are
    // legitimately un-matchable. An earlier draft of this test placed them
    // 149000 apart and failed against a perfectly correct compressor.
    std::vector<std::uint8_t> in(200000, 0);
    const auto blk = make_input(0, 10000, 7);      // incompressible
    std::memcpy(in.data() + 100000, blk.data(), blk.size());
    std::memcpy(in.data() + 124000, blk.data(), blk.size());
    std::string why;
    const std::size_t n = roundtrip(in, &why);
    ASSERT_NE(n, static_cast<std::size_t>(-1)) << why;
    // One copy of the noise is incompressible (~10 KB); the other must be
    // matched away, and 190 KB of zeros costs almost nothing. Re-emitting the
    // second copy would put this over 20 KB.
    EXPECT_LT(n, 15000u) << "long-range match was not found: " << n;
}

TEST(BoltDeflate, ActuallyCompresses) {
    std::string why;
    // A long run: this is the case the uint16 position bug destroyed.
    const auto runs = make_input(1, 100000, 1);
    const std::size_t rn = roundtrip(runs, &why);
    ASSERT_NE(rn, static_cast<std::size_t>(-1)) << why;
    EXPECT_LT(rn, 2000u) << "100k of one byte compressed to " << rn;

    const auto periodic = make_input(3, 100000, 2);
    const std::size_t pn = roundtrip(periodic, &why);
    ASSERT_NE(pn, static_cast<std::size_t>(-1)) << why;
    EXPECT_LT(pn, periodic.size() / 10);

    const auto text = make_input(4, 50000, 3);
    const std::size_t tn = roundtrip(text, &why);
    ASSERT_NE(tn, static_cast<std::size_t>(-1)) << why;
    EXPECT_LT(tn, text.size() * 3 / 4) << "low-entropy text barely shrank";
}

TEST(BoltDeflate, IncompressibleInputDoesNotExpandMuch) {
    // Fixed-Huffman EXPANDS random bytes by ~12%, so the encoder must fall
    // back to STORED blocks: 5 bytes of overhead per 64 KiB, not 12%.
    const auto noise = make_input(0, 200000, 5);
    std::string why;
    const std::size_t n = roundtrip(noise, &why);
    ASSERT_NE(n, static_cast<std::size_t>(-1)) << why;
    EXPECT_LE(n, noise.size() + 64u)
        << "random input expanded by " << (n - noise.size()) << " bytes";
}

TEST(BoltDeflate, RespectsTheOutputBound) {
    static DeflateState st;
    const auto in = make_input(4, 20000, 11);
    std::vector<std::uint8_t> full(static_cast<std::size_t>(
        deflate_bound(in.size())));
    std::uint64_t need = 0;
    ASSERT_TRUE(deflate_raw_compress(in.data(), in.size(), full.data(),
                                     full.size(), &need, &st));
    // Every strictly-smaller capacity must be refused, never truncated -- a
    // truncated DEFLATE stream is one an inflater reads as corrupt data
    // rather than as a short read.
    for (std::uint64_t cap = 0; cap < need; cap += 131) {
        std::vector<std::uint8_t> small(static_cast<std::size_t>(cap) + 1);
        std::uint64_t got = 0;
        EXPECT_FALSE(deflate_raw_compress(in.data(), in.size(), small.data(),
                                          cap, &got, &st))
            << "fit into " << cap << ", needs " << need;
    }
}

// ---- GZIP container -------------------------------------------------------

TEST(BoltDeflate, GzipHeaderAndTrailer) {
    static DeflateState st;
    const auto in = make_input(4, 30000, 13);
    std::vector<std::uint8_t> gz(static_cast<std::size_t>(gzip_bound(in.size())));
    std::uint64_t n = 0;
    ASSERT_TRUE(gzip_compress(in.data(), in.size(), gz.data(), gz.size(), &n,
                              &st));
    ASSERT_GT(n, 18u);
    // RFC 1952 header.
    EXPECT_EQ(gz[0], 0x1F);
    EXPECT_EQ(gz[1], 0x8B);
    EXPECT_EQ(gz[2], 8) << "CM must be 8 (deflate)";
    // MTIME is deliberately zero so output is reproducible.
    EXPECT_EQ(gz[4], 0); EXPECT_EQ(gz[5], 0);
    EXPECT_EQ(gz[6], 0); EXPECT_EQ(gz[7], 0);
    // Trailer: CRC32 of the UNCOMPRESSED bytes, then ISIZE.
    std::uint32_t crc = 0, isize = 0;
    std::memcpy(&crc, gz.data() + n - 8, 4);
    std::memcpy(&isize, gz.data() + n - 4, 4);
    EXPECT_EQ(crc, crc32_ieee(in.data(), in.size(), 0));
    EXPECT_EQ(isize, static_cast<std::uint32_t>(in.size()));

    // The body between header and trailer must be a valid raw DEFLATE stream.
    std::vector<std::uint8_t> back(in.size());
    std::uint64_t got = 0;
    EXPECT_EQ(inflate_raw(gz.data() + 10, n - 18u, back.data(), back.size(),
                          &got), kInflateOk);
    EXPECT_EQ(got, in.size());
    EXPECT_EQ(0, std::memcmp(back.data(), in.data(), in.size()));

    // Byte-for-byte reproducible: no timestamp, no nondeterminism.
    std::vector<std::uint8_t> gz2(gz.size());
    std::uint64_t n2 = 0;
    ASSERT_TRUE(gzip_compress(in.data(), in.size(), gz2.data(), gz2.size(),
                              &n2, &st));
    ASSERT_EQ(n2, n);
    EXPECT_EQ(0, std::memcmp(gz.data(), gz2.data(), static_cast<std::size_t>(n)));
}

TEST(BoltDeflate, Crc32MatchesKnownVectors) {
    // The standard CRC-32 check values. A wrong polynomial or a missing
    // final inversion still produces a stable-looking checksum, so these are
    // the only way to know it is the one gzip readers compute.
    EXPECT_EQ(crc32_ieee(reinterpret_cast<const std::uint8_t*>(""), 0, 0),
              0x00000000u);
    EXPECT_EQ(crc32_ieee(reinterpret_cast<const std::uint8_t*>("a"), 1, 0),
              0xE8B7BE43u);
    EXPECT_EQ(crc32_ieee(reinterpret_cast<const std::uint8_t*>("abc"), 3, 0),
              0x352441C2u);
    EXPECT_EQ(crc32_ieee(
        reinterpret_cast<const std::uint8_t*>("123456789"), 9, 0),
        0xCBF43926u);
}

// ---- the gate must discriminate -------------------------------------------

TEST(BoltDeflate, DiscriminatingPower) {
    // Corrupting the stream must be noticed. If inflate accepted anything,
    // every round-trip above would be vacuous.
    static DeflateState st;
    const auto in = make_input(4, 8000, 17);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(
        deflate_bound(in.size())));
    std::uint64_t n = 0;
    ASSERT_TRUE(deflate_raw_compress(in.data(), in.size(), out.data(),
                                     out.size(), &n, &st));
    int noticed = 0, tried = 0;
    for (std::uint64_t i = 0; i < n; i += 29) {
        auto bad = out;
        bad[static_cast<std::size_t>(i)] ^= 0xFF;
        std::vector<std::uint8_t> back(in.size());
        std::uint64_t got = 0;
        const int rc = inflate_raw(bad.data(), n, back.data(), back.size(),
                                   &got);
        const bool same = (rc == kInflateOk) && got == in.size() &&
                          std::memcmp(back.data(), in.data(), in.size()) == 0;
        ++tried;
        noticed += same ? 0 : 1;
    }
    EXPECT_EQ(noticed, tried) << "a corrupted stream inflated to the original";
}

}  // namespace
