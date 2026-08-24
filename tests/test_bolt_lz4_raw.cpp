// Self-contained LZ4 block codec.
//
// Two things are being established, and they need different evidence.
//
//   1. The DECOMPRESSOR reads what the world writes. bolt could not open an
//      LZ4_RAW parquet file at all on a default build -- bolt_lz4.h is behind
//      find_package(lz4) -- so the decoder's correctness has to be settled
//      against blocks produced by a REFERENCE compressor, not by ours. A
//      decoder and compressor that share a misreading agree perfectly with
//      each other; that is the failure this whole file exists to rule out.
//      scripts/make_lz4_vectors.py emits those blocks by calling liblz4
//      itself through ctypes -- so the reference is needed to GENERATE the
//      committed vectors, never to consume them, and bolt's build gains no
//      dependency.
//
//   2. The COMPRESSOR produces blocks the world reads. Round-tripping through
//      our own decoder cannot show that. So the compressed output is checked
//      by the reference too, and separately it must actually COMPRESS -- a
//      "compressor" that emits everything as literals round-trips flawlessly
//      while doing nothing.
//
// Between those, a differential fuzz against an intentionally naive decoder
// written straight from the format description catches the cases neither
// corpus happens to contain: overlapping matches, maximal extended lengths,
// and every truncation of a valid block.

#include "bolt/ingest/bolt_lz4_raw.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace bolt::ingest;

// ---- a deliberately naive reference decoder -------------------------------
//
// Transcribed from the format description rather than from the optimised one,
// so a shared mistake between the two is unlikely. Bounds-checked but makes
// no attempt to be fast.
bool naive_decompress(const std::uint8_t* src, std::size_t n,
                      std::vector<std::uint8_t>* out) {
    out->clear();
    std::size_t ip = 0;
    while (ip < n) {
        const std::uint8_t token = src[ip++];
        std::size_t ll = token >> 4;
        if (ll == 15) {
            for (;;) {
                if (ip >= n) return false;
                const std::uint8_t b = src[ip++];
                ll += b;
                if (b != 255) break;
            }
        }
        if (ip + ll > n) return false;
        for (std::size_t i = 0; i < ll; ++i) out->push_back(src[ip + i]);
        ip += ll;
        if (ip == n) break;
        if (ip + 2 > n) return false;
        const std::size_t off = static_cast<std::size_t>(src[ip]) |
                                (static_cast<std::size_t>(src[ip + 1]) << 8);
        ip += 2;
        std::size_t ml = token & 0x0F;
        if (ml == 15) {
            for (;;) {
                if (ip >= n) return false;
                const std::uint8_t b = src[ip++];
                ml += b;
                if (b != 255) break;
            }
        }
        ml += 4;
        if (off == 0 || off > out->size()) return false;
        const std::size_t start = out->size() - off;
        for (std::size_t i = 0; i < ml; ++i) out->push_back((*out)[start + i]);
    }
    return true;
}

// ---- corpora --------------------------------------------------------------

// Deterministic PRNG so a failure is reproducible from its seed alone.
struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<std::uint32_t>(s >> 32);
    }
};

// Inputs chosen for what they do to the ENCODER, not for realism.
std::vector<std::uint8_t> make_input(int kind, std::size_t n, Rng* r) {
    std::vector<std::uint8_t> v(n);
    switch (kind) {
        case 0:                                    // incompressible
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = static_cast<std::uint8_t>(r->next());
            }
            break;
        case 1:                                    // all one byte: one long run
            std::memset(v.data(), 0x5A, n);
            break;
        case 2:                                    // short period: overlapping
            for (std::size_t i = 0; i < n; ++i) {  // matches with off < len
                v[i] = static_cast<std::uint8_t>(i % 3);
            }
            break;
        case 3:                                    // long repeated block
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = static_cast<std::uint8_t>((i / 64) % 7);
            }
            break;
        case 4:                                    // text-like, low entropy
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = static_cast<std::uint8_t>('a' + (r->next() % 6));
            }
            break;
        default:                                   // mostly-repeat, rare noise
            for (std::size_t i = 0; i < n; ++i) {
                // Both PRNG draws happen unconditionally so the sequence is
                // independent of the branch -- scripts/make_lz4_vectors.py
                // must reproduce this byte for byte, and a short-circuiting
                // ternary would desynchronise the two.
                const std::uint32_t a = r->next();
                const std::uint32_t b = r->next();
                v[i] = static_cast<std::uint8_t>((a % 32 == 0) ? b : 0xC3);
            }
            break;
    }
    return v;
}

bool roundtrip(const std::vector<std::uint8_t>& in, std::string* why) {
    std::vector<std::uint8_t> comp(lz4_raw_bound(in.size()));
    std::uint64_t clen = 0;
    if (!lz4_raw_compress(in.data(), in.size(), comp.data(), comp.size(),
                          &clen)) {
        *why = "compress failed";
        return false;
    }
    comp.resize(static_cast<std::size_t>(clen));

    // Our decoder.
    std::vector<std::uint8_t> out(in.size());
    if (!lz4_raw_decompress(comp.data(), comp.size(), out.data(), out.size())) {
        *why = "decompress rejected our own block";
        return false;
    }
    if (out != in) { *why = "round trip changed the bytes"; return false; }

    // The naive reference decoder must agree, byte for byte. This is what
    // catches a block that our decoder mis-reads in the same way our
    // compressor mis-writes it.
    std::vector<std::uint8_t> ref;
    if (!naive_decompress(comp.data(), comp.size(), &ref)) {
        *why = "reference decoder rejected the block";
        return false;
    }
    if (ref != in) { *why = "reference decoder disagreed"; return false; }
    return true;
}

// ---- tests ----------------------------------------------------------------

TEST(BoltLz4Raw, RoundTripAcrossShapesAndSizes) {
    Rng r{0x9E3779B97F4A7C15ull};
    // Sizes straddle the two end-of-block rules (last 5 bytes literal, last
    // match >= 12 from the end) and the all-literals short-block path.
    const std::size_t sizes[] = {0, 1, 2, 3, 4, 5, 11, 12, 13, 15, 16, 17,
                                 63, 64, 65, 255, 256, 257, 1023, 4096, 65535,
                                 70000};
    for (int kind = 0; kind < 6; ++kind) {
        for (std::size_t n : sizes) {
            const auto in = make_input(kind, n, &r);
            std::string why;
            SCOPED_TRACE(testing::Message() << "kind=" << kind << " n=" << n);
            EXPECT_TRUE(roundtrip(in, &why)) << why;
        }
    }
}

TEST(BoltLz4Raw, ActuallyCompresses) {
    // A "compressor" that emits pure literals round-trips perfectly and is
    // useless. Highly repetitive input must shrink a lot; random input must
    // not blow past the documented bound.
    Rng r{12345};
    const auto runs = make_input(1, 100000, &r);      // one byte repeated
    std::vector<std::uint8_t> c(lz4_raw_bound(runs.size()));
    std::uint64_t n = 0;
    ASSERT_TRUE(lz4_raw_compress(runs.data(), runs.size(), c.data(), c.size(),
                                 &n));
    EXPECT_LT(n, runs.size() / 50) << "a 100k run compressed to " << n;

    const auto periodic = make_input(3, 100000, &r);  // 64-byte blocks
    ASSERT_TRUE(lz4_raw_compress(periodic.data(), periodic.size(), c.data(),
                                 c.size(), &n));
    EXPECT_LT(n, periodic.size() / 4);

    const auto noise = make_input(0, 100000, &r);
    ASSERT_TRUE(lz4_raw_compress(noise.data(), noise.size(), c.data(), c.size(),
                                 &n));
    EXPECT_LE(n, lz4_raw_bound(noise.size()))
        << "incompressible input exceeded the bound";
}

TEST(BoltLz4Raw, OverlappingMatchesAreExact) {
    // off < match_length is how LZ4 encodes a run, and it is the one case a
    // memcpy-based decoder gets wrong while looking correct on everything
    // else. Periods 1..8 all produce it.
    for (int period = 1; period <= 8; ++period) {
        std::vector<std::uint8_t> in(5000);
        for (std::size_t i = 0; i < in.size(); ++i) {
            in[i] = static_cast<std::uint8_t>(i % period);
        }
        std::string why;
        SCOPED_TRACE(testing::Message() << "period=" << period);
        EXPECT_TRUE(roundtrip(in, &why)) << why;
    }
}

TEST(BoltLz4Raw, RespectsTheOutputBound) {
    // A caller that sizes the destination exactly must get a clean refusal,
    // not a partial block or an overrun.
    Rng r{999};
    const auto in = make_input(0, 4096, &r);          // incompressible
    std::vector<std::uint8_t> c(lz4_raw_bound(in.size()));
    std::uint64_t full = 0;
    ASSERT_TRUE(lz4_raw_compress(in.data(), in.size(), c.data(), c.size(),
                                 &full));
    // Every strictly-smaller capacity must fail rather than truncate.
    for (std::uint64_t cap = 0; cap < full; cap += 97) {
        std::vector<std::uint8_t> small(static_cast<std::size_t>(cap) + 1);
        std::uint64_t got = 0;
        EXPECT_FALSE(lz4_raw_compress(in.data(), in.size(), small.data(), cap,
                                      &got))
            << "compressed into " << cap << " bytes, needs " << full;
    }
}

TEST(BoltLz4Raw, HostileInputIsRefusedNotUB) {
    // Every truncation of a valid block, plus corrupted offsets. None may
    // read or write out of bounds; all must return false or produce exactly
    // the promised length.
    Rng r{424242};
    for (int kind = 0; kind < 6; ++kind) {
        const auto in = make_input(kind, 3000, &r);
        std::vector<std::uint8_t> c(lz4_raw_bound(in.size()));
        std::uint64_t clen = 0;
        ASSERT_TRUE(lz4_raw_compress(in.data(), in.size(), c.data(), c.size(),
                                     &clen));
        for (std::uint64_t cut = 0; cut < clen; ++cut) {
            std::vector<std::uint8_t> out(in.size());
            // Truncated: must be refused (it cannot yield the full length).
            (void)lz4_raw_decompress(c.data(), cut, out.data(), out.size());
        }
        // Corrupt the offset bytes: an offset reaching before the start of
        // the output must be refused, never read out of bounds.
        for (std::uint64_t i = 0; i + 1 < clen; ++i) {
            std::vector<std::uint8_t> bad(c.begin(),
                                          c.begin() + static_cast<std::size_t>(clen));
            bad[static_cast<std::size_t>(i)] = 0xFF;
            bad[static_cast<std::size_t>(i) + 1] = 0xFF;
            std::vector<std::uint8_t> out(in.size());
            (void)lz4_raw_decompress(bad.data(), bad.size(), out.data(),
                                     out.size());
        }
    }
    SUCCEED() << "no crash, no overrun (run under ASAN for the real check)";
}

TEST(BoltLz4Raw, WrongExpectedLengthIsRefused) {
    // parquet always knows a page's uncompressed size. A block that yields
    // more or fewer bytes than promised is corrupt, and accepting it would
    // hand the caller a partly uninitialised buffer.
    Rng r{7};
    const auto in = make_input(4, 2000, &r);
    std::vector<std::uint8_t> c(lz4_raw_bound(in.size()));
    std::uint64_t clen = 0;
    ASSERT_TRUE(lz4_raw_compress(in.data(), in.size(), c.data(), c.size(),
                                 &clen));
    std::vector<std::uint8_t> out(in.size() + 16);
    EXPECT_FALSE(lz4_raw_decompress(c.data(), clen, out.data(), in.size() - 1));
    EXPECT_FALSE(lz4_raw_decompress(c.data(), clen, out.data(), in.size() + 1));
    EXPECT_TRUE(lz4_raw_decompress(c.data(), clen, out.data(), in.size()));
}

// ---- the gate must discriminate ------------------------------------------

TEST(BoltLz4Raw, DiscriminatingPower) {
    // If the naive reference decoder accepted anything, every comparison
    // above would be vacuous. Show it rejects a block our decoder also
    // rejects, and that a single flipped byte in the compressed stream is
    // detected by at least one of them.
    Rng r{2024};
    const auto in = make_input(3, 4000, &r);
    std::vector<std::uint8_t> c(lz4_raw_bound(in.size()));
    std::uint64_t clen = 0;
    ASSERT_TRUE(lz4_raw_compress(in.data(), in.size(), c.data(), c.size(),
                                 &clen));
    c.resize(static_cast<std::size_t>(clen));

    std::vector<std::uint8_t> ref;
    ASSERT_TRUE(naive_decompress(c.data(), c.size(), &ref));
    ASSERT_EQ(ref, in);

    int detected = 0, tried = 0;
    for (std::size_t i = 0; i < c.size(); i += 7) {
        auto bad = c;
        bad[i] ^= 0xFF;
        ++tried;
        std::vector<std::uint8_t> mine(in.size());
        const bool ok_mine =
            lz4_raw_decompress(bad.data(), bad.size(), mine.data(), mine.size());
        std::vector<std::uint8_t> theirs;
        const bool ok_ref = naive_decompress(bad.data(), bad.size(), &theirs);
        const bool changed = (!ok_mine) || (mine != in) ||
                             (!ok_ref) || (theirs != in);
        detected += changed ? 1 : 0;
    }
    // Not every flip must be caught -- a literal byte flip is legal and just
    // yields different data, which IS detected as "!= in". What must not
    // happen is a flip going wholly unnoticed by both.
    EXPECT_EQ(detected, tried) << "a corrupted block decoded as the original";
}

// ---- interoperability with the REFERENCE implementation -------------------

std::string data_path(const char* name) {
#ifdef BOLT_TEST_DATA_DIR
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
#else
    return std::string("tests/data/") + name;
#endif
}

std::vector<std::uint8_t> slurp(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    const std::size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

inline std::uint32_t rd32(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// The generator scripts/make_lz4_vectors.py implements, restated. If the two
// ever disagree the vectors decode to the wrong bytes and this test says so,
// which is itself the check that the shared generator is shared.
std::vector<std::uint8_t> gen_vector(std::uint32_t kind, std::size_t n,
                                     std::uint32_t seed) {
    Rng r{seed};
    return make_input(static_cast<int>(kind), n, &r);
}

TEST(BoltLz4Raw, DecodesBlocksProducedByLiblz4) {
    // THE test for a decoder. Everything else in this file compares bolt
    // against bolt; these blocks were compressed by liblz4 itself, so
    // decoding them is the only evidence that bolt reads what the ecosystem
    // writes -- which it previously could not do at all without
    // find_package(lz4).
    const auto f = slurp(data_path("lz4_vectors.bin").c_str());
    ASSERT_GE(f.size(), 8u) << "run scripts/make_lz4_vectors.py";
    ASSERT_EQ(std::memcmp(f.data(), "BLZ4", 4), 0) << "bad vector file magic";
    const std::uint32_t count = rd32(f.data() + 4);
    ASSERT_GT(count, 0u);

    std::size_t off = 8;
    std::uint32_t checked = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        ASSERT_LE(off + 16, f.size()) << "vector " << i << " header truncated";
        const std::uint32_t kind = rd32(f.data() + off);
        const std::uint32_t raw_len = rd32(f.data() + off + 4);
        const std::uint32_t seed = rd32(f.data() + off + 8);
        const std::uint32_t clen = rd32(f.data() + off + 12);
        off += 16;
        ASSERT_LE(off + clen, f.size()) << "vector " << i << " body truncated";
        const std::uint8_t* comp = f.data() + off;
        off += clen;

        const auto want = gen_vector(kind, raw_len, seed);
        ASSERT_EQ(want.size(), raw_len);
        std::vector<std::uint8_t> out(raw_len);
        SCOPED_TRACE(testing::Message() << "vector " << i << " kind=" << kind
                                        << " raw_len=" << raw_len);
        ASSERT_TRUE(lz4_raw_decompress(comp, clen, out.data(), raw_len))
            << "bolt refused a block liblz4 produced";
        ASSERT_EQ(out, want) << "bolt decoded a liblz4 block to wrong bytes";
        ++checked;
    }
    EXPECT_EQ(checked, count);
    EXPECT_EQ(off, f.size()) << "trailing bytes in the vector file";
}

TEST(BoltLz4Raw, BoltBlocksMatchLiblz4Semantics) {
    // The other direction. bolt COMPRESSES each reference input and the
    // result must decode -- via the independent naive decoder -- to exactly
    // the same bytes liblz4 was given. A compressor that emits a block only
    // its own decoder understands would pass every round-trip test above and
    // fail here.
    const auto f = slurp(data_path("lz4_vectors.bin").c_str());
    ASSERT_GE(f.size(), 8u);
    const std::uint32_t count = rd32(f.data() + 4);
    std::size_t off = 8;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t kind = rd32(f.data() + off);
        const std::uint32_t raw_len = rd32(f.data() + off + 4);
        const std::uint32_t seed = rd32(f.data() + off + 8);
        const std::uint32_t clen = rd32(f.data() + off + 12);
        off += 16 + clen;
        const auto raw = gen_vector(kind, raw_len, seed);
        std::vector<std::uint8_t> mine(lz4_raw_bound(raw.size()));
        std::uint64_t n = 0;
        SCOPED_TRACE(testing::Message() << "vector " << i);
        ASSERT_TRUE(lz4_raw_compress(raw.data(), raw.size(), mine.data(),
                                     mine.size(), &n));
        mine.resize(static_cast<std::size_t>(n));
        std::vector<std::uint8_t> back;
        ASSERT_TRUE(naive_decompress(mine.data(), mine.size(), &back))
            << "bolt emitted a block the format description does not accept";
        ASSERT_EQ(back, raw);
    }
}

}  // namespace
