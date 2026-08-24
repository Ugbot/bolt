// Emits bolt-compressed streams for scripts/deflate_zlib_check.py.
//
// A separate executable rather than part of the gtest suite: the check that
// matters is what an INDEPENDENT inflater makes of these bytes, and that
// inflater is python's zlib, which cannot be linked into a gtest binary. The
// C++ side's job here is only to produce the streams deterministically.
#include "bolt/ingest/bolt_deflate.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

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

}  // namespace

int main(int argc, char** argv) {
    using namespace bolt::ingest;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out>\n", argv[0]);
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "wb");
    if (f == nullptr) return 2;
    static DeflateState st;
    // Sizes straddle the 65535-byte STORED block limit and the 32 KiB window.
    const std::size_t sizes[] = {0, 1, 2, 3, 7, 64, 255, 1023, 32768, 65535,
                                 65536, 100000, 200000};
    for (int kind = 0; kind < 6; ++kind) {
        for (std::size_t n : sizes) {
            const auto in = make_input(
                kind, n, 0x9E3779B9ull ^ n ^ static_cast<std::uint64_t>(kind));
            std::vector<std::uint8_t> raw(
                static_cast<std::size_t>(deflate_bound(n)));
            std::vector<std::uint8_t> gz(
                static_cast<std::size_t>(gzip_bound(n)));
            std::uint64_t rn = 0, gn = 0;
            if (!deflate_raw_compress(in.data(), n, raw.data(), raw.size(), &rn,
                                      &st)) {
                std::fclose(f);
                return 3;
            }
            if (!gzip_compress(in.data(), n, gz.data(), gz.size(), &gn, &st)) {
                std::fclose(f);
                return 4;
            }
            const std::uint32_t hdr[4] = {
                static_cast<std::uint32_t>(kind),
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(rn),
                static_cast<std::uint32_t>(gn)};
            std::fwrite(hdr, 4, 4, f);
            if (n != 0) std::fwrite(in.data(), 1, n, f);
            std::fwrite(raw.data(), 1, static_cast<std::size_t>(rn), f);
            std::fwrite(gz.data(), 1, static_cast<std::size_t>(gn), f);
        }
    }
    std::fclose(f);
    std::printf("wrote %s\n", argv[1]);
    return 0;
}
