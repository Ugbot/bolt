// Emits bolt-compressed snappy blocks for scripts/snappy_ref_check.py.
//
// Separate from the gtest suite because the check that matters is what a
// REFERENCE snappy makes of these bytes, and that reference is pyarrow's
// codec, which cannot be linked into a gtest binary.
#include "bolt/ingest/bolt_snappy.h"

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

// Deliberately compressible shapes -- the point is to prove back-references
// are emitted, which an incompressible corpus could not show.
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
    // Sizes straddle snappy's 64 KiB block boundary and the 60-byte literal
    // tag threshold.
    const std::size_t sizes[] = {0, 1, 2, 15, 16, 59, 60, 61, 255, 256, 1023,
                                 65535, 65536, 65537, 200000, 500000};
    for (int kind = 0; kind < 6; ++kind) {
        for (std::size_t n : sizes) {
            const auto in = make_input(
                kind, n, 0x9E3779B9ull ^ n ^ static_cast<std::uint64_t>(kind));
            std::vector<std::uint8_t> comp(
                static_cast<std::size_t>(snappy_max_compressed_len(n)));
            std::uint64_t cn = 0;
            if (!snappy_compress(in.data(), n, comp.data(), comp.size(), &cn)) {
                std::fclose(f);
                std::fprintf(stderr, "compress failed kind=%d n=%zu\n", kind, n);
                return 3;
            }
            const std::uint32_t hdr[3] = {
                static_cast<std::uint32_t>(kind),
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(cn)};
            std::fwrite(hdr, 4, 3, f);
            if (n != 0) std::fwrite(in.data(), 1, n, f);
            std::fwrite(comp.data(), 1, static_cast<std::size_t>(cn), f);
        }
    }
    std::fclose(f);
    std::printf("wrote %s\n", argv[1]);
    return 0;
}
