// Emits bolt-compressed zstd frames for scripts/zstd_ref_check.py.
//
// Separate from the gtest suite because the only evidence that settles a zstd
// encoder is what the REFERENCE libzstd makes of its frames, and libzstd is
// reached through ctypes on the python side rather than linked here.
#include "bolt/ingest/bolt_zstd_enc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace bolt::ingest;
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <out>\n", argv[0]); return 2; }
    std::FILE* f = std::fopen(argv[1], "wb");
    void* mem = std::malloc((size_t)zstd_enc_state_size());
    ZstdEncState* st = zstd_enc_state_init(mem, zstd_enc_state_size());
    // 6 is a "parquet PLAIN Utf8 page" shape: 4-byte length prefixes and
    // high-cardinality strings. It both COMPRESSES and carries literal runs
    // past 4096, which is the only combination that reaches the 3-byte
    // literals-size header -- the path that shipped broken because the
    // earlier corpus had compressible-with-tiny-literals and
    // incompressible-stored-raw, but nothing with both.
    const size_t sizes[] = {0,1,2,15,100,1000,65536,200000,1020000};
    for (int kind = 0; kind < 6; ++kind) for (size_t n : sizes) {
        std::vector<uint8_t> in(n);
        uint64_t s = 0x9E3779B9ull ^ (uint64_t)(kind*7919) ^ n;
        for (size_t i = 0; i < n; ++i) {
            s ^= s<<13; s ^= s>>7; s ^= s<<17;
            if (kind==0) in[i]=(uint8_t)(s>>32);
            else if (kind==1) in[i]=0x5A;
            else if (kind==2) in[i]=(uint8_t)(i%3);
            else if (kind==3) in[i]=(uint8_t)((i/64)%7);
            else if (kind==4) in[i]=(uint8_t)('a'+((s>>32)%6));
            else in[i]=0;   // kind 5 filled below
        }
        if (kind == 5) {                 // PLAIN Utf8 page shape
            in.clear();
            for (size_t i = 0; in.size() < n; ++i) {
                char x[64];
                const int L = std::snprintf(x, 64, "id-%08zu-x", i);
                const uint32_t u = (uint32_t)L;
                in.insert(in.end(), (uint8_t*)&u, (uint8_t*)&u + 4);
                in.insert(in.end(), x, x + L);
            }
            in.resize(n);
        }
        std::vector<uint8_t> out(zstd_enc_bound(n));
        uint64_t on = 0;
        if (!zstd_compress_self(in.data(), n, out.data(), out.size(), &on, st)) {
            std::printf("compress FAILED kind=%d n=%zu\n", kind, n); return 2;
        }
        uint32_t hdr[3] = {(uint32_t)kind,(uint32_t)n,(uint32_t)on};
        std::fwrite(hdr,4,3,f);
        if (n) std::fwrite(in.data(),1,n,f);
        std::fwrite(out.data(),1,on,f);
    }
    std::fclose(f);
    std::printf("wrote streams\n");
    return 0;
}
