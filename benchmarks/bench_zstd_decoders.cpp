// bench_zstd_decoders — libzstd's decoder vs bolt's (bolt_zstd_dec.h) on the SAME libzstd-produced frames.
//
// bench_zstd_decoders <file> <chunk_bytes> [reps]
// Each chunk of the file is one frame per level (1, 3, 9). Decoders alternate per rep (interleaved, 1 thread);
// the best rep counts. Reports output MB/s and instructions per output byte (proc_pid_rusage on macOS, else 0).

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <zstd.h>

#include "bolt_zstd_dec.h"   // private: bolt's own decoder

#if defined(__APPLE__)
#include <libproc.h>
#include <unistd.h>
#endif

namespace {

uint64_t instructions() noexcept {
#if defined(__APPLE__)
    rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&ri)) != 0) return 0;
    return ri.ri_instructions;
#else
    return 0;
#endif
}

double now_s() noexcept {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Frame { uint64_t src_off, src_len, enc_off, enc_len; };

struct Result { double best_s = 1e30; uint64_t best_ins = UINT64_MAX; };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <file> <chunk_bytes> [reps]\n", argv[0]); return 2; }
    const uint64_t chunk = std::strtoull(argv[2], nullptr, 10);
    const int reps = argc > 3 ? std::atoi(argv[3]) : 7;
    if (chunk == 0 || chunk > (64u << 20) || reps < 1 || reps > 1000) return 2;
    std::FILE* f = std::fopen(argv[1], "rb");
    if (f == nullptr) { std::perror("open"); return 1; }
    std::fseek(f, 0, SEEK_END);
    const uint64_t n = static_cast<uint64_t>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> raw(n);
    if (std::fread(raw.data(), 1, n, f) != n) return 1;
    std::fclose(f);

    std::vector<uint8_t> out(chunk + 64), scratch(bolt::ingest::zstd_scratch_size() + 64);
    void* sc = scratch.data() + ((64 - (reinterpret_cast<uintptr_t>(scratch.data()) & 63)) & 63);
    std::printf("file %s bytes %" PRIu64 " chunk %" PRIu64 " reps %d\n", argv[1], n, chunk, reps);
    std::printf("level ratio  lib_MB/s bolt_MB/s bolt/lib_time  lib_ins/B bolt_ins/B bolt/lib_ins\n");
    for (const int level : {1, 3, 9}) {
        std::vector<uint8_t> enc(ZSTD_compressBound(chunk) * ((n + chunk - 1) / chunk));
        std::vector<Frame> frames;
        uint64_t eo = 0;
        for (uint64_t o = 0; o < n; o += chunk) {                 // bounded: file / chunk
            const uint64_t len = n - o < chunk ? n - o : chunk;
            const size_t r = ZSTD_compress(enc.data() + eo, enc.size() - eo, raw.data() + o, len, level);
            if (ZSTD_isError(r)) return 1;
            frames.push_back({o, len, eo, r});
            eo += r;
        }
        Result lib, own;
        for (int rep = 0; rep < reps; ++rep) {                    // bounded: reps
            for (int which = 0; which < 2; ++which) {             // interleaved
                const bool use_lib = (which ^ (rep & 1)) == 0;
                const uint64_t i0 = instructions();
                const double t0 = now_s();
                for (const Frame& fr : frames) {
                    if (use_lib) {
                        const size_t r = ZSTD_decompress(out.data(), fr.src_len, enc.data() + fr.enc_off, fr.enc_len);
                        if (r != fr.src_len) { std::fprintf(stderr, "libzstd decode failed\n"); return 1; }
                    } else {
                        uint64_t got = 0;
                        const int e = bolt::ingest::zstd_decode_raw(enc.data() + fr.enc_off, fr.enc_len, out.data(),
                                                                    fr.src_len, &got, sc, bolt::ingest::zstd_scratch_size());
                        if (e != 0 || got != fr.src_len) { std::fprintf(stderr, "bolt decode failed %d\n", e); return 1; }
                    }
                    if (rep == 0 && std::memcmp(out.data(), raw.data() + fr.src_off, fr.src_len) != 0) {
                        std::fprintf(stderr, "round trip mismatch\n");
                        return 1;
                    }
                }
                const double dt = now_s() - t0;
                const uint64_t di = instructions() - i0;
                Result& r = use_lib ? lib : own;
                if (rep > 0 && dt < r.best_s) r.best_s = dt;      // rep 0 pays the memcmp
                if (rep > 0 && di < r.best_ins) r.best_ins = di;
            }
        }
        const double mb = static_cast<double>(n) / 1e6;
        std::printf("%5d %5.2f %9.0f %9.0f %13.2f %10.2f %10.2f %12.2f\n", level,
                    static_cast<double>(n) / static_cast<double>(eo), mb / lib.best_s, mb / own.best_s,
                    own.best_s / lib.best_s, static_cast<double>(lib.best_ins) / static_cast<double>(n),
                    static_cast<double>(own.best_ins) / static_cast<double>(n),
                    static_cast<double>(own.best_ins) / static_cast<double>(lib.best_ins));
    }
    return 0;
}
