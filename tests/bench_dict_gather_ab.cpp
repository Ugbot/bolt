// Interleaved, in-process A/B of the parquet dictionary gather.
//
// This box runs several build+benchmark agents at once (load average >200 on
// 18 cores), so any wall-clock comparison between two separately-built binaries
// measures contention, not code. This driver sidesteps that:
//
//   * BOTH implementations are compiled into ONE translation unit and run
//     back-to-back on the SAME buffers in the SAME process, A/B/A/B..., so any
//     load spike lands on both arms.
//   * The reported figure is ns/value and the A:B ratio, not wall time.
//   * min-of-N is taken per arm, which selects the least-preempted run.
//   * Both arms write to the same output buffer and the result is checksummed
//     after each, so a "faster" arm that produces different bytes is caught.
//
// The index streams are REAL: dumped from row group 0 of the official
// ClickBench hits.parquet via pyarrow dictionary_encode, so the dictionary
// sizes and access locality match production exactly.
//
//   bench_dict_gather_ab <idx.bin> <dict_n> <elem_width> [reps]

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace {

// ---- arm A: the shipped implementation (runtime width) ----------------------
// Verbatim shape of dict_gather before the change: the copy size is a runtime
// value, so the compiler must emit a call to libc memmove per value.
void gather_runtime_width(uint8_t* out, const uint8_t* dict,
                          const uint32_t* idx, uint32_t n,
                          uint32_t w, uint32_t dict_n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t id = idx[i];
        if (id >= dict_n) return;                    // per-value bounds branch
        std::memcpy(out + static_cast<uint64_t>(i) * w,
                    dict + static_cast<uint64_t>(id) * w, w);
    }
}

// ---- arm B: compile-time width ---------------------------------------------
template <uint32_t W>
void gather_const_width(uint8_t* out, const uint8_t* dict,
                        const uint32_t* idx, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        std::memcpy(out + static_cast<uint64_t>(i) * W,
                    dict + static_cast<uint64_t>(idx[i]) * W, W);
    }
}

void gather_const_dispatch(uint8_t* out, const uint8_t* dict,
                           const uint32_t* idx, uint32_t n,
                           uint32_t w, uint32_t dict_n) noexcept {
    uint32_t mx = 0;                                 // one vectorizable pass
    for (uint32_t k = 0; k < n; ++k) mx = (idx[k] > mx) ? idx[k] : mx;
    if (n > 0 && mx >= dict_n) return;
    switch (w) {
        case 8:  gather_const_width<8>(out, dict, idx, n);  break;
        case 4:  gather_const_width<4>(out, dict, idx, n);  break;
        case 16: gather_const_width<16>(out, dict, idx, n); break;
        case 2:  gather_const_width<2>(out, dict, idx, n);  break;
        case 1:  gather_const_width<1>(out, dict, idx, n);  break;
        default: break;
    }
}

uint64_t fnv(const uint8_t* p, uint64_t n) noexcept {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

double now_ns() noexcept {
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <idx.bin> <dict_n> <elem_width> [reps]\n",
                     argv[0]);
        return 2;
    }
    const char* idx_path = argv[1];
    const uint32_t dict_n = static_cast<uint32_t>(std::atoi(argv[2]));
    const uint32_t w      = static_cast<uint32_t>(std::atoi(argv[3]));
    const uint32_t reps   = (argc > 4) ? static_cast<uint32_t>(std::atoi(argv[4])) : 15u;

    std::FILE* f = std::fopen(idx_path, "rb");
    if (f == nullptr) { std::perror("fopen"); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const uint32_t n = static_cast<uint32_t>(bytes / 4);
    uint32_t* idx = static_cast<uint32_t*>(std::malloc(static_cast<size_t>(bytes)));
    if (idx == nullptr || std::fread(idx, 1, static_cast<size_t>(bytes), f) != static_cast<size_t>(bytes)) {
        std::fprintf(stderr, "read idx failed\n"); return 1;
    }
    std::fclose(f);

    uint8_t* dict = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(dict_n) * w + 64));
    uint8_t* out  = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(n) * w + 64));
    if (dict == nullptr || out == nullptr) { std::fprintf(stderr, "oom\n"); return 1; }
    for (uint64_t i = 0; i < static_cast<uint64_t>(dict_n) * w; ++i) {
        dict[i] = static_cast<uint8_t>(i * 31u + 7u);
    }

    double best_a = 1e30, best_b = 1e30;
    uint64_t ck_a = 0, ck_b = 0;
    for (uint32_t r = 0; r < reps; ++r) {
        // A then B, same buffers, same process, adjacent in time.
        std::memset(out, 0, static_cast<size_t>(n) * w);
        double t0 = now_ns();
        gather_runtime_width(out, dict, idx, n, w, dict_n);
        double t1 = now_ns();
        ck_a = fnv(out, static_cast<uint64_t>(n) * w);

        std::memset(out, 0, static_cast<size_t>(n) * w);
        double t2 = now_ns();
        gather_const_dispatch(out, dict, idx, n, w, dict_n);
        double t3 = now_ns();
        ck_b = fnv(out, static_cast<uint64_t>(n) * w);

        const double a = (t1 - t0) / n, b = (t3 - t2) / n;
        if (a < best_a) best_a = a;
        if (b < best_b) best_b = b;
    }

    std::printf("values=%u  dict_n=%u  width=%u\n", n, dict_n, w);
    std::printf("  A runtime-width memcpy : %6.3f ns/value  (min of %u)\n", best_a, reps);
    std::printf("  B const-width  memcpy  : %6.3f ns/value  (min of %u)\n", best_b, reps);
    std::printf("  ratio A/B              : %6.2fx\n", best_a / best_b);
    std::printf("  checksum A=%016" PRIx64 "  B=%016" PRIx64 "  %s\n",
                ck_a, ck_b, (ck_a == ck_b) ? "IDENTICAL" : "*** MISMATCH ***");
    return (ck_a == ck_b) ? 0 : 1;
}
