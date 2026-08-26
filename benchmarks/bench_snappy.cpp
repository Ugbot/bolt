// bench_snappy — compression throughput, isolated from the parquet writer.
//
// The vs-Arrow benchmark can only report the codec's cost by SUBTRACTION
// (snappy encode minus uncompressed encode), which folds in every other
// difference between those two runs. This measures the compressor alone,
// per input class, so a change can be attributed instead of inferred.
//
// Input classes matter more than total throughput here: snappy's cost is
// dominated by whether it finds matches. Incompressible input runs the
// skip-accelerated scan and never extends; highly repetitive input barely
// scans and extends enormously. A change that helps one can hurt the other,
// and a single aggregate number hides that.
//
// Reports MB/s and ratio. Ratio is printed because a "faster" compressor
// that stopped finding matches would otherwise look like a win.

#include "bolt/ingest/bolt_snappy.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using bolt::ingest::snappy_compress;
using bolt::ingest::snappy_decompress;
using bolt::ingest::snappy_max_compressed_len;

constexpr int kRepeats = 7;

struct Rng {
    std::uint64_t s;
    std::uint32_t next() noexcept {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<std::uint32_t>(s >> 32);
    }
};

// The shapes a parquet page actually takes. Named so a regression can be
// pinned to a class rather than to "snappy got slower".
std::vector<std::uint8_t> make_input(int kind, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    Rng r{0x9E3779B97F4A7C15ull ^ static_cast<std::uint64_t>(kind)};
    switch (kind) {
        case 0:                                     // incompressible
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = static_cast<std::uint8_t>(r.next());
            }
            break;
        case 1:                                     // one repeated byte
            std::memset(v.data(), 0x5A, n);
            break;
        case 2: {                                   // low-cardinality strings
            static const char* w[8] = {"alpha", "bravo", "charlie", "delta",
                                       "echo", "foxtrot", "golf", "hotel"};
            std::size_t o = 0;
            while (o < n) {
                const char* s = w[r.next() & 7u];
                const std::size_t l = std::strlen(s);
                const std::size_t c = (o + l <= n) ? l : (n - o);
                std::memcpy(v.data() + o, s, c);
                o += c;
            }
            break;
        }
        case 3:                                     // int64 run, PLAIN
            for (std::size_t i = 0; i + 8 <= n; i += 8) {
                const std::int64_t x = static_cast<std::int64_t>(i / 8) * 3;
                std::memcpy(v.data() + i, &x, 8);
            }
            break;
        case 4:                                     // float64, PLAIN
            for (std::size_t i = 0; i + 8 <= n; i += 8) {
                const double d = static_cast<double>(i / 8) * 0.5;
                std::memcpy(v.data() + i, &d, 8);
            }
            break;
        default: {                                  // high-card Utf8 page
            std::size_t o = 0;
            char b[40];
            while (o < n) {
                const int l = std::snprintf(b, sizeof(b), "id-%08u-%04u",
                                            r.next() & 0xFFFFFFu,
                                            r.next() & 0xFFFu);
                const std::size_t c =
                    (o + static_cast<std::size_t>(l) <= n)
                        ? static_cast<std::size_t>(l) : (n - o);
                std::memcpy(v.data() + o, b, c);
                o += c;
            }
            break;
        }
    }
    return v;
}

}  // namespace

int main() {
    const char* names[6] = {"random", "repeat-byte", "low-card str",
                            "int64 plain", "f64 plain", "utf8 high-card"};
    constexpr std::size_t kN = 4u << 20;            // 4 MiB per class

    std::printf("snappy compression, %zu KiB per class, min-of-%d\n\n",
                kN >> 10, kRepeats);
    std::printf("  %-15s %10s %9s %12s\n", "input", "MB/s", "ratio", "comp(KB)");
    std::printf("  %s\n", "------------------------------------------------");

    double total_in = 0.0, total_s = 0.0;
    for (int kind = 0; kind < 6; ++kind) {
        const auto in = make_input(kind, kN);
        std::vector<std::uint8_t> out(snappy_max_compressed_len(in.size()));
        double best = 1e30;
        std::uint64_t clen = 0;
        for (int rep = 0; rep < kRepeats; ++rep) {
            const auto t0 = Clock::now();
            if (!snappy_compress(in.data(), in.size(), out.data(), out.size(),
                                 &clen)) {
                std::printf("  %-15s COMPRESS FAILED\n", names[kind]);
                return 1;
            }
            const double s =
                std::chrono::duration<double>(Clock::now() - t0).count();
            if (s < best) best = s;
        }
        // A faster compressor that stopped decoding correctly is not faster.
        std::vector<std::uint8_t> back(in.size());
        if (!snappy_decompress(out.data(), clen, back.data(), back.size()) ||
            std::memcmp(back.data(), in.data(), in.size()) != 0) {
            std::printf("  %-15s ROUND-TRIP FAILED\n", names[kind]);
            return 1;
        }
        total_in += static_cast<double>(in.size());
        total_s += best;
        std::printf("  %-15s %10.1f %9.3f %12.1f\n", names[kind],
                    static_cast<double>(in.size()) / best / 1e6,
                    static_cast<double>(in.size()) /
                        static_cast<double>(clen),
                    static_cast<double>(clen) / 1e3);
    }
    std::printf("  %s\n", "------------------------------------------------");
    std::printf("  %-15s %10.1f\n", "aggregate", total_in / total_s / 1e6);
    return 0;
}
