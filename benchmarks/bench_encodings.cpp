// bench_encodings — scalar decode ns/value of the FastLanes formats
// (BitPacked / FrameOfRef / DeltaFOR, MSEG B1) at 1K..64K values, plus
// decode-on-access (fastlanes_value) for a view. Scalar reference only; the
// NEON / AVX2 kernels are M4.
//
// Usage: bench_encodings [runs=5] [min_ms_per_run=20]
// Prints one row per (format, T, W, n): median ns/value over `runs`.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_fastlanes.h"

namespace fl = bolt::fastlanes;
using Clock = std::chrono::steady_clock;

namespace {

volatile uint64_t g_sink = 0;

template <class F>
double median_ns_per_value(F&& f, int64_t values_per_call, int runs, double min_ms) {
    std::vector<double> r;
    for (int k = 0; k < runs; ++k) {
        int64_t calls = 0;
        const auto t0 = Clock::now();
        double ms = 0;
        while (ms < min_ms && calls < (int64_t(1) << 30)) {
            f();
            ++calls;
            ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }
        r.push_back(ms * 1e6 / double(calls * values_per_call));
    }
    std::sort(r.begin(), r.end());
    return r[r.size() / 2];
}

template <class V>
std::vector<V> data_for(int fmt, uint32_t w, int64_t n, std::mt19937_64& g) {
    using U = std::make_unsigned_t<V>;
    const uint64_t m = w >= 64 ? ~0ull : ((1ull << w) - 1);
    std::vector<V> v(static_cast<size_t>(n));
    U acc = 1000;
    for (int64_t i = 0; i < n; ++i) {
        if (fmt == 2) acc = U(acc + 1 + (g() & m));   // DeltaFOR: sorted, deltas < 2^w
        else acc = U(5000 + (g() & m));                // FOR: values in [5000, 5000 + 2^w)
        v[size_t(i)] = V(acc);
    }
    return v;
}

template <class V>
void bench_one(int fmt, uint32_t w, int64_t n, int runs, double min_ms) {
    constexpr uint32_t T = sizeof(V) * 8;
    std::mt19937_64 g(42u + static_cast<uint64_t>(n) + w);
    const std::vector<V> v = data_for<V>(fmt, w, n, g);
    int64_t ref = 0;
    uint32_t cw = 0;
    std::vector<uint8_t> buf;
    if (fmt == 2) {
        fl::choose_delta<V>(v.data(), n, &ref, &cw);
        buf.resize(fl::delta_bytes(T, n, cw));
        fl::encode_delta_for<V>(v.data(), n, ref, cw, buf.data());
    } else {
        fl::choose_for<V>(v.data(), n, &ref, &cw);
        if (fmt == 0) { ref = 0; cw = fl::width_of(uint64_t(std::make_unsigned_t<V>(
                            *std::max_element(v.begin(), v.end())))); }
        buf.resize(fl::packed_bytes(T, n, cw));
        fl::encode_for<V>(v.data(), n, ref, cw, buf.data());
    }
    std::vector<V> out(static_cast<size_t>(n));
    const double dec = median_ns_per_value([&] {
        if (fmt == 2) fl::decode_delta_for<V>(buf.data(), n, ref, cw, out.data());
        else fl::decode_for<V>(buf.data(), n, ref, cw, out.data());
        g_sink = g_sink + uint64_t(out[size_t(n - 1)]);
    }, n, runs, min_ms);
    if (out != v) { std::fprintf(stderr, "decode mismatch\n"); std::exit(1); }
    bolt::Arena a;
    const bolt::BoltType bt = T == 32 ? bolt::BoltType::Int32 : bolt::BoltType::Int64;
    const bolt::ColumnFormat cf = fmt == 0 ? bolt::ColumnFormat::BitPacked
                                 : fmt == 1 ? bolt::ColumnFormat::FrameOfRef
                                            : bolt::ColumnFormat::DeltaFOR;
    const bolt::BoltColumn c = bolt::BoltColumn::make_fastlanes(cf, buf.data(), uint8_t(cw),
                                                                ref, n, bt, &a);
    const bolt::BoltColumn view = bolt::BoltColumn::make_view(c, n / 3, n - n / 3);
    const double acc = median_ns_per_value([&] {
        uint64_t s = 0;
        for (int64_t i = 0; i < view.length; i += 7) s += uint64_t(view.fastlanes_value<V>(i));
        g_sink = g_sink + s;
    }, (view.length + 6) / 7, runs, min_ms);
    static const char* kName[3] = {"BitPacked", "FrameOfRef", "DeltaFOR"};
    std::printf("%-10s T=%-2u W=%-2u n=%-6lld decode %.3f ns/value  value_at %.2f ns/value\n",
                kName[fmt], T, cw, (long long)n, dec, acc);
}

}  // namespace

int main(int argc, char** argv) {
    const int runs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    const double min_ms = argc > 2 ? std::max(1.0, std::atof(argv[2])) : 20.0;
    for (int fmt = 0; fmt < 3; ++fmt)
        for (uint32_t w : {3u, 11u, 20u})
            for (int64_t n : {1024, 4096, 16384, 65536}) {
                bench_one<int32_t>(fmt, w, n, runs, min_ms);
                bench_one<int64_t>(fmt, w, n, runs, min_ms);
            }
    return int(g_sink & 0);
}
