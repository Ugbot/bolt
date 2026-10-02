// bench_encodings — scalar decode ns/value of the FastLanes formats
// (BitPacked / FrameOfRef / DeltaFOR, MSEG B1) at 1K..64K values, plus
// decode-on-access (fastlanes_value) for a view. Scalar reference only; the
// NEON / AVX2 kernels are M4.
//
// Usage: bench_encodings [runs=5] [min_ms_per_run=20] [detect]   (detect: K4 cases only)
//        bench_encodings detect <case 0..12> <copy|detect> <n> <iters>
// Prints one row per (format, T, W, n): median ns/value over `runs`.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_fastlanes.h"
#include "bolt/kernels/bolt_page_detect.h"
#include "bolt/kernels/bolt_stats_fused.h"

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

// K4: constant / all-null detection folded into the page copy pass. Each
// case is the same masked copy (null slots zeroed) without and with the
// detection; pages are Flat with distinct values so detection runs to the end.
// The copy loops mirror the kernels' so the difference is the detection.
struct DetectData {
    int64_t n;
    std::vector<int64_t> i64, i64o;
    std::vector<uint8_t> valid, src1, dst1, src16, dst16;
    std::vector<bolt::StringView> sv, svo;
    std::vector<uint8_t> csrc1, csrc16;       // Constant pages (detection runs to the end)
    std::vector<bolt::StringView> csv;
};

void detect_data(DetectData* d, int64_t n) {
    std::mt19937_64 g(7u + uint64_t(n));
    const size_t un = static_cast<size_t>(n);
    d->n = n;
    d->i64.resize(un); d->i64o.resize(un);
    for (auto& x : d->i64) x = int64_t(g() >> 8);
    d->valid.resize((un + 7) / 8 + 8);
    for (auto& b : d->valid) b = uint8_t(g() | 0x11);
    d->src1.resize(un); d->dst1.resize(un); d->src16.resize(un * 16); d->dst16.resize(un * 16);
    for (auto& b : d->src1) b = uint8_t(g());
    for (auto& b : d->src16) b = uint8_t(g());
    d->sv.resize(un); d->svo.resize(un);
    for (auto& v : d->sv) { std::memset(&v, 0, sizeof(v)); v.length = 8; uint64_t x = g(); std::memcpy(v.prefix, &x, 8); }
    d->csrc1.assign(un, 0x5A);
    d->csrc16.resize(un * 16);
    for (size_t i = 0; i < un * 16; ++i) d->csrc16[i] = uint8_t(i & 15);
    d->csv.assign(un, d->sv[0]);
}

BOLT_FORCE_INLINE uint64_t bench_mask(const uint8_t* v, uint32_t i) {
    return v == nullptr ? ~0ull : 0ull - uint64_t((v[i >> 3] >> (i & 7)) & 1u);
}

template <uint32_t W>
BOLT_FORCE_INLINE void masked_row(const uint8_t* s, uint8_t* d, uint64_t m) {
    uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, s, W < 8 ? W : 8);
    if (W > 8) std::memcpy(&hi, s + 8, W - 8);
    lo &= m; hi &= m;
    std::memcpy(d, &lo, W < 8 ? W : 8);
    if (W > 8) std::memcpy(d + 8, &hi, W - 8);
}

// The copy the kernels do without the detection (8-row validity groups).
template <uint32_t W>
void masked_copy(const uint8_t* s, const uint8_t* v, uint32_t n, uint8_t* d) {
    if (v == nullptr) { std::memcpy(d, s, size_t(n) * W); return; }
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint32_t bits = v[i >> 3];
        for (uint32_t j = 0; j < 8; ++j)
            masked_row<W>(s + size_t(i + j) * W, d + size_t(i + j) * W, 0ull - ((bits >> j) & 1u));
    }
    for (; i < n; ++i) masked_row<W>(s + size_t(i) * W, d + size_t(i) * W, bench_mask(v, i));
}

constexpr int kDetectCases = 13;

// Case c (0..kDetectCases-1), detection on or off; returns its name. Cases
// 7..12 are 1..6 over Constant pages.
const char* detect_case(DetectData& d, int c, bool det) {
    namespace pg = bolt::page;
    namespace st = bolt::stats;
    const uint32_t n = uint32_t(d.n);
    const bool cst = c >= 7;
    if (cst) c -= 6;
    const uint8_t* vb = (c == 2 || c == 4 || c == 6) ? d.valid.data() : nullptr;
    if (c == 0) {
        st::FusedStats s;
        st::stats_fused_copy(bolt::BoltType::Int64, d.i64.data(), nullptr, 0, n, d.i64o.data(), &s);
        uint64_t k = uint64_t(s.sum);
        if (det) { uint8_t v[16]; pg::page_value_from_stats(s, v); k += uint64_t(pg::page_kind_from_stats(s)) + v[0]; }
        g_sink = g_sink + k;
        return "int64 (K7 stats)";
    }
    if (c <= 4) {
        const bool w16 = c >= 3;
        const uint8_t* src = cst ? (w16 ? d.csrc16.data() : d.csrc1.data())
                                 : (w16 ? d.src16.data() : d.src1.data());
        uint8_t* dst = w16 ? d.dst16.data() : d.dst1.data();
        if (det) {
            pg::FixedDetectState s; pg::fixed_detect_init(&s, w16 ? 16 : 1);
            pg::fixed_detect_update(&s, src, vb, 0, n, dst);
            uint8_t v[16]; g_sink = g_sink + uint64_t(pg::fixed_detect_finish(s, v));
        } else if (w16) masked_copy<16>(src, vb, n, dst);
        else masked_copy<1>(src, vb, n, dst);
        g_sink = g_sink + dst[0];
        if (cst) return c == 1 ? "CONST w=1 dense" : c == 2 ? "CONST w=1 nulls" : c == 3 ? "CONST w=16 dense" : "CONST w=16 nulls";
        return c == 1 ? "fixed w=1 dense" : c == 2 ? "fixed w=1 nulls" : c == 3 ? "fixed w=16 dense" : "fixed w=16 nulls";
    }
    if (det) {
        pg::SvDetectState s; pg::sv_detect_init(&s);
        pg::sv_detect_update(&s, cst ? d.csv.data() : d.sv.data(), vb, 0, n, d.svo.data(), nullptr);
        bolt::StringView v; g_sink = g_sink + uint64_t(pg::sv_detect_finish(s, &v));
    } else {
        masked_copy<16>(reinterpret_cast<const uint8_t*>(cst ? d.csv.data() : d.sv.data()), vb, n,
                        reinterpret_cast<uint8_t*>(d.svo.data()));
    }
    g_sink = g_sink + d.svo[0].length;
    if (cst) return c == 5 ? "CONST utf8 dense" : "CONST utf8 nulls";
    return c == 5 ? "utf8 inline dense" : "utf8 inline nulls";
}

void bench_detect_all(int64_t n, int runs, double min_ms) {
    DetectData d;
    detect_data(&d, n);
    for (int c = 0; c < kDetectCases; ++c) {
        const double a = median_ns_per_value([&] { detect_case(d, c, false); }, n, runs, min_ms);
        const double b = median_ns_per_value([&] { detect_case(d, c, true); }, n, runs, min_ms);
        std::printf("detect %-18s n=%-6lld copy %.3f  copy+detect %.3f  detect %+.3f ns/value\n",
                    detect_case(d, c, false), (long long)n, a, b, b - a);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // bench_encodings detect <case> <copy|detect> <n> <iters>: one K4 case
    // only, for instruction counting (/usr/bin/time -l) under load.
    if (argc == 6 && std::strcmp(argv[1], "detect") == 0) {
        DetectData d;
        detect_data(&d, std::atoll(argv[4]));
        const int c = std::atoi(argv[2]);
        const bool det = std::strcmp(argv[3], "detect") == 0;
        const int64_t iters = std::atoll(argv[5]);
        if (c < 0 || c >= kDetectCases || d.n <= 0) return 2;
        for (int64_t k = 0; k < iters; ++k) detect_case(d, c, det);
        return int(g_sink & 0);
    }
    const int runs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    const double min_ms = argc > 2 ? std::max(1.0, std::atof(argv[2])) : 20.0;
    const bool detect_only = argc > 3 && std::strcmp(argv[3], "detect") == 0;
    for (int fmt = 0; fmt < 3 && !detect_only; ++fmt)
        for (uint32_t w : {3u, 11u, 20u})
            for (int64_t n : {1024, 4096, 16384, 65536}) {
                bench_one<int32_t>(fmt, w, n, runs, min_ms);
                bench_one<int64_t>(fmt, w, n, runs, min_ms);
            }
    for (int64_t n : {1024, 4096, 65536}) bench_detect_all(n, runs, min_ms);
    return int(g_sink & 0);
}
