// bench_utf8.cpp — micro-benchmarks for bolt_utf8 kernels.
//
// Reports ns/row for: utf8_compare, utf8_filter_eq, utf8_hash,
// utf8_filter_like (exact / prefix / suffix / contains / general),
// utf8_substring.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_utf8.h"

using bolt::Arena;
using bolt::StringView;
namespace ku = bolt::kernels::utf8;

using clk = std::chrono::high_resolution_clock;

template<class F>
static double bench(const char* name, int64_t n, int iters, F fn) {
    std::printf("  starting %s\n", name); std::fflush(stdout);
    // Warmup
    for (int i = 0; i < 3; ++i) fn();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = clk::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double per_row = ns / (double(n) * iters);
    std::printf("  %-36s %8.2f ns/row  (n=%lld, iters=%d)\n",
                name, per_row, (long long)n, iters);
    return per_row;
}

static std::vector<std::string> gen_strings(int64_t n, uint32_t avg_len) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> len_dist(avg_len / 2 + 1, avg_len * 3 / 2 + 1);
    std::uniform_int_distribution<int> byte_dist('a', 'z');
    std::vector<std::string> v;
    v.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t L = len_dist(rng);
        std::string s(L, 'a');
        for (uint32_t k = 0; k < L; ++k) s[k] = char(byte_dist(rng));
        v.push_back(std::move(s));
    }
    return v;
}

// Build inline-or-spilled views from a vector of strings.  Spilled bodies go
// into `pool`; views point into `pool`.
static void build_views(const std::vector<std::string>& src,
                        std::vector<StringView>* out,
                        std::vector<char>* pool) {
    out->resize(src.size());
    pool->clear();
    // Reserve once so c_str pointers stay stable across loop.
    size_t total = 0;
    for (auto& s : src) if (s.size() > 12) total += s.size();
    pool->reserve(total);
    for (size_t i = 0; i < src.size(); ++i) {
        const auto& s = src[i];
        StringView v;
        std::memset(&v, 0, sizeof(v));
        v.length = uint32_t(s.size());
        if (s.size() <= 12) {
            if (!s.empty()) std::memcpy(v.prefix, s.data(),
                                        s.size() < 4 ? s.size() : 4);
            if (s.size() > 4)
                std::memcpy(v.inline_data, s.data() + 4, s.size() - 4);
        } else {
            uint32_t off = uint32_t(pool->size());
            pool->insert(pool->end(), s.begin(), s.end());
            std::memcpy(v.prefix, s.data(), 4);
            v.ref.buf_idx = 0;
            v.ref.offset = off;
        }
        (*out)[i] = v;
    }
}

int main() {
    const int64_t N = 100'000;
    const int ITERS = 20;
    std::printf("starting\n"); std::fflush(stdout);

    // src_short: force <=12 bytes for uniform inline path coverage.
    std::vector<std::string> src_short = gen_strings(N, 6);
    for (auto& s : src_short) if (s.size() > 12) s.resize(12);
    // src_mid: force >12 bytes for uniform spilled path coverage.
    std::vector<std::string> src_mid = gen_strings(N, 16);
    for (auto& s : src_mid) if (s.size() <= 12) s.resize(16, 'q');
    std::vector<StringView> a, b;
    std::vector<char> pool_a, pool_b;
    build_views(src_short, &a, &pool_a);
    build_views(src_short, &b, &pool_b);
    // Make some differ
    for (int64_t i = 0; i < N; i += 3) b[i].prefix[0] ^= 1;

    std::vector<int32_t> out_i32(N);
    std::vector<uint64_t> out_u64(N);
    std::vector<StringView> out_sv(N);

    std::printf("== utf8 kernels (n=%lld, %d iters) ==\n", (long long)N, ITERS);

    bench("utf8_compare (inline, avg 8)", N, ITERS, [&]{
        ku::utf8_compare(a.data(), b.data(), out_i32.data(), N);
    });

    auto needle_short = StringView::from_cstr("aaaa");
    bench("utf8_filter_eq (inline, miss)", N, ITERS, [&]{
        ku::utf8_filter_eq(a.data(), N, needle_short, out_i32.data());
    });

    bench("utf8_hash (inline, avg 8)", N, ITERS, [&]{
        ku::utf8_hash(a.data(), N, out_u64.data());
    });

    // Mid-sized (spilled): hash + compare across 16-byte buffer.
    std::vector<StringView> am, bm;
    std::vector<char> pa, pb;
    build_views(src_mid, &am, &pa);
    build_views(src_mid, &bm, &pb);
    for (int64_t i = 0; i < N; i += 3) bm[i].prefix[0] ^= 1;

    bench("utf8_compare (spilled, avg 16)", N, ITERS, [&]{
        ku::utf8_compare(am.data(), bm.data(), out_i32.data(), N,
                         pa.data(), pb.data());
    });

    bench("utf8_hash (spilled, avg 16)", N, ITERS, [&]{
        ku::utf8_hash(am.data(), N, out_u64.data(), pa.data());
    });

    // LIKE — exercise each kind.
    ku::CompiledLike pat;
    auto cpat = [&](const char* p) {
        auto v = StringView::from_cstr(p);
        ku::utf8_like_compile(v, &pat);
    };

    cpat("aaaa");
    bench("utf8_filter_like exact", N, ITERS, [&]{
        ku::utf8_filter_like(a.data(), N, &pat, out_i32.data());
    });
    cpat("aa%");
    bench("utf8_filter_like prefix", N, ITERS, [&]{
        ku::utf8_filter_like(a.data(), N, &pat, out_i32.data());
    });
    cpat("%aa");
    bench("utf8_filter_like suffix", N, ITERS, [&]{
        ku::utf8_filter_like(a.data(), N, &pat, out_i32.data());
    });
    cpat("%aa%");
    bench("utf8_filter_like contains", N, ITERS, [&]{
        ku::utf8_filter_like(a.data(), N, &pat, out_i32.data());
    });
    cpat("a_b%c_d");
    bench("utf8_filter_like general", N, ITERS, [&]{
        ku::utf8_filter_like(a.data(), N, &pat, out_i32.data());
    });

    // Substring (allocates per row → reset arena each iteration).
    Arena arena;
    auto anchor = reinterpret_cast<const char*>(&arena);
    bench("utf8_substring (inline)", N, ITERS, [&]{
        arena.reset();
        ku::utf8_substring(a.data(), N, 2, 4, out_sv.data(), &arena, anchor);
    });

    return 0;
}
