// bench_json.cpp — bolt::parse::json throughput (fionn-parity wave).
//
// Reports GiB/s for: build_index (tape), sax_parse (events, no tape),
// ndjson_for_each (tape per record, scratch reset), ndjson_for_each_sax,
// and build_index_filtered (1-path skip). The synthetic corpus mirrors
// fionn's headline shape: string-heavy records with mixed scalar fields.
//
// STL is allowed in benchmarks; production code stays Tiger-Style.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_json.h"

namespace fj = bolt::parse::json;
using clk = std::chrono::high_resolution_clock;

namespace {

std::string make_ndjson_corpus(int64_t records, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> word(3, 24);
    std::uniform_int_distribution<int> byte_dist('a', 'z');
    std::uniform_int_distribution<int64_t> num(0, 1u << 30);
    std::string out;
    out.reserve(static_cast<size_t>(records) * 160);
    for (int64_t i = 0; i < records; ++i) {
        out += "{\"id\": ";
        out += std::to_string(i);
        out += ", \"user\": \"";
        const int wl = word(rng) + 12;   // mostly spilled-length strings
        for (int k = 0; k < wl; ++k) out += static_cast<char>(byte_dist(rng));
        out += "\", \"score\": ";
        out += std::to_string(num(rng));
        out += ", \"ratio\": 0.";
        out += std::to_string(100 + (num(rng) % 900));
        out += ", \"active\": ";
        out += (i & 1) ? "true" : "false";
        out += ", \"note\": \"";
        const int nl = word(rng) * 4;    // long plain-ASCII runs
        for (int k = 0; k < nl; ++k) out += static_cast<char>(byte_dist(rng));
        out += "\"}\n";
    }
    return out;
}

std::string make_array_doc(const std::string& ndjson) {
    std::string out = "[";
    bool first = true;
    size_t pos = 0;
    while (pos < ndjson.size()) {
        size_t e = ndjson.find('\n', pos);
        if (e == std::string::npos) e = ndjson.size();
        if (e > pos) {
            if (!first) out += ",";
            out.append(ndjson, pos, e - pos);
            first = false;
        }
        pos = e + 1;
    }
    out += "]";
    return out;
}

template <class F>
double bench_gibs(const char* name, int64_t bytes, int iters, F fn) {
    for (int i = 0; i < 2; ++i) fn();   // warmup
    const auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = clk::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    const double gibs =
        (static_cast<double>(bytes) * iters) / ns;   // bytes/ns == GB/s
    std::printf("  %-34s %8.3f GB/s  (bytes=%lld, iters=%d)\n",
                name, gibs, static_cast<long long>(bytes), iters);
    return gibs;
}

}  // namespace

int main() {
#if BOLT_SIMD_AVX2
    std::printf("bench_json — SIMD tier: AVX2 plain-run + SWAR\n");
#else
    std::printf("bench_json — SIMD tier: SWAR plain-run (no AVX2)\n");
#endif

    const int64_t kRecords = 200'000;
    const std::string ndjson = make_ndjson_corpus(kRecords, 42);
    const std::string array_doc = make_array_doc(ndjson);
    const auto* nd_src = reinterpret_cast<const uint8_t*>(ndjson.data());
    const auto* ar_src = reinterpret_cast<const uint8_t*>(array_doc.data());
    const auto nd_len = static_cast<int64_t>(ndjson.size());
    const auto ar_len = static_cast<int32_t>(array_doc.size());
    std::printf("corpus: %lld records, ndjson=%.1f MiB, array=%.1f MiB\n\n",
                static_cast<long long>(kRecords),
                ndjson.size() / 1048576.0, array_doc.size() / 1048576.0);

    bolt::ArenaConfig big{};
    big.initial_block_size = 256ull * 1024 * 1024;
    big.max_block_size     = 1024ull * 1024 * 1024;
    bolt::Arena tape_arena{big};

    bench_gibs("build_index (array doc -> tape)", ar_len, 5, [&] {
        tape_arena.reset();
        fj::StructuralIndex idx;
        if (!fj::build_index(ar_src, ar_len, &tape_arena, &idx)) {
            std::printf("  !! build_index failed\n");
        }
    });

    static int64_t sink_count;
    fj::SaxHandler count_ints{};
    count_ints.ctx = &sink_count;
    count_ints.on_int64 = [](void* c, int64_t, const char*, int32_t) noexcept {
        ++*static_cast<int64_t*>(c);
        return true;
    };
    bench_gibs("sax_parse (array doc, no tape)", ar_len, 5, [&] {
        sink_count = 0;
        if (!fj::sax_parse(ar_src, ar_len, &count_ints)) {
            std::printf("  !! sax_parse failed\n");
        }
    });

    bolt::ArenaConfig small{};
    small.initial_block_size = 1ull * 1024 * 1024;
    small.max_block_size     = 16ull * 1024 * 1024;
    bolt::Arena scratch{small};

    bench_gibs("ndjson_for_each (tape/record)", nd_len, 5, [&] {
        fj::NdjsonStats st{};
        if (!fj::ndjson_for_each(
                nd_src, nd_len, false, &scratch, nullptr,
                [](void*, const fj::StructuralIndex*, int64_t) noexcept {
                    return true;
                },
                &st) ||
            st.records != kRecords) {
            std::printf("  !! ndjson_for_each failed\n");
        }
    });

    bench_gibs("ndjson_for_each_sax (no arena)", nd_len, 5, [&] {
        sink_count = 0;
        fj::NdjsonStats st{};
        if (!fj::ndjson_for_each_sax(nd_src, nd_len, false, &count_ints, &st)
            || st.records != kRecords) {
            std::printf("  !! ndjson_for_each_sax failed\n");
        }
    });

    {
        const char* paths[] = {"/0"};
        const int32_t lens[] = {2};
        fj::PathFilter pf{};
        bolt::Arena pf_arena{small};
        if (!fj::compile_paths(paths, lens, 1, &pf_arena, &pf)) {
            std::printf("  !! compile_paths failed\n");
            return 1;
        }
        bench_gibs("build_index_filtered (skip ~all)", ar_len, 5, [&] {
            tape_arena.reset();
            fj::StructuralIndex idx;
            if (!fj::build_index_filtered(ar_src, ar_len, &pf,
                                          &tape_arena, &idx)) {
                std::printf("  !! build_index_filtered failed\n");
            }
        });
    }

    std::printf("\ndone\n");
    return 0;
}
