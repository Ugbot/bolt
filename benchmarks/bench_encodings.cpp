// bench_encodings.cpp — MSEG M0 native-format kernels, 1 thread, ns/value per
// page size. Rows: K3 RLE (encode, decode, run-granular filter count / sel).
//
//   bench_encodings [data_dir] [reps]
//   bench_encodings --work decode|count|sel <data_dir> <col> <page> file|clustered <passes>
//     exactly <passes> passes of one kernel, no timing: run under
//     /usr/bin/time -l at two pass counts; the slope is instructions/value.
//
// data_dir holds raw little-endian int16 columns <Name>.i16 (ClickBench
// AdvEngineID / OS / Sex, extracted from hits.parquet). Each column is cut
// into pages of 1K..64K rows and measured in file order and clustered
// (each page sorted, the compaction-with-cluster-key shape). Median of reps.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bolt/kernels/bolt_rle.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace R = bolt::rle;

struct Pages {
    int64_t page = 0;
    std::vector<int16_t> vals;
    std::vector<int32_t> ends;
    std::vector<int64_t> run_off;  // page p's runs: [run_off[p], run_off[p+1])
    std::vector<int64_t> rows;     // rows of page p
};

std::vector<int16_t> load(const std::string& path) {
    std::vector<int16_t> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(bytes) / sizeof(int16_t));
    const size_t got = std::fread(v.data(), sizeof(int16_t), v.size(), f);
    std::fclose(f);
    v.resize(got);
    return v;
}

Pages encode_pages(const std::vector<int16_t>& col, int64_t page) {
    Pages p;
    p.page = page;
    p.run_off.push_back(0);
    const int64_t n = static_cast<int64_t>(col.size());
    for (int64_t b = 0; b < n; b += page) {
        const int64_t len = std::min(page, n - b);
        const size_t at = p.vals.size();
        p.vals.resize(at + static_cast<size_t>(len));
        p.ends.resize(at + static_cast<size_t>(len));
        const int64_t r = R::encode(col.data() + b, nullptr, len, p.vals.data() + at,
                                    p.ends.data() + at, len);
        if (r < 0) std::abort();
        p.vals.resize(at + static_cast<size_t>(r));
        p.ends.resize(at + static_cast<size_t>(r));
        p.run_off.push_back(p.run_off.back() + r);
        p.rows.push_back(len);
    }
    return p;
}

template <class F> double median_ns_per_val(int reps, int64_t rows, F&& pass) {
    int iters = 1;
    for (;;) {  // calibrate: one rep >= 20 ms
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) pass();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (ms >= 20.0 || iters >= (1 << 20)) break;
        iters *= 2;
    }
    std::vector<double> ns;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) pass();
        const double t = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        ns.push_back(t / (static_cast<double>(iters) * static_cast<double>(rows)));
    }
    std::sort(ns.begin(), ns.end());
    return ns[ns.size() / 2];
}

volatile int64_t g_sink;

void bench_column(const char* name, const char* order, const std::vector<int16_t>& col,
                  int64_t page, int reps) {
    const Pages p = encode_pages(col, page);
    const int64_t n = static_cast<int64_t>(col.size());
    const size_t np = p.rows.size();
    std::vector<int16_t> out(static_cast<size_t>(page));
    std::vector<int32_t> sel(static_cast<size_t>(page));
    const int16_t s = col[0];
    std::vector<int16_t> ev(static_cast<size_t>(page));
    std::vector<int32_t> ee(static_cast<size_t>(page));
    const double enc = median_ns_per_val(reps, n, [&] {
        int64_t acc = 0;
        for (int64_t b = 0; b < n; b += page) {
            acc += R::encode(col.data() + b, nullptr, std::min(page, n - b), ev.data(),
                             ee.data(), page);
        }
        g_sink = acc;
    });
    const double dec = median_ns_per_val(reps, n, [&] {
        int64_t acc = 0;
        for (size_t i = 0; i < np; ++i) {
            const int64_t r0 = p.run_off[i];
            R::decode(p.vals.data() + r0, p.ends.data() + r0, p.run_off[i + 1] - r0, p.rows[i],
                      out.data());
            acc += out[0] + out[static_cast<size_t>(p.rows[i] - 1)];
        }
        g_sink = acc;
    });
    const double cnt = median_ns_per_val(reps, n, [&] {
        int64_t acc = 0;
        for (size_t i = 0; i < np; ++i) {
            const int64_t r0 = p.run_off[i];
            acc += R::filter_count(p.vals.data() + r0, p.ends.data() + r0,
                                   p.run_off[i + 1] - r0, nullptr, R::CmpOp::Eq, s);
        }
        g_sink = acc;
    });
    const double sl = median_ns_per_val(reps, n, [&] {
        int64_t acc = 0;
        for (size_t i = 0; i < np; ++i) {
            const int64_t r0 = p.run_off[i];
            acc += R::filter_sel(p.vals.data() + r0, p.ends.data() + r0, p.run_off[i + 1] - r0,
                                 nullptr, R::CmpOp::Ne, s, sel.data());
        }
        g_sink = acc;
    });
    std::printf("rle  %-12s %-9s page=%-6lld rows/run=%9.1f  encode=%.4f  decode=%.4f  "
                "count_eq=%.5f  sel_ne=%.4f  ns/val\n",
                name, order, static_cast<long long>(page),
                static_cast<double>(n) / static_cast<double>(p.vals.size()), enc, dec, cnt, sl);
}

void cluster(std::vector<int16_t>& col, int64_t page) {
    for (size_t b = 0; b < col.size(); b += static_cast<size_t>(page)) {
        const size_t e = std::min(col.size(), b + static_cast<size_t>(page));
        std::sort(col.begin() + static_cast<long>(b), col.begin() + static_cast<long>(e));
    }
}

int work(char** argv) {
    const std::string kind = argv[2];
    std::vector<int16_t> col = load(std::string(argv[3]) + "/" + argv[4] + ".i16");
    const int64_t page = std::atoll(argv[5]);
    if (col.empty() || page <= 0) return 2;
    if (std::string(argv[6]) == "clustered") cluster(col, page);
    const int passes = std::atoi(argv[7]);
    const Pages p = encode_pages(col, page);
    std::vector<int16_t> out(static_cast<size_t>(page));
    std::vector<int32_t> sel(static_cast<size_t>(page));
    const int k = kind == "decode" ? 0 : kind == "count" ? 1 : 2;
    int64_t acc = 0;
    for (int it = 0; it < passes; ++it) {
        for (size_t i = 0; i < p.rows.size(); ++i) {
            const int64_t r0 = p.run_off[i];
            const int64_t nr = p.run_off[i + 1] - r0;
            if (k == 0) {
                R::decode(p.vals.data() + r0, p.ends.data() + r0, nr, p.rows[i], out.data());
                acc += out[0];
            } else if (k == 1) {
                acc += R::filter_count(p.vals.data() + r0, p.ends.data() + r0, nr, nullptr,
                                       R::CmpOp::Eq, col[0]);
            } else {
                acc += R::filter_sel(p.vals.data() + r0, p.ends.data() + r0, nr, nullptr,
                                     R::CmpOp::Ne, col[0], sel.data());
            }
        }
    }
    g_sink = acc;
    std::printf("work %s rows=%zu passes=%d\n", kind.c_str(), col.size(), passes);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 8 && std::string(argv[1]) == "--work") return work(argv);
    const std::string dir = argc > 1 ? argv[1] : ".";
    const int reps = argc > 2 ? std::max(5, std::atoi(argv[2])) : 7;
    const char* names[] = {"AdvEngineID", "OS", "Sex"};
    const int64_t pages[] = {1024, 4096, 16384, 65536};
    for (const char* name : names) {
        std::vector<int16_t> col = load(dir + "/" + name + ".i16");
        if (col.empty()) {
            std::printf("rle  %s: no data at %s/%s.i16, skipped\n", name, dir.c_str(), name);
            continue;
        }
        for (const int64_t page : pages) bench_column(name, "file", col, page, reps);
        for (const int64_t page : pages) {
            std::vector<int16_t> sorted = col;
            cluster(sorted, page);
            bench_column(name, "clustered", sorted, page, reps);
        }
    }
    return 0;
}
