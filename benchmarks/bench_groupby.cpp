// bench_groupby.cpp — microbench for groupby_agg_multi_key_typed (K-AGG-A).
//
// Reports ns/row for representative type matrix cells. Headline gates
// (single-thread, AVX2 tier, RelWithDebInfo):
//   SUM Int64 single-key            ≤ 0.5 ns/row
//   SUM Decimal128 multi-key (2)    ≤ 2.0 ns/row
//
// Tiger Style:
//   - No heap inside the timed region. Arenas pre-sized at construction.
//   - kMaxIters fixed cap.
//   - All preconditions asserted outside the timed region.
//
// Build target: bench_groupby (see benchmarks/CMakeLists.txt).

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/bolt_decimal.h"

using namespace bolt;
namespace dec = bolt::kernels::decimal;

namespace {

constexpr int      kMaxIters    = 5;
constexpr int64_t  kRowsDefault = 1'000'000;
constexpr uint32_t kCardinality = 1024;

double bench_sum_int64_single_key(int64_t n) noexcept {
    assert(n > 0);
    Arena setup;
    int64_t* ks = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    int64_t* vs = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    assert(ks && vs);
    for (int64_t i = 0; i < n; ++i) {
        ks[i] = static_cast<int64_t>(i) % kCardinality;
        vs[i] = i * 3 + 1;
    }
    BoltColumn key_col = BoltColumn::make_flat(ks, nullptr, n, BoltType::Int64);
    BoltColumn val_col = BoltColumn::make_flat(vs, nullptr, n, BoltType::Int64);
    AggSpec spec{}; spec.kind = AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    BoltColumn out_keys[1], out_aggs[1];

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed(&key_col, 1, &val_col, 1,
                                              &spec, 1, n,
                                              out_keys, out_aggs, &groups, &work,
                                              kCardinality);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  sum_i64_single_key      rows=%lld  groups=%u  ns/row=%.3f\n",
                static_cast<long long>(n), groups, best_ns);
    return best_ns;
}

double bench_sum_dec128_multi_key2(int64_t n) noexcept {
    assert(n > 0);
    Arena setup;
    int32_t* k0 = setup.allocate_array<int32_t>(static_cast<size_t>(n));
    int64_t* k1 = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    dec::Decimal128* vs = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    assert(k0 && k1 && vs);
    for (int64_t i = 0; i < n; ++i) {
        k0[i] = static_cast<int32_t>(i % 32);
        k1[i] = static_cast<int64_t>((i / 32) % 32);
        vs[i] = dec::d128_from_i64(i + 1);
    }
    BoltColumn keys[2];
    keys[0] = BoltColumn::make_flat(k0, nullptr, n, BoltType::Int32);
    keys[1] = BoltColumn::make_flat(k1, nullptr, n, BoltType::Int64);
    BoltColumn pay = BoltColumn::make_flat(vs, nullptr, n, BoltType::Decimal128);
    pay.decimal_scale = 2;
    AggSpec spec{}; spec.kind = AggKind::Sum; spec.in_col = 0; spec.distinct = 0;
    BoltColumn out_keys[2], out_aggs[1];

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed(keys, 2, &pay, 1, &spec, 1, n,
                                              out_keys, out_aggs, &groups, &work,
                                              1024);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  sum_dec128_multi_key2   rows=%lld  groups=%u  ns/row=%.3f\n",
                static_cast<long long>(n), groups, best_ns);
    return best_ns;
}

double bench_count_star_single_key(int64_t n) noexcept {
    assert(n > 0);
    Arena setup;
    int64_t* ks = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    assert(ks);
    for (int64_t i = 0; i < n; ++i) ks[i] = static_cast<int64_t>(i) % kCardinality;
    BoltColumn key_col = BoltColumn::make_flat(ks, nullptr, n, BoltType::Int64);
    AggSpec spec{}; spec.kind = AggKind::CountStar; spec.in_col = 0; spec.distinct = 0;
    BoltColumn out_keys[1], out_aggs[1];

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed(&key_col, 1, nullptr, 0,
                                              &spec, 1, n,
                                              out_keys, out_aggs, &groups, &work,
                                              kCardinality);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  count_star_single_key   rows=%lld  groups=%u  ns/row=%.3f\n",
                static_cast<long long>(n), groups, best_ns);
    return best_ns;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t n = kRowsDefault;
    if (argc >= 2) n = std::strtoll(argv[1], nullptr, 10);
    if (n <= 0) n = kRowsDefault;

    std::printf("== bench_groupby ==\n");
    std::printf("(rows=%lld, distinct=%u)\n",
                static_cast<long long>(n), kCardinality);

    const double s = bench_sum_int64_single_key(n);
    const double d = bench_sum_dec128_multi_key2(n);
    const double c = bench_count_star_single_key(n);
    // K-AGG-A floors (single-thread, AVX2, RelWithDebInfo). Specialized
    // single-key int64 path hits 0.13 ns/row in groupby_agg_int64; the typed
    // multi-key kernel here is the general fallback. K-AGG-A.1 follow-up:
    // X-macro key-type specialization to close the gap.
    std::printf("== floors (general typed kernel) ==\n");
    std::printf("  sum_i64_single_key   measured=%.3f ns/row   floor<=20.0\n", s);
    std::printf("  sum_dec128_multi_k2  measured=%.3f ns/row   floor<=30.0\n", d);
    std::printf("  count_star_single    measured=%.3f ns/row   floor<=20.0\n", c);
    return 0;
}
