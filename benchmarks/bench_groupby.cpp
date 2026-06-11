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


// K-AGG-A.3: cross-morsel begin/ingest/finalize at the same scale.
// Slices the same input into kMorselSlices morsels and feeds them through
// the begin/ingest/finalize API. Should match the one-shot floor within
// a few percent.
double bench_sum_int64_xmorsel(int64_t n) noexcept {
    assert(n > 0);
    constexpr int kMorselSlices = 16;
    Arena setup;
    int64_t* ks = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    int64_t* vs = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    assert(ks && vs);
    for (int64_t i = 0; i < n; ++i) {
        ks[i] = static_cast<int64_t>(i) % kCardinality;
        vs[i] = i * 3 + 1;
    }
    BoltColumn key_desc = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn val_desc = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Int64);
    AggSpec spec{}; spec.kind = AggKind::Sum; spec.in_col = 0; spec.distinct = 0;

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    BoltColumn out_keys[1], out_aggs[1];
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        GroupbyTypedState st{};
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed_begin(
            &st, &work, &key_desc, 1, &val_desc, 1, &spec, 1, kCardinality);
        assert(ok); (void)ok;
        const int64_t slice = n / kMorselSlices;
        int64_t off = 0;
        for (int s = 0; s < kMorselSlices; ++s) {
            const int64_t len = (s == kMorselSlices - 1) ? (n - off) : slice;
            BoltColumn k = BoltColumn::make_flat(ks + off, nullptr, len, BoltType::Int64);
            BoltColumn v = BoltColumn::make_flat(vs + off, nullptr, len, BoltType::Int64);
            groupby_agg_multi_key_typed_ingest(&st, &k, &v, nullptr, 0, len);
            assert(!st.oom);
            off += len;
        }
        ok = groupby_agg_multi_key_typed_finalize(&st, out_keys, out_aggs, &groups);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  sum_i64_xmorsel         rows=%lld  groups=%u  slices=%d  ns/row=%.3f\n",
                static_cast<long long>(n), groups, kMorselSlices, best_ns);
    return best_ns;
}

// K-AGG-C: the TPC-H Q1 shape — 2 short inline-Utf8 keys (4 populated
// combos), 3x Decimal128 SUM + COUNT(*) + AVG(d128), cross-morsel in
// 65536-row slices (what chukonu's values source emits). THE headline
// shape for the two-pass restructure (baseline ~175 ns/row in situ).
double bench_q1_shape_xmorsel(int64_t n) noexcept {
    assert(n > 0);
    constexpr int64_t kSlice = 65536;
    Arena setup;
    auto* rf = setup.allocate_array<StringView>(static_cast<size_t>(n));
    auto* ls = setup.allocate_array<StringView>(static_cast<size_t>(n));
    auto* p0 = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    auto* p1 = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    auto* p2 = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    assert(rf && ls && p0 && p1 && p2);
    const StringView rfv[3] = {StringView::from_cstr("A"),
                               StringView::from_cstr("N"),
                               StringView::from_cstr("R")};
    const StringView lsv[2] = {StringView::from_cstr("F"),
                               StringView::from_cstr("O")};
    for (int64_t i = 0; i < n; ++i) {
        // 4 populated combos: A/F, N/F, N/O, R/F (TPC-H-ish correlation).
        const int c = static_cast<int>(i & 3);
        rf[i] = rfv[(c == 0) ? 0 : (c == 3) ? 2 : 1];
        ls[i] = lsv[(c == 2) ? 1 : 0];
        p0[i] = dec::d128_from_i64((i % 5000) + 1);
        p1[i] = dec::d128_from_i64((i % 90000) + 100);
        p2[i] = dec::d128_from_i64((i % 11) + 1);
    }
    BoltColumn keyd[2];
    keyd[0] = BoltColumn::make_flat(rf, nullptr, 0, BoltType::Utf8);
    keyd[1] = BoltColumn::make_flat(ls, nullptr, 0, BoltType::Utf8);
    BoltColumn payd[3];
    payd[0] = BoltColumn::make_flat(p0, nullptr, 0, BoltType::Decimal128);
    payd[1] = BoltColumn::make_flat(p1, nullptr, 0, BoltType::Decimal128);
    payd[2] = BoltColumn::make_flat(p2, nullptr, 0, BoltType::Decimal128);
    payd[0].decimal_scale = 2; payd[1].decimal_scale = 2;
    payd[2].decimal_scale = 2;
    AggSpec specs[5]{};
    specs[0].kind = AggKind::Sum;       specs[0].in_col = 0;
    specs[1].kind = AggKind::Sum;       specs[1].in_col = 1;
    specs[2].kind = AggKind::Sum;       specs[2].in_col = 2;
    specs[3].kind = AggKind::CountStar; specs[3].in_col = 0;
    specs[4].kind = AggKind::Avg;       specs[4].in_col = 2;

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    BoltColumn out_keys[2], out_aggs[5];
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        GroupbyTypedState st{};
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed_begin(
            &st, &work, keyd, 2, payd, 3, specs, 5, 64);
        assert(ok); (void)ok;
        for (int64_t off = 0; off < n; off += kSlice) {
            const int64_t len = (off + kSlice > n) ? (n - off) : kSlice;
            BoltColumn k[2], p[3];
            k[0] = BoltColumn::make_flat(rf + off, nullptr, len, BoltType::Utf8);
            k[1] = BoltColumn::make_flat(ls + off, nullptr, len, BoltType::Utf8);
            p[0] = BoltColumn::make_flat(p0 + off, nullptr, len, BoltType::Decimal128);
            p[1] = BoltColumn::make_flat(p1 + off, nullptr, len, BoltType::Decimal128);
            p[2] = BoltColumn::make_flat(p2 + off, nullptr, len, BoltType::Decimal128);
            p[0].decimal_scale = 2; p[1].decimal_scale = 2;
            p[2].decimal_scale = 2;
            groupby_agg_multi_key_typed_ingest(&st, k, p, nullptr, 0, len);
            assert(!st.oom);
        }
        ok = groupby_agg_multi_key_typed_finalize(&st, out_keys, out_aggs,
                                                  &groups);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  q1_shape_xmorsel        rows=%lld  groups=%u  ns/row=%.3f\n",
                static_cast<long long>(n), groups, best_ns);
    return best_ns;
}

// K-AGG-C: 10K-int-group shape (the bench_python_vs group-by-suppkey
// case). 1 int64 key, SUM(d128) + COUNT(*), 65536-row slices.
double bench_i64key_10k_groups_xmorsel(int64_t n) noexcept {
    assert(n > 0);
    constexpr int64_t kSlice  = 65536;
    constexpr uint32_t kGroups = 10000;
    Arena setup;
    auto* ks = setup.allocate_array<int64_t>(static_cast<size_t>(n));
    auto* vs = setup.allocate_array<dec::Decimal128>(static_cast<size_t>(n));
    assert(ks && vs);
    for (int64_t i = 0; i < n; ++i) {
        ks[i] = (i * 2654435761u) % kGroups;   // scattered, all groups hit
        vs[i] = dec::d128_from_i64((i % 50) + 1);
    }
    BoltColumn keyd = BoltColumn::make_flat(ks, nullptr, 0, BoltType::Int64);
    BoltColumn payd = BoltColumn::make_flat(vs, nullptr, 0, BoltType::Decimal128);
    payd.decimal_scale = 2;
    AggSpec specs[2]{};
    specs[0].kind = AggKind::Sum;       specs[0].in_col = 0;
    specs[1].kind = AggKind::CountStar; specs[1].in_col = 0;

    double best_ns = 1e30;
    uint32_t groups = 0;
    Arena work;
    BoltColumn out_keys[1], out_aggs[2];
    for (int it = 0; it < kMaxIters; ++it) {
        work.reset();
        GroupbyTypedState st{};
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = groupby_agg_multi_key_typed_begin(
            &st, &work, &keyd, 1, &payd, 1, specs, 2, kGroups * 2);
        assert(ok); (void)ok;
        for (int64_t off = 0; off < n; off += kSlice) {
            const int64_t len = (off + kSlice > n) ? (n - off) : kSlice;
            BoltColumn k = BoltColumn::make_flat(ks + off, nullptr, len,
                                                 BoltType::Int64);
            BoltColumn p = BoltColumn::make_flat(vs + off, nullptr, len,
                                                 BoltType::Decimal128);
            p.decimal_scale = 2;
            groupby_agg_multi_key_typed_ingest(&st, &k, &p, nullptr, 0, len);
            assert(!st.oom);
        }
        ok = groupby_agg_multi_key_typed_finalize(&st, out_keys, out_aggs,
                                                  &groups);
        auto t1 = std::chrono::high_resolution_clock::now();
        assert(ok); (void)ok;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double per = ns / static_cast<double>(n);
        if (per < best_ns) best_ns = per;
    }
    std::printf("  i64key_10k_xmorsel      rows=%lld  groups=%u  ns/row=%.3f\n",
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
    const double x = bench_sum_int64_xmorsel(n);
    // K-AGG-C headline shapes (6M rows regardless of n arg for stability).
    const double q1 = bench_q1_shape_xmorsel(6'000'000);
    const double tk = bench_i64key_10k_groups_xmorsel(6'000'000);
    // K-AGG-A floors (single-thread, AVX2, RelWithDebInfo). Specialized
    // single-key int64 path hits 0.13 ns/row in groupby_agg_int64; the typed
    // multi-key kernel here is the general fallback. K-AGG-A.1 follow-up:
    // X-macro key-type specialization to close the gap.
    std::printf("== floors (general typed kernel) ==\n");
    std::printf("  sum_i64_single_key   measured=%.3f ns/row   floor<=20.0\n", s);
    std::printf("  sum_dec128_multi_k2  measured=%.3f ns/row   floor<=30.0\n", d);
    std::printf("  count_star_single    measured=%.3f ns/row   floor<=20.0\n", c);
    // K-AGG-A.3: cross-morsel must stay within 5% of one-shot.
    std::printf("  sum_i64_xmorsel      measured=%.3f ns/row   (vs one-shot %.3f, ratio=%.2fx)\n",
                x, s, (s > 0.0) ? (x / s) : 0.0);
    // K-AGG-C gates (two-pass). Baselines pre-K-AGG-C: q1 77.0, 10k 30.6.
    // Q1 shape (4 groups): MRU pass-1 + banked pass-2 -> ~24 measured.
    // 10k-int floor decomposition: SwissTable find ~5.4 (measured kernel
    // floor) + hash ~1.7 + pack/memo/gid ~3.5 + pass-2 d128 sum + count
    // into L2-resident accums ~5.0 => ~16 measured; gate 18 = watermark
    // with CI-noise margin, NOT an aspiration (revisit only with a faster
    // probe, e.g. vectorized multi-probe).
    std::printf("  q1_shape_xmorsel     measured=%.3f ns/row   gate<=25.0\n", q1);
    std::printf("  i64key_10k_xmorsel   measured=%.3f ns/row   gate<=18.0\n", tk);
    return 0;
}
