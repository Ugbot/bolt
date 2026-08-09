// Equivalence gate for the WINDOWED (memory-level-parallel) join probe.
//
// bolt_joinkernel.h's `jk_probe_one_int64_mlp` reorders WHEN the probe touches
// memory (hash a window -> prefetch the window's ctrl/slot lines -> resolve),
// never WHAT it computes. The contract this file pins is deliberately stronger
// than "same pair multiset": the windowed path must emit the SAME (build_idx,
// probe_idx) SEQUENCE as the scalar path, pair for pair, because the phases
// preserve row order within a window and windows run in order. A downstream
// consumer that (today or tomorrow) depends on probe-row ordering — the
// semi/anti mark pass, first-match residual gates, the overshoot carry in
// hash_join_swiss_probe_emit.inc — is then unaffected by construction.
//
// Coverage, over the shapes that flip the kernel's compile-time arms:
//   * UNIQUE build (jk_max_fanout()==1, every TPC-H PK join) and DUPLICATE
//     build (the chain-walking general lane).
//   * bloom ON and OFF (a filter with no false negatives must not change a
//     single pair).
//   * every window size the dispatcher can select (8 / 16 / 32).
//   * NULL keys on both sides (NULL != NULL), misses, duplicates in the probe.
//   * a skewed probe (one hot key dominates) — the distribution where a
//     window's prefetches collapse onto one line.
//   * probe lengths that are shorter than / not a multiple of the window, plus
//     empty probe and empty build (the phase loops' boundary conditions).
//   * Int32 keys — same OneInt64 key shape, narrower load.
//
// A brute-force O(n*m) oracle backs the smaller cases so "both paths agree"
// cannot degenerate into "both paths are wrong the same way".

#include "bolt/join/bolt_joinkernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace {

using namespace bolt;

constexpr uint32_t k_windows[3] = {8, 16, 32};

// Deterministic 64-bit PRNG (splitmix64) — no <random>, no hidden allocation,
// identical stream on every platform so a failure replays exactly.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) noexcept : s(seed) {}
    uint64_t next() noexcept {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) noexcept { return n == 0 ? 0 : next() % n; }
};

// One probe result: the emitted pair sequence, captured for comparison.
struct Pairs {
    static constexpr int32_t k_cap = 1 << 16;
    int32_t build[k_cap];
    int32_t probe[k_cap];
    size_t  n = 0;
};

// Set/clear bit `i` of an Arrow-style validity bitmap.
void set_valid(uint8_t* bm, int64_t i, bool v) noexcept {
    if (v) bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] | (1u << (i & 7)));
    else   bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] & ~(1u << (i & 7)));
}

// Probe `jb` with `p` under an explicitly forced schedule, capturing pairs.
void run_probe(const JoinBuildTyped* jb, const BoltColumn& p, int64_t n_probe,
               bool mlp, uint32_t window, bool bloom, Pairs* out) noexcept {
    jk_mlp_force(mlp, window);
    out->n = join_probe_typed(jb, &p, n_probe, out->build, out->probe,
                              static_cast<size_t>(Pairs::k_cap), bloom);
}

// gtest-visible sequence comparison (EXPECT_* needs a non-noexcept context).
void expect_same(const Pairs& a, const Pairs& b, const char* what) {
    ASSERT_EQ(a.n, b.n) << what << ": pair COUNT differs";
    for (size_t i = 0; i < a.n; ++i) {
        ASSERT_EQ(a.build[i], b.build[i]) << what << ": build_idx at pair " << i;
        ASSERT_EQ(a.probe[i], b.probe[i]) << what << ": probe_idx at pair " << i;
    }
}

// Drive every windowed variant against the scalar path for one (build, probe).
void expect_all_schedules_agree(const JoinBuildTyped* jb, const BoltColumn& p,
                                int64_t n_probe, const char* what) {
    static Pairs ref, got;
    run_probe(jb, p, n_probe, /*mlp=*/false, 16, /*bloom=*/false, &ref);
    for (uint32_t w : k_windows) {
        for (int bl = 0; bl < 2; ++bl) {
            run_probe(jb, p, n_probe, /*mlp=*/true, w, bl != 0, &got);
            char tag[192];
            std::snprintf(tag, sizeof(tag), "%s [mlp W=%u bloom=%d]", what, w, bl);
            expect_same(ref, got, tag);
        }
    }
    // The scalar path with the bloom on is the other pre-existing arm; it must
    // also be pair-identical, which makes `ref` a schedule-independent oracle.
    run_probe(jb, p, n_probe, /*mlp=*/false, 16, /*bloom=*/true, &got);
    expect_same(ref, got, "scalar+bloom");
}

// Brute-force O(n*m) oracle over int64 keys with optional validity.
void expect_matches_bruteforce(const Pairs& got,
                               const int64_t* bk, const uint8_t* bv, int64_t nb,
                               const int64_t* pk, const uint8_t* pv, int64_t np) {
    size_t seen = 0;
    for (int64_t r = 0; r < np; ++r) {
        const bool pvalid = (pv == nullptr) || ((pv[r >> 3] >> (r & 7)) & 1);
        if (!pvalid) continue;
        for (int64_t b = 0; b < nb; ++b) {
            const bool bvalid = (bv == nullptr) || ((bv[b >> 3] >> (b & 7)) & 1);
            if (!bvalid || bk[b] != pk[r]) continue;
            ASSERT_LT(seen, got.n) << "kernel emitted FEWER pairs than the oracle";
            // Order within one probe row's chain is the table's, not the
            // oracle's — check membership by scanning this row's run.
            bool found = false;
            for (size_t j = 0; j < got.n; ++j) {
                if (got.probe[j] == r && got.build[j] == b) { found = true; break; }
            }
            ASSERT_TRUE(found) << "oracle pair (b=" << b << ",p=" << r
                               << ") missing from kernel output";
            ++seen;
        }
    }
    ASSERT_EQ(seen, got.n) << "kernel emitted MORE pairs than the oracle";
}

// ---------------------------------------------------------------------------

// Unique build side — the jk_max_fanout()==1 fast lane every TPC-H PK join
// takes, and therefore the lane the windowed probe must get exactly right.
TEST(BoltProbeMlp, UniqueBuildRandomKeys) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 4096, np = 20000;
    static int64_t bk[nb], pk[np];
    for (int64_t i = 0; i < nb; ++i) bk[i] = i * 7 + 3;      // distinct
    Rng rng(0xC0FFEEull);
    for (int64_t i = 0; i < np; ++i) {
        // ~2/3 hit an existing key (with duplicates), ~1/3 miss entirely.
        pk[i] = (rng.next() % 3 == 0) ? static_cast<int64_t>(rng.next() | 1ull)
                                      : bk[rng.below(nb)];
    }
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    ASSERT_EQ(jk_max_fanout(&jb), 1u) << "fixture must exercise the Unique lane";
    expect_all_schedules_agree(&jb, p, np, "unique/random");
    jk_mlp_force(true, 16);
}

// Duplicate-heavy build — jk_max_fanout()>1, so both paths walk chain_nodes.
TEST(BoltProbeMlp, DuplicateBuildChainWalk) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 3000, np = 4000;
    static int64_t bk[nb], pk[np];
    for (int64_t i = 0; i < nb; ++i) bk[i] = i % 250;   // each key ~12 times
    Rng rng(0x5EEDull);
    for (int64_t i = 0; i < np; ++i) pk[i] = static_cast<int64_t>(rng.below(400));
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    ASSERT_GT(jk_max_fanout(&jb), 1u) << "fixture must exercise the chain lane";
    expect_all_schedules_agree(&jb, p, np, "duplicates/chain");
    jk_mlp_force(true, 16);
}

// Skew: one key takes ~80% of the probe rows. Every window's prefetches then
// target the same line — the degenerate case for a prefetch schedule.
TEST(BoltProbeMlp, SkewedProbeDistribution) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 2048, np = 12000;
    static int64_t bk[nb], pk[np];
    for (int64_t i = 0; i < nb; ++i) bk[i] = i + 1;
    Rng rng(0xBEEFull);
    for (int64_t i = 0; i < np; ++i) {
        pk[i] = (rng.below(10) < 8) ? 1337 : static_cast<int64_t>(rng.below(nb) + 1);
    }
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    expect_all_schedules_agree(&jb, p, np, "skew");
    jk_mlp_force(true, 16);
}

// NULL keys on BOTH sides. A NULL never matches (not even another NULL), and a
// NULL probe row must not consume a window slot's hash/prefetch either.
TEST(BoltProbeMlp, NullKeysBothSides) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 600, np = 1500;
    static int64_t bk[nb], pk[np];
    static uint8_t bv[(nb + 7) / 8], pv[(np + 7) / 8];
    std::memset(bv, 0xFF, sizeof(bv));
    std::memset(pv, 0xFF, sizeof(pv));
    Rng rng(0xD00Dull);
    for (int64_t i = 0; i < nb; ++i) {
        bk[i] = i;
        if (i % 7 == 0) set_valid(bv, i, false);   // and the payload stays 0
    }
    for (int64_t i = 0; i < np; ++i) {
        pk[i] = static_cast<int64_t>(rng.below(nb + 50));
        if (i % 5 == 0) set_valid(pv, i, false);
    }
    BoltColumn b = BoltColumn::make_flat(bk, bv, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, pv, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    expect_all_schedules_agree(&jb, p, np, "nulls");

    static Pairs got;
    run_probe(&jb, p, np, /*mlp=*/true, 16, /*bloom=*/false, &got);
    expect_matches_bruteforce(got, bk, bv, nb, pk, pv, np);
    jk_mlp_force(true, 16);
}

// Boundaries: probe lengths around/below the window, empty probe, empty build.
TEST(BoltProbeMlp, WindowBoundaryLengths) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 64;
    static int64_t bk[nb], pk[80];
    for (int64_t i = 0; i < nb; ++i) bk[i] = i;
    for (int64_t i = 0; i < 80; ++i) pk[i] = i % (nb + 10);
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    // 1 and 0 rows, then every length that straddles a window edge.
    const int64_t lens[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 65, 80};
    for (int64_t n : lens) {
        BoltColumn p = BoltColumn::make_flat(pk, nullptr, n, BoltType::Int64);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "len=%lld", static_cast<long long>(n));
        expect_all_schedules_agree(&jb, p, n, tag);
    }
    // Empty build: no pairs on any schedule.
    static bolt::Arena a2;
    JoinBuildTyped jb0{};
    BoltColumn b0 = BoltColumn::make_flat(bk, nullptr, 0, BoltType::Int64);
    ASSERT_TRUE(join_build_typed(&b0, 1, 0, &a2, &jb0));
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, 80, BoltType::Int64);
    static Pairs got;
    for (uint32_t w : k_windows) {
        run_probe(&jb0, p, 80, /*mlp=*/true, w, /*bloom=*/false, &got);
        EXPECT_EQ(got.n, 0u) << "empty build must emit nothing (W=" << w << ")";
    }
    jk_mlp_force(true, 16);
}

// Int32 keys take the same OneInt64 shape through a narrower load.
TEST(BoltProbeMlp, Int32Keys) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 1000, np = 3000;
    static int32_t bk[nb], pk[np];
    for (int64_t i = 0; i < nb; ++i) bk[i] = static_cast<int32_t>(i * 3);
    Rng rng(0x1234ull);
    for (int64_t i = 0; i < np; ++i)
        pk[i] = static_cast<int32_t>(rng.below(static_cast<uint64_t>(nb) * 4));
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int32);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, np, BoltType::Int32);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    expect_all_schedules_agree(&jb, p, np, "int32");
    jk_mlp_force(true, 16);
}

// The kill switch must actually reach the kernel: with CHUKONU_HJ_MLP=0 latched
// the scalar arm runs, and it agrees pair-for-pair with the windowed arm.
TEST(BoltProbeMlp, KillSwitchSelectsScalarAndAgrees) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 500, np = 2000;
    static int64_t bk[nb], pk[np];
    for (int64_t i = 0; i < nb; ++i) bk[i] = i * 11;
    Rng rng(0xABCDull);
    for (int64_t i = 0; i < np; ++i) pk[i] = bk[rng.below(nb)];
    BoltColumn b = BoltColumn::make_flat(bk, nullptr, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk, nullptr, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));

    static Pairs off, on;
    jk_mlp_force(false, 16);
    off.n = join_probe_typed(&jb, &p, np, off.build, off.probe, Pairs::k_cap, false);
    jk_mlp_force(true, 16);
    on.n = join_probe_typed(&jb, &p, np, on.build, on.probe, Pairs::k_cap, false);
    EXPECT_EQ(off.n, static_cast<size_t>(np)) << "every probe row should match";
    expect_same(off, on, "killswitch");
    expect_matches_bruteforce(on, bk, nullptr, nb, pk, nullptr, np);
}

}  // namespace
