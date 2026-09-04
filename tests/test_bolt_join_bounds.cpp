// Bounds contract for the hash-join probe kernels (G2GRAPH-27).
//
// WHY THIS FILE IS COMPILED WITH -DNDEBUG
// ---------------------------------------
// Every write bound on the probe path used to be an `assert`:
//     assert(walk < kHJMaxChainLen);   // bolt_joinkernel.h x3, bolt_hashjoin.h
//     assert(out  < pairs_cap);        // beside each of them
// and `pairs_cap` was explicitly `(void)pairs_cap;`-ed so the parameter was
// UNUSED in any build that defines NDEBUG. A stock `-O3 -DNDEBUG` Release —
// what every consumer of this library actually ships — therefore had NO bound
// on an emit loop that writes into a caller-supplied buffer. This translation
// unit defines NDEBUG on purpose so it exercises exactly that configuration:
// a test that only ever runs with asserts live cannot see this bug class at
// all, it just sees a clean abort.
//
// THE DEFECT THE FIXTURES PIN
// ---------------------------
// `jk_max_fanout()` — the ONLY thing a caller has to size its pair buffer from
// (chukonu's hash_join_swiss_probe_emit.inc divides k_hj_pairs_cap by it to
// pick a probe block size) — used to end with
//     if (fan > kHJMaxChainLen) fan = kHJMaxChainLen;
// i.e. the SIZING path silently assumed no chain exceeds 4096 while the
// WALKING path only *asserted* it. Neither enforced it, and the build path
// imposes no per-chain limit whatsoever: 6000 build rows sharing one key make
// one 6000-node chain. The two halves disagreeing is the bug — the caller
// budgets for 4096 matches per probe row and the kernel emits 6000.
//
// This is not hypothetical. Two independent tests in this tree already
// document the abort from real data (chukonu's TAQ trade JOIN quote on `sym`,
// python/tests/native/test_golden_taq.py and spark/test_dataframe_parity.py:
// "Assertion failed: walk < kHJMaxChainLen ... a hard process abort"). With
// NDEBUG those same queries do not abort — they write past the end of the
// pair buffer.
//
// Each fixture puts a canary region immediately after the pair buffers and
// asserts it is untouched, so "did not crash" is never mistaken for "did not
// corrupt": an out-of-bounds write lands in the canary and is caught by value.

#ifndef NDEBUG
#error "test_bolt_join_bounds.cpp must be compiled with -DNDEBUG (see header comment)"
#endif

#include "bolt/join/bolt_joinkernel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using namespace bolt;

// Chain longer than kHJMaxChainLen (4096) — the whole point of the fixture.
constexpr int64_t k_dup_rows = 6000;
static_assert(k_dup_rows > static_cast<int64_t>(kHJMaxChainLen),
              "fixture must build a chain past the old silent clamp");

// A pair buffer with a canary tail. `cap` is what we hand the kernel; anything
// written past it lands in the canary and is detected by value, not by luck.
struct GuardedPairs {
    static constexpr int32_t k_canary = 0x5A5A5A5A;
    std::vector<int32_t> build;
    std::vector<int32_t> probe;
    size_t cap;
    size_t guard;

    GuardedPairs(size_t cap_, size_t guard_) : cap(cap_), guard(guard_) {
        build.assign(cap + guard, k_canary);
        probe.assign(cap + guard, k_canary);
    }
    // Zero findings iff nothing wrote past `cap`.
    size_t canary_breaches() const {
        size_t n = 0;
        for (size_t i = cap; i < build.size(); ++i) {
            if (build[i] != k_canary || probe[i] != k_canary) ++n;
        }
        return n;
    }
};

// Build side: `rows` rows that ALL carry the same key, so the table holds one
// chain of length `rows`.
struct DupBuild {
    bolt::Arena     arena;
    std::vector<int64_t> keys;
    BoltColumn      col{};
    JoinBuildTyped  jb{};

    bool init(int64_t rows, int64_t key, bool force_general) {
        arena.reset();
        keys.assign(static_cast<size_t>(rows), key);
        col = BoltColumn::make_flat(keys.data(), nullptr, rows, BoltType::Int64);
        return force_general
            ? join_build_typed_general(&col, 1, static_cast<uint64_t>(rows),
                                       &arena, &jb)
            : join_build_typed(&col, 1, static_cast<uint64_t>(rows), &arena, &jb);
    }
};

// ---------------------------------------------------------------------------
// 1. Sizing must not assume a limit the walk does not enforce.
// ---------------------------------------------------------------------------
// jk_max_fanout is the caller's ONLY handle on "how many pairs can one probe
// row produce". If it under-reports, every caller that sizes from it is a
// buffer overflow waiting for the right data. Pre-fix this returned 4096 for a
// 6000-long chain.
TEST(BoltJoinBounds, FanoutBoundCoversTheRealChainLength) {
    DupBuild b;
    ASSERT_TRUE(b.init(k_dup_rows, 42, /*force_general=*/false));
    EXPECT_GE(jk_max_fanout(&b.jb), static_cast<uint64_t>(k_dup_rows))
        << "jk_max_fanout under-reports the true fan-out; any caller sizing a "
           "pair buffer from it writes out of bounds on this build side";
    EXPECT_LE(jk_max_fanout(&b.jb), static_cast<uint64_t>(k_dup_rows))
        << "and it must stay a TIGHT bound, not degenerate to build_rows*k";
}

// ---------------------------------------------------------------------------
// 2. The real caller's sizing formula must be safe on this build side.
// ---------------------------------------------------------------------------
// Reproduces chukonu hash_join_swiss_probe_emit.inc verbatim:
//     blk = max(1, k_hj_pairs_cap / jk_max_fanout(build))
// then probes `blk` rows in one kernel call with pairs_cap = k_hj_pairs_cap.
// Pre-fix: fanout 4096 -> blk 2 -> 12,000 pairs into an 8,192-slot buffer.
TEST(BoltJoinBounds, CallerSizedBlockNeverOverflowsThePairBuffer) {
    DupBuild b;
    ASSERT_TRUE(b.init(k_dup_rows, 42, /*force_general=*/false));

    constexpr size_t cap = 8192;
    const uint64_t fan = jk_max_fanout(&b.jb);
    ASSERT_GT(fan, 0u);
    int64_t blk = static_cast<int64_t>(cap / fan);
    if (blk < 1) blk = 1;

    std::vector<int64_t> pkeys(static_cast<size_t>(blk), 42);
    BoltColumn p = BoltColumn::make_flat(pkeys.data(), nullptr, blk,
                                         BoltType::Int64);
    GuardedPairs g(cap, /*guard=*/cap);
    const size_t n = join_probe_typed(&b.jb, &p, blk, g.build.data(),
                                      g.probe.data(), cap, /*use_bloom=*/false);

    EXPECT_NE(n, kJkProbeOverflow)
        << "a block sized from jk_max_fanout must FIT — overflow here means "
           "the bound and the walk still disagree";
    EXPECT_LE(n, cap);
    EXPECT_EQ(g.canary_breaches(), 0u)
        << "probe wrote past pairs_cap: out-of-bounds write reachable from a "
           "user query on a skewed join key";
    // Value check, not just a bound: blk probe rows x k_dup_rows chain.
    EXPECT_EQ(n, static_cast<size_t>(blk) * static_cast<size_t>(k_dup_rows));
}

// ---------------------------------------------------------------------------
// 3. Over-cap is a typed error, never a write. (The lane gate asks for this.)
// ---------------------------------------------------------------------------
// Even with sizing fixed, a caller can hand the kernel a buffer that is too
// small. That must fail closed with the documented sentinel, not scribble.
TEST(BoltJoinBounds, OverCapProbeReturnsTypedErrorNotCorruption) {
    DupBuild b;
    ASSERT_TRUE(b.init(k_dup_rows, 42, /*force_general=*/false));

    constexpr size_t cap = 100;  // one probe row alone needs 6000
    int64_t pk = 42;
    BoltColumn p = BoltColumn::make_flat(&pk, nullptr, 1, BoltType::Int64);

    for (int mlp = 0; mlp < 2; ++mlp) {
        for (int bloom = 0; bloom < 2; ++bloom) {
            jk_mlp_force(mlp != 0, 16);
            GuardedPairs g(cap, /*guard=*/4096);
            const size_t n = join_probe_typed(&b.jb, &p, 1, g.build.data(),
                                              g.probe.data(), cap, bloom != 0);
            EXPECT_EQ(n, kJkProbeOverflow)
                << "mlp=" << mlp << " bloom=" << bloom
                << ": over-cap must return the overflow sentinel";
            EXPECT_EQ(g.canary_breaches(), 0u)
                << "mlp=" << mlp << " bloom=" << bloom
                << ": kernel wrote past pairs_cap";
        }
    }
    jk_mlp_force(true, 16);
}

// The General (composite-hash / typed-compare) lane has its own copy of the
// walk-and-emit loop, so it needs its own gate.
TEST(BoltJoinBounds, GeneralLaneOverCapReturnsTypedErrorNotCorruption) {
    DupBuild b;
    ASSERT_TRUE(b.init(k_dup_rows, 42, /*force_general=*/true));
    EXPECT_GE(jk_max_fanout(&b.jb), static_cast<uint64_t>(k_dup_rows));

    constexpr size_t cap = 100;
    int64_t pk = 42;
    BoltColumn p = BoltColumn::make_flat(&pk, nullptr, 1, BoltType::Int64);
    {
        GuardedPairs g(cap, /*guard=*/4096);
        const size_t n = jk_probe_general<false>(&b.jb, &p, 1, g.build.data(),
                                                 g.probe.data(), cap);
        EXPECT_EQ(n, kJkProbeOverflow) << "bloom=off";
        EXPECT_EQ(g.canary_breaches(), 0u) << "bloom=off";
    }
    if (b.jb.has_bloom != 0) {
        GuardedPairs g(cap, /*guard=*/4096);
        const size_t n = jk_probe_general<true>(&b.jb, &p, 1, g.build.data(),
                                                g.probe.data(), cap);
        EXPECT_EQ(n, kJkProbeOverflow) << "bloom=on";
        EXPECT_EQ(g.canary_breaches(), 0u) << "bloom=on";
    }
}

// ---------------------------------------------------------------------------
// 4. The older standalone chained kernel (bolt_hashjoin.h) has the same shape.
// ---------------------------------------------------------------------------
TEST(BoltJoinBounds, ChainedMultiMatchOverCapReturnsTypedError) {
    static bolt::Arena a;
    a.reset();
    constexpr uint64_t rows = 5000;  // > kHJMaxChainLen
    std::vector<uint64_t> bk(static_cast<size_t>(rows), 7u);
    const uint64_t* bcols[1] = {bk.data()};

    HashJoinBuildCfgChained cfg{};
    cfg.keys[0]             = BoltType::Int64;
    cfg.n_keys              = 1;
    cfg.build_rows          = rows;
    cfg.expected_dup_factor = kHJDefaultExpectedDupFact;
    cfg.max_chain_len       = kHJMaxChainLen;
    HashJoinBuildChained hb{};
    ASSERT_TRUE(hash_join_build_chained(bcols, cfg, &a, &hb));

    uint64_t pk = 7u;
    const uint64_t* pcols[1] = {&pk};
    constexpr size_t cap = 64;
    std::vector<uint32_t> ob(cap + 4096, 0xA5A5A5A5u), op(cap + 4096, 0xA5A5A5A5u);
    const size_t n = hash_join_probe_multi_match(&hb, bcols, pcols, 1,
                                                 ob.data(), op.data(), cap);
    EXPECT_EQ(n, kHJProbeOverflow);
    for (size_t i = cap; i < ob.size(); ++i) {
        ASSERT_EQ(ob[i], 0xA5A5A5A5u) << "wrote past pairs_cap at " << i;
        ASSERT_EQ(op[i], 0xA5A5A5A5u) << "wrote past pairs_cap at " << i;
    }
}

// ---------------------------------------------------------------------------
// 5. The fix must not change any in-bounds result.
// ---------------------------------------------------------------------------
// A short chain (well under the old clamp) is the case every existing suite
// covers; its pairs must be byte-identical to what the pre-fix kernel emitted.
TEST(BoltJoinBounds, ShortChainResultsUnchanged) {
    static bolt::Arena a;
    a.reset();
    constexpr int64_t nb = 3000, np = 500;
    std::vector<int64_t> bk(nb), pk(np);
    for (int64_t i = 0; i < nb; ++i) bk[static_cast<size_t>(i)] = i % 250;
    for (int64_t i = 0; i < np; ++i) pk[static_cast<size_t>(i)] = i % 300;
    BoltColumn b = BoltColumn::make_flat(bk.data(), nullptr, nb, BoltType::Int64);
    BoltColumn p = BoltColumn::make_flat(pk.data(), nullptr, np, BoltType::Int64);
    JoinBuildTyped jb{};
    ASSERT_TRUE(join_build_typed(&b, 1, nb, &a, &jb));
    // 3000 rows / 250 keys = 12 per chain; bound is (rows - distinct + 1).
    EXPECT_EQ(jk_max_fanout(&jb), static_cast<uint64_t>(nb - 250 + 1));

    GuardedPairs g(1u << 16, 4096);
    const size_t n = join_probe_typed(&jb, &p, np, g.build.data(),
                                      g.probe.data(), g.cap, false);
    ASSERT_NE(n, kJkProbeOverflow);
    // 250 of the 300 distinct probe keys hit, 12 build rows each; probe rows
    // repeat the 300-key cycle, so count by construction.
    size_t expect = 0;
    for (int64_t i = 0; i < np; ++i) {
        if (pk[static_cast<size_t>(i)] < 250) expect += 12;
    }
    EXPECT_EQ(n, expect);
    EXPECT_EQ(g.canary_breaches(), 0u);
}

}  // namespace
