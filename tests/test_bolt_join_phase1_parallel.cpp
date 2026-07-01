// test_bolt_join_phase1_parallel — W-PAR inc 3: the chunked parallel Phase 1
// (jk_build_phase1_prepare / pass_a / glue / pass_b) must produce a scatter
// BYTE-IDENTICAL to the serial jk_build_scatter_and_alloc: same offsets, same
// row_ids order (ascending row order within every partition — the chain-order
// contract), same key cells / chain init, and a bloom with the same behaviour.
// The task bodies are plain functions, so the test drives the chunks serially
// (the parallel schedule only changes WHICH thread runs a chunk, not the data).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/join/bolt_joinkernel.h"

namespace {

// Build an Int64 key column with every 97th row NULL (null path coverage).
bolt::BoltColumn make_keys(bolt::Arena* a, std::int64_t n,
                           std::uint64_t seed, bool with_nulls) {
    auto* v = a->allocate_array<std::int64_t>(static_cast<std::size_t>(n));
    std::uint8_t* validity = nullptr;
    if (with_nulls) {
        const std::size_t vb = static_cast<std::size_t>((n + 7) / 8);
        validity = static_cast<std::uint8_t*>(a->allocate(vb, 8));
        std::memset(validity, 0xFF, vb);
    }
    std::uint64_t x = seed;
    for (std::int64_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;      // xorshift64
        v[i] = static_cast<std::int64_t>(x % 20011);  // repeated keys (chains)
        if (with_nulls && (i % 97) == 0) {
            validity[i >> 3] &= static_cast<std::uint8_t>(~(1u << (i & 7)));
        }
    }
    bolt::BoltColumn c = bolt::BoltColumn::make_flat(
        v, validity, n, bolt::BoltType::Int64);
    return c;
}

void run_parallel_phase1(const bolt::BoltColumn* keys, std::uint8_t n_keys,
                         std::uint64_t rows, std::uint32_t chunks,
                         bolt::Arena* arena, bolt::JoinBuildTyped* out,
                         bolt::JoinBuildScatterCtx* scat) {
    bolt::JkPhase1Ctx c{};
    ASSERT_TRUE(bolt::jk_build_phase1_prepare(keys, n_keys, rows, chunks,
                                              arena, out, scat, &c));
    for (std::uint32_t k = 0; k < c.n_chunks; ++k) bolt::jk_build_pass_a(&c, k);
    ASSERT_TRUE(bolt::jk_build_phase1_glue(&c, arena));
    for (std::uint32_t k = 0; k < c.n_chunks; ++k) bolt::jk_build_pass_b(&c, k);
}

void expect_identical(const bolt::JoinBuildTyped& a,
                      const bolt::JoinBuildScatterCtx& sa,
                      const bolt::JoinBuildTyped& b,
                      const bolt::JoinBuildScatterCtx& sb,
                      std::uint64_t rows, std::uint8_t n_keys) {
    for (std::uint32_t p = 0; p <= bolt::kHJNumPartitions; ++p) {
        ASSERT_EQ(sa.offsets[p], sb.offsets[p]) << "offset partition " << p;
    }
    const std::uint32_t total = sa.offsets[bolt::kHJNumPartitions];
    ASSERT_EQ(0, std::memcmp(sa.row_ids, sb.row_ids,
                             total * sizeof(std::int32_t)))
        << "row_ids diverge (chain-order contract broken)";
    ASSERT_EQ(0, std::memcmp(a.keys_flat, b.keys_flat,
                             static_cast<std::size_t>(rows) * n_keys *
                                 sizeof(bolt::GbCell16)));
    for (std::uint64_t i = 0; i < rows; ++i) {
        ASSERT_EQ(a.chain_nodes[i].build_idx, b.chain_nodes[i].build_idx);
        ASSERT_EQ(a.chain_nodes[i].next, b.chain_nodes[i].next);
    }
}

TEST(JoinPhase1Parallel, ByteIdenticalToSerial_WithNulls) {
    bolt::Arena arena;
    const std::uint64_t rows = 300000;   // > 2*kJkP1MinChunkRows -> chunked
    bolt::BoltColumn keys = make_keys(&arena, static_cast<std::int64_t>(rows),
                                      0x9E3779B97F4A7C15ull, /*nulls=*/true);
    bolt::JoinBuildTyped ser{}; bolt::JoinBuildScatterCtx ser_s{};
    ASSERT_TRUE(bolt::jk_build_scatter_and_alloc(&keys, 1, rows, &arena,
                                                 &ser, &ser_s));
    bolt::JoinBuildTyped par{}; bolt::JoinBuildScatterCtx par_s{};
    run_parallel_phase1(&keys, 1, rows, /*chunks=*/7, &arena, &par, &par_s);
    expect_identical(ser, ser_s, par, par_s, rows, 1);
    // Bloom: every reachable key must test positive in BOTH blooms.
    ASSERT_EQ(ser.has_bloom, par.has_bloom);
    if (ser.has_bloom != 0) {
        for (std::uint64_t i = 0; i < rows; ++i) {
            if (bolt::join_row_has_null(&keys, 1, static_cast<std::int64_t>(i)))
                continue;
            const std::uint64_t h =
                bolt::swiss_mix(bolt::jk_read_int64_key(keys,
                    static_cast<std::int64_t>(i)));
            ASSERT_TRUE(bolt::sbbf_test(par.bloom, h)) << "bloom lost row " << i;
        }
    }
}

TEST(JoinPhase1Parallel, ByteIdenticalAcrossChunkCounts) {
    bolt::Arena arena;
    const std::uint64_t rows = 262144;
    bolt::BoltColumn keys = make_keys(&arena, static_cast<std::int64_t>(rows),
                                      0xC0FFEEull, /*nulls=*/false);
    bolt::JoinBuildTyped a{}; bolt::JoinBuildScatterCtx a_s{};
    run_parallel_phase1(&keys, 1, rows, /*chunks=*/2, &arena, &a, &a_s);
    bolt::JoinBuildTyped b{}; bolt::JoinBuildScatterCtx b_s{};
    run_parallel_phase1(&keys, 1, rows, /*chunks=*/13, &arena, &b, &b_s);
    expect_identical(a, a_s, b, b_s, rows, 1);   // geometry-independent
}

TEST(JoinPhase1Parallel, PrepareDeclinesTinyBuild) {
    bolt::Arena arena;
    const std::uint64_t rows = 1000;     // < 2*kJkP1MinChunkRows
    bolt::BoltColumn keys = make_keys(&arena, static_cast<std::int64_t>(rows),
                                      42ull, false);
    bolt::JoinBuildTyped out{}; bolt::JoinBuildScatterCtx scat{};
    bolt::JkPhase1Ctx c{};
    EXPECT_FALSE(bolt::jk_build_phase1_prepare(&keys, 1, rows, 8, &arena,
                                               &out, &scat, &c));
}

}  // namespace
