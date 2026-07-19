// test_bolt_vector_column.cpp — Wave 9.4 Phase 1 coverage.
//
// Pins the contract of:
//   1. BoltColumn::make_flat_vector / make_flat_vector_alloc / vector_dim
//   2. BoltColumn::compute_stats_vector (per-dim Welford)
//   3. bolt_wire round-trip with `fixed_size` round-trip for Embedding
//   4. bolt::vec::kmeans_train_f32 cluster recovery on synthetic GMM
//   5. bolt::vec::bond_zone_rank_scores_f32 monotonicity
//   6. bolt::vec::sigma_bound_remaining_minmax_f32 conservativeness
//
// Tiger Style: deterministic seeds, fixed-capacity arrays, no
// floating-point equality (uses bounded relative epsilon).

#include <gtest/gtest.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_vector_bond.h"
#include "bolt/kernels/bolt_vector_kmeans.h"
#include "bolt/wire/bolt_wire.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kDim = 32u;
constexpr uint32_t kN   = 128u;

// Deterministic PRNG so the test is reproducible.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t s) noexcept : state(s) {}
    float unit() noexcept {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        const uint64_t mant = z >> 11;
        return static_cast<float>(static_cast<double>(mant) *
                                  (1.0 / static_cast<double>(1ULL << 53)));
    }
    float normal() noexcept {
        // Marsaglia polar.
        float u, v, s;
        do {
            u = 2.0f * unit() - 1.0f;
            v = 2.0f * unit() - 1.0f;
            s = u * u + v * v;
        } while (s == 0.0f || s >= 1.0f);
        return u * std::sqrt(-2.0f * std::log(s) / s);
    }
};

}  // namespace

TEST(BoltVectorColumn, FactoriesEncodeStrideAndDim) {
    bolt::Arena arena;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_alloc(kN, kDim, &arena);
    ASSERT_NE(c.data, nullptr);
    EXPECT_EQ(c.type, bolt::BoltType::Embedding);
    EXPECT_EQ(c.format, bolt::ColumnFormat::Flat);
    EXPECT_EQ(c.length, static_cast<int64_t>(kN));
    EXPECT_EQ(c.type_size_bytes, kDim * sizeof(float));
    EXPECT_EQ(c.vector_dim(), kDim);
}

TEST(BoltVectorColumn, FactoryRejectsZeroDim) {
    bolt::Arena arena;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_alloc(kN, 0u, &arena);
    EXPECT_EQ(c.data, nullptr);
    EXPECT_EQ(c.length, 0);
}

TEST(BoltVectorColumn, ComputeStatsVectorRecoversMeanAndRange) {
    bolt::Arena arena;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_alloc(kN, kDim, &arena);
    ASSERT_NE(c.data, nullptr);

    Rng rng(0xC0FFEE01ull);
    for (uint32_t i = 0; i < kN; ++i) {
        float* row = c.vector_row_mut(static_cast<int64_t>(i));
        for (uint32_t d = 0; d < kDim; ++d) {
            row[d] = static_cast<float>(d) +
                     0.01f * static_cast<float>(i) +
                     0.001f * rng.normal();
        }
    }
    bolt::VectorStats* vs = c.compute_stats_vector(&arena);
    ASSERT_NE(vs, nullptr);
    EXPECT_EQ(vs->dim, kDim);
    EXPECT_EQ(vs->row_count, kN);

    // Reference: pass mean + min/max scalar.
    for (uint32_t d = 0; d < kDim; ++d) {
        double sum = 0.0;
        float ref_min = 1e30f, ref_max = -1e30f;
        for (uint32_t i = 0; i < kN; ++i) {
            const float v = c.vector_row(static_cast<int64_t>(i))[d];
            sum += v;
            if (v < ref_min) ref_min = v;
            if (v > ref_max) ref_max = v;
        }
        const float ref_mu = static_cast<float>(sum / static_cast<double>(kN));
        EXPECT_NEAR(vs->mean[d], ref_mu, 1e-3f);
        EXPECT_NEAR(vs->min[d],  ref_min, 1e-4f);
        EXPECT_NEAR(vs->max[d],  ref_max, 1e-4f);
    }
}

TEST(BoltVectorColumn, WireRoundTripPreservesDimAndPayload) {
    bolt::Arena arena;
    constexpr uint32_t kRows = 17u;  // not power-of-two on purpose
    constexpr uint32_t kD    = 384u;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_alloc(kRows, kD, &arena);
    ASSERT_NE(c.data, nullptr);

    Rng rng(0xBEEF1234ull);
    for (uint32_t r = 0; r < kRows; ++r) {
        float* row = c.vector_row_mut(static_cast<int64_t>(r));
        for (uint32_t d = 0; d < kD; ++d) {
            row[d] = rng.normal();
        }
    }

    bolt::BoltBatch b;
    bolt::BoltBatch::init_empty(&b);
    b.arena    = &arena;
    b.num_cols = 1;
    b.num_rows = kRows;
    bolt::BoltBatch::alloc_columns(&b, &arena, 1);  // G2FEAT-47: size columns[2]
    b.schema.num_fields = 1;
    std::memset(&b.schema.fields[0], 0, sizeof(b.schema.fields[0]));
    std::memcpy(b.schema.fields[0].name, "emb", 3);
    b.schema.fields[0].type       = bolt::BoltType::Embedding;
    b.schema.fields[0].fixed_size = kD;
    b.columns[0][0] = c;
    b.columns[1][0] = c;

    const size_t sz = bolt::wire::bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(sz, 64));
    ASSERT_NE(buf, nullptr);
    const size_t written = bolt::wire::bolt_wire_serialize(&b, buf, sz);
    ASSERT_EQ(written, sz);

    bolt::Arena rd_arena;
    bolt::BoltBatch out;
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(buf, sz, &out, &rd_arena));
    EXPECT_EQ(out.num_cols, 1u);
    EXPECT_EQ(out.num_rows, static_cast<int64_t>(kRows));
    EXPECT_EQ(out.schema.fields[0].fixed_size, kD);
    EXPECT_EQ(out.schema.fields[0].type, bolt::BoltType::Embedding);
    const bolt::BoltColumn& rc = out.col(0);
    EXPECT_EQ(rc.vector_dim(), kD);
    EXPECT_EQ(rc.type_size_bytes, kD * sizeof(float));
    // Bit-exact recover of every row.
    for (uint32_t r = 0; r < kRows; ++r) {
        const float* src = c.vector_row(static_cast<int64_t>(r));
        const float* dst = rc.vector_row(static_cast<int64_t>(r));
        for (uint32_t d = 0; d < kD; ++d) {
            ASSERT_EQ(std::memcmp(src + d, dst + d, sizeof(float)), 0)
                << "row=" << r << " dim=" << d;
        }
    }
}

TEST(BoltVectorKmeans, RecoversSyntheticClusters) {
    bolt::Arena arena;
    constexpr uint32_t kD     = 16u;
    constexpr uint32_t kK     = 4u;
    constexpr uint32_t kPerK  = 64u;
    constexpr uint32_t kN_all = kK * kPerK;
    float* vecs = static_cast<float*>(
        arena.allocate(static_cast<size_t>(kN_all) * kD * sizeof(float)));
    ASSERT_NE(vecs, nullptr);

    Rng rng(0xCAFE2025ull);
    // Synthesise kK well-separated centroids.
    float true_cents[kK * kD];
    for (uint32_t k = 0; k < kK; ++k) {
        for (uint32_t d = 0; d < kD; ++d) {
            true_cents[k * kD + d] = static_cast<float>(k * 10) +
                                     0.1f * rng.normal();
        }
    }
    // Sample points around each centroid.
    uint32_t* labels = static_cast<uint32_t*>(
        arena.allocate(kN_all * sizeof(uint32_t)));
    for (uint32_t k = 0; k < kK; ++k) {
        for (uint32_t i = 0; i < kPerK; ++i) {
            const uint32_t idx = k * kPerK + i;
            labels[idx] = k;
            float* row = vecs + static_cast<size_t>(idx) * kD;
            for (uint32_t d = 0; d < kD; ++d) {
                row[d] = true_cents[k * kD + d] + 0.05f * rng.normal();
            }
        }
    }

    float out_cents[kK * kD];
    uint32_t out_assign[kN_all];
    ASSERT_TRUE(bolt::vec::kmeans_train_f32(
        vecs, kN_all, kD, kK,
        /*max_iters=*/20, /*restarts=*/3, /*seed=*/0xBEEFu,
        out_cents, out_assign, &arena));

    // Recovery: every true cluster must dominate (≥ 95% of its points
    // share an assigned label, even if the label index permutes).
    uint32_t hist[kK][kK]{};
    for (uint32_t i = 0; i < kN_all; ++i) {
        const uint32_t t = labels[i];
        const uint32_t a = out_assign[i];
        ASSERT_LT(a, kK);
        ++hist[t][a];
    }
    for (uint32_t t = 0; t < kK; ++t) {
        uint32_t best = 0u;
        for (uint32_t a = 0; a < kK; ++a) {
            if (hist[t][a] > best) best = hist[t][a];
        }
        EXPECT_GE(best, static_cast<uint32_t>(kPerK * 95u / 100u))
            << "true cluster " << t << " best alignment = " << best;
    }
}

// ---------------------------------------------------------------------------
// Wave 9.4 z.5 — multi-precision Embedding family.
// Round-trip f16 / u8 / i8 columns through wire → deserialise. Every test
// asserts bit-exact recovery (no quantisation error at the wire layer — the
// payload is just bytes; the wire format never interprets element values).
// ---------------------------------------------------------------------------

namespace {

// Deterministic byte / half fill, independent of the f32 PRNG in `Rng`.
// Returns a value in [0, 65535] mapped via a simple LCG so the same seed
// produces the same payload across runs.
struct ByteRng {
    uint64_t state;
    explicit ByteRng(uint64_t s) noexcept : state(s ? s : 1ull) {}
    uint16_t next_u16() noexcept {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint16_t>(state >> 48);
    }
    uint8_t next_u8() noexcept {
        return static_cast<uint8_t>(next_u16() & 0xFFu);
    }
};

// Build a BoltBatch with a single vector column of the given type. Caller
// owns the arena and the underlying payload (allocated inside `arena`).
void build_vector_batch(bolt::BoltBatch* b,
                        bolt::Arena* arena,
                        const bolt::BoltColumn& c,
                        uint32_t dim,
                        const char* col_name) noexcept {
    bolt::BoltBatch::init_empty(b);
    b->arena    = arena;
    b->num_cols = 1;
    b->num_rows = c.length;
    bolt::BoltBatch::alloc_columns(b, arena, 1);  // G2FEAT-47: size columns[2]
    b->schema.num_fields = 1;
    std::memset(&b->schema.fields[0], 0, sizeof(b->schema.fields[0]));
    const size_t nlen = std::strlen(col_name);
    std::memcpy(b->schema.fields[0].name, col_name,
                nlen < bolt::kMaxFieldName ? nlen : bolt::kMaxFieldName);
    b->schema.fields[0].type       = c.type;
    b->schema.fields[0].fixed_size = dim;
    b->columns[0][0] = c;
    b->columns[1][0] = c;
}

}  // namespace

TEST(BoltVectorColumn, FactoriesEncodeStrideAndDim_F16_U8_I8) {
    bolt::Arena arena;
    constexpr uint32_t kRows = 11u;
    constexpr uint32_t kD    = 128u;

    bolt::BoltColumn cf16 =
        bolt::BoltColumn::make_flat_vector_f16_alloc(kRows, kD, &arena);
    ASSERT_NE(cf16.data, nullptr);
    EXPECT_EQ(cf16.type, bolt::BoltType::EmbeddingF16);
    EXPECT_EQ(cf16.format, bolt::ColumnFormat::Flat);
    EXPECT_EQ(cf16.type_size_bytes, kD * 2u);
    EXPECT_EQ(cf16.vector_dim(), kD);

    bolt::BoltColumn cu8 =
        bolt::BoltColumn::make_flat_vector_u8_alloc(kRows, kD, &arena);
    ASSERT_NE(cu8.data, nullptr);
    EXPECT_EQ(cu8.type, bolt::BoltType::EmbeddingU8);
    EXPECT_EQ(cu8.type_size_bytes, kD * 1u);
    EXPECT_EQ(cu8.vector_dim(), kD);

    bolt::BoltColumn ci8 =
        bolt::BoltColumn::make_flat_vector_i8_alloc(kRows, kD, &arena);
    ASSERT_NE(ci8.data, nullptr);
    EXPECT_EQ(ci8.type, bolt::BoltType::EmbeddingI8);
    EXPECT_EQ(ci8.type_size_bytes, kD * 1u);
    EXPECT_EQ(ci8.vector_dim(), kD);

    // Zero-dim guard works on every variant.
    EXPECT_EQ(bolt::BoltColumn::make_flat_vector_f16_alloc(kRows, 0u, &arena).data,
              nullptr);
    EXPECT_EQ(bolt::BoltColumn::make_flat_vector_u8_alloc(kRows, 0u, &arena).data,
              nullptr);
    EXPECT_EQ(bolt::BoltColumn::make_flat_vector_i8_alloc(kRows, 0u, &arena).data,
              nullptr);
}

TEST(BoltVectorColumn, WireRoundTripPreservesF16Payload) {
    bolt::Arena arena;
    constexpr uint32_t kRows = 19u;          // non-power-of-two on purpose
    constexpr uint32_t kD    = 384u;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_f16_alloc(kRows, kD, &arena);
    ASSERT_NE(c.data, nullptr);

    // Fill with deterministic u16 payload (the wire never reinterprets
    // these as floats — it's just bytes — but using realistic f16 bit
    // patterns documents the intent).
    ByteRng rng(0xF16D'EAD0'BEEFull);
    uint16_t* src = static_cast<uint16_t*>(c.data);
    for (uint32_t i = 0; i < kRows * kD; ++i) src[i] = rng.next_u16();

    bolt::BoltBatch b;
    build_vector_batch(&b, &arena, c, kD, "emb16");

    const size_t sz = bolt::wire::bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(sz, 64));
    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(bolt::wire::bolt_wire_serialize(&b, buf, sz), sz);

    bolt::Arena rd_arena;
    bolt::BoltBatch out;
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(buf, sz, &out, &rd_arena));
    EXPECT_EQ(out.num_cols, 1u);
    EXPECT_EQ(out.num_rows, static_cast<int64_t>(kRows));
    EXPECT_EQ(out.schema.fields[0].type, bolt::BoltType::EmbeddingF16);
    EXPECT_EQ(out.schema.fields[0].fixed_size, kD);
    const bolt::BoltColumn& rc = out.col(0);
    EXPECT_EQ(rc.type, bolt::BoltType::EmbeddingF16);
    EXPECT_EQ(rc.vector_dim(), kD);
    EXPECT_EQ(rc.type_size_bytes, kD * 2u);
    // Exact bit-for-bit recover on the f16 storage.
    ASSERT_EQ(std::memcmp(c.data, rc.data,
                          static_cast<size_t>(kRows) * kD * 2u), 0);
}

TEST(BoltVectorColumn, WireRoundTripPreservesU8Payload) {
    bolt::Arena arena;
    constexpr uint32_t kRows = 23u;
    constexpr uint32_t kD    = 512u;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_u8_alloc(kRows, kD, &arena);
    ASSERT_NE(c.data, nullptr);

    ByteRng rng(0xC0FFEE'0008ull);
    uint8_t* src = static_cast<uint8_t*>(c.data);
    for (uint32_t i = 0; i < kRows * kD; ++i) src[i] = rng.next_u8();

    bolt::BoltBatch b;
    build_vector_batch(&b, &arena, c, kD, "emb_u8");

    const size_t sz = bolt::wire::bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(sz, 64));
    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(bolt::wire::bolt_wire_serialize(&b, buf, sz), sz);

    bolt::Arena rd_arena;
    bolt::BoltBatch out;
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(buf, sz, &out, &rd_arena));
    const bolt::BoltColumn& rc = out.col(0);
    EXPECT_EQ(rc.type, bolt::BoltType::EmbeddingU8);
    EXPECT_EQ(rc.vector_dim(), kD);
    EXPECT_EQ(rc.type_size_bytes, kD * 1u);
    ASSERT_EQ(std::memcmp(c.data, rc.data,
                          static_cast<size_t>(kRows) * kD), 0);
}

TEST(BoltVectorColumn, WireRoundTripPreservesI8Payload) {
    bolt::Arena arena;
    constexpr uint32_t kRows = 7u;
    constexpr uint32_t kD    = 256u;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_i8_alloc(kRows, kD, &arena);
    ASSERT_NE(c.data, nullptr);

    ByteRng rng(0xDEADBEEF'0008ull);
    int8_t* src = static_cast<int8_t*>(c.data);
    for (uint32_t i = 0; i < kRows * kD; ++i) {
        src[i] = static_cast<int8_t>(rng.next_u8());
    }

    bolt::BoltBatch b;
    build_vector_batch(&b, &arena, c, kD, "emb_i8");

    const size_t sz = bolt::wire::bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(sz, 64));
    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(bolt::wire::bolt_wire_serialize(&b, buf, sz), sz);

    bolt::Arena rd_arena;
    bolt::BoltBatch out;
    ASSERT_TRUE(bolt::wire::bolt_wire_deserialize(buf, sz, &out, &rd_arena));
    const bolt::BoltColumn& rc = out.col(0);
    EXPECT_EQ(rc.type, bolt::BoltType::EmbeddingI8);
    EXPECT_EQ(rc.vector_dim(), kD);
    EXPECT_EQ(rc.type_size_bytes, kD * 1u);
    ASSERT_EQ(std::memcmp(c.data, rc.data,
                          static_cast<size_t>(kRows) * kD), 0);
}

TEST(BoltVectorColumn, WireRejectsUnsupportedFormatPair) {
    // Only (Embedding*, Flat) is legal. The wire layer must reject any
    // (Embedding*, VarBinary) labelling at both sides — sizing returns 0
    // and serialize fails up front. We poke the schema byte directly to
    // simulate a corrupt / mislabelled stream.
    using namespace bolt::wire::detail;
    EXPECT_FALSE(is_supported_format_pair(bolt::BoltType::EmbeddingF16,
                                          bolt::ColumnFormat::VarBinary));
    EXPECT_FALSE(is_supported_format_pair(bolt::BoltType::EmbeddingU8,
                                          bolt::ColumnFormat::VarBinary));
    EXPECT_FALSE(is_supported_format_pair(bolt::BoltType::EmbeddingI8,
                                          bolt::ColumnFormat::VarBinary));
    // Constant / Dictionary etc. are also illegal for the embedding family.
    EXPECT_FALSE(is_supported_format_pair(bolt::BoltType::EmbeddingF16,
                                          bolt::ColumnFormat::Constant));
    EXPECT_FALSE(is_supported_format_pair(bolt::BoltType::EmbeddingU8,
                                          bolt::ColumnFormat::Dictionary));
    // The Flat pair *is* legal — guards against accidentally narrowing
    // the supported set.
    EXPECT_TRUE(is_supported_format_pair(bolt::BoltType::EmbeddingF16,
                                         bolt::ColumnFormat::Flat));
    EXPECT_TRUE(is_supported_format_pair(bolt::BoltType::EmbeddingU8,
                                         bolt::ColumnFormat::Flat));
    EXPECT_TRUE(is_supported_format_pair(bolt::BoltType::EmbeddingI8,
                                         bolt::ColumnFormat::Flat));
}

TEST(BoltVectorColumn, WireRejectsMislabelledFormatByteOnRead) {
    // Construct a valid f16 wire payload, then flip the format byte in
    // both the schema entry (offset 65 of the entry) and the descriptor
    // (offset 48 of the descriptor) so the deserialiser sees
    // (EmbeddingF16, VarBinary). Expect rejection.
    bolt::Arena arena;
    constexpr uint32_t kRows = 4u;
    constexpr uint32_t kD    = 64u;
    bolt::BoltColumn c =
        bolt::BoltColumn::make_flat_vector_f16_alloc(kRows, kD, &arena);
    ASSERT_NE(c.data, nullptr);
    std::memset(c.data, 0xAB, static_cast<size_t>(kRows) * kD * 2u);

    bolt::BoltBatch b;
    build_vector_batch(&b, &arena, c, kD, "emb16");

    const size_t sz = bolt::wire::bolt_wire_size(&b);
    ASSERT_GT(sz, 0u);
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(sz, 64));
    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(bolt::wire::bolt_wire_serialize(&b, buf, sz), sz);

    // Corrupt the schema's format byte. Header is 32 bytes; schema entry
    // 0 starts at offset 32; format byte sits at offset 65.
    buf[32u + 65u] = static_cast<uint8_t>(bolt::ColumnFormat::VarBinary);

    bolt::Arena rd_arena;
    bolt::BoltBatch out;
    EXPECT_FALSE(bolt::wire::bolt_wire_deserialize(buf, sz, &out, &rd_arena))
        << "deserialiser must reject (EmbeddingF16, VarBinary)";
}

TEST(BoltVectorBond, ZoneRankAndMinMaxBound) {
    constexpr uint32_t kD = 16u;
    float q[kD], mu[kD], mn[kD], mx[kD];
    for (uint32_t d = 0; d < kD; ++d) {
        q [d] = static_cast<float>(d);
        mu[d] = static_cast<float>(d) +
                ((d == 5u || d == 11u) ? 5.0f : 0.0f);
        mn[d] = mu[d] - 1.0f;
        mx[d] = mu[d] + 1.0f;
    }
    constexpr uint32_t kZone = 4u;
    float scores[kD / kZone];
    const uint32_t nz = bolt::vec::bond_zone_rank_scores_f32(
        q, mu, kD, kZone, scores);
    ASSERT_EQ(nz, kD / kZone);
    // Zones containing dim 5 (zone 1) and dim 11 (zone 2) score higher
    // than the others.
    EXPECT_GT(scores[1], scores[0]);
    EXPECT_GT(scores[1], scores[3]);
    EXPECT_GT(scores[2], scores[3]);

    // σ-bound is non-negative; bound > 0 because mn/mx are tight.
    const float ub = bolt::vec::sigma_bound_remaining_minmax_f32(
        q, mn, mx, /*visited=*/nullptr, kD);
    EXPECT_GE(ub, 0.0f);
    EXPECT_LT(ub, 1e6f);
}
