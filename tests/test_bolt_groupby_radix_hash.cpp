// G2FEAT-13 — THE radix route-hash invariant property test.
//
// Radix-partitioned aggregation routes each row to a bucket sub-table by
// `route_hash & (P-1)` and later concatenates per-bucket partials WITHOUT
// re-aggregation. That is sound iff, for EVERY row, the per-row route hash
// equals the hash of the row's STORED group representative:
//
//     gb_route_hash_rows(row r) == gb_merge_hash_stored(state, gid_of(r))
//
// Any drift silently splits one logical group across buckets → duplicate
// output groups. This suite asserts the invariant for every row across
// key-type mixes × seeds, through the REAL begin/ingest paths (two-pass
// and fallback windows, gb_grow rehash, sparse sel ingest, spilled-Utf8
// deep-copy, inline-Utf8 canonicalisation with garbage padding).
//
// Also covered here (no test_bolt_radixpart.cpp exists):
//   - radix_partition_hashes basic invariants (offsets, permutation,
//     bucket membership by LOW bits, stability, duplicate hashes)
//   - sel-vector composition (physical-index output, logical-order stable)
//   - integration: route-hash → partition → bucket disjointness by gid
//   - GbSharedScratch begin variant differential vs the default begin

#include "bolt/join/bolt_groupby.h"
#include "bolt/kernels/bolt_radixpart.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace bolt;

constexpr int64_t  kRows   = 5000;
constexpr uint32_t kGroups = 200;

uint64_t splitmix64(uint64_t* s) noexcept {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline AggSpec make_spec(AggKind k, uint8_t in_col) noexcept {
    AggSpec s{};
    s.kind = k;
    s.in_col = in_col;
    return s;
}

// Per-slot key flavour. U8Mixed uses inline views for even group ids and
// spilled (>12-byte) views for odd ones — both in ONE column.
enum class KeyKind : uint8_t { I64, I32, D32, U8Inline, U8Spill, U8Mixed };

// Injective per-(group, slot) scalar derivations (g < kGroups).
int64_t key_i64(uint32_t g, uint32_t k) noexcept {
    return static_cast<int64_t>(
        (static_cast<uint64_t>(g) + 1u) * 0x9E3779B97F4A7C15ULL) ^
        static_cast<int64_t>(k * 1315423911u);
}
int32_t key_i32(uint32_t g, uint32_t k) noexcept {
    // odd-constant multiply is a bijection mod 2^32 → injective over g.
    return static_cast<int32_t>(g * 2654435761u + k * 97u);
}
int32_t key_d32(uint32_t g, uint32_t k) noexcept {
    return static_cast<int32_t>(18000u + g * 3u + k);
}

// Inline Utf8 (len 4..12), group id encoded in the first 4 bytes so the
// value is injective at every length. Unused StringView bytes get
// DETERMINISTIC GARBAGE — canonicalisation must erase it from hashing.
StringView key_sv_inline(uint32_t g, uint32_t k, uint8_t garbage) noexcept {
    char buf[12];
    const uint32_t len = 4u + ((g + k) % 9u);   // 4..12
    buf[0] = static_cast<char>('A' + static_cast<char>(g % 26u));
    buf[1] = static_cast<char>('0' + static_cast<char>((g / 100u) % 10u));
    buf[2] = static_cast<char>('0' + static_cast<char>((g / 10u) % 10u));
    buf[3] = static_cast<char>('0' + static_cast<char>(g % 10u));
    for (uint32_t j = 4; j < len; ++j) {
        buf[j] = static_cast<char>('a' + static_cast<char>((g + j) % 26u));
    }
    StringView v;
    std::memset(&v, garbage, sizeof(v));
    v.length = len;
    std::memcpy(v.prefix, buf, 4);
    std::memcpy(v.inline_data, buf + 4, len - 4u);
    return v;
}

// Spilled Utf8 (len 13..24). Bytes live in `overflow` at slot g*32; the
// view carries prefix + {buf_idx=0, offset}. `overflow` must outlive the
// morsel columns (the state deep-copies on first insert, but later rows
// still hash/compare against the source buffer).
StringView key_sv_spilled(uint32_t g, uint32_t k,
                          std::vector<char>* overflow) noexcept {
    const uint32_t len = 13u + ((g + k) % 12u);   // 13..24
    const uint32_t off = g * 32u;
    char* p = overflow->data() + off;
    p[0] = 'Z';
    p[1] = static_cast<char>('0' + static_cast<char>((g / 100u) % 10u));
    p[2] = static_cast<char>('0' + static_cast<char>((g / 10u) % 10u));
    p[3] = static_cast<char>('0' + static_cast<char>(g % 10u));
    for (uint32_t j = 4; j < len; ++j) {
        p[j] = static_cast<char>('a' + static_cast<char>((g * 7u + j) % 26u));
    }
    StringView v{};
    v.length = len;
    std::memcpy(v.prefix, p, 4);
    v.ref.buf_idx = 0;
    v.ref.offset  = off;
    return v;
}

// One built key column + its backing storage (kept alive by the caller).
struct KeyCol {
    std::vector<int64_t>    i64;
    std::vector<int32_t>    i32;
    std::vector<StringView> sv;
    std::vector<char>       overflow;
    BoltColumn              col;
};

void build_key_col(KeyCol* kc, KeyKind kind, uint32_t k,
                   const std::vector<uint32_t>& gids,
                   uint64_t* rng) noexcept {
    const size_t n = gids.size();
    switch (kind) {
        case KeyKind::I64: {
            kc->i64.resize(n);
            for (size_t i = 0; i < n; ++i) kc->i64[i] = key_i64(gids[i], k);
            kc->col = BoltColumn::make_flat(kc->i64.data(), nullptr,
                                            static_cast<int64_t>(n),
                                            BoltType::Int64);
            return;
        }
        case KeyKind::I32:
        case KeyKind::D32: {
            kc->i32.resize(n);
            for (size_t i = 0; i < n; ++i) {
                kc->i32[i] = (kind == KeyKind::I32) ? key_i32(gids[i], k)
                                                    : key_d32(gids[i], k);
            }
            kc->col = BoltColumn::make_flat(
                kc->i32.data(), nullptr, static_cast<int64_t>(n),
                (kind == KeyKind::I32) ? BoltType::Int32 : BoltType::Date32);
            return;
        }
        case KeyKind::U8Inline:
        case KeyKind::U8Spill:
        case KeyKind::U8Mixed: {
            kc->sv.resize(n);
            kc->overflow.assign(static_cast<size_t>(kGroups) * 32u, '\0');
            for (size_t i = 0; i < n; ++i) {
                const uint32_t g = gids[i];
                const bool spill =
                    (kind == KeyKind::U8Spill) ||
                    (kind == KeyKind::U8Mixed && (g & 1u) != 0u);
                if (spill) {
                    kc->sv[i] = key_sv_spilled(g, k, &kc->overflow);
                } else {
                    const uint8_t garbage =
                        static_cast<uint8_t>(splitmix64(rng) | 1u);
                    kc->sv[i] = key_sv_inline(g, k, garbage);
                }
            }
            kc->col = BoltColumn::make_flat(kc->sv.data(), nullptr,
                                            static_cast<int64_t>(n),
                                            BoltType::Utf8);
            kc->col.str_overflow_base = kc->overflow.data();
            return;
        }
    }
}

// Build columns, run the REAL begin/ingest (dense morsel + sparse sel
// morsel), then assert route_hash == stored hash for EVERY row.
void run_invariant_mix(const KeyKind* kinds, uint32_t nk, uint64_t seed) {
    Arena arena;
    uint64_t rng = seed;

    std::vector<uint32_t> gids(static_cast<size_t>(kRows));
    std::vector<bool> seen(kGroups, false);
    for (int64_t i = 0; i < kRows; ++i) {
        const uint32_t g = static_cast<uint32_t>(splitmix64(&rng) % kGroups);
        gids[static_cast<size_t>(i)] = g;
        seen[g] = true;
    }
    uint32_t distinct = 0;
    for (uint32_t g = 0; g < kGroups; ++g) distinct += seen[g] ? 1u : 0u;

    KeyCol kc[3];
    BoltColumn keys[3];
    for (uint32_t k = 0; k < nk; ++k) {
        build_key_col(&kc[k], kinds[k], k, gids, &rng);
        keys[k] = kc[k].col;
    }
    std::vector<int64_t> vals(static_cast<size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) vals[static_cast<size_t>(i)] = i + 1;
    BoltColumn val = BoltColumn::make_flat(vals.data(), nullptr, kRows,
                                           BoltType::Int64);

    AggSpec specs[2] = {make_spec(AggKind::CountStar, 0),
                        make_spec(AggKind::Sum, 0)};
    GroupbyTypedState st{};
    // Tight hint of 64 with grow_cap 4096: 200 groups FORCE gb_grow, whose
    // rehash itself rides gb_merge_hash_stored — extra invariant pressure.
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(
        &st, &arena, keys, static_cast<uint8_t>(nk), &val, 1, specs, 2,
        /*hint=*/64, /*grow_cap=*/4096));

    // Morsel A: dense rows [0, 3500). Morsel B: rows [3500, 5000) via a
    // sel vector (sparse ingest path).
    groupby_agg_multi_key_typed_ingest(&st, keys, &val, nullptr, 0, 3500);
    std::vector<uint32_t> ing_sel;
    for (int64_t r = 3500; r < kRows; ++r) {
        ing_sel.push_back(static_cast<uint32_t>(r));
    }
    groupby_agg_multi_key_typed_ingest(&st, keys, &val, ing_sel.data(),
                                       static_cast<uint32_t>(ing_sel.size()),
                                       kRows);
    ASSERT_FALSE(st.oom);
    ASSERT_EQ(st.entry_count, distinct);

    // Route hashes in windows (exercises start/count windowing).
    std::vector<uint64_t> route(static_cast<size_t>(kRows));
    for (int64_t start = 0; start < kRows; start += 1024) {
        const int64_t cnt = (kRows - start < 1024) ? (kRows - start) : 1024;
        gb_route_hash_rows(&st, keys, nullptr, start, cnt,
                           route.data() + start);
    }

    // Sel composition: a shuffled sel over all rows must yield the hash of
    // the PHYSICAL row sel[i].
    std::vector<uint32_t> sel(static_cast<size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) {
        sel[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    }
    for (int64_t i = kRows - 1; i > 0; --i) {
        const int64_t j =
            static_cast<int64_t>(splitmix64(&rng) % static_cast<uint64_t>(i + 1));
        std::swap(sel[static_cast<size_t>(i)], sel[static_cast<size_t>(j)]);
    }
    std::vector<uint64_t> route_sel(static_cast<size_t>(kRows));
    gb_route_hash_rows(&st, keys, sel.data(), 0, kRows, route_sel.data());
    for (int64_t i = 0; i < kRows; ++i) {
        ASSERT_EQ(route_sel[static_cast<size_t>(i)],
                  route[sel[static_cast<size_t>(i)]])
            << "sel composition drift at i=" << i;
    }

    // THE invariant, for every row: route hash finds the row's group via
    // the state's own SwissTable, the found group's keys match the row,
    // and the stored hash equals the route hash exactly.
    for (int64_t r = 0; r < kRows; ++r) {
        const uint64_t h = route[static_cast<size_t>(r)];
        const int32_t found = st.table->find(h);
        ASSERT_GE(found, 0) << "route hash misses the SwissTable, row " << r;
        ASSERT_TRUE(gb_detail::keys_equal(keys, nk, r, st.keys_flat,
                                          static_cast<uint32_t>(found),
                                          st.key_str_buf))
            << "route hash found a DIFFERENT group, row " << r;
        ASSERT_EQ(h, gb_merge_hash_stored(&st, static_cast<uint32_t>(found)))
            << "route/stored hash drift, row " << r;
    }
}

constexpr uint64_t kSeeds[3] = {0xC0FFEE0001ULL, 0xBADC0DE99ULL,
                                0x5EEDF00D42ULL};

TEST(BoltGroupbyRadixHash, InvariantInt64) {
    const KeyKind ks[1] = {KeyKind::I64};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 1, s);
}

TEST(BoltGroupbyRadixHash, InvariantInt32) {
    const KeyKind ks[1] = {KeyKind::I32};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 1, s);
}

TEST(BoltGroupbyRadixHash, InvariantDate32) {
    const KeyKind ks[1] = {KeyKind::D32};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 1, s);
}

TEST(BoltGroupbyRadixHash, InvariantInt64Int64) {
    const KeyKind ks[2] = {KeyKind::I64, KeyKind::I64};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 2, s);
}

TEST(BoltGroupbyRadixHash, InvariantInt64Utf8Inline) {
    const KeyKind ks[2] = {KeyKind::I64, KeyKind::U8Inline};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 2, s);
}

TEST(BoltGroupbyRadixHash, InvariantUtf8Spilled) {
    const KeyKind ks[1] = {KeyKind::U8Spill};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 1, s);
}

TEST(BoltGroupbyRadixHash, InvariantInt64Int32Utf8Mixed) {
    const KeyKind ks[3] = {KeyKind::I64, KeyKind::I32, KeyKind::U8Mixed};
    for (uint64_t s : kSeeds) run_invariant_mix(ks, 3, s);
}

// ---------------------------------------------------------------------------
// radix_partition_hashes — kernel invariants.
// ---------------------------------------------------------------------------

TEST(BoltRadixPartHashes, BasicInvariantsDense) {
    constexpr int64_t n = 10000;
    std::vector<uint64_t> hashes(static_cast<size_t>(n));
    uint64_t rng = 0xFEEDFACE7ULL;
    for (int64_t i = 0; i < n; ++i) {
        hashes[static_cast<size_t>(i)] = splitmix64(&rng);
    }
    for (int bits : {1, 4, 8}) {
        const int64_t nb = static_cast<int64_t>(1) << bits;
        const uint64_t mask = static_cast<uint64_t>(nb) - 1u;
        std::vector<int32_t> perm(static_cast<size_t>(n));
        std::vector<int64_t> offsets(static_cast<size_t>(nb) + 1u);
        radix_partition_hashes(hashes.data(), nullptr, n, bits, perm.data(),
                               offsets.data());
        ASSERT_EQ(offsets[0], 0);
        ASSERT_EQ(offsets[static_cast<size_t>(nb)], n);
        std::vector<uint8_t> hit(static_cast<size_t>(n), 0);
        for (int64_t b = 0; b < nb; ++b) {
            ASSERT_LE(offsets[static_cast<size_t>(b)],
                      offsets[static_cast<size_t>(b) + 1u]);
            int32_t prev = -1;
            for (int64_t p = offsets[static_cast<size_t>(b)];
                 p < offsets[static_cast<size_t>(b) + 1u]; ++p) {
                const int32_t row = perm[static_cast<size_t>(p)];
                ASSERT_GE(row, 0);
                ASSERT_LT(row, n);
                ASSERT_EQ(hit[static_cast<size_t>(row)], 0)  // permutation
                    << "row emitted twice";
                hit[static_cast<size_t>(row)] = 1;
                // Bucket = LOW bits of the FINAL hash — never re-mixed.
                ASSERT_EQ(static_cast<int64_t>(
                              hashes[static_cast<size_t>(row)] & mask),
                          b);
                ASSERT_GT(row, prev) << "stability violated in bucket " << b;
                prev = row;
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            ASSERT_EQ(hit[static_cast<size_t>(i)], 1) << "row dropped";
        }
    }
}

TEST(BoltRadixPartHashes, SelCompositionPhysicalIndices) {
    constexpr int64_t n = 4096;
    std::vector<uint64_t> hashes(static_cast<size_t>(n));
    std::vector<uint32_t> sel(static_cast<size_t>(n));
    uint64_t rng = 0xABCDEF123ULL;
    for (int64_t i = 0; i < n; ++i) {
        hashes[static_cast<size_t>(i)] = splitmix64(&rng);
        // Distinct, non-contiguous physical indices.
        sel[static_cast<size_t>(i)] = static_cast<uint32_t>(3 * i + 7);
    }
    constexpr int bits = 4;
    const int64_t nb = 16;
    const uint64_t mask = 15u;
    std::vector<int32_t> perm(static_cast<size_t>(n));
    std::vector<int64_t> offsets(static_cast<size_t>(nb) + 1u);
    radix_partition_hashes(hashes.data(), sel.data(), n, bits, perm.data(),
                           offsets.data());
    ASSERT_EQ(offsets[static_cast<size_t>(nb)], n);
    // Every perm entry is a sel value; bucket chosen by the LOGICAL row's
    // hash; stability in logical order (sel[i], sel[j] keep i<j order).
    for (int64_t b = 0; b < nb; ++b) {
        int64_t prev_logical = -1;
        for (int64_t p = offsets[static_cast<size_t>(b)];
             p < offsets[static_cast<size_t>(b) + 1u]; ++p) {
            const int32_t phys = perm[static_cast<size_t>(p)];
            ASSERT_EQ((phys - 7) % 3, 0) << "not a sel-produced index";
            const int64_t logical = (phys - 7) / 3;
            ASSERT_GE(logical, 0);
            ASSERT_LT(logical, n);
            ASSERT_EQ(static_cast<int64_t>(
                          hashes[static_cast<size_t>(logical)] & mask),
                      b);
            ASSERT_GT(logical, prev_logical) << "logical stability violated";
            prev_logical = logical;
        }
    }
}

TEST(BoltRadixPartHashes, DuplicateHashesSameBucketStable) {
    // Same hash → same bucket, input order preserved.
    constexpr int64_t n = 64;
    std::vector<uint64_t> hashes(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        hashes[static_cast<size_t>(i)] =
            0xDEADBEEF00ULL + static_cast<uint64_t>(i % 4);  // 4 duplicates
    }
    std::vector<int32_t> perm(static_cast<size_t>(n));
    std::vector<int64_t> offsets(4u + 1u);
    radix_partition_hashes(hashes.data(), nullptr, n, 2, perm.data(),
                           offsets.data());
    ASSERT_EQ(offsets[4], n);
    for (int64_t b = 0; b < 4; ++b) {
        int32_t prev = -1;
        for (int64_t p = offsets[static_cast<size_t>(b)];
             p < offsets[static_cast<size_t>(b) + 1u]; ++p) {
            const int32_t row = perm[static_cast<size_t>(p)];
            ASSERT_EQ(static_cast<int64_t>(row % 4), b);
            ASSERT_GT(row, prev);
            prev = row;
        }
    }
}

// Integration: route hashes from a REAL groupby state → partition → every
// gid lives in exactly ONE bucket, and that bucket is the stored hash's
// low bits — the disjointness radix-partitioned aggregation rests on.
TEST(BoltRadixPartHashes, BucketDisjointnessByGid) {
    Arena arena;
    uint64_t rng = 0x77AA55EE11ULL;
    std::vector<uint32_t> gids(static_cast<size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) {
        gids[static_cast<size_t>(i)] =
            static_cast<uint32_t>(splitmix64(&rng) % kGroups);
    }
    KeyCol kc[2];
    BoltColumn keys[2];
    const KeyKind kinds[2] = {KeyKind::I64, KeyKind::U8Mixed};
    for (uint32_t k = 0; k < 2; ++k) {
        build_key_col(&kc[k], kinds[k], k, gids, &rng);
        keys[k] = kc[k].col;
    }
    std::vector<int64_t> vals(static_cast<size_t>(kRows), 1);
    BoltColumn val = BoltColumn::make_flat(vals.data(), nullptr, kRows,
                                           BoltType::Int64);
    AggSpec spec = make_spec(AggKind::Sum, 0);
    GroupbyTypedState st{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&st, &arena, keys, 2, &val,
                                                  1, &spec, 1, /*hint=*/256));
    groupby_agg_multi_key_typed_ingest(&st, keys, &val, nullptr, 0, kRows);
    ASSERT_FALSE(st.oom);

    std::vector<uint64_t> route(static_cast<size_t>(kRows));
    gb_route_hash_rows(&st, keys, nullptr, 0, kRows, route.data());

    constexpr int bits = 4;
    const uint64_t mask = 15u;
    std::vector<int32_t> perm(static_cast<size_t>(kRows));
    std::vector<int64_t> offsets(16u + 1u);
    radix_partition_hashes(route.data(), nullptr, kRows, bits, perm.data(),
                           offsets.data());
    ASSERT_EQ(offsets[16], kRows);

    std::vector<int32_t> gid_bucket(st.entry_count, -1);
    for (int64_t b = 0; b < 16; ++b) {
        for (int64_t p = offsets[static_cast<size_t>(b)];
             p < offsets[static_cast<size_t>(b) + 1u]; ++p) {
            const int32_t row = perm[static_cast<size_t>(p)];
            const int32_t gid = st.table->find(route[static_cast<size_t>(row)]);
            ASSERT_GE(gid, 0);
            ASSERT_TRUE(gb_detail::keys_equal(keys, 2,
                                              static_cast<int64_t>(row),
                                              st.keys_flat,
                                              static_cast<uint32_t>(gid),
                                              st.key_str_buf));
            if (gid_bucket[static_cast<size_t>(gid)] < 0) {
                gid_bucket[static_cast<size_t>(gid)] =
                    static_cast<int32_t>(b);
            }
            ASSERT_EQ(gid_bucket[static_cast<size_t>(gid)],
                      static_cast<int32_t>(b))
                << "group split across buckets — the fatal drift";
            ASSERT_EQ(static_cast<int64_t>(
                          gb_merge_hash_stored(
                              &st, static_cast<uint32_t>(gid)) & mask),
                      b) << "bucket != stored-hash low bits";
        }
    }
}

// ---------------------------------------------------------------------------
// GbSharedScratch begin variant — differential vs the default begin.
// ---------------------------------------------------------------------------

TEST(BoltGroupbySharedScratch, SharedBeginMatchesDefault) {
    Arena arena;
    uint64_t rng = 0x13572468ACULL;
    std::vector<int64_t> ks(static_cast<size_t>(kRows));
    std::vector<int64_t> vs(static_cast<size_t>(kRows));
    for (int64_t i = 0; i < kRows; ++i) {
        const uint32_t g = static_cast<uint32_t>(splitmix64(&rng) % kGroups);
        ks[static_cast<size_t>(i)] = key_i64(g, 0);
        vs[static_cast<size_t>(i)] = static_cast<int64_t>(g) + 1;
    }
    BoltColumn key = BoltColumn::make_flat(ks.data(), nullptr, kRows,
                                           BoltType::Int64);
    BoltColumn val = BoltColumn::make_flat(vs.data(), nullptr, kRows,
                                           BoltType::Int64);
    AggSpec specs[2] = {make_spec(AggKind::CountStar, 0),
                        make_spec(AggKind::Sum, 0)};

    GroupbyTypedState a{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin(&a, &arena, &key, 1, &val,
                                                  1, specs, 2, /*hint=*/256));

    GbSharedScratch scratch{};
    scratch.gid_scratch = arena.allocate_array<int32_t>(kGbIngestWindow);
    ASSERT_NE(scratch.gid_scratch, nullptr);
    GroupbyTypedState b{};
    ASSERT_TRUE(groupby_agg_multi_key_typed_begin_shared(
        &b, &arena, &key, 1, &val, 1, specs, 2, /*hint=*/256, /*grow_cap=*/0,
        &scratch));
    EXPECT_EQ(b.gid_scratch, scratch.gid_scratch);
    EXPECT_EQ(b.bank_acc, nullptr);       // banks disabled for shared states
    EXPECT_EQ(b.bank_cnt, nullptr);

    groupby_agg_multi_key_typed_ingest(&a, &key, &val, nullptr, 0, kRows);
    groupby_agg_multi_key_typed_ingest(&b, &key, &val, nullptr, 0, kRows);
    ASSERT_FALSE(a.oom);
    ASSERT_FALSE(b.oom);
    ASSERT_EQ(a.entry_count, b.entry_count);

    BoltColumn oka[1], oaa[2], okb[1], oab[2];
    uint32_t nga = 0, ngb = 0;
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&a, oka, oaa, &nga));
    ASSERT_TRUE(groupby_agg_multi_key_typed_finalize(&b, okb, oab, &ngb));
    ASSERT_EQ(nga, ngb);
    std::map<int64_t, std::pair<int64_t, int64_t>> ma, mb;
    const int64_t* ka_out = static_cast<const int64_t*>(oka[0].data);
    const int64_t* kb_out = static_cast<const int64_t*>(okb[0].data);
    const int64_t* ca_out = static_cast<const int64_t*>(oaa[0].data);
    const int64_t* cb_out = static_cast<const int64_t*>(oab[0].data);
    const int64_t* sa_out = static_cast<const int64_t*>(oaa[1].data);
    const int64_t* sb_out = static_cast<const int64_t*>(oab[1].data);
    for (uint32_t i = 0; i < nga; ++i) {
        ma[ka_out[i]] = {ca_out[i], sa_out[i]};
        mb[kb_out[i]] = {cb_out[i], sb_out[i]};
    }
    EXPECT_EQ(ma, mb);
}

}  // namespace
